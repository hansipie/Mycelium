use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result};
use hf_hub::api::tokio::ApiBuilder;
use serde::Deserialize;
use tracing::{info, warn};

use crate::{
    config::settings::ModelArchitecture,
    model::engine::ModelInfo,
};

// ── Public types ──────────────────────────────────────────────────────────────

/// Resolved paths for all files needed to load a model.
#[derive(Debug, Clone)]
pub struct ModelFiles {
    pub config_path: PathBuf,
    pub tokenizer_path: PathBuf,
    pub weight_paths: Vec<PathBuf>,
}

/// Minimal HuggingFace config.json fields we need.
#[derive(Debug, Deserialize)]
struct HfModelConfig {
    pub architectures: Option<Vec<String>>,
    pub num_hidden_layers: Option<usize>,
    pub max_position_embeddings: Option<usize>,
    pub vocab_size: Option<usize>,
}

// ── Cache resolution (no network) ────────────────────────────────────────────

/// Try to resolve model files from a pre-populated cache directory.
/// Returns an error if any required file is missing.
pub fn resolve_model_files(cache_dir: &Path, model_id: &str) -> Result<ModelFiles> {
    // hf-hub cache layout: {cache_dir}/models--{org}--{name}/snapshots/{commit}/
    let dir_name = model_id.replace('/', "--");
    let model_dir = cache_dir.join(format!("models--{dir_name}"));

    // Walk snapshots to find the most recent one
    let snapshots_dir = model_dir.join("snapshots");
    if !snapshots_dir.exists() {
        anyhow::bail!(
            "model not in cache: {} (expected {})",
            model_id,
            snapshots_dir.display()
        );
    }

    let snapshot = find_latest_snapshot(&snapshots_dir)?;

    let config_path = snapshot.join("config.json");
    let tokenizer_path = snapshot.join("tokenizer.json");

    if !config_path.exists() {
        anyhow::bail!("config.json not found in snapshot: {}", snapshot.display());
    }
    if !tokenizer_path.exists() {
        anyhow::bail!(
            "tokenizer.json not found in snapshot: {}",
            snapshot.display()
        );
    }

    let weight_paths = collect_weight_files(&snapshot)?;

    Ok(ModelFiles {
        config_path,
        tokenizer_path,
        weight_paths,
    })
}

fn find_latest_snapshot(snapshots_dir: &Path) -> Result<PathBuf> {
    let mut entries = std::fs::read_dir(snapshots_dir)
        .context("failed to read snapshots dir")?
        .filter_map(|e| e.ok())
        .filter(|e| e.path().is_dir())
        .collect::<Vec<_>>();

    entries.sort_by_key(|e| {
        e.metadata()
            .and_then(|m| m.modified())
            .unwrap_or(SystemTime::UNIX_EPOCH)
    });

    entries
        .last()
        .map(|e| e.path())
        .ok_or_else(|| anyhow::anyhow!("no snapshots found"))
}

fn collect_weight_files(snapshot: &Path) -> Result<Vec<PathBuf>> {
    // Try sharded index first, then single file
    let index_path = snapshot.join("model.safetensors.index.json");
    if index_path.exists() {
        let index_bytes = std::fs::read(&index_path)?;
        let index: serde_json::Value = serde_json::from_slice(&index_bytes)?;
        let files: std::collections::HashSet<String> = index["weight_map"]
            .as_object()
            .context("invalid index.json")?
            .values()
            .filter_map(|v| v.as_str().map(String::from))
            .collect();
        let mut paths: Vec<PathBuf> = files
            .into_iter()
            .map(|f| snapshot.join(&f))
            .collect();
        paths.sort();
        return Ok(paths);
    }

    let single = snapshot.join("model.safetensors");
    if single.exists() {
        return Ok(vec![single]);
    }

    anyhow::bail!(
        "no safetensors weights found in snapshot: {}",
        snapshot.display()
    );
}

// ── Downloader (async, cache-first) ──────────────────────────────────────────

/// Download a model from HuggingFace Hub if not already cached.
/// Uses `hf-hub`'s built-in cache-first logic (checks ETag, no re-download).
pub async fn download_model(model_id: &str, cache_dir: &Path) -> Result<ModelInfo> {
    std::fs::create_dir_all(cache_dir)
        .with_context(|| format!("failed to create cache dir: {}", cache_dir.display()))?;

    info!(%model_id, "checking model cache...");

    let mut builder = ApiBuilder::new().with_cache_dir(cache_dir.to_path_buf());
    if let Ok(token) = std::env::var("HF_TOKEN") {
        builder = builder.with_token(Some(token));
    }
    let api = builder.build().context("failed to build HF Hub API")?;

    let repo = api.model(model_id.to_string());

    // Download metadata files (cache-first: no network if already cached)
    info!("fetching config.json");
    let config_path = repo
        .get("config.json")
        .await
        .with_context(|| format!("failed to fetch config.json for {model_id}"))?;

    info!("fetching tokenizer.json");
    let tokenizer_path = repo
        .get("tokenizer.json")
        .await
        .with_context(|| format!("failed to fetch tokenizer.json for {model_id}"))?;

    // Parse config to determine architecture and sharding
    let cfg_bytes = std::fs::read(&config_path)?;
    let hf_cfg: HfModelConfig = serde_json::from_slice(&cfg_bytes)
        .context("failed to parse config.json")?;

    let architecture = detect_architecture(&hf_cfg)
        .with_context(|| format!("unsupported model architecture for {model_id}"))?;

    let num_layers = hf_cfg.num_hidden_layers.unwrap_or(0);
    let context_length = hf_cfg.max_position_embeddings.unwrap_or(4096);
    let vocab_size = hf_cfg.vocab_size.unwrap_or(0);

    // Download weight files
    let weight_paths = download_weights(&repo, &config_path).await?;

    let loaded_at = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);

    let cache_path = config_path
        .parent()
        .unwrap_or(cache_dir)
        .to_path_buf();

    info!(
        %model_id,
        architecture = %architecture,
        num_layers,
        context_length,
        "model ready"
    );

    Ok(ModelInfo {
        id: model_id.to_string(),
        architecture,
        num_layers,
        context_length,
        vocab_size,
        cache_path,
        loaded_at,
    })
}

fn detect_architecture(cfg: &HfModelConfig) -> Result<ModelArchitecture> {
    let archs = cfg.architectures.as_deref().unwrap_or(&[]);
    for arch in archs {
        let lower = arch.to_lowercase();
        if lower.contains("llama") {
            return Ok(ModelArchitecture::Llama);
        }
        if lower.contains("mistral") {
            return Ok(ModelArchitecture::Mistral);
        }
    }
    anyhow::bail!("unknown architecture: {:?}", archs)
}

async fn download_weights(
    repo: &hf_hub::api::tokio::ApiRepo,
    config_path: &Path,
) -> Result<Vec<PathBuf>> {
    let snapshot_dir = config_path.parent().unwrap();

    // Check for sharded index
    let index_result = repo.get("model.safetensors.index.json").await;
    if let Ok(index_path) = index_result {
        let index_bytes = std::fs::read(&index_path)?;
        let index: serde_json::Value = serde_json::from_slice(&index_bytes)?;
        let shard_names: std::collections::HashSet<String> = index["weight_map"]
            .as_object()
            .context("invalid index.json")?
            .values()
            .filter_map(|v| v.as_str().map(String::from))
            .collect();

        let mut paths = Vec::new();
        for shard in &shard_names {
            info!("fetching weight shard: {shard}");
            let path = repo
                .get(shard)
                .await
                .with_context(|| format!("failed to fetch shard {shard}"))?;
            paths.push(path);
        }
        paths.sort();
        return Ok(paths);
    }

    // Single weight file
    warn!("no shard index found, trying model.safetensors");
    let path = repo
        .get("model.safetensors")
        .await
        .context("failed to fetch model.safetensors")?;
    Ok(vec![path])
}

