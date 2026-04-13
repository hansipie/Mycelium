use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use candle_core::Device;
use futures_util::stream;
use tracing::info;

use crate::{
    config::ServerConfig,
    inference::backend::{GenerateRequest, InferenceBackend, InferenceError, Token, TokenStream},
    model::{engine::Engine, loader::download_model},
};

// ── LocalInferenceBackend ─────────────────────────────────────────────────────

/// CPU inference backend backed by a candle Llama engine.
///
/// Concurrency model (Phase 1):
/// - The `Engine` owns the KV cache which is mutated during generation.
/// - Requests are serialized via `Mutex` — CPU inference is single-threaded anyway.
/// - SC-004 (≥2 concurrent requests) is satisfied: requests queue correctly,
///   no corruption, no crash.
pub struct LocalInferenceBackend {
    engine: Arc<Mutex<Engine>>,
    model_id: String,
}

impl LocalInferenceBackend {
    /// Download (or load from cache) the model and build the backend.
    pub async fn from_config(cfg: &ServerConfig) -> anyhow::Result<Self> {
        let model_info = download_model(&cfg.model.id, &cfg.model.cache_dir).await?;
        let model_id = model_info.id.clone();

        let files =
            crate::model::loader::resolve_model_files(&cfg.model.cache_dir, &cfg.model.id)?;

        let device = Device::Cpu;
        let engine = Engine::load(
            model_info,
            &files.config_path,
            &files.tokenizer_path,
            &files.weight_paths,
            &device,
        )?;

        info!(%model_id, "model loaded, backend ready");

        Ok(Self {
            engine: Arc::new(Mutex::new(engine)),
            model_id,
        })
    }
}

#[async_trait]
impl InferenceBackend for LocalInferenceBackend {
    async fn generate(&self, req: GenerateRequest) -> Result<TokenStream, InferenceError> {
        let engine = Arc::clone(&self.engine);
        let prompt = req.prompt.clone();
        let config = req.config.clone();

        // Run blocking CPU inference on a dedicated thread pool thread.
        // block_in_place is correct here: we hold no async resources while blocking.
        let tokens = tokio::task::block_in_place(move || {
            let mut eng = engine
                .lock()
                .map_err(|e| InferenceError::ModelNotReady(e.to_string()))?;

            let mut collected: Vec<Token> = Vec::new();
            eng.generate_tokens(&prompt, &config, |token| {
                collected.push(token);
                Ok(false) // continue until EOS or max_tokens
            })?;
            Ok::<Vec<Token>, InferenceError>(collected)
        })?;

        let token_stream = stream::iter(tokens.into_iter().map(Ok::<Token, InferenceError>));
        Ok(Box::pin(token_stream))
    }

    fn model_id(&self) -> Option<String> {
        Some(self.model_id.clone())
    }
}
