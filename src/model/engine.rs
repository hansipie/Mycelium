use std::path::PathBuf;

use anyhow::{Context, Result};
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::{
    generation::{LogitsProcessor, Sampling},
    models::llama::{Cache, Config, Llama, LlamaConfig, LlamaEosToks},
    utils::apply_repeat_penalty,
};
use tokenizers::Tokenizer;

use crate::{
    config::settings::ModelArchitecture,
    inference::backend::{GenerationConfig, InferenceError, Token},
    model::tokenizer::TokenOutputStream,
};

// ── Public data types ─────────────────────────────────────────────────────────

/// Metadata about a loaded model.
#[derive(Debug, Clone)]
pub struct ModelInfo {
    pub id: String,
    pub architecture: ModelArchitecture,
    pub num_layers: usize,
    pub context_length: usize,
    pub vocab_size: usize,
    pub cache_path: PathBuf,
    pub loaded_at: u64,
}

// ── Engine ────────────────────────────────────────────────────────────────────

/// Wraps a loaded Llama model with its KV cache and tokenizer.
/// Phase 1 supports Llama only. Mistral support added in Phase 2 when
/// candle-transformers ships a stable Mistral module.
pub struct Engine {
    pub info: ModelInfo,
    model: Llama,
    cache: Cache,
    eos_tokens: Vec<u32>,
    tokenizer: Tokenizer,
    device: Device,
}

impl Engine {
    /// Load a model from a pre-populated cache directory.
    pub fn load(
        info: ModelInfo,
        config_path: &PathBuf,
        tokenizer_path: &PathBuf,
        weight_paths: &[PathBuf],
        device: &Device,
    ) -> Result<Self> {
        if info.architecture != ModelArchitecture::Llama {
            anyhow::bail!(
                "Phase 1 supports Llama only; got {:?}",
                info.architecture
            );
        }

        let tokenizer = Tokenizer::from_file(tokenizer_path)
            .map_err(|e| anyhow::anyhow!("tokenizer load failed: {e}"))?;

        let cfg_bytes = std::fs::read(config_path)?;
        let llama_cfg: LlamaConfig = serde_json::from_slice(&cfg_bytes)
            .context("failed to parse LlamaConfig")?;

        let eos_tokens = match &llama_cfg.eos_token_id {
            Some(LlamaEosToks::Single(id)) => vec![*id],
            Some(LlamaEosToks::Multiple(ids)) => ids.clone(),
            None => vec![2], // Llama 2 default
        };

        let cfg: Config = llama_cfg.into_config(false /* no flash attn on CPU */);

        let dtype = DType::F32;
        // SAFETY: mmap is valid while we hold the file open; candle validates shapes.
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(weight_paths, dtype, device)
                .context("failed to mmap safetensors")?
        };

        let use_kv_cache = true;
        let cache = Cache::new(use_kv_cache, dtype, &cfg, device)
            .context("failed to create KV cache")?;
        let model = Llama::load(vb, &cfg).context("failed to load Llama")?;

        Ok(Self {
            info,
            model,
            cache,
            eos_tokens,
            tokenizer,
            device: device.clone(),
        })
    }

    /// Apply the Llama 3 chat template to a list of (role, content) pairs.
    pub fn apply_chat_template(&self, messages: &[(String, String)]) -> String {
        let mut out = String::from("<|begin_of_text|>");
        for (role, content) in messages {
            out.push_str(&format!(
                "<|start_header_id|>{role}<|end_header_id|>\n\n{content}<|eot_id|>"
            ));
        }
        out.push_str("<|start_header_id|>assistant<|end_header_id|>\n\n");
        out
    }

    /// Run the token generation loop.
    /// Calls `on_token` for each generated token; stops when `on_token` returns
    /// `Ok(true)`, on EOS, or when `config.max_tokens` is reached.
    pub fn generate_tokens(
        &mut self,
        prompt: &str,
        config: &GenerationConfig,
        mut on_token: impl FnMut(Token) -> Result<bool, InferenceError>,
    ) -> Result<(), InferenceError> {
        let encoding = self
            .tokenizer
            .encode(prompt, true)
            .map_err(|e| InferenceError::TokenizationFailed(e.to_string()))?;

        let input_ids = encoding.get_ids();

        if input_ids.len() >= self.info.context_length {
            return Err(InferenceError::ContextTooLong {
                tokens: input_ids.len(),
                limit: self.info.context_length,
            });
        }

        let mut logits_processor = LogitsProcessor::from_sampling(
            config.seed,
            Sampling::TopP {
                p: config.top_p as f64,
                temperature: config.temperature as f64,
            },
        );

        let mut token_stream = TokenOutputStream::new(self.tokenizer.clone());
        let mut all_tokens: Vec<u32> = input_ids.to_vec();
        let mut index_pos = 0usize;

        for step in 0..config.max_tokens {
            let (input_slice, curr_index_pos) = if step == 0 {
                (all_tokens.as_slice(), 0usize)
            } else {
                let last = all_tokens.len() - 1;
                (&all_tokens[last..], index_pos)
            };

            let input_tensor = Tensor::new(input_slice, &self.device)
                .and_then(|t| t.unsqueeze(0))
                .map_err(|e| InferenceError::GenerationFailed(e.to_string()))?;

            // forward: [1, seq_len, vocab_size]
            let logits = self
                .model
                .forward(&input_tensor, curr_index_pos, &mut self.cache)
                .map_err(|e| InferenceError::GenerationFailed(e.to_string()))?;

            // Extract last token logits → [vocab_size]
            let seq_len = logits.dim(1)
                .map_err(|e| InferenceError::GenerationFailed(e.to_string()))?;
            let logits = logits
                .squeeze(0)
                .and_then(|t| t.narrow(0, seq_len - 1, 1))
                .and_then(|t| t.squeeze(0))
                .map_err(|e| InferenceError::GenerationFailed(e.to_string()))?;

            // Repeat penalty over recent context
            let logits = if config.repeat_penalty != 1.0 {
                let start = all_tokens.len().saturating_sub(64);
                apply_repeat_penalty(&logits, config.repeat_penalty, &all_tokens[start..])
                    .map_err(|e| InferenceError::GenerationFailed(e.to_string()))?
            } else {
                logits
            };

            let next_token = logits_processor
                .sample(&logits)
                .map_err(|e| InferenceError::GenerationFailed(e.to_string()))?;

            index_pos = curr_index_pos + input_slice.len();
            all_tokens.push(next_token);

            let is_eos = self.eos_tokens.contains(&next_token);

            if let Some(text) = token_stream
                .next_token(next_token)
                .map_err(|e| InferenceError::TokenizationFailed(e.to_string()))?
            {
                let stop = on_token(Token { id: next_token, text, is_eos })?;
                if stop || is_eos {
                    break;
                }
            } else if is_eos {
                if let Some(text) = token_stream.flush() {
                    if !text.is_empty() {
                        on_token(Token { id: next_token, text, is_eos: true })?;
                    }
                }
                break;
            }
        }

        // Signal end of stream
        on_token(Token { id: 0, text: String::new(), is_eos: true })?;
        Ok(())
    }
}
