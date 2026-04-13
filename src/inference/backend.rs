use std::pin::Pin;

use async_trait::async_trait;
use futures_util::Stream;
use thiserror::Error;

// ── Errors ────────────────────────────────────────────────────────────────────

#[derive(Debug, Error)]
pub enum InferenceError {
    #[error("model not ready: {0}")]
    ModelNotReady(String),

    #[error("generation failed: {0}")]
    GenerationFailed(String),

    #[error("tokenization failed: {0}")]
    TokenizationFailed(String),

    #[error("context too long: {tokens} tokens exceeds limit of {limit}")]
    ContextTooLong { tokens: usize, limit: usize },
}

// ── Core types ────────────────────────────────────────────────────────────────

/// A single generated token with its decoded text fragment.
#[derive(Debug, Clone)]
pub struct Token {
    pub id: u32,
    /// Decoded text fragment (may be empty if codepoint is not yet complete).
    pub text: String,
    /// True when this token is the EOS token.
    pub is_eos: bool,
}

/// Parameters for a generation request passed to the backend.
#[derive(Debug, Clone)]
pub struct GenerateRequest {
    /// Prompt already formatted with the model chat template.
    pub prompt: String,
    pub config: GenerationConfig,
}

/// Sampling parameters with sensible defaults.
#[derive(Debug, Clone)]
pub struct GenerationConfig {
    pub temperature: f32,
    pub top_p: f32,
    pub max_tokens: usize,
    pub seed: u64,
    pub repeat_penalty: f32,
}

impl Default for GenerationConfig {
    fn default() -> Self {
        Self {
            temperature: 0.7,
            top_p: 0.9,
            max_tokens: 256,
            seed: rand_seed(),
            repeat_penalty: 1.1,
        }
    }
}

fn rand_seed() -> u64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos() as u64)
        .unwrap_or(42)
}

// ── Trait ─────────────────────────────────────────────────────────────────────

pub type TokenStream =
    Pin<Box<dyn Stream<Item = Result<Token, InferenceError>> + Send>>;

#[async_trait]
pub trait InferenceBackend: Send + Sync {
    /// Stream tokens for the given request.
    async fn generate(&self, req: GenerateRequest) -> Result<TokenStream, InferenceError>;

    /// Identifier of the currently loaded model, if any.
    fn model_id(&self) -> Option<String>;
}
