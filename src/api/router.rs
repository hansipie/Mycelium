use std::sync::Arc;

use axum::{
    routing::{get, post},
    Router,
};

use crate::{
    api::{chat, health, models},
    config::ServerConfig,
    inference::{backend::InferenceBackend, local::LocalInferenceBackend},
};

// ── AppState ──────────────────────────────────────────────────────────────────

#[derive(Clone)]
pub struct AppState {
    pub backend: Arc<dyn InferenceBackend>,
    pub model_id: String,
    pub node_name: String,
}

// ── Router builders ───────────────────────────────────────────────────────────

/// Build router by downloading/loading the model then wiring `LocalInferenceBackend`.
pub async fn build_router(cfg: ServerConfig) -> anyhow::Result<Router> {
    let backend = Arc::new(LocalInferenceBackend::from_config(&cfg).await?);
    build_router_with(cfg, backend).await
}

/// Build router with an explicit backend (used in tests and US2 wiring).
pub async fn build_router_with(
    cfg: ServerConfig,
    backend: Arc<dyn InferenceBackend>,
) -> anyhow::Result<Router> {
    let state = AppState {
        model_id: cfg.model.id.clone(),
        node_name: cfg.node.name.clone(),
        backend,
    };

    let router = Router::new()
        .route("/v1/chat/completions", post(chat::chat_completions))
        .route("/v1/models", get(models::list_models))
        .route("/health", get(health::health_check))
        .with_state(state);

    Ok(router)
}

// ── Mock backend (Phase 1 / US1) ──────────────────────────────────────────────

struct MockInferenceBackend {
    model_id: String,
}

impl MockInferenceBackend {
    fn new(model_id: String) -> Self {
        Self { model_id }
    }
}

#[async_trait::async_trait]
impl InferenceBackend for MockInferenceBackend {
    async fn generate(
        &self,
        req: crate::inference::backend::GenerateRequest,
    ) -> Result<crate::inference::backend::TokenStream, crate::inference::backend::InferenceError>
    {
        use crate::inference::backend::Token;
        use futures_util::stream;

        let prompt_echo = format!("Echo: {}", req.prompt);
        let tokens = vec![
            Ok(Token {
                id: 0,
                text: prompt_echo,
                is_eos: false,
            }),
            Ok(Token {
                id: 1,
                text: String::new(),
                is_eos: true,
            }),
        ];
        Ok(Box::pin(stream::iter(tokens)))
    }

    fn model_id(&self) -> Option<String> {
        Some(self.model_id.clone())
    }
}
