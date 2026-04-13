use std::sync::Arc;

use axum::{
    body::Body,
    http::{Request, StatusCode},
};
use futures_util::StreamExt;
use tower::ServiceExt;

use mycelium::{
    api::router::build_router,
    config::ServerConfig,
    inference::backend::{GenerateRequest, InferenceBackend, InferenceError, Token, TokenStream},
};

// ── Mock backend ──────────────────────────────────────────────────────────────

struct MockBackend {
    tokens: Vec<&'static str>,
    model: String,
}

impl MockBackend {
    fn new(tokens: Vec<&'static str>) -> Self {
        Self {
            tokens,
            model: "mock-model".to_string(),
        }
    }
}

#[async_trait::async_trait]
impl InferenceBackend for MockBackend {
    async fn generate(&self, _req: GenerateRequest) -> Result<TokenStream, InferenceError> {
        let tokens: Vec<Result<Token, InferenceError>> = self
            .tokens
            .iter()
            .enumerate()
            .map(|(i, t)| {
                Ok(Token {
                    id: i as u32,
                    text: t.to_string(),
                    is_eos: false,
                })
            })
            .collect();
        let stream = futures_util::stream::iter(tokens);
        Ok(Box::pin(stream))
    }

    fn model_id(&self) -> Option<String> {
        Some(self.model.clone())
    }
}

async fn test_app() -> axum::Router {
    let cfg = ServerConfig::default();
    build_router_with_backend(cfg, Arc::new(MockBackend::new(vec!["Hello", " world"])))
        .await
        .unwrap()
}

// Helper to build router with an explicit backend (for testing)
async fn build_router_with_backend(
    cfg: ServerConfig,
    backend: Arc<dyn InferenceBackend>,
) -> anyhow::Result<axum::Router> {
    mycelium::api::router::build_router_with(cfg, backend).await
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[tokio::test]
async fn post_chat_non_streaming_returns_complete_json() {
    let app = test_app().await;

    let body = serde_json::json!({
        "model": "mock-model",
        "messages": [{"role": "user", "content": "Hi"}],
        "stream": false
    });

    let resp = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header("content-type", "application/json")
                .body(Body::from(serde_json::to_string(&body).unwrap()))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(resp.status(), StatusCode::OK);

    let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX)
        .await
        .unwrap();
    let json: serde_json::Value = serde_json::from_slice(&bytes).unwrap();

    assert_eq!(json["object"], "chat.completion");
    assert!(json["choices"][0]["message"]["content"].is_string());
    assert!(json["usage"]["prompt_tokens"].is_number());
    assert!(json["usage"]["completion_tokens"].is_number());
}

#[tokio::test]
async fn post_chat_streaming_returns_sse_events_and_done() {
    let app = test_app().await;

    let body = serde_json::json!({
        "model": "mock-model",
        "messages": [{"role": "user", "content": "Hi"}],
        "stream": true
    });

    let resp = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header("content-type", "application/json")
                .body(Body::from(serde_json::to_string(&body).unwrap()))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(resp.status(), StatusCode::OK);
    assert_eq!(
        resp.headers()
            .get("content-type")
            .unwrap()
            .to_str()
            .unwrap(),
        "text/event-stream"
    );

    let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX)
        .await
        .unwrap();
    let body_str = std::str::from_utf8(&bytes).unwrap();

    assert!(body_str.contains("data:"), "should have SSE data lines");
    assert!(body_str.contains("[DONE]"), "should terminate with [DONE]");
}

#[tokio::test]
async fn post_chat_empty_messages_returns_422() {
    let app = test_app().await;

    let body = serde_json::json!({
        "model": "mock-model",
        "messages": []
    });

    let resp = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header("content-type", "application/json")
                .body(Body::from(serde_json::to_string(&body).unwrap()))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(resp.status(), StatusCode::UNPROCESSABLE_ENTITY);
}

#[tokio::test]
async fn post_chat_invalid_temperature_returns_422() {
    let app = test_app().await;

    let body = serde_json::json!({
        "model": "mock-model",
        "messages": [{"role": "user", "content": "Hi"}],
        "temperature": 5.0
    });

    let resp = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header("content-type", "application/json")
                .body(Body::from(serde_json::to_string(&body).unwrap()))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(resp.status(), StatusCode::UNPROCESSABLE_ENTITY);
}
