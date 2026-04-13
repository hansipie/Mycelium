use std::sync::Arc;
use axum::{body::Body, http::{Request, StatusCode}};
use tower::ServiceExt;
use mycelium::{
    api::router::build_router_with,
    config::ServerConfig,
    inference::backend::{GenerateRequest, InferenceBackend, InferenceError, TokenStream},
};

struct MockBackend;

#[async_trait::async_trait]
impl InferenceBackend for MockBackend {
    async fn generate(&self, _req: GenerateRequest) -> Result<TokenStream, InferenceError> {
        Err(InferenceError::ModelNotReady("mock".into()))
    }
    fn model_id(&self) -> Option<String> { Some("mock-model".to_string()) }
}

async fn make_app() -> axum::Router {
    build_router_with(ServerConfig::default(), Arc::new(MockBackend)).await.unwrap()
}

#[tokio::test]
async fn get_models_returns_openai_format() {
    let resp = make_app().await.oneshot(
        Request::builder().method("GET").uri("/v1/models").body(Body::empty()).unwrap()
    ).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let b = axum::body::to_bytes(resp.into_body(), usize::MAX).await.unwrap();
    let j: serde_json::Value = serde_json::from_slice(&b).unwrap();
    assert_eq!(j["object"], "list");
    assert!(j["data"].is_array());
    assert!(!j["data"].as_array().unwrap().is_empty());
    assert_eq!(j["data"][0]["object"], "model");
}

#[tokio::test]
async fn get_health_returns_status() {
    let resp = make_app().await.oneshot(
        Request::builder().method("GET").uri("/health").body(Body::empty()).unwrap()
    ).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let b = axum::body::to_bytes(resp.into_body(), usize::MAX).await.unwrap();
    let j: serde_json::Value = serde_json::from_slice(&b).unwrap();
    assert!(j["status"].is_string());
}
