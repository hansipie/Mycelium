use std::sync::Arc;

use axum::{
    body::Body,
    http::{Request, StatusCode},
};
use tower::ServiceExt;

use mycelium::{
    config::ServerConfig,
    inference::backend::{GenerateRequest, InferenceBackend, InferenceError, TokenStream},
};

struct MockBackend;

#[async_trait::async_trait]
impl InferenceBackend for MockBackend {
    async fn generate(&self, _req: GenerateRequest) -> Result<TokenStream, InferenceError> {
        Err(InferenceError::ModelNotReady("mock".into()))
    }

    fn model_id(&self) -> Option<String> {
        Some("mock-model".to_string())
    }
}

async fn test_app() -> axum::Router {
    let cfg = ServerConfig::default();
    mycelium::api::router::build_router_with(cfg, Arc::new(MockBackend))
        .await
        .unwrap()
}

#[tokio::test]
async fn get_models_returns_openai_format() {
    let app = test_app().await;

    let resp = app
        .oneshot(
            Request::builder()
                .method("GET")
                .uri("/v1/models")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(resp.status(), StatusCode::OK);

    let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX)
        .await
        .unwrap();
    let json: serde_json::Value = serde_json::from_slice(&bytes).unwrap();

    assert_eq!(json["object"], "list");
    assert!(json["data"].is_array());
    assert!(!json["data"].as_array().unwrap().is_empty());
    assert_eq!(json["data"][0]["object"], "model");
}

#[tokio::test]
async fn get_health_returns_status() {
    let app = test_app().await;

    let resp = app
        .oneshot(
            Request::builder()
                .method("GET")
                .uri("/health")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(resp.status(), StatusCode::OK);

    let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX)
        .await
        .unwrap();
    let json: serde_json::Value = serde_json::from_slice(&bytes).unwrap();

    assert!(json["status"].is_string());
}
