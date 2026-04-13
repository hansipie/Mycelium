use std::sync::Arc;
use axum::{body::Body, http::{Request, StatusCode}};
use tower::ServiceExt;
use mycelium::{
    api::router::build_router_with,
    config::ServerConfig,
    inference::backend::{GenerateRequest, InferenceBackend, InferenceError, Token, TokenStream},
};

struct MockBackend { model: String }
impl MockBackend { fn new() -> Self { Self { model: "mock-model".to_string() } } }

#[async_trait::async_trait]
impl InferenceBackend for MockBackend {
    async fn generate(&self, _req: GenerateRequest) -> Result<TokenStream, InferenceError> {
        let tokens = vec![
            Ok(Token { id: 0, text: "Hello".into(), is_eos: false }),
            Ok(Token { id: 1, text: " world".into(), is_eos: true }),
        ];
        Ok(Box::pin(futures_util::stream::iter(tokens)))
    }
    fn model_id(&self) -> Option<String> { Some(self.model.clone()) }
}

async fn make_app() -> axum::Router {
    build_router_with(ServerConfig::default(), Arc::new(MockBackend::new())).await.unwrap()
}

fn json_req(v: serde_json::Value) -> Request<Body> {
    Request::builder()
        .method("POST").uri("/v1/chat/completions")
        .header("content-type", "application/json")
        .body(Body::from(serde_json::to_vec(&v).unwrap()))
        .unwrap()
}

#[tokio::test]
async fn non_streaming_returns_complete_json() {
    let resp = make_app().await.oneshot(json_req(serde_json::json!({
        "model": "mock-model",
        "messages": [{"role": "user", "content": "Hi"}],
        "stream": false
    }))).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let b = axum::body::to_bytes(resp.into_body(), usize::MAX).await.unwrap();
    let j: serde_json::Value = serde_json::from_slice(&b).unwrap();
    assert_eq!(j["object"], "chat.completion");
    assert!(j["choices"][0]["message"]["content"].is_string());
    assert!(j["usage"]["prompt_tokens"].is_number());
    assert!(j["usage"]["completion_tokens"].is_number());
}

#[tokio::test]
async fn streaming_returns_sse_with_done() {
    let resp = make_app().await.oneshot(json_req(serde_json::json!({
        "model": "mock-model",
        "messages": [{"role": "user", "content": "Hi"}],
        "stream": true
    }))).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let ct = resp.headers().get("content-type").unwrap().to_str().unwrap();
    assert!(ct.starts_with("text/event-stream"), "got {ct}");
    let b = axum::body::to_bytes(resp.into_body(), usize::MAX).await.unwrap();
    let s = std::str::from_utf8(&b).unwrap();
    assert!(s.contains("data:"), "missing SSE data lines");
    assert!(s.contains("[DONE]"), "missing [DONE] terminator");
}

#[tokio::test]
async fn empty_messages_returns_422() {
    let resp = make_app().await.oneshot(json_req(serde_json::json!({
        "model": "mock-model", "messages": []
    }))).await.unwrap();
    assert_eq!(resp.status(), StatusCode::UNPROCESSABLE_ENTITY);
}

#[tokio::test]
async fn invalid_temperature_returns_422() {
    let resp = make_app().await.oneshot(json_req(serde_json::json!({
        "model": "mock-model",
        "messages": [{"role": "user", "content": "Hi"}],
        "temperature": 5.0
    }))).await.unwrap();
    assert_eq!(resp.status(), StatusCode::UNPROCESSABLE_ENTITY);
}
