use std::sync::Arc;

use axum::{
    body::Body,
    http::{Request, StatusCode},
};
use tower::ServiceExt;

use mycelium::{
    api::router::build_router_with,
    config::ServerConfig,
    inference::backend::{GenerateRequest, InferenceBackend, InferenceError, Token, TokenStream},
};

// ── Mock backend ──────────────────────────────────────────────────────────────

struct MockBackend;

#[async_trait::async_trait]
impl InferenceBackend for MockBackend {
    async fn generate(&self, _req: GenerateRequest) -> Result<TokenStream, InferenceError> {
        let tokens = vec![
            Ok(Token { id: 0, text: "Hello".into(), is_eos: false }),
            Ok(Token { id: 1, text: " world".into(), is_eos: false }),
            Ok(Token { id: 2, text: "!".into(), is_eos: true }),
        ];
        Ok(Box::pin(futures_util::stream::iter(tokens)))
    }

    fn model_id(&self) -> Option<String> {
        Some("mock-model".to_string())
    }
}

async fn test_app() -> axum::Router {
    build_router_with(ServerConfig::default(), Arc::new(MockBackend))
        .await
        .unwrap()
}

fn post_chat(body: serde_json::Value) -> Request<Body> {
    Request::builder()
        .method("POST")
        .uri("/v1/chat/completions")
        .header("content-type", "application/json")
        .body(Body::from(serde_json::to_vec(&body).unwrap()))
        .unwrap()
}

// ── T035: Flow e2e complet (backend mock) ─────────────────────────────────────

#[tokio::test]
async fn e2e_non_streaming_full_flow() {
    let app = test_app().await;

    let resp = app
        .oneshot(post_chat(serde_json::json!({
            "model": "mock-model",
            "messages": [{"role": "user", "content": "Bonjour !"}],
            "stream": false
        })))
        .await
        .unwrap();

    assert_eq!(resp.status(), StatusCode::OK);

    let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX).await.unwrap();
    let json: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    assert!(json["choices"][0]["message"]["content"].as_str().unwrap().len() > 0);
}

#[tokio::test]
async fn e2e_streaming_full_flow() {
    let app = test_app().await;

    let resp = app
        .oneshot(post_chat(serde_json::json!({
            "model": "mock-model",
            "messages": [{"role": "user", "content": "Bonjour !"}],
            "stream": true
        })))
        .await
        .unwrap();

    assert_eq!(resp.status(), StatusCode::OK);
    let body = axum::body::to_bytes(resp.into_body(), usize::MAX).await.unwrap();
    let text = std::str::from_utf8(&body).unwrap();
    assert!(text.contains("data:"));
    assert!(text.contains("[DONE]"));
}

#[tokio::test]
async fn e2e_models_endpoint() {
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
    let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX).await.unwrap();
    let json: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(json["object"], "list");
    assert!(!json["data"].as_array().unwrap().is_empty());
}

#[tokio::test]
async fn e2e_health_endpoint() {
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
    let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX).await.unwrap();
    let json: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(json["status"], "ready");
    assert!(json["model"].is_string());
}

// ── T037: Conformité OpenAI ───────────────────────────────────────────────────

#[tokio::test]
async fn openai_conformance_non_streaming_required_fields() {
    let app = test_app().await;

    let resp = app
        .oneshot(post_chat(serde_json::json!({
            "model": "mock-model",
            "messages": [{"role": "user", "content": "Hi"}],
            "stream": false
        })))
        .await
        .unwrap();

    let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX).await.unwrap();
    let j: serde_json::Value = serde_json::from_slice(&bytes).unwrap();

    // Champs obligatoires OpenAI
    assert!(j["id"].as_str().unwrap().starts_with("chatcmpl-"), "id doit commencer par chatcmpl-");
    assert_eq!(j["object"], "chat.completion");
    assert!(j["created"].is_number(), "created doit être un timestamp");
    assert!(j["model"].is_string(), "model est requis");

    // choices
    let choices = j["choices"].as_array().unwrap();
    assert!(!choices.is_empty(), "choices ne doit pas être vide");
    let choice = &choices[0];
    assert_eq!(choice["index"], 0);
    assert_eq!(choice["message"]["role"], "assistant");
    assert!(choice["message"]["content"].is_string());
    assert_eq!(choice["finish_reason"], "stop", "finish_reason doit être 'stop'");

    // usage (non-streaming uniquement)
    assert!(j["usage"]["prompt_tokens"].is_number());
    assert!(j["usage"]["completion_tokens"].is_number());
    assert!(j["usage"]["total_tokens"].is_number());
    let total = j["usage"]["total_tokens"].as_u64().unwrap();
    let prompt = j["usage"]["prompt_tokens"].as_u64().unwrap();
    let completion = j["usage"]["completion_tokens"].as_u64().unwrap();
    assert_eq!(total, prompt + completion, "total_tokens = prompt + completion");
}

#[tokio::test]
async fn openai_conformance_streaming_chunks_and_done() {
    let app = test_app().await;

    let resp = app
        .oneshot(post_chat(serde_json::json!({
            "model": "mock-model",
            "messages": [{"role": "user", "content": "Hi"}],
            "stream": true
        })))
        .await
        .unwrap();

    assert!(
        resp.headers()
            .get("content-type")
            .unwrap()
            .to_str()
            .unwrap()
            .starts_with("text/event-stream"),
        "Content-Type doit être text/event-stream"
    );

    let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX).await.unwrap();
    let text = std::str::from_utf8(&bytes).unwrap();

    // Terminaison obligatoire OpenAI
    assert!(text.contains("data: [DONE]"), "stream doit se terminer par data: [DONE]");

    // Vérifier les chunks JSON
    let data_lines: Vec<&str> = text
        .lines()
        .filter(|l| l.starts_with("data:") && !l.contains("[DONE]"))
        .collect();
    assert!(!data_lines.is_empty(), "doit avoir au moins un chunk de données");

    let first_chunk: serde_json::Value =
        serde_json::from_str(data_lines[0].trim_start_matches("data:").trim()).unwrap();
    assert!(first_chunk["id"].as_str().unwrap().starts_with("chatcmpl-"));
    assert_eq!(first_chunk["object"], "chat.completion.chunk");
    assert!(first_chunk["created"].is_number());
    assert!(first_chunk["model"].is_string());
    let chunk_choices = first_chunk["choices"].as_array().unwrap();
    assert_eq!(chunk_choices[0]["index"], 0);
}

// ── T038: Concurrence SC-004 ──────────────────────────────────────────────────

#[tokio::test]
async fn two_concurrent_requests_succeed_without_corruption() {
    // Deux requêtes simultanées — vérifier qu'aucune ne plante
    // et que les réponses sont des JSON valides distincts.
    let app1 = test_app().await;
    let app2 = test_app().await;

    let req1 = post_chat(serde_json::json!({
        "model": "mock-model",
        "messages": [{"role": "user", "content": "First request"}],
        "stream": false
    }));
    let req2 = post_chat(serde_json::json!({
        "model": "mock-model",
        "messages": [{"role": "user", "content": "Second request"}],
        "stream": false
    }));

    let (resp1, resp2) = tokio::join!(app1.oneshot(req1), app2.oneshot(req2));

    let resp1 = resp1.unwrap();
    let resp2 = resp2.unwrap();
    assert_eq!(resp1.status(), StatusCode::OK, "première requête doit réussir");
    assert_eq!(resp2.status(), StatusCode::OK, "deuxième requête doit réussir");

    let b1 = axum::body::to_bytes(resp1.into_body(), usize::MAX).await.unwrap();
    let b2 = axum::body::to_bytes(resp2.into_body(), usize::MAX).await.unwrap();

    // Les deux corps doivent être du JSON valide (pas de corruption)
    let j1: serde_json::Value = serde_json::from_slice(&b1).expect("réponse 1 doit être JSON valide");
    let j2: serde_json::Value = serde_json::from_slice(&b2).expect("réponse 2 doit être JSON valide");

    // Chaque réponse a son propre id unique
    assert_ne!(j1["id"], j2["id"], "chaque réponse doit avoir un id distinct");
}

// ── Tests modèle réel (cache requis — cargo test --test integration) ──────────

/// Démarre le serveur complet avec LocalInferenceBackend.
/// Requiert le modèle en cache (~/.cache/mycelium/models).
/// Lancer avec : cargo test --test integration -- --include-ignored
#[tokio::test]
#[ignore = "requires model in cache — run with --include-ignored"]
async fn real_model_non_streaming() {
    use mycelium::api::router::build_router;
    let cfg = ServerConfig::default();
    let app = build_router(cfg).await.expect("failed to build router with real model");

    let resp = app
        .oneshot(post_chat(serde_json::json!({
            "model": "meta-llama/Llama-3.2-3B-Instruct",
            "messages": [{"role": "user", "content": "Say 'hello' in one word."}],
            "stream": false,
            "max_tokens": 10
        })))
        .await
        .unwrap();

    assert_eq!(resp.status(), StatusCode::OK);
    let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX).await.unwrap();
    let j: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    assert!(j["choices"][0]["message"]["content"].as_str().unwrap().len() > 0);
}

#[tokio::test]
#[ignore = "requires model in cache — run with --include-ignored"]
async fn real_model_streaming() {
    use mycelium::api::router::build_router;
    let cfg = ServerConfig::default();
    let app = build_router(cfg).await.expect("failed to build router with real model");

    let resp = app
        .oneshot(post_chat(serde_json::json!({
            "model": "meta-llama/Llama-3.2-3B-Instruct",
            "messages": [{"role": "user", "content": "Say 'hello' in one word."}],
            "stream": true,
            "max_tokens": 10
        })))
        .await
        .unwrap();

    assert_eq!(resp.status(), StatusCode::OK);
    let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX).await.unwrap();
    let text = std::str::from_utf8(&bytes).unwrap();
    assert!(text.contains("[DONE]"));
}
