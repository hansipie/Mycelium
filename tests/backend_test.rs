use futures_util::StreamExt;

use mycelium::inference::backend::{
    GenerateRequest, GenerationConfig, InferenceBackend, InferenceError, Token, TokenStream,
};

// ── Mock backend ──────────────────────────────────────────────────────────────

struct EchoBackend {
    reply: Vec<String>,
}

#[async_trait::async_trait]
impl InferenceBackend for EchoBackend {
    async fn generate(&self, _req: GenerateRequest) -> Result<TokenStream, InferenceError> {
        let tokens: Vec<Result<Token, InferenceError>> = self
            .reply
            .iter()
            .enumerate()
            .map(|(i, t)| {
                Ok(Token {
                    id: i as u32,
                    text: t.clone(),
                    is_eos: i == self.reply.len() - 1,
                })
            })
            .collect();
        Ok(Box::pin(futures_util::stream::iter(tokens)))
    }

    fn model_id(&self) -> Option<String> {
        Some("echo".to_string())
    }
}

struct ErrorBackend;

#[async_trait::async_trait]
impl InferenceBackend for ErrorBackend {
    async fn generate(&self, _req: GenerateRequest) -> Result<TokenStream, InferenceError> {
        Err(InferenceError::ModelNotReady("not loaded".into()))
    }

    fn model_id(&self) -> Option<String> {
        None
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[tokio::test]
async fn echo_backend_streams_tokens_and_terminates() {
    let backend = EchoBackend {
        reply: vec!["Hello".to_string(), " world".to_string()],
    };

    let req = GenerateRequest {
        prompt: "test".to_string(),
        config: GenerationConfig::default(),
    };

    let mut stream = backend.generate(req).await.unwrap();

    let mut collected = Vec::new();
    while let Some(result) = stream.next().await {
        collected.push(result.unwrap());
    }

    assert_eq!(collected.len(), 2);
    assert_eq!(collected[0].text, "Hello");
    assert_eq!(collected[1].text, " world");
    assert!(collected[1].is_eos);
}

#[tokio::test]
async fn error_backend_returns_error() {
    let backend = ErrorBackend;

    let req = GenerateRequest {
        prompt: "test".to_string(),
        config: GenerationConfig::default(),
    };

    let result = backend.generate(req).await;
    let err = match result {
        Err(e) => e,
        Ok(_) => panic!("expected an error but generate succeeded"),
    };
    assert!(matches!(err, InferenceError::ModelNotReady(_)));
}

#[tokio::test]
async fn generation_config_has_correct_defaults() {
    let cfg = GenerationConfig::default();
    assert_eq!(cfg.temperature, 0.7);
    assert_eq!(cfg.top_p, 0.9);
    assert_eq!(cfg.max_tokens, 256);
    assert_eq!(cfg.repeat_penalty, 1.1);
}
