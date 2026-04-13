use std::convert::Infallible;

use axum::{
    extract::State,
    http::StatusCode,
    response::{
        sse::{Event, Sse},
        IntoResponse, Response,
    },
    Json,
};
use futures_util::{stream, StreamExt};
use serde::{Deserialize, Serialize};
use tracing::{error, warn};
use uuid::Uuid;

use crate::{
    api::router::AppState,
    inference::backend::{GenerateRequest, GenerationConfig, InferenceError},
    sse::openai::{
        self, ChatCompletion, ChatCompletionChunk, Choice, ChunkChoice, CompletionMessage, Delta,
        Usage,
    },
};

// ── Request / Response types ──────────────────────────────────────────────────

#[derive(Debug, Deserialize)]
pub struct ChatRequest {
    pub model: String,
    pub messages: Vec<ChatMessage>,
    #[serde(default)]
    pub stream: bool,
    pub temperature: Option<f32>,
    pub top_p: Option<f32>,
    pub max_tokens: Option<usize>,
    pub seed: Option<u64>,
}

#[derive(Debug, Deserialize, Clone)]
pub struct ChatMessage {
    pub role: MessageRole,
    pub content: String,
}

#[derive(Debug, Deserialize, Clone, PartialEq)]
#[serde(rename_all = "lowercase")]
pub enum MessageRole {
    System,
    User,
    Assistant,
}

impl std::fmt::Display for MessageRole {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MessageRole::System => write!(f, "system"),
            MessageRole::User => write!(f, "user"),
            MessageRole::Assistant => write!(f, "assistant"),
        }
    }
}

// ── Validation ────────────────────────────────────────────────────────────────

#[derive(Debug, Serialize)]
struct ErrorResponse {
    error: ErrorDetail,
}

#[derive(Debug, Serialize)]
struct ErrorDetail {
    message: String,
    r#type: String,
    code: Option<String>,
}

fn validation_error(msg: impl Into<String>) -> Response {
    let body = ErrorResponse {
        error: ErrorDetail {
            message: msg.into(),
            r#type: "invalid_request_error".to_string(),
            code: None,
        },
    };
    (StatusCode::UNPROCESSABLE_ENTITY, Json(body)).into_response()
}

fn validate_request(req: &ChatRequest) -> Option<Response> {
    if req.messages.is_empty() {
        return Some(validation_error("messages must not be empty"));
    }
    if let Some(t) = req.temperature {
        if !(0.0..=2.0).contains(&t) {
            return Some(validation_error(
                "temperature must be between 0.0 and 2.0",
            ));
        }
    }
    if let Some(p) = req.top_p {
        if !(f32::EPSILON..=1.0).contains(&p) {
            return Some(validation_error("top_p must be in (0.0, 1.0]"));
        }
    }
    if let Some(m) = req.max_tokens {
        if m == 0 {
            return Some(validation_error("max_tokens must be > 0"));
        }
    }
    None
}

// ── Handler ───────────────────────────────────────────────────────────────────

pub async fn chat_completions(
    State(state): State<AppState>,
    Json(req): Json<ChatRequest>,
) -> Response {
    if let Some(err) = validate_request(&req) {
        return err;
    }

    let gen_config = GenerationConfig {
        temperature: req.temperature.unwrap_or(0.7),
        top_p: req.top_p.unwrap_or(0.9),
        max_tokens: req.max_tokens.unwrap_or(256),
        seed: req.seed.unwrap_or_else(|| {
            use std::time::{SystemTime, UNIX_EPOCH};
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map(|d| d.as_nanos() as u64)
                .unwrap_or(42)
        }),
        repeat_penalty: 1.1,
    };

    let prompt = format_chat_prompt(&req.messages);
    let prompt_token_estimate = prompt.split_whitespace().count(); // rough estimate for mock

    let gen_req = GenerateRequest {
        prompt,
        config: gen_config,
    };

    let model_id = req.model.clone();

    if req.stream {
        stream_response(state, gen_req, model_id).await
    } else {
        non_stream_response(state, gen_req, model_id, prompt_token_estimate).await
    }
}

fn format_chat_prompt(messages: &[ChatMessage]) -> String {
    // Simple concatenation for the mock backend.
    // LocalInferenceBackend applies the proper chat template per architecture.
    messages
        .iter()
        .map(|m| format!("{}: {}", m.role, m.content))
        .collect::<Vec<_>>()
        .join("\n")
}

async fn non_stream_response(
    state: AppState,
    req: GenerateRequest,
    model_id: String,
    prompt_token_estimate: usize,
) -> Response {
    let stream_result = state.backend.generate(req).await;

    let mut token_stream = match stream_result {
        Ok(s) => s,
        Err(InferenceError::ModelNotReady(msg)) => {
            warn!("model not ready: {msg}");
            return service_unavailable(&msg);
        }
        Err(InferenceError::ContextTooLong { tokens, limit }) => {
            return (
                StatusCode::BAD_REQUEST,
                Json(ErrorResponse {
                    error: ErrorDetail {
                        message: format!("context too long: {tokens} tokens (limit: {limit})"),
                        r#type: "invalid_request_error".to_string(),
                        code: Some("context_length_exceeded".to_string()),
                    },
                }),
            )
                .into_response();
        }
        Err(e) => {
            error!("generation error: {e}");
            return internal_error(&e.to_string());
        }
    };

    let mut content = String::new();
    let mut completion_tokens = 0usize;

    while let Some(result) = token_stream.next().await {
        match result {
            Ok(token) => {
                if !token.is_eos {
                    content.push_str(&token.text);
                    completion_tokens += 1;
                }
            }
            Err(e) => {
                error!("stream error: {e}");
                return internal_error(&e.to_string());
            }
        }
    }

    let response = ChatCompletion {
        id: format!("chatcmpl-{}", Uuid::new_v4()),
        object: "chat.completion",
        created: openai::unix_now(),
        model: model_id,
        choices: vec![Choice {
            index: 0,
            message: CompletionMessage {
                role: "assistant".to_string(),
                content,
            },
            finish_reason: Some("stop".to_string()),
        }],
        usage: Usage {
            prompt_tokens: prompt_token_estimate,
            completion_tokens,
            total_tokens: prompt_token_estimate + completion_tokens,
        },
    };

    (StatusCode::OK, Json(response)).into_response()
}

async fn stream_response(state: AppState, req: GenerateRequest, model_id: String) -> Response {
    let completion_id = format!("chatcmpl-{}", Uuid::new_v4());
    let created = openai::unix_now();

    let token_stream = match state.backend.generate(req).await {
        Ok(s) => s,
        Err(InferenceError::ModelNotReady(msg)) => {
            warn!("model not ready: {msg}");
            return service_unavailable(&msg);
        }
        Err(e) => {
            error!("generation error: {e}");
            return internal_error(&e.to_string());
        }
    };

    let id = completion_id.clone();
    let model = model_id.clone();

    let chunk_stream = token_stream
        .map(move |result| {
            let id = id.clone();
            let model = model.clone();

            match result {
                Ok(token) if token.is_eos => {
                    // Final chunk with finish_reason
                    let chunk = ChatCompletionChunk {
                        id,
                        object: "chat.completion.chunk",
                        created,
                        model,
                        choices: vec![ChunkChoice {
                            index: 0,
                            delta: Delta {
                                role: None,
                                content: None,
                            },
                            finish_reason: Some("stop".to_string()),
                        }],
                    };
                    Ok::<Event, Infallible>(
                        Event::default().data(serde_json::to_string(&chunk).unwrap()),
                    )
                }
                Ok(token) => {
                    let chunk = ChatCompletionChunk {
                        id,
                        object: "chat.completion.chunk",
                        created,
                        model,
                        choices: vec![ChunkChoice {
                            index: 0,
                            delta: Delta {
                                role: None,
                                content: Some(token.text),
                            },
                            finish_reason: None,
                        }],
                    };
                    Ok(Event::default().data(serde_json::to_string(&chunk).unwrap()))
                }
                Err(e) => {
                    error!("stream token error: {e}");
                    // Best-effort: send error as last event before [DONE]
                    Ok(Event::default().data(format!("[ERROR] {e}")))
                }
            }
        })
        .chain(stream::once(async move {
            Ok::<Event, Infallible>(Event::default().data("[DONE]"))
        }));

    Sse::new(chunk_stream)
        .keep_alive(axum::response::sse::KeepAlive::default())
        .into_response()
}

fn service_unavailable(msg: &str) -> Response {
    (
        StatusCode::SERVICE_UNAVAILABLE,
        Json(ErrorResponse {
            error: ErrorDetail {
                message: msg.to_string(),
                r#type: "server_error".to_string(),
                code: Some("model_not_ready".to_string()),
            },
        }),
    )
        .into_response()
}

fn internal_error(msg: &str) -> Response {
    (
        StatusCode::INTERNAL_SERVER_ERROR,
        Json(ErrorResponse {
            error: ErrorDetail {
                message: msg.to_string(),
                r#type: "server_error".to_string(),
                code: None,
            },
        }),
    )
        .into_response()
}
