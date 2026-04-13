use axum::{extract::State, Json};
use serde::Serialize;

use crate::api::router::AppState;

#[derive(Serialize)]
pub struct HealthResponse {
    pub status: String,
    pub model: Option<String>,
}

pub async fn health_check(State(state): State<AppState>) -> Json<HealthResponse> {
    let model = state.backend.model_id();
    let status = if model.is_some() {
        "ready".to_string()
    } else {
        "initializing".to_string()
    };

    Json(HealthResponse { status, model })
}
