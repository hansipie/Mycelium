use axum::{extract::State, Json};

use crate::{
    api::router::AppState,
    sse::openai::{unix_now, ModelInfo, ModelList},
};

pub async fn list_models(State(state): State<AppState>) -> Json<ModelList> {
    let models = match state.backend.model_id() {
        Some(id) => vec![ModelInfo {
            id,
            object: "model",
            created: unix_now(),
            owned_by: "mycelium".to_string(),
        }],
        None => vec![],
    };

    Json(ModelList {
        object: "list",
        data: models,
    })
}
