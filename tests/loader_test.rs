use std::path::PathBuf;
use mycelium::model::loader::{resolve_model_files, ModelFiles};

// These tests verify cache-first behavior without hitting the network.
// They rely on a pre-populated cache directory (integration tests require model).

#[test]
fn resolve_model_files_returns_error_on_missing_cache() {
    let missing = PathBuf::from("/nonexistent/path/to/cache");
    let result = resolve_model_files(&missing, "meta-llama/Llama-3.2-3B-Instruct");
    assert!(result.is_err(), "should fail when cache dir does not exist");
}

#[test]
fn resolve_model_files_returns_error_on_empty_dir() {
    let dir = tempfile::tempdir().unwrap();
    let result = resolve_model_files(dir.path(), "meta-llama/Llama-3.2-3B-Instruct");
    assert!(result.is_err(), "should fail when no model files present");
}

#[test]
fn model_files_struct_has_required_fields() {
    // Compile-time check: ModelFiles must expose config_path, tokenizer_path, weight_paths
    let _ = |f: ModelFiles| {
        let _: &PathBuf = &f.config_path;
        let _: &PathBuf = &f.tokenizer_path;
        let _: &Vec<PathBuf> = &f.weight_paths;
    };
}
