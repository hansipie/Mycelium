# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

# Mycelium Development Guidelines

Auto-generated from feature plans + manual additions. Last updated: 2026-04-13

## Active Technologies

- **Rust stable ≥ 1.75** (no nightly) — (001-inference-locale)
- **candle-transformers** (HuggingFace git) — inférence Llama/Mistral sur CPU
- **axum 0.7** — HTTP server + SSE streaming
- **tokio** — async runtime
- **hf-hub 0.3** — téléchargement et cache modèles HuggingFace Hub
- **tokenizers 0.22** — tokenizer BPE (Llama/Mistral)

## Project Structure

```text
src/
├── main.rs              # Entrypoint CLI (clap) — commande `serve`
├── config/              # ServerConfig, TOML loading, env overrides
├── model/               # HF Hub download, candle engine, tokenizer
├── inference/           # trait InferenceBackend + LocalInferenceBackend
├── api/                 # Axum router, handlers chat/models/health
└── sse/                 # Types SSE OpenAI (ChatCompletionChunk)

tests/
├── api/
├── inference/
├── model/
└── integration/
```

## Commands

```bash
cargo build --release                        # Build optimisé
cargo build --release --features cuda        # Build avec support CUDA
cargo build --release --features metal       # Build avec support Metal (Apple)
cargo test                                   # Tous les tests unitaires + intégration (mock)
cargo test config::settings                  # Un module spécifique
cargo test --test integration                # Tests d'intégration (mock, pas de modèle requis)
cargo test --test integration -- --include-ignored  # + tests avec vrai modèle (cache requis)
cargo clippy                                 # Linting
cargo run -- serve                           # Lancer le serveur (port 9090 par défaut)
cargo run -- serve --port 8080 --model mistralai/Mistral-7B-Instruct-v0.2
```

## Architecture

La frontière architecturale centrale est `inference/backend.rs` :
- `trait InferenceBackend` — interface commune local/distribué
- `LocalInferenceBackend` — implémentation Phase 1 (candle, CPU) ; utilise `Arc<Mutex<Engine>>` car le KV cache est muté pendant la génération (RwLock inadapté)
- La couche HTTP (`api/`) ne dépend jamais directement de `candle-transformers`

Le router expose deux constructeurs :
- `build_router(cfg)` — production, télécharge/charge le vrai modèle
- `build_router_with(cfg, backend)` — injecte un backend arbitraire (utilisé dans tous les tests)

## Configuration

Ordre de priorité (du plus faible au plus fort) : valeurs par défaut → `~/.config/mycelium/config.toml` → variables d'environnement → flags CLI.

Variables d'environnement disponibles : `MYCELIUM_PORT`, `MYCELIUM_LISTEN`, `MYCELIUM_MODEL_ID`, `MYCELIUM_CACHE_DIR`, `MYCELIUM_NODE_NAME`, `MYCELIUM_NAMESPACE`, `MYCELIUM_LOG_FORMAT` (`json` ou texte).

## Workflow TDD (NON-NÉGOCIABLE)

Les tests doivent être écrits **avant** l'implémentation (Red → Green → Refactor). Aucun code d'implémentation ne peut être soumis sans tests préalables. Cette règle est issue de la constitution et prévaut sur toute contrainte de délai.

## Code Style

- Rust idiomatique stable — pas de `unsafe` sauf `VarBuilder::from_mmaped_safetensors` dans `model/engine.rs`
- Erreurs : `anyhow` pour les fonctions internes, `thiserror` pour les types publics (`InferenceError`)
- Logging : `tracing` partout (pas de `println!` en dehors des tests)

## Constitution

Voir `.specify/memory/constitution.md` — 4 principes non-négociables :
I. Rust-First | II. Distributed-by-Design | III. OpenAI-Compatible API | IV. Test-First

## Recent Changes

- 001-inference-locale: Implémentation complète Phase 1 — inférence locale (T001-T038 tous terminés)
  - `model/` : téléchargement HF Hub cache-first, chargement safetensors Llama, boucle génération KV cache
  - `inference/local.rs` : `LocalInferenceBackend` opérationnel
  - `config/settings.rs` : priorité CLI > env > TOML > defaults
  - `tests/integration/` : tests e2e, conformité OpenAI, concurrence

<!-- MANUAL ADDITIONS START -->
<!-- MANUAL ADDITIONS END -->
