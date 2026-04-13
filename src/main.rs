mod api;
mod config;
mod inference;
mod model;
mod sse;

use clap::{Parser, Subcommand};
use tracing::info;

#[derive(Parser)]
#[command(name = "mycelium", about = "Distributed LLM inference server", version)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Start the inference HTTP server
    Serve {
        /// Override listen port
        #[arg(long)]
        port: Option<u16>,
        /// Override model ID (HuggingFace repo, e.g. meta-llama/Llama-3.2-3B-Instruct)
        #[arg(long)]
        model: Option<String>,
    },
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Tracing: human text by default, JSON if MYCELIUM_LOG_FORMAT=json
    let use_json = std::env::var("MYCELIUM_LOG_FORMAT")
        .map(|v| v.eq_ignore_ascii_case("json"))
        .unwrap_or(false);

    if use_json {
        tracing_subscriber::fmt()
            .json()
            .with_env_filter(
                tracing_subscriber::EnvFilter::try_from_default_env()
                    .unwrap_or_else(|_| "mycelium=info".into()),
            )
            .init();
    } else {
        tracing_subscriber::fmt()
            .with_env_filter(
                tracing_subscriber::EnvFilter::try_from_default_env()
                    .unwrap_or_else(|_| "mycelium=info".into()),
            )
            .init();
    }

    let cli = Cli::parse();

    match cli.command {
        Commands::Serve { port, model } => {
            serve(port, model).await?;
        }
    }

    Ok(())
}

async fn serve(port_override: Option<u16>, model_override: Option<String>) -> anyhow::Result<()> {
    // Load config: defaults → TOML → env overrides
    let mut cfg = config::ServerConfig::load()?;

    // CLI takes highest priority
    if let Some(p) = port_override {
        cfg.network.port = p;
    }
    if let Some(m) = model_override {
        cfg.model.id = m;
    }

    let addr = cfg.socket_addr();

    info!(
        port = cfg.network.port,
        model_id = %cfg.model.id,
        cache_dir = %cfg.model.cache_dir.display(),
        namespace = %cfg.node.namespace,
        "starting mycelium server"
    );

    info!("loading model (this may take a moment on first run)...");
    let app = api::router::build_router(cfg).await?;

    let listener = tokio::net::TcpListener::bind(addr).await?;
    info!("server ready — http://{addr}/v1/chat/completions");

    axum::serve(listener, app).await?;

    Ok(())
}
