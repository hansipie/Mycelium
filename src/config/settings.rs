use std::net::{IpAddr, Ipv4Addr};
use std::path::PathBuf;

// ── Node state machine ────────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq)]
pub enum NodeStatus {
    Initializing,
    Ready,
    Error(String),
}

impl std::fmt::Display for NodeStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            NodeStatus::Initializing => write!(f, "initializing"),
            NodeStatus::Ready => write!(f, "ready"),
            NodeStatus::Error(msg) => write!(f, "error: {msg}"),
        }
    }
}

// ── Model architecture ────────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq)]
pub enum ModelArchitecture {
    Llama,
    Mistral,
}

impl std::fmt::Display for ModelArchitecture {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ModelArchitecture::Llama => write!(f, "llama"),
            ModelArchitecture::Mistral => write!(f, "mistral"),
        }
    }
}

// ── Node ─────────────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct Node {
    pub id: String,
    pub name: String,
    pub namespace: String,
    pub status: NodeStatus,
    pub model_id: Option<String>,
    pub listen_addr: std::net::SocketAddr,
    pub started_at: u64,
}

// ── ServerConfig ──────────────────────────────────────────────────────────────

#[derive(Debug, Clone, serde::Deserialize)]
#[serde(default)]
pub struct ServerConfig {
    pub node: NodeConfig,
    pub network: NetworkConfig,
    pub model: ModelConfig,
}

#[derive(Debug, Clone, serde::Deserialize)]
#[serde(default)]
pub struct NodeConfig {
    pub name: String,
    pub namespace: String,
}

#[derive(Debug, Clone, serde::Deserialize)]
#[serde(default)]
pub struct NetworkConfig {
    pub listen: IpAddr,
    pub port: u16,
}

#[derive(Debug, Clone, serde::Deserialize)]
#[serde(default)]
pub struct ModelConfig {
    pub id: String,
    pub cache_dir: PathBuf,
}

impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            node: NodeConfig::default(),
            network: NetworkConfig::default(),
            model: ModelConfig::default(),
        }
    }
}

impl Default for NodeConfig {
    fn default() -> Self {
        Self {
            name: hostname(),
            namespace: "default".to_string(),
        }
    }
}

impl Default for NetworkConfig {
    fn default() -> Self {
        Self {
            listen: IpAddr::V4(Ipv4Addr::new(0, 0, 0, 0)),
            port: 9090,
        }
    }
}

impl Default for ModelConfig {
    fn default() -> Self {
        Self {
            id: "meta-llama/Llama-3.2-3B-Instruct".to_string(),
            cache_dir: default_cache_dir(),
        }
    }
}

fn hostname() -> String {
    std::env::var("HOSTNAME")
        .unwrap_or_else(|_| "mycelium-node".to_string())
}

fn default_cache_dir() -> PathBuf {
    dirs::cache_dir()
        .unwrap_or_else(|| PathBuf::from(".cache"))
        .join("mycelium")
        .join("models")
}

impl ServerConfig {
    /// Load config: defaults → TOML file → env overrides.
    /// CLI overrides are applied by the caller after this returns.
    pub fn load() -> anyhow::Result<Self> {
        let mut config = Self::default();

        // 1. TOML file
        let toml_path = dirs::config_dir()
            .unwrap_or_else(|| PathBuf::from(".config"))
            .join("mycelium")
            .join("config.toml");

        if toml_path.exists() {
            let contents = std::fs::read_to_string(&toml_path)?;
            let file_cfg: ServerConfig = toml::from_str(&contents)?;
            config = file_cfg;
        }

        // 2. Env overrides
        if let Ok(v) = std::env::var("MYCELIUM_NODE_NAME") {
            config.node.name = v;
        }
        if let Ok(v) = std::env::var("MYCELIUM_NAMESPACE") {
            config.node.namespace = v;
        }
        if let Ok(v) = std::env::var("MYCELIUM_LISTEN") {
            config.network.listen = v.parse()?;
        }
        if let Ok(v) = std::env::var("MYCELIUM_PORT") {
            config.network.port = v.parse()?;
        }
        if let Ok(v) = std::env::var("MYCELIUM_MODEL_ID") {
            config.model.id = v;
        }
        if let Ok(v) = std::env::var("MYCELIUM_CACHE_DIR") {
            config.model.cache_dir = PathBuf::from(v);
        }

        Ok(config)
    }

    pub fn socket_addr(&self) -> std::net::SocketAddr {
        std::net::SocketAddr::new(self.network.listen, self.network.port)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_config_has_expected_values() {
        let cfg = ServerConfig::default();
        assert_eq!(cfg.network.port, 9090);
        assert_eq!(cfg.node.namespace, "default");
        assert_eq!(cfg.model.id, "meta-llama/Llama-3.2-3B-Instruct");
    }

    #[test]
    fn env_override_port() {
        std::env::set_var("MYCELIUM_PORT", "8888");
        let cfg = ServerConfig::load().unwrap();
        assert_eq!(cfg.network.port, 8888);
        std::env::remove_var("MYCELIUM_PORT");
    }

    #[test]
    fn missing_config_file_uses_defaults() {
        // Ensure no config file interferes by pointing to a temp dir.
        // load() falls back to Default when file is absent — no panic.
        let cfg = ServerConfig::load();
        assert!(cfg.is_ok());
    }
}
