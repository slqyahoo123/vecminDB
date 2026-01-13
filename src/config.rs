use serde::{Deserialize, Serialize};
use std::path::PathBuf;

/// Main configuration for VecminDB
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Config {
    /// Storage configuration
    pub storage: StorageConfig,
    
    /// Cache configuration
    pub cache: CacheConfig,
    
    /// Resource management configuration
    pub resource: ResourceConfig,
    
    /// HTTP server configuration (if enabled)
    #[cfg(feature = "http-server")]
    pub server: ServerConfig,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            storage: StorageConfig::default(),
            cache: CacheConfig::default(),
            resource: ResourceConfig::default(),
            #[cfg(feature = "http-server")]
            server: ServerConfig::default(),
        }
    }
}

/// Storage configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StorageConfig {
    /// Path to storage directory
    pub path: PathBuf,
    
    /// Enable compression
    pub compression: bool,
    
    /// Cache size in bytes
    pub cache_size: usize,
    
    /// Enable write-ahead log
    pub wal_enabled: bool,
}

impl Default for StorageConfig {
    fn default() -> Self {
        Self {
            path: PathBuf::from("./data"),
            compression: true,
            cache_size: 100 * 1024 * 1024, // 100MB
            wal_enabled: true,
        }
    }
}

/// Cache configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheConfig {
    /// Enable caching
    pub enabled: bool,
    
    /// Memory cache size in bytes
    pub memory_size: usize,
    
    /// Disk cache size in bytes
    pub disk_size: usize,
    
    /// Enable distributed cache (Redis)
    #[cfg(feature = "distributed")]
    pub redis_url: Option<String>,
}

impl Default for CacheConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            memory_size: 50 * 1024 * 1024, // 50MB
            disk_size: 500 * 1024 * 1024,  // 500MB
            #[cfg(feature = "distributed")]
            redis_url: None,
        }
    }
}

/// Resource management configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceConfig {
    /// Maximum memory usage in bytes
    pub max_memory: usize,
    
    /// Number of worker threads
    pub num_threads: usize,
    
    /// Enable GPU acceleration
    #[cfg(feature = "gpu")]
    pub gpu_enabled: bool,
}

impl Default for ResourceConfig {
    fn default() -> Self {
        Self {
            max_memory: 1024 * 1024 * 1024, // 1GB
            num_threads: num_cpus::get(),
            #[cfg(feature = "gpu")]
            gpu_enabled: false,
        }
    }
}

/// HTTP server configuration
#[cfg(feature = "http-server")]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServerConfig {
    /// Server host
    pub host: String,
    
    /// Server port
    pub port: u16,
    
    /// Number of worker threads
    pub workers: usize,
    
    /// Enable CORS
    pub cors_enabled: bool,
    
    /// Request timeout in seconds
    pub timeout_secs: u64,
}

#[cfg(feature = "http-server")]
impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            host: "127.0.0.1".to_string(),
            port: 8080,
            workers: num_cpus::get(),
            cors_enabled: true,
            timeout_secs: 30,
        }
    }
}

impl Config {
    /// Load configuration from file
    pub fn from_file(path: impl AsRef<std::path::Path>) -> crate::Result<Self> {
        let content = std::fs::read_to_string(path)?;
        let config: Config = serde_json::from_str(&content)?;
        Ok(config)
    }
    
    /// Save configuration to file
    pub fn to_file(&self, path: impl AsRef<std::path::Path>) -> crate::Result<()> {
        let content = serde_json::to_string_pretty(self)?;
        std::fs::write(path, content)?;
        Ok(())
    }
}

