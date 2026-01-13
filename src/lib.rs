//! VecminDB - High-performance vector database
//!
//! VecminDB is a high-performance vector database with multiple index algorithms,
//! optimizers, and auto-tuning capabilities.
//!
//! # Features
//!
//! - Multiple index algorithms (HNSW, IVF, PQ, LSH, etc.)
//! - Multi-objective optimization
//! - Auto-tuning capabilities
//! - Parallel processing
//! - Flexible storage with transaction support
//! - Multi-tier caching
//! - Resource management
//! - Built-in monitoring
//!
//! # Quick Start
//!
//! ```rust,no_run
//! use vecmindb::{VectorDB, Result};
//!
//! #[tokio::main]
//! async fn main() -> Result<()> {
//!     let db = VectorDB::new("./data")?;
//!     Ok(())
//! }
//! ```

// Core modules
pub mod error;
pub mod config;
pub mod status;

// Compatibility module for removed training/model code
pub mod compat;

// Create module aliases for backward compatibility
pub use compat as model;
pub use compat as training;

// Re-export task_scheduler from compat
pub use compat::task_scheduler;

// Core infrastructure
pub mod core;
pub mod interfaces;
pub mod event;
pub mod types;
pub mod utils;
pub mod hashing;
pub mod data;
pub mod network;

// Vector database modules
pub mod vector;
pub mod algorithm;
pub mod storage;
pub mod cache;
pub mod cache_common;
pub mod resource;
pub mod memory;
pub mod monitoring;

// API module (optional)
#[cfg(feature = "http-server")]
pub mod api;

// Re-export commonly used types
pub use error::{Error, Result};
pub use config::Config;
pub use vector::{
    VectorDB, 
    types::{Vector, VectorId, VectorEntry},
    index::{IndexType, IndexConfig, SearchResult, VectorIndex},
    search::{VectorSearchParams, VectorSearchResult, VectorCollection, VectorCollectionConfig, VectorQuery},
    core::operations::SimilarityMetric,
};

// Component and status types
pub use status::{StatusTracker, StatusTrackerTrait};

/// Component type enum
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ComponentType {
    Storage,
    Cache,
    Index,
    Monitor,
    Resource,
}

/// Component status enum  
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ComponentStatus {
    Starting,
    Running,
    Stopping,
    Shutdown,
    Error,
}

/// Component trait
pub trait Component: Send + Sync {
    fn name(&self) -> &str;
    fn component_type(&self) -> ComponentType;
    fn status(&self) -> ComponentStatus;
    fn start(&mut self) -> Result<()>;
    fn stop(&mut self) -> Result<()>;
}

/// Version information
pub const VERSION: &str = env!("CARGO_PKG_VERSION");
pub const NAME: &str = env!("CARGO_PKG_NAME");

/// Initialize the library
pub fn init() -> Result<()> {
    env_logger::try_init()
        .map_err(|e| Error::config(format!("Failed to initialize logger: {}", e)))?;
    log::info!("VecminDB {} initialized", VERSION);
    Ok(())
}
