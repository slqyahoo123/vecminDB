use thiserror::Error;

/// Result type for vecmindb operations
pub type Result<T> = std::result::Result<T, Error>;

impl Error {
    /// Create a storage error
    pub fn storage(msg: impl Into<String>) -> Self {
        Error::Storage(msg.into())
    }
    
    /// Create a serialization error
    pub fn serialization(msg: impl Into<String>) -> Self {
        Error::Serialization(msg.into())
    }
    
    /// Create an invalid input error
    pub fn invalid_input(msg: impl Into<String>) -> Self {
        Error::InvalidInput(msg.into())
    }
    
    /// Create a not found error
    pub fn not_found(msg: impl Into<String>) -> Self {
        Error::NotFound(msg.into())
    }
    
    /// Create an internal error
    pub fn internal(msg: impl Into<String>) -> Self {
        Error::Internal(msg.into())
    }
    
    /// Create a lock error
    pub fn lock(msg: impl Into<String>) -> Self {
        Error::Lock(msg.into())
    }

    /// Create a lock‑poison error (wrapper around `Lock`)
    pub fn locks_poison(msg: impl Into<String>) -> Self {
        Error::Lock(format!("Lock poisoned: {}", msg.into()))
    }
    
    /// Create a config error
    pub fn config(msg: impl Into<String>) -> Self {
        Error::Config(msg.into())
    }
    
    /// Create an index error
    pub fn index(msg: impl Into<String>) -> Self {
        Error::Index(msg.into())
    }
    
    /// Create a cache error
    pub fn cache(msg: impl Into<String>) -> Self {
        Error::Cache(msg.into())
    }

    /// Create a processing error for data/stream pipelines
    pub fn processing(msg: impl Into<String>) -> Self {
        Error::Processing(msg.into())
    }
    
    /// Create a system error
    pub fn system(msg: impl Into<String>) -> Self {
        Error::Internal(msg.into())
    }
    
    /// Create an IO error
    pub fn io_error(msg: impl Into<String>) -> Self {
        Error::Io(std::io::Error::new(
            std::io::ErrorKind::Other,
            format!("{}", msg.into()),
        ))
    }
    
    /// Create an invalid data error
    pub fn invalid_data(msg: impl Into<String>) -> Self {
        Error::InvalidInput(msg.into())
    }
    
    /// Create an invalid argument error
    pub fn invalid_argument(msg: impl Into<String>) -> Self {
        Error::InvalidInput(msg.into())
    }
    
    /// Create an invalid state error (for inconsistent or unsupported runtime states)
    pub fn invalid_state(msg: impl Into<String>) -> Self {
        Error::InvalidState(msg.into())
    }
    
    /// Create a feature not enabled error
    pub fn feature_not_enabled(feature: impl Into<String>) -> Self {
        Error::Config(format!("Feature '{}' is not enabled", feature.into()))
    }
    
    /// Create an unsupported file type error
    pub fn unsupported_file_type(file_type: impl Into<String>) -> Self {
        Error::InvalidInput(format!("Unsupported file type: {}", file_type.into()))
    }

    /// Create a database operation error
    pub fn database_operation(msg: impl Into<String>) -> Self {
        Error::Database(msg.into())
    }

    /// Create a network error
    pub fn network(msg: impl Into<String>) -> Self {
        Error::Network(msg.into())
    }

    /// Create a vector error
    pub fn vector(msg: impl Into<String>) -> Self {
        Error::Vector(msg.into())
    }

    /// Create a resource error
    pub fn resource(msg: impl Into<String>) -> Self {
        Error::Resource(msg.into())
    }

    /// Create an already exists error
    pub fn already_exists(msg: impl Into<String>) -> Self {
        Error::AlreadyExists(msg.into())
    }

    /// Create an other error
    pub fn other(msg: impl Into<String>) -> Self {
        Error::Other(msg.into())
    }

    /// Create a transaction error
    pub fn transaction(msg: impl Into<String>) -> Self {
        Error::Transaction(msg.into())
    }

    /// Create a validation error
    pub fn validation(msg: impl Into<String>) -> Self {
        Error::Validation(msg.into())
    }

    /// Create an execution error (for automation / workflow failures)
    pub fn execution_error(msg: impl Into<String>) -> Self {
        Error::ExecutionError(msg.into())
    }

    /// Create a not‑implemented error for optional subsystems
    pub fn not_implemented(msg: impl Into<String>) -> Self {
        Error::NotImplemented(msg.into())
    }

    /// Create a timeout error (for long‑running operations)
    pub fn timeout(msg: impl Into<String>) -> Self {
        Error::Timeout(msg.into())
    }
}

/// Error context trait for adding context to errors
pub trait WithErrorContext {
    fn with_context(self, context: impl Into<String>) -> Self;
}

impl<T> WithErrorContext for Result<T> {
    fn with_context(self, context: impl Into<String>) -> Self {
        // Attach human‑readable context information to error variants so
        // that logs and client responses carry richer diagnostic details.
        self.map_err(|e| {
            let context_str = context.into();
            match e {
                Error::Internal(msg) => Error::Internal(format!("{}: {}", context_str, msg)),
                Error::InvalidInput(msg) => Error::InvalidInput(format!("{}: {}", context_str, msg)),
                Error::NotFound(msg) => Error::NotFound(format!("{}: {}", context_str, msg)),
                Error::Config(msg) => Error::Config(format!("{}: {}", context_str, msg)),
                Error::Storage(msg) => Error::Storage(format!("{}: {}", context_str, msg)),
                Error::Database(msg) => Error::Database(format!("{}: {}", context_str, msg)),
                Error::Serialization(msg) => Error::Serialization(format!("{}: {}", context_str, msg)),
                Error::Index(msg) => Error::Index(format!("{}: {}", context_str, msg)),
                Error::Cache(msg) => Error::Cache(format!("{}: {}", context_str, msg)),
                Error::Lock(msg) => Error::Lock(format!("{}: {}", context_str, msg)),
                Error::Vector(msg) => Error::Vector(format!("{}: {}", context_str, msg)),
                Error::Resource(msg) => Error::Resource(format!("{}: {}", context_str, msg)),
                Error::AlreadyExists(msg) => Error::AlreadyExists(format!("{}: {}", context_str, msg)),
                Error::Validation(msg) => Error::Validation(format!("{}: {}", context_str, msg)),
                Error::Network(msg) => Error::Network(format!("{}: {}", context_str, msg)),
                Error::Processing(msg) => Error::Processing(format!("{}: {}", context_str, msg)),
                Error::Other(msg) => Error::Other(format!("{}: {}", context_str, msg)),
            }
        })
    }
}

/// Error context type alias
pub type ErrorContext = String;

/// Main error type for vecmindb
#[derive(Error, Debug)]
pub enum Error {
    /// Vector-related errors
    #[error("Vector error: {0}")]
    Vector(String),

    /// Index-related errors
    #[error("Index error: {0}")]
    Index(String),

    /// Storage-related errors
    #[error("Storage error: {0}")]
    Storage(String),

    /// Database operation errors
    #[error("Database error: {0}")]
    Database(String),

    /// Cache-related errors
    #[error("Cache error: {0}")]
    Cache(String),

    /// Resource management errors
    #[error("Resource error: {0}")]
    Resource(String),

    /// Configuration errors
    #[error("Configuration error: {0}")]
    Config(String),

    /// Serialization/Deserialization errors
    #[error("Serialization error: {0}")]
    Serialization(String),

    /// Generic processing errors (data pipelines, async loaders, etc.)
    #[error("Processing error: {0}")]
    Processing(String),

    /// I/O errors
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    /// Invalid input or parameters
    #[error("Invalid input: {0}")]
    InvalidInput(String),

    /// Not found errors
    #[error("Not found: {0}")]
    NotFound(String),

    /// Already exists errors
    #[error("Already exists: {0}")]
    AlreadyExists(String),

    /// Internal errors
    #[error("Internal error: {0}")]
    Internal(String),

    /// Lock errors
    #[error("Lock error: {0}")]
    Lock(String),

    /// Validation errors
    #[error("Validation error: {0}")]
    Validation(String),

    /// Transaction coordination / 2PC errors
    #[error("Transaction error: {0}")]
    Transaction(String),

    /// Invalid state errors (e.g. cycle dependencies, inconsistent container state)
    #[error("Invalid state: {0}")]
    InvalidState(String),

    /// Automation / execution errors
    #[error("Execution error: {0}")]
    ExecutionError(String),

    /// Not implemented errors for optional features
    #[error("Not implemented: {0}")]
    NotImplemented(String),

    /// Network / transport layer errors
    #[error("Network error: {0}")]
    Network(String),

    /// Timeout errors
    #[error("Timeout: {0}")]
    Timeout(String),

    /// Other errors
    #[error("{0}")]
    Other(String),
}

// Implement From for common error types
impl From<serde_json::Error> for Error {
    fn from(err: serde_json::Error) -> Self {
        Error::Serialization(err.to_string())
    }
}

impl From<bincode::Error> for Error {
    fn from(err: bincode::Error) -> Self {
        Error::Serialization(err.to_string())
    }
}

impl From<sled::Error> for Error {
    fn from(err: sled::Error) -> Self {
        Error::Storage(err.to_string())
    }
}

impl From<rocksdb::Error> for Error {
    fn from(err: rocksdb::Error) -> Self {
        Error::Storage(err.to_string())
    }
}

impl From<Box<dyn std::error::Error>> for Error {
    fn from(err: Box<dyn std::error::Error>) -> Self {
        Error::Other(err.to_string())
    }
}

impl<T> From<std::sync::PoisonError<T>> for Error {
    fn from(err: std::sync::PoisonError<T>) -> Self {
        Error::Lock(err.to_string())
    }
}

impl From<std::string::FromUtf8Error> for Error {
    fn from(err: std::string::FromUtf8Error) -> Self {
        Error::InvalidInput(format!("UTF-8转换错误: {}", err))
    }
}

impl From<tokio::task::JoinError> for Error {
    fn from(err: tokio::task::JoinError) -> Self {
        Error::ExecutionError(format!("任务执行失败: {}", err))
    }
}

impl From<anyhow::Error> for Error {
    fn from(err: anyhow::Error) -> Self {
        Error::ExecutionError(format!("执行错误: {}", err))
    }
}

impl From<regex::Error> for Error {
    fn from(err: regex::Error) -> Self {
        Error::InvalidInput(format!("正则表达式错误: {}", err))
    }
}

impl From<csv::Error> for Error {
    fn from(err: csv::Error) -> Self {
        Error::InvalidInput(format!("CSV处理错误: {}", err))
    }
}

impl From<String> for Error {
    fn from(msg: String) -> Self {
        Error::InvalidInput(msg)
    }
}

