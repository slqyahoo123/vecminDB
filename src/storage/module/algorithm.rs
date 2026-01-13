// 算法存储模块
// 提供算法代码和元数据的存储和检索功能

use crate::Result;

// Algorithm related constants
pub const ALGO_CODE_PREFIX: &str = "algo:code:";
pub const CF_ALGORITHMS: &str = "algorithms";

/// Algorithm operations trait
pub trait AlgorithmOperations {
    /// Store algorithm code
    fn store_algorithm(&self, algorithm_id: &str, code: &str) -> Result<()>;
    
    /// Retrieve algorithm code
    fn get_algorithm(&self, algorithm_id: &str) -> Result<Option<String>>;
    
    /// Store algorithm metadata
    fn store_algorithm_metadata(&self, algorithm_id: &str, metadata: &serde_json::Value) -> Result<()>;
    
    /// Retrieve algorithm metadata
    fn get_algorithm_metadata(&self, algorithm_id: &str) -> Result<Option<serde_json::Value>>;
    
    /// Get algorithm for a specific version
    fn get_algorithm_version(&self, algorithm_id: &str, version_id: &str) -> Result<Option<String>>;
    
    /// Delete algorithm
    fn delete_algorithm(&self, algorithm_id: &str) -> Result<()>;
}

// Implementation is left for the Storage struct in core.rs 