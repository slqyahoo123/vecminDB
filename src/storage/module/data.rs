// 数据存储模块
// 提供数据存储和检索的相关功能

use crate::data::DataFormat;
use crate::Result;
use chrono::{DateTime, Utc};
use serde::{Serialize, Deserialize};
use serde_json::Value;
use std::path::Path;

// Data related constants
pub const DATA_RAW_PREFIX: &str = "data:raw:";
pub const DATA_PROCESSED_PREFIX: &str = "data:processed:";
pub const CF_RAW_DATA: &str = "raw_data";
pub const CF_PROCESSED_DATA: &str = "processed_data";

/// Data information structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataInfo {
    pub id: String,
    pub name: String,
    pub format: DataFormat,
    pub size: u64,
    pub created_at: DateTime<Utc>,
}

/// Data operations trait
pub trait DataOperations {
    /// Store raw data into storage
    fn store_raw_data(&self, data_id: &str, data: Vec<String>) -> Result<()>;
    
    /// Retrieve raw data from storage
    fn get_raw_data(&self, data_id: &str) -> Result<Option<Vec<String>>>;
    
    /// Store processed data into storage
    fn store_processed_data(&self, data_id: &str, data: Vec<String>) -> Result<()>;
    
    /// Retrieve processed data from storage
    fn get_processed_data(&self, data_id: &str) -> Result<Option<Vec<String>>>;
    
    /// Import data from a file
    fn import_data(&self, name: &str, path: &Path, format: DataFormat) -> Result<String>;
    
    /// Convert data format
    fn convert_data_format(&self, id: &str, from: DataFormat, to: DataFormat) -> Result<()>;
    
    /// Get data as JSON
    fn get_data_as_json(&self, id: &str) -> Result<Value>;
    
    /// Get data as CSV
    fn get_data_as_csv(&self, id: &str) -> Result<String>;
    
    /// List all available data
    fn list_data(&self) -> Result<Vec<(String, String)>>;
    
    /// Get information about specific data
    fn get_data_info(&self, id: &str) -> Result<DataInfo>;
}

// Implementation is left for the Storage struct in core.rs 