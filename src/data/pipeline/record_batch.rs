// src/data/pipeline/record_batch.rs
//
// 记录批次定义
// 用于在数据管道中处理批量记录

use std::collections::HashMap;
use crate::data::record::Record;
use crate::data::schema::Schema;

/// 记录批次
/// 用于在数据管道中批量处理记录
#[derive(Debug, Clone)]
pub struct RecordBatch {
    /// 批次ID
    pub id: String,
    /// 批次中的记录
    pub records: Vec<Record>,
    /// 批次的模式
    pub schema: Option<Schema>,
    /// 批次元数据
    pub metadata: HashMap<String, String>,
    /// 创建时间
    pub created_at: chrono::DateTime<chrono::Utc>,
}

impl RecordBatch {
    /// 创建新的记录批次
    pub fn new() -> Self {
        Self {
            id: uuid::Uuid::new_v4().to_string(),
            records: Vec::new(),
            schema: None,
            metadata: HashMap::new(),
            created_at: chrono::Utc::now(),
        }
    }
    
    /// 创建指定容量的记录批次
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            id: uuid::Uuid::new_v4().to_string(),
            records: Vec::with_capacity(capacity),
            schema: None,
            metadata: HashMap::new(),
            created_at: chrono::Utc::now(),
        }
    }
    
    /// 设置模式
    pub fn with_schema(mut self, schema: Schema) -> Self {
        self.schema = Some(schema);
        self
    }
    
    /// 添加记录
    pub fn add_record(&mut self, record: Record) {
        self.records.push(record);
    }
    
    /// 批量添加记录
    pub fn add_records(&mut self, records: Vec<Record>) {
        self.records.extend(records);
    }
    
    /// 获取记录数量
    pub fn record_count(&self) -> usize {
        self.records.len()
    }
    
    /// 是否为空
    pub fn is_empty(&self) -> bool {
        self.records.is_empty()
    }
    
    /// 添加元数据
    pub fn add_metadata(&mut self, key: &str, value: &str) -> &mut Self {
        self.metadata.insert(key.to_string(), value.to_string());
        self
    }
    
    /// 获取元数据
    pub fn get_metadata(&self, key: &str) -> Option<&String> {
        self.metadata.get(key)
    }
    
    /// 将批次分割成多个小批次
    pub fn split(&self, batch_size: usize) -> Vec<RecordBatch> {
        if batch_size == 0 || self.is_empty() {
            return vec![self.clone()];
        }
        
        let mut result = Vec::new();
        let chunks = self.records.chunks(batch_size);
        
        for chunk in chunks {
            let mut batch = RecordBatch::with_capacity(chunk.len());
            if let Some(schema) = &self.schema {
                batch.schema = Some(schema.clone());
            }
            
            // 复制元数据
            for (key, value) in &self.metadata {
                batch.metadata.insert(key.clone(), value.clone());
            }
            
            // 添加记录
            batch.records.extend_from_slice(chunk);
            
            result.push(batch);
        }
        
        result
    }
}

impl Default for RecordBatch {
    fn default() -> Self {
        Self::new()
    }
}

/// 从Vec<Record>创建RecordBatch
impl From<Vec<Record>> for RecordBatch {
    fn from(records: Vec<Record>) -> Self {
        Self {
            id: uuid::Uuid::new_v4().to_string(),
            records,
            schema: None,
            created_at: chrono::Utc::now(),
            metadata: HashMap::new(),
        }
    }
}

/// 将RecordBatch转换为Vec<Record>
impl From<RecordBatch> for Vec<Record> {
    fn from(batch: RecordBatch) -> Self {
        batch.records
    }
} 