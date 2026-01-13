use std::collections::HashMap;
use uuid::Uuid;
use crate::data::DataBatch;

/// 批次数据摘要
#[derive(Debug, Clone)]
pub struct BatchSummary {
    /// 批次ID
    pub batch_id: String,
    /// 记录数量
    pub record_count: usize,
    /// 字段数量
    pub field_count: usize,
    /// 字段名称列表
    pub field_names: Vec<String>,
    /// 批次创建时间
    pub created_at: String,
}

impl From<&DataBatch> for BatchSummary {
    fn from(batch: &DataBatch) -> Self {
        let field_names = if let Some(schema) = &batch.schema {
            schema.fields().iter().map(|f| f.name.clone()).collect()
        } else {
            Vec::new()
        };
        
        Self {
            batch_id: batch.id.clone().unwrap_or_else(|| Uuid::new_v4().to_string()),
            record_count: batch.records.len(),
            field_count: field_names.len(),
            field_names,
            created_at: batch.created_at.to_string(),
        }
    }
}

/// 导入结果结构体
#[derive(Debug, Clone)]
pub struct ImportResult {
    /// 导入ID
    pub id: String,
    /// 开始时间
    pub start_time: chrono::DateTime<chrono::Utc>,
    /// 结束时间
    pub end_time: Option<chrono::DateTime<chrono::Utc>>,
    /// 持续时间（毫秒）
    pub duration: Option<u64>,
    /// 是否成功
    pub success: bool,
    /// 结果消息
    pub message: String,
    /// 处理的记录数
    pub records_processed: usize,
    /// 失败的记录数
    pub records_failed: usize,
    /// 错误信息
    pub errors: Vec<String>,
    /// 警告信息
    pub warnings: Vec<String>,
    /// 元数据
    pub metadata: HashMap<String, String>,
    /// 处理行数，与记录可能不同
    pub processed_rows: usize,
    /// 导入的批次数据摘要
    pub batch_summary: Option<BatchSummary>,
}

impl ImportResult {
    /// 创建新的成功结果
    pub fn success(message: &str, records_processed: usize) -> Self {
        let mut result = Self::default();
        result.success = true;
        result.message = message.to_string();
        result.records_processed = records_processed;
        result
    }
    
    /// 创建新的失败结果
    pub fn failure(message: &str, error: &str) -> Self {
        let mut result = Self::default();
        result.success = false;
        result.message = message.to_string();
        result.errors.push(error.to_string());
        result
    }
    
    /// 添加警告信息
    pub fn add_warning(&mut self, warning: &str) -> &mut Self {
        self.warnings.push(warning.to_string());
        self
    }
    
    /// 添加错误信息
    pub fn add_error(&mut self, error: &str) -> &mut Self {
        self.errors.push(error.to_string());
        self
    }
    
    /// 添加元数据
    pub fn add_metadata(&mut self, key: &str, value: &str) -> &mut Self {
        self.metadata.insert(key.to_string(), value.to_string());
        self
    }
    
    /// 记录处理完成的记录
    pub fn record_processed(&mut self, processed: usize, failed: usize) -> &mut Self {
        self.records_processed = processed;
        self.records_failed = failed;
        self
    }
    
    /// 添加批次摘要
    pub fn with_batch(&mut self, batch: &DataBatch) -> &mut Self {
        self.batch_summary = Some(BatchSummary::from(batch));
        self
    }
    
    /// 检查是否有警告
    pub fn has_warnings(&self) -> bool {
        !self.warnings.is_empty()
    }
    
    /// 检查是否有错误
    pub fn has_errors(&self) -> bool {
        !self.errors.is_empty()
    }

    /// 合并另一个导入结果到当前结果
    pub fn merge(&mut self, other: &ImportResult) -> &mut Self {
        if !other.success {
            self.success = false;
        }
        
        self.records_processed += other.records_processed;
        self.records_failed += other.records_failed;
        self.processed_rows += other.processed_rows;
        
        for error in &other.errors {
            self.errors.push(error.clone());
        }
        
        for warning in &other.warnings {
            self.warnings.push(warning.clone());
        }
        
        for (key, value) in &other.metadata {
            if !self.metadata.contains_key(key) {
                self.metadata.insert(key.clone(), value.clone());
            }
        }
        
        // 保留最后一个批次摘要
        if other.batch_summary.is_some() {
            self.batch_summary = other.batch_summary.clone();
        }
        
        self
    }

    /// 获取总记录数
    pub fn total_records(&self) -> usize {
        self.records_processed + self.records_failed
    }

    /// 获取成功率
    pub fn success_rate(&self) -> f64 {
        if self.total_records() == 0 {
            return 0.0;
        }
        
        self.records_processed as f64 / self.total_records() as f64
    }

    /// 获取处理速率（每秒处理记录数）
    pub fn processing_rate(&self) -> Option<f64> {
        if let Some(duration_str) = self.metadata.get("processing_time_ms") {
            if let Ok(duration_ms) = duration_str.parse::<u64>() {
                if duration_ms > 0 {
                    let duration_secs = duration_ms as f64 / 1000.0;
                    return Some(self.total_records() as f64 / duration_secs);
                }
            }
        }
        
        None
    }
}

impl Default for ImportResult {
    fn default() -> Self {
        Self {
            id: String::new(),
            start_time: chrono::Utc::now(),
            end_time: None,
            duration: None,
            success: false,
            message: String::new(),
            records_processed: 0,
            records_failed: 0,
            errors: Vec::new(),
            warnings: Vec::new(),
            metadata: HashMap::new(),
            processed_rows: 0,
            batch_summary: None,
        }
    }
}

/// 批量导入结果
#[derive(Debug, Clone)]
pub struct BatchImportResult {
    /// 总体结果
    pub summary: ImportResult,
    /// 各文件的单独结果
    pub file_results: HashMap<String, ImportResult>,
}

impl BatchImportResult {
    /// 创建新的批量导入结果
    pub fn new() -> Self {
        Self {
            summary: ImportResult::default(),
            file_results: HashMap::new(),
        }
    }
    
    /// 添加单个文件的导入结果
    pub fn add_file_result(&mut self, file_path: &str, result: ImportResult) -> &mut Self {
        // 更新总体结果
        self.summary.merge(&result);
        
        // 保存单个文件的结果
        self.file_results.insert(file_path.to_string(), result);
        
        self
    }
    
    /// 获取成功导入的文件数量
    pub fn successful_files(&self) -> usize {
        self.file_results.values()
            .filter(|r| r.success)
            .count()
    }
    
    /// 获取失败导入的文件数量
    pub fn failed_files(&self) -> usize {
        self.file_results.values()
            .filter(|r| !r.success)
            .count()
    }
    
    /// 获取总文件数量
    pub fn total_files(&self) -> usize {
        self.file_results.len()
    }
    
    /// 检查是否所有文件都导入成功
    pub fn all_successful(&self) -> bool {
        self.summary.success && self.failed_files() == 0
    }
} 