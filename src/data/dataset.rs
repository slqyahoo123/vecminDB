use std::collections::HashMap;
use std::sync::Arc;
use serde::{Serialize, Deserialize};
use chrono::{DateTime, Utc};
use crate::error::Result;
use crate::data::types::{DataFormat, DatasetMetadata, DataStatus, ProcessingStep};
use crate::data::loader::DataLoader;
use crate::data::schema::DataSchema;
use crate::core::TensorData;

/// 数据集结构
#[derive(Clone, Serialize, Deserialize)]
pub struct Dataset {
    pub id: String,
    pub name: String,
    pub description: Option<String>,
    pub format: DataFormat,
    pub size: usize,
    pub created_at: chrono::DateTime<chrono::Utc>,
    pub updated_at: chrono::DateTime<chrono::Utc>,
    pub metadata: DatasetMetadata,
    pub path: String,
    pub processed: bool,
    #[serde(skip_serializing, skip_deserializing, default = "default_loader")]
    pub loader: Arc<dyn DataLoader>,
    pub batch_size: usize,
    pub schema: Option<DataSchema>,
    /// 数据批次列表
    pub batches: Vec<String>,
}

/// 提供默认的data loader
fn default_loader() -> Arc<dyn DataLoader> {
    Arc::new(crate::data::loader::memory::MemoryDataLoader::new())
}

impl std::fmt::Debug for Dataset {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Dataset")
            .field("id", &self.id)
            .field("name", &self.name)
            .field("description", &self.description)
            .field("format", &self.format)
            .field("size", &self.size)
            .field("created_at", &self.created_at)
            .field("updated_at", &self.updated_at)
            .field("metadata", &self.metadata)
            .field("path", &self.path)
            .field("processed", &self.processed)
            .field("loader", &format!("<DataLoader: {}>", self.loader.name()))
            .field("batch_size", &self.batch_size)
            .field("schema", &self.schema)
            .field("batches", &self.batches)
            .finish()
    }
}

impl Dataset {
    /// 创建新的数据集
    pub fn new(features: Vec<Vec<f64>>, labels: Option<Vec<Vec<f64>>>) -> Result<Self> {
        let id = uuid::Uuid::new_v4().to_string();
        let now = chrono::Utc::now();
        
        // 计算数据集大小
        let size = features.len();
        
        // 创建基础元数据
        let metadata = DatasetMetadata {
            id: id.clone(),
            name: format!("Dataset-{}", &id[..8]),
            description: Some("Auto-generated dataset".to_string()),
            created_at: now,
            updated_at: now,
            version: "1.0".to_string(),
            owner: "system".to_string(),
            schema: None,
            properties: HashMap::new(),
            tags: Vec::new(),
            records_count: size,
            size_bytes: (size * std::mem::size_of::<f64>() * features.first().map(|f| f.len()).unwrap_or(0)) as u64,
        };
        
        Ok(Self {
            id: id.clone(),
            name: metadata.name.clone(),
            description: metadata.description.clone(),
            format: DataFormat::Vector,
            size,
            created_at: now,
            updated_at: now,
            metadata,
            path: format!("/tmp/dataset-{}", id),
            processed: false,
            loader: Arc::new(crate::data::loader::memory::MemoryDataLoader::new()),
            batch_size: 32,
            schema: None,
            batches: Vec::new(),
        })
    }
    
    /// 从ManagerDataset创建Dataset
    pub fn from_manager_dataset(manager_dataset: crate::data::manager::ManagerDataset) -> Result<Self> {
        let created_at = chrono::DateTime::<Utc>::from_timestamp(manager_dataset.created_at, 0)
            .unwrap_or_else(chrono::Utc::now);
        let updated_at = chrono::DateTime::<Utc>::from_timestamp(manager_dataset.updated_at, 0)
            .unwrap_or_else(chrono::Utc::now);
        
        // 转换数据格式
        let format = match manager_dataset.format.as_str() {
            "json" => DataFormat::Json,
            "csv" => DataFormat::Csv,
            "binary" | "bin" => DataFormat::Binary,
            "text" | "txt" => DataFormat::Text,
            "vector" => DataFormat::Vector,
            "matrix" => DataFormat::Matrix { rows: 0, cols: 0, dtype: "float32".to_string() },
            _ => DataFormat::Json,
        };
        
        // 创建数据集元数据
        let mut properties = HashMap::new();
        properties.insert("original_format".to_string(), manager_dataset.format.clone());
        properties.insert("dataset_type".to_string(), format!("{:?}", manager_dataset.dataset_type));
        
        // 合并原有元数据
        for (key, value) in manager_dataset.metadata.iter() {
            properties.insert(key.clone(), value.clone());
        }
        
        let metadata = DatasetMetadata {
            id: manager_dataset.id.clone(),
            name: manager_dataset.name.clone(),
            description: Some(format!("Converted from ManagerDataset - {}", manager_dataset.name)),
            created_at,
            updated_at,
            version: "1.0".to_string(),
            owner: "system".to_string(),
            schema: None,
            properties,
            tags: Vec::new(),
            records_count: manager_dataset.size,
            size_bytes: (manager_dataset.size as u64) * 1024, // 估算大小
        };
        
        // 创建适当的数据加载器
        let loader: Arc<dyn DataLoader> = Arc::new(crate::data::loader::memory::MemoryDataLoader::new());
        
        Ok(Self {
            id: manager_dataset.id,
            name: manager_dataset.name,
            description: Some(format!("Dataset converted from ManagerDataset")),
            format,
            size: manager_dataset.size,
            created_at,
            updated_at,
            path: format!("/data/managed/{}", metadata.id),
            metadata,
            processed: matches!(manager_dataset.dataset_type, crate::data::manager::DatasetType::Processed),
            loader,
            batch_size: 32,
            schema: None,
            batches: Vec::new(),
        })
    }
    
    /// 更新数据集信息
    pub fn update_metadata(&mut self, metadata: DatasetMetadata) {
        self.metadata = metadata;
        self.updated_at = chrono::Utc::now();
    }
    
    /// 标记为已处理
    pub fn mark_processed(&mut self) {
        self.processed = true;
        self.updated_at = chrono::Utc::now();
    }

    /// 创建批次迭代器
    pub fn create_batches(&self, batch_size: usize) -> Result<BatchIterator> {
        Ok(BatchIterator {
            dataset: Arc::new(self.clone()),
            batch_size,
            current_pos: 0,
        })
    }

    /// 将数据集转换为DataBatch
    pub fn to_batch(&self) -> Result<crate::data::DataBatch> {
        // 创建一个空的DataBatch，因为Dataset本身不包含实际数据
        // 实际数据需要通过loader加载
        let schema = self.schema.clone();
        let mut batch = crate::data::DataBatch::new(&self.id, 0, 0);
        batch.id = Some(format!("batch_{}", uuid::Uuid::new_v4()));
        batch.schema = schema;
        batch.data = Some(Vec::new()); // 空数据，需要通过loader加载
        batch.metadata = HashMap::new();
        
        Ok(batch)
    }
}

/// 批次迭代器
pub struct BatchIterator {
    dataset: Arc<Dataset>,
    batch_size: usize,
    current_pos: usize,
}

impl Iterator for BatchIterator {
    type Item = Result<crate::data::DataBatch>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.current_pos >= self.dataset.size {
            return None;
        }

        let start = self.current_pos;
        let end = std::cmp::min(start + self.batch_size, self.dataset.size);
        self.current_pos = end;

        // 在这里，我们假设DataLoader可以处理范围加载。
        // 这是一个简化的实现，实际应用中可能需要更复杂的逻辑。
        let source = crate::data::loader::types::DataSource::File(self.dataset.path.clone());
        // 从文件路径推断格式
        let format = if let Some(ext) = std::path::Path::new(&self.dataset.path).extension() {
            match ext.to_string_lossy().to_lowercase().as_str() {
                "csv" => crate::data::loader::types::DataFormat::Csv {
                    delimiter: ',',
                    has_header: true,
                    quote: '"',
                    escape: '"',
                },
                "json" => crate::data::loader::types::DataFormat::Json {
                    is_lines: false,
                    is_array: true,
                    options: Vec::new(),
                },
                "txt" | "text" => crate::data::loader::types::DataFormat::Text {
                    new_line: "\n".to_string(),
                    encoding: "utf-8".to_string(),
                },
                _ => crate::data::loader::types::DataFormat::CustomBinary(ext.to_string_lossy().to_string()),
            }
        } else {
            crate::data::loader::types::DataFormat::CustomBinary("bin".to_string())
        };
        
        let batch_result = futures::executor::block_on(
            self.dataset.loader.load_batch(&source, &format, end - start, start)
        );

        Some(batch_result)
    }
}

/// 扩展数据集
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ExtendedDataset {
    /// 数据集ID
    pub id: String,
    /// 数据集名称
    pub name: String,
    /// 数据集描述
    pub description: Option<String>,
    /// 数据来源
    pub source: DataSource,
    /// 数据格式
    pub format: DataFormat,
    /// 数据大小（字节）
    pub size: Option<usize>,
    /// 记录数量
    pub record_count: Option<usize>,
    /// 列信息（列名，类型）
    pub columns: Option<Vec<(String, String)>>,
    /// 数据状态
    pub status: DataStatus,
    /// 创建时间
    pub created_at: DateTime<Utc>,
    /// 更新时间
    pub updated_at: DateTime<Utc>,
    /// 元数据
    pub metadata: HashMap<String, String>,
}

impl ExtendedDataset {
    /// 创建新的扩展数据集
    pub fn new(name: &str, source: DataSource, format: DataFormat) -> Self {
        let now = Utc::now();
        Self {
            id: uuid::Uuid::new_v4().to_string(),
            name: name.to_string(),
            description: None,
            source,
            format,
            size: None,
            record_count: None,
            columns: None,
            status: DataStatus::Created,
            created_at: now,
            updated_at: now,
            metadata: HashMap::new(),
        }
    }

    /// 设置描述
    pub fn with_description(mut self, description: &str) -> Self {
        self.description = Some(description.to_string());
        self
    }

    /// 设置大小
    pub fn with_size(mut self, size: usize) -> Self {
        self.size = Some(size);
        self
    }

    /// 设置记录数量
    pub fn with_record_count(mut self, count: usize) -> Self {
        self.record_count = Some(count);
        self
    }

    /// 设置列信息
    pub fn with_columns(mut self, columns: Vec<(String, String)>) -> Self {
        self.columns = Some(columns);
        self
    }

    /// 添加元数据
    pub fn add_metadata(&mut self, key: &str, value: &str) {
        self.metadata.insert(key.to_string(), value.to_string());
        self.updated_at = Utc::now();
    }

    /// 更新状态
    pub fn update_status(&mut self, status: DataStatus) {
        self.status = status;
        self.updated_at = Utc::now();
    }
}

/// 已处理数据集
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ProcessedDataset {
    /// 数据集ID
    pub id: String,
    /// 原始数据集ID
    pub original_dataset_id: String,
    /// 数据集名称
    pub name: String,
    /// 数据集描述
    pub description: Option<String>,
    /// 处理步骤
    pub processing_steps: Vec<ProcessingStep>,
    /// 数据格式
    pub format: DataFormat,
    /// 数据大小（字节）
    pub size: Option<usize>,
    /// 记录数量
    pub record_count: Option<usize>,
    /// 列信息（列名，类型）
    pub columns: Option<Vec<(String, String)>>,
    /// 数据状态
    pub status: DataStatus,
    /// 创建时间
    pub created_at: DateTime<Utc>,
    /// 更新时间
    pub updated_at: DateTime<Utc>,
    /// 元数据
    pub metadata: HashMap<String, String>,
    /// 批次大小
    pub batch_size: usize,
    /// 数据批次
    pub batches: Vec<crate::data::processor::ProcessedBatch>,
}

impl ProcessedDataset {
    /// 创建新的已处理数据集
    pub fn new(original_dataset_id: &str, name: &str, format: DataFormat) -> Self {
        let now = Utc::now();
        Self {
            id: uuid::Uuid::new_v4().to_string(),
            original_dataset_id: original_dataset_id.to_string(),
            name: name.to_string(),
            description: None,
            processing_steps: Vec::new(),
            format,
            size: None,
            record_count: None,
            columns: None,
            status: DataStatus::Created,
            created_at: now,
            updated_at: now,
            metadata: HashMap::new(),
            batch_size: 32,
            batches: Vec::new(),
        }
    }

    /// 使用数据创建已处理数据集
    pub fn from_data(
        id: String,
        content: Vec<f32>,
        dimensions: Vec<usize>,
        labels: Option<Vec<String>>,
        metadata: HashMap<String, String>,
    ) -> Self {
        let now = Utc::now();
        
        // 根据content和dimensions创建批次
        let mut batches = Vec::new();
        if !content.is_empty() {
            // 计算每个样本的大小
            let sample_size = if dimensions.is_empty() { 1 } else { dimensions.iter().product() };
            let batch_size = 32; // 默认批次大小
            
            // 将数据分割成批次
            for chunk_start in (0..content.len()).step_by(batch_size * sample_size) {
                let chunk_end = std::cmp::min(chunk_start + batch_size * sample_size, content.len());
                let chunk_data = content[chunk_start..chunk_end].to_vec();
                
                // 创建批次标签
                let batch_labels = if let Some(ref all_labels) = labels {
                    let label_start = chunk_start / sample_size;
                    let label_end = std::cmp::min(label_start + batch_size, all_labels.len());
                    Some(all_labels[label_start..label_end].to_vec())
                } else {
                    None
                };
                
                // 创建ProcessedBatch
                // 构造 TensorData
                let tensor = TensorData {
                    id: uuid::Uuid::new_v4().to_string(),
                    shape: dimensions.clone(),
                    data: chunk_data,
                    dtype: "float32".to_string(),
                    device: "cpu".to_string(),
                    requires_grad: false,
                    metadata: metadata.clone(),
                    created_at: chrono::Utc::now(),
                    updated_at: chrono::Utc::now(),
                };
                let labels_tensor = batch_labels.as_ref().map(|labels_vec| TensorData {
                    id: uuid::Uuid::new_v4().to_string(),
                    shape: vec![labels_vec.len()],
                    data: labels_vec.iter().map(|_| 0.0_f32).collect(), // 占位数值，实际标签另存metadata
                    dtype: "float32".to_string(),
                    device: "cpu".to_string(),
                    requires_grad: false,
                    metadata: {
                        let mut m = metadata.clone();
                        m.insert("labels_text".to_string(), labels_vec.join(","));
                        m
                    },
                    created_at: chrono::Utc::now(),
                    updated_at: chrono::Utc::now(),
                });
                let batch = crate::data::processor::ProcessedBatch::new(
                    uuid::Uuid::new_v4().to_string(),
                    tensor,
                    labels_tensor,
                    metadata.clone(),
                    crate::data::DataFormat::Vector,
                );
                
                batches.push(batch);
            }
        }
        
        Self {
            id: id.clone(),
            original_dataset_id: id.clone(),
            name: format!("Dataset-{}", &id[..8]),
            description: None,
            processing_steps: Vec::new(),
            format: DataFormat::Vector,
            size: Some(content.len() * std::mem::size_of::<f32>()),
            record_count: Some(content.len() / dimensions.iter().product::<usize>().max(1)),
            columns: None,
            status: DataStatus::Loaded,
            created_at: now,
            updated_at: now,
            metadata,
            batch_size: 32,
            batches,
        }
    }

    /// 获取批次数量
    pub fn batch_count(&self) -> usize {
        self.batches.len()
    }

    /// 获取指定索引的批次
    pub async fn get_batch(&self, batch_idx: usize) -> Result<crate::data::processor::ProcessedBatch> {
        if batch_idx >= self.batches.len() {
            return Err(crate::error::Error::invalid_data(format!("批次索引超出范围: {}", batch_idx)));
        }
        Ok(self.batches[batch_idx].clone())
    }

    /// 添加批次
    pub fn add_batch(&mut self, batch: crate::data::processor::ProcessedBatch) {
        self.batches.push(batch);
        self.updated_at = Utc::now();
    }

    /// 设置批次大小
    pub fn with_batch_size(mut self, batch_size: usize) -> Self {
        self.batch_size = batch_size;
        self
    }

    /// 设置描述
    pub fn with_description(mut self, description: &str) -> Self {
        self.description = Some(description.to_string());
        self
    }

    /// 添加处理步骤
    pub fn add_processing_step(&mut self, step: ProcessingStep) {
        self.processing_steps.push(step);
        self.updated_at = Utc::now();
    }

    /// 设置大小
    pub fn with_size(mut self, size: usize) -> Self {
        self.size = Some(size);
        self
    }

    /// 设置记录数量
    pub fn with_record_count(mut self, count: usize) -> Self {
        self.record_count = Some(count);
        self
    }

    /// 设置列信息
    pub fn with_columns(mut self, columns: Vec<(String, String)>) -> Self {
        self.columns = Some(columns);
        self
    }

    /// 添加元数据
    pub fn add_metadata(&mut self, key: &str, value: &str) {
        self.metadata.insert(key.to_string(), value.to_string());
        self.updated_at = Utc::now();
    }

    /// 更新状态
    pub fn update_status(&mut self, status: DataStatus) {
        self.status = status;
        self.updated_at = Utc::now();
    }

    /// 从DataBatch创建ProcessedDataset
    pub fn from_data_batch(data_batch: crate::data::exports::DataBatch) -> Result<Self> {
        let now = Utc::now();
        let batch_id: String = data_batch.id.clone();
        
        // 将exports::DataBatch转换为ProcessedBatch
        let processed_batch = crate::data::processor::ProcessedBatch::from_data_batch(data_batch)?;
        
        Ok(Self {
            id: batch_id.clone(),
            original_dataset_id: batch_id.clone(),
            name: format!("ProcessedDataset-{}", &batch_id[..8]),
            description: Some("从DataBatch创建的ProcessedDataset".to_string()),
            processing_steps: Vec::new(),
            format: DataFormat::JSON,
            size: Some(processed_batch.features.data.len() * std::mem::size_of::<f32>()),
            record_count: Some(processed_batch.records.len()),
            columns: None,
            status: DataStatus::Loaded,
            created_at: now,
            updated_at: now,
            metadata: HashMap::new(),
            batch_size: processed_batch.records.len().max(1),
            batches: vec![processed_batch],
        })
    }
}

/// 数据来源枚举
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum DataSource {
    /// 文件系统
    FileSystem(String),
    /// 数据库
    Database(LocalDatabaseConfig),
    /// API
    Api(ApiConfig),
    /// 内存数据
    Memory,
    /// 自定义
    Custom(String),
}

/// 本地数据库配置
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LocalDatabaseConfig {
    /// 数据库类型
    pub db_type: String,
    /// 连接字符串
    pub connection_string: String,
    /// 查询语句
    pub query: Option<String>,
    /// 表名
    pub table: Option<String>,
}

/// API配置
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ApiConfig {
    /// API URL
    pub url: String,
    /// HTTP方法
    pub method: String,
    /// 请求头
    pub headers: HashMap<String, String>,
    /// 请求体
    pub body: Option<String>,
} 