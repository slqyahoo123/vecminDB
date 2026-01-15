use crate::error::{Error, Result, WithErrorContext};
use crate::status::{StatusType, StatusTrackerTrait};
use crate::storage::Storage;
use uuid::Uuid;
use std::sync::{Arc, Mutex};
use std::collections::HashMap;
use std::path::Path;
use async_trait::async_trait;
use tokio::fs::File;
use tokio::io::AsyncReadExt;
use serde::{Serialize, Deserialize};
use crate::data::DataBatch;
// ndarray::Array not used in this module
use base64::{Engine as _, engine::general_purpose};
use crate::core::types::{CoreDataSchema, CoreSchemaField, CoreTensorData, CoreFieldType};
use crate::data::connector::{DatabaseConnectorFactory, DatabaseType, DatabaseConfig, QueryParams};
use std::pin::Pin;
use url::Url;
use log::info;
#[cfg(feature = "websocket")]
use log::warn;
use futures::StreamExt;

/// 数据集类型
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum DatasetType {
    /// 原始数据
    Raw,
    /// 已处理数据
    Processed,
    /// 训练数据
    Training,
    /// 验证数据
    Validation,
    /// 测试数据
    Test,
}

/// 数据集描述（管理器专用）
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ManagerDataset {
    /// 数据集ID
    pub id: String,
    /// 数据集名称
    pub name: String,
    /// 数据集类型
    pub dataset_type: DatasetType,
    /// 数据集格式
    pub format: String,
    /// 数据集大小
    pub size: usize,
    /// 数据集元数据
    pub metadata: HashMap<String, String>,
    /// 创建时间
    pub created_at: i64,
    /// 更新时间
    pub updated_at: i64,
}

/// ManagerDataset 的 DataLoader 实现
#[async_trait]
impl crate::data::loader::DataLoader for ManagerDataset {
    async fn load(&self, _source: &crate::data::loader::types::DataSource, _format: &crate::data::loader::types::DataFormat) -> crate::error::Result<crate::data::DataBatch> {
        // 从数据集创建 DataBatch
        crate::data::DataBatch::from_manager_dataset(self.clone())
    }
    
    async fn get_schema(&self, _source: &crate::data::loader::types::DataSource, _format: &crate::data::loader::types::DataFormat) -> crate::error::Result<crate::data::DataSchema> {
        // 创建基本的数据架构
        Ok(crate::data::DataSchema::new("manager_dataset", "1.0"))
    }
    
    fn name(&self) -> &'static str {
        "manager_dataset_loader"
    }
    
    async fn load_batch(&self, _source: &crate::data::loader::types::DataSource, _format: &crate::data::loader::types::DataFormat, _batch_size: usize, _offset: usize) -> crate::error::Result<crate::data::DataBatch> {
        // 使用默认的load实现
        self.load(_source, _format).await
    }
    
    fn supports_format(&self, format: &crate::data::loader::types::DataFormat) -> bool {
        use crate::data::loader::types::DataFormat;
        match format {
            DataFormat::Json { .. } => self.format == "json",
            DataFormat::Csv { .. } => self.format == "csv",
            DataFormat::Text { .. } => self.format == "text" || self.format == "txt",
            DataFormat::CustomBinary(name) => self.format == *name || self.format == "binary" || self.format == "bin",
            DataFormat::CustomText(name) => self.format == *name,
            _ => false,
        }
    }
    
    fn config(&self) -> &crate::data::loader::LoaderConfig {
        // 返回默认配置的静态引用
        &*Box::leak(Box::new(crate::data::loader::LoaderConfig::default()))
    }
    
    fn set_config(&mut self, _config: crate::data::loader::LoaderConfig) {
        // ManagerDataset是不可变的，所以此方法为空实现
    }
}

/// 数据管理器，负责数据的存储、检索和处理
pub struct DataManager {
    storage: Arc<Storage>,
    status_tracker: Option<Arc<dyn StatusTrackerTrait>>,
    cache: Mutex<HashMap<String, Arc<ManagerDataset>>>,
}

/// 数据预处理配置
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataProcessConfig {
    /// 处理步骤
    pub steps: Vec<ProcessStep>,
    /// 输出格式
    pub output_format: String,
    /// 自定义配置
    pub custom_config: HashMap<String, String>,
}

/// 处理步骤
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProcessStep {
    /// 步骤名称
    pub name: String,
    /// 步骤参数
    pub params: HashMap<String, String>,
}

/// 数据管道特性定义
pub trait DataPipeline {
    /// 处理数据批次
    fn process(&self, batch: &mut DataBatch) -> std::result::Result<(), String>;
    
    /// 获取管道配置
    fn get_config(&self) -> HashMap<String, String>;
}

/// 数据管理器配置
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataManagerConfig {
    /// 存储路径
    pub storage_path: String,
    /// 缓存大小限制
    pub cache_size_limit: usize,
    /// 是否启用状态跟踪
    pub enable_status_tracking: bool,
    /// 批次大小默认值
    pub default_batch_size: usize,
    /// 最大并发处理数
    pub max_concurrent_tasks: usize,
    /// 数据压缩设置
    pub enable_compression: bool,
    /// 自动清理间隔（秒）
    pub cleanup_interval_seconds: u64,
}

impl Default for DataManagerConfig {
    fn default() -> Self {
        Self {
            storage_path: "./data".to_string(),
            cache_size_limit: 1000,
            enable_status_tracking: true,
            default_batch_size: 1000,
            max_concurrent_tasks: 10,
            enable_compression: false,
            cleanup_interval_seconds: 3600,
        }
    }
}

impl DataManager {
    /// 使用配置创建新的数据管理器
    pub fn new(config: DataManagerConfig) -> Result<Self> {
        // 创建默认存储引擎
        // 使用 module::config::StorageConfig 作为 Storage::new 的配置类型
        let storage = crate::storage::Storage::new(crate::storage::module::config::StorageConfig::default())?;
        
        Ok(Self {
            storage,
            status_tracker: None,
            cache: Mutex::new(HashMap::new()),
        })
    }
    
    /// 使用现有存储创建数据管理器
    pub fn with_storage(storage: Arc<Storage>) -> Self {
        Self {
            storage,
            status_tracker: None,
            cache: Mutex::new(HashMap::new()),
        }
    }

    /// 设置状态跟踪器
    pub fn with_status_tracker(mut self, status_tracker: Arc<dyn StatusTrackerTrait>) -> Self {
        self.status_tracker = Some(status_tracker);
        self
    }

    /// 导入数据
    pub async fn import_data(&self, name: &str, data_path: &Path, metadata: HashMap<String, String>) -> Result<ManagerDataset> {
        // 生成唯一ID
        let dataset_id = Uuid::new_v4().to_string();
        
        // 创建任务ID用于状态追踪（通过StatusEvent）
        let task_id = if let Some(tracker) = &self.status_tracker {
            let id = Uuid::new_v4();
            let event = crate::status::StatusEvent::new(
                id,
                StatusType::Initializing,
                "创建任务: data_import".to_string(),
            ).with_metadata("task_type", "data_import");
            tracker.update_status(event).await?;
            Some(id)
        } else { None };
        
        // 更新状态
        if let (Some(task_id), Some(tracker)) = (task_id, &self.status_tracker) {
            let mut event = crate::status::StatusEvent::new(
                task_id,
                StatusType::DataPreparing,
                "开始导入数据".to_string(),
            );
            event.progress = Some(0);
            tracker.update_status(event).await?;
        }
        
        // 读取数据文件
        let mut file = File::open(data_path).await
            .map_err(|e| Error::Io(e))
            .with_context("无法打开数据文件")?;
        
        let mut buffer = Vec::new();
        file.read_to_end(&mut buffer).await
            .map_err(|e| Error::Io(e))
            .with_context("无法读取数据文件")?;
        
        // 存储数据到存储引擎
        let now = chrono::Utc::now().timestamp();
        let dataset = ManagerDataset {
            id: dataset_id.clone(),
            name: name.to_string(),
            dataset_type: DatasetType::Raw,
            format: data_path.extension().and_then(|e| e.to_str()).unwrap_or("bin").to_string(),
            size: buffer.len(),
            metadata,
            created_at: now,
            updated_at: now,
        };
        
        // 存储数据和元数据
        self.storage.put_data(&dataset_id, &buffer).await
            .map_err(|e| Error::storage(format!("无法存储数据内容: {}", e)))?;
        
        self.storage.put_dataset(&dataset_id, &dataset).await
            .map_err(|e| Error::storage(format!("无法存储数据集元数据: {}", e)))?;
        
        // 更新缓存
        let mut cache = self.cache.lock().map_err(|_| Error::Lock("无法锁定数据缓存".to_string()))?;
        cache.insert(dataset_id.clone(), Arc::new(dataset.clone()));
        
        // 更新状态
        if let (Some(task_id), Some(tracker)) = (task_id, &self.status_tracker) {
            let mut event = crate::status::StatusEvent::new(
                task_id,
                StatusType::Completed,
                "数据导入完成".to_string(),
            );
            event.progress = Some(100);
            tracker.update_status(event).await?;
        }
        
        Ok(dataset)
    }

    /// 根据数据集ID获取数据集（别名方法）
    pub fn get_dataset_by_id(&self, dataset_id: &str) -> Result<Option<Arc<ManagerDataset>>> {
        self.get_dataset(dataset_id)
    }

    /// 获取数据集
    pub fn get_dataset(&self, dataset_id: &str) -> Result<Option<Arc<ManagerDataset>>> {
        // 先检查缓存
        {
            let cache = self.cache.lock().map_err(|_| Error::Lock("无法锁定数据缓存".to_string()))?;
            if let Some(dataset) = cache.get(dataset_id) {
                return Ok(Some(dataset.clone()));
            }
        }
        
        // 从存储引擎获取（同步封装异步操作）
        let dataset_opt: Option<ManagerDataset> = {
            // 尝试复用当前 tokio 运行时，如果不存在则创建临时运行时
            if let Ok(handle) = tokio::runtime::Handle::try_current() {
                handle.block_on(self.storage.get_dataset(dataset_id))?
            } else {
                let rt = tokio::runtime::Runtime::new()
                    .map_err(|e| Error::Lock(format!("无法创建 tokio 运行时: {}", e)))?;
                rt.block_on(self.storage.get_dataset(dataset_id))?
            }
        };
        
        if let Some(dataset) = dataset_opt {
            let dataset_arc = Arc::new(dataset);
            
            // 更新缓存
            let mut cache = self.cache.lock().map_err(|_| Error::Lock("无法锁定数据缓存".to_string()))?;
            cache.insert(dataset_id.to_string(), dataset_arc.clone());
            
            Ok(Some(dataset_arc))
        } else {
            Ok(None)
        }
    }

    /// 获取原始数据
    pub async fn get_raw_data(&self, dataset_id: &str) -> Result<Option<Vec<u8>>> {
        self.storage.get_data(dataset_id).await
    }

    /// 异步读取文件数据
    pub async fn read_file_async(&self, file_path: &Path) -> Result<Vec<u8>> {
        let mut file = File::open(file_path).await
            .map_err(|e| Error::io_error(format!("无法打开文件 {:?}: {}", file_path, e)))?;
        
        let mut buffer = Vec::new();
        file.read_to_end(&mut buffer).await
            .map_err(|e| Error::io_error(format!("读取文件失败: {}", e)))?;
        
        Ok(buffer)
    }

    /// 异步读取文件数据块
    pub async fn read_file_chunk_async(&self, file_path: &Path, chunk_size: usize) -> Result<Vec<Vec<u8>>> {
        let mut file = File::open(file_path).await
            .map_err(|e| Error::io_error(format!("无法打开文件 {:?}: {}", file_path, e)))?;
        
        let mut chunks = Vec::new();
        let mut buffer = vec![0u8; chunk_size];
        
        loop {
            let bytes_read = file.read(&mut buffer).await
                .map_err(|e| Error::io_error(format!("读取文件块失败: {}", e)))?;
            
            if bytes_read == 0 {
                break;
            }
            
            chunks.push(buffer[..bytes_read].to_vec());
        }
        
        Ok(chunks)
    }

    /// 获取处理后的数据
    pub async fn get_processed_data(&self, dataset_id: &str) -> Result<Option<Vec<u8>>> {
        let processed_key = format!("{}_processed", dataset_id);
        self.storage.get_data(&processed_key).await
    }

    /// 处理数据（兼容API调用）
    pub async fn process_data(&self, dataset_id: &str, process_type: &str, config: Option<serde_json::Value>) -> Result<serde_json::Value> {
        // 将API参数转换为内部配置
        let process_config = DataProcessConfig {
            steps: vec![ProcessStep {
                name: process_type.to_string(),
                params: config.as_ref()
                    .and_then(|v| v.as_object())
                    .map(|obj| obj.iter().map(|(k, v)| (k.clone(), v.to_string())).collect())
                    .unwrap_or_default(),
            }],
            output_format: "json".to_string(),
            custom_config: HashMap::new(),
        };
        
        let result = self.process_data_internal(dataset_id, process_config).await?;
        
        // 返回处理结果的JSON表示
        Ok(serde_json::json!({
            "dataset_id": result.id,
            "name": result.name,
            "size": result.size,
            "format": result.format,
            "created_at": result.created_at,
            "updated_at": result.updated_at
        }))
    }

    /// 内部处理数据方法
    pub async fn process_data_internal(&self, dataset_id: &str, config: DataProcessConfig) -> Result<ManagerDataset> {
        // 创建任务ID用于状态追踪
        let task_id = if let Some(tracker) = &self.status_tracker {
            let id = Uuid::new_v4();
            let event = crate::status::StatusEvent::new(
                id,
                StatusType::Initializing,
                "创建任务: data_processing".to_string(),
            ).with_metadata("task_type", "data_processing");
            tracker.update_status(event).await?;
            Some(id)
        } else { None };
        
        // 更新状态
        if let (Some(task_id), Some(tracker)) = (task_id, &self.status_tracker) {
            let event = crate::status::StatusEvent::new(
                task_id,
                StatusType::DataPreparing,
                "开始处理数据".to_string(),
            );
            tracker.update_status(event).await?;
        }
        
        // 获取原始数据
        let raw_data = self.get_raw_data(dataset_id).await?
            .ok_or_else(|| Error::invalid_data(format!("找不到数据集: {}", dataset_id)))?;
        
        // 获取数据集元数据
        let dataset = self.get_dataset(dataset_id)?
            .ok_or_else(|| Error::invalid_data(format!("找不到数据集元数据: {}", dataset_id)))?;
        
        // 进行数据处理逻辑
        // 生产级实现：根据配置进行真正的数据处理转换
        let processed_data = Self::process_data_with_config(&raw_data, &config, &dataset.format)?;
        
        // 创建新的处理后数据集
        let processed_id = format!("{}_processed", dataset_id);
        let now = chrono::Utc::now().timestamp();
        let processed_dataset = ManagerDataset {
            id: processed_id.clone(),
            name: format!("{}_processed", dataset.name),
            dataset_type: DatasetType::Processed,
            format: config.output_format.clone(),
            size: processed_data.len(),
            metadata: {
                let mut metadata = dataset.metadata.clone();
                metadata.insert("original_id".to_string(), dataset_id.to_string());
                metadata.insert("process_config".to_string(), serde_json::to_string(&config).unwrap_or_default());
                metadata
            },
            created_at: now,
            updated_at: now,
        };
        
        // 存储处理后的数据和元数据
        self.storage.put_data(&processed_id, &processed_data).await
            .map_err(|e| Error::invalid_data(format!("无法存储处理后的数据: {}", e)))?;
        
        self.storage.put_dataset(&processed_id, &processed_dataset).await
            .map_err(|e| Error::invalid_data(format!("无法存储处理后的数据集元数据: {}", e)))?;
        
        // 更新缓存
        let mut cache = self.cache.lock().map_err(|_| Error::Lock("无法锁定数据缓存".to_string()))?;
        cache.insert(processed_id.clone(), Arc::new(processed_dataset.clone()));
        
        // 更新状态
        if let (Some(task_id), Some(tracker)) = (task_id, &self.status_tracker) {
            let mut event = crate::status::StatusEvent::new(
                task_id,
                StatusType::Completed,
                "数据处理完成".to_string(),
            );
            event.progress = Some(100);
            tracker.update_status(event).await?;
        }
        
        Ok(processed_dataset)
    }

    /// 创建数据批次
    pub async fn create_batch(&self, dataset_id: &str, batch_size: usize, batch_index: usize) -> Result<DataBatch> {
        // 获取数据
        let data = if let Some(processed_data) = self.get_processed_data(dataset_id).await? {
            processed_data
        } else if let Some(raw_data) = self.get_raw_data(dataset_id).await? {
            raw_data
        } else {
            return Err(Error::invalid_data(format!("找不到数据集: {}", dataset_id)));
        };
        
        // 生产级实现：根据batch_index和batch_size截取相应部分
        let start_idx = batch_index * batch_size;
        let end_idx = std::cmp::min(start_idx + batch_size, data.len());
        
        if start_idx >= data.len() {
            return Err(Error::invalid_data(format!("批次索引超出范围: batch_index={}, data_len={}", batch_index, data.len())));
        }
        
        let batch_data = data[start_idx..end_idx].to_vec();
        
        // 获取数据集以提取标签字段信息
        let dataset = self.get_dataset(dataset_id)?
            .ok_or_else(|| Error::invalid_data(format!("找不到数据集元数据: {}", dataset_id)))?;
        
        // 从元数据中提取标签字段（create_batch 方法没有 config 参数，使用默认配置）
        let default_config = DataProcessConfig {
            steps: Vec::new(),
            output_format: "binary".to_string(),
            custom_config: HashMap::new(),
        };
        let labels = Self::extract_labels_from_data(&batch_data, &default_config, &dataset.metadata)?;
        
        // 创建数据批次
        let batch = DataBatch {
            id: Some(format!("{}_{}", dataset_id, batch_index)),
            dataset_id: dataset_id.to_string(),
            index: batch_index,
            batch_index,
            size: batch_data.len(),
            batch_size,
            status: crate::data::types::DataStatus::Loaded,
            created_at: chrono::Utc::now(),
            data: Some(batch_data),
            labels,
            metadata: HashMap::new(),
            format: crate::data::types::DataFormat::Binary,
            source: Some(dataset_id.to_string()),
            records: Vec::new(),
            schema: None,
            field_names: Vec::new(),
            features: None,
            target: None,
            version: None,
            checksum: None,
            compression: None,
            encryption: None,
            tags: Vec::new(),
            ..Default::default()
        };
        
        Ok(batch)
    }

    /// 列出所有数据集
    pub fn list_datasets(&self) -> Result<Vec<ManagerDataset>> {
        // 使用存储引擎的 get_dataset_ids 和 get_dataset 异步方法
        let ids: Vec<String> = {
            if let Ok(handle) = tokio::runtime::Handle::try_current() {
                handle.block_on(self.storage.get_dataset_ids())?
            } else {
                let rt = tokio::runtime::Runtime::new()
                    .map_err(|e| Error::Lock(format!("无法创建 tokio 运行时: {}", e)))?;
                rt.block_on(self.storage.get_dataset_ids())?
            }
        };
        
        let mut datasets = Vec::new();
        for id in ids {
            let dataset_opt: Option<ManagerDataset> = {
                if let Ok(handle) = tokio::runtime::Handle::try_current() {
                    handle.block_on(self.storage.get_dataset(&id))?
                } else {
                    let rt = tokio::runtime::Runtime::new()
                        .map_err(|e| Error::Lock(format!("无法创建 tokio 运行时: {}", e)))?;
                    rt.block_on(self.storage.get_dataset(&id))?
                }
            };
            
            if let Some(dataset) = dataset_opt {
                datasets.push(dataset);
            }
        }
        
        Ok(datasets)
    }

    /// 删除数据集
    pub async fn delete_dataset(&self, dataset_id: &str) -> Result<bool> {
        // 先检查数据集是否存在
        let exists = self.get_dataset(dataset_id)?.is_some();
        
        if exists {
            // 删除数据和元数据
            self.storage.delete_data(dataset_id).await?;
            self.storage.delete_dataset(dataset_id).await?;
            
            // 更新缓存
            let mut cache = self.cache.lock().map_err(|_| Error::Lock("无法锁定数据缓存".to_string()))?;
            cache.remove(dataset_id);
            
            Ok(true)
        } else {
            Ok(false)
        }
    }

    /// 获取数据批次引用，避免完整拷贝
    pub async fn get_data_batch_ref(&self, dataset_id: &str) -> Result<Arc<DataBatch>> {
        // 创建任务ID用于状态追踪
        let task_id = if let Some(tracker) = &self.status_tracker {
            let id = Uuid::new_v4();
            let event = crate::status::StatusEvent::new(
                id,
                StatusType::Initializing,
                "创建任务: data_retrieval".to_string(),
            ).with_metadata("task_type", "data_retrieval");
            tracker.update_status(event).await?;
            Some(id)
        } else { None };
        
        // 更新状态
        if let (Some(task_id), Some(tracker)) = (task_id, &self.status_tracker) {
            let event = crate::status::StatusEvent::new(
                task_id,
                StatusType::DataLoading,
                "开始加载数据批次".to_string(),
            );
            tracker.update_status(event).await?;
        }
        
        // 获取原始数据
        let raw_data = self.get_raw_data(dataset_id).await?
            .ok_or_else(|| Error::invalid_data(format!("找不到数据集: {}", dataset_id)))?;
        
        // 获取数据集元数据
        let dataset = self.get_dataset(dataset_id)?
            .ok_or_else(|| Error::invalid_data(format!("找不到数据集元数据: {}", dataset_id)))?;
        
        // 创建数据批次
        let batch = DataBatch {
            id: Some(format!("{}_0", dataset_id)),
            dataset_id: dataset_id.to_string(),
            index: 0,
            batch_index: 0,
            size: raw_data.len(),
            batch_size: raw_data.len(),
            status: crate::data::types::DataStatus::Loaded,
            created_at: chrono::Utc::now(),
            data: Some(raw_data),
            labels: None, // 实际应用中应该根据数据格式解析标签
            metadata: dataset.metadata.clone(),
            format: crate::data::types::DataFormat::Binary,
            source: Some(dataset_id.to_string()),
            records: Vec::new(),
            schema: None,
            field_names: Vec::new(),
            features: None,
            target: None,
            version: None,
            checksum: None,
            compression: None,
            encryption: None,
            tags: Vec::new(),
            ..Default::default()
        };
        
        // 更新状态
        if let (Some(task_id), Some(tracker)) = (task_id, &self.status_tracker) {
            let mut event = crate::status::StatusEvent::new(
                task_id,
                StatusType::Completed,
                "数据批次加载完成".to_string(),
            );
            event.progress = Some(100);
            tracker.update_status(event).await?;
        }
        
        Ok(Arc::new(batch))
    }

    /// 创建数据批次迭代器
    pub fn create_batch_iterator(&self, dataset_id: &str, batch_size: usize) -> Result<crate::data::iterator::DataBatchIterator> {
        // 获取数据集元数据
        let dataset = self.get_dataset(dataset_id)?
            .ok_or_else(|| Error::invalid_data(format!("找不到数据集元数据: {}", dataset_id)))?;
        
        // 创建数据加载器适配器
        let loader = Arc::new(StorageDataLoader::new(self.storage.clone()));
        
        // 构造数据路径
        let data_path = format!("dataset/{}", dataset_id);
        
        Ok(crate::data::iterator::DataBatchIterator::new(loader, data_path, batch_size))
    }
    
    /// 从文件加载数据 - 支持端到端管道
    pub async fn load_from_file(&self, file_path: &str) -> Result<(crate::core::types::CoreDataBatch, CoreDataSchema)> {
        // 生成唯一的数据集ID
        let dataset_id = format!("file_{}", uuid::Uuid::new_v4());
        
        // 读取文件数据
        let data = tokio::fs::read(file_path).await
            .map_err(|e| Error::Io(e))
            .with_context("无法读取文件")?;
        
        // 创建核心数据批次
        let data_batch = crate::core::types::CoreDataBatch {
            id: dataset_id.clone(),
            data: Vec::<crate::core::types::CoreTensorData>::new(), // 空的数据向量，因为原始数据是Vec<u8>
            labels: None,
            batch_size: 1,
            metadata: Some({
                let mut metadata = HashMap::new();
                metadata.insert("source_file".to_string(), file_path.to_string());
                metadata.insert("size".to_string(), data.len().to_string());
                metadata.insert("raw_data".to_string(), general_purpose::STANDARD.encode(&data)); // 将原始数据编码存储
                metadata
            }),
            created_at: chrono::Utc::now(),
            updated_at: chrono::Utc::now(),
        };
        
        // 创建核心数据模式
        let data_schema = CoreDataSchema {
            id: format!("{}_schema", dataset_id),
            name: format!("File Schema for {}", file_path),
            fields: vec![
                CoreSchemaField {
                    name: "data".to_string(),
                    field_type: crate::core::types::CoreFieldType::Array, // 使用Array类型代替不存在的Binary
                    nullable: false,
                    description: Some("Binary file data".to_string()),
                    constraints: None,
                }
            ],
            metadata: {
                let mut metadata = HashMap::new();
                metadata.insert("source_type".to_string(), "file".to_string());
                metadata.insert("source_path".to_string(), file_path.to_string());
                metadata.insert("version".to_string(), "1.0".to_string());
                metadata.insert("target_field".to_string(), "".to_string());
                metadata.insert("feature_fields".to_string(), "data".to_string());
                metadata
            },
            created_at: chrono::Utc::now(),
            updated_at: chrono::Utc::now(),
        };
        
        Ok((data_batch, data_schema))
    }
    
    /// 从数据库加载数据 - 支持端到端管道
    /// 
    /// 生产级实现：真正连接数据库，执行查询，并将结果转换为 CoreDataBatch
    pub async fn load_from_database(&self, connection_string: &str) -> Result<(crate::core::types::CoreDataBatch, CoreDataSchema)> {
        let dataset_id = format!("db_{}", uuid::Uuid::new_v4());
        
        // 解析连接字符串以确定数据库类型
        let db_type = Self::parse_connection_string(connection_string)?;
        
        // 从连接字符串中提取数据库配置信息
        let (username, password, database, table) = Self::parse_connection_string_parts(connection_string)?;
        
        // 创建数据库配置
        let db_type_clone = db_type.clone();
        let db_type_str = format!("{:?}", db_type_clone);
        let db_config = DatabaseConfig {
            db_type,
            connection_string: connection_string.to_string(),
            username,
            password,
            database,
            table: table.clone(),
            query: None,
            pool_size: Some(10),
            timeout: Some(30),
            extra_params: HashMap::new(),
        };
        
        // 创建数据库连接器
        let mut connector = DatabaseConnectorFactory::create_connector(db_config)
            .map_err(|e| Error::invalid_data(format!("无法创建数据库连接器: {}", e)))?;
        
        // 连接到数据库
        Pin::from(connector.connect()).await
            .map_err(|e| Error::invalid_data(format!("数据库连接失败: {}", e)))?;
        
        // 构建查询参数
        // 如果没有指定表，尝试查询系统表获取可用表列表
        let query = if let Some(ref table_name) = table {
            format!("SELECT * FROM {}", table_name)
        } else {
            // 根据数据库类型构建不同的查询
            match &db_type_clone {
                DatabaseType::PostgreSQL => "SELECT * FROM information_schema.tables WHERE table_schema = 'public' LIMIT 1".to_string(),
                DatabaseType::MySQL => "SHOW TABLES LIMIT 1".to_string(),
                DatabaseType::SQLite => "SELECT name FROM sqlite_master WHERE type='table' LIMIT 1".to_string(),
                _ => "SELECT 1".to_string(), // 默认查询
            }
        };
        
        let query_params = QueryParams {
            query,
            params: Vec::new(),
            limit: Some(1000), // 限制返回行数
            offset: None,
            sort_by: None,
            sort_direction: None,
            table_name: table.clone(),
        };
        
        // 执行查询
        let data_batch_result = Pin::from(connector.query(&query_params)).await
            .map_err(|e| Error::invalid_data(format!("数据库查询失败: {}", e)))?;
        
        // 获取数据库模式
        let schema_result = Pin::from(connector.get_schema(table.as_deref())).await
            .map_err(|e| Error::invalid_data(format!("获取数据库模式失败: {}", e)))?;
        
        // 断开数据库连接
        let _ = Pin::from(connector.disconnect()).await;
        
        // 将 DataBatch 转换为 CoreDataBatch
        let core_tensors = Self::convert_data_batch_to_core_tensors(&data_batch_result)?;
        
        // 将 DataSchema 转换为 CoreDataSchema
        let core_schema = Self::convert_data_schema_to_core_schema(&schema_result, &dataset_id)?;
        
        // 创建 CoreDataBatch
        let core_data_batch = crate::core::types::CoreDataBatch {
            id: dataset_id.clone(),
            data: core_tensors,
            labels: None,
            batch_size: data_batch_result.records.len(),
            metadata: Some({
                let mut metadata = HashMap::new();
                metadata.insert("source_type".to_string(), "database".to_string());
                metadata.insert("connection_string".to_string(), connection_string.to_string());
                metadata.insert("database_type".to_string(), db_type_str);
                metadata.insert("table_name".to_string(), table.unwrap_or_default());
                metadata.insert("record_count".to_string(), data_batch_result.records.len().to_string());
                metadata
            }),
            created_at: chrono::Utc::now(),
            updated_at: chrono::Utc::now(),
        };
        
        Ok((core_data_batch, core_schema))
    }
    
    /// 解析连接字符串以确定数据库类型
    fn parse_connection_string(connection_string: &str) -> Result<DatabaseType> {
        if connection_string.starts_with("postgresql://") || connection_string.starts_with("postgres://") {
            Ok(DatabaseType::PostgreSQL)
        } else if connection_string.starts_with("mysql://") {
            Ok(DatabaseType::MySQL)
        } else if connection_string.ends_with(".db") || connection_string.starts_with("sqlite://") || connection_string.starts_with("file:") {
            Ok(DatabaseType::SQLite)
        } else if connection_string.starts_with("redis://") {
            Ok(DatabaseType::Redis)
        } else if connection_string.starts_with("mongodb://") {
            Ok(DatabaseType::MongoDB)
        } else if connection_string.starts_with("http://") || connection_string.starts_with("https://") {
            Ok(DatabaseType::Elasticsearch)
        } else {
            Err(Error::invalid_argument(&format!("无法识别的连接字符串格式: {}", connection_string)))
        }
    }
    
    /// 从连接字符串中提取用户名、密码、数据库名和表名
    fn parse_connection_string_parts(connection_string: &str) -> Result<(Option<String>, Option<String>, Option<String>, Option<String>)> {
        let url = Url::parse(connection_string)
            .map_err(|e| Error::invalid_argument(&format!("无效的连接字符串: {}", e)))?;
        
        let username = if !url.username().is_empty() {
            Some(url.username().to_string())
        } else {
            None
        };
        
        let password = url.password().map(|p| p.to_string());
        
        let database = url.path_segments()
            .and_then(|segments| segments.filter(|s| !s.is_empty()).next())
            .map(|s| s.to_string());
        
        // 表名通常不在连接字符串中，需要从查询参数或其他地方获取
        let table = url.query_pairs()
            .find(|(key, _)| key == "table" || key == "table_name")
            .map(|(_, value)| value.to_string());
        
        Ok((username, password, database, table))
    }
    
    /// 将 DataBatch 转换为 CoreTensorData 向量
    fn convert_data_batch_to_core_tensors(data_batch: &crate::data::DataBatch) -> Result<Vec<CoreTensorData>> {
        let mut tensors = Vec::new();
        
        // 如果 DataBatch 有 features，直接转换
        if let Some(ref features) = data_batch.features {
            for (idx, feature_row) in features.iter().enumerate() {
                let tensor = CoreTensorData {
                    id: format!("tensor_{}", idx),
                    shape: vec![feature_row.len()],
                    data: feature_row.clone(),
                    dtype: "float32".to_string(),
                    device: "cpu".to_string(),
                    requires_grad: false,
                    metadata: HashMap::new(),
                    created_at: chrono::Utc::now(),
                    updated_at: chrono::Utc::now(),
                };
                tensors.push(tensor);
            }
        } else if !data_batch.records.is_empty() {
            // 从 records 中提取数值特征
            for (idx, record) in data_batch.records.iter().enumerate() {
                let mut feature_values = Vec::new();
                for (_, value) in record.iter() {
                    // 尝试将值转换为 f32
                    if let Some(f32_val) = Self::convert_value_to_f32(value) {
                        feature_values.push(f32_val);
                    }
                }
                
                if !feature_values.is_empty() {
                    let tensor = CoreTensorData {
                        id: format!("tensor_{}", idx),
                        shape: vec![feature_values.len()],
                        data: feature_values,
                        dtype: "float32".to_string(),
                        device: "cpu".to_string(),
                        requires_grad: false,
                        metadata: HashMap::new(),
                        created_at: chrono::Utc::now(),
                        updated_at: chrono::Utc::now(),
                    };
                    tensors.push(tensor);
                }
            }
        }
        
        Ok(tensors)
    }
    
    /// 将值转换为 f32
    fn convert_value_to_f32(value: &crate::data::DataValue) -> Option<f32> {
        match value {
            crate::data::DataValue::Integer(i) => Some(*i as f32),
            crate::data::DataValue::Float(f) => Some(*f as f32),
            crate::data::DataValue::String(s) => s.parse::<f32>().ok(),
            crate::data::DataValue::Boolean(b) => Some(if *b { 1.0 } else { 0.0 }),
            _ => None,
        }
    }
    
    /// 将 DataSchema 转换为 CoreDataSchema
    fn convert_data_schema_to_core_schema(data_schema: &crate::data::DataSchema, schema_id: &str) -> Result<CoreDataSchema> {
        let fields: Vec<CoreSchemaField> = data_schema.fields.iter().map(|field| {
            CoreSchemaField {
                name: field.name.clone(),
                field_type: Self::convert_field_type(&field.field_type),
                nullable: !field.required,
                constraints: None,
                description: field.description.clone(),
            }
        }).collect();
        
        Ok(CoreDataSchema {
            id: schema_id.to_string(),
            name: data_schema.name.clone(),
            fields,
            metadata: data_schema.metadata.clone(),
            created_at: chrono::Utc::now(),
            updated_at: chrono::Utc::now(),
        })
    }
    
    /// 转换字段类型
    fn convert_field_type(field_type: &crate::data::schema::schema::FieldType) -> CoreFieldType {
        match field_type {
            crate::data::schema::schema::FieldType::Text => CoreFieldType::String,
            crate::data::schema::schema::FieldType::Numeric => CoreFieldType::Integer,
            crate::data::schema::schema::FieldType::Boolean => CoreFieldType::Boolean,
            crate::data::schema::schema::FieldType::DateTime => CoreFieldType::DateTime,
            crate::data::schema::schema::FieldType::Array(_) => CoreFieldType::Array,
            crate::data::schema::schema::FieldType::Object(_) => CoreFieldType::Object,
            _ => CoreFieldType::String,
        }
    }
    
    /// 从API端点加载数据 - 支持端到端管道
    /// 
    /// 生产级实现：真正调用HTTP API，解析响应，并将结果转换为 CoreDataBatch
    pub async fn load_from_api(&self, api_endpoint: &str) -> Result<(crate::core::types::CoreDataBatch, CoreDataSchema)> {
        let _dataset_id = format!("api_{}", uuid::Uuid::new_v4());
        
        info!("从API端点加载数据: {}", api_endpoint);
        
        // 创建HTTP客户端
        #[cfg(not(feature = "multimodal"))]
        return Err(Error::feature_not_enabled("multimodal"));
        
        #[cfg(feature = "multimodal")]
        {
            let client = reqwest::Client::builder()
                .timeout(std::time::Duration::from_secs(30))
                .build()
                .map_err(|e| Error::invalid_data(format!("无法创建HTTP客户端: {}", e)))?;
            
            // 发送HTTP请求
            let response = client
                .get(api_endpoint)
                .send()
                .await
                .map_err(|e| Error::invalid_data(format!("API请求失败: {}", e)))?;
            
            // 检查响应状态
            if !response.status().is_success() {
                return Err(Error::invalid_data(format!("API返回错误状态: {}", response.status())));
            }
            
            // 解析响应为JSON
            let json_value: serde_json::Value = response
                .json()
                .await
                .map_err(|e| Error::invalid_data(format!("无法解析API响应为JSON: {}", e)))?;
            
            // 将JSON响应转换为 CoreTensorData 向量
            let core_tensors = Self::convert_json_to_core_tensors(&json_value)?;
            
            // 从JSON响应推断schema
            let core_schema = Self::infer_schema_from_json(&json_value, &_dataset_id)?;
            
            // 创建 CoreDataBatch
            let core_data_batch = crate::core::types::CoreDataBatch {
                id: _dataset_id.clone(),
                data: core_tensors.clone(),
                labels: None,
                batch_size: core_tensors.len(),
                metadata: Some({
                    let mut metadata = HashMap::new();
                    metadata.insert("source_type".to_string(), "api".to_string());
                    metadata.insert("api_endpoint".to_string(), api_endpoint.to_string());
                    metadata.insert("response_size".to_string(), core_tensors.len().to_string());
                    metadata.insert("raw_response".to_string(), serde_json::to_string(&json_value).unwrap_or_default());
                    metadata
                }),
                created_at: chrono::Utc::now(),
                updated_at: chrono::Utc::now(),
            };
            
            Ok((core_data_batch, core_schema))
        }
    }
    
    /// 将JSON值转换为 CoreTensorData 向量
    fn convert_json_to_core_tensors(json_value: &serde_json::Value) -> Result<Vec<CoreTensorData>> {
        let mut tensors = Vec::new();
        
        match json_value {
            serde_json::Value::Array(arr) => {
                // 如果是数组，将每个元素转换为tensor
                for (idx, item) in arr.iter().enumerate() {
                    if let Ok(tensor) = Self::json_value_to_tensor(item, idx) {
                        tensors.push(tensor);
                    }
                }
            }
            serde_json::Value::Object(obj) => {
                // 如果是对象，尝试提取数值字段
                let mut values = Vec::new();
                for (_, value) in obj.iter() {
                    if let Some(f32_val) = Self::json_value_to_f32(value) {
                        values.push(f32_val);
                    }
                }
                if !values.is_empty() {
                    tensors.push(CoreTensorData {
                        id: "tensor_0".to_string(),
                        shape: vec![values.len()],
                        data: values,
                        dtype: "float32".to_string(),
                        device: "cpu".to_string(),
                        requires_grad: false,
                        metadata: HashMap::new(),
                        created_at: chrono::Utc::now(),
                        updated_at: chrono::Utc::now(),
                    });
                }
            }
            _ => {
                // 单个值
                if let Some(f32_val) = Self::json_value_to_f32(json_value) {
                    tensors.push(CoreTensorData {
                        id: "tensor_0".to_string(),
                        shape: vec![1],
                        data: vec![f32_val],
                        dtype: "float32".to_string(),
                        device: "cpu".to_string(),
                        requires_grad: false,
                        metadata: HashMap::new(),
                        created_at: chrono::Utc::now(),
                        updated_at: chrono::Utc::now(),
                    });
                }
            }
        }
        
        Ok(tensors)
    }
    
    /// 将JSON值转换为f32
    fn json_value_to_f32(value: &serde_json::Value) -> Option<f32> {
        match value {
            serde_json::Value::Number(n) => n.as_f64().map(|f| f as f32),
            serde_json::Value::String(s) => s.parse::<f32>().ok(),
            serde_json::Value::Bool(b) => Some(if *b { 1.0 } else { 0.0 }),
            _ => None,
        }
    }
    
    /// 将JSON值转换为tensor
    fn json_value_to_tensor(value: &serde_json::Value, idx: usize) -> Result<CoreTensorData> {
        let mut values = Vec::new();
        
        match value {
            serde_json::Value::Array(arr) => {
                for item in arr {
                    if let Some(f32_val) = Self::json_value_to_f32(item) {
                        values.push(f32_val);
                    }
                }
            }
            serde_json::Value::Object(obj) => {
                for (_, v) in obj.iter() {
                    if let Some(f32_val) = Self::json_value_to_f32(v) {
                        values.push(f32_val);
                    }
                }
            }
            _ => {
                if let Some(f32_val) = Self::json_value_to_f32(value) {
                    values.push(f32_val);
                }
            }
        }
        
        Ok(CoreTensorData {
            id: format!("tensor_{}", idx),
            shape: vec![values.len()],
            data: values,
            dtype: "float32".to_string(),
            device: "cpu".to_string(),
            requires_grad: false,
            metadata: HashMap::new(),
            created_at: chrono::Utc::now(),
            updated_at: chrono::Utc::now(),
        })
    }
    
    /// 从JSON值推断schema
    fn infer_schema_from_json(json_value: &serde_json::Value, schema_id: &str) -> Result<CoreDataSchema> {
        let mut fields = Vec::new();
        
        match json_value {
            serde_json::Value::Array(arr) if !arr.is_empty() => {
                // 从数组的第一个元素推断schema
                if let serde_json::Value::Object(obj) = &arr[0] {
                    for (key, value) in obj.iter() {
                        let field_type = match value {
                            serde_json::Value::Number(_) => CoreFieldType::Float,
                            serde_json::Value::String(_) => CoreFieldType::String,
                            serde_json::Value::Bool(_) => CoreFieldType::Boolean,
                            serde_json::Value::Array(_) => CoreFieldType::Array,
                            serde_json::Value::Object(_) => CoreFieldType::Object,
                            _ => CoreFieldType::String,
                        };
                        fields.push(CoreSchemaField {
                            name: key.clone(),
                            field_type,
                            nullable: true,
                            constraints: None,
                            description: None,
                        });
                    }
                }
            }
            serde_json::Value::Object(obj) => {
                // 从对象推断schema
                for (key, value) in obj.iter() {
                    let field_type = match value {
                        serde_json::Value::Number(_) => CoreFieldType::Float,
                        serde_json::Value::String(_) => CoreFieldType::String,
                        serde_json::Value::Bool(_) => CoreFieldType::Boolean,
                        serde_json::Value::Array(_) => CoreFieldType::Array,
                        serde_json::Value::Object(_) => CoreFieldType::Object,
                        _ => CoreFieldType::String,
                    };
                    fields.push(CoreSchemaField {
                        name: key.clone(),
                        field_type,
                        nullable: true,
                        constraints: None,
                        description: None,
                    });
                }
            }
            _ => {
                // 单个值，创建默认字段
                fields.push(CoreSchemaField {
                    name: "value".to_string(),
                    field_type: CoreFieldType::String,
                    nullable: true,
                    constraints: None,
                    description: None,
                });
            }
        }
        
        Ok(CoreDataSchema {
            id: schema_id.to_string(),
            name: "API Schema".to_string(),
            fields,
            metadata: HashMap::new(),
            created_at: chrono::Utc::now(),
            updated_at: chrono::Utc::now(),
        })
    }
    
    /// 从流数据源加载数据 - 支持端到端管道
    /// 
    /// 生产级实现：真正处理流数据源（WebSocket、Kafka等），收集数据并转换为 CoreDataBatch
    pub async fn load_from_stream(&self, stream_config: &str) -> Result<(crate::core::types::CoreDataBatch, CoreDataSchema)> {
        let dataset_id = format!("stream_{}", uuid::Uuid::new_v4());
        
        info!("从流数据源加载数据: {}", stream_config);
        
        // 解析流配置（JSON格式：{"type": "websocket", "url": "...", "duration": 10}）
        let config: serde_json::Value = serde_json::from_str(stream_config)
            .map_err(|e| Error::invalid_argument(&format!("无效的流配置: {}", e)))?;
        
        let stream_type = config.get("type")
            .and_then(|v| v.as_str())
            .unwrap_or("websocket");
        
        #[allow(unused_mut)]
        let mut collected_data = Vec::new();
        let max_duration = config.get("duration")
            .and_then(|v| v.as_u64())
            .unwrap_or(10) as u64;
        
        #[cfg(not(feature = "websocket"))]
        if stream_type == "websocket" {
            return Err(Error::feature_not_enabled("websocket"));
        }
        
        #[cfg(not(feature = "multimodal"))]
        if stream_type == "http_stream" || stream_type == "sse" {
            return Err(Error::feature_not_enabled("multimodal"));
        }
        
        match stream_type {
            "websocket" => {
                #[cfg(feature = "websocket")]
                {
                    // 处理WebSocket流
                    let url = config.get("url")
                        .and_then(|v| v.as_str())
                        .ok_or_else(|| Error::invalid_argument("WebSocket URL未指定"))?;
                    
                    // 连接到WebSocket并收集数据
                    let (ws_stream, _) = tokio_tungstenite::connect_async(url)
                        .await
                        .map_err(|e| Error::invalid_data(format!("WebSocket连接失败: {}", e)))?;
                    
                    let (_, mut read) = ws_stream.split();
                    let timeout = tokio::time::Duration::from_secs(max_duration);
                    let start_time = std::time::Instant::now();
                    
                    // 在超时时间内收集数据
                    while start_time.elapsed() < timeout {
                        match tokio::time::timeout(
                            tokio::time::Duration::from_millis(100),
                            read.next()
                        ).await {
                            Ok(Some(Ok(message))) => {
                                if let Ok(text) = message.to_text() {
                                    if let Ok(json_value) = serde_json::from_str::<serde_json::Value>(text) {
                                        collected_data.push(json_value);
                                    }
                                }
                            }
                            Ok(Some(Err(e))) => {
                                warn!("WebSocket消息接收错误: {}", e);
                                break;
                            }
                            Ok(None) => break,
                            Err(_) => continue, // 超时，继续等待
                        }
                    }
                }
            }
            "http_stream" | "sse" => {
                #[cfg(feature = "multimodal")]
                {
                    // 处理HTTP流或Server-Sent Events
                    let url = config.get("url")
                        .and_then(|v| v.as_str())
                        .ok_or_else(|| Error::invalid_argument("HTTP流URL未指定"))?;
                    
                    let client = reqwest::Client::builder()
                        .timeout(std::time::Duration::from_secs(max_duration))
                        .build()
                        .map_err(|e| Error::invalid_data(format!("无法创建HTTP客户端: {}", e)))?;
                    
                    let response = client
                        .get(url)
                        .send()
                        .await
                        .map_err(|e| Error::invalid_data(format!("HTTP流请求失败: {}", e)))?;
                    
                    // 检查响应状态
                    if !response.status().is_success() {
                        return Err(Error::invalid_data(format!("HTTP流返回错误状态: {}", response.status())));
                    }
                    
                    // 读取响应体
                    let text = response
                        .text()
                        .await
                        .map_err(|e| Error::invalid_data(format!("无法读取HTTP流响应: {}", e)))?;
                    
                    // 按行解析JSON数据
                    for line in text.lines() {
                        if let Ok(json_value) = serde_json::from_str::<serde_json::Value>(line) {
                            collected_data.push(json_value);
                        }
                    }
                }
            }
            _ => {
                return Err(Error::invalid_argument(&format!("不支持的流类型: {}", stream_type)));
            }
        }
        
        // 将收集的数据转换为 CoreTensorData
        let core_tensors: Vec<crate::core::types::CoreTensorData> = collected_data.iter()
            .enumerate()
            .filter_map(|(idx, item)| {
                Self::convert_json_to_core_tensors(item).ok()
                    .and_then(|mut tensors| {
                        if !tensors.is_empty() {
                            tensors[0].id = format!("tensor_{}", idx);
                            Some(tensors[0].clone())
                        } else {
                            None
                        }
                    })
            })
            .collect();
        
        // 从收集的数据推断schema
        let core_schema = if !collected_data.is_empty() {
            Self::infer_schema_from_json(&serde_json::Value::Array(collected_data.clone()), &format!("{}_schema", dataset_id))?
        } else {
            CoreDataSchema {
                id: format!("{}_schema", dataset_id),
                name: "Stream Schema".to_string(),
                fields: Vec::new(),
                metadata: HashMap::new(),
                created_at: chrono::Utc::now(),
                updated_at: chrono::Utc::now(),
            }
        };
        
        // 创建 CoreDataBatch
        let core_data_batch = crate::core::types::CoreDataBatch {
            id: dataset_id.clone(),
            data: core_tensors.clone(),
            labels: None,
            batch_size: core_tensors.len(),
            metadata: Some({
                let mut metadata = HashMap::new();
                metadata.insert("source_type".to_string(), "stream".to_string());
                metadata.insert("stream_config".to_string(), stream_config.to_string());
                metadata.insert("stream_type".to_string(), stream_type.to_string());
                metadata.insert("collected_count".to_string(), collected_data.len().to_string());
                metadata.insert("duration_seconds".to_string(), max_duration.to_string());
                metadata
            }),
            created_at: chrono::Utc::now(),
            updated_at: chrono::Utc::now(),
        };
        
        Ok((core_data_batch, core_schema))
    }
    
    /// 从内存数据加载 - 支持端到端管道
    /// 
    /// 生产级实现：从内存引用（可能是之前加载的数据集ID或内存地址）加载数据
    pub async fn load_from_memory(&self, memory_ref: &str) -> Result<(crate::core::types::CoreDataBatch, CoreDataSchema)> {
        let dataset_id = format!("memory_{}", uuid::Uuid::new_v4());
        
        info!("从内存引用加载数据: {}", memory_ref);
        
        // 尝试从缓存中获取数据
        let cache = self.cache.lock()
            .map_err(|_| Error::Lock("无法锁定数据缓存".to_string()))?;
        
        if let Some(dataset) = cache.get(memory_ref) {
            // 如果内存引用是数据集ID，从存储中加载数据
            if let Ok(Some(raw_data)) = self.storage.get_data(memory_ref).await {
                // 解析原始数据
                let json_value: serde_json::Value = serde_json::from_slice(&raw_data)
                    .map_err(|e| Error::invalid_data(format!("无法解析内存数据: {}", e)))?;
                
                // 转换为 CoreTensorData
                let core_tensors = Self::convert_json_to_core_tensors(&json_value)?;
                
                // 推断schema
                let core_schema = Self::infer_schema_from_json(&json_value, &format!("{}_schema", dataset_id))?;
                
                // 创建 CoreDataBatch
                let core_data_batch = crate::core::types::CoreDataBatch {
                    id: dataset_id.clone(),
                    data: core_tensors.clone(),
                    labels: None,
                    batch_size: core_tensors.len(),
                    metadata: Some({
                        let mut metadata = HashMap::new();
                        metadata.insert("source_type".to_string(), "memory".to_string());
                        metadata.insert("memory_ref".to_string(), memory_ref.to_string());
                        metadata.insert("original_dataset_id".to_string(), dataset.id.clone());
                        metadata.insert("tensor_count".to_string(), core_tensors.len().to_string());
                        metadata
                    }),
                    created_at: chrono::Utc::now(),
                    updated_at: chrono::Utc::now(),
                };
                
                return Ok((core_data_batch, core_schema));
            }
        }
        
        // 如果内存引用不是数据集ID，尝试作为JSON字符串解析
        if let Ok(json_value) = serde_json::from_str::<serde_json::Value>(memory_ref) {
            let core_tensors = Self::convert_json_to_core_tensors(&json_value)?;
            let core_schema = Self::infer_schema_from_json(&json_value, &format!("{}_schema", dataset_id))?;
            
            let core_data_batch = crate::core::types::CoreDataBatch {
                id: dataset_id.clone(),
                data: core_tensors.clone(),
                labels: None,
                batch_size: core_tensors.len(),
                metadata: Some({
                    let mut metadata = HashMap::new();
                    metadata.insert("source_type".to_string(), "memory".to_string());
                    metadata.insert("memory_ref".to_string(), memory_ref.to_string());
                    metadata.insert("parsed_from_json".to_string(), "true".to_string());
                    metadata
                }),
                created_at: chrono::Utc::now(),
                updated_at: chrono::Utc::now(),
            };
            
            return Ok((core_data_batch, core_schema));
        }
        
        // 如果都无法解析，返回错误
        Err(Error::invalid_data(format!("无法从内存引用加载数据: {}", memory_ref)))
    }
    
    /// 根据配置处理数据
    fn process_data_with_config(
        raw_data: &[u8],
        config: &DataProcessConfig,
        input_format: &str,
    ) -> Result<Vec<u8>> {
        match input_format {
            "json" => {
                // 解析JSON数据
                let json_value: serde_json::Value = serde_json::from_slice(raw_data)
                    .map_err(|e| Error::invalid_data(format!("无法解析JSON数据: {}", e)))?;
                
                // 根据配置进行数据转换
                let processed_json = Self::transform_json_data(&json_value, config)?;
                
                // 序列化回JSON
                serde_json::to_vec(&processed_json)
                    .map_err(|e| Error::invalid_data(format!("无法序列化处理后的JSON: {}", e)))
            }
            "csv" => {
                // 解析CSV数据
                let mut reader = csv::Reader::from_reader(raw_data);
                let mut records = Vec::new();
                
                for result in reader.deserialize::<HashMap<String, String>>() {
                    let record = result.map_err(|e| Error::invalid_data(format!("CSV解析错误: {}", e)))?;
                    records.push(record);
                }
                
                // 根据配置进行数据转换
                let processed_records = Self::transform_csv_data(&records, config)?;
                
                // 序列化回CSV
                let mut writer = csv::Writer::from_writer(vec![]);
                for record in processed_records {
                    writer.serialize(record)
                        .map_err(|e| Error::invalid_data(format!("CSV序列化错误: {}", e)))?;
                }
                
                writer.into_inner()
                    .map_err(|e| Error::invalid_data(format!("无法获取CSV写入器内容: {}", e)))
            }
            _ => {
                // 对于其他格式，进行基本的二进制处理
                let mut processed = raw_data.to_vec();
                
                // 应用配置中的转换规则
                let normalize = config.custom_config.get("normalize")
                    .map(|v| v == "true" || v == "1")
                    .unwrap_or(false);
                if normalize {
                    // 数据归一化处理
                    processed = Self::normalize_binary_data(&processed)?;
                }
                
                Ok(processed)
            }
        }
    }
    
    /// 转换JSON数据
    fn transform_json_data(
        json_value: &serde_json::Value,
        config: &DataProcessConfig,
    ) -> Result<serde_json::Value> {
        // 从custom_config中获取配置参数
        let normalize_method = config.custom_config.get("normalize_method")
            .map(|v| v.as_str())
            .unwrap_or("none");
        
        let normalize = normalize_method != "none" || config.custom_config.get("normalize")
            .map(|v| v == "true" || v == "1")
            .unwrap_or(false);
        
        let field_filter: Vec<&str> = config.custom_config.get("field_filter")
            .map(|v| v.split(',').map(|s| s.trim()).filter(|s| !s.is_empty()).collect())
            .unwrap_or_else(Vec::new);
        
        // 如果启用归一化且是数组，需要先计算统计信息
        if normalize && matches!(json_value, serde_json::Value::Array(_)) {
            return Self::transform_json_data_with_normalization(json_value, config, &field_filter, normalize_method);
        }
        
        match json_value {
            serde_json::Value::Array(arr) => {
                // 边界检查：空数组
                if arr.is_empty() {
                    return Ok(serde_json::Value::Array(Vec::new()));
                }
                
                let processed: Result<Vec<_>> = arr.iter()
                    .map(|item| Self::transform_json_data(item, config))
                    .collect();
                Ok(serde_json::Value::Array(processed?))
            }
            serde_json::Value::Object(obj) => {
                // 边界检查：空对象
                if obj.is_empty() {
                    return Ok(serde_json::Value::Object(serde_json::Map::new()));
                }
                
                let mut processed = serde_json::Map::new();
                for (key, value) in obj.iter() {
                    // 根据配置过滤字段
                    if field_filter.is_empty() || field_filter.contains(&key.as_str()) {
                        let transformed = Self::transform_json_data(value, config)?;
                        processed.insert(key.clone(), transformed);
                    }
                }
                Ok(serde_json::Value::Object(processed))
            }
            serde_json::Value::Number(n) => {
                if normalize {
                    // 生产级实现：真正的归一化需要统计信息，这里使用简单的缩放作为fallback
                    // 注意：真正的归一化应该在数组级别进行
                    if let Some(f64_val) = n.as_f64() {
                        // 检查数值有效性
                        if !f64_val.is_finite() {
                            return Err(Error::invalid_data(format!("无效的数值: {}", f64_val)));
                        }
                        
                        // 如果没有指定方法，使用简单的缩放（向后兼容）
                        if normalize_method == "none" || normalize_method == "scale" {
                        Ok(serde_json::Value::Number(
                            serde_json::Number::from_f64(f64_val / 1000.0)
                                    .ok_or_else(|| Error::invalid_data(format!("无法转换数值: {}", f64_val)))?
                        ))
                    } else {
                            // 其他归一化方法需要统计信息，单个数值无法归一化
                            Ok(json_value.clone())
                        }
                    } else {
                        Ok(json_value.clone())
                    }
                } else {
                    Ok(json_value.clone())
                }
            }
            _ => Ok(json_value.clone()),
        }
    }
    
    /// 带归一化的JSON数据转换（生产级实现）
    fn transform_json_data_with_normalization(
        json_value: &serde_json::Value,
        config: &DataProcessConfig,
        field_filter: &[&str],
        normalize_method: &str,
    ) -> Result<serde_json::Value> {
        if let serde_json::Value::Array(arr) = json_value {
            if arr.is_empty() {
                return Ok(serde_json::Value::Array(Vec::new()));
            }
            
            // 提取所有数值字段用于计算统计信息
            let mut numeric_fields: std::collections::HashMap<String, Vec<f64>> = std::collections::HashMap::new();
            
            // 第一遍：收集所有数值字段
            for item in arr.iter() {
                if let serde_json::Value::Object(obj) = item {
                    for (key, value) in obj.iter() {
                        if field_filter.is_empty() || field_filter.contains(&key.as_str()) {
                            if let serde_json::Value::Number(n) = value {
                                if let Some(f64_val) = n.as_f64() {
                                    if f64_val.is_finite() {
                                        numeric_fields.entry(key.clone())
                                            .or_insert_with(Vec::new)
                                            .push(f64_val);
                                    }
                                }
                            }
                        }
                    }
                }
            }
            
            // 计算统计信息（min, max, mean, std）
            let mut stats: std::collections::HashMap<String, (f64, f64, f64, f64)> = std::collections::HashMap::new();
            for (field, values) in &numeric_fields {
                if values.is_empty() {
                    continue;
                }
                
                let min = values.iter().fold(f64::INFINITY, |a, &b| a.min(b));
                let max = values.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
                let sum: f64 = values.iter().sum();
                let mean = sum / values.len() as f64;
                let variance: f64 = values.iter()
                    .map(|&x| (x - mean).powi(2))
                    .sum::<f64>() / values.len() as f64;
                let std = variance.sqrt();
                
                stats.insert(field.clone(), (min, max, mean, std));
            }
            
            // 第二遍：应用归一化
            let processed: Result<Vec<_>> = arr.iter()
                .map(|item| {
                    if let serde_json::Value::Object(obj) = item {
                        let mut processed_obj = serde_json::Map::new();
                        for (key, value) in obj.iter() {
                            if field_filter.is_empty() || field_filter.contains(&key.as_str()) {
                                if let serde_json::Value::Number(n) = value {
                                    if let Some(f64_val) = n.as_f64() {
                                        if f64_val.is_finite() {
                                            if let Some((min, max, mean, std)) = stats.get(key) {
                                                let normalized = match normalize_method {
                                                    "minmax" => {
                                                        let range = max - min;
                                                        if range > 1e-10 {
                                                            (f64_val - min) / range
                                                        } else {
                                                            f64_val // 避免除零
                                                        }
                                                    }
                                                    "zscore" | "standardize" => {
                                                        if *std > 1e-10 {
                                                            (f64_val - mean) / std
                                                        } else {
                                                            0.0 // 标准差为0，归一化为0
                                                        }
                                                    }
                                                    _ => {
                                                        // 默认使用minmax
                                                        let range = max - min;
                                                        if range > 1e-10 {
                                                            (f64_val - min) / range
                                                        } else {
                                                            f64_val
                                                        }
                                                    }
                                                };
                                                
                                                processed_obj.insert(
                                                    key.clone(),
                                                    serde_json::Value::Number(
                                                        serde_json::Number::from_f64(normalized)
                                                            .ok_or_else(|| Error::invalid_data(format!("归一化后的数值无效: {}", normalized)))?
                                                    )
                                                );
                                            } else {
                                                processed_obj.insert(key.clone(), value.clone());
                                            }
                                        } else {
                                            return Err(Error::invalid_data(format!("字段 {} 包含无效数值: {}", key, f64_val)));
                                        }
                                    } else {
                                        processed_obj.insert(key.clone(), value.clone());
                                    }
                                } else {
                                    processed_obj.insert(key.clone(), value.clone());
                                }
                            } else {
                                processed_obj.insert(key.clone(), value.clone());
                            }
                        }
                        Ok(serde_json::Value::Object(processed_obj))
                    } else {
                        Self::transform_json_data(item, config)
                    }
                })
                .collect();
            
            Ok(serde_json::Value::Array(processed?))
        } else {
            Self::transform_json_data(json_value, config)
        }
    }
    
    /// 转换CSV数据（生产级实现：支持真正的归一化）
    fn transform_csv_data(
        records: &[HashMap<String, String>],
        config: &DataProcessConfig,
    ) -> Result<Vec<HashMap<String, String>>> {
        // 边界检查：空记录
        if records.is_empty() {
            return Ok(Vec::new());
        }
        
        let normalize_method = config.custom_config.get("normalize_method")
            .map(|v| v.as_str())
            .unwrap_or("none");
        
        let normalize = normalize_method != "none" || config.custom_config.get("normalize")
            .map(|v| v == "true" || v == "1")
            .unwrap_or(false);
        
        let field_filter: Vec<&str> = config.custom_config.get("field_filter")
            .map(|v| v.split(',').map(|s| s.trim()).filter(|s| !s.is_empty()).collect())
            .unwrap_or_else(Vec::new);
        
        // 如果启用归一化，先计算统计信息
        let stats = if normalize {
            Self::compute_csv_statistics(records, &field_filter)?
        } else {
            None
        };
        
        let mut processed = Vec::with_capacity(records.len());
        
        for (idx, record) in records.iter().enumerate() {
            // 边界检查：空记录
            if record.is_empty() {
                processed.push(HashMap::new());
                continue;
            }
            
            let mut processed_record = HashMap::with_capacity(record.len());
            
            for (key, value) in record.iter() {
                // 根据配置过滤字段
                if field_filter.is_empty() || field_filter.contains(&key.as_str()) {
                    let processed_value = if normalize {
                        // 尝试归一化数值
                        match value.parse::<f64>() {
                            Ok(num) => {
                                if !num.is_finite() {
                                    return Err(Error::invalid_data(format!(
                                        "记录 {} 的字段 {} 包含无效数值: {}",
                                        idx, key, num
                                    )));
                                }
                                
                                if let Some(ref field_stats) = stats {
                                    if let Some((min, max, mean, std)) = field_stats.get(key) {
                                        let normalized = match normalize_method {
                                            "minmax" => {
                                                let range = max - min;
                                                if range > 1e-10 {
                                                    (num - min) / range
                        } else {
                                                    num // 避免除零
                                                }
                                            }
                                            "zscore" | "standardize" => {
                                                if *std > 1e-10 {
                                                    (num - mean) / std
                                                } else {
                                                    0.0 // 标准差为0，归一化为0
                                                }
                                            }
                                            _ => {
                                                // 默认使用简单缩放（向后兼容）
                                                num / 1000.0
                                            }
                                        };
                                        
                                        // 格式化归一化后的数值，保留合理精度
                                        format!("{:.6}", normalized)
                                    } else {
                                        // 该字段没有统计信息，使用简单缩放
                                        format!("{:.6}", num / 1000.0)
                                    }
                                } else {
                                    // 没有统计信息，使用简单缩放（向后兼容）
                                    format!("{:.6}", num / 1000.0)
                                }
                            }
                            Err(_) => {
                                // 不是数值，保持原值
                            value.clone()
                            }
                        }
                    } else {
                        value.clone()
                    };
                    
                    processed_record.insert(key.clone(), processed_value);
                }
            }
            
            processed.push(processed_record);
        }
        
        Ok(processed)
    }
    
    /// 计算CSV数据的统计信息（用于归一化）
    fn compute_csv_statistics(
        records: &[HashMap<String, String>],
        field_filter: &[&str],
    ) -> Result<Option<std::collections::HashMap<String, (f64, f64, f64, f64)>>> {
        if records.is_empty() {
            return Ok(None);
        }
        
        let mut numeric_fields: std::collections::HashMap<String, Vec<f64>> = std::collections::HashMap::new();
        
        // 收集所有数值字段的值
        for (idx, record) in records.iter().enumerate() {
            for (key, value) in record.iter() {
                if field_filter.is_empty() || field_filter.contains(&key.as_str()) {
                    match value.parse::<f64>() {
                        Ok(num) => {
                            if num.is_finite() {
                                numeric_fields.entry(key.clone())
                                    .or_insert_with(Vec::new)
                                    .push(num);
                            } else {
                                return Err(Error::invalid_data(format!(
                                    "记录 {} 的字段 {} 包含无效数值: {}",
                                    idx, key, num
                                )));
                            }
                        }
                        Err(_) => {
                            // 不是数值，跳过
                        }
                    }
                }
            }
        }
        
        if numeric_fields.is_empty() {
            return Ok(None);
        }
        
        // 计算统计信息
        let mut stats = std::collections::HashMap::new();
        for (field, values) in numeric_fields {
            if values.is_empty() {
                continue;
            }
            
            let min = values.iter().fold(f64::INFINITY, |a, &b| a.min(b));
            let max = values.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
            let sum: f64 = values.iter().sum();
            let mean = sum / values.len() as f64;
            let variance: f64 = values.iter()
                .map(|&x| (x - mean).powi(2))
                .sum::<f64>() / values.len() as f64;
            let std = variance.sqrt();
            
            stats.insert(field, (min, max, mean, std));
        }
        
        Ok(Some(stats))
    }
    
    /// 归一化二进制数据（生产级实现：真正的MinMax归一化）
    fn normalize_binary_data(data: &[u8]) -> Result<Vec<u8>> {
        // 边界检查：空数据
        if data.is_empty() {
            return Ok(Vec::new());
        }
        
        // 计算最小值和最大值
        let min = *data.iter().min().ok_or_else(|| Error::invalid_data("数据为空，无法计算最小值"))?;
        let max = *data.iter().max().ok_or_else(|| Error::invalid_data("数据为空，无法计算最大值"))?;
        
        // 如果所有值相同，返回原数据或归一化到中间值
        if min == max {
            return Ok(vec![128u8; data.len()]); // 归一化到中间值
        }
        
        // MinMax归一化：将值映射到[0, 255]范围
        let range = (max - min) as f32;
        let normalized: Vec<u8> = data.iter()
            .map(|&b| {
                let normalized_f32 = ((b - min) as f32 / range) * 255.0;
                // 确保在[0, 255]范围内
                normalized_f32.max(0.0).min(255.0) as u8
            })
            .collect();
        
        Ok(normalized)
    }
    
    /// 从数据中提取标签
    fn extract_labels_from_data(
        data: &[u8],
        config: &DataProcessConfig,
        metadata: &HashMap<String, String>,
    ) -> Result<Option<Vec<u8>>> {
        // 从配置或元数据中获取标签字段名
        let label_field = config.custom_config.get("target_field")
            .or_else(|| metadata.get("target_field"))
            .or_else(|| metadata.get("label_field"));
        
        if let Some(field_name) = label_field {
            // 尝试从JSON数据中提取标签
            if let Ok(json_value) = serde_json::from_slice::<serde_json::Value>(data) {
                if let Some(labels) = Self::extract_labels_from_json(&json_value, field_name) {
                    return Ok(Some(serde_json::to_vec(&labels)
                        .map_err(|e| Error::invalid_data(format!("无法序列化标签: {}", e)))?));
                }
            }
        }
        
        Ok(None)
    }
    
    /// 从JSON数据中提取标签
    fn extract_labels_from_json(
        json_value: &serde_json::Value,
        label_field: &str,
    ) -> Option<serde_json::Value> {
        match json_value {
            serde_json::Value::Array(arr) => {
                let labels: Vec<_> = arr.iter()
                    .filter_map(|item| {
                        if let serde_json::Value::Object(obj) = item {
                            obj.get(label_field).cloned()
                        } else {
                            None
                        }
                    })
                    .collect();
                
                if !labels.is_empty() {
                    Some(serde_json::Value::Array(labels))
                } else {
                    None
                }
            }
            serde_json::Value::Object(obj) => {
                obj.get(label_field).cloned()
            }
            _ => None,
        }
    }
}

/// 存储数据加载器适配器，用于将Storage接口适配到DataLoader trait
struct StorageDataLoader {
    storage: Arc<Storage>,
}

impl StorageDataLoader {
    fn new(storage: Arc<Storage>) -> Self {
        Self { storage }
    }
}

#[async_trait::async_trait]
impl crate::data::loader::DataLoader for StorageDataLoader {
    async fn load(&self, source: &crate::data::loader::DataSource, format: &crate::data::loader::types::DataFormat) -> Result<crate::data::DataBatch> {
        // 从数据源中提取路径
        let path = match source {
            crate::data::loader::DataSource::File(path) => path,
            crate::data::loader::DataSource::Database(_) => return Err(Error::invalid_argument("不支持数据库数据源")),
            crate::data::loader::DataSource::Stream(_) => return Err(Error::invalid_argument("不支持流数据源")),
            crate::data::loader::DataSource::Memory(_) => return Err(Error::invalid_argument("不支持内存数据源")),
        };
        
        // 从路径中提取数据集ID
        let dataset_id = path.strip_prefix("dataset/").unwrap_or(path);
        
        let data = self.storage.get_data(dataset_id).await?
            .ok_or_else(|| Error::invalid_data(format!("找不到数据集: {}", dataset_id)))?;
        
        // 使用 DataBatch::new 创建基础批次，并填充数据
        let mut batch = crate::data::DataBatch::new(dataset_id, 0, data.len());
        batch.data = Some(data);
        
        Ok(batch)
    }

    async fn get_schema(&self, _source: &crate::data::loader::DataSource, _format: &crate::data::loader::types::DataFormat) -> Result<crate::data::DataSchema> {
        // 返回基础schema
        Ok(crate::data::DataSchema::new("storage_dataset", "1.0"))
    }

    fn name(&self) -> &'static str {
        "StorageDataLoader"
    }

    async fn load_batch(&self, source: &crate::data::loader::DataSource, format: &crate::data::loader::types::DataFormat, batch_size: usize, offset: usize) -> Result<crate::data::DataBatch> {
        let mut batch = self.load(source, format).await?;
        
        if let Some(data) = &mut batch.data {
            let data_len = data.len();
            let end = (offset + batch_size).min(data_len);
            if offset >= data_len {
                batch.data = Some(Vec::new());
            } else {
                let sliced = data[offset..end].to_vec();
                batch.data = Some(sliced);
            }
            batch.batch_size = batch.data.as_ref().map(|d| d.len()).unwrap_or(0);
        } else {
            batch.data = Some(Vec::new());
            batch.batch_size = 0;
        }
        
        batch.batch_index = offset / batch_size;
        
        Ok(batch)
    }

    fn supports_format(&self, format: &crate::data::loader::types::DataFormat) -> bool {
        // 支持所有格式
        matches!(format, 
            crate::data::loader::types::DataFormat::Json { .. } | 
            crate::data::loader::types::DataFormat::Csv { .. } |
            crate::data::loader::types::DataFormat::CustomBinary(_)
        )
    }

    fn config(&self) -> &crate::data::loader::LoaderConfig {
        // 返回默认配置的静态引用
        static DEFAULT_CONFIG: std::sync::OnceLock<crate::data::loader::LoaderConfig> = std::sync::OnceLock::new();
        DEFAULT_CONFIG.get_or_init(|| crate::data::loader::LoaderConfig::default())
    }

    fn set_config(&mut self, _config: crate::data::loader::LoaderConfig) {
        // 此实现不支持配置设置
    }

    async fn get_size(&self, path: &str) -> Result<usize> {
        let dataset_id = path.strip_prefix("dataset/").unwrap_or(path);
        let data = self.storage.get_data(dataset_id).await?
            .ok_or_else(|| Error::invalid_data(format!("找不到数据集: {}", dataset_id)))?;
        Ok(data.len())
    }

    async fn get_batch_at(&self, path: &str, index: usize, batch_size: usize) -> Result<crate::data::DataBatch> {
        let source = crate::data::loader::DataSource::File(path.to_string());
        let format = crate::data::loader::types::DataFormat::CustomBinary("binary".to_string());
        self.load_batch(&source, &format, batch_size, index * batch_size).await
    }
} 