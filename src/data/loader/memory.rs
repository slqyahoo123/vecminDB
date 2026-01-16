/// Memory Data Loader
/// 内存数据加载器，用于从内存中加载数据

use std::collections::HashMap;
use std::sync::Arc;
use async_trait::async_trait;
// serde derives not used in this module
use crate::error::{Result, Error};
use crate::data::DataBatch;
use super::{DataLoader, LoaderConfig};
use crate::data::loader::types::DataSource;
use crate::data::schema::DataSchema;
use futures::stream::{self, StreamExt, FuturesUnordered};
use tokio::task;
use tokio::sync::{RwLock, Semaphore};
use std::time::{Duration, Instant};
use log::{debug, info, warn, error};

/// 异步批处理配置
#[derive(Debug, Clone)]
pub struct AsyncBatchConfig {
    /// 最大并发数
    pub max_concurrent: usize,
    /// 批处理大小
    pub batch_size: usize,
    /// 超时时间
    pub timeout: Duration,
    /// 是否启用缓存
    pub enable_cache: bool,
    /// 缓存过期时间
    pub cache_ttl: Duration,
}

impl Default for AsyncBatchConfig {
    fn default() -> Self {
        Self {
            max_concurrent: num_cpus::get() * 2,
            batch_size: 100,
            timeout: Duration::from_secs(30),
            enable_cache: true,
            cache_ttl: Duration::from_secs(300), // 5分钟
        }
    }
}

/// 缓存项
#[derive(Debug, Clone)]
pub struct CacheItem {
    /// 缓存的数据批次
    pub batch: DataBatch,
    /// 创建时间
    pub created_at: Instant,
    /// 访问次数
    pub access_count: usize,
}

/// 内存数据加载器
#[derive(Debug, Clone)]
pub struct MemoryDataLoader {
    /// 内存中的数据存储
    data_store: Arc<RwLock<HashMap<String, Vec<u8>>>>,
    /// 加载器配置
    config: LoaderConfig,
    /// 异步批处理配置
    batch_config: AsyncBatchConfig,
    /// 并发控制信号量
    semaphore: Arc<Semaphore>,
    /// 缓存存储
    cache: Arc<RwLock<HashMap<String, CacheItem>>>,
}

impl MemoryDataLoader {
    /// 创建新的内存数据加载器
    pub fn new() -> Self {
        let batch_config = AsyncBatchConfig::default();
        Self {
            data_store: Arc::new(RwLock::new(HashMap::new())),
            config: LoaderConfig::default(),
            semaphore: Arc::new(Semaphore::new(batch_config.max_concurrent)),
            cache: Arc::new(RwLock::new(HashMap::new())),
            batch_config,
        }
    }

    /// 使用配置创建内存数据加载器
    pub fn with_config(config: LoaderConfig) -> Self {
        let batch_config = AsyncBatchConfig::default();
        Self {
            data_store: Arc::new(RwLock::new(HashMap::new())),
            config,
            semaphore: Arc::new(Semaphore::new(batch_config.max_concurrent)),
            cache: Arc::new(RwLock::new(HashMap::new())),
            batch_config,
        }
    }

    /// 使用批处理配置创建加载器
    pub fn with_batch_config(config: LoaderConfig, batch_config: AsyncBatchConfig) -> Self {
        Self {
            data_store: Arc::new(RwLock::new(HashMap::new())),
            config,
            semaphore: Arc::new(Semaphore::new(batch_config.max_concurrent)),
            cache: Arc::new(RwLock::new(HashMap::new())),
            batch_config,
        }
    }

    /// 向内存中添加数据
    pub async fn add_data(&self, key: String, data: Vec<u8>) -> Result<()> {
        let mut store = self.data_store.write().await;
        store.insert(key.clone(), data);
        
        // 清除相关缓存
        if self.batch_config.enable_cache {
            let mut cache = self.cache.write().await;
            cache.remove(&key);
        }
        
        debug!("Added data to memory store: {}", key);
        Ok(())
    }

    /// 从内存中获取数据
    pub async fn get_data(&self, key: &str) -> Result<Option<Vec<u8>>> {
        let store = self.data_store.read().await;
        Ok(store.get(key).cloned())
    }

    /// 删除内存中的数据
    pub async fn remove_data(&self, key: &str) -> Result<Option<Vec<u8>>> {
        let mut store = self.data_store.write().await;
        let result = store.remove(key);
        
        // 清除相关缓存
        if self.batch_config.enable_cache {
            let mut cache = self.cache.write().await;
            cache.remove(key);
        }
        
        Ok(result)
    }

    /// 清空所有数据
    pub async fn clear(&self) -> Result<()> {
        let mut store = self.data_store.write().await;
        store.clear();
        
        // 清空缓存
        if self.batch_config.enable_cache {
            let mut cache = self.cache.write().await;
            cache.clear();
        }
        
        Ok(())
    }

    /// 获取所有键
    pub async fn keys(&self) -> Result<Vec<String>> {
        let store = self.data_store.read().await;
        Ok(store.keys().cloned().collect())
    }

    /// 获取数据数量
    pub async fn len(&self) -> Result<usize> {
        let store = self.data_store.read().await;
        Ok(store.len())
    }

    /// 检查是否为空
    pub async fn is_empty(&self) -> Result<bool> {
        let store = self.data_store.read().await;
        Ok(store.is_empty())
    }

    /// 从JSON字符串加载数据
    pub async fn load_from_json(&self, key: String, json_data: &str) -> Result<()> {
        let data = json_data.as_bytes().to_vec();
        self.add_data(key, data).await
    }

    /// 从CSV字符串加载数据
    pub async fn load_from_csv(&self, key: String, csv_data: &str) -> Result<()> {
        let data = csv_data.as_bytes().to_vec();
        self.add_data(key, data).await
    }

    /// 从二进制数据加载
    pub async fn load_from_bytes(&self, key: String, bytes: Vec<u8>) -> Result<()> {
        self.add_data(key, bytes).await
    }

    /// 异步批量加载数据（真正的并发实现）
    pub async fn load_batch_concurrent(&self, paths: &[String]) -> Result<Vec<DataBatch>> {
        let start_time = Instant::now();
        info!("开始并发批量加载 {} 个数据项", paths.len());

        // 检查缓存
        let mut cached_results = HashMap::new();
        let mut uncached_paths = Vec::new();

        if self.batch_config.enable_cache {
            let cache = self.cache.read().await;
            for path in paths {
                if let Some(cache_item) = cache.get(path) {
                    // 检查缓存是否过期
                    if cache_item.created_at.elapsed() < self.batch_config.cache_ttl {
                        cached_results.insert(path.clone(), cache_item.batch.clone());
                        debug!("Cache hit for path: {}", path);
                    } else {
                        uncached_paths.push(path.clone());
                        debug!("Cache expired for path: {}", path);
                    }
                } else {
                    uncached_paths.push(path.clone());
                }
            }
        } else {
            uncached_paths = paths.to_vec();
        }

        info!("缓存命中: {}, 需要加载: {}", cached_results.len(), uncached_paths.len());

        // 并发处理未缓存的路径
        let mut results = Vec::new();
        if !uncached_paths.is_empty() {
            // 将路径分块处理
            let chunks: Vec<Vec<String>> = uncached_paths
                .chunks(self.batch_config.batch_size)
                .map(|chunk| chunk.to_vec())
                .collect();

            // 并发处理每个块
            let mut futures = FuturesUnordered::new();
            
            for chunk in chunks {
                let loader = self.clone();
                let future = task::spawn(async move {
                    loader.process_chunk_concurrent(chunk).await
                });
                futures.push(future);
            }

            // 收集所有结果
            while let Some(chunk_result) = futures.next().await {
                match chunk_result {
                    Ok(Ok(chunk_batches)) => {
                        results.extend(chunk_batches);
                    }
                    Ok(Err(e)) => {
                        error!("Chunk processing failed: {}", e);
                        return Err(e);
                    }
                    Err(e) => {
                        error!("Task join failed: {}", e);
                        return Err(Error::Internal(format!("Task execution failed: {}", e)));
                    }
                }
            }

            // 更新缓存
            if self.batch_config.enable_cache {
                let mut cache = self.cache.write().await;
                for batch in &results {
                    if let Some(id) = &batch.id {
                        let cache_item = CacheItem {
                            batch: batch.clone(),
                            created_at: Instant::now(),
                            access_count: 1,
                        };
                        cache.insert(id.clone(), cache_item);
                    }
                }
            }
        }

        // 合并缓存结果和新加载的结果
        let mut final_results = Vec::new();
        for path in paths {
            if let Some(cached_batch) = cached_results.get(path) {
                final_results.push(cached_batch.clone());
                
                // 更新访问计数
                if self.batch_config.enable_cache {
                    let mut cache = self.cache.write().await;
                    if let Some(cache_item) = cache.get_mut(path) {
                        cache_item.access_count += 1;
                    }
                }
            } else {
                // 从新加载的结果中查找
                if let Some(batch) = results.iter().find(|b| b.id.as_ref() == Some(&path.to_string())) {
                    final_results.push(batch.clone());
                }
            }
        }

        let duration = start_time.elapsed();
        info!("并发批量加载完成，耗时: {:?}ms，成功加载: {} 个", 
              duration.as_millis(), final_results.len());

        Ok(final_results)
    }

    /// 处理单个块的并发加载
    async fn process_chunk_concurrent(&self, paths: Vec<String>) -> Result<Vec<DataBatch>> {
        let mut futures = FuturesUnordered::new();

        for path in paths {
            let loader = self.clone();
            let future = async move {
                // 获取信号量许可
                let _permit = loader.semaphore.acquire().await
                    .map_err(|e| Error::Internal(format!("Failed to acquire semaphore: {}", e)))?;

                // 加载单个数据项
                let result = tokio::time::timeout(
                    loader.batch_config.timeout,
                    loader.load(&DataSource::Memory(path.as_bytes().to_vec()), &crate::data::loader::types::DataFormat::json())
                ).await;

                match result {
                    Ok(Ok(batch)) => Ok(batch),
                    Ok(Err(e)) => {
                        warn!("Failed to load path {}: {}", path, e);
                        Err(e)
                    }
                    Err(_) => {
                        warn!("Timeout loading path: {}", path);
                        Err(Error::Internal(format!("Timeout loading path: {}", path)))
                    }
                }
            };
            futures.push(future);
        }

        let mut results = Vec::new();
        while let Some(result) = futures.next().await {
            match result {
                Ok(batch) => results.push(batch),
                Err(e) => {
                    error!("Individual load failed: {}", e);
                    // 继续处理其他项目，不中断整个批次
                }
            }
        }

        Ok(results)
    }

    /// 流式异步加载数据
    pub async fn load_stream(&self, paths: Vec<String>) -> impl futures::Stream<Item = Result<DataBatch>> {
        let loader = self.clone();
        stream::iter(paths.into_iter().map(move |path| {
            let loader = loader.clone();
            async move {
                let _permit = loader.semaphore.acquire().await
                    .map_err(|e| Error::Internal(format!("Failed to acquire semaphore: {}", e)))?;
                loader.load(&DataSource::Memory(path.as_bytes().to_vec()), &crate::data::loader::types::DataFormat::json()).await
            }
        }))
        .buffer_unordered(self.batch_config.max_concurrent)
    }

    /// 清理过期缓存
    pub async fn cleanup_cache(&self) -> Result<usize> {
        if !self.batch_config.enable_cache {
            return Ok(0);
        }

        let mut cache = self.cache.write().await;
        let initial_size = cache.len();
        
        cache.retain(|_, item| {
            item.created_at.elapsed() < self.batch_config.cache_ttl
        });

        let cleaned = initial_size - cache.len();
        if cleaned > 0 {
            info!("清理了 {} 个过期缓存项", cleaned);
        }
        
        Ok(cleaned)
    }

    /// 获取缓存统计信息
    pub async fn get_cache_stats(&self) -> Result<HashMap<String, usize>> {
        let mut stats = HashMap::new();
        
        if self.batch_config.enable_cache {
            let cache = self.cache.read().await;
            stats.insert("total_items".to_string(), cache.len());
            stats.insert("total_access_count".to_string(), 
                        cache.values().map(|item| item.access_count).sum());
            
            let expired_count = cache.values()
                .filter(|item| item.created_at.elapsed() >= self.batch_config.cache_ttl)
                .count();
            stats.insert("expired_items".to_string(), expired_count);
        } else {
            stats.insert("total_items".to_string(), 0);
            stats.insert("total_access_count".to_string(), 0);
            stats.insert("expired_items".to_string(), 0);
        }
        
        Ok(stats)
    }
}

impl Default for MemoryDataLoader {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl DataLoader for MemoryDataLoader {
    async fn load(&self, source: &DataSource, format: &crate::data::loader::types::DataFormat) -> Result<DataBatch> {
        // 从DataSource中提取路径
        let path = match source {
            DataSource::File(path) => path.clone(),
            DataSource::Stream(url) => url.clone(),
            DataSource::Memory(data) => {
                // 对于内存数据源，将字节数据转换为字符串并用作键
                String::from_utf8_lossy(data).to_string()
            },
            DataSource::Database(_) => return Err(Error::invalid_input("内存加载器不支持数据库数据源")),
            DataSource::Custom(name, _) => return Err(Error::invalid_input(&format!("内存加载器不支持自定义数据源: {}", name))),
        };

        // 从内存中获取数据
        let data = self.get_data(&path).await?
            .ok_or_else(|| Error::not_found(format!("内存中未找到数据: {}", path)))?;

        // 解析数据
        let features = match format {
            crate::data::loader::types::DataFormat::Json { .. } => self.parse_json_data(&data)?,
            crate::data::loader::types::DataFormat::Csv { .. } => self.parse_csv_data(&data)?,
            _ => return Err(Error::unsupported_file_type(format!("不支持的格式: {:?}", format))),
        };

        // 创建数据批次 - 使用正确的构造函数
        let mut batch = DataBatch::new("memory_data", 0, features.len());
        
        // 设置批次ID
        batch.id = Some(path.to_string());
        
        // 将features转换为records格式
        for (i, feature_vec) in features.iter().enumerate() {
            let mut record = HashMap::new();
            for (j, value) in feature_vec.iter().enumerate() {
                record.insert(format!("feature_{}", j), crate::data::DataValue::Float(*value as f64));
            }
            batch.records.push(record);
        }
        
        // 添加元数据
        batch.add_metadata("source", "memory");
        batch.add_metadata("path", &path);
        batch.add_metadata("format", &format!("{:?}", format));
        batch.add_metadata("size", &data.len().to_string());
        batch.add_metadata("loaded_at", &chrono::Utc::now().to_rfc3339());

        Ok(batch)
    }

    async fn get_schema(&self, source: &DataSource, format: &crate::data::loader::types::DataFormat) -> Result<DataSchema> {
        // 从DataSource中提取路径
        let path = match source {
            DataSource::File(path) => path,
            DataSource::Memory(_key) => {
                // 内存数据源使用默认模式
                return MemoryDataLoader::create_default_schema();
            },
            _ => return Err(Error::invalid_input("不支持的数据源类型")),
        };

        // 从内存中获取数据
        let data = self.get_data(path).await?
            .ok_or_else(|| Error::not_found(format!("内存中未找到数据: {}", path)))?;

        // 根据格式推断schema
        match format {
            crate::data::loader::types::DataFormat::Json { .. } => {
                self.infer_json_schema(&data)
            },
            crate::data::loader::types::DataFormat::Csv { .. } => {
                self.infer_csv_schema(&data)
            },
            _ => Err(Error::unsupported_file_type(format!("不支持的格式进行schema推断: {:?}", format))),
        }
    }

    fn name(&self) -> &'static str {
        "memory_loader"
    }

    async fn load_batch(&self, source: &DataSource, format: &crate::data::loader::types::DataFormat, batch_size: usize, offset: usize) -> Result<DataBatch> {
        // 先加载完整数据
        let mut batch = self.load(source, format).await?;
        
        // 然后切片获取指定范围的数据
        let start = offset;
        let features_vec = batch.features();
        let features_len = features_vec.len();
        let end = std::cmp::min(start + batch_size, features_len);
        
        if start >= features_len {
            // 如果偏移量超出范围，返回空批次
            batch.features = Some(Vec::new());
        } else {
            batch.features = Some(features_vec[start..end].to_vec());
        }
        
        // 更新元数据
        batch.metadata.insert("batch_start".to_string(), start.to_string());
        batch.metadata.insert("batch_end".to_string(), end.to_string());
        batch.metadata.insert("batch_size".to_string(), (end - start).to_string());
        
        Ok(batch)
    }

    fn supports_format(&self, format: &crate::data::loader::types::DataFormat) -> bool {
        matches!(format, 
            crate::data::loader::types::DataFormat::Json { .. } | 
            crate::data::loader::types::DataFormat::Csv { .. }
        )
    }

    fn config(&self) -> &LoaderConfig {
        &self.config
    }

    fn set_config(&mut self, config: LoaderConfig) {
        self.config = config;
    }
}

impl MemoryDataLoader {
    /// 创建默认schema
    fn create_default_schema() -> Result<DataSchema> {
        use crate::data::schema::schema::{FieldDefinition, FieldType as SchemaFieldType};
        
        let mut schema = DataSchema::new("default_memory_schema", "1.0");
        schema.description = Some("内存数据源的默认schema".to_string());
        
        // 添加默认字段
        let fields = vec![
            FieldDefinition {
                name: "id".to_string(),
                field_type: SchemaFieldType::Text,
                data_type: None,
                required: false,
                nullable: true,
                primary_key: false,
                foreign_key: None,
                description: None,
                default_value: None,
                constraints: None,
                metadata: HashMap::new(),
            },
            FieldDefinition {
                name: "data".to_string(),
                field_type: SchemaFieldType::Text,
                data_type: None,
                required: false,
                nullable: true,
                primary_key: false,
                foreign_key: None,
                description: None,
                default_value: None,
                constraints: None,
                metadata: HashMap::new(),
            },
        ];
        
        for field in fields {
            schema.fields.push(field);
        }
        
        Ok(schema)
    }
    
    /// 解析JSON数据
    fn parse_json_data(&self, data: &[u8]) -> Result<Vec<Vec<f32>>> {
        let json_str = String::from_utf8(data.to_vec())
            .map_err(|e| Error::invalid_input(format!("无法解析UTF-8数据: {}", e)))?;

        let json_value: serde_json::Value = serde_json::from_str(&json_str)
            .map_err(|e| Error::invalid_input(format!("无法解析JSON: {}", e)))?;

        match json_value {
            serde_json::Value::Array(arr) => {
                let mut features = Vec::new();
                for item in arr {
                    match item {
                        serde_json::Value::Array(row) => {
                            let mut feature_row = Vec::new();
                            for val in row {
                                let f_val = match val {
                                    serde_json::Value::Number(n) => {
                                        n.as_f64().unwrap_or(0.0) as f32
                                    },
                                    serde_json::Value::String(s) => {
                                        s.parse::<f32>().unwrap_or(0.0)
                                    },
                                    _ => 0.0,
                                };
                                feature_row.push(f_val);
                            }
                            features.push(feature_row);
                        },
                        serde_json::Value::Object(obj) => {
                            // 将对象的值转换为特征向量
                            let mut feature_row = Vec::new();
                            for (_, val) in obj {
                                let f_val = match val {
                                    serde_json::Value::Number(n) => {
                                        n.as_f64().unwrap_or(0.0) as f32
                                    },
                                    serde_json::Value::String(s) => {
                                        s.parse::<f32>().unwrap_or(0.0)
                                    },
                                    _ => 0.0,
                                };
                                feature_row.push(f_val);
                            }
                            features.push(feature_row);
                        },
                        _ => {
                            return Err(Error::invalid_input("不支持的JSON数据格式".to_string()));
                        }
                    }
                }
                Ok(features)
            },
            _ => Err(Error::invalid_input("JSON数据必须是数组格式".to_string())),
        }
    }

    /// 解析CSV数据
    fn parse_csv_data(&self, data: &[u8]) -> Result<Vec<Vec<f32>>> {
        let csv_str = String::from_utf8(data.to_vec())
            .map_err(|e| Error::invalid_input(format!("无法解析UTF-8数据: {}", e)))?;

        let mut features = Vec::new();
        let mut lines = csv_str.lines();

        // 跳过标题行（如果配置要求）
        if self.config.has_header.unwrap_or(true) {
            lines.next();
        }

        for line in lines {
            if line.trim().is_empty() {
                continue;
            }

            let values: Vec<&str> = line.split(',').collect();
            let mut feature_row = Vec::new();

            for value in values {
                let f_val = value.trim().parse::<f32>().unwrap_or(0.0);
                feature_row.push(f_val);
            }

            if !feature_row.is_empty() {
                features.push(feature_row);
            }
        }

        Ok(features)
    }

    /// 推断JSON数据的schema
    fn infer_json_schema(&self, data: &[u8]) -> Result<DataSchema> {
        use crate::data::schema::schema::{FieldDefinition, FieldType as SchemaFieldType};
        
        let json_str = String::from_utf8(data.to_vec())
            .map_err(|e| Error::invalid_input(format!("无法解析UTF-8数据: {}", e)))?;

        let json_value: serde_json::Value = serde_json::from_str(&json_str)
            .map_err(|e| Error::invalid_input(format!("无法解析JSON: {}", e)))?;

        let mut fields = Vec::new();

        match json_value {
            serde_json::Value::Array(arr) if !arr.is_empty() => {
                match &arr[0] {
                    serde_json::Value::Array(row) => {
                        // 数组的数组格式
                        for (i, val) in row.iter().enumerate() {
                            let field_type = match val {
                                serde_json::Value::Number(_) => SchemaFieldType::Numeric,
                                serde_json::Value::String(_) => SchemaFieldType::Text,
                                serde_json::Value::Bool(_) => SchemaFieldType::Boolean,
                                _ => SchemaFieldType::Text,
                            };
                            fields.push(FieldDefinition {
                                name: format!("column_{}", i),
                                field_type,
                                data_type: None,
                                required: false,
                                nullable: true,
                                primary_key: false,
                                foreign_key: None,
                                description: None,
                                default_value: None,
                                constraints: None,
                                metadata: HashMap::new(),
                            });
                        }
                    },
                    serde_json::Value::Object(obj) => {
                        // 对象数组格式
                        for (key, val) in obj.iter() {
                            let field_type = match val {
                                serde_json::Value::Number(_) => SchemaFieldType::Numeric,
                                serde_json::Value::String(_) => SchemaFieldType::Text,
                                serde_json::Value::Bool(_) => SchemaFieldType::Boolean,
                                _ => SchemaFieldType::Text,
                            };
                            fields.push(FieldDefinition {
                                name: key.clone(),
                                field_type,
                                data_type: None,
                                required: false,
                                nullable: true,
                                primary_key: false,
                                foreign_key: None,
                                description: None,
                                default_value: None,
                                constraints: None,
                                metadata: HashMap::new(),
                            });
                        }
                    },
                    _ => {
                        return Err(Error::invalid_input("不支持的JSON数据格式".to_string()));
                    }
                }
            },
            _ => {
                return Err(Error::invalid_input("JSON数据必须是数组格式".to_string()));
            }
        }

        Ok(DataSchema {
            name: "inferred_json_schema".to_string(),
            version: "1.0".to_string(),
            description: Some("从JSON数据推断的schema".to_string()),
            fields,
            primary_key: None,
            indexes: None,
            relationships: None,
            metadata: HashMap::new(),
        })
    }

    /// 推断CSV数据的schema
    fn infer_csv_schema(&self, data: &[u8]) -> Result<DataSchema> {
        use crate::data::schema::schema::{FieldDefinition, FieldType as SchemaFieldType};
        
        let csv_str = String::from_utf8(data.to_vec())
            .map_err(|e| Error::invalid_input(format!("无法解析UTF-8数据: {}", e)))?;

        let mut lines = csv_str.lines();
        let mut fields = Vec::new();

        // 检查是否有标题行
        let has_header = self.config.has_header.unwrap_or(true);
        
        if has_header {
            if let Some(header_line) = lines.next() {
                let headers: Vec<&str> = header_line.split(',').collect();
                
                // 如果有数据行，分析第一行数据来推断类型
                if let Some(first_data_line) = lines.next() {
                    let values: Vec<&str> = first_data_line.split(',').collect();
                    
                    for (header, value) in headers.iter().zip(values.iter()) {
                        let field_type = if value.trim().parse::<f64>().is_ok() {
                            SchemaFieldType::Numeric
                        } else if value.trim().parse::<i64>().is_ok() {
                            SchemaFieldType::Numeric
                        } else {
                            SchemaFieldType::Text
                        };

                        fields.push(FieldDefinition {
                            name: header.trim().to_string(),
                            field_type,
                            data_type: None,
                            required: false,
                            nullable: true,
                            primary_key: false,
                            foreign_key: None,
                            description: None,
                            default_value: None,
                            constraints: None,
                            metadata: HashMap::new(),
                        });
                    }
                } else {
                    // 只有标题行，假设所有字段都是字符串类型
                    for header in headers.iter() {
                        fields.push(FieldDefinition {
                            name: header.trim().to_string(),
                            field_type: SchemaFieldType::Text,
                            data_type: None,
                            required: false,
                            nullable: true,
                            primary_key: false,
                            foreign_key: None,
                            description: None,
                            default_value: None,
                            constraints: None,
                            metadata: HashMap::new(),
                        });
                    }
                }
            }
        } else {
            // 没有标题行，分析第一行数据
            if let Some(first_line) = lines.next() {
                let values: Vec<&str> = first_line.split(',').collect();
                
                for (i, value) in values.iter().enumerate() {
                    let field_type = if value.trim().parse::<f64>().is_ok() {
                        SchemaFieldType::Numeric
                    } else if value.trim().parse::<i64>().is_ok() {
                        SchemaFieldType::Numeric
                    } else {
                        SchemaFieldType::Text
                    };

                    fields.push(FieldDefinition {
                        name: format!("column_{}", i),
                        field_type,
                        data_type: None,
                        required: false,
                        nullable: true,
                        primary_key: false,
                        foreign_key: None,
                        description: None,
                        default_value: None,
                        constraints: None,
                        metadata: HashMap::new(),
                    });
                }
            }
        }

        if fields.is_empty() {
            return Err(Error::invalid_input("无法从CSV数据中推断schema".to_string()));
        }

        Ok(DataSchema {
            name: "inferred_csv_schema".to_string(),
            version: "1.0".to_string(),
            description: Some("从CSV数据推断的schema".to_string()),
            fields,
            primary_key: None,
            indexes: None,
            relationships: None,
            metadata: HashMap::new(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio;

    #[tokio::test]
    async fn test_memory_loader_basic() {
        let loader = MemoryDataLoader::new();
        
        // 添加测试数据
        let json_data = r#"[[1.0, 2.0], [3.0, 4.0]]"#;
        loader.load_from_json("test".to_string(), json_data).await.unwrap();
        
        // 加载数据
        let batch = loader.load("test").await.unwrap();
        assert_eq!(batch.features.len(), 2);
        assert_eq!(batch.features[0], vec![1.0, 2.0]);
        assert_eq!(batch.features[1], vec![3.0, 4.0]);
    }

    #[tokio::test]
    async fn test_memory_loader_csv() {
        let mut config = LoaderConfig::default();
        config.has_header = Some(true);
        let loader = MemoryDataLoader::with_config(config);
        
        let csv_data = "header1,header2\n1.5,2.5\n3.5,4.5";
        loader.load_from_csv("test_csv".to_string(), csv_data).await.unwrap();
        
        let batch = loader.load("test_csv").await.unwrap();
        assert_eq!(batch.features.len(), 2);
        assert_eq!(batch.features[0], vec![1.5, 2.5]);
        assert_eq!(batch.features[1], vec![3.5, 4.5]);
    }

    #[tokio::test]
    async fn test_concurrent_batch_loading() {
        let loader = MemoryDataLoader::new();
        
        // 添加多个测试数据
        for i in 0..10 {
            let json_data = format!(r#"[[{}.0, {}.0]]"#, i, i + 1);
            loader.load_from_json(format!("test_{}", i), &json_data).await.unwrap();
        }
        
        // 并发批量加载
        let paths: Vec<String> = (0..10).map(|i| format!("test_{}", i)).collect();
        let batches = loader.load_batch_concurrent(&paths).await.unwrap();
        
        assert_eq!(batches.len(), 10);
        for (i, batch) in batches.iter().enumerate() {
            assert_eq!(batch.features[0], vec![i as f32, (i + 1) as f32]);
        }
    }

    #[tokio::test]
    async fn test_cache_functionality() {
        let mut batch_config = AsyncBatchConfig::default();
        batch_config.enable_cache = true;
        batch_config.cache_ttl = Duration::from_secs(1);
        
        let config = LoaderConfig::default();
        let loader = MemoryDataLoader::with_batch_config(config, batch_config);
        
        // 添加测试数据
        let json_data = r#"[[1.0, 2.0]]"#;
        loader.load_from_json("cache_test".to_string(), json_data).await.unwrap();
        
        // 第一次加载（应该缓存）
        let _batch1 = loader.load("cache_test").await.unwrap();
        
        // 检查缓存统计
        let stats = loader.get_cache_stats().await.unwrap();
        assert_eq!(stats.get("total_items").unwrap_or(&0), &1);
        
        // 等待缓存过期
        tokio::time::sleep(Duration::from_secs(2)).await;
        
        // 清理过期缓存
        let cleaned = loader.cleanup_cache().await.unwrap();
        assert_eq!(cleaned, 1);
    }

    #[tokio::test]
    async fn test_memory_operations() {
        let loader = MemoryDataLoader::new();
        
        // 测试基本操作
        assert!(loader.is_empty().await.unwrap());
        assert_eq!(loader.len().await.unwrap(), 0);
        
        // 添加数据
        loader.add_data("key1".to_string(), vec![1, 2, 3]).await.unwrap();
        loader.add_data("key2".to_string(), vec![4, 5, 6]).await.unwrap();
        
        assert!(!loader.is_empty().await.unwrap());
        assert_eq!(loader.len().await.unwrap(), 2);
        
        // 检查键
        let keys = loader.keys().await.unwrap();
        assert_eq!(keys.len(), 2);
        assert!(keys.contains(&"key1".to_string()));
        assert!(keys.contains(&"key2".to_string()));
        
        // 获取数据
        let data = loader.get_data("key1").await.unwrap();
        assert_eq!(data, Some(vec![1, 2, 3]));
        
        // 删除数据
        let removed = loader.remove_data("key1").await.unwrap();
        assert_eq!(removed, Some(vec![1, 2, 3]));
        assert_eq!(loader.len().await.unwrap(), 1);
        
        // 清空所有数据
        loader.clear().await.unwrap();
        assert!(loader.is_empty().await.unwrap());
    }
}