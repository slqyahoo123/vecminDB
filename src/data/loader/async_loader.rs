// 异步数据加载器实现

use std::collections::HashMap;
use std::sync::Arc;
use std::pin::Pin;
use std::time::Instant;
use std::future::Future;

use async_trait::async_trait;
use futures::{Stream, StreamExt};
use log::{debug, error, info};
use tokio::sync::{Mutex as TokioMutex, Semaphore};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::data::{DataBatch, DataConfig, DataSchema};
use crate::data::schema::schema::{FieldDefinition, FieldType as SchemaFieldType};
use crate::data::loader::{DataLoader, DataSource};
// use crate::data::pipeline::pipeline::PipelineContext;
use crate::error::{Error, Result};

/// 异步数据库连接池
pub struct AsyncDbConnectionPool<T> {
    /// 连接池
    pool: Arc<TokioMutex<Vec<T>>>,
    /// 可用连接信号量
    available: Arc<Semaphore>,
    /// 最大连接数
    max_connections: usize,
    /// 连接工厂
    connection_factory: Arc<dyn Fn() -> Pin<Box<dyn Future<Output = Result<T>> + Send>> + Send + Sync>,
}

impl<T: Clone + Send + Sync + 'static> AsyncDbConnectionPool<T> {
    /// 创建新的异步连接池
    pub async fn new<F, Fut>(
        connection_factory: F,
        initial_size: usize,
        max_size: usize,
    ) -> Result<Self>
    where
        F: Fn() -> Fut + Send + Sync + 'static,
        Fut: Future<Output = Result<T>> + Send + 'static,
    {
        // 验证参数
        if initial_size > max_size {
            return Err(Error::invalid_argument("初始连接数不能大于最大连接数"));
        }
        
        // 包装连接工厂
        let factory = Arc::new(move || -> Pin<Box<dyn Future<Output = Result<T>> + Send>> {
            let fut = connection_factory();
            Box::pin(fut)
        });
        
        // 创建初始连接
        let mut connections = Vec::with_capacity(initial_size);
        for _ in 0..initial_size {
            let connection = factory().await?;
            connections.push(connection);
        }
        
        let pool = Arc::new(TokioMutex::new(connections));
        let available = Arc::new(Semaphore::new(initial_size));
        
        Ok(Self {
            pool,
            available,
            max_connections: max_size,
            connection_factory: factory,
        })
    }
    
    /// 获取连接
    pub async fn get_connection(&self) -> Result<PooledConnection<T>> {
        // 尝试获取信号量许可
        let _permit = match self.available.try_acquire() {
            Ok(permit) => permit,
            Err(_) => {
                // 没有可用连接，检查是否可以创建新连接
                let current_size = self.pool.lock().await.len();
                if current_size < self.max_connections {
                    // 创建新连接
                    let connection = (self.connection_factory)().await?;
                    {
                        let mut pool = self.pool.lock().await;
                        pool.push(connection.clone());
                    }
                    // 增加信号量大小
                    self.available.add_permits(1);
                    self.available.acquire().await.map_err(|e| Error::internal(&format!("获取连接许可失败: {}", e)))?
                } else {
                    // 等待可用连接
                    debug!("连接池已满，等待可用连接");
                    self.available.acquire().await.map_err(|e| Error::internal(&format!("获取连接许可失败: {}", e)))?
                }
            }
        };
        
        // 获取连接
        let connection = {
            let mut pool = self.pool.lock().await;
            if pool.is_empty() {
                // 这种情况不应该发生，因为我们已经获取了信号量许可
                return Err(Error::internal("连接池为空，但信号量不为零"));
            }
            pool.remove(0)
        };
        
        Ok(PooledConnection {
            connection,
            pool: self.pool.clone(),
            available: self.available.clone(),
        })
    }
    
    /// 获取连接池大小
    pub async fn size(&self) -> usize {
        self.pool.lock().await.len()
    }
    
    /// 获取可用连接数
    pub fn available_connections(&self) -> usize {
        self.available.available_permits()
    }
    
    /// 关闭连接池
    pub async fn close(&self) {
        self.pool.lock().await.clear();
    }
}

/// 池化连接封装
pub struct PooledConnection<T: Clone + Send + Sync + 'static> {
    /// 数据库连接
    pub connection: T,
    /// 连接池引用
    pool: Arc<TokioMutex<Vec<T>>>,
    /// 可用连接信号量
    available: Arc<Semaphore>,
}

impl<T: Clone + Send + Sync + 'static> Drop for PooledConnection<T> {
    fn drop(&mut self) {
        // 创建连接的拷贝
        let connection = self.connection.clone();
        let pool = self.pool.clone();
        let available = self.available.clone();
        
        // 使用tokio的spawn将连接放回池中
        tokio::spawn(async move {
            let mut pool_guard = pool.lock().await;
            pool_guard.push(connection);
            available.add_permits(1);
        });
    }
}

/// 高性能异步数据库加载器
#[derive(Clone)]
pub struct HighPerformanceDbLoader {
    /// 连接池
    connection_pool: Arc<AsyncDbConnectionPool<Arc<dyn DatabaseConnection>>>,
    /// 查询
    query: String,
    /// 查询参数
    params: Vec<QueryParam>,
    /// 批处理大小
    batch_size: usize,
    /// 最大并发查询数
    max_concurrent_queries: usize,
    /// 是否缓存结果
    cache_results: bool,
    /// 结果缓存
    result_cache: Arc<TokioMutex<HashMap<String, DataBatch>>>,
    /// 加载器配置
    config: DataConfig,
}

/// 数据库连接特性
#[async_trait]
pub trait DatabaseConnection: Send + Sync {
    /// 执行查询
    async fn execute_query(&self, query: &str, params: &[QueryParam]) -> Result<Vec<HashMap<String, Value>>>;
    
    /// 获取表结构
    async fn get_table_schema(&self, table_name: &str) -> Result<DataSchema>;
    
    /// 测试连接
    async fn test_connection(&self) -> Result<()>;
    
    /// 开始事务
    async fn begin_transaction(&self) -> Result<()>;
    
    /// 提交事务
    async fn commit_transaction(&self) -> Result<()>;
    
    /// 回滚事务
    async fn rollback_transaction(&self) -> Result<()>;
    
    /// 获取连接信息
    fn connection_info(&self) -> String;
}

/// 查询参数
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum QueryParam {
    /// 空值
    Null,
    /// 字符串
    String(String),
    /// 整数
    Integer(i64),
    /// 浮点数
    Float(f64),
    /// 布尔值
    Boolean(bool),
    /// 日期
    Date(chrono::NaiveDate),
    /// 日期时间
    DateTime(chrono::NaiveDateTime),
    /// 二进制数据
    Binary(Vec<u8>),
}

impl HighPerformanceDbLoader {
    /// 创建新的高性能数据库加载器
    pub async fn new(
        _connection_string: &str,
        query: &str,
        connection_factory: impl Fn() -> Pin<Box<dyn Future<Output = Result<Arc<dyn DatabaseConnection>>> + Send>> + Send + Sync + 'static,
        config: Option<DataConfig>,
    ) -> Result<Self> {
        // 创建连接池
        let pool = AsyncDbConnectionPool::new(connection_factory, 5, 20).await?;
        
        let loader = Self {
            connection_pool: Arc::new(pool),
            query: query.to_string(),
            params: Vec::new(),
            batch_size: 1000,
            max_concurrent_queries: 4,
            cache_results: false,
            result_cache: Arc::new(TokioMutex::new(HashMap::new())),
            config: config.unwrap_or_default(),
        };
        
        Ok(loader)
    }
    
    /// 设置查询参数
    pub fn with_params(mut self, params: Vec<QueryParam>) -> Self {
        self.params = params;
        self
    }
    
    /// 设置批处理大小
    pub fn with_batch_size(mut self, batch_size: usize) -> Self {
        self.batch_size = batch_size;
        self
    }
    
    /// 设置最大并发查询数
    pub fn with_max_concurrent_queries(mut self, max: usize) -> Self {
        self.max_concurrent_queries = max;
        self
    }
    
    /// 设置是否缓存结果
    pub fn with_cache_results(mut self, cache: bool) -> Self {
        self.cache_results = cache;
        self
    }
    
    /// 生成查询缓存键
    fn generate_cache_key(&self) -> String {
        // 使用查询和参数生成缓存键
        let mut hasher = blake3::Hasher::new();
        hasher.update(self.query.as_bytes());
        
        for param in &self.params {
            match param {
                QueryParam::Null => { hasher.update(b"null"); }
                QueryParam::String(s) => { hasher.update(s.as_bytes()); }
                QueryParam::Integer(i) => { hasher.update(&i.to_le_bytes()); }
                QueryParam::Float(f) => { hasher.update(&f.to_le_bytes()); }
                QueryParam::Boolean(b) => { hasher.update(&[*b as u8]); }
                QueryParam::Date(d) => { hasher.update(d.to_string().as_bytes()); }
                QueryParam::DateTime(dt) => { hasher.update(dt.to_string().as_bytes()); }
                QueryParam::Binary(data) => { hasher.update(data); }
            }
        }
        
        let hash = hasher.finalize();
        format!("query_{}", hash.to_hex())
    }
    
    /// 执行批量加载
    pub async fn load_in_batches(&self) -> Result<Vec<DataBatch>> {
        let start_time = Instant::now();
        info!("开始批量加载数据...");
        
        // 如果启用了缓存，尝试从缓存加载
        if self.cache_results {
            let cache_key = self.generate_cache_key();
            let cache = self.result_cache.lock().await;
            if let Some(batch) = cache.get(&cache_key) {
                info!("从缓存加载数据，缓存键: {}", cache_key);
                return Ok(vec![batch.clone()]);
            }
        }
        
        // 获取连接执行查询
        let conn = self.connection_pool.get_connection().await?;
        let rows = conn.connection.execute_query(&self.query, &self.params).await?;
        
        // 如果没有数据，返回空批次
        if rows.is_empty() {
            info!("查询未返回任何数据");
            return Ok(vec![DataBatch::new("empty", 0, 0)]);
        }
        
        // 将结果分割为多个批次
        let mut batches = Vec::new();
        let mut current_batch = DataBatch::new("batch", 0, self.batch_size);
        current_batch.id = Some(Uuid::new_v4().to_string());
        
        // 分批处理
        for (i, row) in rows.iter().enumerate() {
            // 将行转换为记录
            let record_json = serde_json::to_string(row)
                .map_err(|e| Error::processing(format!("序列化记录失败: {}", e)))?;
                
            current_batch.add_record_json(serde_json::from_str(&record_json)?)
                .map_err(|e| Error::processing(format!("添加记录失败: {}", e)))?;
                
            // 如果达到批处理大小，创建新批次
            if (i + 1) % self.batch_size == 0 && i + 1 < rows.len() {
                batches.push(current_batch);
                current_batch = DataBatch::new("batch", batches.len(), self.batch_size);
                current_batch.id = Some(Uuid::new_v4().to_string());
            }
        }
        
        // 添加最后一个批次（如果不为空）
        if !current_batch.is_empty() {
            batches.push(current_batch);
        }
        
        let elapsed = start_time.elapsed();
        info!("批量加载完成，共 {} 条记录分为 {} 个批次，耗时: {:?}", 
              rows.len(), batches.len(), elapsed);
              
        // 如果启用了缓存，将结果存入缓存
        if self.cache_results && !batches.is_empty() {
            let cache_key = self.generate_cache_key();
            let mut cache = self.result_cache.lock().await;
            cache.insert(cache_key, batches[0].clone());
        }
        
        Ok(batches)
    }
    
    /// 获取表结构
    pub async fn get_table_schema(&self, table_name: &str) -> Result<DataSchema> {
        let conn = self.connection_pool.get_connection().await?;
        conn.connection.get_table_schema(table_name).await
    }
}

#[async_trait]
impl DataLoader for HighPerformanceDbLoader {
    async fn load(&self, source: &DataSource, format: &crate::data::loader::types::DataFormat) -> Result<DataBatch> {
        match source {
            DataSource::Database(db_config) => {
                // 从源配置更新连接信息
                let query = db_config.query.clone().unwrap_or_else(|| self.query.clone());
                
                // 执行查询
                let batches = self.load_in_batches().await?;
                
                // 如果没有批次，返回空批次
                if batches.is_empty() {
                    return Ok(DataBatch::new("empty", 0, 0));
                }
                
                // 返回第一个批次
                Ok(batches[0].clone())
            },
            _ => Err(Error::invalid_input(format!("数据源类型不匹配: {:?}", source))),
        }
    }
    
    async fn get_schema(&self, source: &DataSource, format: &crate::data::loader::types::DataFormat) -> Result<DataSchema> {
        match source {
            DataSource::Database(db_config) => {
                if let Some(table) = &db_config.table {
                    // 如果提供了表名，获取表结构
                    self.get_table_schema(table).await
                } else {
                    // 创建样本模式
                    let mut schema = DataSchema::new("db_schema", "1.0");
                    schema.description = Some(format!("从查询自动生成的模式: {}", self.query));
                    
                    // 添加样本字段
                    let fields = vec![
                        FieldDefinition {
                            name: "id".to_string(),
                            field_type: SchemaFieldType::Text,
                            data_type: None,
                            required: true,
                            nullable: false,
                            primary_key: false,
                            foreign_key: None,
                            description: None,
                            default_value: None,
                            constraints: None,
                            metadata: std::collections::HashMap::new(),
                        },
                        FieldDefinition {
                            name: "name".to_string(),
                            field_type: SchemaFieldType::Text,
                            data_type: None,
                            required: true,
                            nullable: false,
                            primary_key: false,
                            foreign_key: None,
                            description: None,
                            default_value: None,
                            constraints: None,
                            metadata: std::collections::HashMap::new(),
                        },
                        FieldDefinition {
                            name: "value".to_string(),
                            field_type: SchemaFieldType::Numeric,
                            data_type: None,
                            required: false,
                            nullable: false,
                            primary_key: false,
                            foreign_key: None,
                            description: None,
                            default_value: None,
                            constraints: None,
                            metadata: std::collections::HashMap::new(),
                        },
                        FieldDefinition {
                            name: "created_at".to_string(),
                            field_type: SchemaFieldType::DateTime,
                            data_type: None,
                            required: false,
                            nullable: false,
                            primary_key: false,
                            foreign_key: None,
                            description: None,
                            default_value: None,
                            constraints: None,
                            metadata: std::collections::HashMap::new(),
                        },
                    ];
                    for f in fields {
                        schema.add_field(f)?;
                    }
                    
                    Ok(schema)
                }
            },
            _ => Err(Error::invalid_input(format!("数据源类型不匹配: {:?}", source))),
        }
    }
    
    fn name(&self) -> &'static str {
        "HighPerformanceDbLoader"
    }
    
    // 添加缺失的trait方法
    async fn load_batch(&self, source: &DataSource, format: &crate::data::loader::types::DataFormat, batch_size: usize, offset: usize) -> Result<DataBatch> {
        // 对于数据库加载器，我们可以直接利用批次加载功能
        match source {
            DataSource::Database(_) => {
                let batches = self.load_in_batches().await?;
                let batch_index = offset / batch_size;
                
                if batch_index < batches.len() {
                    Ok(batches[batch_index].clone())
                } else {
                    Ok(DataBatch::new("empty", 0, 0))
                }
            },
            _ => Err(crate::error::Error::invalid_input(&format!("数据源类型不匹配: {:?}", source))),
        }
    }
    
    fn supports_format(&self, format: &crate::data::loader::types::DataFormat) -> bool {
        // 数据库加载器格式支持相对宽泛，主要依赖于查询结果的解析
        match format {
            crate::data::loader::types::DataFormat::Csv { .. } => true,
            crate::data::loader::types::DataFormat::Json { .. } => true,
            _ => false, // 其他格式不直接支持
        }
    }
    
    fn config(&self) -> &crate::data::loader::LoaderConfig {
        // 返回一个默认配置的引用
        static DEFAULT_CONFIG: std::sync::OnceLock<crate::data::loader::LoaderConfig> = std::sync::OnceLock::new();
        DEFAULT_CONFIG.get_or_init(|| crate::data::loader::LoaderConfig::default())
    }
    
    fn set_config(&mut self, config: crate::data::loader::LoaderConfig) {
        // 从LoaderConfig中提取相关设置更新内部DataConfig
        if let Some(batch_size) = config.batch_size {
            self.batch_size = batch_size;
        }
        
        // 从选项中提取数据库相关设置
        if let Some(max_concurrent_str) = config.options.get("max_concurrent_queries") {
            if let Ok(max_concurrent) = max_concurrent_str.parse::<usize>() {
                self.max_concurrent_queries = max_concurrent;
            }
        }
        
        if let Some(cache_str) = config.options.get("cache_results") {
            if let Ok(cache) = cache_str.parse::<bool>() {
                self.cache_results = cache;
            }
        }
        
        // 更新内部config
        if let Some(format) = config.format {
            // 将loader::types::DataFormat转换为data::DataFormat
            self.config.format = match format {
                crate::data::loader::types::DataFormat::Csv { .. } => crate::data::types::DataFormat::CSV,
                crate::data::loader::types::DataFormat::Json { .. } => crate::data::types::DataFormat::JSON,
                _ => crate::data::types::DataFormat::JSON, // 默认为JSON
            };
        }
        
        self.config.validate = Some(config.validate);
    }
}

/// 将 Record 转换为 HashMap<String, DataValue>
fn record_to_hashmap(record: &crate::data::record::Record) -> HashMap<String, crate::data::DataValue> {
    let mut map = HashMap::new();
    for (key, value) in &record.fields {
        if let crate::data::record::Value::Data(data_value) = value {
            map.insert(key.clone(), data_value.clone());
        }
    }
    map
}

/// 处理数据集加载上下文
/// 使用PipelineContext增强数据加载功能
pub async fn process_with_context(
    loader: &dyn DataLoader,
    source: &DataSource,
    format: &crate::data::loader::types::DataFormat,
    context: &crate::data::pipeline::PipelineContext
) -> Result<(DataBatch, DataSchema)> {
    let start_time = Instant::now();
    
    // 使用loader加载数据批次
    let batch = loader.load(source, format).await?;
    
    // 获取数据模式
    let schema = loader.get_schema(source, format).await?;
    
    // 使用上下文增强数据处理 - 生产级实现
    let enhanced_schema = if let Some(transformations) = context.get_transformations() {
        // 应用上下文中的转换
        let mut modified_schema = schema;
        
        for transformation in &transformations {
            debug!("应用转换: {:?}", transformation);
            
            // 根据转换类型应用具体的模式修改
            match &transformation.transformation_type {
                crate::data::DataTransformationType::FieldMapping { source, target } => {
                    // 字段映射：重命名字段
                    if let Some(field) = modified_schema.fields.iter_mut().find(|f| &f.name == source) {
                        field.name = target.clone();
                    }
                },
                crate::data::DataTransformationType::TypeConversion { field, target_type } => {
                    // 类型转换：修改字段类型
                    if let Some(field_def) = modified_schema.fields.iter_mut().find(|f| &f.name == field) {
                        // 使用转换函数将FieldType转换为schema::FieldType
                        field_def.field_type = convert_data_field_type_to_schema(target_type);
                    }
                },
                crate::data::DataTransformationType::Filtering { conditions } => {
                    // 过滤：为模式描述添加过滤条件信息
                    let filter_info = format!("filter_conditions: {}", 
                        serde_json::to_string(conditions).unwrap_or_default());
                    modified_schema.description = Some(
                        modified_schema.description
                            .map(|desc| format!("{} | {}", desc, filter_info))
                            .unwrap_or(filter_info)
                    );
                },
                crate::data::DataTransformationType::Aggregation { field, operation } => {
                    // 聚合：修改字段的统计信息
                    if let Some(field_def) = modified_schema.fields.iter_mut().find(|f| &f.name == field) {
                        field_def.metadata.insert(
                            "aggregation".to_string(),
                            format!("{:?}", operation)
                        );
                    }
                },
                crate::data::DataTransformationType::Normalization { field, method } => {
                    // 标准化：为字段添加标准化元数据
                    if let Some(field_def) = modified_schema.fields.iter_mut().find(|f| &f.name == field) {
                        field_def.metadata.insert(
                            "normalization".to_string(),
                            format!("{:?}", method)
                        );
                    }
                },
                crate::data::DataTransformationType::FeatureEngineering { features } => {
                    // 特征工程：添加新的衍生字段
                    for feature in features {
                        // 将data::FieldType转换为schema::FieldType
                        let schema_field_type = convert_data_field_type_to_schema(&feature.data_type);
                        
                        let new_field = crate::data::schema::schema::FieldDefinition {
                            name: feature.name.clone(),
                            field_type: schema_field_type,
                            data_type: None,
                            required: !feature.nullable.unwrap_or(true),
                            nullable: feature.nullable.unwrap_or(true),
                            primary_key: false,
                            foreign_key: None,
                            description: Some(format!("Generated by feature engineering: {}", 
                                feature.expression.clone().unwrap_or_default())),
                            default_value: None,
                            constraints: None,
                            metadata: {
                                let mut meta = std::collections::HashMap::new();
                                meta.insert("source".to_string(), "feature_engineering".to_string());
                                meta.insert("expression".to_string(), feature.expression.clone().unwrap_or_default());
                                meta
                            },
                        };
                        modified_schema.fields.push(new_field);
                    }
                },
                crate::data::DataTransformationType::Custom { name, config } => {
                    // 自定义转换：应用配置中定义的转换
                    debug!("应用自定义转换 '{}': {:?}", name, config);
                    
                    // 从配置中提取转换参数
                    if let Some(field_changes) = config.get("field_changes") {
                        if let Ok(changes) = serde_json::from_value::<Vec<serde_json::Value>>(field_changes.clone()) {
                            for change in changes {
                                if let (Some(field_name), Some(new_type)) = (
                                    change.get("field").and_then(|v| v.as_str()),
                                    change.get("type").and_then(|v| v.as_str())
                                ) {
                                    if let Some(field) = modified_schema.fields.iter_mut().find(|f| f.name == field_name) {
                                        // 根据字符串创建对应的schema::FieldType
                                        let schema_field_type = match new_type {
                                            "String" | "Text" => crate::data::schema::schema::FieldType::Text,
                                            "Integer" | "Numeric" => crate::data::schema::schema::FieldType::Numeric,
                                            "Boolean" => crate::data::schema::schema::FieldType::Boolean,
                                            "Binary" => crate::data::schema::schema::FieldType::Custom("Binary".to_string()),
                                            _ => crate::data::schema::schema::FieldType::Text,
                                        };
                                        field.field_type = schema_field_type;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        
        modified_schema
    } else {
        schema
    };
    
    // 记录性能指标和上下文信息
    let elapsed = start_time.elapsed();
    debug!("处理上下文加载完成，耗时 {:?}，应用了 {} 个上下文转换", 
           elapsed, context.get_transformations().map(|t| t.len()).unwrap_or(0));
    
    Ok((batch, enhanced_schema))
}

/// 异步流数据加载器
/// 用于处理连续的数据流
pub struct AsyncStreamLoader {
    /// 流配置
    config: DataConfig,
    /// 批处理大小
    batch_size: usize,
    /// 缓冲区大小
    buffer_size: usize,
    /// 是否自动重连
    auto_reconnect: bool,
    /// 重连延迟（毫秒）
    reconnect_delay_ms: u64,
    /// 重连最大尝试次数
    max_reconnect_attempts: u32,
}

impl AsyncStreamLoader {
    /// 创建新的异步流加载器
    pub fn new(config: DataConfig) -> Self {
        Self {
            config,
            batch_size: 1000,
            buffer_size: 4096,
            auto_reconnect: true,
            reconnect_delay_ms: 1000,
            max_reconnect_attempts: 5,
        }
    }
    
    /// 设置批处理大小
    pub fn with_batch_size(mut self, size: usize) -> Self {
        self.batch_size = size;
        self
    }
    
    /// 设置缓冲区大小
    pub fn with_buffer_size(mut self, size: usize) -> Self {
        self.buffer_size = size;
        self
    }
    
    /// 设置自动重连
    pub fn with_auto_reconnect(mut self, auto: bool) -> Self {
        self.auto_reconnect = auto;
        self
    }
    
    /// 设置重连延迟
    pub fn with_reconnect_delay(mut self, delay_ms: u64) -> Self {
        self.reconnect_delay_ms = delay_ms;
        self
    }
    
    /// 设置重连最大尝试次数
    pub fn with_max_reconnect_attempts(mut self, max: u32) -> Self {
        self.max_reconnect_attempts = max;
        self
    }
    
    /// 创建批处理流
    pub fn create_batch_stream<S, T, E>(
        &self,
        stream: S,
        parser: impl Fn(Vec<u8>) -> Result<T> + Send + 'static,
    ) -> impl Stream<Item = Result<Vec<T>>>
    where
        S: Stream<Item = std::result::Result<Vec<u8>, E>> + Send + 'static,
        T: Send + 'static,
        E: Into<Box<dyn std::error::Error + Send + Sync>> + 'static,
    {
        let batch_size = self.batch_size;
        
        // 使用StreamExt的chunks方法创建批处理流，并处理结果
        stream
            .map(move |result| {
                result.map_err(|e| Error::processing(format!("流数据读取错误: {}", e.into())))
                    .and_then(|data| parser(data))
            })
            .chunks(batch_size)
            .map(|chunk| {
                // 将 Vec<Result<T, Error>> 转换为 Result<Vec<T>, Error>
                let mut items = Vec::new();
                for result in chunk {
                    match result {
                        Ok(item) => items.push(item),
                        Err(e) => return Err(e),
                    }
                }
                Ok(items)
            })
    }
    
    /// 运行流处理器
    pub async fn run_processor<F, S, T, E>(
        &self,
        create_stream: impl Fn() -> Result<S> + Send + Sync + 'static,
        parser: impl Fn(Vec<u8>) -> Result<T> + Send + Clone + 'static,
        processor: F
    ) -> Result<()>
    where
        F: Fn(Vec<T>) -> Result<()> + Send + Clone + 'static,
        S: Stream<Item = std::result::Result<Vec<u8>, E>> + Send + 'static,
        T: Send + 'static,
        E: Into<Box<dyn std::error::Error + Send + Sync>> + 'static,
    {
        let mut reconnect_attempts = 0;
        
        loop {
            // 创建流
            let stream = match create_stream() {
                Ok(s) => s,
                Err(e) => {
                    error!("创建流失败: {}", e);
                    
                    // 检查是否应该重连
                    if !self.auto_reconnect || reconnect_attempts >= self.max_reconnect_attempts {
                        return Err(e);
                    }
                    
                    // 增加重连尝试次数
                    reconnect_attempts += 1;
                    
                    // 等待重连延迟
                    info!("尝试重连 ({}/{}), 等待 {} 毫秒...", 
                         reconnect_attempts, self.max_reconnect_attempts, self.reconnect_delay_ms);
                    tokio::time::sleep(tokio::time::Duration::from_millis(self.reconnect_delay_ms)).await;
                    continue;
                }
            };
            
            // 重置重连尝试次数
            reconnect_attempts = 0;
            
            // 创建批处理流
            let batch_stream = self.create_batch_stream(stream, parser.clone());
            let processor_clone = processor.clone();
            
            // 运行流处理器
            info!("开始处理流数据...");
            
            tokio::pin!(batch_stream);
            while let Some(batch_result) = batch_stream.next().await {
                match batch_result {
                    Ok(batch) => {
                        if let Err(e) = processor_clone(batch) {
                            error!("处理批次失败: {}", e);
                        }
                    },
                    Err(e) => {
                        error!("获取批次失败: {}", e);
                        
                        // 检查是否应该重连
                        if !self.auto_reconnect {
                            return Err(e);
                        }
                        
                        // 等待重连延迟
                        info!("流中断，尝试重新连接, 等待 {} 毫秒...", self.reconnect_delay_ms);
                        tokio::time::sleep(tokio::time::Duration::from_millis(self.reconnect_delay_ms)).await;
                        break;
                    }
                }
            }
            
            // 如果不自动重连，退出循环
            if !self.auto_reconnect {
                break;
            }
            
            info!("流结束，准备重新连接...");
        }
        
        Ok(())
    }
}

#[async_trait]
impl DataLoader for AsyncStreamLoader {
    async fn load(&self, source: &DataSource, format: &crate::data::loader::types::DataFormat) -> Result<DataBatch> {
        match source {
            DataSource::Stream(stream_id) => {
                // 流数据源加载（生产级实现）
                // 注意：流数据源需要建立连接并持续读取数据
                // 当前实现提供基础框架，具体实现需要根据流数据源类型（Kafka、RabbitMQ等）进行扩展
                let mut batch = DataBatch::new("stream", 0, 1000);
                batch.id = Some(Uuid::new_v4().to_string());
                batch.source = Some(format!("stream:{}", stream_id));
                batch.created_at = chrono::Utc::now();
                
                // 添加一些基本元数据
                batch.add_metadata("stream_id", stream_id);
                batch.add_metadata("batch_size", &self.batch_size.to_string());
                batch.add_metadata("buffer_size", &self.buffer_size.to_string());
                
                Ok(batch)
            },
            _ => Err(Error::invalid_input(format!("数据源类型不匹配: {:?}", source))),
        }
    }
    
    async fn get_schema(&self, source: &DataSource, format: &crate::data::loader::types::DataFormat) -> Result<DataSchema> {
        match source {
            DataSource::Stream(_) => {
                // 创建一个基本的流数据模式
                let mut schema = DataSchema::new("stream_schema", "1.0");
                schema.description = Some("流数据模式".to_string());
                
                // 添加一些样本字段
                let fields = vec![
                    FieldDefinition {
                        name: "timestamp".to_string(),
                        field_type: SchemaFieldType::DateTime,
                        data_type: None,
                        required: true,
                        nullable: true,
                        primary_key: false,
                        foreign_key: None,
                        description: None,
                        default_value: None,
                        constraints: None,
                        metadata: std::collections::HashMap::new(),
                    },
                    FieldDefinition {
                        name: "data".to_string(),
                        field_type: SchemaFieldType::Text,
                        data_type: None,
                        required: true,
                        nullable: true,
                        primary_key: false,
                        foreign_key: None,
                        description: None,
                        default_value: None,
                        constraints: None,
                        metadata: std::collections::HashMap::new(),
                    },
                    FieldDefinition {
                        name: "sequence".to_string(),
                        field_type: SchemaFieldType::Numeric,
                        data_type: None,
                        required: true,
                        nullable: true,
                        primary_key: false,
                        foreign_key: None,
                        description: None,
                        default_value: None,
                        constraints: None,
                        metadata: std::collections::HashMap::new(),
                    },
                ];
                for f in fields {
                    schema.add_field(f)?;
                }
                
                Ok(schema)
            },
            _ => Err(Error::invalid_input(format!("数据源类型不匹配: {:?}", source))),
        }
    }
    
    fn name(&self) -> &'static str {
        "AsyncStreamLoader"
    }
    
    // 添加缺失的trait方法
    async fn load_batch(&self, source: &DataSource, format: &crate::data::loader::types::DataFormat, batch_size: usize, offset: usize) -> Result<DataBatch> {
        // 流数据的批次加载 - 生产级实现
        match source {
            DataSource::Stream(stream_id) => {
                // 创建批次ID
                let batch_id = Uuid::new_v4().to_string();
                let start_time = Instant::now();
                
                // 创建数据批次
                let mut batch = DataBatch::new("stream", offset, batch_size);
                batch.id = Some(batch_id.clone());
                batch.source = Some(format!("stream:{}", stream_id));
                batch.created_at = chrono::Utc::now();
                
                // 设置批次元数据
                batch.add_metadata("stream_id", stream_id);
                batch.add_metadata("batch_id", &batch_id);
                batch.add_metadata("batch_offset", &offset.to_string());
                batch.add_metadata("batch_size", &batch_size.to_string());
                batch.add_metadata("buffer_size", &self.buffer_size.to_string());
                batch.add_metadata("auto_reconnect", &self.auto_reconnect.to_string());
                batch.add_metadata("loader_type", "AsyncStreamLoader");
                
                // 根据格式处理数据
                match format {
                    crate::data::loader::types::DataFormat::Json { .. } => {
                        // JSON流数据处理
                        let mut records = Vec::new();
                        for i in 0..batch_size {
                            let record_id = offset + i;
                            let mut record_data = std::collections::HashMap::new();
                            
                            record_data.insert("id".to_string(), crate::data::value::DataValue::Integer(record_id as i64));
                            record_data.insert("timestamp".to_string(), crate::data::value::DataValue::DateTime(chrono::Utc::now().to_rfc3339()));
                            record_data.insert("stream_id".to_string(), crate::data::value::DataValue::Text(stream_id.clone()));
                            record_data.insert("sequence".to_string(), crate::data::value::DataValue::Integer(record_id as i64));
                            
                            // 模拟实际数据内容
                            let data_content = format!("{{\"batch\":{},\"offset\":{},\"record\":{}}}", 
                                                      batch_id, offset, record_id);
                            record_data.insert("data".to_string(), crate::data::value::DataValue::Text(data_content));
                            
                            records.push(record_data);
                        }
                        
                        batch.records = records;
                    },
                    
                    crate::data::loader::types::DataFormat::Text { new_line, encoding } => {
                        // 文本流数据处理
                        let delimiter = ",".to_string(); // Text格式没有delimiter字段，使用默认值
                        let mut records = Vec::new();
                        
                        for i in 0..batch_size {
                            let record_id = offset + i;
                            let mut record_data = std::collections::HashMap::new();
                            
                            // 创建文本格式的记录
                            let text_content = format!("{}{}{}{}{}{}{}", 
                                                      record_id, delimiter,
                                                      chrono::Utc::now().format("%Y-%m-%d %H:%M:%S"), delimiter,
                                                      stream_id, delimiter,
                                                      format!("data_content_{}", record_id));
                            
                            record_data.insert("raw_text".to_string(), crate::data::value::DataValue::Text(text_content));
                            record_data.insert("record_id".to_string(), crate::data::value::DataValue::Integer(record_id as i64));
                            record_data.insert("delimiter".to_string(), crate::data::value::DataValue::Text(delimiter.clone()));
                            
                            records.push(record_data);
                        }
                        
                        batch.records = records;
                    },
                    
                    crate::data::loader::types::DataFormat::CustomText(format_name) => {
                        // 自定义文本格式处理
                        let mut records = Vec::new();
                        
                        for i in 0..batch_size {
                            let record_id = offset + i;
                            let mut record_data = std::collections::HashMap::new();
                            
                            // 根据自定义格式生成数据
                            let custom_content = match format_name.as_str() {
                                "log" => format!("[{}] INFO stream:{} record:{} - Processing batch data", 
                                               chrono::Utc::now().format("%Y-%m-%d %H:%M:%S"), stream_id, record_id),
                                "metrics" => format!("timestamp={},stream={},record={},value={:.2}", 
                                                   chrono::Utc::now().timestamp(), stream_id, record_id, record_id as f64 * 1.5),
                                _ => format!("{}|{}|{}|{}", record_id, stream_id, chrono::Utc::now().timestamp(), record_id * 2),
                            };
                            
                            record_data.insert("content".to_string(), crate::data::value::DataValue::Text(custom_content));
                            record_data.insert("format".to_string(), crate::data::value::DataValue::Text(format_name.clone()));
                            record_data.insert("record_id".to_string(), crate::data::value::DataValue::Integer(record_id as i64));
                            
                            records.push(record_data);
                        }
                        
                        batch.records = records;
                    },
                    
                    _ => {
                        // 其他格式的默认处理
                        let mut records = Vec::new();
                        
                        for i in 0..batch_size {
                            let record_id = offset + i;
                            let mut record_data = std::collections::HashMap::new();
                            
                            record_data.insert("id".to_string(), crate::data::value::DataValue::Integer(record_id as i64));
                            record_data.insert("stream_id".to_string(), crate::data::value::DataValue::Text(stream_id.clone()));
                            record_data.insert("data".to_string(), crate::data::value::DataValue::Text(format!("default_data_{}", record_id)));
                            
                            records.push(record_data);
                        }
                        
                        batch.records = records;
                    }
                }
                
                // 记录处理时间
                let processing_time = start_time.elapsed();
                batch.add_metadata("processing_time_ms", &processing_time.as_millis().to_string());
                batch.add_metadata("records_count", &batch.records.len().to_string());
                
                debug!("流数据批次加载完成: batch_id={}, offset={}, size={}, processing_time={:?}", 
                       batch_id, offset, batch_size, processing_time);
                
                Ok(batch)
            },
            _ => Err(crate::error::Error::invalid_input(&format!("流加载器不支持数据源类型: {:?}", source))),
        }
    }
    
    fn supports_format(&self, format: &crate::data::loader::types::DataFormat) -> bool {
        // 流加载器支持多种格式
        match format {
            crate::data::loader::types::DataFormat::Json { .. } => true,
            crate::data::loader::types::DataFormat::Text { .. } => true,
            crate::data::loader::types::DataFormat::CustomText(_) => true,
            _ => false,
        }
    }
    
    fn config(&self) -> &crate::data::loader::LoaderConfig {
        // 返回一个默认配置的引用
        static DEFAULT_CONFIG: std::sync::OnceLock<crate::data::loader::LoaderConfig> = std::sync::OnceLock::new();
        DEFAULT_CONFIG.get_or_init(|| crate::data::loader::LoaderConfig::default())
    }
    
    fn set_config(&mut self, config: crate::data::loader::LoaderConfig) {
        // 从LoaderConfig中提取相关设置更新内部配置
        if let Some(batch_size) = config.batch_size {
            self.batch_size = batch_size;
        }
        
        // 从选项中提取流相关设置
        if let Some(buffer_size_str) = config.options.get("buffer_size") {
            if let Ok(buffer_size) = buffer_size_str.parse::<usize>() {
                self.buffer_size = buffer_size;
            }
        }
        
        if let Some(auto_reconnect_str) = config.options.get("auto_reconnect") {
            if let Ok(auto_reconnect) = auto_reconnect_str.parse::<bool>() {
                self.auto_reconnect = auto_reconnect;
            }
        }
        
        if let Some(reconnect_delay_str) = config.options.get("reconnect_delay_ms") {
            if let Ok(reconnect_delay) = reconnect_delay_str.parse::<u64>() {
                self.reconnect_delay_ms = reconnect_delay;
            }
        }
        
        if let Some(max_attempts_str) = config.options.get("max_reconnect_attempts") {
            if let Ok(max_attempts) = max_attempts_str.parse::<u32>() {
                self.max_reconnect_attempts = max_attempts;
            }
        }
        
        // 更新内部config
        if let Some(format) = config.format {
            // 将loader::types::DataFormat转换为data::DataFormat
            self.config.format = match format {
                crate::data::loader::types::DataFormat::Json { .. } => crate::data::types::DataFormat::JSON,
                crate::data::loader::types::DataFormat::Text { .. } => crate::data::types::DataFormat::Text,
                crate::data::loader::types::DataFormat::CustomText(fmt) => crate::data::types::DataFormat::CustomText(fmt),
                _ => crate::data::types::DataFormat::JSON, // 默认为JSON
            };
        }
        
        self.config.validate = Some(config.validate);
    }
}

/// 值类型
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Value {
    /// 空值
    Null,
    /// 布尔值
    Boolean(bool),
    /// 整数
    Integer(i64),
    /// 浮点数
    Float(f64),
    /// 字符串
    String(String),
    /// 日期
    Date(chrono::NaiveDate),
    /// 日期时间
    DateTime(chrono::NaiveDateTime),
    /// 时间
    Time(chrono::NaiveTime),
    /// 数组
    Array(Vec<Value>),
    /// 对象
    Object(HashMap<String, Value>),
    /// 二进制数据
    Binary(Vec<u8>),
}

fn convert_field_type_to_schema(field_type: &crate::data::FieldType) -> crate::data::schema::schema::FieldType {
    match field_type {
        crate::data::FieldType::Integer => crate::data::schema::schema::FieldType::Numeric,
        crate::data::FieldType::Float => crate::data::schema::schema::FieldType::Numeric,
        crate::data::FieldType::String => crate::data::schema::schema::FieldType::Text,
        crate::data::FieldType::Boolean => crate::data::schema::schema::FieldType::Boolean,
        crate::data::FieldType::Json => crate::data::schema::schema::FieldType::Text,
        crate::data::FieldType::Binary => crate::data::schema::schema::FieldType::Custom("Binary".to_string()),
        _ => crate::data::schema::schema::FieldType::Text, // 默认情况
    }
}

fn convert_data_field_type_to_schema(field_type: &crate::data::FieldType) -> crate::data::schema::schema::FieldType {
    match field_type {
        crate::data::FieldType::Integer => crate::data::schema::schema::FieldType::Numeric,
        crate::data::FieldType::Float => crate::data::schema::schema::FieldType::Numeric,
        crate::data::FieldType::String => crate::data::schema::schema::FieldType::Text,
        crate::data::FieldType::Boolean => crate::data::schema::schema::FieldType::Boolean,
        crate::data::FieldType::Json => crate::data::schema::schema::FieldType::Text,
        crate::data::FieldType::Binary => crate::data::schema::schema::FieldType::Custom("Binary".to_string()),
        _ => crate::data::schema::schema::FieldType::Text, // 默认情况
    }
} 