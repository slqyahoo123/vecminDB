// 加载器工厂实现

use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use lazy_static::lazy_static;
use log::{debug, error, info, warn};

use crate::data::{DataBatch, DataConfig, DataSchema};
use crate::data::loader::{DataLoader, DataSource, types::DataFormat};
use crate::error::{Error, Result};
use crate::data::loader::common::DataCache;
use crate::data::value::DataValue;

/// 为 HashMap<String, DataValue> 提供扩展方法
trait RecordMapExt {
    fn has_field(&self, name: &str) -> bool;
    fn set_field(&mut self, name: &str, value: impl Into<DataValue>) -> Result<()>;
}

impl RecordMapExt for HashMap<String, DataValue> {
    fn has_field(&self, name: &str) -> bool {
        self.contains_key(name)
    }
    
    fn set_field(&mut self, name: &str, value: impl Into<DataValue>) -> Result<()> {
        self.insert(name.to_string(), value.into());
        Ok(())
    }
}

/// 数据加载器工厂
pub struct LoaderFactory {
    registry: HashMap<String, Box<dyn Fn(Option<DataConfig>) -> Result<Box<dyn DataLoader>> + Send + Sync>>,
    cache: Option<Box<dyn DataCache>>,
    schema_registry: HashMap<String, Arc<DataSchema>>,
    batch_transformers: Vec<Arc<dyn Fn(&mut DataBatch) -> Result<()> + Send + Sync>>,
}

impl LoaderFactory {
    /// 创建新的加载器工厂
    pub fn new() -> Self {
        let mut factory = Self {
            registry: HashMap::new(),
            cache: None,
            schema_registry: HashMap::new(),
            batch_transformers: Vec::new(),
        };
        
        // 注册默认加载器
        factory.register("file", |config| {
            let config = config.unwrap_or_default();
            Ok(Box::new(crate::data::loader::file::FileDataLoader::new(config)) as Box<dyn DataLoader>)
        });
        
        factory.register("database", |_| {
            // 将 ImportedDatabaseConfig 转换为 database::DatabaseConfig
            let imported_config = crate::data::ImportedDatabaseConfig::default();
            let local_config = crate::data::loader::database::DatabaseLoader::convert_to_local_config(&imported_config);
            Ok(Box::new(crate::data::loader::database::DatabaseLoader::new(local_config)) as Box<dyn DataLoader>)
        });
        
        factory.register("stream", |config| {
            let config = config.unwrap_or_default();
            Ok(Box::new(crate::data::loader::stream::StreamDataLoader::new(config)) as Box<dyn DataLoader>)
        });
        
        factory.register("binary", |_| {
            use crate::data::loader::binary::{CustomBinaryLoader, Endianness, CustomBinaryFormat};
            Ok(Box::new(CustomBinaryLoader::new(
                Endianness::Little,
                16,
                CustomBinaryFormat::RecordBatch
            )) as Box<dyn DataLoader>)
        });
        
        factory.register("common", |config| {
            let config = config.unwrap_or_default();
            let result = crate::data::loader::common::CommonDataLoader::new(config)?;
            Ok(Box::new(result) as Box<dyn DataLoader>)
        });
        
        factory
    }
    
    /// 注册加载器创建函数
    pub fn register<F>(&mut self, name: &str, creator: F)
    where
        F: Fn(Option<DataConfig>) -> Result<Box<dyn DataLoader>> + 'static + Send + Sync,
    {
        self.registry.insert(name.to_string(), Box::new(creator));
    }
    
    /// 创建加载器实例
    pub fn create(&self, source_type: &str, config: Option<DataConfig>) -> Result<Box<dyn DataLoader>> {
        if let Some(creator) = self.registry.get(source_type) {
            creator(config)
        } else {
            Err(Error::invalid_argument(format!("不支持的数据源类型: {}", source_type)))
        }
    }
    
    /// 获取已注册的加载器类型列表
    pub fn get_registered_types(&self) -> Vec<String> {
        self.registry.keys().cloned().collect()
    }
    
    /// 检查是否支持指定类型
    pub fn supports_type(&self, source_type: &str) -> bool {
        self.registry.contains_key(source_type)
    }
    
    /// 注册所有默认加载器
    pub fn register_defaults(&mut self) {
        // 文件加载器
        self.register("csv", |config| {
            let config = config.unwrap_or_default();
            Ok(Box::new(crate::data::loader::file::FileDataLoader::new(config)) as Box<dyn DataLoader>)
        });
        
        self.register("json", |config| {
            let config = config.unwrap_or_default();
            Ok(Box::new(crate::data::loader::file::FileDataLoader::new(config)) as Box<dyn DataLoader>)
        });
        
        self.register("parquet", |config| {
            let config = config.unwrap_or_default();
            Ok(Box::new(crate::data::loader::file::FileDataLoader::new(config)) as Box<dyn DataLoader>)
        });
        
        // 更多默认加载器注册...
    }
    
    /// 设置数据缓存系统
    pub fn with_cache(&mut self, cache: Box<dyn DataCache>) -> &mut Self {
        self.cache = Some(cache);
        self
    }
    
    /// 获取数据缓存系统
    pub fn get_cache(&self) -> Option<&dyn DataCache> {
        self.cache.as_deref()
    }
    
    /// 注册数据模式
    pub fn register_schema(&mut self, name: &str, schema: DataSchema) -> &mut Self {
        self.schema_registry.insert(name.to_string(), Arc::new(schema));
        self
    }
    
    /// 获取已注册的数据模式
    pub fn get_schema(&self, name: &str) -> Option<Arc<DataSchema>> {
        self.schema_registry.get(name).cloned()
    }
    
    /// 添加批次转换器
    pub fn add_batch_transformer<F>(&mut self, transformer: F) -> &mut Self 
    where
        F: Fn(&mut DataBatch) -> Result<()> + Send + Sync + 'static
    {
        self.batch_transformers.push(Arc::new(transformer));
        self
    }
    
    /// 应用所有转换器到数据批次
    pub fn transform_batch(&self, batch: &mut DataBatch) -> Result<()> {
        for transformer in &self.batch_transformers {
            transformer(batch)?;
        }
        Ok(())
    }
}

impl Default for LoaderFactory {
    fn default() -> Self {
        Self::new()
    }
}

// 创建一个全局工厂实例
lazy_static! {
    pub(crate) static ref GLOBAL_LOADER_FACTORY: Mutex<LoaderFactory> = Mutex::new(LoaderFactory::new());
}

/// 初始化加载器工厂
pub fn initialize(
    default_loaders: HashMap<String, String>,
    cache: Box<dyn DataCache>,
    _connection_pool: Arc<dyn crate::data::loader::common::DataSourceConnector>
) -> Result<()> {
    if let Ok(mut factory) = GLOBAL_LOADER_FACTORY.lock() {
        // 设置缓存
        factory.with_cache(cache);
        
        // 注册默认加载器映射
        for (format, loader_type) in default_loaders {
            debug!("注册默认加载器映射: {} -> {}", format, loader_type);
        }
        
        // 注册内置模式
        use crate::data::schema::schema::{FieldDefinition, FieldType};
        let mut common_schema = DataSchema::new("common_schema", "1.0");
        let fields = vec![
            FieldDefinition {
                name: "id".to_string(),
                field_type: FieldType::Text,
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
                name: "name".to_string(),
                field_type: FieldType::Text,
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
                field_type: FieldType::Numeric,
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
                name: "timestamp".to_string(),
                field_type: FieldType::DateTime,
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
        ];
        for f in fields {
            common_schema.add_field(f)?;
        }
        
        factory.register_schema("common", common_schema);
        
        // 添加标准批次转换器
        factory.add_batch_transformer(|batch| {
            // 添加加载时间戳元数据
            batch.add_metadata("loaded_at", &chrono::Utc::now().to_rfc3339());
            
            // 对所有记录应用通用转换
            for record in &mut batch.records {
                // 例如，确保所有记录都有id字段
                if !record.has_field("id") {
                    record.set_field("id", uuid::Uuid::new_v4().to_string())?;
                }
            }
            
            info!("应用通用批次转换器处理了{}条记录", batch.size);
            Ok(())
        });
        
        Ok(())
    } else {
        Err(Error::internal("无法访问全局加载器工厂"))
    }
}

/// 创建数据加载器的工厂函数（旧版本，现在更建议使用LoaderFactory直接创建）
#[deprecated(since = "0.2.0", note = "请使用LoaderFactory直接创建")]
pub fn create_loader(source_type: &str, config: Option<DataConfig>) -> Result<Box<dyn DataLoader>> {
    if let Ok(factory) = GLOBAL_LOADER_FACTORY.lock() {
        factory.create(source_type, config)
    } else {
        Err(Error::internal("无法访问全局加载器工厂"))
    }
}

/// 注册自定义加载器到全局工厂
pub fn register_loader<F>(name: &str, creator: F) -> Result<()>
where
    F: Fn(Option<DataConfig>) -> Result<Box<dyn DataLoader>> + 'static + Send + Sync,
{
    if let Ok(mut factory) = GLOBAL_LOADER_FACTORY.lock() {
        factory.register(name, creator);
        Ok(())
    } else {
        Err(Error::internal("无法访问全局加载器工厂"))
    }
}

/// 创建加载器v2版本，包含更多高级功能
pub fn create_loader_v2(source: &str) -> Result<Box<dyn DataLoader>> {
    debug!("正在创建数据加载器: {}", source);
    
    // 解析源字符串，提取类型和路径
    let config = match source.contains(":") {
        true => {
            let parts: Vec<&str> = source.splitn(2, ':').collect();
            let source_type = parts[0];
            let source_path = parts[1];
            
            // 使用Arc共享配置
            let shared_config = Arc::new(DataConfig {
                path: Some(source_path.to_string()),
                ..DataConfig::default()
            });
            
            info!("使用共享配置创建加载器: {}", source_type);
            Some((*shared_config).clone())
        },
        false => {
            warn!("未提供详细的源信息，使用默认配置");
            None
        }
    };
    
    // 获取全局工厂并创建加载器
    if let Ok(factory) = GLOBAL_LOADER_FACTORY.lock() {
        let loader_type = source.split(':').next().unwrap_or(source);
        match factory.create(loader_type, config) {
            Ok(loader) => {
                debug!("成功创建加载器: {}", loader.name());
                Ok(loader)
            },
            Err(e) => {
                error!("创建加载器失败: {}", e);
                Err(e)
            }
        }
    } else {
        error!("无法访问全局加载器工厂，锁可能被污染");
        Err(Error::internal("无法访问全局加载器工厂"))
    }
}

/// 获取所有已注册的加载器类型
pub fn get_registered_loader_types() -> Result<Vec<String>> {
    if let Ok(factory) = GLOBAL_LOADER_FACTORY.lock() {
        Ok(factory.get_registered_types())
    } else {
        Err(Error::internal("无法访问全局加载器工厂"))
    }
}

/// 预处理数据批次
pub fn process_batch(batch: &mut DataBatch) -> Result<()> {
    if let Ok(factory) = GLOBAL_LOADER_FACTORY.lock() {
        factory.transform_batch(batch)
    } else {
        Err(Error::internal("无法访问全局加载器工厂"))
    }
}

/// 根据数据源和格式创建合适的加载器
pub fn create_loader_for_source(source: &DataSource, format: &DataFormat) -> Result<Box<dyn DataLoader>> {
    use crate::data::types::DataFormat as CoreDataFormat;
    
    let loader_type = match source {
        DataSource::File(_) => "file",
        DataSource::Database(_) => "database",
        DataSource::Stream(_) => "stream",
        DataSource::Custom(type_name, _) => type_name,
        DataSource::Memory(_) => "memory",
    };
    
    // 创建基础配置
    let mut config = DataConfig::default();
    
    // 根据数据源类型设置配置
    match source {
        DataSource::File(path) => {
            config.path = Some(path.clone());
        },
        DataSource::Database(db_config) => {
            // 将数据库配置信息存储到 extra_params 中
            config.add_parameter("connection_string", db_config.connection_string.clone());
            if let Some(query) = &db_config.query {
                config.add_parameter("query", query.clone());
            }
            if let Some(username) = &db_config.username {
                config.add_parameter("username", username.clone());
            }
            if let Some(password) = &db_config.password {
                config.add_parameter("password", password.clone());
            }
            if let Some(database) = &db_config.database {
                config.add_parameter("database", database.clone());
            }
            if let Some(table) = &db_config.table {
                config.add_parameter("table", table.clone());
            }
            // 存储数据库类型
            let db_type_str = format!("{:?}", db_config.db_type);
            config.add_parameter("db_type", db_type_str);
        },
        DataSource::Stream(stream_id) => {
            // 将流ID存储到 extra_params 中
            config.add_parameter("stream_id", stream_id.clone());
        },
        DataSource::Custom(_, params) => {
            for (key, value) in params {
                config.add_parameter(key, value);
            }
        },
        DataSource::Memory(_) => {
            // 内存数据源使用默认配置
        },
    }
    
    // 将 data::loader::types::DataFormat 转换为 data::types::DataFormat
    let core_format = match format {
        DataFormat::Csv { .. } => CoreDataFormat::CSV,
        DataFormat::Json { .. } => CoreDataFormat::JSON,
        DataFormat::Parquet { .. } => CoreDataFormat::Parquet,
        DataFormat::Avro { .. } => CoreDataFormat::Avro,
        DataFormat::Excel { .. } => CoreDataFormat::Custom("excel".to_string()),
        DataFormat::Text { .. } => CoreDataFormat::Text,
        DataFormat::Tensor { .. } => CoreDataFormat::Tensor,
        DataFormat::CustomText(format_name) => CoreDataFormat::CustomText(format_name.clone()),
        DataFormat::CustomBinary(format_name) => CoreDataFormat::Custom(format_name.clone()),
    };
    
    // 设置格式
    config.format = core_format;
    
    // 创建加载器
    if let Ok(factory) = GLOBAL_LOADER_FACTORY.lock() {
        factory.create(loader_type, Some(config))
    } else {
        Err(Error::internal("无法访问全局加载器工厂"))
    }
} 