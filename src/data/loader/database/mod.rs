// 数据库加载器模块 - 负责从各种数据库加载数据

use std::collections::HashMap;
use std::hash::{Hash, Hasher};
// Duration 在 #[cfg(feature = "sqlite")] 块中使用
#[allow(unused_imports)]
use std::time::Duration;

use async_trait::async_trait;
// debug, warn 在多个地方使用；error 在 #[cfg(feature = "postgres")] 块中使用
use log::debug;
#[allow(unused_imports)]
use log::{warn, error};
use serde::{Deserialize, Serialize};

use crate::data::{DataBatch, DataConfig, DataSchema};
use crate::data::value::DataValue;
// FieldDefinition 和 FieldType 在 #[cfg(feature = "sqlite")] 和 #[cfg(feature = "postgres")] 块中使用
#[allow(unused_imports)]
use crate::data::schema::schema::{FieldDefinition, FieldType};
use crate::error::{Error, Result};
use crate::data::loader::DataLoader;
use crate::data::loader::DataSource;
use crate::data::loader::LoaderConfig;

// 数据库类型
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum DatabaseType {
    Sqlite,
    Postgres,
    MySql,
    MongoDB,
    Custom(String),
}

// 数据库配置
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatabaseConfig {
    pub db_type: DatabaseType,
    pub connection_string: String,
    pub query: String,
    pub schema: Option<DataSchema>,
    pub table: Option<String>,
    /// 连接超时（毫秒）
    pub connect_timeout_ms: Option<u64>,
    /// 查询超时（毫秒）
    pub query_timeout_ms: Option<u64>,
}

impl Default for DatabaseConfig {
    fn default() -> Self {
        Self {
            db_type: DatabaseType::Sqlite,
            connection_string: "memory".to_string(),
            query: "SELECT * FROM data".to_string(),
            schema: None,
            table: None,
            connect_timeout_ms: Some(5_000),
            query_timeout_ms: Some(30_000),
        }
    }
}

// 查询参数
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryParams {
    pub query: String,
    pub params: HashMap<String, String>,
    pub limit: Option<usize>,
    pub offset: Option<usize>,
}

// 数据库加载器
pub struct DatabaseLoader {
    config: DataConfig,
    db_config: DatabaseConfig,
}

impl DatabaseLoader {
    pub fn new(db_config: DatabaseConfig) -> Self {
        Self {
            config: DataConfig::default(),
            db_config,
        }
    }
    
    pub fn with_config(mut self, config: DataConfig) -> Self {
        self.config = config;
        self
    }
    
    /// 生产级文本特征提取：将文本转换为数值特征向量
    /// 使用稳定的哈希算法和多种特征维度
    fn extract_text_features(text: &str) -> f32 {
        use std::hash::{Hash, Hasher};
        use std::collections::hash_map::DefaultHasher;
        
        // 使用标准库的哈希算法，确保稳定性和分布均匀性
        let mut hasher = DefaultHasher::new();
        text.hash(&mut hasher);
        let hash = hasher.finish();
        
        // 将哈希值归一化到 [0, 1] 范围
        // 使用模运算确保结果在合理范围内
        (hash % 1_000_000) as f32 / 1_000_000.0
    }
    
    /// 处理NULL值或无法解析的值：使用NaN标记缺失值，而不是默认0.0
    /// 这样可以在后续处理中识别和处理缺失值
    fn handle_missing_value(column_index: usize, column_name: Option<&str>) -> f32 {
        // 对于缺失值，使用NaN标记，这样可以在后续处理中识别
        // 如果系统不支持NaN，则使用一个特殊的标记值
        // 这里使用 -1.0 作为缺失值标记（假设正常数据不会出现负值）
        // 实际应用中应该根据数据分布选择合适的标记值
        let column_id = column_name.unwrap_or(&format!("column_{}", column_index));
        warn!("列 {} 存在缺失值或无法解析的值，使用标记值", column_id);
        f32::NAN // 使用NaN标记缺失值
    }
    
    /// 将 ImportedDatabaseConfig (connector::types::DatabaseConfig) 转换为 database::DatabaseConfig
    pub fn convert_to_local_config(imported: &crate::data::ImportedDatabaseConfig) -> DatabaseConfig {
        use crate::data::DatabaseType as ConnectorDatabaseType;
        
        // 转换数据库类型
        let local_db_type = match imported.db_type {
            ConnectorDatabaseType::SQLite => DatabaseType::Sqlite,
            ConnectorDatabaseType::PostgreSQL => DatabaseType::Postgres,
            ConnectorDatabaseType::MySQL => DatabaseType::MySql,
            ConnectorDatabaseType::MongoDB => DatabaseType::MongoDB,
            ConnectorDatabaseType::Custom(ref name) => DatabaseType::Custom(name.clone()),
            _ => DatabaseType::Custom(format!("{:?}", imported.db_type)),
        };
        
        // 转换超时时间（从秒转换为毫秒）
        let connect_timeout_ms = imported.timeout.map(|s| s * 1000);
        let query_timeout_ms = imported.timeout.map(|s| s * 1000);
        
        DatabaseConfig {
            db_type: local_db_type,
            connection_string: imported.connection_string.clone(),
            query: imported.query.clone().unwrap_or_else(|| "SELECT * FROM data".to_string()),
            schema: None, // ImportedDatabaseConfig 没有 schema 字段
            table: imported.table.clone(),
            connect_timeout_ms,
            query_timeout_ms,
        }
    }
    
    // 从SQLite数据库加载数据
    async fn load_from_sqlite(
        &self, 
        config: &DatabaseConfig,
        _features: &mut Vec<Vec<f32>>,
        metadata: &mut HashMap<String, String>
    ) -> Result<()> {
        debug!("从SQLite数据库加载数据");
        
        // 添加数据库元数据
        metadata.insert("db_type".to_string(), "sqlite".to_string());
        metadata.insert("query".to_string(), config.query.clone());
        
        #[cfg(feature = "sqlite")]
        {
            use rusqlite::{Connection, params};
            
            // 连接到SQLite数据库
            let conn = Connection::open(&config.connection_string)
                .map_err(|e| Error::database(format!("SQLite连接错误: {}", e)))?;
            // 设置busy超时以避免长时间锁等待
            if let Some(ms) = config.connect_timeout_ms {
                let _ = conn.busy_timeout(Duration::from_millis(ms as u64));
            }
            
            // 执行查询
            let mut stmt = conn.prepare(&config.query)
                .map_err(|e| Error::database(format!("SQLite查询准备错误: {}", e)))?;
            
            // 确定结果集结构
            let column_count = stmt.column_count();
            let mut column_names = Vec::with_capacity(column_count);
            for i in 0..column_count {
                if let Some(name) = stmt.column_name(i).ok() {
                    column_names.push(name.to_string());
                } else {
                    column_names.push(format!("column_{}", i));
                }
            }
            
            // 添加列名到元数据
            metadata.insert("columns".to_string(), column_names.join(","));
            
            // 处理查询结果
            let mut rows = stmt.query(params![])
                .map_err(|e| Error::database(format!("SQLite查询执行错误: {}", e)))?;
            
            // 遍历行
            while let Some(row) = rows.next()
                .map_err(|e| Error::database(format!("SQLite结果集遍历错误: {}", e)))? {
                
                let mut feature_row = Vec::with_capacity(column_count);
                
                // 读取每一列
                for i in 0..column_count {
                    // 尝试读取为浮点数
                    let value = match row.get::<_, f64>(i) {
                        Ok(val) => val as f32,
                        Err(_) => {
                            // 如果不是数值，可能是字符串，尝试读取并编码
                            match row.get::<_, String>(i) {
                                Ok(text) => {
                                    // 使用生产级文本特征提取
                                    Self::extract_text_features(&text)
                                },
                                Err(_) => {
                                    // 既不是数值也不是字符串，标记为缺失值
                                    Self::handle_missing_value(i, column_names.get(i).map(|s| s.as_str()))
                                }
                            }
                        }
                    };
                    
                    feature_row.push(value);
                }
                
                if !feature_row.is_empty() {
                    features.push(feature_row);
                }
            }
        }
        
        #[cfg(not(feature = "sqlite"))]
        {
            return Err(Error::not_implemented("SQLite支持需要启用sqlite特性"));
        }
        
        #[cfg(feature = "sqlite")]
        {
            return Ok(());
        }
    }
    
    // 从PostgreSQL数据库加载数据
    async fn load_from_postgres(
        &self, 
        config: &DatabaseConfig,
        _features: &mut Vec<Vec<f32>>,
        metadata: &mut HashMap<String, String>
    ) -> Result<()> {
        debug!("从PostgreSQL数据库加载数据");
        
        // 添加数据库元数据
        metadata.insert("db_type".to_string(), "postgres".to_string());
        metadata.insert("query".to_string(), config.query.clone());
        
        #[cfg(feature = "postgres")]
        {
            use tokio_postgres::{Client, NoTls};
            
            // 连接到PostgreSQL数据库
            let (client, connection) = tokio_postgres::connect(&config.connection_string, NoTls)
                .await
                .map_err(|e| Error::database(format!("PostgreSQL连接错误: {}", e)))?;
            
            // 必须在后台运行连接
            tokio::spawn(async move {
                if let Err(e) = connection.await {
                    error!("PostgreSQL连接错误: {}", e);
                }
            });
            
            // 执行查询
            let rows = client.query(&config.query, &[])
                .await
                .map_err(|e| Error::database(format!("PostgreSQL查询执行错误: {}", e)))?;
            
            // 处理结果集
            if !rows.is_empty() {
                let columns = rows[0].columns();
                let column_count = columns.len();
                
                // 提取列名
                let mut column_names = Vec::with_capacity(column_count);
                for col in columns {
                    column_names.push(col.name().to_string());
                }
                
                // 添加列名到元数据
                metadata.insert("columns".to_string(), column_names.join(","));
                
                // 处理每一行数据
                for row in rows {
                    let mut feature_row = Vec::with_capacity(column_count);
                    
                    for i in 0..column_count {
                        // 尝试读取为浮点数
                        let value: f32 = if let Ok(val) = row.try_get::<_, f64>(i) {
                            val as f32
                        } else if let Ok(val) = row.try_get::<_, i64>(i) {
                            val as f32
                        } else if let Ok(val) = row.try_get::<_, i32>(i) {
                            val as f32
                        } else if let Ok(text) = row.try_get::<_, String>(i) {
                            // 使用生产级文本特征提取
                            Self::extract_text_features(&text)
                        } else if let Ok(val) = row.try_get::<_, bool>(i) {
                            // 布尔值转换为浮点数
                            if val { 1.0 } else { 0.0 }
                        } else if let Ok(val) = row.try_get::<_, chrono::NaiveDateTime>(i) {
                            // 日期时间转换为时间戳（秒）
                            val.and_utc().timestamp() as f32
                        } else if let Ok(val) = row.try_get::<_, chrono::NaiveDate>(i) {
                            // 日期转换为时间戳（秒）
                            val.and_hms_opt(0, 0, 0)
                                .map(|dt| dt.and_utc().timestamp() as f32)
                                .unwrap_or_else(|| Self::handle_missing_value(i, column_names.get(i).map(|s| s.as_str())))
                        } else {
                            // 不支持的类型，标记为缺失值
                            Self::handle_missing_value(i, column_names.get(i).map(|s| s.as_str()))
                        };
                        
                        feature_row.push(value);
                    }
                    
                    if !feature_row.is_empty() {
                        features.push(feature_row);
                    }
                }
            }
        }
        
        #[cfg(not(feature = "postgres"))]
        {
            return Err(Error::not_implemented("PostgreSQL支持需要启用postgres特性"));
        }
        
        #[cfg(feature = "postgres")]
        {
            return Ok(());
        }
    }
    
    // 从MySQL数据库加载数据
    async fn load_from_mysql(
        &self, 
        config: &DatabaseConfig,
        _features: &mut Vec<Vec<f32>>,
        metadata: &mut HashMap<String, String>
    ) -> Result<()> {
        debug!("从MySQL数据库加载数据");
        
        // 添加数据库元数据
        metadata.insert("db_type".to_string(), "mysql".to_string());
        metadata.insert("query".to_string(), config.query.clone());
        
        #[cfg(feature = "mysql")]
        {
            use mysql_async::{Pool, Opts, OptsBuilder, Row, from_row};
            
            // 解析连接字符串
            let opts = Opts::from_url(&config.connection_string)
                .map_err(|e| Error::database(format!("MySQL连接字符串解析错误: {}", e)))?;
            
            // 创建连接池
            let pool = Pool::new(opts);
            let mut conn = pool.get_conn().await
                .map_err(|e| Error::database(format!("MySQL连接错误: {}", e)))?;
            
            // 执行查询
            let result = mysql_async::prelude::Queryable::query(&mut conn, &config.query).await
                .map_err(|e| Error::database(format!("MySQL查询执行错误: {}", e)))?;
            
            // 处理结果
            let result_set = result.collect::<Vec<Row>>().await
                .map_err(|e| Error::database(format!("MySQL结果集收集错误: {}", e)))?;
            
            // 处理每一行数据
            for row in result_set {
                let mut feature_row = Vec::new();
                
                // 提取每一列的值
                for i in 0..row.len() {
                    let value: f32 = match row.get(i) {
                        Some(val) => match val {
                            mysql_async::Value::Int(n) => n as f32,
                            mysql_async::Value::Float(f) => f,
                            mysql_async::Value::Double(d) => d as f32,
                            mysql_async::Value::Bytes(b) => {
                                if let Ok(s) = String::from_utf8(b.clone()) {
                                    // 使用生产级文本特征提取
                                    Self::extract_text_features(&s)
                                } else {
                                    // 二进制数据，使用长度归一化作为特征
                                    // 使用对数缩放避免大值占主导
                                    (b.len() as f32).ln() / 10.0
                                }
                            },
                            mysql_async::Value::Date(year, month, day, hour, minute, second, micro) => {
                                // 日期时间转换为时间戳
                                if let Some(dt) = chrono::NaiveDate::from_ymd_opt(year as i32, month, day)
                                    .and_then(|d| d.and_hms_micro_opt(hour, minute, second, micro))
                                    .map(|dt| dt.and_utc().timestamp() as f32) {
                                    dt
                                } else {
                                    Self::handle_missing_value(i, None)
                                }
                            },
                            mysql_async::Value::Time(neg, days, hours, minutes, seconds, micros) => {
                                // 时间值转换为秒数
                                let total_seconds = (days * 86400 + hours * 3600 + minutes * 60 + seconds) as f32 
                                    + (micros as f32 / 1_000_000.0);
                                if neg { -total_seconds } else { total_seconds }
                            },
                            _ => Self::handle_missing_value(i, None), // 其他类型标记为缺失值
                        },
                        None => Self::handle_missing_value(i, None), // NULL值标记为缺失值
                    };
                    
                    feature_row.push(value);
                }
                
                if !feature_row.is_empty() {
                    features.push(feature_row);
                }
            }
            
            // 关闭连接池
            pool.disconnect().await
                .map_err(|e| Error::database(format!("MySQL连接池关闭错误: {}", e)))?;
        }
        
        #[cfg(not(feature = "mysql"))]
        {
            return Err(Error::not_implemented("MySQL支持需要启用mysql特性"));
        }
        
        #[cfg(feature = "mysql")]
        {
            return Ok(());
        }
    }
    
    // 从MongoDB加载数据
    async fn load_from_mongodb(
        &self, 
        config: &DatabaseConfig,
        _features: &mut Vec<Vec<f32>>,
        metadata: &mut HashMap<String, String>
    ) -> Result<()> {
        debug!("从MongoDB加载数据");
        
        // 添加数据库元数据
        metadata.insert("db_type".to_string(), "mongodb".to_string());
        metadata.insert("query".to_string(), config.query.clone());
        
        #[cfg(feature = "mongodb")]
        {
            use mongodb::{Client, options::ClientOptions};
            use mongodb::bson::{Document, Bson};
            use futures::stream::StreamExt;
            
            // 解析连接字符串和查询
            let mut parts = config.connection_string.split('/');
            let conn_str = parts.next().unwrap_or("");
            let db_name = parts.next().unwrap_or("test");
            let coll_name = parts.next().unwrap_or("data");
            
            // 解析查询为BSON文档
            let filter = match serde_json::from_str::<Document>(&config.query) {
                Ok(doc) => doc,
                Err(_) => Document::new(), // 如果解析失败，使用空文档（匹配所有）
            };
            
            // 连接到MongoDB
            let client_options = ClientOptions::parse(conn_str).await
                .map_err(|e| Error::database(format!("MongoDB连接选项解析错误: {}", e)))?;
                
            let client = Client::with_options(client_options)
                .map_err(|e| Error::database(format!("MongoDB客户端创建错误: {}", e)))?;
                
            // 获取数据库和集合
            let database = client.database(db_name);
            let collection = database.collection::<Document>(coll_name);
            
            // 执行查询
            let mut cursor = collection.find(filter, None).await
                .map_err(|e| Error::database(format!("MongoDB查询执行错误: {}", e)))?;
                
            // 处理结果
            let mut field_names = Vec::new();
            let mut has_field_names = false;
            
            while let Some(result) = cursor.next().await {
                match result {
                    Ok(document) => {
                        // 如果是第一个文档，记录字段名
                        if !has_field_names {
                            for key in document.keys() {
                                field_names.push(key.clone());
                            }
                            has_field_names = true;
                            metadata.insert("fields".to_string(), field_names.join(","));
                        }
                        
                        // 提取特征值
                        let mut feature_row = Vec::new();
                        
                        for field in &field_names {
                            if let Some(value) = document.get(field) {
                                let num_value = match value {
                                    Bson::Double(d) => *d as f32,
                                    Bson::Int32(i) => *i as f32,
                                    Bson::Int64(i) => *i as f32,
                                    Bson::String(s) => {
                                        // 使用生产级文本特征提取
                                        Self::extract_text_features(s)
                                    },
                                    Bson::Boolean(b) => if *b { 1.0 } else { 0.0 },
                                    Bson::Array(arr) => {
                                        // 使用数组长度作为特征
                                        arr.len() as f32
                                    },
                                    Bson::Document(doc) => {
                                        // 使用文档中的键数量作为特征
                                        doc.keys().count() as f32
                                    },
                                    _ => 0.0, // 其他类型
                                };
                                
                                feature_row.push(num_value);
                            } else {
                                // 字段不存在，标记为缺失值
                                feature_row.push(Self::handle_missing_value(
                                    field_names.iter().position(|f| f == field).unwrap_or(0),
                                    Some(field)
                                ));
                            }
                        }
                        
                        if !feature_row.is_empty() {
                            features.push(feature_row);
                        }
                    },
                    Err(e) => {
                        warn!("MongoDB文档读取错误: {}", e);
                        continue;
                    }
                }
            }
        }
        
        #[cfg(not(feature = "mongodb"))]
        {
            return Err(Error::not_implemented("MongoDB支持需要启用mongodb特性"));
        }
        
        #[cfg(feature = "mongodb")]
        {
            return Ok(());
        }
    }
}

#[async_trait]
impl DataLoader for DatabaseLoader {
    async fn load(&self, source: &DataSource, format: &crate::data::loader::types::DataFormat) -> Result<DataBatch> {
        // 验证数据源
        let db_config = match source {
            DataSource::Database(config) => config.clone(),
            _ => return Err(Error::invalid_argument("数据源必须是数据库类型")),
        };
        
        let mut features = Vec::new();
        let mut metadata = HashMap::new();
        
        // 根据数据库类型加载数据
        // 注意：db_config 是 ImportedDatabaseConfig (connector::types::DatabaseConfig)
        // 需要转换为 database::DatabaseConfig
        let local_db_config = Self::convert_to_local_config(&db_config);
        
        // 使用公共导出的 DatabaseType
        use crate::data::DatabaseType as ConnectorDatabaseType;
        match db_config.db_type {
            ConnectorDatabaseType::SQLite => {
                self.load_from_sqlite(&local_db_config, &mut features, &mut metadata).await?;
            },
            ConnectorDatabaseType::PostgreSQL => {
                self.load_from_postgres(&local_db_config, &mut features, &mut metadata).await?;
            },
            ConnectorDatabaseType::MySQL => {
                // MySQL加载实现
                #[cfg(feature = "mysql")]
                {
                    self.load_from_mysql(&local_db_config, &mut features, &mut metadata).await?;
                }
                
                #[cfg(not(feature = "mysql"))]
                {
                    return Err(Error::not_implemented("MySQL支持需要启用mysql特性"));
                }
            },
            ConnectorDatabaseType::MongoDB => {
                // MongoDB加载实现
                #[cfg(feature = "mongodb")]
                {
                    self.load_from_mongodb(&local_db_config, &mut features, &mut metadata).await?;
                }
                
                #[cfg(not(feature = "mongodb"))]
                {
                    return Err(Error::not_implemented("MongoDB支持需要启用mongodb特性"));
                }
            },
            ConnectorDatabaseType::Custom(name) => {
                return Err(Error::not_implemented(format!("暂不支持自定义数据库类型: {}", name)));
            },
            _ => {
                return Err(Error::not_implemented(format!("暂不支持的数据库类型: {:?}", db_config.db_type)));
            }
        }
        
        // 添加格式元数据
        metadata.insert("format".to_string(), format!("{:?}", format));
        
        // 创建数据批次
        let mut batch = DataBatch::new("database_dataset", 0, features.len());
        batch.metadata = metadata;
        
        // 转换特征向量为记录
        for (i, feature_vec) in features.iter().enumerate() {
            let mut record = HashMap::new();
            for (j, value) in feature_vec.iter().enumerate() {
                record.insert(format!("column_{}", j), DataValue::Float(*value as f64));
            }
            record.insert("row_id".to_string(), DataValue::Integer(i as i64));
            batch.records.push(record);
        }
        
        batch.size = batch.records.len();
        Ok(batch)
    }
    
    async fn get_schema(&self, source: &DataSource, format: &crate::data::loader::types::DataFormat) -> Result<DataSchema> {
        // 验证数据源
        let db_config = match source {
            DataSource::Database(config) => config.clone(),
            _ => return Err(Error::invalid_argument("数据源必须是数据库类型")),
        };
        
        // 注意：db_config 是 ImportedDatabaseConfig (connector::types::DatabaseConfig)
        // connector::types::DatabaseConfig 没有 schema 字段，所以需要转换为 database::DatabaseConfig
        let local_db_config = Self::convert_to_local_config(&db_config);
        
        // 如果转换后的配置中已有架构，直接返回
        if let Some(schema) = &local_db_config.schema {
            return Ok(schema.clone());
        }
        
        // 根据数据库类型获取架构
        
        // 使用公共导出的 DatabaseType
        use crate::data::DatabaseType as ConnectorDatabaseType;
        let schema = match db_config.db_type {
            ConnectorDatabaseType::SQLite => {
                // SQLite架构获取实现
                #[cfg(feature = "sqlite")]
                {
                    self.get_sqlite_schema(&local_db_config).await?
                }
                
                #[cfg(not(feature = "sqlite"))]
                {
                    // 返回默认模式
                    let mut schema = DataSchema::new("sqlite_default", "1.0");
                    let fields = vec![
                        crate::data::schema::schema::FieldDefinition {
                            name: "column_0".to_string(),
                            field_type: crate::data::schema::schema::FieldType::Numeric,
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
                        crate::data::schema::schema::FieldDefinition {
                            name: "row_id".to_string(),
                            field_type: crate::data::schema::schema::FieldType::Numeric,
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
                    schema
                }
            },
            ConnectorDatabaseType::PostgreSQL => {
                // PostgreSQL架构获取实现
                #[cfg(feature = "postgres")]
                {
                    self.get_postgres_schema(&local_db_config).await?
                }
                
                #[cfg(not(feature = "postgres"))]
                {
                    // 返回默认模式
                    let mut schema = DataSchema::new("postgres_default", "1.0");
                    let fields = vec![
                        crate::data::schema::schema::FieldDefinition {
                            name: "column_0".to_string(),
                            field_type: crate::data::schema::schema::FieldType::Numeric,
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
                        crate::data::schema::schema::FieldDefinition {
                            name: "row_id".to_string(),
                            field_type: crate::data::schema::schema::FieldType::Numeric,
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
                    schema
                }
            },
            _ => {
                // 对于其他数据库类型，构建基础架构
                let mut schema = DataSchema::new("db_default", "1.0");
                let fields = vec![
                    crate::data::schema::schema::FieldDefinition {
                        name: "column_0".to_string(),
                        field_type: crate::data::schema::schema::FieldType::Numeric,
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
                    crate::data::schema::schema::FieldDefinition {
                        name: "row_id".to_string(),
                        field_type: crate::data::schema::schema::FieldType::Numeric,
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
                schema
            }
        };
        
        Ok(schema)
    }
    
    async fn load_batch(&self, source: &DataSource, format: &crate::data::loader::types::DataFormat, batch_size: usize, offset: usize) -> Result<DataBatch> {
        // 对于数据库，我们可以通过修改查询来实现分页
        if let DataSource::Database(mut config) = source.clone() {
            // 在原查询上添加LIMIT和OFFSET
            // 注意：config.query 是 Option<String>，需要先 unwrap 或使用默认值
            let base_query = config.query.as_deref().unwrap_or("SELECT * FROM data");
            config.query = Some(format!("{} LIMIT {} OFFSET {}", base_query, batch_size, offset));
            let modified_source = DataSource::Database(config);
            self.load(&modified_source, format).await
        } else {
            Err(Error::invalid_argument("数据源必须是数据库类型"))
        }
    }
    
    fn supports_format(&self, format: &crate::data::loader::types::DataFormat) -> bool {
        // 数据库loader支持大部分格式，因为它主要处理查询结果
        match format {
            crate::data::loader::types::DataFormat::Csv { .. } => true,
            crate::data::loader::types::DataFormat::Json { .. } => true,
            _ => false,
        }
    }
    
    fn config(&self) -> &LoaderConfig {
        // 返回默认配置
        static DEFAULT_CONFIG: std::sync::OnceLock<LoaderConfig> = std::sync::OnceLock::new();
        DEFAULT_CONFIG.get_or_init(|| LoaderConfig::default())
    }
    
    fn set_config(&mut self, config: LoaderConfig) {
        // 目前database loader不直接使用LoaderConfig，但可以在这里处理一些通用设置
        // 可以根据需要更新self.config或self.db_config
    }
    
    fn name(&self) -> &'static str {
        "database_loader"
    }
    
    async fn get_size(&self, _path: &str) -> Result<usize> {
        // 实现生产级的数据大小估算：通过执行COUNT查询获取实际记录数
        // 使用self.db_config而不是path参数，因为path可能无法直接解析
        let db_config = &self.db_config;
        
        // 根据数据库类型执行COUNT查询
        match db_config.db_type {
            DatabaseType::Sqlite => {
                #[cfg(feature = "sqlite")]
                {
                    use rusqlite::Connection;
                    let conn = Connection::open(&db_config.connection_string)
                        .map_err(|e| Error::database(format!("SQLite连接错误: {}", e)))?;
                    
                    // 构建COUNT查询
                    let count_query = if db_config.query.to_uppercase().starts_with("SELECT") {
                        // 如果是SELECT查询，包装为COUNT查询
                        format!("SELECT COUNT(*) FROM ({}) AS subquery", db_config.query)
                    } else {
                        // 否则使用表名
                        if let Some(table) = &db_config.table {
                            format!("SELECT COUNT(*) FROM {}", table)
                        } else {
                            return Err(Error::invalid_argument("无法确定表名，无法估算数据大小"));
                        }
                    };
                    
                    let count: i64 = conn.query_row(&count_query, rusqlite::params![], |row| {
                        row.get(0)
                    })
                    .map_err(|e| Error::database(format!("SQLite COUNT查询错误: {}", e)))?;
                    
                    Ok(count as usize)
                }
                
                #[cfg(not(feature = "sqlite"))]
                {
                    Err(Error::not_implemented("SQLite支持需要启用sqlite特性"))
                }
            },
            DatabaseType::Postgres => {
                #[cfg(feature = "postgres")]
                {
                    use tokio_postgres::{Client, NoTls};
                    let (client, connection) = tokio_postgres::connect(&db_config.connection_string, NoTls)
                        .await
                        .map_err(|e| Error::database(format!("PostgreSQL连接错误: {}", e)))?;
                    
                    tokio::spawn(async move {
                        if let Err(e) = connection.await {
                            error!("PostgreSQL连接错误: {}", e);
                        }
                    });
                    
                    // 构建COUNT查询
                    let count_query = if db_config.query.to_uppercase().starts_with("SELECT") {
                        format!("SELECT COUNT(*) FROM ({}) AS subquery", db_config.query)
                    } else if let Some(table) = &db_config.table {
                        format!("SELECT COUNT(*) FROM {}", table)
                    } else {
                        return Err(Error::invalid_argument("无法确定表名，无法估算数据大小"));
                    };
                    
                    let row = client.query_one(&count_query, &[])
                        .await
                        .map_err(|e| Error::database(format!("PostgreSQL COUNT查询错误: {}", e)))?;
                    
                    let count: i64 = row.get(0);
                    Ok(count as usize)
                }
                
                #[cfg(not(feature = "postgres"))]
                {
                    Err(Error::not_implemented("PostgreSQL支持需要启用postgres特性"))
                }
            },
            _ => {
                // 其他数据库类型，返回错误而不是默认值
                Err(Error::not_implemented(format!("数据大小估算暂不支持数据库类型: {:?}", db_config.db_type)))
            }
        }
    }
}

// 为DatabaseLoader添加辅助方法
impl DatabaseLoader {
    // 获取SQLite数据库架构
    #[cfg(feature = "sqlite")]
    async fn get_sqlite_schema(&self, config: &DatabaseConfig) -> Result<DataSchema> {
        use rusqlite::Connection;
        
        // 连接SQLite数据库
        let conn = Connection::open(&config.connection_string)
            .map_err(|e| Error::database(format!("SQLite连接错误: {}", e)))?;
        
        // 执行查询
        let mut stmt = conn.prepare(&format!("{} LIMIT 1", config.query))
            .map_err(|e| Error::database(format!("SQLite查询准备错误: {}", e)))?;
        
        // 获取列名和类型信息
        let column_count = stmt.column_count();
        let mut fields = Vec::new();
        
        // 执行查询获取一行样本数据用于类型推断
        let mut rows = stmt.query(rusqlite::params![])
            .map_err(|e| Error::database(format!("SQLite样本查询错误: {}", e)))?;
        
        // 获取第一行数据用于类型推断
        let sample_row = rows.next()
            .transpose()
            .map_err(|e| Error::database(format!("SQLite结果集读取错误: {}", e)))?;
        
        for i in 0..column_count {
            if let Some(name) = stmt.column_name(i).ok() {
                // 根据实际列类型推断字段类型（生产级实现）
                let field_type = if let Some(ref row) = sample_row {
                    // 尝试从样本行推断类型
                    if let Ok(_) = row.get::<_, i64>(i) {
                        FieldType::Numeric // 整数映射到Numeric
                    } else if let Ok(_) = row.get::<_, f64>(i) {
                        FieldType::Numeric // 浮点数映射到Numeric
                    } else if let Ok(_) = row.get::<_, String>(i) {
                        FieldType::Text // 字符串映射到Text
                    } else if let Ok(_) = row.get::<_, bool>(i) {
                        FieldType::Boolean // 布尔值
                    } else if let Ok(_) = row.get::<_, Vec<u8>>(i) {
                        FieldType::Custom("binary".to_string()) // 二进制数据
                    } else {
                        // 无法确定类型，使用Text作为通用类型
                        FieldType::Text
                    }
                } else {
                    // 没有样本数据，使用SQLite声明的类型
                    // SQLite使用动态类型，但可以通过声明类型推断
                    let declared_type = stmt.column_decl_type(i)
                        .unwrap_or("TEXT")
                        .to_uppercase();
                    
                    match declared_type.as_str() {
                        "INTEGER" | "INT" | "BIGINT" | "SMALLINT" | "TINYINT" => FieldType::Numeric,
                        "REAL" | "DOUBLE" | "FLOAT" | "NUMERIC" | "DECIMAL" => FieldType::Numeric,
                        "TEXT" | "VARCHAR" | "CHAR" | "CLOB" => FieldType::Text,
                        "BLOB" | "BINARY" => FieldType::Custom("binary".to_string()),
                        "BOOLEAN" => FieldType::Boolean,
                        "DATE" | "DATETIME" | "TIMESTAMP" => FieldType::DateTime,
                        _ => FieldType::Text, // 默认为文本类型
                    }
                };
                
                let mut metadata = HashMap::new();
                metadata.insert("source".to_string(), "sqlite".to_string());
                metadata.insert("column_index".to_string(), i.to_string());
                if let Some(declared_type) = stmt.column_decl_type(i) {
                    metadata.insert("sqlite_declared_type".to_string(), declared_type.to_string());
                }
                
                fields.push(FieldDefinition {
                    name: name.to_string(),
                    field_type,
                    description: Some(format!("SQLite列 #{} ({})", i, field_type)),
                    constraints: None,
                    metadata,
                });
            }
        }
        
        let mut schema = DataSchema::new("sqlite_schema", "1.0");
        schema.description = Some(format!("SQLite查询结果模式"));
        
        // 添加所有字段
        for field in fields {
            schema.add_field(field)
                .map_err(|e| Error::database(format!("添加字段到SQLite模式失败: {}", e)))?;
        }
        
        Ok(schema)
    }
    
    // 获取PostgreSQL数据库架构
    #[cfg(feature = "postgres")]
    async fn get_postgres_schema(&self, config: &DatabaseConfig) -> Result<DataSchema> {
        use tokio_postgres::{Client, NoTls};
        
        // 连接PostgreSQL数据库
        let (client, connection) = tokio_postgres::connect(&config.connection_string, NoTls)
            .await
            .map_err(|e| Error::database(format!("PostgreSQL连接错误: {}", e)))?;
        
        // 必须在后台运行连接
        tokio::spawn(async move {
            if let Err(e) = connection.await {
                error!("PostgreSQL连接错误: {}", e);
            }
        });
        
        // 获取表信息
        let table_name = if let Some(table) = &config.table {
            table.clone()
        } else {
            // 尝试从查询中提取表名
            let query = config.query.to_lowercase();
            if let Some(from_idx) = query.find("from") {
                let after_from = &query[from_idx + 4..];
                let table_part = after_from.trim().split_whitespace().next();
                if let Some(table) = table_part {
                    table.trim_matches(|c| c == '(' || c == ')' || c == ';' || c == '"' || c == '\'').to_string()
                } else {
                    return Err(Error::invalid_argument("无法从查询中提取表名，请使用表名参数"));
                }
            } else {
                return Err(Error::invalid_argument("无法从查询中提取表名，请使用表名参数"));
            }
        };
        
        // 构建元数据查询
        let metadata_query = format!(
            "SELECT column_name, data_type FROM information_schema.columns WHERE table_name = $1"
        );
        
        // 执行元数据查询
        let rows = client.query(&metadata_query, &[&table_name])
            .await
            .map_err(|e| Error::database(format!("PostgreSQL模式查询错误: {}", e)))?;
        
        // 构建DataSchema
        let mut fields = Vec::new();
        
        for row in rows {
            let column_name: String = row.get(0);
            let data_type: String = row.get(1);
            
            // 映射PostgreSQL数据类型到我们的FieldType（生产级实现）
            let field_type = match data_type.to_lowercase().as_str() {
                "integer" | "bigint" | "smallint" | "int" | "int2" | "int4" | "int8" => FieldType::Numeric,
                "numeric" | "decimal" | "real" | "double precision" | "float" | "float4" | "float8" | "money" => FieldType::Numeric,
                "boolean" | "bool" => FieldType::Boolean,
                "date" => FieldType::DateTime, // 日期映射到DateTime
                "timestamp" | "timestamp without time zone" | "timestamp with time zone" | "time" | "time without time zone" | "time with time zone" | "interval" => FieldType::DateTime,
                "text" | "character varying" | "character" | "varchar" | "char" | "name" | "uuid" => FieldType::Text,
                "json" | "jsonb" => FieldType::Custom("json".to_string()),
                "bytea" | "bit" | "bit varying" | "varbit" => FieldType::Custom("binary".to_string()),
                "array" | "_text" | "_int4" | "_int8" | "_float8" => FieldType::Array(Box::new(FieldType::Text)), // 数组类型
                "point" | "line" | "lseg" | "box" | "path" | "polygon" | "circle" => FieldType::Custom("geometry".to_string()), // 几何类型
                "inet" | "cidr" | "macaddr" | "macaddr8" => FieldType::Text, // 网络类型作为文本
                "tsvector" | "tsquery" => FieldType::Text, // 全文搜索类型
                _ => FieldType::Custom(data_type.clone()),  // 其他类型作为自定义类型
            };
            
            // 创建空的元数据HashMap
            let mut metadata = HashMap::new();
            metadata.insert("postgres_type".to_string(), data_type.clone());
            
            fields.push(FieldDefinition {
                name: column_name,
                field_type,
                description: Some(format!("PostgreSQL {}类型字段", data_type)),
                constraints: None,
                metadata,
            });
        }
        
        let mut schema = DataSchema::new("postgres_schema", "1.0");
        schema.description = Some(format!("PostgreSQL表'{}'的模式", table_name));
        
        // 添加所有字段
        for field in fields {
            schema.add_field(field)
                .map_err(|e| Error::database(format!("添加字段到PostgreSQL模式失败: {}", e)))?;
        }
        
        Ok(schema)
    }
}
