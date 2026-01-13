// connector/mod.rs - 数据库连接器模块

mod types;
mod mysql_connector;
mod postgresql_connector;
mod sqlite_connector;
mod mongodb_connector;
mod redis_connector;
mod cassandra_connector;
mod elasticsearch_connector;
mod neo4j_connector;
mod manager;
mod factory;
mod advanced_pool;

// 重新导出所有公共类型
pub use types::*;

// 重新导出所有连接器实现
pub use mysql_connector::MySQLConnector;
pub use postgresql_connector::PostgreSQLConnector;
pub use sqlite_connector::SQLiteConnector;
#[cfg(feature = "mongodb")]
pub use mongodb_connector::MongoDBConnector;
#[cfg(feature = "redis")]
pub use redis_connector::RedisConnector;
#[cfg(feature = "cassandra")]
pub use cassandra_connector::CassandraConnector;
#[cfg(feature = "elasticsearch")]
pub use elasticsearch_connector::ElasticsearchConnector;
#[cfg(feature = "neo4rs")]
pub use neo4j_connector::Neo4jConnector;
pub use manager::{DatabaseManager, DatabaseConnection};
pub use factory::DatabaseConnectorFactory;

// 重新导出高级连接池
pub use advanced_pool::{
    AdvancedConnectionPool, PooledConnection, ConnectionFactory, PoolConfig,
    PoolStats, PoolMonitor, HealthChecker, PoolOptimizer, ConnectionState,
    ConnectionMetadata, HealthCheckResult, OptimizationAction
};

/// 数据库字段映射器 - 统一处理不同数据库的字段类型转换
pub struct DatabaseFieldMapper;

impl DatabaseFieldMapper {
    /// 创建标准的数字字段定义
    pub fn create_numeric_field(name: &str, sql_type: &str, description: Option<&str>) -> crate::data::schema::schema::FieldDefinition {
        crate::data::schema::schema::FieldDefinition {
            name: name.to_string(),
            field_type: crate::data::schema::schema::FieldType::Numeric,
            data_type: Some(sql_type.to_string()),
            required: true,
            nullable: false,
            primary_key: name == "id",
            foreign_key: None,
            description: description.map(|s| s.to_string()),
            default_value: None,
            constraints: None,
            metadata: std::collections::HashMap::new(),
        }
    }
    
    /// 创建标准的文本字段定义
    pub fn create_text_field(name: &str, sql_type: &str, description: Option<&str>) -> crate::data::schema::schema::FieldDefinition {
        crate::data::schema::schema::FieldDefinition {
            name: name.to_string(),
            field_type: crate::data::schema::schema::FieldType::Text,
            data_type: Some(sql_type.to_string()),
            required: false,
            nullable: true,
            primary_key: false,
            foreign_key: None,
            description: description.map(|s| s.to_string()),
            default_value: None,
            constraints: None,
            metadata: std::collections::HashMap::new(),
        }
    }
    
    /// 创建标准的布尔字段定义
    pub fn create_boolean_field(name: &str, sql_type: &str, description: Option<&str>) -> crate::data::schema::schema::FieldDefinition {
        crate::data::schema::schema::FieldDefinition {
            name: name.to_string(),
            field_type: crate::data::schema::schema::FieldType::Boolean,
            data_type: Some(sql_type.to_string()),
            required: false,
            nullable: true,
            primary_key: false,
            foreign_key: None,
            description: description.map(|s| s.to_string()),
            default_value: None,
            constraints: None,
            metadata: std::collections::HashMap::new(),
        }
    }
    
    /// 创建标准的日期时间字段定义
    pub fn create_datetime_field(name: &str, sql_type: &str, description: Option<&str>) -> crate::data::schema::schema::FieldDefinition {
        crate::data::schema::schema::FieldDefinition {
            name: name.to_string(),
            field_type: crate::data::schema::schema::FieldType::DateTime,
            data_type: Some(sql_type.to_string()),
            required: false,
            nullable: true,
            primary_key: false,
            foreign_key: None,
            description: description.map(|s| s.to_string()),
            default_value: None,
            constraints: None,
            metadata: std::collections::HashMap::new(),
        }
    }
    
    /// 创建可配置的字段定义
    pub fn create_field(
        name: &str, 
        field_type: crate::data::schema::schema::FieldType,
        sql_type: &str,
        required: bool,
        nullable: bool,
        primary_key: bool,
        description: Option<&str>
    ) -> crate::data::schema::schema::FieldDefinition {
        crate::data::schema::schema::FieldDefinition {
            name: name.to_string(),
            field_type,
            data_type: Some(sql_type.to_string()),
            required,
            nullable,
            primary_key,
            foreign_key: None,
            description: description.map(|s| s.to_string()),
            default_value: None,
            constraints: None,
            metadata: std::collections::HashMap::new(),
        }
    }
}

/// 查询参数转换器 - 统一处理不同数据库的参数转换
pub struct QueryParamConverter;

impl QueryParamConverter {
    /// 转换查询参数为字符串格式
    pub fn convert_params(params: &[QueryParam]) -> Vec<String> {
        params.iter()
            .map(|p| match &p.value {
                QueryParamValue::String(s) => format!("'{}'", s.replace("'", "''")), // SQL注入防护
                QueryParamValue::Integer(i) => i.to_string(),
                QueryParamValue::Float(f) => f.to_string(),
                QueryParamValue::Boolean(b) => if *b { "1".to_string() } else { "0".to_string() },
                QueryParamValue::Null => "NULL".to_string(),
                _ => "NULL".to_string(),
            })
            .collect()
    }
}

/// 查询构建器 - 统一处理SQL查询构建
pub struct QueryBuilder;

impl QueryBuilder {
    /// 构建查询字符串
    pub fn build_query(params: &QueryParams) -> String {
        let mut query = params.query.clone();
        
        // 添加排序 - 必须在LIMIT之前
        if let Some(sort_by) = &params.sort_by {
            let direction = match params.sort_direction {
                Some(SortDirection::Ascending) => "ASC",
                Some(SortDirection::Descending) => "DESC",
                None => "ASC",
            };
            
            query = format!("{} ORDER BY {} {}", query, sort_by, direction);
        }
        
        // 添加LIMIT和OFFSET
        if let Some(limit) = params.limit {
            query = format!("{} LIMIT {}", query, limit);
        }
        
        if let Some(offset) = params.offset {
            query = format!("{} OFFSET {}", query, offset);
        }
        
        query
    }
}

/// 结果转换器 - 统一处理查询结果转换
pub struct ResultConverter;

impl ResultConverter {
    /// 将查询结果转换为DataBatch
    pub fn convert_results(results: Vec<std::collections::HashMap<String, String>>) -> crate::Result<crate::data::batch::DataBatch> {
        if results.is_empty() {
            return Ok(crate::data::batch::DataBatch::new("", 0, 0));
        }
        
        // 收集所有字段名
        let fields: Vec<String> = results[0].keys().cloned().collect();
        
        // 创建特征矩阵
        let mut features = Vec::new();
        
        for row in &results {
            let mut feature_row = Vec::new();
            
            for field in &fields {
                let value = row.get(field).cloned().unwrap_or_default();
                // 尝试转换为数字，失败则用哈希值
                let num_value = if let Ok(val) = value.parse::<f32>() {
                    val
                } else {
                    // 对非数字字符串进行简单的哈希转换
                    (value.len() as f32) % 1000.0
                };
                feature_row.push(num_value);
            }
            
            features.push(feature_row);
        }
        
        // 创建DataBatch
        let batch_size = features.len();
        Ok(crate::data::batch::DataBatch::new("connector", 0, batch_size)
            .with_features(features))
    }
} 