// connector/types.rs - 共享类型定义

use std::collections::HashMap;
use crate::Result;
use crate::Error;
use crate::data::{DataBatch, DataSchema};
use serde::{Serialize, Deserialize};
use crate::data::connector::{
    MySQLConnector, PostgreSQLConnector,
};
// 以下连接器依赖未启用的特性/外部库，类型定义中暂不直接使用具体实现
// use crate::data::connector::elasticsearch_connector::ElasticsearchConnector;
// use crate::data::connector::MongoDBConnector;
// use crate::data::connector::Neo4jConnector;

/// 数据库类型枚举
#[derive(Debug, Clone)]
pub enum DatabaseType {
    // 关系型数据库
    MySQL,
    PostgreSQL,
    SQLite,
    SQLServer,
    Oracle,
    
    // 非关系型数据库
    Redis,
    Cassandra,
    Elasticsearch,
    Neo4j,
    MongoDB,
    
    // 其他类型
    Custom(String),
}

/// 数据库配置
#[derive(Debug, Clone)]
pub struct DatabaseConfig {
    /// 数据库类型
    pub db_type: DatabaseType,
    /// 连接字符串或地址
    pub connection_string: String,
    /// 用户名
    pub username: Option<String>,
    /// 密码
    pub password: Option<String>,
    /// 数据库名称
    pub database: Option<String>,
    /// 表名或集合名
    pub table: Option<String>,
    /// 查询语句
    pub query: Option<String>,
    /// 连接池大小
    pub pool_size: Option<usize>,
    /// 连接超时（秒）
    pub timeout: Option<u64>,
    /// 额外参数
    pub extra_params: HashMap<String, String>,
}

impl Default for DatabaseConfig {
    fn default() -> Self {
        Self {
            db_type: DatabaseType::SQLite,
            connection_string: "memory".to_string(),
            username: None,
            password: None,
            database: None,
            table: None,
            query: None,
            pool_size: Some(10),
            timeout: Some(30),
            extra_params: HashMap::new(),
        }
    }
}

/// 查询参数
#[derive(Debug, Clone)]
pub struct QueryParams {
    /// SQL查询或NoSQL查询
    pub query: String,
    /// 查询参数
    pub params: Vec<QueryParam>,
    /// 限制返回的行数
    pub limit: Option<usize>,
    /// 跳过的行数
    pub offset: Option<usize>,
    /// 排序字段
    pub sort_by: Option<String>,
    /// 排序方向（升序或降序）
    pub sort_direction: Option<SortDirection>,
    /// 表名或集合名
    pub table_name: Option<String>,
}

/// 查询参数值
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum QueryParamValue {
    String(String),
    Integer(i64),
    Float(f64),
    Boolean(bool),
    Null,
    Array(Vec<QueryParamValue>),
    Object(HashMap<String, QueryParamValue>),
}

/// 查询参数
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryParam {
    pub field: String,
    pub value: QueryParamValue,
    pub operator: QueryOperator,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum QueryOperator {
    Equals,
    NotEquals,
    GreaterThan,
    LessThan,
    GreaterOrEqual,
    LessOrEqual,
    Like,
    In,
    NotIn,
    Between,
}

/// 排序方向
#[derive(Debug, Clone)]
pub enum SortDirection {
    Ascending,
    Descending,
}

/// 写入模式
#[derive(Debug)]
pub enum WriteMode {
    /// 插入新数据
    Insert,
    /// 更新现有数据
    Update,
    /// 插入或更新（upsert）
    Upsert,
    /// 替换所有数据
    Replace,
    /// 追加数据
    Append,
}

/// 数据库连接器工厂
pub struct DatabaseConnectorFactory;

impl DatabaseConnectorFactory {
    /// 创建数据库连接器
    pub fn create_connector(config: DatabaseConfig) -> Result<Box<dyn DatabaseConnector>> {
        match config.db_type {
            DatabaseType::MySQL => {
                // 创建MySQL连接器
                let connector = MySQLConnector::new(config)?;
                Ok(Box::new(connector))
            },
            DatabaseType::PostgreSQL => {
                // 创建PostgreSQL连接器
                let connector = PostgreSQLConnector::new(config)?;
                Ok(Box::new(connector))
            },
            // 下面这些数据库类型依赖尚未启用的外部驱动，实现占位错误返回，避免 cfg(feature=...) 警告
            DatabaseType::MongoDB => {
                Err(Error::invalid_data("MongoDB connector feature is not enabled".to_string()))
            },
            DatabaseType::Elasticsearch => {
                Err(Error::invalid_data("Elasticsearch connector feature is not enabled".to_string()))
            },
            DatabaseType::Neo4j => {
                Err(Error::invalid_data("Neo4j connector feature is not enabled".to_string()))
            },
            _ => Err(Error::not_implemented(format!("不支持的数据库类型: {:?}", config.db_type)))
        }
    }
}

/// 数据库连接器特质
pub trait DatabaseConnector: Send + Sync {
    /// 连接到数据库
    fn connect<'life0>(&'life0 mut self) -> Box<dyn std::future::Future<Output = Result<()>> + Send + 'life0>;
    
    /// 断开数据库连接
    fn disconnect<'life0>(&'life0 mut self) -> Box<dyn std::future::Future<Output = Result<()>> + Send + 'life0>;
    
    /// 测试数据库连接
    fn test_connection<'life0>(&'life0 self) -> Box<dyn std::future::Future<Output = Result<bool>> + Send + 'life0>;
    
    /// 执行查询并返回数据批次
    fn query<'life0, 'life1>(&'life0 self, params: &'life1 QueryParams) -> Box<dyn std::future::Future<Output = Result<DataBatch>> + Send + 'life0>;
    
    /// 获取数据库模式
    fn get_schema<'life0, 'life1>(&'life0 self, table_name: Option<&'life1 str>) -> Box<dyn std::future::Future<Output = Result<DataSchema>> + Send + 'life0>;
    
    /// 将数据写入数据库
    fn write_data<'life0, 'life1, 'life2>(&'life0 self, batch: &'life1 DataBatch, table_name: &'life2 str, mode: WriteMode) -> Box<dyn std::future::Future<Output = Result<usize>> + Send + 'life0>;
    
    /// 获取数据库类型
    fn get_type(&self) -> DatabaseType;
    
    /// 获取数据库配置
    fn get_config(&self) -> &DatabaseConfig;
}