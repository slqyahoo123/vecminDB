// connector/factory.rs - 数据库连接器工厂

use crate::Result;
use super::{
    DatabaseConnector, 
    DatabaseConfig, 
    DatabaseType,
    mysql_connector::MySQLConnector,
    postgresql_connector::PostgreSQLConnector,
    sqlite_connector::SQLiteConnector,
};
#[cfg(feature = "mongodb")]
use super::mongodb_connector::MongoDBConnector;
use super::{
    redis_connector::RedisConnector,
    cassandra_connector::CassandraConnector,
    elasticsearch_connector::ElasticsearchConnector,
};
#[cfg(feature = "neo4rs")]
use super::neo4j_connector::Neo4jConnector;
use std::io::{Error, ErrorKind};

/// 数据库连接器工厂，用于创建不同类型的数据库连接器
pub struct DatabaseConnectorFactory;

impl DatabaseConnectorFactory {
    /// 根据数据库配置创建相应的数据库连接器
    pub fn create(config: DatabaseConfig) -> Result<Box<dyn DatabaseConnector>> {
        match config.db_type {
            DatabaseType::MySQL => {
                let connector = MySQLConnector::new(config)?;
                Ok(Box::new(connector))
            },
            DatabaseType::PostgreSQL => {
                let connector = PostgreSQLConnector::new(config)?;
                Ok(Box::new(connector))
            },
            DatabaseType::SQLite => {
                let connector = SQLiteConnector::new(config)?;
                Ok(Box::new(connector))
            },
            #[cfg(feature = "mongodb")]
            DatabaseType::MongoDB => {
                let connector = MongoDBConnector::new(config)?;
                Ok(Box::new(connector))
            },
            DatabaseType::Redis => {
                let connector = RedisConnector::new(config)?;
                Ok(Box::new(connector))
            },
            DatabaseType::Cassandra => {
                let connector = CassandraConnector::new(config)?;
                Ok(Box::new(connector))
            },
            DatabaseType::Elasticsearch => {
                let connector = ElasticsearchConnector::new(config)?;
                Ok(Box::new(connector))
            },
            #[cfg(feature = "neo4rs")]
            DatabaseType::Neo4j => {
                let connector = Neo4jConnector::new(config)?;
                Ok(Box::new(connector))
            },
            _ => Err(Error::new(ErrorKind::Unsupported, "不支持的数据库类型").into()),
        }
    }
    
    /// create_connector 作为 create 方法的别名，保持兼容性
    pub fn create_connector(config: DatabaseConfig) -> Result<Box<dyn DatabaseConnector>> {
        Self::create(config)
    }
} 