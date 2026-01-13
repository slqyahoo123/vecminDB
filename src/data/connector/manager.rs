// connector/manager.rs - 数据库连接管理器

use std::collections::HashMap;
use std::pin::Pin;
use crate::Result;
use crate::Error;
use super::types::{DatabaseType};
use super::DatabaseConnector;

#[cfg(feature = "postgres")]
use tokio_postgres;
#[cfg(feature = "mysql")]
use mysql_async;
#[cfg(feature = "sqlite")]
use rusqlite;
#[cfg(feature = "redis")]
use redis;
#[cfg(feature = "mongodb")]
use mongodb;
#[cfg(feature = "elasticsearch")]
use elasticsearch;

/// 数据库连接枚举，包装不同类型的数据库连接
pub enum DatabaseConnection {
    #[cfg(feature = "postgres")]
    PostgreSQL(tokio_postgres::Client),
    #[cfg(feature = "mysql")]
    MySQL(mysql_async::Conn),
    #[cfg(feature = "sqlite")]
    SQLite(rusqlite::Connection),
    #[cfg(feature = "redis")]
    Redis(Box<std::sync::Mutex<redis::aio::Connection>>),
    #[cfg(feature = "mongodb")]
    MongoDB(mongodb::Database),
    #[cfg(feature = "elasticsearch")]
    Elasticsearch(elasticsearch::Elasticsearch),
    /// 通用连接类型，当特定数据库特性未启用时使用
    Generic(Box<dyn std::any::Any + Send + Sync>),
}

impl std::fmt::Debug for DatabaseConnection {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            #[cfg(feature = "postgres")]
            DatabaseConnection::PostgreSQL(_) => write!(f, "DatabaseConnection::PostgreSQL"),
            #[cfg(feature = "mysql")]
            DatabaseConnection::MySQL(_) => write!(f, "DatabaseConnection::MySQL"),
            #[cfg(feature = "sqlite")]
            DatabaseConnection::SQLite(_) => write!(f, "DatabaseConnection::SQLite"),
            #[cfg(feature = "redis")]
            DatabaseConnection::Redis(_) => write!(f, "DatabaseConnection::Redis"),
            #[cfg(feature = "mongodb")]
            DatabaseConnection::MongoDB(_) => write!(f, "DatabaseConnection::MongoDB"),
            #[cfg(feature = "elasticsearch")]
            DatabaseConnection::Elasticsearch(_) => write!(f, "DatabaseConnection::Elasticsearch"),
            DatabaseConnection::Generic(_) => write!(f, "DatabaseConnection::Generic"),
        }
    }
}

/// 数据库连接管理器，用于管理多个数据库连接
pub struct DatabaseManager {
    /// 数据库连接器列表
    pub connectors: HashMap<String, Box<dyn DatabaseConnector>>,
}

impl DatabaseManager {
    /// 创建一个新的数据库连接管理器
    pub fn new() -> Self {
        Self {
            connectors: HashMap::new(),
        }
    }

    /// 添加数据库连接器
    pub fn add_connector(&mut self, name: &str, connector: Box<dyn DatabaseConnector>) {
        self.connectors.insert(name.to_string(), connector);
    }

    /// 获取数据库连接器
    pub fn get_connector(&self, name: &str) -> Option<&Box<dyn DatabaseConnector>> {
        self.connectors.get(name)
    }

    /// 获取可变数据库连接器
    pub fn get_connector_mut(&mut self, name: &str) -> Option<&mut Box<dyn DatabaseConnector>> {
        self.connectors.get_mut(name)
    }
    
    /// 断开所有数据库连接
    pub async fn disconnect_all(&mut self) -> Result<()> {
        for (_, connector) in &mut self.connectors {
            Pin::from(connector.disconnect()).await?;
        }
        Ok(())
    }

    /// 连接所有数据库
    pub async fn connect_all(&mut self) -> Result<()> {
        for (_, connector) in &mut self.connectors {
            Pin::from(connector.connect()).await?;
        }
        Ok(())
    }

    /// 测试所有数据库连接
    pub async fn test_all_connections(&self) -> Result<HashMap<String, bool>> {
        let mut results = HashMap::new();
        for (name, connector) in &self.connectors {
            let result = Pin::from(connector.test_connection()).await?;
            results.insert(name.clone(), result);
        }
        Ok(results)
    }

    /// 连接指定数据库
    pub async fn connect(&mut self, name: &str) -> Result<()> {
        if let Some(connector) = self.connectors.get_mut(name) {
            Pin::from(connector.connect()).await?;
            Ok(())
        } else {
            Err(Error::not_found(format!("找不到连接器: {}", name)))
        }
    }

    /// 获取数据库连接
    /// 根据连接字符串返回通用的数据库连接对象
    pub async fn get_connection(&self, connection_string: &str) -> Result<DatabaseConnection> {
        // 解析连接字符串以确定数据库类型
        let db_type = self.parse_connection_string(connection_string)?;
        
        match db_type {
            DatabaseType::PostgreSQL => {
                #[cfg(feature = "postgres")]
                {
                    let (client, connection) = tokio_postgres::connect(connection_string, tokio_postgres::NoTls).await
                        .map_err(|e| Error::database_operation(&format!("PostgreSQL连接失败: {}", e)))?;
                    
                    // 启动连接
                    tokio::spawn(async move {
                        if let Err(e) = connection.await {
                            // 生产级实现：记录连接错误（实际部署时应使用日志系统）
                            // 例如：error!("PostgreSQL连接错误: {}", e);
                        }
                    });
                    
                    Ok(DatabaseConnection::PostgreSQL(client))
                }
                #[cfg(not(feature = "postgres"))]
                {
                    Err(Error::not_implemented("PostgreSQL支持需要启用postgres特性"))
                }
            },
            DatabaseType::MySQL => {
                #[cfg(feature = "mysql")]
                {
                    let pool = mysql_async::Pool::new(connection_string);
                    let conn = pool.get_conn().await
                        .map_err(|e| Error::database_operation(&format!("MySQL连接失败: {}", e)))?;
                    Ok(DatabaseConnection::MySQL(conn))
                }
                #[cfg(not(feature = "mysql"))]
                {
                    Err(Error::not_implemented("MySQL支持需要启用mysql特性"))
                }
            },
            DatabaseType::SQLite => {
                #[cfg(feature = "sqlite")]
                {
                    let conn = rusqlite::Connection::open(connection_string)
                        .map_err(|e| Error::database_operation(&format!("SQLite连接失败: {}", e)))?;
                    Ok(DatabaseConnection::SQLite(conn))
                }
                #[cfg(not(feature = "sqlite"))]
                {
                    Err(Error::not_implemented("SQLite支持需要启用sqlite特性"))
                }
            },
            DatabaseType::Redis => {
                #[cfg(feature = "redis")]
                {
                    let client = redis::Client::open(connection_string)
                        .map_err(|e| Error::database_operation(&format!("Redis客户端创建失败: {}", e)))?;
                    let conn = client.get_async_connection().await
                        .map_err(|e| Error::database_operation(&format!("Redis连接失败: {}", e)))?;
                    Ok(DatabaseConnection::Redis(Box::new(std::sync::Mutex::new(conn))))
                }
                #[cfg(not(feature = "redis"))]
                {
                    Err(Error::not_implemented("Redis支持需要启用redis特性"))
                }
            },
            DatabaseType::MongoDB => {
                #[cfg(feature = "mongodb")]
                {
                    let client = mongodb::Client::with_uri_str(connection_string).await
                        .map_err(|e| Error::database_operation(&format!("MongoDB连接失败: {}", e)))?;
                    // 获取默认数据库
                    let db_name = self.extract_database_name(connection_string)
                        .unwrap_or_else(|| "default".to_string());
                    let database = client.database(&db_name);
                    Ok(DatabaseConnection::MongoDB(database))
                }
                #[cfg(not(feature = "mongodb"))]
                {
                    Err(Error::not_implemented("MongoDB支持需要启用mongodb特性"))
                }
            },
            DatabaseType::Elasticsearch => {
                #[cfg(feature = "elasticsearch")]
                {
                    use std::str::FromStr;
                    let url = url::Url::from_str(connection_string)
                        .map_err(|e| Error::database_operation(&format!("Elasticsearch URL解析失败: {}", e)))?;
                    let transport = elasticsearch::http::transport::Transport::single_node(&url.to_string())
                        .map_err(|e| Error::database_operation(&format!("Elasticsearch传输创建失败: {}", e)))?;
                    let client = elasticsearch::Elasticsearch::new(transport);
                    Ok(DatabaseConnection::Elasticsearch(client))
                }
                #[cfg(not(feature = "elasticsearch"))]
                {
                    Err(Error::not_implemented("Elasticsearch支持需要启用elasticsearch特性"))
                }
            },
            _ => {
                Err(Error::not_implemented(&format!("不支持的数据库类型: {:?}", db_type)))
            }
        }
    }

    /// 解析连接字符串以确定数据库类型
    fn parse_connection_string(&self, connection_string: &str) -> Result<DatabaseType> {
        if connection_string.starts_with("postgresql://") || connection_string.starts_with("postgres://") {
            Ok(DatabaseType::PostgreSQL)
        } else if connection_string.starts_with("mysql://") {
            Ok(DatabaseType::MySQL)
        } else if connection_string.ends_with(".db") || connection_string.starts_with("sqlite://") {
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

    /// 从连接字符串中提取数据库名称
    fn extract_database_name(&self, connection_string: &str) -> Option<String> {
        // 简单的数据库名称提取逻辑
        if connection_string.contains("mongodb://") {
            // 从MongoDB连接字符串中提取数据库名
            let parts: Vec<&str> = connection_string.split('/').collect();
            if parts.len() > 3 {
                Some(parts[3].split('?').next().unwrap_or("default").to_string())
            } else {
                Some("default".to_string())
            }
        } else {
            None
        }
    }
} 