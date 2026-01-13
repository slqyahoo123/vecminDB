use std::sync::Arc;
use tokio::sync::RwLock;
use crate::Result;
use crate::Error;
use crate::storage::config::{StorageConfig, StorageConfigUpdate};
use super::types::StorageStatistics;

/// 配置管理器服务
#[derive(Clone)]
pub struct ConfigManagerService {
    config: StorageConfig,
    db: Arc<RwLock<sled::Db>>,
}

impl ConfigManagerService {
    /// 创建新的配置管理器服务
    pub fn new(config: StorageConfig, db: Arc<RwLock<sled::Db>>) -> Self {
        Self { config, db }
    }

    /// 获取存储配置
    pub fn get_config(&self) -> &StorageConfig {
        &self.config
    }

    /// 更新配置
    pub async fn update_config(&self, new_config: &serde_json::Value) -> Result<()> {
        // 验证配置更新
        let config_update: StorageConfigUpdate = serde_json::from_value(new_config.clone())?;
        self.validate_config_update(&config_update)?;
        
        // 应用配置更新
        let key = "storage_config";
        let data = serde_json::to_vec(new_config)?;
        
        let db = self.db.write().await;
        db.insert(key.as_bytes(), data)?;
        
        Ok(())
    }

    /// 获取存储信息
    pub async fn get_info(&self) -> Result<serde_json::Value> {
        let db = self.db.read().await;
        let info = serde_json::json!({
            "version": "1.0.0",
            "path": self.config.path.to_string_lossy(),
            "size": db.size_on_disk()?,
            "len": db.len(),
            "is_empty": db.is_empty(),
        });
        
        Ok(info)
    }

    /// 获取存储引擎状态
    pub async fn get_storage_engine_status(&self) -> Result<serde_json::Value> {
        let mut status = serde_json::Map::new();
        
        // 基础状态
        status.insert("version".to_string(), serde_json::Value::String("1.0.0".to_string()));
        status.insert("path".to_string(), serde_json::Value::String(self.config.path.to_string_lossy().to_string()));
        
        // 数据库状态
        let db = self.db.read().await;
        status.insert("total_keys".to_string(), serde_json::Value::Number(serde_json::Number::from(db.len())));
        status.insert("is_empty".to_string(), serde_json::Value::Bool(db.len() == 0));
        
        // 配置信息
        let mut config_info = serde_json::Map::new();
        config_info.insert("max_file_size".to_string(), serde_json::Value::Number(serde_json::Number::from(self.config.max_file_size)));
        config_info.insert("cache_size".to_string(), serde_json::Value::Number(serde_json::Number::from(self.config.cache_size)));
        config_info.insert("use_compression".to_string(), serde_json::Value::Bool(self.config.use_compression));
        config_info.insert("compression_level".to_string(), serde_json::Value::Number(serde_json::Number::from(self.config.compression.unwrap_or(0))));
        status.insert("config".to_string(), serde_json::Value::Object(config_info));
        
        Ok(serde_json::Value::Object(status))
    }

    /// 获取存储统计信息
    pub async fn get_statistics(&self) -> Result<StorageStatistics> {
        let db = self.db.read().await;
        let size = db.size_on_disk()?;
        let len = db.len();
        let _is_empty = db.is_empty();
        
        // 计算缓存命中率（简化实现）
        let cache_hit_rate = 0.85; // 示例值
        
        // 计算平均操作时间（简化实现）
        let average_operation_time_ms = 5.0; // 示例值
        
        // 获取内存使用情况（简化实现）
        let peak_memory_usage_bytes = 1024 * 1024 * 100; // 示例值：100MB
        
        // 获取活跃连接数（简化实现）
        let active_connections = 10; // 示例值
        
        // 获取最后备份时间（简化实现）
        let last_backup_time = chrono::Utc::now().timestamp() as u64;
        
        // 计算数据库健康评分
        let health_score = self.calculate_database_health_score().await?;
        
        Ok(StorageStatistics {
            total_objects: len as u64,
            total_size_bytes: size,
            read_operations: 0, // 需要实现计数器
            write_operations: 0, // 需要实现计数器
            cache_hit_rate,
            average_operation_time_ms,
            peak_memory_usage_bytes,
            active_connections: active_connections as u32,
            last_backup_time,
            database_health_score: health_score,
        })
    }

    /// 计算数据库健康评分
    async fn calculate_database_health_score(&self) -> Result<f64> {
        let db = self.db.read().await;
        let mut score: f64 = 100.0;
        
        // 检查数据库大小
        let size = db.size_on_disk()?;
        if size > 1024 * 1024 * 1024 * 10 { // 超过10GB
            score -= 20.0;
        }
        
        // 检查是否为空
        if db.is_empty() {
            score -= 10.0;
        }
        
        // 确保分数在0-100范围内
        Ok(score.max(0.0_f64).min(100.0_f64))
    }

    /// 验证配置更新
    fn validate_config_update(&self, update: &StorageConfigUpdate) -> Result<()> {
        // 验证配置更新的合法性
        if let Some(write_buffer_size) = update.write_buffer_size {
            if write_buffer_size < 1024 * 1024 { // 最小1MB
                return Err(Error::InvalidInput("Write buffer size must be at least 1MB".to_string()));
            }
        }
        
        if let Some(cache_size_mb) = update.cache_size_mb {
            if cache_size_mb < 1 { // 最小1MB
                return Err(Error::InvalidInput("Cache size must be at least 1MB".to_string()));
            }
        }
        
        Ok(())
    }

    /// 获取数据库大小
    pub fn size(&self) -> Result<u64> {
        let db = self.db.blocking_read();
        Ok(db.size_on_disk()?)
    }
    
    /// 获取数据库长度
    pub fn len(&self) -> Result<usize> {
        let db = self.db.blocking_read();
        Ok(db.len())
    }
    
    /// 检查数据库是否为空
    pub fn is_empty(&self) -> Result<bool> {
        let db = self.db.blocking_read();
        Ok(db.is_empty())
    }
    
    /// 刷新数据库
    pub fn flush(&self) -> Result<()> {
        let db = self.db.blocking_write();
        db.flush()?;
        Ok(())
    }
    
    /// 关闭数据库
    pub fn close(&self) -> Result<()> {
        let db = self.db.blocking_write();
        db.flush()?;
        Ok(())
    }
} 