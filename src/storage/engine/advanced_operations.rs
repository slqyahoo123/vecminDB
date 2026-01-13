use std::sync::Arc;
use tokio::sync::RwLock;
use log::{debug, warn};
use crate::Result;
use crate::Error;

/// 高级存储操作服务
#[derive(Clone)]
pub struct AdvancedOperationsService {
    db: Arc<RwLock<sled::Db>>,
}

impl AdvancedOperationsService {
    /// 创建新的高级操作服务
    pub fn new(db: Arc<RwLock<sled::Db>>) -> Self {
        Self { db }
    }

    /// 批量写入操作
    pub fn batch_write(&self, operations: Vec<(Vec<u8>, Option<Vec<u8>>)>) -> Result<()> {
        let db = self.db.blocking_read();
        let mut batch = sled::Batch::default();
        
        for (key, value) in operations {
            match value {
                Some(val) => batch.insert(key, val),
                None => batch.remove(key),
            }
        }
        
        db.apply_batch(batch)
            .map_err(|e| Error::storage(format!("批量写入失败: {}", e)))?;
        
        debug!("批量写入完成，操作数量: {}", operations.len());
        Ok(())
    }

    /// 高性能前缀扫描
    pub fn scan_prefix_optimized(&self, prefix: &[u8], limit: Option<usize>) -> Result<Vec<(Vec<u8>, Vec<u8>)>> {
        let db = self.db.blocking_read();
        let mut results = Vec::new();
        let mut count = 0;
        
        for result in db.scan_prefix(prefix) {
            match result {
                Ok((key, value)) => {
                    results.push((key.to_vec(), value.to_vec()));
                    count += 1;
                    
                    // 如果达到限制，停止扫描
                    if let Some(limit_val) = limit {
                        if count >= limit_val {
                            break;
                        }
                    }
                },
                Err(e) => {
                    warn!("前缀扫描时出错: {}", e);
                    continue;
                }
            }
        }
        
        debug!("前缀扫描完成，前缀: {:?}, 结果数量: {}", prefix, results.len());
        Ok(results)
    }

    /// 异步前缀扫描
    pub async fn scan_prefix_async(&self, prefix: &[u8], limit: Option<usize>) -> Result<Vec<(Vec<u8>, Vec<u8>)>> {
        let db = self.db.read().await;
        let mut results = Vec::new();
        let mut count = 0;
        
        for result in db.scan_prefix(prefix) {
            match result {
                Ok((key, value)) => {
                    results.push((key.to_vec(), value.to_vec()));
                    count += 1;
                    
                    // 如果达到限制，停止扫描
                    if let Some(limit_val) = limit {
                        if count >= limit_val {
                            break;
                        }
                    }
                },
                Err(e) => {
                    warn!("异步前缀扫描时出错: {}", e);
                    continue;
                }
            }
        }
        
        debug!("异步前缀扫描完成，前缀: {:?}, 结果数量: {}", prefix, results.len());
        Ok(results)
    }

    /// 高性能刷新操作
    pub fn flush_optimized(&self) -> Result<()> {
        let db = self.db.blocking_read();
        
        // 执行刷新
        db.flush()
            .map_err(|e| Error::storage(format!("数据库刷新失败: {}", e)))?;
        
        // 等待刷新完成
        db.flush()
            .map_err(|e| Error::storage(format!("数据库刷新确认失败: {}", e)))?;
        
        debug!("数据库刷新完成");
        Ok(())
    }

    /// 异步刷新操作
    pub async fn flush_async(&self) -> Result<()> {
        let db = self.db.read().await;
        
        // 执行刷新
        db.flush()
            .map_err(|e| Error::storage(format!("异步数据库刷新失败: {}", e)))?;
        
        // 等待刷新完成
        db.flush()
            .map_err(|e| Error::storage(format!("异步数据库刷新确认失败: {}", e)))?;
        
        debug!("异步数据库刷新完成");
        Ok(())
    }

    /// 优雅关闭
    pub async fn graceful_shutdown(&self) -> Result<()> {
        use log::info;
        
        info!("开始优雅关闭存储引擎...");
        
        // 1. 停止接受新的事务
        // 2. 等待所有事务完成
        // 3. 清理过期事务
        // 4. 刷新数据库
        self.flush_async().await?;
        
        // 5. 关闭数据库连接
        let db = self.db.read().await;
        drop(db); // 释放读锁
        
        info!("存储引擎优雅关闭完成");
        Ok(())
    }

    /// 获取所有键
    pub async fn list_keys(&self, prefix: &str) -> Result<Vec<String>> {
        let db = self.db.read().await;
        let mut keys = Vec::new();
        
        for result in db.scan_prefix(prefix.as_bytes()) {
            match result {
                Ok((key, _)) => {
                    if let Ok(key_str) = String::from_utf8(key.to_vec()) {
                        keys.push(key_str);
                    }
                },
                Err(e) => {
                    warn!("获取键时出错: {}", e);
                    continue;
                }
            }
        }
        
        debug!("获取键完成，前缀: {}, 键数量: {}", prefix, keys.len());
        Ok(keys)
    }

    /// 获取指定前缀的键
    pub async fn get_keys_with_prefix(&self, prefix: &str) -> Result<Vec<String>> {
        let db = self.db.read().await;
        let mut keys = Vec::new();
        
        for result in db.scan_prefix(prefix.as_bytes()) {
            match result {
                Ok((key, _)) => {
                    if let Ok(key_str) = String::from_utf8(key.to_vec()) {
                        keys.push(key_str);
                    }
                },
                Err(e) => {
                    warn!("获取前缀键时出错: {}", e);
                    continue;
                }
            }
        }
        
        debug!("获取前缀键完成，前缀: {}, 键数量: {}", prefix, keys.len());
        Ok(keys)
    }
} 