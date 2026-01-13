use std::sync::{Arc, Mutex};
use std::collections::HashMap;
// use log::debug; // reserve for future detailed tracing
use tokio::sync::RwLock;

use crate::Result;
use crate::Error;
// remove unused cross-module imports; this module focuses on algorithm storage only

use super::transaction::TransactionManager;

/// 算法管理服务
#[derive(Clone)]
pub struct AlgorithmManager {
    storage: Arc<RwLock<sled::Db>>,
    transaction_manager: Arc<Mutex<TransactionManager>>,
}

impl AlgorithmManager {
    pub fn new(
        storage: Arc<RwLock<sled::Db>>,
        transaction_manager: Arc<Mutex<TransactionManager>>,
    ) -> Self {
        Self {
            storage,
            transaction_manager,
        }
    }

    /// 存储算法
    pub async fn store_algorithm(&self, algorithm_id: &str, algorithm: &crate::algorithm::types::Algorithm) -> Result<()> {
        let key = format!("algorithm:{}", algorithm_id);
        let data = bincode::serialize(algorithm)?;
        self.put_raw(key.as_bytes(), &data).await?;

        // 保存算法元数据
        let metadata = HashMap::from([
            ("id".to_string(), algorithm_id.to_string()),
            ("name".to_string(), algorithm.name.clone()),
            ("version".to_string(), algorithm.version.to_string()),
            ("created_at".to_string(), chrono::Utc::now().to_rfc3339()),
        ]);

        let metadata_key = format!("algorithm:{}:metadata", algorithm_id);
        let metadata_data = serde_json::to_vec(&metadata)?;
        self.put_raw(metadata_key.as_bytes(), &metadata_data).await
    }

    /// 加载算法
    pub async fn load_algorithm(&self, algorithm_id: &str) -> Result<Option<crate::algorithm::types::Algorithm>> {
        let key = format!("algorithm:{}", algorithm_id);
        if let Some(data) = self.get_raw(&key).await? {
            Ok(Some(bincode::deserialize(&data)?))
        } else {
            Ok(None)
        }
    }

    /// 删除算法
    pub async fn delete_algorithm(&self, algorithm_id: &str) -> Result<()> {
        let prefixes = vec![
            format!("algorithm:{}", algorithm_id),
            format!("algorithm:{}:metadata", algorithm_id),
        ];

        for prefix in prefixes {
            let entries = self.scan_prefix_raw(prefix.as_bytes())?;
            for (key, _) in entries {
                self.delete_raw(&key).await?;
            }
        }

        Ok(())
    }

    /// 列出所有算法
    pub async fn list_algorithms(&self) -> Result<Vec<String>> {
        let mut algorithms = Vec::new();
        let prefix = "algorithm:";
        
        let entries = self.scan_prefix_raw(prefix.as_bytes())?;
        for (key, _) in entries {
            if let Ok(key_str) = String::from_utf8(key) {
                if key_str.contains(":metadata") {
                    continue; // 跳过元数据键
                }
                if let Some(algorithm_id) = key_str.strip_prefix("algorithm:") {
                    algorithms.push(algorithm_id.to_string());
                }
            }
        }
        
        Ok(algorithms)
    }

    /// 列出过滤后的算法
    pub async fn list_algorithms_filtered(&self, category: Option<&str>, limit: Option<usize>) -> Result<Vec<crate::algorithm::types::Algorithm>> {
        let mut algorithms = Vec::new();
        let prefix = "algorithm:";
        
        let entries = self.scan_prefix_raw(prefix.as_bytes())?;
        for (key, value) in entries {
            if let Ok(key_str) = String::from_utf8(key) {
                if key_str.contains(":metadata") {
                    continue; // 跳过元数据键
                }
                
                if let Ok(algorithm) = bincode::deserialize::<crate::algorithm::types::Algorithm>(&value) {
                    // 应用分类过滤器
                    if let Some(cat) = category {
                        if algorithm.algorithm_type.to_string() != cat {
                            continue;
                        }
                    }
                    
                    algorithms.push(algorithm);
                    
                    // 应用限制
                    if let Some(lim) = limit {
                        if algorithms.len() >= lim {
                            break;
                        }
                    }
                }
            }
        }
        
        Ok(algorithms)
    }

    /// 检查算法是否存在
    pub async fn algorithm_exists(&self, algorithm_id: &str) -> Result<bool> {
        let key = format!("algorithm:{}", algorithm_id);
        self.exists_raw(key.as_bytes())
    }

    /// 获取算法统计信息
    pub async fn get_algorithm_stats(&self) -> Result<HashMap<String, usize>> {
        let mut stats = HashMap::new();
        let prefix = "algorithm:";
        
        let entries = self.scan_prefix_raw(prefix.as_bytes())?;
        for (key, value) in entries {
            if let Ok(key_str) = String::from_utf8(key) {
                if key_str.contains(":metadata") {
                    continue; // 跳过元数据键
                }
                
                if let Ok(algorithm) = bincode::deserialize::<crate::algorithm::types::Algorithm>(&value) {
                    let count = stats.entry(algorithm.algorithm_type.to_string()).or_insert(0);
                    *count += 1;
                }
            }
        }
        
        Ok(stats)
    }

    /// 搜索算法
    pub async fn search_algorithms(&self, query: &str) -> Result<Vec<crate::algorithm::types::Algorithm>> {
        let mut algorithms = Vec::new();
        let prefix = "algorithm:";
        
        let entries = self.scan_prefix_raw(prefix.as_bytes())?;
        for (key, value) in entries {
            if let Ok(key_str) = String::from_utf8(key) {
                if key_str.contains(":metadata") {
                    continue; // 跳过元数据键
                }
                
                if let Ok(algorithm) = bincode::deserialize::<crate::algorithm::types::Algorithm>(&value) {
                    // 搜索算法名称和描述
                    if algorithm.name.to_lowercase().contains(&query.to_lowercase()) ||
                       algorithm.description.as_ref().map_or(false, |desc| desc.to_lowercase().contains(&query.to_lowercase())) {
                        algorithms.push(algorithm);
                    }
                }
            }
        }
        
        Ok(algorithms)
    }

    /// 获取算法版本历史
    pub async fn get_algorithm_versions(&self, algorithm_id: &str) -> Result<Vec<String>> {
        let mut versions = Vec::new();
        let prefix = format!("algorithm:{}:version:", algorithm_id);
        
        let entries = self.scan_prefix_raw(prefix.as_bytes())?;
        for (key, _) in entries {
            if let Ok(key_str) = String::from_utf8(key) {
                if let Some(version) = key_str.strip_prefix(&format!("algorithm:{}:version:", algorithm_id)) {
                    versions.push(version.to_string());
                }
            }
        }
        
        Ok(versions)
    }

    /// 保存算法版本
    pub async fn save_algorithm_version(&self, algorithm_id: &str, version: &str, algorithm: &crate::algorithm::types::Algorithm) -> Result<()> {
        let key = format!("algorithm:{}:version:{}", algorithm_id, version);
        let data = bincode::serialize(algorithm)?;
        self.put_raw(key.as_bytes(), &data).await
    }

    /// 获取算法版本
    pub async fn get_algorithm_version(&self, algorithm_id: &str, version: &str) -> Result<Option<crate::algorithm::types::Algorithm>> {
        let key = format!("algorithm:{}:version:{}", algorithm_id, version);
        if let Some(data) = self.get_raw(&key).await? {
            Ok(Some(bincode::deserialize(&data)?))
        } else {
            Ok(None)
        }
    }

    /// 删除算法版本
    pub async fn delete_algorithm_version(&self, algorithm_id: &str, version: &str) -> Result<()> {
        let key = format!("algorithm:{}:version:{}", algorithm_id, version);
        self.delete_raw(key.as_bytes()).await
    }

    /// 获取算法使用统计
    pub async fn get_algorithm_usage_stats(&self, algorithm_id: &str) -> Result<HashMap<String, u64>> {
        let key = format!("algorithm:{}:usage_stats", algorithm_id);
        if let Some(data) = self.get_raw(&key).await? {
            Ok(serde_json::from_slice(&data)?)
        } else {
            Ok(HashMap::new())
        }
    }

    /// 更新算法使用统计
    pub async fn update_algorithm_usage_stats(&self, algorithm_id: &str, usage_type: &str) -> Result<()> {
        let mut stats = self.get_algorithm_usage_stats(algorithm_id).await?;
        let count = stats.entry(usage_type.to_string()).or_insert(0);
        *count += 1;
        
        let key = format!("algorithm:{}:usage_stats", algorithm_id);
        let data = serde_json::to_vec(&stats)?;
        self.put_raw(key.as_bytes(), &data).await
    }

    /// 获取算法依赖关系
    pub async fn get_algorithm_dependencies(&self, algorithm_id: &str) -> Result<Vec<String>> {
        let key = format!("algorithm:{}:dependencies", algorithm_id);
        if let Some(data) = self.get_raw(&key).await? {
            Ok(serde_json::from_slice(&data)?)
        } else {
            Ok(Vec::new())
        }
    }

    /// 保存算法依赖关系
    pub async fn save_algorithm_dependencies(&self, algorithm_id: &str, dependencies: &[String]) -> Result<()> {
        let key = format!("algorithm:{}:dependencies", algorithm_id);
        let data = serde_json::to_vec(dependencies)?;
        self.put_raw(key.as_bytes(), &data).await
    }

    /// 验证算法完整性
    pub async fn validate_algorithm(&self, algorithm_id: &str) -> Result<bool> {
        // 检查算法是否存在
        if !self.algorithm_exists(algorithm_id).await? {
            return Ok(false);
        }

        // 检查算法数据是否完整
        if let Some(algorithm) = self.load_algorithm(algorithm_id).await? {
            // 验证算法基本属性
            if algorithm.name.is_empty() || algorithm.version == 0 {
                return Ok(false);
            }

            // 检查依赖关系
            let dependencies = self.get_algorithm_dependencies(algorithm_id).await?;
            for dep_id in dependencies {
                if !self.algorithm_exists(&dep_id).await? {
                    return Ok(false);
                }
            }

            Ok(true)
        } else {
            Ok(false)
        }
    }

    /// 清理孤立的算法
    pub async fn cleanup_orphaned_algorithms(&self) -> Result<Vec<String>> {
        let mut orphaned = Vec::new();
        let algorithms = self.list_algorithms().await?;
        
        for algorithm_id in algorithms {
            if !self.validate_algorithm(&algorithm_id).await? {
                orphaned.push(algorithm_id);
            }
        }
        
        Ok(orphaned)
    }

    // 私有方法：底层存储操作
    async fn get_raw(&self, key: &str) -> Result<Option<Vec<u8>>> {
        let db = self.storage.read().await;
        Ok(db.get(key.as_bytes())?.map(|v| v.to_vec()))
    }

    async fn put_raw(&self, key: &[u8], value: &[u8]) -> Result<()> {
        let db = self.storage.write().await;
        db.insert(key, value)?;
        Ok(())
    }

    async fn delete_raw(&self, key: &[u8]) -> Result<()> {
        let db = self.storage.write().await;
        db.remove(key)?;
        Ok(())
    }

    fn exists_raw(&self, key: &[u8]) -> Result<bool> {
        let db = self.storage.try_read().map_err(|e| Error::LockError(format!("锁错误: {}", e)))?;
        Ok(db.contains_key(key)?)
    }

    fn scan_prefix_raw(&self, prefix: &[u8]) -> Result<Vec<(Vec<u8>, Vec<u8>)>> {
        let db = self.storage.try_read().map_err(|e| Error::LockError(format!("锁错误: {}", e)))?;
        let mut results = Vec::new();
        
        for result in db.scan_prefix(prefix) {
            let (key, value) = result?;
            results.push((key.to_vec(), value.to_vec()));
        }
        
        Ok(results)
    }
} 