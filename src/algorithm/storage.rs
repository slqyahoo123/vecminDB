use async_trait::async_trait;
use crate::error::{Error, Result};
use crate::storage::Storage;
use crate::algorithm::types::{Algorithm, AlgorithmTask, AlgorithmMetadata};
use std::sync::{Arc, RwLock};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use tokio::fs;
use serde_json;
use uuid::Uuid;
use log::{info, warn, error, debug};
use serde::{Serialize, Deserialize};
use crate::storage::config::StorageConfig;
use chrono;

/// 算法存储接口，用于算法的持久化管理
pub trait AlgorithmStorage: Send + Sync {
    /// 保存算法
    fn save_algorithm(&self, id: &str, algorithm: &Algorithm) -> Result<()>;
    
    /// 加载算法
    fn load_algorithm(&self, id: &str) -> Result<Algorithm>;
    
    /// 更新算法
    fn update_algorithm(&self, id: &str, algorithm: &Algorithm) -> Result<()>;
    
    /// 删除算法
    fn delete_algorithm(&self, id: &str) -> Result<()>;
    
    /// 检查算法是否存在
    fn algorithm_exists(&self, id: &str) -> Result<bool>;
    
    /// 列出所有算法
    fn list_algorithms(&self) -> Result<Vec<String>>;
}

/// 异步算法存储接口
#[async_trait]
pub trait AsyncAlgorithmStorage: Send + Sync {
    /// 异步保存算法
    async fn save_algorithm(&self, id: &str, algorithm: &Algorithm) -> Result<()>;
    
    /// 异步加载算法
    async fn load_algorithm(&self, id: &str) -> Result<Algorithm>;
    
    /// 异步更新算法
    async fn update_algorithm(&self, id: &str, algorithm: &Algorithm) -> Result<()>;
    
    /// 异步删除算法
    async fn delete_algorithm(&self, id: &str) -> Result<()>;
    
    /// 异步检查算法是否存在
    async fn algorithm_exists(&self, id: &str) -> Result<bool>;
    
    /// 异步列出所有算法
    async fn list_algorithms(&self) -> Result<Vec<String>>;
    
    /// 异步导出算法元数据
    async fn export_metadata(&self, id: &str) -> Result<AlgorithmMetadata>;
    
    /// 异步导入算法任务
    async fn import_task(&self, task: &AlgorithmTask) -> Result<()>;
    
    /// 异步更新任务索引
    async fn update_task_index(&self, task_id: &str, algorithm_id: &str) -> Result<()> {
        // 默认实现，可由具体实现类重写
        debug!("默认任务索引更新实现");
        Ok(())
    }
}

/// 本地算法存储实现
pub struct LocalAlgorithmStorage {
    /// 存储配置
    config: StorageConfig,
    /// 核心存储引用
    storage: Arc<crate::storage::Storage>,
    /// 内存缓存
    cache: RwLock<HashMap<String, Algorithm>>,
}

impl LocalAlgorithmStorage {
    /// 创建新的本地算法存储
    pub fn new(storage: Arc<crate::storage::Storage>, config: StorageConfig) -> Self {
        debug!("创建新的本地算法存储");
        Self {
            config,
            storage,
            cache: RwLock::new(HashMap::new()),
        }
    }
    
    /// 生成算法存储键
    fn algorithm_key(&self, id: &str) -> String {
        format!("algorithm:{}", id)
    }
    
    /// 从文件系统加载算法
    pub fn load_from_file(&self, path: &Path) -> Result<Algorithm> {
        debug!("从文件加载算法: {:?}", path);
        let data = std::fs::read_to_string(path)
            .map_err(|e| Error::io(format!("无法读取算法文件: {}", e)))?;
        
        let algorithm: Algorithm = serde_json::from_str(&data)
            .map_err(|e| Error::invalid_input(format!("无法解析算法文件: {}", e)))?;
        
        Ok(algorithm)
    }
    
    /// 保存算法到文件系统
    pub fn save_to_file(&self, algorithm: &Algorithm, path: &Path) -> Result<()> {
        debug!("保存算法到文件: {:?}", path);
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)
                .map_err(|e| Error::io(format!("无法创建目录: {}", e)))?;
        }
        
        let data = serde_json::to_string_pretty(algorithm)
            .map_err(|e| Error::internal(format!("无法序列化算法: {}", e)))?;
        
        std::fs::write(path, data)
            .map_err(|e| Error::io(format!("无法写入算法文件: {}", e)))?;
        
        Ok(())
    }
    
    /// 创建算法副本
    pub fn clone_algorithm(&self, id: &str, new_id: &str) -> Result<Algorithm> {
        info!("创建算法副本: {} -> {}", id, new_id);
        let algorithm = self.load_algorithm(id)?;
        
        // 创建副本需要生成新的UUID
        let mut cloned = algorithm.clone();
        cloned.id = Uuid::new_v4().to_string();
        
        self.save_algorithm(new_id, &cloned)?;
        
        Ok(cloned)
    }
}

impl AlgorithmStorage for LocalAlgorithmStorage {
    fn save_algorithm(&self, id: &str, algorithm: &Algorithm) -> Result<()> {
        info!("保存算法: {}", id);
        // 序列化算法
        let serialized = bincode::serialize(algorithm)?;
        
        // 保存到存储引擎
        let key = self.algorithm_key(id);
        self.storage.put(&key, &serialized)?;
        
        // 更新缓存
        {
            let mut cache = self.cache.write().map_err(|_| 
                Error::internal("无法获取缓存写锁"))?;
            cache.insert(id.to_string(), algorithm.clone());
        }
        
        debug!("算法保存成功: {}", id);
        Ok(())
    }
    
    fn load_algorithm(&self, id: &str) -> Result<Algorithm> {
        debug!("加载算法: {}", id);
        // 先检查缓存
        {
            let cache = self.cache.read().map_err(|_| 
                Error::internal("无法获取缓存读锁"))?;
            if let Some(algorithm) = cache.get(id) {
                debug!("从缓存加载算法: {}", id);
                return Ok(algorithm.clone());
            }
        }
        
        // 从存储引擎加载
        let key = self.algorithm_key(id);
        let value = self.storage.get(&key)?
            .ok_or_else(|| {
                warn!("算法不存在: {}", id);
                Error::not_found(format!("算法不存在: {}", id))
            })?;
        
        // 反序列化
        let algorithm: Algorithm = bincode::deserialize(&value)?;
        
        // 更新缓存
        {
            let mut cache = self.cache.write().map_err(|_| 
                Error::internal("无法获取缓存写锁"))?;
            cache.insert(id.to_string(), algorithm.clone());
        }
        
        info!("成功加载算法: {}", id);
        Ok(algorithm)
    }
    
    fn update_algorithm(&self, id: &str, algorithm: &Algorithm) -> Result<()> {
        info!("更新算法: {}", id);
        // 检查算法是否存在
        if !self.algorithm_exists(id)? {
            warn!("尝试更新不存在的算法: {}", id);
            return Err(Error::not_found(format!("算法不存在: {}", id)));
        }
        
        // 重用保存方法
        self.save_algorithm(id, algorithm)
    }
    
    fn delete_algorithm(&self, id: &str) -> Result<()> {
        info!("删除算法: {}", id);
        // 从存储引擎删除
        let key = self.algorithm_key(id);
        self.storage.delete(&key)?;
        
        // 从缓存中删除
        {
            let mut cache = self.cache.write().map_err(|_| 
                Error::internal("无法获取缓存写锁"))?;
            cache.remove(id);
        }
        
        debug!("算法删除成功: {}", id);
        Ok(())
    }
    
    fn algorithm_exists(&self, id: &str) -> Result<bool> {
        debug!("检查算法是否存在: {}", id);
        // 检查缓存
        {
            let cache = self.cache.read().map_err(|_| 
                Error::internal("无法获取缓存读锁"))?;
            if cache.contains_key(id) {
                return Ok(true);
            }
        }
        
        // 检查存储引擎
        let key = self.algorithm_key(id);
        self.storage.exists(&key)
    }
    
    fn list_algorithms(&self) -> Result<Vec<String>> {
        debug!("列出所有算法");
        // 获取前缀匹配
        let prefix = "algorithm:";
        let keys = self.storage.list_keys_with_prefix(prefix)?;
        
        // 提取算法ID
        let mut algorithm_ids = Vec::new();
        for key in keys {
            if let Some(id) = key.strip_prefix(prefix) {
                algorithm_ids.push(id.to_string());
            }
        }
        
        info!("找到 {} 个算法", algorithm_ids.len());
        Ok(algorithm_ids)
    }
}

/// 文件系统算法存储实现
pub struct FileSystemAlgorithmStorage {
    base_path: PathBuf,
    cache: RwLock<HashMap<String, Algorithm>>,
}

impl FileSystemAlgorithmStorage {
    pub fn new(base_path: PathBuf) -> Self {
        info!("创建文件系统算法存储: {:?}", base_path);
        // 确保目录存在
        if !base_path.exists() {
            std::fs::create_dir_all(&base_path).unwrap_or_else(|e| {
                error!("无法创建算法存储目录: {}", e);
            });
        }
        
        Self {
            base_path,
            cache: RwLock::new(HashMap::new()),
        }
    }
    
    fn algorithm_path(&self, id: &str) -> PathBuf {
        self.base_path.join(format!("{}.json", id))
    }
}

#[async_trait]
impl AsyncAlgorithmStorage for FileSystemAlgorithmStorage {
    async fn save_algorithm(&self, id: &str, algorithm: &Algorithm) -> Result<()> {
        info!("异步保存算法: {}", id);
        let path = self.algorithm_path(id);
        let data = serde_json::to_string_pretty(algorithm)
            .map_err(|e| Error::internal(format!("无法序列化算法: {}", e)))?;
        
        fs::write(path, data).await
            .map_err(|e| Error::io(format!("无法写入算法文件: {}", e)))?;
        
        // 更新缓存
        {
            let mut cache = self.cache.write().map_err(|_| 
                Error::internal("无法获取缓存写锁"))?;
            cache.insert(id.to_string(), algorithm.clone());
        }
        
        debug!("异步算法保存成功: {}", id);
        Ok(())
    }
    
    async fn load_algorithm(&self, id: &str) -> Result<Algorithm> {
        debug!("异步加载算法: {}", id);
        // 先检查缓存
        {
            let cache = self.cache.read().map_err(|_| 
                Error::internal("无法获取缓存读锁"))?;
            if let Some(algorithm) = cache.get(id) {
                debug!("从缓存异步加载算法: {}", id);
                return Ok(algorithm.clone());
            }
        }
        
        // 从文件加载
        let path = self.algorithm_path(id);
        let data = fs::read_to_string(path).await
            .map_err(|e| {
                warn!("无法读取算法文件: {}, 错误: {}", id, e);
                Error::not_found(format!("算法不存在: {}", id))
            })?;
        
        let algorithm: Algorithm = serde_json::from_str(&data)
            .map_err(|e| Error::invalid_input(format!("无法解析算法文件: {}", e)))?;
        
        // 更新缓存
        {
            let mut cache = self.cache.write().map_err(|_| 
                Error::internal("无法获取缓存写锁"))?;
            cache.insert(id.to_string(), algorithm.clone());
        }
        
        info!("成功异步加载算法: {}", id);
        Ok(algorithm)
    }
    
    async fn update_algorithm(&self, id: &str, algorithm: &Algorithm) -> Result<()> {
        info!("异步更新算法: {}", id);
        // 检查算法是否存在
        if !self.algorithm_exists(id).await? {
            warn!("尝试异步更新不存在的算法: {}", id);
            return Err(Error::not_found(format!("算法不存在: {}", id)));
        }
        
        // 重用保存方法
        self.save_algorithm(id, algorithm).await
    }
    
    async fn delete_algorithm(&self, id: &str) -> Result<()> {
        info!("异步删除算法: {}", id);
        // 从文件系统删除
        let path = self.algorithm_path(id);
        fs::remove_file(path).await
            .map_err(|e| Error::io(format!("无法删除算法文件: {}", e)))?;
        
        // 从缓存中删除
        {
            let mut cache = self.cache.write().map_err(|_| 
                Error::internal("无法获取缓存写锁"))?;
            cache.remove(id);
        }
        
        debug!("异步算法删除成功: {}", id);
        Ok(())
    }
    
    async fn algorithm_exists(&self, id: &str) -> Result<bool> {
        debug!("异步检查算法是否存在: {}", id);
        // 检查缓存
        {
            let cache = self.cache.read().map_err(|_| 
                Error::internal("无法获取缓存读锁"))?;
            if cache.contains_key(id) {
                return Ok(true);
            }
        }
        
        // 检查文件系统
        let path = self.algorithm_path(id);
        Ok(path.exists())
    }
    
    async fn list_algorithms(&self) -> Result<Vec<String>> {
        debug!("异步列出所有算法");
        let mut entries = fs::read_dir(&self.base_path).await
            .map_err(|e| Error::io(format!("无法读取算法目录: {}", e)))?;
        
        let mut algorithm_ids = Vec::new();
        while let Ok(Some(entry)) = entries.next_entry().await {
            if let Ok(file_type) = entry.file_type().await {
                if file_type.is_file() {
                    if let Some(file_name) = entry.file_name().to_str() {
                        if file_name.ends_with(".json") {
                            if let Some(id) = file_name.strip_suffix(".json") {
                                algorithm_ids.push(id.to_string());
                            }
                        }
                    }
                }
            }
        }
        
        info!("异步找到 {} 个算法", algorithm_ids.len());
        Ok(algorithm_ids)
    }
    
    /// 异步导出算法元数据
    async fn export_metadata(&self, id: &str) -> Result<AlgorithmMetadata> {
        info!("导出算法元数据: {}", id);
        
        // 加载算法
        let algorithm = self.load_algorithm(id).await?;
        
        // 创建导出路径
        let export_path = self.base_path.join("exports").join(format!("{}_metadata.json", id));
        debug!("导出路径: {:?}", export_path);
        
        // 确保导出目录存在
        if let Some(parent) = export_path.parent() {
            if !parent.exists() {
                fs::create_dir_all(parent).await
                    .map_err(|e| Error::io(format!("无法创建导出目录: {}", e)))?;
                debug!("创建导出目录: {:?}", parent);
            }
        }
        
        // 从算法创建元数据
        let metadata = AlgorithmMetadata::from(algorithm);
        
        // 将元数据序列化并写入文件
        let json_data = serde_json::to_string_pretty(&metadata)
            .map_err(|e| Error::internal(format!("无法序列化元数据: {}", e)))?;
            
        fs::write(&export_path, json_data).await
            .map_err(|e| Error::io(format!("无法写入元数据文件: {}", e)))?;
            
        info!("成功导出算法元数据: {}, 路径: {:?}", id, export_path);
        
        Ok(metadata)
    }
    
    /// 异步导入算法任务
    async fn import_task(&self, task: &AlgorithmTask) -> Result<()> {
        debug!("导入算法任务: {}", task.id);
        
        // 检查任务对应的算法是否存在
        if !self.algorithm_exists(&task.algorithm_id).await? {
            warn!("任务对应的算法不存在: {}", task.algorithm_id);
            return Err(Error::not_found(format!("算法不存在: {}", task.algorithm_id)));
        }
        
        // 创建任务导入路径
        let task_path = self.base_path.join("tasks").join(format!("{}.json", task.id));
        debug!("任务导入路径: {:?}", task_path);
        
        // 确保任务目录存在
        if let Some(parent) = task_path.parent() {
            if !parent.exists() {
                fs::create_dir_all(parent).await
                    .map_err(|e| Error::io(format!("无法创建任务目录: {}", e)))?;
                debug!("创建任务目录: {:?}", parent);
            }
        }
        
        // 序列化任务并写入文件
        let json_data = serde_json::to_string_pretty(task)
            .map_err(|e| Error::internal(format!("无法序列化任务: {}", e)))?;
            
        fs::write(&task_path, json_data).await
            .map_err(|e| Error::io(format!("无法写入任务文件: {}", e)))?;
            
        info!("成功导入算法任务: {}, 路径: {:?}", task.id, task_path);
        
        // 更新相关缓存和索引
        self.update_task_index(&task.id, &task.algorithm_id).await?;
        
        Ok(())
    }
    
    // 添加一个新的辅助方法，用于更新任务索引
    async fn update_task_index(&self, task_id: &str, algorithm_id: &str) -> Result<()> {
        debug!("更新任务索引: {}", task_id);
        
        // 创建索引文件路径
        let index_path = self.base_path.join("indexes").join("tasks_index.json");
        
        // 确保索引目录存在
        if let Some(parent) = index_path.parent() {
            if !parent.exists() {
                fs::create_dir_all(parent).await
                    .map_err(|e| Error::io(format!("无法创建索引目录: {}", e)))?;
                debug!("创建索引目录: {:?}", parent);
            }
        }
        
        // 读取现有索引或创建新索引
        let mut index: HashMap<String, String> = if index_path.exists() {
            match fs::read_to_string(&index_path).await {
                Ok(data) => serde_json::from_str(&data)
                    .unwrap_or_else(|_| {
                        warn!("无法解析索引文件，创建新索引");
                        HashMap::new()
                    }),
                Err(e) => {
                    warn!("无法读取索引文件: {}, 创建新索引", e);
                    HashMap::new()
                }
            }
        } else {
            debug!("索引文件不存在，创建新索引");
            HashMap::new()
        };
        
        // 更新索引
        index.insert(task_id.to_string(), algorithm_id.to_string());
        
        // 保存更新后的索引
        let json_data = serde_json::to_string_pretty(&index)
            .map_err(|e| Error::internal(format!("无法序列化索引: {}", e)))?;
            
        fs::write(&index_path, json_data).await
            .map_err(|e| Error::io(format!("无法写入索引文件: {}", e)))?;
            
        debug!("成功更新任务索引");
        
        Ok(())
    }
}

/// 算法元数据序列化和反序列化实现
impl AlgorithmMetadata {
    /// 序列化元数据为JSON字符串
    pub fn to_json(&self) -> Result<String> {
        serde_json::to_string_pretty(self)
            .map_err(|e| Error::serialization(format!("无法序列化算法元数据: {}", e)))
    }
    
    /// 从JSON字符串反序列化元数据
    pub fn from_json(json: &str) -> Result<Self> {
        serde_json::from_str(json)
            .map_err(|e| Error::deserialization(format!("无法反序列化算法元数据: {}", e)))
    }
}

/// 从Algorithm创建AlgorithmMetadata的转换实现
impl From<Algorithm> for AlgorithmMetadata {
    fn from(algorithm: Algorithm) -> Self {
        use crate::algorithm::management_types::ParameterDefinition;
        use crate::algorithm::management_types::ParameterType;
        use crate::algorithm::management_types::ValidationStatus;
        
        // 创建参数定义
        let mut parameters = std::collections::HashMap::new();
        for (key, value) in &algorithm.metadata {
            parameters.insert(key.clone(), ParameterDefinition {
                name: key.clone(),
                parameter_type: ParameterType::String, // 简化处理，实际应该根据值类型判断
                required: false,
                default_value: Some(serde_json::Value::String(value.clone())),
                description: None,
                constraints: None,
            });
        }
        
        Self {
            id: algorithm.id.clone(),
            name: algorithm.name.clone(),
            version: algorithm.version.to_string(),
            description: Some(algorithm.description.clone()),
            author: algorithm.metadata.get("author").cloned(),
            tags: algorithm.metadata.get("tags")
                .and_then(|t| t.split(',').map(|s| s.trim().to_string()).collect::<Vec<_>>().into())
                .unwrap_or_default(),
            algorithm_type: algorithm.algorithm_type.clone(),
            created_at: chrono::Utc::now(), // 使用当前时间，实际应该从算法中获取
            updated_at: chrono::Utc::now(),
            dependencies: algorithm.metadata.get("dependencies")
                .and_then(|d| serde_json::from_str(d).ok())
                .unwrap_or_default(),
            parameters,
            performance_metrics: None, // 需要从算法执行历史中获取
            validation_status: ValidationStatus::Pending,
        }
    }
}

/// 算法备份数据结构
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct AlgorithmBackup {
    /// 算法ID
    pub id: String,
    /// 算法数据
    pub algorithm: Option<Algorithm>,
    /// 元数据
    pub metadata: AlgorithmMetadata,
    /// 备份时间戳
    pub timestamp: u64,
    /// 版本信息
    pub version: String,
    /// 备份描述
    pub description: Option<String>,
}

impl AlgorithmBackup {
    /// 创建新的算法备份
    pub fn new(id: String, algorithm: Algorithm, metadata: AlgorithmMetadata) -> Self {
        Self {
            id,
            algorithm: Some(algorithm),
            metadata,
            timestamp: chrono::Utc::now().timestamp() as u64,
            version: "1.0".to_string(),
            description: None,
        }
    }
    
    /// 设置备份描述
    pub fn with_description(mut self, description: String) -> Self {
        self.description = Some(description);
        self
    }
    
    /// 序列化为JSON字符串
    pub fn to_json(&self) -> Result<String> {
        serde_json::to_string_pretty(self)
            .map_err(|e| Error::serialization(format!("无法序列化算法备份: {}", e)))
    }
    
    /// 从JSON字符串反序列化
    pub fn from_json(json: &str) -> Result<Self> {
        serde_json::from_str(json)
            .map_err(|e| Error::deserialization(format!("无法反序列化算法备份: {}", e)))
    }
    
    /// 创建不包含算法数据的轻量级备份
    pub fn lightweight_backup(id: String, metadata: AlgorithmMetadata) -> Self {
        Self {
            id,
            algorithm: None,
            metadata,
            timestamp: chrono::Utc::now().timestamp() as u64,
            version: "1.0".to_string(),
            description: None,
        }
    }
}

/// 文件系统算法存储实现
/// 添加数据序列化备份功能
impl FileSystemAlgorithmStorage {
    /// 创建算法数据的序列化备份
    pub fn create_serialized_backup(&self, algorithm: &Algorithm) -> Result<Vec<u8>> {
        debug!("创建算法序列化备份");
        
        // 使用serde的Serialize特性将算法序列化为JSON格式
        let serialized = serde_json::to_vec_pretty(algorithm)
            .map_err(|e| Error::internal(format!("序列化算法失败: {}", e)))?;
            
        Ok(serialized)
    }
    
    /// 从序列化数据恢复算法
    pub fn restore_from_serialized(&self, data: &[u8]) -> Result<Algorithm> {
        debug!("从序列化数据恢复算法");
        
        // 使用serde的Deserialize特性从JSON数据恢复算法对象
        let algorithm: Algorithm = serde_json::from_slice(data)
            .map_err(|e| Error::internal(format!("反序列化算法失败: {}", e)))?;
            
        Ok(algorithm)
    }
    
    /// 使用Storage进行算法数据备份
    pub async fn backup_algorithm_to_storage(&self, id: &str, storage: &Storage) -> Result<()> {
        info!("备份算法到中央存储: {}", id);
        
        // 从文件系统加载算法
        let algorithm = self.load_algorithm(id).await?;
        
        // 创建元数据
        let metadata = AlgorithmMetadata::from(algorithm.clone());
        
        // 创建备份对象
        let backup = AlgorithmBackup::new(id.to_string(), algorithm, metadata);
        
        // 序列化备份对象
        let serialized = serde_json::to_vec(&backup)
            .map_err(|e| Error::internal(format!("序列化备份对象失败: {}", e)))?;
        
        // 使用Storage存储备份数据
        let backup_key = format!("algorithm_backup:{}", id);
        storage.put_raw(backup_key.as_str(), &serialized).await?;
        
        info!("算法备份完成: {}", id);
        Ok(())
    }
    
    /// 从Storage恢复算法数据
    pub async fn restore_algorithm_from_storage(&self, id: &str, storage: &Storage) -> Result<Algorithm> {
        info!("从中央存储恢复算法: {}", id);
        
        // 从Storage获取备份数据
        let backup_key = format!("algorithm_backup:{}", id);
        let backup_data = storage.get_raw(backup_key.as_str()).await?
            .ok_or_else(|| Error::not_found(format!("算法备份不存在: {}", id)))?;
        
        // 反序列化备份对象
        let backup: AlgorithmBackup = serde_json::from_slice(&backup_data)
            .map_err(|e| Error::internal(format!("反序列化备份对象失败: {}", e)))?;
        
        // 从备份中获取算法
        let algorithm = backup.algorithm
            .ok_or_else(|| Error::internal(format!("备份中不包含算法数据: {}", id)))?;
        
        // 保存到本地文件系统
        self.save_algorithm(id, &algorithm).await?;
        
        info!("算法恢复完成: {}", id);
        Ok(algorithm)
    }
} 