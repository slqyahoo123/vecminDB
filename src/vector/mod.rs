use crate::Result;
use std::path::Path;
use std::collections::HashMap;
use std::sync::Arc;
use uuid::Uuid;
use tokio::sync::RwLock;
// use uuid::Uuid; // reserved for future collection IDs or vector IDs generation
use crate::Error;
use crate::{Component, ComponentType, ComponentStatus};
use crate::vector::index::FlatIndex;
// IndexConfig和SimilarityMetric在下方统一重导出

// 子模块
pub mod types;
pub mod core;
pub mod index;
pub mod search;
pub mod search_options;
pub mod utils;
pub mod parallel;
pub mod feature;
pub mod optimizer;

// 重导出常用类型和函数
pub use self::types::{
    Vector, VectorList, VectorId, VectorEntry, DistanceFunction,
    VectorOperationResult, VectorBatchResult, VectorStats
};
pub use self::search::{
    VectorSearchParams, VectorSearchResult, VectorMetadata, 
    VectorQuery, VectorCollection, VectorCollectionConfig
};
pub use self::index::{VectorIndex, IndexParams, IndexType, IndexStatus, SearchResult};
pub use self::search_options::{
    SearchOptions, FilterOptions, FilterCondition, FilterOperator, FilterValue, FilterLogic,
    SortOrder, RerankOptions, VectorSearchQuery
};
// 统一从core模块导出SimilarityMetric，避免重复导入
pub use self::core::operations::SimilarityMetric;
pub use self::parallel::{ParallelConfig, WorkerPool, TaskType, BatchProcessor, VectorPipeline};
pub use self::feature::{
    FeatureSet, FeatureDescriptor, FeatureType, FeatureExtractionConfig,
    FeatureManager, extraction::FeatureExtractor, reduction::DimensionReducer,
    transform::VectorTransformer
};

/// 生成全局唯一的向量或集合ID
pub fn generate_uuid() -> String {
    Uuid::new_v4().to_string()
}

// 统一向量服务抽象（契约）
// 说明：为上层（如 API 层）提供统一的向量服务接口，实现与具体实现（VectorDB、VectorEngine 等）的解耦。
// 当前先提供最小稳定契约；后续可逐步扩展（如集合管理、检索、索引维护等异步接口）。
#[async_trait::async_trait]
pub trait VectorService: Send + Sync {
    async fn add_vector(&self, id: &str, values: &[f32], metadata: Option<&serde_json::Value>) -> Result<()>;
    async fn get_vector(&self, id: &str) -> Result<Option<(Vec<f32>, Option<crate::vector::search::VectorMetadata>)>>;
    async fn delete_vector(&self, id: &str) -> Result<bool>;
    async fn search_vectors(
        &self,
        query_vector: &[f32],
        top_k: usize,
        metric: &str,
        filter: Option<&serde_json::Value>,
    ) -> Result<Vec<(String, f32, Vec<f32>, Option<crate::vector::search::VectorMetadata>)>>;
    async fn count_vectors(&self) -> Result<usize>;
    async fn rebuild_index(&self) -> Result<()>;
    async fn update_vector(&self, id: &str, values: &[f32], metadata: Option<&serde_json::Value>) -> Result<()>;
    async fn get_stats(&self) -> Result<VectorStats>;
}

// 用 VectorStorageManager 作为 VectorService 的一个实现
#[async_trait::async_trait]
impl VectorService for crate::vector::storage::storage::VectorStorageManager {
    async fn add_vector(&self, id: &str, values: &[f32], metadata: Option<&serde_json::Value>) -> Result<()> {
        self.add_vector(id, values, metadata).await
    }

    async fn get_vector(&self, id: &str) -> Result<Option<(Vec<f32>, Option<crate::vector::search::VectorMetadata>)>> {
        self.get_vector(id).await
    }

    async fn delete_vector(&self, id: &str) -> Result<bool> {
        self.delete_vector(id).await
    }

    async fn search_vectors(
        &self,
        query_vector: &[f32],
        top_k: usize,
        metric: &str,
        filter: Option<&serde_json::Value>,
    ) -> Result<Vec<(String, f32, Vec<f32>, Option<crate::vector::search::VectorMetadata>)>> {
        self.search_vectors(query_vector, top_k, metric, filter).await
    }

    async fn count_vectors(&self) -> Result<usize> {
        self.count_vectors().await
    }

    async fn rebuild_index(&self) -> Result<()> {
        self.rebuild_index_async().await
    }
    
    async fn update_vector(&self, id: &str, values: &[f32], metadata: Option<&serde_json::Value>) -> Result<()> {
        // 先删除旧向量，再添加新向量
        self.delete_vector(id).await?;
        self.add_vector(id, values, metadata).await
    }
    
    async fn get_stats(&self) -> Result<VectorStats> {
        let count = self.count_vectors().await?;
        // 暂时使用默认维度 128，后续可根据集合实际维度统计
        let dimension = if count > 0 { 128 } else { 0 };
        let index_type = "Flat".to_string();
        
        Ok(VectorStats {
            count,
            dimension,
            index_type,
            memory_usage: Some(0), // 暂时返回0，需要实际实现
        })
    }
}

// 便捷别名，便于在调用侧声明参数类型
pub type DynVectorService = dyn VectorService + Send + Sync;

/// 向量组件
pub struct VectorComponent {
    // 组件配置和状态管理
    status: ComponentStatus,
    config: VectorComponentConfig,
    
    // 向量集合
    collections: std::collections::HashMap<String, VectorCollection>,
}

/// 向量组件配置
#[derive(Debug, Clone)]
pub struct VectorComponentConfig {
    /// 最大向量集合数量
    pub max_collections: usize,
    /// 默认相似度度量
    pub default_metric: SimilarityMetric,
    /// 是否启用并行处理
    pub parallel_enabled: bool,
    /// 并行处理线程数
    pub parallel_threads: usize,
}

impl Default for VectorComponentConfig {
    fn default() -> Self {
        Self {
            max_collections: 100,
            default_metric: SimilarityMetric::Cosine,
            parallel_enabled: true,
            parallel_threads: num_cpus::get(),
        }
    }
}

impl VectorComponent {
    /// 创建新的向量组件
    pub fn new(config: VectorComponentConfig) -> Self {
        Self {
            status: ComponentStatus::Initializing,
            config,
            collections: std::collections::HashMap::new(),
        }
    }
    
    /// 创建使用默认配置的向量组件
    pub fn default() -> Self {
        Self::new(VectorComponentConfig::default())
    }
    
    /// 创建新的向量集合
    pub fn create_collection(&mut self, name: &str, config: VectorCollectionConfig) -> Result<()> {
        if self.collections.contains_key(name) {
            return Err(Error::vector(format!("Collection '{}' already exists", name)));
        }
        
        if self.collections.len() >= self.config.max_collections {
            return Err(Error::vector(format!(
                "Maximum number of collections reached: {}", 
                self.config.max_collections
            )));
        }
        
        // 构造索引配置并创建默认的 Flat 索引
        let mut index_config = IndexConfig::default();
        index_config.index_type = config.index_type;
        index_config.dimension = config.dimension;
        index_config.similarity_metric = SimilarityMetric::Euclidean;
        index_config.metric = index_config.similarity_metric;

        let index = Box::new(FlatIndex::new(index_config.clone()));
        let mut collection_config = config.clone();
        collection_config.name = name.to_string();

        let collection = VectorCollection::new(collection_config, index);
        self.collections.insert(name.to_string(), collection);
        
        Ok(())
    }
    
    /// 获取向量集合
    pub fn get_collection(&self, name: &str) -> Result<&VectorCollection> {
        self.collections.get(name)
            .ok_or_else(|| Error::vector(format!("Collection '{}' not found", name)))
    }
    
    /// 获取可变向量集合
    pub fn get_collection_mut(&mut self, name: &str) -> Result<&mut VectorCollection> {
        self.collections.get_mut(name)
            .ok_or_else(|| Error::vector(format!("Collection '{}' not found", name)))
    }
    
    /// 删除向量集合
    pub fn delete_collection(&mut self, name: &str) -> Result<()> {
        if self.collections.remove(name).is_none() {
            return Err(Error::vector(format!("Collection '{}' not found", name)));
        }
        
        Ok(())
    }
    
    /// 获取所有集合名称
    pub fn list_collections(&self) -> Vec<String> {
        self.collections.keys().cloned().collect()
    }
    
    /// 获取向量特征管理器
    pub fn feature_manager(&self) -> FeatureManager {
        FeatureManager::new()
    }
}

impl Component for VectorComponent {
    fn name(&self) -> &str {
        "vector_component"
    }
    
    fn component_type(&self) -> ComponentType {
        ComponentType::Vector
    }
    
    fn status(&self) -> ComponentStatus {
        self.status
    }
    
    fn start(&mut self) -> Result<()> {
        // 运行轻量就绪检查：验证各索引类型的工厂接线
        crate::vector::index::factory::VectorIndexFactory::readiness_probe()?;
        self.status = ComponentStatus::Running;
        Ok(())
    }
    
    fn stop(&mut self) -> Result<()> {
        self.status = ComponentStatus::Shutdown;
        Ok(())
    }
}

// 存储模块
pub mod storage;

// 类型和索引模块
// pub mod search_options;

// 重组后的结构
// 对外保持模块导出的一致性
pub mod distance {
    pub use super::core::distance::*;
}

pub mod ops {
    pub use super::core::ops::*;
}

pub mod operations {
    pub use super::core::operations::*;
}

pub mod benchmark {
    pub use super::utils::benchmark::*;
}

pub mod examples {
    pub use super::utils::examples::*;
}

// 新增向量优化器的导出
pub mod optimizers {
    pub use super::optimizer::{
        IndexOptimizer, OptimizerConfig, OptimizationTarget, OptimizationResult,
        run_optimizer_example, create_optimizer,
        // 自动调优相关导出
        VectorIndexAutoTuner, AutoTuneConfig, create_vector_optimizer, run_auto_tune_example,
        // 参数空间相关导出
        ParameterRange, ParameterSpace, ParameterValue, ParameterType, Parameter,
        // 多目标优化相关导出
        MultiObjectiveAlgorithm, MultiObjectiveConfig, MultiObjectiveOptimizer, MultiObjectiveResult,
    };
}

// 向量索引相关导出使用命名空间方式
pub mod indexes {
    pub use super::index::{
        types::{IndexType, IndexConfig, SearchResult},
        flat::FlatIndex,
        hnsw::HNSWIndex,
        ivf::IVFIndex,
        pq::PQIndex,
        ivfpq::IVFPQIndex,
        ivfhnsw::IVFHNSWIndex,
        lsh::LSHIndex,
        annoy::ANNOYIndex,
        kmeans::KMeansIndex,
        ngt::NGTIndex,
        hierarchical_clustering::HierarchicalClusteringIndex,
        graph_index::GraphIndex,
        vptree::VPTreeIndex,
        distance::Distance,
        VectorIndex,
        VectorIndexEnum,
    };
}

// 搜索相关导出
pub mod search_types {
    pub use super::search::{
        VectorQuery, VectorSearchResult, VectorCollection, VectorCollectionConfig, 
        VectorMetadata, VectorIndexFactory
    };
}

/// 向量数据库，管理多个向量集合
#[derive(Debug)]
pub struct VectorDB {
    collections: HashMap<String, Arc<RwLock<VectorCollection>>>,
    storage_path: Option<String>,
}

impl VectorDB {
    /// 创建新的向量数据库
    /// 
    /// # 参数
    /// * `path` - 存储路径，用于持久化数据
    /// 
    /// # 示例
    /// ```
    /// use vecmindb::VectorDB;
    /// let db = VectorDB::new("./data")?;
    /// ```
    pub fn new(path: impl AsRef<std::path::Path>) -> Result<Self> {
        let path_str = path.as_ref().to_string_lossy().to_string();
        // 确保目录存在
        std::fs::create_dir_all(&path_str)
            .map_err(|e| Error::Io(e))?;
        
        Ok(Self {
            collections: HashMap::new(),
            storage_path: Some(path_str),
        })
    }

    /// 创建新的向量数据库（无持久化存储）
    pub fn new_in_memory() -> Self {
        Self {
            collections: HashMap::new(),
            storage_path: None,
        }
    }

    /// 使用存储路径创建向量数据库（向后兼容）
    pub fn with_storage(path: &str) -> Self {
        Self {
            collections: HashMap::new(),
            storage_path: Some(path.to_string()),
        }
    }

    pub async fn collection_exists(&self, name: &str) -> Result<bool> {
        Ok(self.collections.contains_key(name))
    }

    /// 创建集合（便捷方法，匹配README示例）
    /// 
    /// # 参数
    /// * `name` - 集合名称
    /// * `dimension` - 向量维度
    /// * `index_type` - 索引类型
    /// 
    /// # 示例
    /// ```
    /// use vecmindb::{VectorDB, IndexType};
    /// let mut db = VectorDB::new("./data")?;
    /// let collection = db.create_collection("my_vectors", 128, IndexType::HNSW)?;
    /// ```
    pub fn create_collection(
        &mut self,
        name: &str,
        dimension: usize,
        index_type: IndexType,
    ) -> Result<Arc<RwLock<VectorCollection>>> {
        // 检查集合是否已存在
        if self.collections.contains_key(name) {
            return Err(Error::AlreadyExists(format!(
                "Collection already exists: {}", name
            )));
        }
        
        // 创建集合配置
        let config = VectorCollectionConfig {
            name: name.to_string(),
            dimension,
            index_type,
            metadata_schema: None,
        };
        
        // 创建索引配置
        use crate::vector::index::types::IndexConfig;
        
        // 注意：HNSW 索引当前未完全集成到 VectorIndex trait
        // 如果请求 HNSW，暂时使用 Flat 索引作为替代
        let actual_index_type = if index_type == IndexType::HNSW {
            eprintln!("Warning: HNSW index is not fully integrated, using Flat index as fallback");
            IndexType::Flat
        } else {
            index_type
        };
        
        let index_config = IndexConfig {
            index_type: actual_index_type,
            metric: SimilarityMetric::Cosine, // 默认使用余弦相似度
            dimension,
            ..Default::default()
        };
        
        // 创建索引（使用search模块中的工厂方法）
        let index = crate::vector::search::VectorIndexFactory::create_index(index_config)?;
        
        // 创建集合
        let collection = VectorCollection::new(config.clone(), index);
        
        // 存储集合
        let collection_arc = Arc::new(RwLock::new(collection));
        self.collections.insert(name.to_string(), collection_arc.clone());
        
        // 如果配置了存储路径，则保存配置
        if let Some(path) = &self.storage_path {
            self.save_collection_config(&config, path)?;
        }
        
        Ok(collection_arc)
    }

    /// 创建集合（使用完整配置）
    pub async fn create_collection_with_config(&mut self, config: VectorCollectionConfig) -> Result<()> {
        // 检查集合是否已存在
        if self.collections.contains_key(&config.name) {
            return Err(Error::AlreadyExists(format!(
                "Collection already exists: {}", config.name
            )));
        }
        
        // 创建索引配置
        let index_config = IndexConfig {
            index_type: config.index_type,
            metric: SimilarityMetric::Euclidean, // 可以从config中获取或使用默认值
            dimension: config.dimension,
            ..Default::default()
        };
        
        // 创建索引
        let index = crate::vector::search::VectorIndexFactory::create_index(index_config)?;
        
        // 创建集合
        let collection = VectorCollection::new(config.clone(), index);
        
        // 存储集合
        self.collections.insert(
            config.name.clone(),
            Arc::new(RwLock::new(collection))
        );
        
        // 如果配置了存储路径，则保存配置
        if let Some(path) = &self.storage_path {
            self.save_collection_config(&config, path)?;
        }
        
        Ok(())
    }

    /// 获取集合（返回可直接使用的集合引用）
    pub fn get_collection(&self, name: &str) -> Result<Arc<RwLock<VectorCollection>>> {
        self.collections.get(name)
            .cloned()
            .ok_or_else(|| Error::NotFound(format!("Collection not found: {}", name)))
    }

    pub async fn delete_collection(&mut self, name: &str) -> Result<()> {
        if self.collections.remove(name).is_none() {
            return Err(Error::NotFound(format!("Collection not found: {}", name)));
        }
        
        // 如果配置了存储路径，则删除存储的配置
        if let Some(path) = &self.storage_path {
            self.delete_collection_config(name, path)?;
        }
        
        Ok(())
    }

    pub async fn list_collections(&self) -> Vec<String> {
        self.collections.keys().cloned().collect()
    }

    pub async fn drop_collection(&mut self, name: &str) -> Result<()> {
        if self.collections.remove(name).is_some() {
            // 如果配置了存储路径，则删除持久化数据
            if let Some(path) = &self.storage_path {
                self.remove_collection_storage(name, path)?;
            }
            Ok(())
        } else {
            Err(Error::NotFound(format!("Collection not found: {}", name)))
        }
    }

    // 私有辅助方法
    fn save_collection_config(&self, config: &VectorCollectionConfig, base_path: &str) -> Result<()> {
        // 生产级配置持久化实现
        use std::fs;
        use std::path::Path;
        
        // 1. 验证输入参数
        if config.name.is_empty() {
            return Err(Error::InvalidInput("Collection name cannot be empty".to_string()));
        }
        
        if base_path.is_empty() {
            return Err(Error::InvalidInput("Base path cannot be empty".to_string()));
        }
        
        // 2. 创建目录结构
        let base_dir = Path::new(base_path);
        let config_dir = base_dir.join("configs");
        let backup_dir = base_dir.join("backups");
        
        fs::create_dir_all(&config_dir)
            .map_err(|e| Error::storage(format!("Failed to create config directory: {}", e)))?;
        fs::create_dir_all(&backup_dir)
            .map_err(|e| Error::storage(format!("Failed to create backup directory: {}", e)))?;
        
        // 3. 生成配置文件路径
        let config_filename = format!("{}.json", config.name);
        let config_path = config_dir.join(&config_filename);
        let backup_path = backup_dir.join(format!("{}.backup.json", config.name));
        let temp_path = config_dir.join(format!("{}.tmp", config_filename));
        
        // 4. 备份现有配置（如果存在）
        if config_path.exists() {
            fs::copy(&config_path, &backup_path)
                .map_err(|e| Error::storage(format!("Failed to backup existing config: {}", e)))?;
            log::info!("Backed up existing config to: {:?}", backup_path);
        }
        
        // 5. 序列化配置
        let config_json = serde_json::to_string_pretty(config)
            .map_err(|e| Error::InvalidInput(format!("Failed to serialize config: {}", e)))?;
        
        // 6. 原子写入（先写临时文件，再重命名）
        fs::write(&temp_path, &config_json)
            .map_err(|e| Error::storage(format!("Failed to write temp config file: {}", e)))?;
        
        fs::rename(&temp_path, &config_path)
            .map_err(|e| Error::storage(format!("Failed to rename temp config file: {}", e)))?;
        
        // 7. 验证写入的文件
        let written_content = fs::read_to_string(&config_path)
            .map_err(|e| Error::storage(format!("Failed to read back config file: {}", e)))?;
        
        let _parsed_config: VectorCollectionConfig = serde_json::from_str(&written_content)
            .map_err(|e| Error::storage(format!("Config file validation failed: {}", e)))?;
        
        // 8. 记录操作日志
        log::info!("Successfully saved collection config '{}' to: {:?}", config.name, config_path);
        
        // 9. 清理旧备份文件（保留最近5个）
        self.cleanup_old_backups(&backup_dir, &config.name, 5)?;
        
        Ok(())
    }
    
    /// 清理旧备份文件
    fn cleanup_old_backups(&self, backup_dir: &Path, collection_name: &str, keep_count: usize) -> Result<()> {
        let backup_pattern = format!("{}.backup", collection_name);
        
        let mut backup_files: Vec<_> = std::fs::read_dir(backup_dir)
            .map_err(|e| Error::storage(format!("Failed to read backup directory: {}", e)))?
            .filter_map(|entry| {
                let entry = entry.ok()?;
                let filename = entry.file_name().to_string_lossy().to_string();
                if filename.starts_with(&backup_pattern) {
                    let metadata = entry.metadata().ok()?;
                    let modified = metadata.modified().ok()?;
                    Some((entry.path(), modified))
                } else {
                    None
                }
            })
            .collect();
        
        // 按修改时间排序（最新的在前）
        backup_files.sort_by(|a, b| b.1.cmp(&a.1));
        
        // 删除多余的备份文件
        for (path, _) in backup_files.into_iter().skip(keep_count) {
            if let Err(e) = std::fs::remove_file(&path) {
                log::warn!("Failed to remove old backup file {:?}: {}", path, e);
            } else {
                log::debug!("Removed old backup file: {:?}", path);
            }
        }
        
        Ok(())
    }

    fn delete_collection_config(&self, name: &str, base_path: &str) -> Result<()> {
        let path = Path::new(base_path).join(format!("{}.json", name));
        if path.exists() {
            std::fs::remove_file(path)?;
        }
        Ok(())
    }

    fn remove_collection_storage(&self, name: &str, base_path: &str) -> Result<()> {
        let collection_path = format!("{}/{}.json", base_path, name);
        if std::fs::metadata(&collection_path).is_ok() {
            std::fs::remove_file(&collection_path)
                .map_err(|e| Error::Io(e))?;
        }
        Ok(())
    }
}

/// 基于 VectorDB 的 VectorService 适配器
#[derive(Clone)]
pub struct VectorDBServiceAdapter {
    db: Arc<tokio::sync::RwLock<VectorDB>>,
    default_collection: String,
    default_dimension: usize,
    default_index: self::indexes::IndexType,
}

impl VectorDBServiceAdapter {
    pub fn new(db: VectorDB, default_dimension: usize) -> Self {
        Self {
            db: Arc::new(tokio::sync::RwLock::new(db)),
            default_collection: "default".to_string(),
            default_dimension,
            default_index: self::indexes::IndexType::HNSW,
        }
    }

    async fn ensure_default_collection(&self) -> Result<()> {
        let mut guard = self.db.write().await;
        if !guard.collection_exists(&self.default_collection).await? {
            let cfg = VectorCollectionConfig {
                name: self.default_collection.clone(),
                dimension: self.default_dimension,
                index_type: self::indexes::IndexType::HNSW,
                metadata_schema: None,
            };
            guard.create_collection(&cfg.name, cfg.dimension, cfg.index_type)?;
        }
        Ok(())
    }

    async fn get_default_collection(&self) -> Result<Arc<RwLock<VectorCollection>>> {
        self.ensure_default_collection().await?;
        let guard = self.db.read().await;
        guard.get_collection(&self.default_collection)
    }
}

#[async_trait::async_trait]
impl VectorService for VectorDBServiceAdapter {
    async fn add_vector(&self, id: &str, values: &[f32], metadata: Option<&serde_json::Value>) -> Result<()> {
        let coll = self.get_default_collection().await?;
        let mut coll_guard = coll.write().await;
        coll_guard.add_vector_with_id(id, Vector { id: id.to_string(), data: values.to_vec(), metadata: metadata.map(|m| crate::vector::search::VectorMetadata { properties: serde_json::from_value(m.clone()).unwrap_or_default() }) }).await
    }

    async fn get_vector(&self, id: &str) -> Result<Option<(Vec<f32>, Option<crate::vector::search::VectorMetadata>)>> {
        let coll = self.get_default_collection().await?;
        let coll_guard = coll.read().await;
        Ok(coll_guard.vectors.get(id).map(|v| (v.data.clone(), v.metadata.clone())))
    }

    async fn delete_vector(&self, id: &str) -> Result<bool> {
        let coll = self.get_default_collection().await?;
        let mut coll_guard = coll.write().await;
        let existed = coll_guard.vectors.contains_key(id);
        if existed { coll_guard.delete_vector(id)?; }
        Ok(existed)
    }

    async fn search_vectors(
        &self,
        query_vector: &[f32],
        top_k: usize,
        _metric: &str,
        _filter: Option<&serde_json::Value>,
    ) -> Result<Vec<(String, f32, Vec<f32>, Option<crate::vector::search::VectorMetadata>)>> {
        let coll = self.get_default_collection().await?;
        let coll_guard = coll.read().await;
        let query = crate::vector::search::VectorQuery {
            vector: query_vector.to_vec(),
            filter: None,
            top_k,
            include_metadata: true,
            include_vectors: true,
        };
        let results = coll_guard.search(&query)?;
        Ok(results
            .into_iter()
            .map(|r| (r.id, r.score, r.vector.unwrap_or_default(), r.metadata))
            .collect())
    }

    async fn count_vectors(&self) -> Result<usize> {
        let coll = self.get_default_collection().await?;
        let coll_guard = coll.read().await;
        Ok(coll_guard.vectors.len())
    }

    async fn rebuild_index(&self) -> Result<()> {
        let coll = self.get_default_collection().await?;
        let mut coll_guard = coll.write().await;
        coll_guard.rebuild_index().await
    }

    async fn update_vector(&self, id: &str, values: &[f32], metadata: Option<&serde_json::Value>) -> Result<()> {
        let coll = self.get_default_collection().await?;
        let mut coll_guard = coll.write().await;
        // 如果存在则删除再添加，确保索引同步更新
        if coll_guard.vectors.contains_key(id) {
            coll_guard.delete_vector(id)?;
        }
        let vector = Vector { id: id.to_string(), data: values.to_vec(), metadata: metadata.map(|m| crate::vector::search::VectorMetadata { properties: serde_json::from_value(m.clone()).unwrap_or_default() }) };
        coll_guard.add_vector_with_id(id, vector).await
    }

    async fn get_stats(&self) -> Result<VectorStats> {
        let coll = self.get_default_collection().await?;
        let coll_guard = coll.read().await;
        let count = coll_guard.vectors.len();
        let dimension = coll_guard
            .vectors
            .values()
            .next()
            .map(|v| v.data.len())
            .unwrap_or(0);
        let index_type = format!("{:?}", self.default_index);
        Ok(VectorStats { count, dimension, index_type, memory_usage: Some(0) })
    }
}

/// 通用闭包适配器：用于将任意引擎（如可能存在的 VectorEngine）包装为 VectorService
pub struct FnVectorServiceAdapter {
    add_fn: std::sync::Arc<dyn Fn(String, Vec<f32>, Option<serde_json::Value>) -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<()>> + Send>> + Send + Sync>,
    get_fn: std::sync::Arc<dyn Fn(String) -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<Option<(Vec<f32>, Option<crate::vector::search::VectorMetadata>)>>> + Send>> + Send + Sync>,
    del_fn: std::sync::Arc<dyn Fn(String) -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<bool>> + Send>> + Send + Sync>,
    search_fn: std::sync::Arc<dyn Fn(Vec<f32>, usize, String, Option<serde_json::Value>) -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<Vec<(String, f32, Vec<f32>, Option<crate::vector::search::VectorMetadata>)>>> + Send>> + Send + Sync>,
    count_fn: std::sync::Arc<dyn Fn() -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<usize>> + Send>> + Send + Sync>,
    rebuild_fn: std::sync::Arc<dyn Fn() -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<()>> + Send>> + Send + Sync>,
}

impl FnVectorServiceAdapter {
    pub fn new(
        add_fn: impl Fn(String, Vec<f32>, Option<serde_json::Value>) -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<()>> + Send>> + Send + Sync + 'static,
        get_fn: impl Fn(String) -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<Option<(Vec<f32>, Option<crate::vector::search::VectorMetadata>)>>> + Send>> + Send + Sync + 'static,
        del_fn: impl Fn(String) -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<bool>> + Send>> + Send + Sync + 'static,
        search_fn: impl Fn(Vec<f32>, usize, String, Option<serde_json::Value>) -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<Vec<(String, f32, Vec<f32>, Option<crate::vector::search::VectorMetadata>)>>> + Send>> + Send + Sync + 'static,
        count_fn: impl Fn() -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<usize>> + Send>> + Send + Sync + 'static,
        rebuild_fn: impl Fn() -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<()>> + Send>> + Send + Sync + 'static,
    ) -> Self {
        Self {
            add_fn: std::sync::Arc::new(add_fn),
            get_fn: std::sync::Arc::new(get_fn),
            del_fn: std::sync::Arc::new(del_fn),
            search_fn: std::sync::Arc::new(search_fn),
            count_fn: std::sync::Arc::new(count_fn),
            rebuild_fn: std::sync::Arc::new(rebuild_fn),
        }
    }
}

#[async_trait::async_trait]
impl VectorService for FnVectorServiceAdapter {
    async fn add_vector(&self, id: &str, values: &[f32], metadata: Option<&serde_json::Value>) -> Result<()> {
        (self.add_fn)(id.to_string(), values.to_vec(), metadata.cloned()).await
    }

    async fn get_vector(&self, id: &str) -> Result<Option<(Vec<f32>, Option<crate::vector::search::VectorMetadata>)>> {
        (self.get_fn)(id.to_string()).await
    }

    async fn delete_vector(&self, id: &str) -> Result<bool> {
        (self.del_fn)(id.to_string()).await
    }

    async fn search_vectors(&self, query_vector: &[f32], top_k: usize, metric: &str, filter: Option<&serde_json::Value>) -> Result<Vec<(String, f32, Vec<f32>, Option<crate::vector::search::VectorMetadata>)>> {
        (self.search_fn)(query_vector.to_vec(), top_k, metric.to_string(), filter.cloned()).await
    }

    async fn count_vectors(&self) -> Result<usize> {
        (self.count_fn)().await
    }

    async fn rebuild_index(&self) -> Result<()> {
        (self.rebuild_fn)().await
    }

    async fn update_vector(&self, id: &str, values: &[f32], metadata: Option<&serde_json::Value>) -> Result<()> {
        // 通过先删后加的方式实现更新，保证与底层索引一致
        (self.del_fn)(id.to_string()).await?;
        (self.add_fn)(id.to_string(), values.to_vec(), metadata.cloned()).await
    }

    async fn get_stats(&self) -> Result<VectorStats> {
        // 由于闭包未提供详细统计接口，这里提供最小可用实现
        let count = (self.count_fn)().await?;
        Ok(VectorStats { count, dimension: 0, index_type: "unknown".to_string(), memory_usage: None })
    }
}

impl Component for VectorDB {
    fn name(&self) -> &str {
        "VectorDB"
    }
    
    fn component_type(&self) -> ComponentType {
        ComponentType::Vector
    }
    
    fn status(&self) -> ComponentStatus {
        // 根据内部状态返回适当的状态
        if self.collections.is_empty() {
            ComponentStatus::Initializing
        } else {
            ComponentStatus::Ready
        }
    }
    
    fn start(&mut self) -> Result<()> {
        // VectorDB 启动逻辑
        // 在实际实现中，这里可以初始化索引、加载数据等
        Ok(())
    }
    
    fn stop(&mut self) -> Result<()> {
        // VectorDB 停止逻辑
        // 在实际实现中，这里可以保存数据、清理资源等
        Ok(())
    }
}

// 统一从index模块导出IndexConfig，避免重复定义
pub use self::index::types::IndexConfig;

#[cfg(test)]
mod tests {
    use super::*;
    use super::indexes::IndexType;

    #[tokio::test]
    async fn test_vector_db() {
        let mut db = VectorDB::new();
        
        // 创建集合
        let config = VectorCollectionConfig {
            name: "test".to_string(),
            dimension: 128,
            index_type: IndexType::Flat,
            metadata_schema: None,
        };
        
        db.create_collection(config.clone()).await.unwrap();
        
        // 获取集合
        let collection = db.get_collection("test").await.unwrap();
        
        // 添加向量
        let vector = Vector {
            id: Uuid::new_v4().to_string(),
            data: vec![1.0; 128],
            metadata: None,
        };
        
        {
            let mut collection = collection.write().await;
            collection.add_vector(vector.clone()).unwrap();
        }
        
        // 查询向量
        let query = VectorQuery {
            vector: vec![1.0; 128],
            filter: None,
            top_k: 10,
            include_metadata: true,
            include_vectors: true,
        };
        
        let results = {
            let collection = collection.read().await;
            collection.search(&query).unwrap()
        };
        
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].id, vector.id);
        
        // 删除集合
        db.delete_collection("test").await.unwrap();
        
        // 确认已删除
        let result = db.get_collection("test").await;
        assert!(result.is_err());
    }
} 