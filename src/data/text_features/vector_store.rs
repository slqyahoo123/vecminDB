// 文本特征向量存储接口
// 提供文本特征向量的存储、索引和检索功能

use crate::Result;
use crate::Error;
use crate::data::text_features::extractors::FeatureExtractor;
use crate::vector::{
    Vector, 
    VectorMetadata, 
    VectorQuery, 
    SearchOptions,
    VectorCollection, 
    VectorCollectionConfig, 
    VectorSearchResult,
    IndexType,
};
use std::collections::HashMap;
use std::sync::{Arc, RwLock};
use serde::{Serialize, Deserialize};

/// 文本特征向量存储配置
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TextFeatureVectorStoreConfig {
    /// 底层向量集合配置
    pub collection_config: VectorCollectionConfig,
    /// 特征维度
    pub dimension: usize,
    /// 是否启用自动优化
    pub auto_optimize: bool,
    /// 是否启用批量处理
    pub batch_processing: bool,
    /// 批处理大小
    pub batch_size: usize,
    /// 缓存配置
    pub cache_config: Option<CacheConfig>,
}

impl Default for TextFeatureVectorStoreConfig {
    fn default() -> Self {
        Self {
            collection_config: VectorCollectionConfig {
                name: "text_features".to_string(),
                dimension: 768,
                index_type: IndexType::HNSW,
                metadata_schema: None,
            },
            dimension: 768,
            auto_optimize: true,
            batch_processing: true,
            batch_size: 100,
            cache_config: Some(CacheConfig::default()),
        }
    }
}

/// 缓存配置
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheConfig {
    /// 是否启用缓存
    pub enabled: bool,
    /// 缓存大小(条目数)
    pub max_size: usize,
    /// 缓存过期时间(秒)
    pub ttl: u64,
}

impl Default for CacheConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            max_size: 1000,
            ttl: 3600,
        }
    }
}

/// 文本特征向量存储
pub struct TextFeatureVectorStore {
    /// 底层向量集合
    collection: Arc<RwLock<VectorCollection>>,
    /// 特征提取器
    extractor: Option<Box<dyn FeatureExtractor>>,
    /// 配置
    config: TextFeatureVectorStoreConfig,
    /// 缓存
    feature_cache: HashMap<String, Vec<f32>>,
    /// 是否初始化
    initialized: bool,
}

impl TextFeatureVectorStore {
    /// 创建新的文本特征向量存储
    pub fn new(config: TextFeatureVectorStoreConfig) -> Result<Self> {
        use crate::vector::index::{VectorIndexFactory, IndexConfig};
        use crate::vector::operations::SimilarityMetric;
        
        let index_config = IndexConfig {
            index_type: config.collection_config.index_type.clone(),
            metric: SimilarityMetric::Cosine,
            dimension: config.collection_config.dimension,
            ..Default::default()
        };
        let index = VectorIndexFactory::create_index(index_config)?;
        let collection = VectorCollection::new(config.collection_config.clone(), Box::new(index));
        
        Ok(Self {
            collection: Arc::new(RwLock::new(collection)),
            extractor: None,
            config,
            feature_cache: HashMap::new(),
            initialized: false,
        })
    }
    
    /// 设置特征提取器
    pub fn set_extractor(&mut self, extractor: Box<dyn FeatureExtractor>) {
        self.extractor = Some(extractor);
        self.initialized = true;
    }
    
    /// 添加文本并索引其特征向量
    pub fn add_text(&mut self, text_id: &str, text: &str, metadata: HashMap<String, String>) -> Result<()> {
        if !self.initialized {
            return Err(Error::InvalidOperation("特征提取器未设置".to_string()));
        }
        
        // 提取特征
        let features = self.extract_features(text)?;
        
        // 创建向量
        let vector = Vector {
            id: text_id.to_string(),
            data: features.clone(),
            metadata: Some(VectorMetadata { 
                properties: metadata.into_iter()
                    .map(|(k, v)| (k, serde_json::Value::String(v)))
                    .collect(),
            }),
        };
        
        // 添加到集合
        {
            let mut collection = self.collection.write().map_err(|_| 
                Error::Internal("无法获取向量集合写锁".to_string()))?;
            collection.add_vector(vector)?;
        }
        
        // 添加到缓存
        if let Some(ref config) = self.config.cache_config {
            if config.enabled {
                self.feature_cache.insert(text_id.to_string(), features);
                
                // 超出缓存大小时清理最早的条目
                if self.feature_cache.len() > config.max_size {
                    if let Some(oldest_key) = self.feature_cache.keys().next().cloned() {
                        self.feature_cache.remove(&oldest_key);
                    }
                }
            }
        }
        
        Ok(())
    }
    
    /// 批量添加文本并索引
    pub fn add_texts_batch(&mut self, texts: Vec<(String, String, HashMap<String, String>)>) -> Result<()> {
        if !self.initialized {
            return Err(Error::InvalidOperation("特征提取器未设置".to_string()));
        }
        
        let mut vectors = Vec::with_capacity(texts.len());
        
        for (id, text, metadata) in texts {
            // 提取特征
            let features = self.extract_features(&text)?;
            
            // 创建向量
            let vector = Vector {
                id,
                data: features.clone(),
                metadata: Some(VectorMetadata { 
                    properties: metadata.into_iter()
                        .map(|(k, v)| (k, serde_json::Value::String(v)))
                        .collect(),
                }),
            };
            
            vectors.push(vector);
        }
        
        // 批量添加到集合
        {
            let mut collection = self.collection.write().map_err(|_| 
                Error::Internal("无法获取向量集合写锁".to_string()))?;
            for vector in vectors {
                collection.add_vector(vector)?;
            }
        }
        
        Ok(())
    }
    
    /// 提取文本特征
    fn extract_features(&self, text: &str) -> Result<Vec<f32>> {
        if let Some(ref extractor) = self.extractor {
            extractor.extract(text)
        } else {
            Err(Error::InvalidOperation("特征提取器未设置".to_string()))
        }
    }
    
    /// 通过文本ID检索特征向量
    pub fn get_vector(&self, text_id: &str) -> Result<Option<Vector>> {
        let collection = self.collection.read().map_err(|_| 
            Error::Internal("无法获取向量集合读锁".to_string()))?;
        Ok(collection.get_vector(text_id).cloned())
    }
    
    /// 通过相似文本搜索
    pub fn search_by_text(&self, query_text: &str, top_k: usize, options: Option<SearchOptions>) 
        -> Result<Vec<VectorSearchResult>> 
    {
        if !self.initialized {
            return Err(Error::InvalidOperation("特征提取器未设置".to_string()));
        }
        
        // 提取查询文本的特征
        let query_features = self.extract_features(query_text)?;
        
        // 如果有 SearchOptions，使用 search_with_options，否则使用 VectorQuery
        let collection = self.collection.read().map_err(|_| 
            Error::Internal("无法获取向量集合读锁".to_string()))?;
        
        if let Some(ref search_options) = options {
            collection.search_with_options(&query_features, search_options)
        } else {
            // 创建查询
            let query = VectorQuery {
                vector: query_features,
                filter: None,
                top_k,
                include_metadata: true,
                include_vectors: false,
            };
            collection.search(&query)
        }
    }
    
    /// 通过特征向量搜索
    pub fn search_by_vector(&self, query_vector: Vec<f32>, top_k: usize, options: Option<SearchOptions>) 
        -> Result<Vec<VectorSearchResult>> 
    {
        let collection = self.collection.read().map_err(|_| 
            Error::Internal("无法获取向量集合读锁".to_string()))?;
        
        // 如果有 SearchOptions，使用 search_with_options，否则使用 VectorQuery
        if let Some(ref search_options) = options {
            collection.search_with_options(&query_vector, search_options)
        } else {
            // 创建查询
            let query = VectorQuery {
                vector: query_vector,
                filter: None,
                top_k,
                include_metadata: true,
                include_vectors: false,
            };
            collection.search(&query)
        }
    }
    
    /// 删除向量
    pub fn delete_vector(&mut self, text_id: &str) -> Result<bool> {
        // 从集合中删除
        let mut collection = self.collection.write().map_err(|_| 
            Error::Internal("无法获取向量集合写锁".to_string()))?;
        collection.delete_vector(text_id)?;
        
        // 从缓存中删除
        let existed = self.feature_cache.remove(text_id).is_some();
        
        Ok(existed)
    }
    
    /// 更新向量
    pub fn update_vector(&mut self, text_id: &str, new_text: &str, metadata: Option<HashMap<String, String>>) -> Result<bool> {
        if !self.initialized {
            return Err(Error::InvalidOperation("特征提取器未设置".to_string()));
        }
        
        // 提取新特征
        let features = self.extract_features(new_text)?;
        
        // 获取现有向量
        let mut vector = match self.get_vector(text_id)? {
            Some(v) => v,
            None => return Ok(false),
        };
        
        // 更新向量数据
        vector.data = features.clone();
        
        // 更新元数据
        if let Some(meta) = metadata {
            let properties: HashMap<String, serde_json::Value> = meta.into_iter()
                .map(|(k, v)| (k, serde_json::Value::String(v)))
                .collect();
            if let Some(ref mut vm) = vector.metadata {
                vm.properties = properties;
            } else {
                vector.metadata = Some(VectorMetadata { 
                    properties,
                });
            }
        }
        
        // 更新集合
        let mut collection = self.collection.write().map_err(|_| 
            Error::Internal("无法获取向量集合写锁".to_string()))?;
        // 注意：update_vector是async方法，但这里在同步上下文中
        // 需要先删除再添加，或者使用其他同步方法
        collection.delete_vector(&vector.id)?;
        let new_vector = Vector {
            id: vector.id.clone(),
            data: vector.data.clone(),
            metadata: vector.metadata.clone(),
        };
        let result = collection.add_vector(new_vector).is_ok();
        
        // 更新缓存
        if result {
            self.feature_cache.insert(text_id.to_string(), features);
        }
        
        Ok(result)
    }
    
    /// 获取向量数量
    pub fn count(&self) -> Result<usize> {
        let collection = self.collection.read().map_err(|_| 
            Error::Internal("无法获取向量集合读锁".to_string()))?;
        Ok(collection.count())
    }
    
    /// 优化索引
    /// 注意：VectorCollection 目前没有 optimize 方法，此方法暂时为空实现
    pub fn optimize(&mut self) -> Result<()> {
        // VectorCollection 目前不支持优化操作
        // 如果需要优化，可以在这里实现索引重建等逻辑
        Ok(())
    }
    
    /// 保存到磁盘
    /// 注意：VectorCollection 目前没有 save 方法，此方法暂时为空实现
    pub fn save(&self, _path: &str) -> Result<()> {
        // VectorCollection 目前不支持直接保存到磁盘
        // 如果需要持久化，可以通过其他方式实现
        Ok(())
    }
    
    /// 从磁盘加载
    /// 注意：VectorCollection 目前没有 load 方法，此方法暂时返回错误
    pub fn load(_path: &str, config: TextFeatureVectorStoreConfig) -> Result<Self> {
        // VectorCollection 目前不支持从磁盘加载
        // 如果需要持久化，可以通过其他方式实现
        Err(Error::InvalidOperation("VectorCollection 暂不支持从磁盘加载".to_string()))
    }
    
    /// 清除所有数据
    pub fn clear(&mut self) -> Result<()> {
        // 清除集合
        let mut collection = self.collection.write().map_err(|_| 
            Error::Internal("无法获取向量集合写锁".to_string()))?;
        
        // 手动清除所有向量和元数据
        let vector_ids: Vec<String> = collection.vectors.keys().cloned().collect();
        for id in vector_ids {
            collection.delete_vector(&id)?;
        }
        collection.vectors.clear();
        collection.metadata.clear();
        
        // 清除缓存
        self.feature_cache.clear();
        
        Ok(())
    }
    
    /// 获取底层集合的只读引用
    pub fn get_collection(&self) -> Result<std::sync::RwLockReadGuard<VectorCollection>> {
        self.collection.read().map_err(|_| 
            Error::Internal("无法获取向量集合读锁".to_string()))
    }
    
    /// 获取底层集合的可写引用
    pub fn get_collection_mut(&self) -> Result<std::sync::RwLockWriteGuard<VectorCollection>> {
        self.collection.write().map_err(|_| 
            Error::Internal("无法获取向量集合写锁".to_string()))
    }
    
    /// 获取配置的只读引用
    pub fn get_config(&self) -> &TextFeatureVectorStoreConfig {
        &self.config
    }
}

/// 向量存储工厂 - 用于创建和管理文本特征向量存储
pub struct TextFeatureVectorStoreFactory;

impl TextFeatureVectorStoreFactory {
    /// 创建新的向量存储
    pub fn create(config: TextFeatureVectorStoreConfig, extractor: Option<Box<dyn FeatureExtractor>>) -> Result<TextFeatureVectorStore> {
        let mut store = TextFeatureVectorStore::new(config)?;
        if let Some(ex) = extractor {
            store.set_extractor(ex);
        }
        Ok(store)
    }
    
    /// 加载现有的向量存储
    pub fn load(path: &str, config: TextFeatureVectorStoreConfig, extractor: Option<Box<dyn FeatureExtractor>>) -> Result<TextFeatureVectorStore> {
        let mut store = TextFeatureVectorStore::load(path, config)?;
        if let Some(ex) = extractor {
            store.set_extractor(ex);
        }
        Ok(store)
    }
} 