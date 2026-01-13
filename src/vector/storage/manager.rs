use std::sync::{Arc, RwLock};
use crate::{Result, Error, Storage};
use crate::vector::types::Vector;
use crate::vector::search::{VectorQuery, VectorSearchResult, VectorMetadata};
use serde::{Serialize, Deserialize};
use crate::vector::generate_uuid;
use std::collections::HashMap;
use crate::vector::index::{VectorIndexEnum, IndexConfig};

/// 向量元数据
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VectorMetadata {
    /// 向量ID
    pub id: String,
    /// 集合名称
    pub collection: String,
    /// 创建时间
    pub created_at: u64,
    /// 额外属性
    pub attributes: HashMap<String, String>,
}

/// 集合元数据
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CollectionMetadata {
    /// 集合名称
    pub name: String,
    /// 向量维度
    pub dimension: usize,
    /// 向量数量
    pub count: usize,
    /// 创建时间
    pub created_at: u64,
    /// 更新时间
    pub updated_at: u64,
    /// 描述
    pub description: Option<String>,
    /// 额外属性
    pub attributes: HashMap<String, String>,
}

/// 向量批量请求
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VectorBatchRequest {
    /// 向量列表
    pub vectors: Vec<Vector>,
    /// 集合名称
    pub collection: String,
}

/// 向量搜索响应
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VectorSearchResponse {
    /// 结果列表
    pub results: Vec<VectorSearchResult>,
    /// 查询时间(毫秒)
    pub time_ms: f32,
    /// 集合名称
    pub collection: String,
}

/// 向量存储管理器
pub struct VectorStorageManager {
    /// 存储引擎
    storage: Arc<Storage>,
    /// 元数据缓存
    collection_cache: Arc<RwLock<HashMap<String, CollectionMetadata>>>,
}

impl VectorStorageManager {
    /// 创建新的向量存储管理器
    pub fn new(storage: Arc<Storage>) -> Self {
        Self {
            storage,
            collection_cache: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// 就绪检查：验证集合元数据路径、缓存与存储接线
    pub async fn readiness_probe(&self) -> Result<()> {
        let probe_collection = "__vector_probe__";
        // 如果集合不存在则创建
        if !self.collection_exists(probe_collection).await? {
            let _ = self.create_collection(probe_collection, 4, Some("probe".to_string())).await?;
        }
        // 读取一次元数据，确保缓存更新
        let _meta = self.get_collection_metadata(probe_collection).await?;
        Ok(())
    }

    /// 创建集合
    pub async fn create_collection(
        &self,
        name: &str,
        dimension: usize,
        description: Option<String>,
    ) -> Result<CollectionMetadata> {
        // 检查集合是否已存在
        let key = format!("collection:{}:metadata", name);
        if self.storage.exists(&key).await? {
            return Err(Error::invalid_operation(format!("Collection '{}' already exists", name)));
        }

        // 创建集合元数据
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        let metadata = CollectionMetadata {
            name: name.to_string(),
            dimension,
            count: 0,
            created_at: now,
            updated_at: now,
            description,
            attributes: HashMap::new(),
        };

        // 保存元数据
        let metadata_json = serde_json::to_string(&metadata)
            .map_err(|e| Error::data(format!("Failed to serialize collection metadata: {}", e)))?;
        self.storage.put(&key, metadata_json.as_bytes()).await?;

        // 更新缓存
        {
            let mut cache = self.collection_cache.write()
                .map_err(|_| Error::lock("Failed to acquire write lock for collection cache"))?;
            cache.insert(name.to_string(), metadata.clone());
        }

        Ok(metadata)
    }

    /// 检查集合是否存在
    pub async fn collection_exists(&self, name: &str) -> Result<bool> {
        // 先检查缓存
        {
            let cache = self.collection_cache.read()
                .map_err(|_| Error::lock("Failed to acquire read lock for collection cache"))?;
            if cache.contains_key(name) {
                return Ok(true);
            }
        }

        // 检查存储
        let key = format!("collection:{}:metadata", name);
        self.storage.exists(&key).await
    }

    /// 获取集合元数据
    pub async fn get_collection_metadata(&self, name: &str) -> Result<CollectionMetadata> {
        // 先检查缓存
        {
            let cache = self.collection_cache.read()
                .map_err(|_| Error::lock("Failed to acquire read lock for collection cache"))?;
            if let Some(metadata) = cache.get(name) {
                return Ok(metadata.clone());
            }
        }

        // 从存储加载
        let key = format!("collection:{}:metadata", name);
        let data = self.storage.get(&key).await?
            .ok_or_else(|| Error::not_found(format!("Collection '{}' not found", name)))?;

        let metadata: CollectionMetadata = serde_json::from_slice(&data)
            .map_err(|e| Error::data(format!("Failed to deserialize collection metadata: {}", e)))?;

        // 更新缓存
        {
            let mut cache = self.collection_cache.write()
                .map_err(|_| Error::lock("Failed to acquire write lock for collection cache"))?;
            cache.insert(name.to_string(), metadata.clone());
        }

        Ok(metadata)
    }

    /// 存储向量
    pub async fn store_vector(&self, vector: Vector, collection: &str) -> Result<String> {
        // 检查集合是否存在
        let metadata = self.get_collection_metadata(collection).await?;

        // 检查向量维度
        if vector.values.len() != metadata.dimension {
            return Err(Error::invalid_input(format!(
                "Vector dimension mismatch: expected {}, got {}",
                metadata.dimension,
                vector.values.len()
            )));
        }

        // 为向量生成ID（如果没有）
        let vector_id = vector.id.clone().unwrap_or_else(|| generate_uuid());
        let vector_with_id = Vector {
            id: Some(vector_id.clone()),
            ..vector
        };

        // 向量键
        let key = format!("vector:{}:{}", collection, vector_id);

        // 序列化向量
        let vector_json = serde_json::to_string(&vector_with_id)
            .map_err(|e| Error::data(format!("Failed to serialize vector: {}", e)))?;

        // 存储向量
        self.storage.put(&key, vector_json.as_bytes()).await?;

        // 更新集合计数
        self.update_collection_count(collection, true).await?;

        Ok(vector_id)
    }

    /// 批量存储向量
    pub async fn store_vectors(&self, batch: VectorBatchRequest) -> Result<Vec<String>> {
        let collection = batch.collection;
        let mut ids = Vec::with_capacity(batch.vectors.len());

        // 检查集合是否存在
        let metadata = self.get_collection_metadata(&collection).await?;

        // 批量操作
        let mut operations = Vec::with_capacity(batch.vectors.len());
        
        for vector in batch.vectors {
            // 检查向量维度
            if vector.values.len() != metadata.dimension {
                return Err(Error::invalid_input(format!(
                    "Vector dimension mismatch: expected {}, got {}",
                    metadata.dimension,
                    vector.values.len()
                )));
            }

            // 为向量生成ID（如果没有）
            let vector_id = vector.id.clone().unwrap_or_else(|| generate_uuid());
            let vector_with_id = Vector {
                id: Some(vector_id.clone()),
                ..vector
            };

            // 向量键
            let key = format!("vector:{}:{}", collection, vector_id);

            // 序列化向量
            let vector_json = serde_json::to_string(&vector_with_id)
                .map_err(|e| Error::data(format!("Failed to serialize vector: {}", e)))?;

            // 添加到操作列表
            operations.push((key, vector_json.into_bytes()));
            ids.push(vector_id);
        }

        // 执行批量存储
        if !operations.is_empty() {
            self.storage.batch_put(&operations).await?;
            self.update_collection_count(&collection, operations.len()).await?;
        }

        Ok(ids)
    }

    /// 获取向量
    pub async fn get_vector(&self, id: &str, collection: &str) -> Result<Vector> {
        let key = format!("vector:{}:{}", collection, id);
        let data = self.storage.get(&key).await?
            .ok_or_else(|| Error::not_found(format!("Vector '{}' not found in collection '{}'", id, collection)))?;

        let vector: Vector = serde_json::from_slice(&data)
            .map_err(|e| Error::data(format!("Failed to deserialize vector: {}", e)))?;

        Ok(vector)
    }

    /// 删除向量
    pub async fn delete_vector(&self, id: &str, collection: &str) -> Result<()> {
        let key = format!("vector:{}:{}", collection, id);
        
        // 检查向量是否存在
        if !self.storage.exists(&key).await? {
            return Err(Error::not_found(format!("Vector '{}' not found in collection '{}'", id, collection)));
        }

        // 删除向量
        self.storage.delete(&key).await?;

        // 更新集合计数
        self.update_collection_count(collection, false).await?;

        Ok(())
    }

    /// 获取集合中的向量数量
    pub async fn count(&self, collection: &str) -> Result<usize> {
        let metadata = self.get_collection_metadata(collection).await?;
        Ok(metadata.count)
    }

    /// 搜索向量
    pub async fn search(
        &self,
        query: VectorQuery,
        collection: &str,
    ) -> Result<VectorSearchResponse> {
        // 验证集合是否存在
        let metadata = self.get_collection_metadata(collection).await?;
        
        // 验证查询向量维度
        if query.vector.values.len() != metadata.dimension {
            return Err(Error::invalid_input(format!(
                "Query vector dimension mismatch: expected {}, got {}",
                metadata.dimension,
                query.vector.values.len()
            )));
        }

        // 记录开始时间
        let start_time = std::time::Instant::now();

        // 尝试使用索引进行高效搜索
        let results = if let Ok(index_results) = self.search_with_index(&query, collection).await {
            index_results
        } else {
            // 如果索引不可用，回退到全量搜索
            self.search_brute_force(&query, collection).await?
        };

        // 计算查询时间
        let elapsed = start_time.elapsed();
        let time_ms = elapsed.as_secs_f32() * 1000.0;

        Ok(VectorSearchResponse {
            results,
            time_ms,
            collection: collection.to_string(),
        })
    }

    /// 使用索引进行高效搜索
    async fn search_with_index(
        &self,
        query: &VectorQuery,
        collection: &str,
    ) -> Result<Vec<VectorSearchResult>> {
        // 尝试获取集合的索引
        let index_key = format!("index:{}", collection);
        let index_data = self.storage.get(&index_key).await?
            .ok_or_else(|| Error::not_found("Index not found for collection"))?;

        // 反序列化索引配置
        let index_config: IndexConfig = serde_json::from_slice(&index_data)
            .map_err(|e| Error::data(format!("Failed to deserialize index config: {}", e)))?;

        // 创建相应的索引实例
        let index = self.create_index_from_config(&index_config, collection).await?;

        // 使用索引进行搜索
        let search_results = {
            let index_guard = index.read()
                .map_err(|_| Error::lock("Failed to acquire read lock for index"))?;
            index_guard.search(&query.vector.values, query.limit.unwrap_or(10))?
        };

        // 转换搜索结果并获取完整向量数据
        let mut results = Vec::new();
        for search_result in search_results {
            // 获取完整的向量数据
            if let Ok(vector) = self.get_vector(&search_result.id, collection).await {
                // 应用过滤器
                if let Some(filter) = &query.filter {
                    if !self.apply_filter(filter, &vector) {
                        continue;
                    }
                }

                results.push(VectorSearchResult {
                    id: search_result.id,
                    vector,
                    score: search_result.score,
                });
            }
        }

        Ok(results)
    }

    /// 暴力搜索（备用方案）
    async fn search_brute_force(
        &self,
        query: &VectorQuery,
        collection: &str,
    ) -> Result<Vec<VectorSearchResult>> {
        let prefix = format!("vector:{}:", collection);
        let keys = self.storage.get_keys_with_prefix(&prefix).await?;

        let mut results = Vec::new();
        let mut processed_count = 0;
        let max_processed = 10000; // 限制处理的向量数量以避免超时

        for key in keys {
            if processed_count >= max_processed {
                break;
            }

            let data = match self.storage.get(&key).await? {
                Some(data) => data,
                None => continue,
            };

            let vector: Vector = match serde_json::from_slice(&data) {
                Ok(v) => v,
                Err(_) => continue,
            };

            // 应用预过滤以提高效率
            if let Some(filter) = &query.filter {
                if !self.apply_filter(filter, &vector) {
                    continue;
                }
            }

            // 计算相似度
            let similarity = match query.vector.metric.as_deref().unwrap_or("cosine") {
                "cosine" => calculate_cosine_similarity(&query.vector.values, &vector.values),
                "euclidean" => calculate_euclidean_distance(&query.vector.values, &vector.values),
                "dot_product" => calculate_dot_product(&query.vector.values, &vector.values),
                _ => calculate_cosine_similarity(&query.vector.values, &vector.values),
            };

            results.push(VectorSearchResult {
                id: vector.id.clone().unwrap_or_default(),
                vector,
                score: similarity,
            });

            processed_count += 1;
        }

        // 按相似度排序
        results.sort_by(|a, b| {
            b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal)
        });

        // 限制结果数量
        if let Some(limit) = query.limit {
            if limit > 0 && limit < results.len() {
                results.truncate(limit);
            }
        }

        Ok(results)
    }

    /// 从配置创建索引实例
    async fn create_index_from_config(
        &self,
        config: &IndexConfig,
        collection: &str,
    ) -> Result<Arc<RwLock<VectorIndexEnum>>> {
        let index = crate::vector::index::factory::VectorIndexFactory::create_index(config.clone())?;
        
        // 加载索引数据
        let index_data_key = format!("index_data:{}", collection);
        if let Some(serialized_data) = self.storage.get(&index_data_key).await? {
            // 反序列化索引数据
            let mut index_guard = index.write()
                .map_err(|_| Error::lock("Failed to acquire write lock for index"))?;
            
            // 这里应该调用索引的反序列化方法
            // 由于不同索引类型的反序列化方法可能不同，这里简化处理
            drop(index_guard);
        }

        Ok(index)
    }

    /// 更新集合计数
    async fn update_collection_count(&self, collection: &str, update: i32) -> Result<()> {
        // 获取当前元数据
        let mut metadata = self.get_collection_metadata(collection).await?;
        
        // 更新计数和时间戳
        if update > 0 {
            metadata.count += update as usize;
        } else if update < 0 && metadata.count > (-update) as usize {
            metadata.count -= (-update) as usize;
        } else if update < 0 {
            metadata.count = 0;
        }
        
        metadata.updated_at = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        
        // 保存更新后的元数据
        let key = format!("collection:{}:metadata", collection);
        let metadata_json = serde_json::to_string(&metadata)
            .map_err(|e| Error::data(format!("Failed to serialize collection metadata: {}", e)))?;
        self.storage.put(&key, metadata_json.as_bytes()).await?;
        
        // 更新缓存
        {
            let mut cache = self.collection_cache.write()
                .map_err(|_| Error::lock("Failed to acquire write lock for collection cache"))?;
            cache.insert(collection.to_string(), metadata);
        }
        
        Ok(())
    }

    /// 应用过滤器
    fn apply_filter(&self, filter: &HashMap<String, String>, vector: &Vector) -> bool {
        if filter.is_empty() {
            return true;
        }

        // 检查所有过滤条件
        for (key, value) in filter {
            if let Some(metadata) = &vector.metadata {
                if let Some(attr_value) = metadata.get(key) {
                    if attr_value != value {
                        return false;
                    }
                } else {
                    return false; // 缺少属性
                }
            } else {
                return false; // 没有元数据
            }
        }

        true
    }
}

/// 计算余弦相似度
pub fn calculate_cosine_similarity(vec1: &[f32], vec2: &[f32]) -> f32 {
    if vec1.len() != vec2.len() || vec1.is_empty() {
        return 0.0;
    }
    
    let mut dot_product = 0.0;
    let mut norm1 = 0.0;
    let mut norm2 = 0.0;
    
    for i in 0..vec1.len() {
        dot_product += vec1[i] as f64 * vec2[i] as f64;
        norm1 += vec1[i] as f64 * vec1[i] as f64;
        norm2 += vec2[i] as f64 * vec2[i] as f64;
    }
    
    let norm = (norm1 * norm2).sqrt();
    if norm < 1e-10 {
        return 0.0;
    }
    
    (dot_product / norm) as f32
}

/// 计算欧几里得距离（转换为相似度分数）
pub fn calculate_euclidean_distance(vec1: &[f32], vec2: &[f32]) -> f32 {
    if vec1.len() != vec2.len() || vec1.is_empty() {
        return 0.0;
    }
    
    let mut sum_sq_diff = 0.0;
    for i in 0..vec1.len() {
        let diff = vec1[i] as f64 - vec2[i] as f64;
        sum_sq_diff += diff * diff;
    }
    
    let distance = sum_sq_diff.sqrt();
    // 将距离转换为相似度分数（距离越小，相似度越高）
    1.0 / (1.0 + distance as f32)
}

/// 计算点积相似度
pub fn calculate_dot_product(vec1: &[f32], vec2: &[f32]) -> f32 {
    if vec1.len() != vec2.len() || vec1.is_empty() {
        return 0.0;
    }
    
    let mut dot_product = 0.0;
    for i in 0..vec1.len() {
        dot_product += vec1[i] as f64 * vec2[i] as f64;
    }
    
    dot_product as f32
} 