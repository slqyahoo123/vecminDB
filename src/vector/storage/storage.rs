use crate::{Error, Result};
use crate::vector::types::Vector;
use crate::vector::index::{VectorIndex, SearchResult, IndexConfig, VectorIndexEnum, FlatIndex, HNSWIndex, IVFIndex, PQIndex, LSHIndex, VPTreeIndex, IndexType};
use std::path::Path;
use std::sync::{Arc, RwLock};
use serde::{Deserialize, Serialize};
use serde_json;
use rayon::prelude::*;
use crate::vector::index::VectorIndexFactory;
use std::collections::HashMap;

/// 向量批量操作请求
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VectorBatchRequest {
    pub vectors: Vec<Vector>,
}

/// 向量搜索请求
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VectorSearchRequest {
    pub query: Vec<f32>,
    pub top_k: usize,
    pub filter: Option<HashMap<String, String>>,
    pub include_metadata: bool,
    pub include_vectors: bool,
}

/// 向量搜索响应
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VectorSearchResponse {
    pub results: Vec<SearchResult>,
    pub took_ms: u64,
}

/// 向量搜索结果
// 兼容别名，复用统一的 SearchResult
pub type VectorSearchResult = SearchResult;

/// 向量存储管理器
#[derive(Clone)]
pub struct VectorStorageManager {
    db: sled::Db,
    index: Arc<RwLock<VectorIndexEnum>>,
    config: Arc<RwLock<IndexConfig>>,
    path: String,
    vectors: Arc<RwLock<HashMap<String, Vector>>>,
}

impl VectorStorageManager {
    /// 创建新的向量存储管理器
    pub fn new(path: impl AsRef<Path>, config: IndexConfig) -> Result<Self> {
        let db = sled::open(path.as_ref().join("vectors"))
            .map_err(|e| Error::Storage(format!("Failed to open vector database: {}", e)))?;
            
        // 创建向量索引
        let index = match config.index_type {
            IndexType::Flat => {
                let flat_index = FlatIndex::new(config.clone());
                Arc::new(RwLock::new(VectorIndexEnum::Flat(flat_index)))
            },
            IndexType::HNSW => {
                // 从IndexConfig提取HNSW所需的参数
                let m = config.hnsw_m;
                let ef_construction = config.hnsw_ef_construction;
                let distance_type = crate::vector::index::hnsw::types::DistanceType::from_similarity_metric(config.metric);
                let ml = 1.0 / (m as f32).ln(); // HNSW算法的层级参数
                let max_level_limit = 16; // 默认最大层级限制
                let index_id = format!("hnsw_{}", uuid::Uuid::new_v4());
                
                let hnsw_index = HNSWIndex::new(
                    config.dimension,
                    m,
                    ef_construction,
                    distance_type,
                    ml,
                    max_level_limit,
                    index_id,
                );
                Arc::new(RwLock::new(VectorIndexEnum::HNSW(hnsw_index)))
            },
            IndexType::IVF => {
                let ivf_index = IVFIndex::new(config.clone())?;
                Arc::new(RwLock::new(VectorIndexEnum::IVF(ivf_index)))
            },
            IndexType::PQ => {
                let pq_index = PQIndex::new(config.clone())?;
                Arc::new(RwLock::new(VectorIndexEnum::PQ(pq_index)))
            },
            IndexType::LSH => {
                let lsh_index = LSHIndex::new(config.clone())?;
                Arc::new(RwLock::new(VectorIndexEnum::LSH(lsh_index)))
            },
            IndexType::VPTree => {
                let vptree_index = VPTreeIndex::new(config.clone())?;
                Arc::new(RwLock::new(VectorIndexEnum::VPTree(vptree_index)))
            },
            _ => return Err(Error::vector(format!("Unsupported index type: {:?}", config.index_type))),
        };
        
        // 从数据库加载向量到索引
        let manager = Self { 
            db, 
            index, 
            config: Arc::new(RwLock::new(config)),
            path: path.as_ref().to_string_lossy().to_string(),
            vectors: Arc::new(RwLock::new(HashMap::new())),
        };
        manager.load_vectors_to_index()?;
        
        Ok(manager)
    }

    /// 轻量级就绪检查：验证DB可访问、索引读锁可获取、工厂可根据当前配置创建实例
    pub fn readiness_probe(&self) -> Result<()> {
        // 1) DB探针：尝试读取一个键迭代器项（不消费，不写入）
        let _ = self.db.iter().next();

        // 2) 索引读锁：确保索引对象可读
        let _index_guard = self.index.read().map_err(|_| Error::vector("Failed to acquire read lock on vector index".to_string()))?;

        // 3) 工厂创建：用当前配置尝试创建一个最小索引实例，验证参数接线
        let cfg = self.config.read().map_err(|_| Error::vector("Failed to acquire read lock on index config".to_string()))?.clone();
        let _ = VectorIndexFactory::create_index(cfg)?;

        Ok(())
    }
    
    /// 从数据库加载向量到索引
    fn load_vectors_to_index(&self) -> Result<()> {
        let mut index = self.index.write().map_err(|_| 
            Error::vector("Failed to acquire write lock on index".to_string())
        )?;
        
        // 并行加载向量
        let vectors: Result<Vec<Vector>> = self.db.iter()
            .map(|item| {
                let (_, value) = item.map_err(|e| 
                    Error::Storage(format!("Failed to iterate over vectors: {}", e))
                )?;
                
                let vector: Vector = serde_json::from_slice(&value)
                    .map_err(|e| Error::Storage(format!("Failed to deserialize vector: {}", e)))?;
                    
                Ok(vector)
            })
            .collect();
        
        // 添加向量到索引
        for vector in vectors? {
            index.add(vector)?;
        }
        
        Ok(())
    }
    
    /// 插入单个向量
    pub fn insert(&self, vector: &Vector) -> Result<()> {
        // 验证向量维度
        if vector.data.len() != self.config.read().map_err(|_| 
            Error::vector("Failed to acquire read lock on config".to_string())
        )?.dimension {
            return Err(Error::vector(format!(
                "Vector dimension mismatch: expected {}, got {}",
                self.config.read().map_err(|_| 
                    Error::vector("Failed to acquire read lock on config".to_string())
                )?.dimension, vector.data.len()
            )));
        }
        
        // 序列化向量
        let value = serde_json::to_vec(vector)
            .map_err(|e| Error::Storage(format!("Failed to serialize vector: {}", e)))?;
            
        // 存储到数据库
        self.db
            .insert(vector.id.as_bytes(), value)
            .map_err(|e| Error::Storage(format!("Failed to insert vector: {}", e)))?;
            
        // 添加到索引
        let mut index = self.index.write().map_err(|_| 
            Error::vector("Failed to acquire write lock on index".to_string())
        )?;
        
        index.add(vector.clone())?;
        
        // 刷新数据库
        self.db.flush()
            .map_err(|e| Error::Storage(format!("Failed to flush database: {}", e)))?;
            
        Ok(())
    }
    
    /// 批量插入向量
    pub fn batch_insert(&self, vectors: &[Vector]) -> Result<()> {
        if vectors.is_empty() {
            return Ok(());
        }
        
        // 验证向量维度
        for vector in vectors {
            if vector.data.len() != self.config.read().map_err(|_| 
                Error::vector("Failed to acquire read lock on config".to_string())
            )?.dimension {
                return Err(Error::vector(format!(
                    "Vector dimension mismatch: expected {}, got {}",
                    self.config.read().map_err(|_| 
                        Error::vector("Failed to acquire read lock on config".to_string())
                    )?.dimension, vector.data.len()
                )));
            }
        }
        
        // 批量写入数据库
        let batch = vectors.par_iter().map(|vector| {
            let key = vector.id.as_bytes().to_vec();
            let value = serde_json::to_vec(vector)
                .map_err(|e| Error::Storage(format!("Failed to serialize vector: {}", e)))?;
                
            Ok((key, value))
        }).collect::<Result<Vec<(Vec<u8>, Vec<u8>)>>>()?;
        
        // 使用事务批量写入
        let mut db_batch = sled::Batch::default();
        for (key, value) in batch {
            db_batch.insert(key, value);
        }
        
        self.db.apply_batch(db_batch)
            .map_err(|e| Error::Storage(format!("Failed to apply batch: {}", e)))?;
            
        // 添加到索引
        let mut index = self.index.write().map_err(|_| 
            Error::vector("Failed to acquire write lock on index".to_string())
        )?;
        
        for vector in vectors {
            index.add(vector.clone())?;
        }
        
        // 刷新数据库
        self.db.flush()
            .map_err(|e| Error::Storage(format!("Failed to flush database: {}", e)))?;
            
        Ok(())
    }
    
    /// 获取单个向量
    pub fn get(&self, id: &str) -> Result<Option<Vector>> {
        match self.db.get(id.as_bytes())
            .map_err(|e| Error::Storage(format!("Failed to get vector: {}", e)))? {
            Some(data) => {
                let vector = serde_json::from_slice(&data)
                    .map_err(|e| Error::Storage(format!("Failed to deserialize vector: {}", e)))?;
                Ok(Some(vector))
            }
            None => Ok(None),
        }
    }
    
    /// 删除单个向量
    pub fn delete(&self, id: &str) -> Result<()> {
        // 从数据库删除
        self.db
            .remove(id.as_bytes())
            .map_err(|e| Error::Storage(format!("Failed to delete vector: {}", e)))?;
            
        // 从索引删除
        let mut index = self.index.write().map_err(|_| 
            Error::vector("Failed to acquire write lock on index".to_string())
        )?;
        
        index.delete(id)?;
        
        // 刷新数据库
        self.db.flush()
            .map_err(|e| Error::Storage(format!("Failed to flush database: {}", e)))?;
            
        Ok(())
    }
    
    /// 批量删除向量
    pub fn batch_delete(&self, ids: &[String]) -> Result<()> {
        if ids.is_empty() {
            return Ok(());
        }
        
        // 批量从数据库删除
        let mut db_batch = sled::Batch::default();
        for id in ids {
            db_batch.remove(id.as_bytes());
        }
        
        self.db.apply_batch(db_batch)
            .map_err(|e| Error::Storage(format!("Failed to apply batch: {}", e)))?;
            
        // 从索引删除
        let mut index = self.index.write().map_err(|_| 
            Error::vector("Failed to acquire write lock on index".to_string())
        )?;
        
        for id in ids {
            index.delete(id)?;
        }
        
        // 刷新数据库
        self.db.flush()
            .map_err(|e| Error::Storage(format!("Failed to flush database: {}", e)))?;
            
        Ok(())
    }
    
    /// 搜索相似向量
    pub fn search(&self, request: &VectorSearchRequest) -> Result<VectorSearchResponse> {
        let start_time = std::time::Instant::now();
        
        // 验证查询向量维度
        if request.query.len() != self.config.read().map_err(|_| 
            Error::vector("Failed to acquire read lock on config".to_string())
        )?.dimension {
            return Err(Error::vector(format!(
                "Query vector dimension mismatch: expected {}, got {}",
                self.config.read().map_err(|_| 
                    Error::vector("Failed to acquire read lock on config".to_string())
                )?.dimension, request.query.len()
            )));
        }
        
        // 获取索引读锁
        let index = self.index.read().map_err(|_| 
            Error::vector("Failed to acquire read lock on index".to_string())
        )?;
        
        // 执行搜索
        let mut results = index.search(&request.query, request.top_k)?;
        
        // 应用过滤器
        if let Some(filter) = &request.filter {
            results = self.apply_filter(results, filter)?;
        }
        
        let took_ms = start_time.elapsed().as_millis() as u64;
        
        Ok(VectorSearchResponse {
            results,
            took_ms,
        })
    }
    
    /// 应用元数据过滤器
    fn apply_filter(&self, results: Vec<SearchResult>, filter: &HashMap<String, String>) -> Result<Vec<SearchResult>> {
        let filtered_results = results.into_iter()
            .filter(|result| {
                if let Some(metadata) = &result.metadata {
                    // 简单的相等过滤
                    let mut match_count = 0;
                    for (key, value) in filter {
                        if let Some(metadata_value) = metadata.as_object().and_then(|obj| obj.get(key)) {
                            if metadata_value.as_str() == Some(value) {
                                match_count += 1;
                            }
                        }
                    }
                    return match_count == filter.len();
                }
                false
            })
            .collect();
            
        Ok(filtered_results)
    }
    
    /// 获取向量数量
    pub fn count(&self) -> Result<usize> {
        let index = self.index.read().map_err(|_| 
            Error::vector("Failed to acquire read lock on index".to_string())
        )?;
        
        Ok(index.size())
    }
    
    /// 重建索引
    pub fn rebuild_index(&self) -> Result<()> {
        // 清空索引
        let mut index = self.index.write().map_err(|_| 
            Error::vector("Failed to acquire write lock on index".to_string())
        )?;
        
        // 获取配置
        let config = self.config.read().map_err(|_| 
            Error::vector("Failed to acquire read lock on config".to_string())
        )?.clone();
        
        // 创建新索引
        let new_index = match config.index_type {
            IndexType::Flat => {
                VectorIndexEnum::Flat(FlatIndex::new(config))
            },
            IndexType::HNSW => {
                // 从IndexConfig提取HNSW所需的参数
                let m = config.hnsw_m;
                let ef_construction = config.hnsw_ef_construction;
                let distance_type = crate::vector::index::hnsw::types::DistanceType::from_similarity_metric(config.metric);
                let ml = 1.0 / (m as f32).ln();
                let max_level_limit = 16;
                let index_id = format!("hnsw_{}", uuid::Uuid::new_v4());
                
                let hnsw_index = HNSWIndex::new(
                    config.dimension,
                    m,
                    ef_construction,
                    distance_type,
                    ml,
                    max_level_limit,
                    index_id,
                );
                VectorIndexEnum::HNSW(hnsw_index)
            },
            IndexType::IVF => {
                VectorIndexEnum::IVF(IVFIndex::new(config)?)
            },
            IndexType::PQ => {
                VectorIndexEnum::PQ(PQIndex::new(config)?)
            },
            IndexType::LSH => {
                VectorIndexEnum::LSH(LSHIndex::new(config)?)
            },
            IndexType::VPTree => {
                VectorIndexEnum::VPTree(VPTreeIndex::new(config)?)
            },
            _ => return Err(Error::vector(format!("Unsupported index type: {:?}", config.index_type))),
        };
        
        // 替换索引
        *index = new_index;
        
        // 重新加载向量
        drop(index); // 释放写锁
        self.load_vectors_to_index()?;
        
        Ok(())
    }
    
    /// 获取所有向量
    pub fn get_all_vectors(&self) -> Result<Vec<Vector>> {
        let vectors: Result<Vec<Vector>> = self.db.iter()
            .map(|item| {
                let (_, value) = item.map_err(|e| 
                    Error::Storage(format!("Failed to read vector from database: {}", e))
                )?;
                
                let vector: Vector = bincode::deserialize(&value)
                    .map_err(|e| Error::Storage(format!("Failed to deserialize vector: {}", e)))?;
                
                Ok(vector)
            })
            .collect();
            
        vectors
    }
    
    /// 获取当前索引配置
    pub fn get_index_config(&self) -> Result<IndexConfig> {
        let config = self.config.read().map_err(|_| 
            Error::vector("Failed to acquire read lock on config".to_string())
        )?;
        
        Ok(config.clone())
    }
    
    /// 设置索引配置
    pub fn set_index_config(&self, config: IndexConfig) -> Result<()> {
        let mut current_config = self.config.write().map_err(|_| 
            Error::vector("Failed to acquire write lock on config".to_string())
        )?;
        
        // 确保维度保持一致
        let mut new_config = config;
        new_config.dimension = current_config.dimension;
        
        *current_config = new_config;
        Ok(())
    }
    
    /// 获取相似度度量方式
    pub fn get_similarity_metric(&self) -> Result<crate::vector::core::operations::SimilarityMetric> {
        let config = self.config.read().map_err(|_| 
            Error::vector("Failed to acquire read lock on config".to_string())
        )?;
        
        Ok(config.metric)
    }
    
    /// 导出索引数据
    pub fn export_index(&self) -> Result<Vec<u8>> {
        VectorIndexFactory::serialize_index(&self.index, self.config.read().map_err(|_| 
            Error::vector("Failed to acquire read lock on config".to_string())
        )?.index_type)
    }
    
    /// 导入索引数据
    pub fn import_index(&self, data: &[u8]) -> Result<()> {
        let new_index = VectorIndexFactory::from_serialized(data, self.config.read().map_err(|_| 
            Error::Storage("Failed to acquire read lock on config".to_string())
        )?.clone())?;
        
        let mut current_index = self.index.write().map_err(|_| 
            Error::Storage("Failed to acquire write lock on index".to_string())
        )?;
        
        // 使用新索引的数据反序列化当前索引
        current_index.deserialize(data)?;
        
        Ok(())
    }

    /// 添加向量 (异步版本)
    pub async fn add_vector(&self, id: &str, values: &[f32], metadata: Option<&serde_json::Value>) -> Result<()> {
        let vector = Vector {
            id: id.to_string(),
            data: values.to_vec(),
            metadata: metadata.map(|m| crate::vector::search::VectorMetadata {
                properties: serde_json::from_value(m.clone()).unwrap_or_default(),
            }),
        };
        
        self.insert(&vector)
    }

    /// 获取向量 (异步版本)
    pub async fn get_vector(&self, id: &str) -> Result<Option<(Vec<f32>, Option<crate::vector::search::VectorMetadata>)>> {
        match self.get(id)? {
            Some(vector) => {
                let metadata = vector.metadata;
                Ok(Some((vector.data, metadata)))
            }
            None => Ok(None),
        }
    }

    /// 删除向量 (异步版本)
    pub async fn delete_vector(&self, id: &str) -> Result<bool> {
        match self.db.get(id.as_bytes())
            .map_err(|e| Error::vector(format!("Failed to get vector: {}", e)))? {
            Some(_) => {
                self.delete(id)?;
                Ok(true)
            }
            None => Ok(false),
        }
    }

    /// 搜索向量 (异步版本)
    pub async fn search_vectors(
        &self, 
        query_vector: &[f32], 
        top_k: usize, 
        _metric: &str, 
        filter: Option<&serde_json::Value>
    ) -> Result<Vec<(String, f32, Vec<f32>, Option<crate::vector::search::VectorMetadata>)>> {
        let request = VectorSearchRequest {
            query: query_vector.to_vec(),
            top_k,
            filter: filter.map(|f| {
                if let serde_json::Value::Object(obj) = f {
                    obj.into_iter()
                        .filter_map(|(k, v)| v.as_str().map(|s| (k.to_string(), s.to_string())))
                        .collect()
                } else {
                    HashMap::new()
                }
            }),
            include_metadata: true,
            include_vectors: true,
        };

        let response = self.search(&request)?;

        let mut results = Vec::new();
        for search_item in response.results {
            let vector_data: Option<Vec<f32>> = None; // SearchResult 不携带向量，保持为空
            let metadata = search_item.metadata.as_ref().and_then(|v| v.as_object()).map(|obj| {
                crate::vector::search::VectorMetadata {
                    properties: obj.iter().map(|(k, v)| (k.clone(), v.clone())).collect(),
                }
            });
            results.push((
                search_item.id.clone(),
                search_item.score(),
                vector_data.map(|_| Vec::<f32>::new()).unwrap_or_default(),
                metadata,
            ));
        }
        
        Ok(results)
    }

    /// 获取向量数量 (异步版本)
    pub async fn count_vectors(&self) -> Result<usize> {
        self.count()
    }

    /// 重建索引 (异步版本)
    pub async fn rebuild_index_async(&self) -> Result<()> {
        // 异步重建索引实现
        let _config = {
            let config_guard = self.config.read().map_err(|_| 
                Error::vector("Failed to acquire read lock on config".to_string())
            )?;
            config_guard.clone()
        };
        
        // 在异步上下文中执行重建操作
        tokio::task::spawn_blocking(move || {
            // 这里可以添加异步特定的重建逻辑
            // 例如：并行处理、异步IO等
        }).await.map_err(|e| Error::vector(format!("Failed to rebuild index: {}", e)))?;
        
        // 调用同步版本完成实际重建
        self.rebuild_index()
    }
} 