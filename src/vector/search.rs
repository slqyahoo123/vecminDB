use crate::{Error, Result};
use crate::vector::types::Vector;
use serde::{Serialize, Deserialize};
use std::collections::HashMap;
use super::index::{VectorIndex, IndexType, SearchResult};
use super::index::IndexConfig;
use super::search_options::{SearchOptions, VectorSearchQuery};
use rayon::prelude::*;

/// 向量元数据，用于存储向量的附加信息
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct VectorMetadata {
    pub properties: HashMap<String, serde_json::Value>,
}

/// 向量查询参数
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct VectorQuery {
    pub vector: Vec<f32>,
    pub filter: Option<VectorMetadata>,
    pub top_k: usize,
    pub include_metadata: bool,
    pub include_vectors: bool,
}

/// 向量搜索结果
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct VectorSearchResult {
    pub id: String,
    pub score: f32,
    pub metadata: Option<VectorMetadata>,
    pub vector: Option<Vec<f32>>,
}

/// 向量集合配置
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct VectorCollectionConfig {
    pub name: String,
    pub dimension: usize,
    pub index_type: IndexType,
    pub metadata_schema: Option<HashMap<String, String>>,
}

/// 向量集合，用于管理向量数据和进行向量搜索
pub struct VectorCollection {
    pub config: VectorCollectionConfig,
    pub index: Box<dyn VectorIndex>,
    pub vectors: HashMap<String, Vector>,
    pub metadata: HashMap<String, VectorMetadata>,
}

impl std::fmt::Debug for VectorCollection {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("VectorCollection")
            .field("config", &self.config)
            .field("vector_count", &self.vectors.len())
            .finish()
    }
}

impl VectorCollection {
    pub fn new(config: VectorCollectionConfig, index: Box<dyn VectorIndex>) -> Self {
        Self {
            config,
            index,
            vectors: HashMap::new(),
            metadata: HashMap::new(),
        }
    }

    /// 添加向量（便捷方法，匹配README示例）
    /// 
    /// # 参数
    /// * `id` - 向量ID
    /// * `values` - 向量值
    /// * `metadata` - 可选的元数据
    /// 
    /// # 示例
    /// ```
    /// collection.add_vector("vec1", &vec![0.1; 128], None).await?;
    /// ```
    pub async fn add_vector(
        &mut self,
        id: &str,
        values: &[f32],
        metadata: Option<&serde_json::Value>,
    ) -> Result<()> {
        // 验证向量维度
        if values.len() != self.config.dimension {
            return Err(Error::InvalidInput(format!(
                "Vector dimension mismatch: expected {}, got {}",
                self.config.dimension,
                values.len()
            )));
        }

        // 转换元数据
        let vector_metadata = metadata.map(|m| {
            use std::collections::HashMap;
            let mut properties = HashMap::new();
            if let Some(obj) = m.as_object() {
                for (k, v) in obj {
                    properties.insert(k.clone(), v.clone());
                }
            }
            VectorMetadata { properties }
        });

        // 创建向量对象
        let vector = Vector {
            id: id.to_string(),
            data: values.to_vec(),
            metadata: vector_metadata.clone(),
        };

        // 添加到索引
        self.index.add(vector.clone())?;

        // 存储向量和元数据
        if let Some(meta) = vector_metadata {
            self.metadata.insert(id.to_string(), meta);
        }
        
        self.vectors.insert(id.to_string(), vector);
        
        Ok(())
    }

    /// 添加向量（使用Vector对象）
    pub fn add_vector_obj(&mut self, vector: Vector) -> Result<()> {
        // 验证向量维度
        if vector.data.len() != self.config.dimension {
            return Err(Error::InvalidInput(format!(
                "Vector dimension mismatch: expected {}, got {}",
                self.config.dimension,
                vector.data.len()
            )));
        }

        // 添加到索引
        self.index.add(vector.clone())?;

        // 存储向量和元数据
        if let Some(metadata) = &vector.metadata {
            self.metadata.insert(vector.id.clone(), metadata.clone());
        }
        
        self.vectors.insert(vector.id.clone(), vector);
        
        Ok(())
    }

    /// 搜索向量（便捷方法，匹配README示例）
    /// 
    /// # 参数
    /// * `vector` - 查询向量
    /// * `top_k` - 返回结果数量
    /// * `metric` - 相似度度量方法
    /// 
    /// # 示例
    /// ```
    /// use vecmindb::SimilarityMetric;
    /// let results = collection.search(&vec![0.15; 128], 10, SimilarityMetric::Cosine).await?;
    /// ```
    pub async fn search(
        &self,
        vector: &[f32],
        top_k: usize,
        _metric: crate::vector::core::operations::SimilarityMetric,
    ) -> Result<Vec<VectorSearchResult>> {
        // 验证查询向量维度
        if vector.len() != self.config.dimension {
            return Err(Error::InvalidInput(format!(
                "Query vector dimension mismatch: expected {}, got {}",
                self.config.dimension,
                vector.len()
            )));
        }

        // 执行搜索
        let results = self.index.search(vector, top_k)?;
        
        // 转换为搜索结果
        let mut search_results = Vec::with_capacity(results.len());
        for result in results {
            let id = result.id.clone();
            
            // 获取元数据
            let metadata = self.metadata.get(&id).cloned();
            
            // 不包含向量数据（节省内存）
            let vector_data = None;
            
            search_results.push(VectorSearchResult {
                id,
                score: result.score(),
                metadata,
                vector: vector_data,
            });
        }
        
        Ok(search_results)
    }

    /// 搜索向量（使用VectorQuery对象）
    /// 搜索向量（使用VectorQuery对象，重命名以避免冲突）
    pub fn search_with_vector_query(&self, query: &VectorQuery) -> Result<Vec<VectorSearchResult>> {
        // 验证查询向量维度
        if query.vector.len() != self.config.dimension {
            return Err(Error::InvalidInput(format!(
                "Query vector dimension mismatch: expected {}, got {}",
                self.config.dimension,
                query.vector.len()
            )));
        }

        // 执行搜索
        let results = self.index.search(&query.vector, query.top_k)?;
        
        // 转换为搜索结果
        let mut search_results = Vec::with_capacity(results.len());
        for result in results {
            let id = result.id.clone();
            
            // 获取元数据（如果需要）
            let metadata = if query.include_metadata {
                self.metadata.get(&id).cloned()
            } else {
                None
            };
            
            // 获取向量数据（如果需要）
            let vector = if query.include_vectors {
                self.vectors.get(&id).map(|v| v.data.clone())
            } else {
                None
            };
            
            search_results.push(VectorSearchResult {
                id,
                score: result.score(),
                metadata,
                vector,
            });
        }
        
        Ok(search_results)
    }

    /// 使用高级搜索选项进行搜索
    pub fn search_with_options(&self, vector: &[f32], options: &SearchOptions) -> Result<Vec<VectorSearchResult>> {
        // 验证查询向量维度
        if vector.len() != self.config.dimension {
            return Err(Error::InvalidInput(format!(
                "Query vector dimension mismatch: expected {}, got {}",
                self.config.dimension,
                vector.len()
            )));
        }

        // 如果指定了从某个ID的向量搜索
        let search_vector = if let Some(from_id) = &options.from_vector_id {
            if let Some(v) = self.vectors.get(from_id) {
                v.data.clone()
            } else {
                return Err(Error::NotFound(format!("Vector with ID {} not found", from_id)));
            }
        } else {
            vector.to_vec()
        };

        // 是否使用并行搜索
        let mut results = if options.parallel && self.vectors.len() > 1000 {
            // 并行搜索实现
            self.parallel_search(&search_vector, options)?
        } else {
            // 标准搜索实现
            self.index.search(&search_vector, options.limit)?
        };

        // 应用预过滤ID
        if let Some(pre_filter_ids) = &options.pre_filter_ids {
            let id_set: std::collections::HashSet<_> = pre_filter_ids.iter().collect();
            results = results.into_iter()
                .filter(|r| id_set.contains(&r.id))
                .collect();
        }

        // 应用相似度阈值过滤
        if let Some(threshold) = options.score_threshold {
            results = results.into_iter()
                .filter(|r| r.score() >= threshold)
                .collect();
        }

        // 对结果进行排序（如果特殊排序选项）
        match &options.sort_by {
            super::search_options::SortOrder::ScoreAscending => {
                results.sort_by(|a, b| a.score().partial_cmp(&b.score()).unwrap_or(std::cmp::Ordering::Equal));
            },
            super::search_options::SortOrder::ScoreDescending => {
                results.sort_by(|a, b| b.score().partial_cmp(&a.score()).unwrap_or(std::cmp::Ordering::Equal));
            },
            super::search_options::SortOrder::IdAscending => {
                results.sort_by(|a, b| a.id.cmp(&b.id));
            },
            super::search_options::SortOrder::IdDescending => {
                results.sort_by(|a, b| b.id.cmp(&a.id));
            },
            super::search_options::SortOrder::Custom(field) => {
                // 由于SearchResult中没有完整的元数据，这里需要从元数据映射中获取
                results.sort_by(|a, b| {
                    let a_val = self.metadata.get(&a.id)
                        .and_then(|m| m.properties.get(field))
                        .map(|v| v.to_string());
                    let b_val = self.metadata.get(&b.id)
                        .and_then(|m| m.properties.get(field))
                        .map(|v| v.to_string());
                    a_val.cmp(&b_val)
                });
            },
        }

        // 应用过滤条件
        if let Some(filter) = &options.filter {
            let original_size = results.len();
            results = results.into_iter()
                .filter(|r| {
                    if let Some(metadata) = self.metadata.get(&r.id) {
                        filter.matches(metadata)
                    } else {
                        // 如果没有元数据，根据过滤条件决定是否保留
                        filter.matches_empty()
                    }
                })
                .collect();
            
            if results.len() < options.limit && results.len() < original_size {
                // 过滤后结果数量不足，可以考虑扩大搜索范围再过滤
                // 实际项目中可以根据需求实现
                log::debug!("Filter reduced results from {} to {}", original_size, results.len());
            }
        }

        // 处理重排序（如果配置了重排序）
        if let Some(rerank) = &options.rerank {
            if let Some(model) = &rerank.model {
                // 使用重排序模型
                // 注意：这里通常需要一个外部模型的实现
                log::info!("Applying reranking model: {}", model);
                
                // 实际项目中可以添加重排序实现
                // 简单示例：根据向量值与额外特征重新计算分数
                if rerank.params.contains_key("boost_field") {
                    let boost_field = &rerank.params["boost_field"];
                    
                    for result in &mut results {
                        if let Some(metadata) = self.metadata.get(&result.id) {
                            if let Some(boost_value) = metadata.properties.get(boost_field) {
                                if let Some(boost) = boost_value.as_f64() {
                                    // 简单的分数调整
                                    result.distance = result.distance * (boost as f32);
                                }
                            }
                        }
                    }
                    
                    // 根据新分数重新排序
                    results.sort_by(|a, b| b.distance.partial_cmp(&a.distance).unwrap_or(std::cmp::Ordering::Equal));
                }
            }
        }

        // 限制结果数量
        if results.len() > options.limit {
            results.truncate(options.limit);
        }

        // 转换为搜索结果
        let mut search_results = Vec::with_capacity(results.len());
        for result in results {
            let id = result.id.clone();
            
            // 获取元数据（如果需要）
            let metadata = if options.include_metadata {
                self.metadata.get(&id).cloned()
            } else {
                None
            };
            
            // 获取向量数据（如果需要）
            let vector = if options.include_vectors {
                self.vectors.get(&id).map(|v| v.data.clone())
            } else {
                None
            };
            
            search_results.push(VectorSearchResult {
                id,
                score: result.score(),
                metadata,
                vector,
            });
        }
        
        Ok(search_results)
    }

    /// 使用查询对象执行搜索
    pub fn search_with_query(&self, query: &VectorSearchQuery) -> Result<Vec<VectorSearchResult>> {
        query.validate()?;
        
        // 如果提供了ID列表，使用ID查询
        if let Some(ids) = &query.ids {
            let mut results = Vec::new();
            
            for id in ids {
                if let Some(vector) = self.vectors.get(id) {
                    let score = if query.vector.is_empty() {
                        1.0 // 如果没有提供查询向量，则默认分数为1.0
                    } else {
                        // 计算与查询向量的相似度
                        self.compute_similarity(&query.vector, &vector.data, query.options.metric)?
                    };
                    
                    // 过滤：如果配置了过滤条件，则基于元数据应用
                    if let Some(filter) = &query.options.filter {
                        if let Some(metadata) = self.metadata.get(id) {
                            if !filter.matches(metadata) {
                                continue; // 不匹配则跳过
                            }
                        } else if !filter.matches_empty() {
                            continue; // 无元数据且不匹配空条件则跳过
                        }
                    }

                    results.push(SearchResult { id: id.clone(), distance: score, metadata: None });
                }
            }
            
            // 根据分数排序
            results.sort_by(|a, b| b.distance.partial_cmp(&a.distance).unwrap_or(std::cmp::Ordering::Equal));
            
            // 转换并返回结果
            let mut search_results = Vec::with_capacity(results.len());
            for result in results {
                search_results.push(VectorSearchResult {
                    id: result.id.clone(),
                    score: result.distance,
                    metadata: if query.options.include_metadata {
                        self.metadata.get(&result.id).cloned()
                    } else {
                        None
                    },
                    vector: if query.options.include_vectors {
                        self.vectors.get(&result.id).map(|v| v.data.clone())
                    } else {
                        None
                    },
                });
            }
            
            return Ok(search_results);
        }
        
        // 使用向量搜索
        self.search_with_options(&query.vector, &query.options)
    }

    /// 并行搜索实现（用于大型集合）
    fn parallel_search(&self, query: &[f32], options: &SearchOptions) -> Result<Vec<SearchResult>> {
        // 使用rayon并行处理
        // 此函数仅在集合足够大时才有意义
        let results: Vec<SearchResult> = self.vectors.par_iter()
            .map(|(id, vector)| {
                let distance = self.compute_similarity(query, &vector.data, options.metric)
                    .unwrap_or(0.0);
                
                SearchResult {
                    id: id.clone(),
                    distance,
                    metadata: None,
                }
            })
            .collect();
        
        // 对结果排序并限制数量
        let mut sorted_results = results;
        sorted_results.sort_by(|a, b| b.distance.partial_cmp(&a.distance).unwrap_or(std::cmp::Ordering::Equal));
        
        if sorted_results.len() > options.limit {
            sorted_results.truncate(options.limit);
        }
        
        Ok(sorted_results)
    }

    /// 计算两个向量之间的相似度
    fn compute_similarity(&self, a: &[f32], b: &[f32], metric: super::SimilarityMetric) -> Result<f32> {
        if a.len() != b.len() {
            return Err(Error::InvalidInput(format!(
                "Vector dimension mismatch: {} vs {}", a.len(), b.len()
            )));
        }
        
        match metric {
            super::SimilarityMetric::Cosine => {
                let mut dot_product = 0.0;
                let mut a_norm = 0.0;
                let mut b_norm = 0.0;
                
                for i in 0..a.len() {
                    dot_product += a[i] * b[i];
                    a_norm += a[i] * a[i];
                    b_norm += b[i] * b[i];
                }
                
                a_norm = a_norm.sqrt();
                b_norm = b_norm.sqrt();
                
                if a_norm == 0.0 || b_norm == 0.0 {
                    Ok(0.0)
                } else {
                    Ok(dot_product / (a_norm * b_norm))
                }
            },
            super::SimilarityMetric::Euclidean => {
                let mut sum = 0.0;
                
                for i in 0..a.len() {
                    let diff = a[i] - b[i];
                    sum += diff * diff;
                }
                
                // 欧氏距离转换为相似度（1 / (1 + distance)）
                let distance = sum.sqrt();
                Ok(1.0 / (1.0 + distance))
            },
            super::SimilarityMetric::DotProduct => {
                let mut dot_product = 0.0;
                
                for i in 0..a.len() {
                    dot_product += a[i] * b[i];
                }
                
                Ok(dot_product)
            },
            super::SimilarityMetric::Manhattan => {
                let mut sum = 0.0;
                
                for i in 0..a.len() {
                    sum += (a[i] - b[i]).abs();
                }
                
                // 曼哈顿距离转换为相似度（1 / (1 + distance)）
                Ok(1.0 / (1.0 + sum))
            },
        }
    }

    pub fn get_vector(&self, id: &str) -> Option<&Vector> {
        self.vectors.get(id)
    }

    pub fn get_metadata(&self, id: &str) -> Option<&VectorMetadata> {
        self.metadata.get(id)
    }

    pub fn delete_vector(&mut self, id: &str) -> Result<()> {
        // 从索引中删除
        self.index.delete(id)?;
        
        // 从存储中删除
        self.vectors.remove(id);
        self.metadata.remove(id);
        
        Ok(())
    }

    pub fn count(&self) -> usize {
        self.vectors.len()
    }

    /// 添加向量（指定ID版本）
    pub async fn add_vector_with_id(&mut self, id: &str, vector: Vector) -> Result<()> {
        // 创建带ID的向量
        let mut new_vector = vector;
        new_vector.id = id.to_string();
        
        // 验证向量维度
        if new_vector.data.len() != self.config.dimension {
            return Err(Error::InvalidInput(format!(
                "Vector dimension mismatch: expected {}, got {}",
                self.config.dimension,
                new_vector.data.len()
            )));
        }

        // 添加到索引
        self.index.add(new_vector.clone())?;

        // 存储向量和元数据
        if let Some(metadata) = &new_vector.metadata {
            self.metadata.insert(new_vector.id.clone(), metadata.clone());
        }
        
        self.vectors.insert(new_vector.id.clone(), new_vector);
        
        Ok(())
    }
    
    /// 获取向量（异步版本）
    pub async fn get_vector_async(&self, id: &str) -> Result<Vector> {
        self.vectors.get(id)
            .cloned()
            .ok_or_else(|| Error::NotFound(format!("Vector not found: {}", id)))
    }
    
    /// 删除向量（异步版本）
    pub async fn delete_vector_async(&mut self, id: &str) -> Result<()> {
        // 从索引中删除
        self.index.delete(id)?;
        
        // 从存储中删除
        self.vectors.remove(id);
        self.metadata.remove(id);
        
        Ok(())
    }
    
    /// 搜索向量（异步版本）
    pub async fn search_async(&self, query: VectorQuery) -> Result<Vec<VectorSearchResult>> {
        // 验证查询向量维度
        if query.vector.len() != self.config.dimension {
            return Err(Error::InvalidInput(format!(
                "Query vector dimension mismatch: expected {}, got {}",
                self.config.dimension,
                query.vector.len()
            )));
        }

        // 执行搜索
        let results = self.index.search(&query.vector, query.top_k)?;
        
        // 转换为搜索结果
        let mut search_results = Vec::with_capacity(results.len());
        for result in results {
            let id = result.id.clone();
            
            // 获取元数据（如果需要）
            let metadata = if query.include_metadata {
                self.metadata.get(&id).cloned()
            } else {
                None
            };
            
            // 获取向量数据（如果需要）
            let vector = if query.include_vectors {
                self.vectors.get(&id).map(|v| v.data.clone())
            } else {
                None
            };
            
            search_results.push(VectorSearchResult {
                id,
                score: result.score(),
                metadata,
                vector,
            });
        }
        
        Ok(search_results)
    }
    
    /// 统计向量数量（异步版本）
    pub async fn count_async(&self) -> Result<usize> {
        Ok(self.vectors.len())
    }
    
    /// 更新向量
    pub async fn update_vector(&mut self, id: &str, data: Vec<f32>, metadata: Option<VectorMetadata>) -> Result<()> {
        // 检查向量是否存在
        if !self.vectors.contains_key(id) {
            return Err(Error::NotFound(format!("Vector not found: {}", id)));
        }
        
        // 验证向量维度
        if data.len() != self.config.dimension {
            return Err(Error::InvalidInput(format!(
                "Vector dimension mismatch: expected {}, got {}",
                self.config.dimension,
                data.len()
            )));
        }
        
        // 创建新向量
        let new_vector = Vector {
            id: id.to_string(),
            data,
            metadata: metadata.clone(),
        };
        
        // 从索引中删除旧向量
        self.index.delete(id)?;
        
        // 添加新向量到索引
        self.index.add(new_vector.clone())?;
        
        // 更新存储
        self.vectors.insert(id.to_string(), new_vector);
        
        if let Some(meta) = metadata {
            self.metadata.insert(id.to_string(), meta);
        } else {
            self.metadata.remove(id);
        }
        
        Ok(())
    }
    /// API需要的方法 - 重建索引（无返回值版本）
    pub async fn rebuild_index(&mut self) -> Result<()> {
        // 清空索引
        // 注意：这里需要重新创建索引，因为VectorIndex trait没有clear方法
        let new_index = VectorIndexFactory::create_index_with_type(
            self.config.index_type.clone(), 
            self.config.dimension
        )?;
        self.index = new_index;
        
        // 重新添加所有向量到索引
        for vector in self.vectors.values() {
            self.index.add(vector.clone())?;
        }
        
        Ok(())
    }
}

/// 向量索引工厂，用于创建不同类型的向量索引
pub struct VectorIndexFactory;

impl VectorIndexFactory {
    /// 创建向量索引
    /// 接收完整的IndexConfig配置来保持与底层工厂一致的接口
    pub fn create_index(config: IndexConfig) -> Result<Box<dyn VectorIndex>> {
        // 创建索引
        let index_enum = super::index::factory::VectorIndexFactory::create_index(config)?;
        
        // 将枚举转换为特性对象
        // 注意：VectorIndexEnum 实现了 VectorIndex trait，可以直接使用
        // 虽然 HNSW 的部分方法会返回错误，但至少可以编译和运行
        Ok(Box::new(index_enum))
    }
    
    /// 快捷方法，创建具有指定类型和维度的索引
    pub fn create_index_with_type(index_type: IndexType, dimension: usize) -> Result<Box<dyn VectorIndex>> {
        // 创建索引配置
        let mut config = IndexConfig::default();
        config.index_type = index_type;
        config.dimension = dimension;
        
        Self::create_index(config)
    }
}

/// 向量搜索参数（VectorSearchQuery的别名）
pub type VectorSearchParams = VectorSearchQuery; 