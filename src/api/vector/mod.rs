//! 向量API模块
//! 提供向量数据库的HTTP API接口

pub mod handlers;
pub mod types;

use std::sync::Arc;
use crate::vector::VectorService;
use crate::Result;
use crate::vector::search::VectorMetadata;

/// 向量API实现
#[derive(Clone)]
pub struct VectorApi {
    service: Arc<dyn VectorService>,
}

impl VectorApi {
    /// 创建新的向量API（基于统一 VectorService 抽象）
    pub fn new(service: Arc<dyn VectorService>) -> Self {
        Self { service }
    }
    
    /// 创建向量
    pub async fn create_vector(&self, request: types::CreateVectorRequest) -> Result<types::Vector> {
        let vector = types::Vector {
            id: request.id.clone(),
            values: request.values.clone(),
            metadata: request.metadata.clone(),
            created_at: chrono::Utc::now().to_rfc3339(),
            updated_at: chrono::Utc::now().to_rfc3339(),
        };
        
        // 委派给统一向量服务
        self.service.add_vector(&request.id, &request.values, request.metadata.as_ref()).await?;
        
        Ok(vector)
    }
    
    /// 批量创建向量
    pub async fn batch_create_vectors(&self, request: types::BatchCreateVectorsRequest) -> Result<Vec<types::Vector>> {
        let mut vectors = Vec::new();
        
        for create_request in request.vectors {
            let vector = types::Vector {
                id: create_request.id.clone(),
                values: create_request.values.clone(),
                metadata: create_request.metadata.clone(),
                created_at: chrono::Utc::now().to_rfc3339(),
                updated_at: chrono::Utc::now().to_rfc3339(),
            };
            
            // 存储向量
            self.service.add_vector(&create_request.id, &create_request.values, create_request.metadata.as_ref()).await?;
            vectors.push(vector);
        }
        
        Ok(vectors)
    }
    
    /// 获取向量
    pub async fn get_vector(&self, vector_id: &str) -> Result<Option<types::Vector>> {
        if let Some((values, metadata)) = self.service.get_vector(vector_id).await? {
            let vector = types::Vector {
                id: vector_id.to_string(),
                values,
                metadata: metadata.as_ref().map(|m| {
                    // 转换VectorMetadata到JSON值
                    serde_json::json!({
                        "properties": m.properties.clone()
                    })
                }),
                created_at: chrono::Utc::now().to_rfc3339(), // 实际应该从存储中获取
                updated_at: chrono::Utc::now().to_rfc3339(),
            };
            Ok(Some(vector))
        } else {
            Ok(None)
        }
    }
    
    /// 删除向量
    pub async fn delete_vector(&self, vector_id: &str) -> Result<bool> {
        self.service.delete_vector(vector_id).await
    }
    
    /// 批量删除向量
    pub async fn batch_delete_vectors(&self, request: types::BatchDeleteVectorsRequest) -> Result<usize> {
        let mut deleted_count = 0;
        
        for vector_id in request.ids {
            if self.service.delete_vector(&vector_id).await? {
                deleted_count += 1;
            }
        }
        
        Ok(deleted_count)
    }
    
    /// 搜索向量
    pub async fn search_vectors(&self, request: types::SearchVectorsRequest) -> Result<Vec<types::VectorSearchResult>> {
        let results = self.service.search_vectors(
            &request.query_vector,
            request.top_k,
            &request.metric,
            request.filter.as_ref()
        ).await?;
        
        let mut search_results = Vec::new();
        
        // 转换搜索结果类型
        for (vector_id, score, vector, metadata) in results {
            let search_result = types::VectorSearchResult {
                id: vector_id.clone(),
                score,
                vector: Some(types::Vector {
                    id: vector_id.clone(),
                    values: vector.clone(),
                    metadata: metadata.as_ref().map(|m| {
                        serde_json::json!({
                            "properties": m.properties.clone()
                        })
                    }),
                    created_at: chrono::Utc::now().to_rfc3339(),
                    updated_at: chrono::Utc::now().to_rfc3339(),
                }),
                metadata: metadata.map(|m| {
                    serde_json::json!({
                        "properties": m.properties
                    })
                }),
            };
            search_results.push(search_result);
        }
        
        Ok(search_results)
    }
    
    /// 更新向量元数据
    pub async fn update_vector_metadata(&self, vector_id: &str, metadata: serde_json::Value) -> Result<bool> {
        // 获取现有向量
        if let Some((values, _)) = self.service.get_vector(vector_id).await? {
            // 更新元数据 - 使用现有的update_vector方法
            self.service.update_vector(vector_id, &values, Some(&metadata)).await?;
            Ok(true)
        } else {
            Ok(false)
        }
    }
    
    /// 获取向量统计信息
    pub async fn get_vector_statistics(&self) -> Result<types::VectorStats> {
        // 使用现有的方法获取统计信息
        let stats = self.service.get_stats().await?;
        Ok(types::VectorStats {
            total_count: stats.count,
            avg_dimension: stats.dimension as f32,
            max_dimension: stats.dimension,
            min_dimension: stats.dimension,
            created_at_range: None,
        })
    }
    
    /// 验证向量数据
    pub fn validate_vector_data(&self, values: &[f32]) -> Result<bool> {
        if values.is_empty() {
            return Err(crate::Error::invalid_input("向量值不能为空"));
        }
        
        if values.iter().any(|&v| v.is_nan() || v.is_infinite()) {
            return Err(crate::Error::invalid_input("向量值包含无效数值"));
        }
        
        Ok(true)
    }
    
    /// 标准化向量值
    pub fn normalize_vector(&self, values: &[f32]) -> Vec<f32> {
        let magnitude: f32 = values.iter().map(|&x| x * x).sum::<f32>().sqrt();
        
        if magnitude > 0.0 {
            values.iter().map(|&x| x / magnitude).collect()
        } else {
            values.to_vec()
        }
    }
    
    /// 统计向量数量
    pub async fn count_vectors(&self) -> Result<usize> {
        self.service.count_vectors().await
    }
    
    /// 重建索引
    pub async fn rebuild_index(&self) -> Result<()> {
        self.service.rebuild_index().await
    }
}

// 重新导出 handlers 中的函数
pub use handlers::{
    create_vector_actix,
    batch_create_vectors_actix,
    get_vector_actix,
    delete_vector_actix,
    batch_delete_vectors_actix,
    search_vectors_actix,
    count_vectors_actix,
    rebuild_index_actix,
};

// 重新导出 types 中的类型
pub use types::*;

