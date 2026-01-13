// 向量距离计算接口和实现
// 提供用于计算向量之间距离的接口和常用距离度量的实现

use crate::Result;
use std::fmt::Debug;

/// 向量距离计算接口
pub trait Distance: Send + Sync + Debug {
    /// 计算两个向量之间的距离
    fn calculate(&self, a: &[f32], b: &[f32]) -> f32;
    
    /// 计算向量与多个向量之间的距离
    fn batch_calculate(&self, query: &[f32], vectors: &[Vec<f32>]) -> Result<Vec<f32>> {
        let mut distances = Vec::with_capacity(vectors.len());
        for vector in vectors {
            distances.push(self.calculate(query, vector));
        }
        Ok(distances)
    }
    
    /// 返回Distance trait对象
    fn as_dyn_distance(self: Box<Self>) -> Box<dyn Distance + Send + Sync>;
}

/// 欧几里得距离
#[derive(Clone, Copy, Debug)]
pub struct EuclideanDistance;

impl Distance for EuclideanDistance {
    fn calculate(&self, a: &[f32], b: &[f32]) -> f32 {
        if a.len() != b.len() {
            return f32::MAX; // 出错时返回最大距离
        }
        
        // 使用VectorOps的欧几里得距离，取负值使其符合距离的语义（越小越相似）
        // 注意VectorOps::euclidean_distance返回的是负值，而我们这里需要正值表示距离
        -crate::vector::core::operations::VectorOps::euclidean_distance(a, b)
    }
    
    fn as_dyn_distance(self: Box<Self>) -> Box<dyn Distance + Send + Sync> {
        self
    }
}

/// 余弦距离
#[derive(Clone, Copy, Debug)]
pub struct CosineDistance;

impl Distance for CosineDistance {
    fn calculate(&self, a: &[f32], b: &[f32]) -> f32 {
        if a.len() != b.len() || a.is_empty() {
            return f32::MAX;
        }
        
        // 使用VectorOps的余弦相似度，转换为距离（1-相似度）
        1.0 - crate::vector::core::operations::VectorOps::cosine_similarity(a, b)
    }
    
    fn as_dyn_distance(self: Box<Self>) -> Box<dyn Distance + Send + Sync> {
        self
    }
}

/// 点积距离
#[derive(Clone, Copy, Debug)]
pub struct DotProductDistance;

impl Distance for DotProductDistance {
    fn calculate(&self, a: &[f32], b: &[f32]) -> f32 {
        if a.len() != b.len() {
            return f32::MAX;
        }
        
        // 使用VectorOps的点积，取负值使其符合距离的语义（越小越相似）
        -crate::vector::core::operations::VectorOps::dot_product(a, b)
    }
    
    fn as_dyn_distance(self: Box<Self>) -> Box<dyn Distance + Send + Sync> {
        self
    }
}

/// 曼哈顿距离
#[derive(Clone, Copy, Debug)]
pub struct ManhattanDistance;

impl Distance for ManhattanDistance {
    fn calculate(&self, a: &[f32], b: &[f32]) -> f32 {
        if a.len() != b.len() {
            return f32::MAX; // 出错时返回最大距离
        }
        
        // 使用VectorOps的曼哈顿距离，注意需要取负值转换为正距离
        -crate::vector::core::operations::VectorOps::manhattan_distance(a, b)
    }
    
    fn as_dyn_distance(self: Box<Self>) -> Box<dyn Distance + Send + Sync> {
        self
    }
}

/// Jaccard距离
#[derive(Clone, Copy, Debug)]
pub struct JaccardDistance;

impl Distance for JaccardDistance {
    fn calculate(&self, a: &[f32], b: &[f32]) -> f32 {
        if a.len() != b.len() {
            return f32::MAX; // 出错时返回最大距离
        }
        
        // Jaccard相似度转换为距离（1-相似度）
        1.0 - crate::vector::core::operations::VectorOps::jaccard_similarity(a, b)
    }
    
    fn as_dyn_distance(self: Box<Self>) -> Box<dyn Distance + Send + Sync> {
        self
    }
}

/// 计算两个向量之间的距离
pub fn compute_distance(a: &[f32], b: &[f32], distance_type: &Box<dyn Distance + Send + Sync>) -> f32 {
    distance_type.calculate(a, b)
}

/// 计算两个向量之间的原始距离，不经过任何转换
pub fn compute_distance_raw(a: &[f32], b: &[f32], distance_type: &Box<dyn Distance + Send + Sync>) -> f32 {
    distance_type.calculate(a, b)
} 