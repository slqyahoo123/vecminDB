use ndarray::ArrayView1;
use crate::{Error, Result};
use std::cmp::Ordering;
use serde::{Serialize, Deserialize};
use crate::vector::types::Vector;

/// 相似度计算方法枚举
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum SimilarityMetric {
    Cosine,
    Euclidean,
    DotProduct,
    Manhattan,
    Jaccard,
}

impl Default for SimilarityMetric {
    fn default() -> Self {
        SimilarityMetric::Cosine
    }
}

/// 向量操作工具
pub struct VectorOps;

impl VectorOps {
    /// 计算两个向量之间的相似度
    pub fn compute_similarity(
        v1: &[f32],
        v2: &[f32],
        metric: SimilarityMetric,
    ) -> f32 {
        match metric {
            SimilarityMetric::Cosine => Self::cosine_similarity(v1, v2),
            SimilarityMetric::Euclidean => Self::euclidean_distance(v1, v2),
            SimilarityMetric::Manhattan => Self::manhattan_distance(v1, v2),
            SimilarityMetric::DotProduct => Self::dot_product(v1, v2),
            SimilarityMetric::Jaccard => Self::jaccard_similarity(v1, v2),
        }
    }

    /// 计算余弦相似度
    pub fn cosine_similarity(v1: &[f32], v2: &[f32]) -> f32 {
        let v1_view = ArrayView1::from(v1);
        let v2_view = ArrayView1::from(v2);
        
        let dot_product = v1_view.dot(&v2_view);
        let norm1 = (v1_view.dot(&v1_view)).sqrt();
        let norm2 = (v2_view.dot(&v2_view)).sqrt();
        
        if norm1 == 0.0 || norm2 == 0.0 {
            return 0.0;
        }
        
        dot_product / (norm1 * norm2)
    }

    /// 计算欧氏距离
    pub fn euclidean_distance(v1: &[f32], v2: &[f32]) -> f32 {
        let v1_view = ArrayView1::from(v1);
        let v2_view = ArrayView1::from(v2);
        
        let diff = &v1_view - &v2_view;
        let squared_diff = diff.mapv(|x| x * x);
        let sum_squared_diff = squared_diff.sum();
        
        // 返回距离的负值，使得较大的值表示更相似
        -sum_squared_diff.sqrt()
    }

    /// 计算点积
    pub fn dot_product(v1: &[f32], v2: &[f32]) -> f32 {
        let v1_view = ArrayView1::from(v1);
        let v2_view = ArrayView1::from(v2);
        
        v1_view.dot(&v2_view)
    }

    /// 计算曼哈顿距离
    pub fn manhattan_distance(v1: &[f32], v2: &[f32]) -> f32 {
        let v1_view = ArrayView1::from(v1);
        let v2_view = ArrayView1::from(v2);
        
        let diff = &v1_view - &v2_view;
        let abs_diff = diff.mapv(|x| x.abs());
        let sum_abs_diff = abs_diff.sum();
        
        // 返回距离的负值，使得较大的值表示更相似
        -sum_abs_diff
    }

    /// 计算Jaccard相似度
    pub fn jaccard_similarity(v1: &[f32], v2: &[f32]) -> f32 {
        let threshold = 0.5; // 二值化阈值
        
        let mut intersection = 0;
        let mut union = 0;
        
        for i in 0..v1.len() {
            let b1 = v1[i] > threshold;
            let b2 = v2[i] > threshold;
            
            if b1 && b2 {
                intersection += 1;
            }
            
            if b1 || b2 {
                union += 1;
            }
        }
        
        if union == 0 {
            return 0.0;
        }
        
        intersection as f32 / union as f32
    }

    /// 向量归一化
    pub fn normalize(vector: &mut [f32]) -> Result<()> {
        let v_view = ArrayView1::from(&*vector);
        let norm = (v_view.dot(&v_view)).sqrt();
        
        if norm == 0.0 {
            return Err(Error::vector("Cannot normalize zero vector".to_string()));
        }
        
        for val in vector.iter_mut() {
            *val /= norm;
        }
        
        Ok(())
    }

    /// 向量比较结果，用于排序
    pub fn compare_vectors(
        query: &[f32], 
        v1: &Vector, 
        v2: &Vector, 
        metric: SimilarityMetric
    ) -> Result<Ordering> {
        // 与查询向量比较，较高的相似度优先
        let sim1 = Self::compute_similarity(query, &v1.data, metric);
        let sim2 = Self::compute_similarity(query, &v2.data, metric);
        
        sim1.partial_cmp(&sim2).ok_or_else(|| 
            Error::vector("Failed to compare vectors".to_string())
        )
    }

    /// 向量对象之间的相似度计算
    pub fn compute_vector_similarity(
        v1: &crate::vector::types::Vector,
        v2: &crate::vector::types::Vector,
        metric: SimilarityMetric,
    ) -> f32 {
        Self::compute_similarity(&v1.data, &v2.data, metric)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cosine_similarity() {
        let v1 = vec![1.0, 2.0, 3.0];
        let v2 = vec![4.0, 5.0, 6.0];
        
        let sim = VectorOps::cosine_similarity(&v1, &v2);
        assert!((sim - 0.9746318).abs() < 1e-6);
    }

    #[test]
    fn test_normalize() {
        let mut v = vec![3.0, 4.0];
        VectorOps::normalize(&mut v).unwrap();
        
        assert!((v[0] - 0.6).abs() < 1e-6);
        assert!((v[1] - 0.8).abs() < 1e-6);
    }
} 