//! HNSW索引的距离计算函数
//!
//! 本模块实现了向量之间的各种距离计算方法，包括欧氏距离、余弦距离、
//! 点积距离和曼哈顿距离等。

use crate::vector::index::hnsw::types::DistanceType;
// 使用标准库，移除SIMD特性
// use std::simd::prelude::*;

/// 计算两个向量之间的距离
pub fn calculate_distance(v1: &[f32], v2: &[f32], distance_type: DistanceType) -> f32 {
    match distance_type {
        DistanceType::Euclidean => euclidean_distance(v1, v2),
        DistanceType::Cosine => cosine_distance(v1, v2),
        DistanceType::DotProduct => dot_product_distance(v1, v2),
        DistanceType::Manhattan => manhattan_distance(v1, v2),
    }
}

/// 计算欧氏距离(常规实现)
fn euclidean_distance_naive(v1: &[f32], v2: &[f32]) -> f32 {
    debug_assert_eq!(v1.len(), v2.len(), "向量维度不匹配");
    
    let mut sum = 0.0;
    for i in 0..v1.len() {
        let diff = v1[i] - v2[i];
        sum += diff * diff;
    }
    sum.sqrt()
}

/// 计算欧氏距离 (优化版本)
/// 
/// 实现了分块计算以提高性能，但不使用不稳定的SIMD特性
pub fn euclidean_distance(v1: &[f32], v2: &[f32]) -> f32 {
    debug_assert_eq!(v1.len(), v2.len(), "向量维度不匹配");
    
    // 如果向量太短，直接使用标准方法
    if v1.len() < 8 {
        return euclidean_distance_naive(v1, v2);
    }
    
    let len = v1.len();
    let mut sum = 0.0;
    
    // 以4个元素为一组进行分块处理，减少循环开销
    let chunks = len / 4;
    for c in 0..chunks {
        let i = c * 4;
        let mut block_sum = 0.0;
        
        // 处理4个元素的块
        for j in 0..4 {
            let idx = i + j;
            let diff = v1[idx] - v2[idx];
            block_sum += diff * diff;
        }
        
        sum += block_sum;
    }
    
    // 处理剩余元素
    for i in (chunks * 4)..len {
        let diff = v1[i] - v2[i];
        sum += diff * diff;
    }
    
    sum.sqrt()
}

/// 计算余弦距离 (1 - 余弦相似度)
pub fn cosine_distance(v1: &[f32], v2: &[f32]) -> f32 {
    debug_assert_eq!(v1.len(), v2.len(), "向量维度不匹配");
    
    let mut dot_product = 0.0;
    let mut norm1 = 0.0;
    let mut norm2 = 0.0;
    
    for i in 0..v1.len() {
        dot_product += v1[i] * v2[i];
        norm1 += v1[i] * v1[i];
        norm2 += v2[i] * v2[i];
    }
    
    // 避免除以零
    if norm1 == 0.0 || norm2 == 0.0 {
        return 1.0; // 最大距离
    }
    
    let cosine_similarity = dot_product / (norm1.sqrt() * norm2.sqrt());
    // 确保余弦相似度在[-1,1]范围内
    let cosine_similarity = cosine_similarity.max(-1.0).min(1.0);
    
    // 余弦距离 = 1 - 余弦相似度
    1.0 - cosine_similarity
}

/// 计算点积距离 (1 - 归一化点积)
pub fn dot_product_distance(v1: &[f32], v2: &[f32]) -> f32 {
    debug_assert_eq!(v1.len(), v2.len(), "向量维度不匹配");
    
    let mut dot_product = 0.0;
    
    for i in 0..v1.len() {
        dot_product += v1[i] * v2[i];
    }
    
    // 归一化到[0,1]范围的距离
    // 这里假设点积值域为[-1,1]，转换为距离[0,2]，再归一化到[0,1]
    (1.0 - dot_product) / 2.0
}

/// 计算曼哈顿距离(L1距离)
pub fn manhattan_distance(v1: &[f32], v2: &[f32]) -> f32 {
    debug_assert_eq!(v1.len(), v2.len(), "向量维度不匹配");
    
    let mut sum = 0.0;
    for i in 0..v1.len() {
        sum += (v1[i] - v2[i]).abs();
    }
    sum
}

/// 计算向量的L2范数(欧几里得范数)
pub fn l2_norm(v: &[f32]) -> f32 {
    let mut sum = 0.0;
    for val in v {
        sum += val * val;
    }
    sum.sqrt()
}

/// 对向量进行L2归一化
pub fn normalize_vector(v: &mut [f32]) {
    let norm = l2_norm(v);
    
    // 避免除以零
    if norm == 0.0 {
        return;
    }
    
    for val in v {
        *val /= norm;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f32::EPSILON;
    
    #[test]
    fn test_euclidean_distance() {
        let v1 = vec![1.0, 2.0, 3.0];
        let v2 = vec![4.0, 5.0, 6.0];
        
        // 手动计算: sqrt((4-1)^2 + (5-2)^2 + (6-3)^2) = sqrt(9 + 9 + 9) = sqrt(27) = 5.196
        let expected = 5.196152;
        let result = euclidean_distance(&v1, &v2);
        
        assert!((result - expected).abs() < 0.0001);
    }
    
    #[test]
    fn test_cosine_distance() {
        let v1 = vec![1.0, 2.0, 3.0];
        let v2 = vec![4.0, 5.0, 6.0];
        
        // 计算余弦相似度: (1*4 + 2*5 + 3*6) / (sqrt(1^2+2^2+3^2) * sqrt(4^2+5^2+6^2))
        // = 32 / (sqrt(14) * sqrt(77)) = 32 / 32.83 = 0.9747
        // 余弦距离 = 1 - 0.9747 = 0.0253
        let expected = 0.0253;
        let result = cosine_distance(&v1, &v2);
        
        assert!((result - expected).abs() < 0.0001);
    }
    
    #[test]
    fn test_dot_product_distance() {
        let v1 = vec![1.0, 0.0, 0.0];
        let v2 = vec![0.0, 1.0, 0.0];
        
        // 正交向量，点积为0，距离为0.5
        let result = dot_product_distance(&v1, &v2);
        assert!((result - 0.5).abs() < EPSILON);
        
        // 相同向量，点积为1，距离为0
        let result = dot_product_distance(&v1, &v1);
        assert!((result - 0.0).abs() < EPSILON);
        
        // 相反向量，点积为-1，距离为1
        let v3 = vec![-1.0, 0.0, 0.0];
        let result = dot_product_distance(&v1, &v3);
        assert!((result - 1.0).abs() < EPSILON);
    }
    
    #[test]
    fn test_manhattan_distance() {
        let v1 = vec![1.0, 2.0, 3.0];
        let v2 = vec![4.0, 5.0, 6.0];
        
        // |4-1| + |5-2| + |6-3| = 3 + 3 + 3 = 9
        let expected = 9.0;
        let result = manhattan_distance(&v1, &v2);
        
        assert_eq!(result, expected);
    }
    
    #[test]
    fn test_normalization() {
        let mut v = vec![3.0, 4.0, 0.0];
        let expected_norm_before = 5.0;
        
        // 检查初始范数
        assert!((l2_norm(&v) - expected_norm_before).abs() < EPSILON);
        
        normalize_vector(&mut v);
        
        // 归一化后的向量范数应为1
        assert!((l2_norm(&v) - 1.0).abs() < EPSILON);
        
        // 检查归一化后的值
        assert!((v[0] - 0.6).abs() < EPSILON);
        assert!((v[1] - 0.8).abs() < EPSILON);
        assert!((v[2] - 0.0).abs() < EPSILON);
    }
} 