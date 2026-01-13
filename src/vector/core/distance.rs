use crate::error::{Error, Result};
use std::fmt::Debug;
use serde::{Serialize, Deserialize};

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
        
        let mut sum = 0.0;
        for i in 0..a.len() {
            let diff = a[i] - b[i];
            sum += diff * diff;
        }
        
        sum.sqrt()
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
        
        let mut dot = 0.0;
        let mut a_norm = 0.0;
        let mut b_norm = 0.0;
        
        for i in 0..a.len() {
            dot += a[i] * b[i];
            a_norm += a[i] * a[i];
            b_norm += b[i] * b[i];
        }
        
        a_norm = a_norm.sqrt();
        b_norm = b_norm.sqrt();
        
        if a_norm == 0.0 || b_norm == 0.0 {
            return f32::MAX;
        }
        
        1.0 - (dot / (a_norm * b_norm))
    }
    
    fn as_dyn_distance(self: Box<Self>) -> Box<dyn Distance + Send + Sync> {
        self
    }
}

/// 点积距离（实际上是相似度，但为了一致性，我们将其转换为距离）
#[derive(Clone, Copy, Debug)]
pub struct DotProductDistance;

impl Distance for DotProductDistance {
    fn calculate(&self, a: &[f32], b: &[f32]) -> f32 {
        if a.len() != b.len() {
            return f32::MAX; // 出错时返回最大距离
        }
        
        let mut dot = 0.0;
        for i in 0..a.len() {
            dot += a[i] * b[i];
        }
        
        // 将点积转换为距离（越大越相似，所以我们用负值）
        -dot
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
        
        let mut sum = 0.0;
        for i in 0..a.len() {
            sum += (a[i] - b[i]).abs();
        }
        
        sum
    }
    
    fn as_dyn_distance(self: Box<Self>) -> Box<dyn Distance + Send + Sync> {
        self
    }
}

/// 切比雪夫距离
#[derive(Clone, Copy, Debug)]
pub struct ChebyshevDistance;

impl Distance for ChebyshevDistance {
    fn calculate(&self, a: &[f32], b: &[f32]) -> f32 {
        if a.len() != b.len() {
            return f32::MAX; // 出错时返回最大距离
        }
        
        let mut max_diff = 0.0;
        for i in 0..a.len() {
            let diff = (a[i] - b[i]).abs();
            if diff > max_diff {
                max_diff = diff;
            }
        }
        
        max_diff
    }
    
    fn as_dyn_distance(self: Box<Self>) -> Box<dyn Distance + Send + Sync> {
        self
    }
}

/// 闵可夫斯基距离
#[derive(Clone, Copy, Debug)]
pub struct MinkowskiDistance {
    p: f32,
}

impl MinkowskiDistance {
    pub fn new(p: f32) -> Self {
        Self { p }
    }
}

impl Distance for MinkowskiDistance {
    fn calculate(&self, a: &[f32], b: &[f32]) -> f32 {
        if a.len() != b.len() {
            return f32::MAX; // 出错时返回最大距离
        }
        
        let mut sum = 0.0;
        for i in 0..a.len() {
            let diff = (a[i] - b[i]).abs();
            sum += diff.powf(self.p);
        }
        
        sum.powf(1.0 / self.p)
    }
    
    fn as_dyn_distance(self: Box<Self>) -> Box<dyn Distance + Send + Sync> {
        self
    }
}

/// 距离度量类型枚举
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum DistanceMetric {
    Euclidean,
    Cosine,
    DotProduct,
    Manhattan,
}

impl Default for DistanceMetric {
    fn default() -> Self {
        DistanceMetric::Euclidean
    }
}

/// 创建距离函数
pub fn create_distance(name: &str) -> Result<Box<dyn Distance + Send + Sync>> {
    match name {
        "euclidean" => Ok(Box::new(EuclideanDistance)),
        "cosine" => Ok(Box::new(CosineDistance)),
        "dot" => Ok(Box::new(DotProductDistance)),
        "manhattan" => Ok(Box::new(ManhattanDistance)),
        "chebyshev" => Ok(Box::new(ChebyshevDistance)),
        _ => Err(Error::InvalidArgument(format!("Unsupported distance function: {}", name))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_euclidean_distance() {
        let dist = EuclideanDistance;
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![4.0, 5.0, 6.0];
        
        let result = dist.calculate(&a, &b);
        let expected = 5.196152; // sqrt(27)
        
        assert!((result - expected).abs() < 1e-6);
    }
    
    #[test]
    fn test_cosine_distance() {
        let dist = CosineDistance;
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![4.0, 5.0, 6.0];
        
        let result = dist.calculate(&a, &b);
        let expected = 1.0 - 0.9746318; // 1 - cos(angle)
        
        assert!((result - expected).abs() < 1e-6);
    }
    
    #[test]
    fn test_dot_product_distance() {
        let dist = DotProductDistance;
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![4.0, 5.0, 6.0];
        
        let result = dist.calculate(&a, &b);
        let expected = -32.0; // -(1*4 + 2*5 + 3*6)
        
        assert!((result - expected).abs() < 1e-6);
    }
    
    #[test]
    fn test_manhattan_distance() {
        let dist = ManhattanDistance;
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![4.0, 5.0, 6.0];
        
        let result = dist.calculate(&a, &b);
        let expected = 9.0; // |1-4| + |2-5| + |3-6|
        
        assert!(result == expected);
    }
    
    #[test]
    fn test_chebyshev_distance() {
        let dist = ChebyshevDistance;
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![4.0, 5.0, 6.0];
        
        let result = dist.calculate(&a, &b);
        let expected = 3.0; // max(|1-4|, |2-5|, |3-6|)
        
        assert!((result - expected).abs() < 1e-6);
    }
    
    #[test]
    fn test_batch_calculate() {
        let dist = EuclideanDistance;
        let query = vec![1.0, 2.0, 3.0];
        let vectors = vec![
            vec![4.0, 5.0, 6.0],
            vec![7.0, 8.0, 9.0],
        ];
        
        let results = dist.batch_calculate(&query, &vectors).unwrap();
        
        assert_eq!(results.len(), 2);
        assert!((results[0] - 5.196152).abs() < 1e-6);
        assert!((results[1] - 10.392304).abs() < 1e-6);
    }
} 