// vector/ops.rs - 向量操作模块

use crate::Result;

/// 向量操作特性
pub trait VectorOps {
    /// 计算向量的欧几里得距离
    fn euclidean_distance(&self, other: &[f32]) -> Result<f32>;
    
    /// 计算向量的曼哈顿距离
    fn manhattan_distance(&self, other: &[f32]) -> Result<f32>;
    
    /// 计算向量的余弦相似度
    fn cosine_similarity(&self, other: &[f32]) -> Result<f32>;
    
    /// 计算向量的点积
    fn dot_product(&self, other: &[f32]) -> Result<f32>;
    
    /// 归一化向量
    fn normalize(&mut self) -> Result<()>;
    
    /// 缩放向量
    fn scale(&mut self, factor: f32) -> Result<()>;
}

/// 为向量类型实现向量操作特性
impl VectorOps for Vec<f32> {
    fn euclidean_distance(&self, other: &[f32]) -> Result<f32> {
        if self.len() != other.len() {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                format!("向量维度不匹配: {} vs {}", self.len(), other.len())
            ).into());
        }
        
        let sum_squared: f32 = self.iter()
            .zip(other.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum();
            
        Ok(sum_squared.sqrt())
    }
    
    fn manhattan_distance(&self, other: &[f32]) -> Result<f32> {
        if self.len() != other.len() {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                format!("向量维度不匹配: {} vs {}", self.len(), other.len())
            ).into());
        }
        
        let sum: f32 = self.iter()
            .zip(other.iter())
            .map(|(a, b)| (a - b).abs())
            .sum();
            
        Ok(sum)
    }
    
    fn cosine_similarity(&self, other: &[f32]) -> Result<f32> {
        if self.len() != other.len() {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                format!("向量维度不匹配: {} vs {}", self.len(), other.len())
            ).into());
        }
        
        let dot_product: f32 = self.iter()
            .zip(other.iter())
            .map(|(a, b)| a * b)
            .sum();
            
        let self_magnitude: f32 = self.iter()
            .map(|a| a.powi(2))
            .sum::<f32>()
            .sqrt();
            
        let other_magnitude: f32 = other.iter()
            .map(|b| b.powi(2))
            .sum::<f32>()
            .sqrt();
            
        if self_magnitude.abs() < f32::EPSILON || other_magnitude.abs() < f32::EPSILON {
            return Ok(0.0);
        }
        
        Ok(dot_product / (self_magnitude * other_magnitude))
    }
    
    fn dot_product(&self, other: &[f32]) -> Result<f32> {
        if self.len() != other.len() {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                format!("向量维度不匹配: {} vs {}", self.len(), other.len())
            ).into());
        }
        
        let dot: f32 = self.iter()
            .zip(other.iter())
            .map(|(a, b)| a * b)
            .sum();
            
        Ok(dot)
    }
    
    fn normalize(&mut self) -> Result<()> {
        let magnitude: f32 = self.iter()
            .map(|a| a.powi(2))
            .sum::<f32>()
            .sqrt();
            
        if magnitude.abs() < f32::EPSILON {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "零向量无法归一化"
            ).into());
        }
        
        for element in self.iter_mut() {
            *element /= magnitude;
        }
        
        Ok(())
    }
    
    fn scale(&mut self, factor: f32) -> Result<()> {
        for element in self.iter_mut() {
            *element *= factor;
        }
        
        Ok(())
    }
}

/// 向量距离计算函数
pub struct EuclideanDistance;
impl EuclideanDistance {
    pub fn calculate(a: &[f32], b: &[f32]) -> Result<f32> {
        let vec_a = a.to_vec();
        vec_a.euclidean_distance(b)
    }
}

/// 向量距离计算函数
pub struct ManhattanDistance;
impl ManhattanDistance {
    pub fn calculate(a: &[f32], b: &[f32]) -> Result<f32> {
        let vec_a = a.to_vec();
        vec_a.manhattan_distance(b)
    }
}

/// 余弦相似度计算函数
pub struct CosineSimilarity;
impl CosineSimilarity {
    pub fn calculate(a: &[f32], b: &[f32]) -> Result<f32> {
        let vec_a = a.to_vec();
        vec_a.cosine_similarity(b)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_euclidean_distance() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![4.0, 5.0, 6.0];
        
        let distance = a.euclidean_distance(&b).unwrap();
        assert!((distance - 5.196152).abs() < 0.00001);
    }
    
    #[test]
    fn test_manhattan_distance() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![4.0, 5.0, 6.0];
        
        let distance = a.manhattan_distance(&b).unwrap();
        assert_eq!(distance, 9.0);
    }
    
    #[test]
    fn test_cosine_similarity() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![4.0, 5.0, 6.0];
        
        let similarity = a.cosine_similarity(&b).unwrap();
        assert!((similarity - 0.974631).abs() < 0.00001);
    }
    
    #[test]
    fn test_normalize() {
        let mut a = vec![3.0, 4.0];
        a.normalize().unwrap();
        
        assert!((a[0] - 0.6).abs() < 0.00001);
        assert!((a[1] - 0.8).abs() < 0.00001);
    }
} 