// Feature Vector Module
// 特征向量模块

use std::collections::HashMap;
use std::ops::{Add, Sub, Mul, Div};
use std::fmt;
use std::fmt::Debug;
use std::hash::{Hash, Hasher};
use std::sync::Arc;

use serde::{Serialize, Deserialize};
use thiserror::Error;
use ndarray::{Array1, Array2};

use crate::error::Result;

/// 向量操作错误
#[derive(Error, Debug)]
pub enum VectorError {
    #[error("维度不匹配：预期 {expected}，实际 {actual}")]
    DimensionMismatch { expected: usize, actual: usize },
    
    #[error("空向量")]
    EmptyVector,
    
    #[error("索引越界：索引 {index}，大小 {size}")]
    IndexOutOfBounds { index: usize, size: usize },
    
    #[error("不支持的操作：{0}")]
    UnsupportedOperation(String),
    
    #[error("数值错误：{0}")]
    NumericError(String),
}

/// 特征向量
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Vector {
    /// 向量的唯一标识符
    pub id: String,
    
    /// 向量的维度
    pub dimension: usize,
    
    /// 向量的值
    pub values: Vec<f32>,
    
    /// 向量的元数据
    pub metadata: HashMap<String, String>,
}

impl PartialEq for Vector {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id
    }
}

impl Eq for Vector {}

impl Hash for Vector {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.id.hash(state);
    }
}

impl Vector {
    /// 创建新的特征向量
    pub fn new(id: impl Into<String>, values: Vec<f32>) -> Self {
        let id = id.into();
        let dimension = values.len();
        
        Self {
            id,
            dimension,
            values,
            metadata: HashMap::new(),
        }
    }
    
    /// 添加元数据
    pub fn with_metadata(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.metadata.insert(key.into(), value.into());
        self
    }
    
    /// 获取元数据值
    pub fn get_metadata(&self, key: &str) -> Option<&String> {
        self.metadata.get(key)
    }
    
    /// 设置元数据值
    pub fn set_metadata(&mut self, key: impl Into<String>, value: impl Into<String>) {
        self.metadata.insert(key.into(), value.into());
    }
    
    /// 计算两个向量之间的欧几里得距离
    pub fn euclidean_distance(&self, other: &Self) -> f32 {
        if self.dimension != other.dimension {
            return f32::NAN;
        }
        
        self.values.iter()
            .zip(other.values.iter())
            .map(|(a, b)| (a - b) * (a - b))
            .sum::<f32>()
            .sqrt()
    }
    
    /// 计算两个向量之间的余弦相似度
    pub fn cosine_similarity(&self, other: &Self) -> f32 {
        if self.dimension != other.dimension {
            return f32::NAN;
        }
        
        let dot_product = self.values.iter()
            .zip(other.values.iter())
            .map(|(a, b)| a * b)
            .sum::<f32>();
            
        let norm_a = self.values.iter()
            .map(|a| a * a)
            .sum::<f32>()
            .sqrt();
            
        let norm_b = other.values.iter()
            .map(|b| b * b)
            .sum::<f32>()
            .sqrt();
            
        if norm_a == 0.0 || norm_b == 0.0 {
            return 0.0;
        }
        
        dot_product / (norm_a * norm_b)
    }
    
    /// 计算两个向量之间的点积
    pub fn dot_product(&self, other: &Self) -> f32 {
        if self.dimension != other.dimension {
            return f32::NAN;
        }
        
        self.values.iter()
            .zip(other.values.iter())
            .map(|(a, b)| a * b)
            .sum()
    }
    
    /// 计算向量的L2范数（欧几里得范数）
    pub fn l2_norm(&self) -> f32 {
        self.values.iter()
            .map(|a| a * a)
            .sum::<f32>()
            .sqrt()
    }
    
    /// 计算向量的L1范数（曼哈顿范数）
    pub fn l1_norm(&self) -> f32 {
        self.values.iter()
            .map(|a| a.abs())
            .sum()
    }
    
    /// 检查向量是否为零向量
    pub fn is_zero(&self) -> bool {
        self.values.iter().all(|&x| x == 0.0)
    }
    
    /// 检查向量是否包含NaN值
    pub fn has_nan(&self) -> bool {
        self.values.iter().any(|x| x.is_nan())
    }
    
    /// 检查向量是否包含无穷值
    pub fn has_infinite(&self) -> bool {
        self.values.iter().any(|x| x.is_infinite())
    }
    
    /// 检查向量是否有效（不包含NaN或无穷值）
    pub fn is_valid(&self) -> bool {
        !self.has_nan() && !self.has_infinite()
    }
}

impl fmt::Display for Vector {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Vector(id={}, dim={}, metadata={})", 
            self.id, 
            self.dimension,
            self.metadata.len()
        )
    }
}

/// 向量批次
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VectorBatch {
    /// 批次ID
    pub batch_id: String,
    
    /// 批次中的向量
    pub vectors: Vec<Vector>,
    
    /// 批次元数据
    pub metadata: HashMap<String, String>,
}

impl VectorBatch {
    /// 创建新的向量批次
    pub fn new(batch_id: impl Into<String>, vectors: Vec<Vector>) -> Self {
        Self {
            batch_id: batch_id.into(),
            vectors,
            metadata: HashMap::new(),
        }
    }
    
    /// 添加元数据
    pub fn with_metadata(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.metadata.insert(key.into(), value.into());
        self
    }
    
    /// 获取元数据值
    pub fn get_metadata(&self, key: &str) -> Option<&String> {
        self.metadata.get(key)
    }
    
    /// 设置元数据值
    pub fn set_metadata(&mut self, key: impl Into<String>, value: impl Into<String>) {
        self.metadata.insert(key.into(), value.into());
    }
    
    /// 添加向量到批次
    pub fn add_vector(&mut self, vector: Vector) {
        self.vectors.push(vector);
    }
    
    /// 获取批次大小
    pub fn size(&self) -> usize {
        self.vectors.len()
    }
    
    /// 检查批次是否为空
    pub fn is_empty(&self) -> bool {
        self.vectors.is_empty()
    }
    
    /// 获取具有相同维度的向量
    pub fn get_vectors_with_dimension(&self, dimension: usize) -> Vec<&Vector> {
        self.vectors.iter()
            .filter(|v| v.dimension == dimension)
            .collect()
    }
    
    /// 按ID获取向量
    pub fn get_vector_by_id(&self, id: &str) -> Option<&Vector> {
        self.vectors.iter().find(|v| v.id == id)
    }
    
    /// 计算批次中所有向量的平均值
    pub fn mean(&self) -> Option<Vector> {
        if self.is_empty() {
            return None;
        }
        
        // 确保所有向量具有相同的维度
        let first_dim = self.vectors.first()?.dimension;
        if !self.vectors.iter().all(|v| v.dimension == first_dim) {
            return None;
        }
        
        let mut mean_values = vec![0.0; first_dim];
        let count = self.vectors.len() as f32;
        
        for vector in &self.vectors {
            for (i, &value) in vector.values.iter().enumerate() {
                mean_values[i] += value / count;
            }
        }
        
        Some(Vector::new(
            format!("{}_mean", self.batch_id),
            mean_values
        ))
    }
    
    /// 将批次分割为多个子批次
    pub fn split(&self, batch_size: usize) -> Vec<VectorBatch> {
        if batch_size == 0 || self.is_empty() {
            return vec![];
        }
        
        self.vectors.chunks(batch_size)
            .enumerate()
            .map(|(i, chunk)| {
                VectorBatch::new(
                    format!("{}_{}", self.batch_id, i),
                    chunk.to_vec()
                )
            })
            .collect()
    }
    
    /// 合并多个批次
    pub fn merge(batches: &[VectorBatch]) -> Option<VectorBatch> {
        if batches.is_empty() {
            return None;
        }
        
        let mut all_vectors = Vec::new();
        let batch_ids = batches.iter()
            .map(|b| b.batch_id.clone())
            .collect::<Vec<_>>()
            .join("_");
            
        for batch in batches {
            all_vectors.extend(batch.vectors.clone());
        }
        
        Some(VectorBatch::new(
            format!("merged_{}", batch_ids),
            all_vectors
        ))
    }
}

impl fmt::Display for VectorBatch {
    fn fmt(&self, f: &mut fmt::Display<'_>) -> fmt::Result {
        write!(f, "VectorBatch(id={}, vectors={}, metadata={})", 
            self.batch_id, 
            self.vectors.len(),
            self.metadata.len()
        )
    }
}

/// 向量存储接口
pub trait VectorStorage: Send + Sync + Debug {
    /// 保存向量
    fn save(&self, vector: &Vector) -> Result<()>;
    
    /// 批量保存向量
    fn save_batch(&self, batch: &VectorBatch) -> Result<()>;
    
    /// 根据ID获取向量
    fn get(&self, id: &str) -> Result<Option<Vector>>;
    
    /// 根据ID列表批量获取向量
    fn get_batch(&self, ids: &[String]) -> Result<Vec<Vector>>;
    
    /// 删除向量
    fn delete(&self, id: &str) -> Result<bool>;
    
    /// 批量删除向量
    fn delete_batch(&self, ids: &[String]) -> Result<usize>;
    
    /// 获取向量数量
    fn count(&self) -> Result<usize>;
    
    /// 获取所有向量
    fn get_all(&self) -> Result<Vec<Vector>>;
    
    /// 根据标签获取向量
    fn get_by_label(&self, label: &str) -> Result<Vec<Vector>>;
    
    /// 根据元数据查询向量
    fn query_by_metadata(&self, key: &str, value: &str) -> Result<Vec<Vector>>;
}

/// 内存向量存储
#[derive(Debug, Default)]
pub struct InMemoryVectorStorage {
    /// 向量映射表
    vectors: std::sync::RwLock<HashMap<String, Vector>>,
}

impl InMemoryVectorStorage {
    /// 创建新的内存向量存储
    pub fn new() -> Self {
        Self {
            vectors: std::sync::RwLock::new(HashMap::new()),
        }
    }
    
    /// 获取存储容量
    pub fn capacity(&self) -> usize {
        self.vectors.read().unwrap().capacity()
    }
    
    /// 清空存储
    pub fn clear(&self) -> Result<()> {
        self.vectors.write().unwrap().clear();
        Ok(())
    }
}

impl VectorStorage for InMemoryVectorStorage {
    fn save(&self, vector: &Vector) -> Result<()> {
        self.vectors.write().unwrap().insert(vector.id.clone(), vector.clone());
        Ok(())
    }
    
    fn save_batch(&self, batch: &VectorBatch) -> Result<()> {
        let mut vectors = self.vectors.write().unwrap();
        for vector in &batch.vectors {
            vectors.insert(vector.id.clone(), vector.clone());
        }
        Ok(())
    }
    
    fn get(&self, id: &str) -> Result<Option<Vector>> {
        let vectors = self.vectors.read().unwrap();
        Ok(vectors.get(id).cloned())
    }
    
    fn get_batch(&self, ids: &[String]) -> Result<Vec<Vector>> {
        let vectors = self.vectors.read().unwrap();
        let result = ids.iter()
            .filter_map(|id| vectors.get(id).cloned())
            .collect();
        Ok(result)
    }
    
    fn delete(&self, id: &str) -> Result<bool> {
        Ok(self.vectors.write().unwrap().remove(id).is_some())
    }
    
    fn delete_batch(&self, ids: &[String]) -> Result<usize> {
        let mut vectors = self.vectors.write().unwrap();
        let mut count = 0;
        for id in ids {
            if vectors.remove(id).is_some() {
                count += 1;
            }
        }
        Ok(count)
    }
    
    fn count(&self) -> Result<usize> {
        Ok(self.vectors.read().unwrap().len())
    }
    
    fn get_all(&self) -> Result<Vec<Vector>> {
        let vectors = self.vectors.read().unwrap();
        Ok(vectors.values().cloned().collect())
    }
    
    fn get_by_label(&self, label: &str) -> Result<Vec<Vector>> {
        let vectors = self.vectors.read().unwrap();
        let result = vectors.values()
            .filter(|v| v.metadata.get("label").map_or(false, |v| v == label))
            .cloned()
            .collect();
        Ok(result)
    }
    
    fn query_by_metadata(&self, key: &str, value: &str) -> Result<Vec<Vector>> {
        let vectors = self.vectors.read().unwrap();
        let result = vectors.values()
            .filter(|v| v.metadata.get(key).map_or(false, |v| v == value))
            .cloned()
            .collect();
        Ok(result)
    }
}

/// 向量距离计算器
#[derive(Debug, Clone, Copy)]
pub enum DistanceMetric {
    /// 欧几里得距离
    Euclidean,
    /// 余弦相似度
    Cosine,
    /// 曼哈顿距离
    Manhattan,
    /// 点积
    DotProduct,
}

impl DistanceMetric {
    /// 计算两个向量间的距离
    pub fn calculate(&self, a: &Vector, b: &Vector) -> Result<f32> {
        match self {
            DistanceMetric::Euclidean => {
                if a.dimension != b.dimension {
                    return Err(format!(
                        "向量维度不匹配: {} vs {}", 
                        a.dimension, 
                        b.dimension
                    ).into());
                }
                
                let sum = a.values.iter()
                    .zip(b.values.iter())
                    .map(|(&x, &y)| (x - y) * (x - y))
                    .sum::<f32>();
                    
                Ok(sum.sqrt())
            }
            DistanceMetric::Cosine => {
                let dot = a.values.iter()
                    .zip(b.values.iter())
                    .map(|(&x, &y)| x * y)
                    .sum::<f32>();
                    
                let self_norm = a.values.iter()
                    .map(|&x| x * x)
                    .sum::<f32>()
                    .sqrt();
                    
                let other_norm = b.values.iter()
                    .map(|&x| x * x)
                    .sum::<f32>()
                    .sqrt();
                    
                if self_norm == 0.0 || other_norm == 0.0 {
                    return Ok(1.0); // 1.0表示最大距离
                }
                
                // 将余弦相似度转换为距离
                Ok(1.0 - (dot / (self_norm * other_norm)))
            }
            DistanceMetric::Manhattan => {
                if a.dimension != b.dimension {
                    return Err(format!(
                        "向量维度不匹配: {} vs {}", 
                        a.dimension, 
                        b.dimension
                    ).into());
                }
                
                let sum = a.values.iter()
                    .zip(b.values.iter())
                    .map(|(&x, &y)| (x - y).abs())
                    .sum::<f32>();
                    
                Ok(sum)
            }
            DistanceMetric::DotProduct => {
                if a.dimension != b.dimension {
                    return Err(format!(
                        "向量维度不匹配: {} vs {}", 
                        a.dimension, 
                        b.dimension
                    ).into());
                }
                
                let dot = a.values.iter()
                    .zip(b.values.iter())
                    .map(|(&x, &y)| x * y)
                    .sum::<f32>();
                    
                // 对于点积，我们返回负点积作为距离度量
                // 这样点积越大，距离越小
                Ok(-dot)
            }
        }
    }
    
    /// 计算一组向量与目标向量的距离
    pub fn calculate_batch(&self, query: &Vector, vectors: &[Vector]) -> Result<Vec<f32>> {
        vectors.iter().map(|v| self.calculate(query, v)).collect()
    }
}

/// 向量检索结果
#[derive(Debug, Clone)]
pub struct SearchResult {
    /// 向量ID
    pub id: String,
    
    /// 向量
    pub vector: Option<Vector>,
    
    /// 距离或相似度分数
    pub score: f32,
    
    /// 向量标签
    pub label: Option<String>,
    
    /// 元数据
    pub metadata: HashMap<String, String>,
}

impl SearchResult {
    /// 创建新的检索结果
    pub fn new(id: impl Into<String>, score: f32) -> Self {
        Self {
            id: id.into(),
            vector: None,
            score,
            label: None,
            metadata: HashMap::new(),
        }
    }
    
    /// 创建带向量的检索结果
    pub fn with_vector(id: impl Into<String>, vector: Vector, score: f32) -> Self {
        let id_str = id.into();
        let label = vector.metadata.get("label").cloned();
        let metadata = vector.metadata.clone();
        
        Self {
            id: id_str,
            vector: Some(vector),
            score,
            label,
            metadata,
        }
    }
}

/// 向量检索器接口
pub trait VectorSearcher: Send + Sync + Debug {
    /// 搜索最近邻向量
    fn search(&self, query: &Vector, k: usize) -> Result<Vec<SearchResult>>;
    
    /// 添加向量到索引
    fn add(&self, vector: &Vector) -> Result<()>;
    
    /// 批量添加向量到索引
    fn add_batch(&self, vectors: &[Vector]) -> Result<()>;
    
    /// 移除向量
    fn remove(&self, id: &str) -> Result<bool>;
    
    /// 获取索引大小
    fn size(&self) -> Result<usize>;
    
    /// 重建索引
    fn rebuild(&self) -> Result<()>;
}

/// 线性搜索器
#[derive(Debug)]
pub struct LinearSearcher {
    /// 向量存储
    storage: Arc<dyn VectorStorage>,
    
    /// 距离度量
    metric: DistanceMetric,
}

impl LinearSearcher {
    /// 创建新的线性搜索器
    pub fn new(storage: Arc<dyn VectorStorage>, metric: DistanceMetric) -> Self {
        Self {
            storage,
            metric,
        }
    }
}

impl VectorSearcher for LinearSearcher {
    fn search(&self, query: &Vector, k: usize) -> Result<Vec<SearchResult>> {
        let vectors = self.storage.get_all()?;
        
        let mut results: Vec<(usize, f32)> = Vec::with_capacity(vectors.len());
        for (i, vector) in vectors.iter().enumerate() {
            let distance = self.metric.calculate(query, vector)?;
            results.push((i, distance));
        }
        
        // 按距离排序
        results.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        
        // 截取前k个结果
        let k = std::cmp::min(k, results.len());
        let search_results = results.into_iter()
            .take(k)
            .map(|(i, score)| {
                let vector = &vectors[i];
                SearchResult::with_vector(vector.id.clone(), vector.clone(), score)
            })
            .collect();
            
        Ok(search_results)
    }
    
    fn add(&self, vector: &Vector) -> Result<()> {
        self.storage.save(vector)
    }
    
    fn add_batch(&self, vectors: &[Vector]) -> Result<()> {
        let batch = VectorBatch::new("batch", vectors.to_vec());
        self.storage.save_batch(&batch)
    }
    
    fn remove(&self, id: &str) -> Result<bool> {
        self.storage.delete(id)
    }
    
    fn size(&self) -> Result<usize> {
        self.storage.count()
    }
    
    fn rebuild(&self) -> Result<()> {
        // 线性搜索不需要重建索引
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_vector_creation() {
        let v = Vector::new(vec![1.0, 2.0, 3.0]);
        assert_eq!(v.dimension, 3);
        assert_eq!(v.values, vec![1.0, 2.0, 3.0]);
        
        let v_zeros = Vector::zeros(3);
        assert_eq!(v_zeros.values, vec![0.0, 0.0, 0.0]);
        
        let v_ones = Vector::ones(3);
        assert_eq!(v_ones.values, vec![1.0, 1.0, 1.0]);
    }
    
    #[test]
    fn test_vector_operations() {
        let v1 = Vector::new(vec![1.0, 2.0, 3.0]);
        let v2 = Vector::new(vec![4.0, 5.0, 6.0]);
        
        let sum = (v1.clone() + v2.clone()).unwrap();
        assert_eq!(sum.values, vec![5.0, 7.0, 9.0]);
        
        let diff = (v2.clone() - v1.clone()).unwrap();
        assert_eq!(diff.values, vec![3.0, 3.0, 3.0]);
        
        let scaled = v1.clone() * 2.0;
        assert_eq!(scaled.values, vec![2.0, 4.0, 6.0]);
        
        let divided = (v2 / 2.0).unwrap();
        assert_eq!(divided.values, vec![2.0, 2.5, 3.0]);
    }
    
    #[test]
    fn test_vector_similarity() {
        let v1 = Vector::new(vec![1.0, 0.0, 0.0]);
        let v2 = Vector::new(vec![0.0, 1.0, 0.0]);
        
        let similarity = v1.cosine_similarity(&v2).unwrap();
        assert_eq!(similarity, 0.0);
        
        let v3 = Vector::new(vec![1.0, 1.0, 0.0]);
        let v4 = Vector::new(vec![1.0, 1.0, 0.0]);
        
        let similarity = v3.cosine_similarity(&v4).unwrap();
        assert_eq!(similarity, 1.0);
    }
    
    #[test]
    fn test_vector_set() {
        let v1 = Vector::new(vec![1.0, 2.0, 3.0]);
        let v2 = Vector::new(vec![4.0, 5.0, 6.0]);
        
        let mut set = VectorSet::new();
        set.add(v1.clone());
        set.add(v2.clone());
        
        assert_eq!(set.len(), 2);
        assert_eq!(set.get(0).unwrap().values, v1.values);
        assert_eq!(set.get(1).unwrap().values, v2.values);
        
        let mean = set.mean().unwrap();
        assert_eq!(mean.values, vec![2.5, 3.5, 4.5]);
    }
} 