// 向量特征提取模块
//
// 提供用于向量特征提取的各种工具和算法

// 当前模块不持有共享状态，不需要 Arc/Mutex
use std::collections::HashMap;
use rayon::prelude::*;

use crate::Result;
use crate::Error;

pub mod extraction;
pub mod reduction;
pub mod transform;

pub use extraction::FeatureExtractor;
pub use reduction::DimensionReducer;
pub use transform::VectorTransformer;

/// 特征类型
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum FeatureType {
    /// 数值特征
    Numeric,
    /// 类别特征
    Categorical,
    /// 文本特征
    Text,
    /// 图像特征
    Image,
    /// 时间序列特征
    TimeSeries,
    /// 空间特征
    Spatial,
    /// 图结构特征
    Graph,
    /// 自定义特征
    Custom,
}

/// 特征描述
#[derive(Debug, Clone)]
pub struct FeatureDescriptor {
    /// 特征名称
    pub name: String,
    /// 特征类型
    pub feature_type: FeatureType,
    /// 特征维度
    pub dimension: usize,
    /// 特征描述
    pub description: Option<String>,
    /// 其他元数据
    pub metadata: HashMap<String, String>,
}

/// 特征集合
#[derive(Debug, Clone)]
pub struct FeatureSet {
    /// 特征描述符
    pub descriptors: Vec<FeatureDescriptor>,
    /// 特征数据
    pub data: Vec<Vec<f32>>,
    /// 特征索引映射
    pub feature_indices: HashMap<String, usize>,
}

impl FeatureSet {
    /// 创建新的特征集合
    pub fn new() -> Self {
        Self {
            descriptors: Vec::new(),
            data: Vec::new(),
            feature_indices: HashMap::new(),
        }
    }
    
    /// 添加特征
    pub fn add_feature(&mut self, descriptor: FeatureDescriptor, data: Vec<f32>) -> Result<()> {
        // 检查特征名称是否已存在
        if self.feature_indices.contains_key(&descriptor.name) {
            return Err(Error::vector(format!(
                "Feature '{}' already exists in the feature set", descriptor.name
            )));
        }
        
        // 检查维度是否匹配
        if descriptor.dimension != data.len() {
            return Err(Error::vector(format!(
                "Feature data dimension mismatch: expected {}, got {}",
                descriptor.dimension, data.len()
            )));
        }
        
        // 添加特征
        let index = self.descriptors.len();
        self.feature_indices.insert(descriptor.name.clone(), index);
        self.descriptors.push(descriptor);
        self.data.push(data);
        
        Ok(())
    }
    
    /// 获取特征
    pub fn get_feature(&self, name: &str) -> Option<(&FeatureDescriptor, &Vec<f32>)> {
        self.feature_indices.get(name).map(|&index| {
            (&self.descriptors[index], &self.data[index])
        })
    }
    
    /// 获取特征数量
    pub fn len(&self) -> usize {
        self.descriptors.len()
    }
    
    /// 检查特征集合是否为空
    pub fn is_empty(&self) -> bool {
        self.descriptors.is_empty()
    }
    
    /// 合并两个特征集
    pub fn merge(&mut self, other: &FeatureSet) -> Result<()> {
        for (i, descriptor) in other.descriptors.iter().enumerate() {
            if !self.feature_indices.contains_key(&descriptor.name) {
                self.add_feature(descriptor.clone(), other.data[i].clone())?;
            }
        }
        
        Ok(())
    }
    
    /// 提取子集
    pub fn subset(&self, feature_names: &[String]) -> Result<FeatureSet> {
        let mut subset = FeatureSet::new();
        
        for name in feature_names {
            if let Some((descriptor, data)) = self.get_feature(name) {
                subset.add_feature(descriptor.clone(), data.clone())?;
            } else {
                return Err(Error::vector(format!(
                    "Feature '{}' not found in the feature set", name
                )));
            }
        }
        
        Ok(subset)
    }
    
    /// 连接特征为向量
    pub fn to_vector(&self) -> Vec<f32> {
        let total_dim: usize = self.data.iter().map(|v| v.len()).sum();
        let mut result = Vec::with_capacity(total_dim);
        
        for data in &self.data {
            result.extend_from_slice(data);
        }
        
        result
    }
    
    /// 构建特征矩阵
    pub fn to_matrix(&self) -> Vec<Vec<f32>> {
        let n = self.len();
        if n == 0 {
            return Vec::new();
        }
        
        // 假设所有特征具有相同的样本数量
        let sample_count = self.data[0].len();
        let mut matrix = vec![Vec::with_capacity(n); sample_count];
        
        for i in 0..sample_count {
            for feature_data in &self.data {
                if i < feature_data.len() {
                    matrix[i].push(feature_data[i]);
                }
            }
        }
        
        matrix
    }
}

/// 特征提取配置
#[derive(Debug, Clone)]
pub struct FeatureExtractionConfig {
    /// 特征类型
    pub feature_types: Vec<FeatureType>,
    /// 最大特征数量
    pub max_features: Option<usize>,
    /// 最小特征频率
    pub min_frequency: Option<f32>,
    /// 并行处理
    pub parallel: bool,
    /// 其他配置参数
    pub params: HashMap<String, String>,
}

impl Default for FeatureExtractionConfig {
    fn default() -> Self {
        Self {
            feature_types: vec![FeatureType::Numeric],
            max_features: None,
            min_frequency: None,
            parallel: true,
            params: HashMap::new(),
        }
    }
}

/// 特征工程管理器
pub struct FeatureManager {
    /// 特征提取器
    extractors: HashMap<String, Box<dyn FeatureExtractor>>,
    /// 维度约简器
    reducers: HashMap<String, Box<dyn DimensionReducer>>,
    /// 向量转换器
    transformers: HashMap<String, Box<dyn VectorTransformer>>,
}

impl FeatureManager {
    /// 创建新的特征管理器
    pub fn new() -> Self {
        Self {
            extractors: HashMap::new(),
            reducers: HashMap::new(),
            transformers: HashMap::new(),
        }
    }
    
    /// 注册特征提取器
    pub fn register_extractor<T: FeatureExtractor + 'static>(
        &mut self, 
        name: &str, 
        extractor: T
    ) -> Result<()> {
        if self.extractors.contains_key(name) {
            return Err(Error::vector(format!(
                "Feature extractor '{}' already registered", name
            )));
        }
        
        self.extractors.insert(name.to_string(), Box::new(extractor));
        Ok(())
    }
    
    /// 注册维度约简器
    pub fn register_reducer<T: DimensionReducer + 'static>(
        &mut self, 
        name: &str, 
        reducer: T
    ) -> Result<()> {
        if self.reducers.contains_key(name) {
            return Err(Error::vector(format!(
                "Dimension reducer '{}' already registered", name
            )));
        }
        
        self.reducers.insert(name.to_string(), Box::new(reducer));
        Ok(())
    }
    
    /// 注册向量转换器
    pub fn register_transformer<T: VectorTransformer + 'static>(
        &mut self, 
        name: &str, 
        transformer: T
    ) -> Result<()> {
        if self.transformers.contains_key(name) {
            return Err(Error::vector(format!(
                "Vector transformer '{}' already registered", name
            )));
        }
        
        self.transformers.insert(name.to_string(), Box::new(transformer));
        Ok(())
    }
    
    /// 提取特征
    pub fn extract_features(
        &self,
        extractor_name: &str,
        vectors: &[Vec<f32>],
        config: &FeatureExtractionConfig
    ) -> Result<FeatureSet> {
        let extractor = self.extractors.get(extractor_name)
            .ok_or_else(|| Error::vector(format!(
                "Feature extractor '{}' not found", extractor_name
            )))?;
        
        extractor.extract_features(vectors, config)
    }
    
    /// 降低维度
    pub fn reduce_dimensions(
        &self,
        reducer_name: &str,
        vectors: &[Vec<f32>],
        target_dim: usize
    ) -> Result<Vec<Vec<f32>>> {
        let reducer = self.reducers.get(reducer_name)
            .ok_or_else(|| Error::vector(format!(
                "Dimension reducer '{}' not found", reducer_name
            )))?;
        
        reducer.reduce_dimensions(vectors, target_dim)
    }
    
    /// 转换向量
    pub fn transform_vectors(
        &self,
        transformer_name: &str,
        vectors: &[Vec<f32>]
    ) -> Result<Vec<Vec<f32>>> {
        let transformer = self.transformers.get(transformer_name)
            .ok_or_else(|| Error::vector(format!(
                "Vector transformer '{}' not found", transformer_name
            )))?;
        
        transformer.transform(vectors)
    }
} 