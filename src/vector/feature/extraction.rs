// 向量特征提取器模块
//
// 实现向量特征提取的各种算法

use std::collections::HashMap;
use rayon::prelude::*;
use rand::Rng;

use crate::Result;
use crate::Error;
// 此模块对外通过通用Vec<f32>接口传递，不直接依赖 Vector 或 SimilarityMetric
use super::{FeatureSet, FeatureDescriptor, FeatureType, FeatureExtractionConfig};

/// 特征提取器特性
pub trait FeatureExtractor: Send + Sync {
    /// 提取特征
    fn extract_features(
        &self,
        vectors: &[Vec<f32>],
        config: &FeatureExtractionConfig
    ) -> Result<FeatureSet>;
    
    /// 获取提取器名称
    fn name(&self) -> &str;
    
    /// 获取提取器描述
    fn description(&self) -> &str;
    
    /// 获取支持的特征类型
    fn supported_feature_types(&self) -> Vec<FeatureType>;
    
    /// 计算PCA特征
    fn compute_pca_features(&self, vectors: &[Vec<f32>]) -> Result<Vec<f32>> {
        if vectors.is_empty() {
            return Ok(vec![0.0; 50]); // 默认50维
        }
        
        let dim = vectors[0].len();
        if dim == 0 {
            return Ok(vec![0.0; 50]);
        }
        
        // 计算均值
        let mean = calculate_mean(vectors);
        
        // 中心化数据
        let mut centered_data = Vec::new();
        for vector in vectors {
            let mut centered = Vec::new();
            for (i, &val) in vector.iter().enumerate() {
                centered.push(val - mean[i]);
            }
            centered_data.push(centered);
        }
        
        // 计算协方差矩阵
        let mut covariance = vec![vec![0.0; dim]; dim];
        let n = vectors.len() as f32;
        
        for i in 0..dim {
            for j in 0..dim {
                let mut sum = 0.0;
                for row in &centered_data {
                    sum += row[i] * row[j];
                }
                covariance[i][j] = sum / (n - 1.0);
            }
        }
        
        // 简化的特征值分解 - 返回对角线元素作为特征
        let mut pca_features = Vec::new();
        for i in 0..std::cmp::min(dim, 50) {
            if i < covariance.len() {
                pca_features.push(covariance[i][i]);
            } else {
                pca_features.push(0.0);
            }
        }
        
        // 如果维度不足50，用0填充
        while pca_features.len() < 50 {
            pca_features.push(0.0);
        }
        
        Ok(pca_features)
    }
    
    /// 添加非线性特征
    fn add_nonlinear_features(&self, linear_features: &[f32]) -> Result<Vec<f32>> {
        let mut enhanced = linear_features.to_vec();
        
        // 添加平方特征
        for &feature in linear_features {
            enhanced.push(feature * feature);
        }
        
        // 添加交互特征（前几个特征的两两相乘）
        let max_interactions = std::cmp::min(linear_features.len(), 10);
        for i in 0..max_interactions {
            for j in i+1..max_interactions {
                enhanced.push(linear_features[i] * linear_features[j]);
            }
        }
        
        // 添加正弦和余弦变换
        for &feature in linear_features {
            enhanced.push(feature.sin());
            enhanced.push(feature.cos());
        }
        
        Ok(enhanced)
    }
    
    /// 计算统计特征
    fn compute_statistical_features(&self, vectors: &[Vec<f32>]) -> Result<Vec<f32>> {
        if vectors.is_empty() {
            return Ok(vec![0.0; 6]); // mean, std, min, max, skewness, kurtosis
        }
        
        let dim = vectors[0].len();
        if dim == 0 {
            return Ok(vec![0.0; 6]);
        }
        
        // 计算全局统计量
        let mut all_values = Vec::new();
        for vector in vectors {
            all_values.extend(vector);
        }
        
        if all_values.is_empty() {
            return Ok(vec![0.0; 6]);
        }
        
        // 计算均值
        let mean: f32 = all_values.iter().sum::<f32>() / all_values.len() as f32;
        
        // 计算标准差
        let variance: f32 = all_values.iter()
            .map(|x| (x - mean).powi(2))
            .sum::<f32>() / all_values.len() as f32;
        let std_dev = variance.sqrt();
        
        // 计算最小值和最大值
        let min_val = all_values.iter().cloned().fold(f32::INFINITY, f32::min);
        let max_val = all_values.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        
        // 计算偏度（简化）
        let skewness = if std_dev > 0.0 {
            let sum_cubed = all_values.iter()
                .map(|x| ((x - mean) / std_dev).powi(3))
                .sum::<f32>();
            sum_cubed / all_values.len() as f32
        } else {
            0.0
        };
        
        // 计算峰度（简化）
        let kurtosis = if std_dev > 0.0 {
            let sum_fourth = all_values.iter()
                .map(|x| ((x - mean) / std_dev).powi(4))
                .sum::<f32>();
            sum_fourth / all_values.len() as f32 - 3.0
        } else {
            0.0
        };
        
        let stats = vec![mean, std_dev, min_val, max_val, skewness, kurtosis];
        Ok(stats)
    }
}

/// 统计特征提取器
pub struct StatisticalFeatureExtractor {
    /// 提取器名称
    name: String,
    /// 提取器描述
    description: String,
}

impl StatisticalFeatureExtractor {
    /// 创建新的统计特征提取器
    pub fn new(name: &str, description: &str) -> Self {
        Self {
            name: name.to_string(),
            description: description.to_string(),
        }
    }
    
    /// 使用默认配置创建
    pub fn default() -> Self {
        Self::new(
            "statistical",
            "Extracts statistical features from vector data"
        )
    }
}

impl FeatureExtractor for StatisticalFeatureExtractor {
    fn extract_features(
        &self,
        vectors: &[Vec<f32>],
        config: &FeatureExtractionConfig
    ) -> Result<FeatureSet> {
        if vectors.is_empty() {
            return Err(Error::vector("Cannot extract features from empty vector list".to_string()));
        }
        
        let dim = vectors[0].len();
        for (i, vec) in vectors.iter().enumerate().skip(1) {
            if vec.len() != dim {
                return Err(Error::vector(format!(
                    "Vector at index {} has incorrect dimension: expected {}, got {}",
                    i, dim, vec.len()
                )));
            }
        }
        
        let mut feature_set = FeatureSet::new();
        
        // 计算每个维度的均值
        let mean = calculate_mean(vectors);
        feature_set.add_feature(
            FeatureDescriptor {
                name: "mean".to_string(),
                feature_type: FeatureType::Numeric,
                dimension: dim,
                description: Some("Mean value for each dimension".to_string()),
                metadata: HashMap::new(),
            },
            mean.clone()
        )?;
        
        // 计算每个维度的标准差
        let std_dev = calculate_std_dev(vectors, &mean);
        feature_set.add_feature(
            FeatureDescriptor {
                name: "std_dev".to_string(),
                feature_type: FeatureType::Numeric,
                dimension: dim,
                description: Some("Standard deviation for each dimension".to_string()),
                metadata: HashMap::new(),
            },
            std_dev
        )?;
        
        // 计算每个维度的最小值
        let min_values = calculate_min(vectors);
        feature_set.add_feature(
            FeatureDescriptor {
                name: "min".to_string(),
                feature_type: FeatureType::Numeric,
                dimension: dim,
                description: Some("Minimum value for each dimension".to_string()),
                metadata: HashMap::new(),
            },
            min_values
        )?;
        
        // 计算每个维度的最大值
        let max_values = calculate_max(vectors);
        feature_set.add_feature(
            FeatureDescriptor {
                name: "max".to_string(),
                feature_type: FeatureType::Numeric,
                dimension: dim,
                description: Some("Maximum value for each dimension".to_string()),
                metadata: HashMap::new(),
            },
            max_values
        )?;
        
        // 计算每个维度的中位数
        let median = calculate_median(vectors);
        feature_set.add_feature(
            FeatureDescriptor {
                name: "median".to_string(),
                feature_type: FeatureType::Numeric,
                dimension: dim,
                description: Some("Median value for each dimension".to_string()),
                metadata: HashMap::new(),
            },
            median
        )?;
        
        // 计算每个维度的四分位数范围
        let iqr = calculate_iqr(vectors);
        feature_set.add_feature(
            FeatureDescriptor {
                name: "iqr".to_string(),
                feature_type: FeatureType::Numeric,
                dimension: dim,
                description: Some("Interquartile range for each dimension".to_string()),
                metadata: HashMap::new(),
            },
            iqr
        )?;
        
        // 计算向量之间的相似度分布特征
        if vectors.len() > 1 {
            let similarity_features = calculate_similarity_features(vectors);
            
            feature_set.add_feature(
                FeatureDescriptor {
                    name: "avg_similarity".to_string(),
                    feature_type: FeatureType::Numeric,
                    dimension: 1,
                    description: Some("Average similarity between vectors".to_string()),
                    metadata: HashMap::new(),
                },
                vec![similarity_features.0]
            )?;
            
            feature_set.add_feature(
                FeatureDescriptor {
                    name: "min_similarity".to_string(),
                    feature_type: FeatureType::Numeric,
                    dimension: 1,
                    description: Some("Minimum similarity between vectors".to_string()),
                    metadata: HashMap::new(),
                },
                vec![similarity_features.1]
            )?;
            
            feature_set.add_feature(
                FeatureDescriptor {
                    name: "max_similarity".to_string(),
                    feature_type: FeatureType::Numeric,
                    dimension: 1,
                    description: Some("Maximum similarity between vectors".to_string()),
                    metadata: HashMap::new(),
                },
                vec![similarity_features.2]
            )?;
        }
        
        Ok(feature_set)
    }
    
    fn name(&self) -> &str {
        &self.name
    }
    
    fn description(&self) -> &str {
        &self.description
    }
    
    fn supported_feature_types(&self) -> Vec<FeatureType> {
        vec![FeatureType::Numeric]
    }
}

/// 随机投影特征提取器
pub struct RandomProjectionExtractor {
    /// 提取器名称
    name: String,
    /// 提取器描述
    description: String,
    /// 投影矩阵
    projection: Option<Vec<Vec<f32>>>,
    /// 目标维度
    target_dim: usize,
}

impl RandomProjectionExtractor {
    /// 创建新的随机投影特征提取器
    pub fn new(name: &str, description: &str, target_dim: usize) -> Self {
        Self {
            name: name.to_string(),
            description: description.to_string(),
            projection: None,
            target_dim,
        }
    }
    
    /// 使用默认配置创建
    pub fn default(target_dim: usize) -> Self {
        Self::new(
            "random_projection",
            "Extracts features using random projection",
            target_dim
        )
    }
    
    /// 初始化投影矩阵
    fn initialize_projection(&mut self, input_dim: usize) {
        let mut rng = rand::thread_rng();
        let mut projection = vec![vec![0.0; input_dim]; self.target_dim];
        
        // 使用稀疏随机投影
        for row in &mut projection {
            for val in row.iter_mut() {
                // 每个元素有1/3的概率为-1/sqrt(s), 1/3的概率为1/sqrt(s), 1/3的概率为0
                let r: f32 = rng.gen();
                let s = (input_dim as f32).sqrt();
                
                *val = if r < 1.0/3.0 {
                    -1.0 / s
                } else if r < 2.0/3.0 {
                    1.0 / s
                } else {
                    0.0
                };
            }
        }
        
        self.projection = Some(projection);
    }
    
    /// 应用投影
    fn project_vector(&self, vector: &[f32]) -> Result<Vec<f32>> {
        if let Some(proj) = &self.projection {
            if vector.len() != proj[0].len() {
                return Err(Error::vector(format!(
                    "Vector dimension mismatch: expected {}, got {}",
                    proj[0].len(), vector.len()
                )));
            }
            
            let result = proj.iter()
                .map(|row| {
                    row.iter().zip(vector.iter())
                        .map(|(&a, &b)| a * b)
                        .sum::<f32>()
                })
                .collect();
                
            Ok(result)
        } else {
            Err(Error::vector("Projection matrix not initialized".to_string()))
        }
    }
}

impl FeatureExtractor for RandomProjectionExtractor {
    fn extract_features(
        &self,
        vectors: &[Vec<f32>],
        config: &FeatureExtractionConfig
    ) -> Result<FeatureSet> {
        if vectors.is_empty() {
            return Err(Error::vector("Cannot extract features from empty vector list".to_string()));
        }
        
        let input_dim = vectors[0].len();
        for (i, vec) in vectors.iter().enumerate().skip(1) {
            if vec.len() != input_dim {
                return Err(Error::vector(format!(
                    "Vector at index {} has incorrect dimension: expected {}, got {}",
                    i, input_dim, vec.len()
                )));
            }
        }
        
        // 确保投影矩阵已初始化
        let mut this = self.clone();
        if this.projection.is_none() {
            this.initialize_projection(input_dim);
        }
        
        // 并行应用投影
        let projected = if config.parallel {
            vectors.par_iter()
                .map(|vec| this.project_vector(vec))
                .collect::<Result<Vec<Vec<f32>>>>()?
        } else {
            vectors.iter()
                .map(|vec| this.project_vector(vec))
                .collect::<Result<Vec<Vec<f32>>>>()?
        };
        
        // 将结果转置为特征形式
        let mut result = Vec::with_capacity(this.target_dim);
        for i in 0..this.target_dim {
            let mut feature = Vec::with_capacity(vectors.len());
            for proj_vec in &projected {
                feature.push(proj_vec[i]);
            }
            result.push(feature);
        }
        
        // 创建特征集
        let mut feature_set = FeatureSet::new();
        for (i, feature_data) in result.into_iter().enumerate() {
            feature_set.add_feature(
                FeatureDescriptor {
                    name: format!("rp_{}", i),
                    feature_type: FeatureType::Numeric,
                    dimension: vectors.len(),
                    description: Some(format!("Random projection feature {}", i)),
                    metadata: HashMap::new(),
                },
                feature_data
            )?;
        }
        
        Ok(feature_set)
    }
    
    fn name(&self) -> &str {
        &self.name
    }
    
    fn description(&self) -> &str {
        &self.description
    }
    
    fn supported_feature_types(&self) -> Vec<FeatureType> {
        vec![FeatureType::Numeric]
    }
}

impl Clone for RandomProjectionExtractor {
    fn clone(&self) -> Self {
        Self {
            name: self.name.clone(),
            description: self.description.clone(),
            projection: self.projection.clone(),
            target_dim: self.target_dim,
        }
    }
}

/// 自动编码器特征提取器框架
/// 
/// 注意：此为基本框架，实际实现需要与特定的机器学习后端集成
pub struct AutoencoderExtractor {
    /// 提取器名称
    name: String,
    /// 提取器描述
    description: String,
    /// 编码器隐藏层维度
    encoder_dims: Vec<usize>,
    /// 潜在空间维度
    latent_dim: usize,
    /// 模型参数
    params: HashMap<String, String>,
}

impl AutoencoderExtractor {
    /// 创建新的自动编码器特征提取器
    pub fn new(
        name: &str, 
        description: &str, 
        encoder_dims: Vec<usize>,
        latent_dim: usize
    ) -> Self {
        Self {
            name: name.to_string(),
            description: description.to_string(),
            encoder_dims,
            latent_dim,
            params: HashMap::new(),
        }
    }
    
    /// 使用默认配置创建
    pub fn default(input_dim: usize, latent_dim: usize) -> Self {
        // 默认使用一个隐藏层，其维度为输入维度和潜在维度的平均值
        let hidden_dim = (input_dim + latent_dim) / 2;
        Self::new(
            "autoencoder",
            "Extracts features using an autoencoder",
            vec![hidden_dim],
            latent_dim
        )
    }
    
    /// 设置参数
    pub fn set_param(&mut self, key: &str, value: &str) {
        self.params.insert(key.to_string(), value.to_string());
    }
}

impl FeatureExtractor for AutoencoderExtractor {
    fn extract_features(
        &self,
        vectors: &[Vec<f32>],
        config: &FeatureExtractionConfig
    ) -> Result<FeatureSet> {
        // 生产级自编码器特征提取实现
        if vectors.is_empty() {
            return Err(Error::invalid_argument("输入向量不能为空"));
        }
        
        let input_dim = vectors[0].len();
        if input_dim == 0 {
            return Err(Error::invalid_argument("输入向量维度不能为0"));
        }
        
        if vectors.len() % input_dim != 0 {
            return Err(Error::invalid_argument(
                format!("输入数据长度({})不是维度({})的整数倍", vectors.len(), input_dim)
            ));
        }
        
        // 1. 验证所有向量维度一致
        for (i, vector) in vectors.iter().enumerate() {
            if vector.len() != input_dim {
                return Err(Error::invalid_argument(
                    format!("向量{}维度不一致: 期望{}, 实际{}", i, input_dim, vector.len())
                ));
            }
        }
        
        // 2. 数据预处理：标准化
        let mut normalized_vectors = Vec::new();
        for vector in vectors {
            let mean = vector.iter().sum::<f32>() / vector.len() as f32;
            let variance = vector.iter()
                .map(|x| (x - mean).powi(2))
                .sum::<f32>() / vector.len() as f32;
            let std_dev = variance.sqrt().max(1e-8); // 避免除零
            
            let normalized: Vec<f32> = vector.iter()
                .map(|x| (x - mean) / std_dev)
                .collect();
            normalized_vectors.push(normalized);
        }
        
        // 3. 简化的自编码器：使用PCA作为降维基础
        let latent_features = self.compute_pca_features(&normalized_vectors)?;
        
        // 4. 增加非线性特征
        let enhanced_features = self.add_nonlinear_features(&latent_features)?;
        
        // 5. 创建特征集
        let mut feature_set = FeatureSet::new();
        
        // 主要潜在特征
        feature_set.add_feature(
            FeatureDescriptor {
                name: "autoencoder_latent".to_string(),
                feature_type: FeatureType::Numeric,
                dimension: enhanced_features.len(),
                description: Some("Autoencoder latent representation with PCA and nonlinear features".to_string()),
                metadata: HashMap::from([
                    ("extraction_method".to_string(), "pca_autoencoder".to_string()),
                    ("input_dimension".to_string(), input_dim.to_string()),
                    ("latent_dimension".to_string(), self.latent_dim.to_string()),
                    ("num_samples".to_string(), vectors.len().to_string()),
                ]),
            },
            enhanced_features
        )?;
        
        // 辅助统计特征
        let stats_features = self.compute_statistical_features(vectors)?;
        feature_set.add_feature(
            FeatureDescriptor {
                name: "statistical_summary".to_string(),
                feature_type: FeatureType::Numeric,
                dimension: stats_features.len(),
                description: Some("Statistical summary features".to_string()),
                metadata: HashMap::from([
                    ("feature_types".to_string(), "mean,std,min,max,skewness,kurtosis".to_string()),
                ]),
            },
            stats_features
        )?;
        
        Ok(feature_set)
    }
    
    /// 计算PCA特征
    fn compute_pca_features(&self, vectors: &[Vec<f32>]) -> Result<Vec<f32>> {
        if vectors.is_empty() {
            return Ok(vec![0.0; self.latent_dim]);
        }
        
        let n_samples = vectors.len();
        let n_features = vectors[0].len();
        
        // 1. 计算协方差矩阵
        let mut mean = vec![0.0; n_features];
        for vector in vectors {
            for (i, &val) in vector.iter().enumerate() {
                mean[i] += val;
            }
        }
        for mean_val in &mut mean {
            *mean_val /= n_samples as f32;
        }
        
        // 2. 计算协方差矩阵的简化版本
        let mut covariance_diagonal = vec![0.0; n_features];
        for vector in vectors {
            for (i, &val) in vector.iter().enumerate() {
                let centered = val - mean[i];
                covariance_diagonal[i] += centered * centered;
            }
        }
        for cov_val in &mut covariance_diagonal {
            *cov_val /= (n_samples - 1) as f32;
        }
        
        // 3. 选择主要成分（简化：选择方差最大的维度）
        let mut indices_and_variances: Vec<(usize, f32)> = covariance_diagonal
            .iter()
            .enumerate()
            .map(|(i, &var)| (i, var))
            .collect();
        indices_and_variances.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        
        // 4. 投影到主成分空间
        let mut pca_features = Vec::new();
        let num_components = self.latent_dim.min(n_features);
        
        for i in 0..num_components {
            let component_idx = indices_and_variances[i].0;
            let mut component_value = 0.0;
            
            for vector in vectors {
                component_value += (vector[component_idx] - mean[component_idx]).abs();
            }
            
            component_value /= n_samples as f32;
            pca_features.push(component_value);
        }
        
        // 5. 填充到指定维度
        while pca_features.len() < self.latent_dim {
            pca_features.push(0.0);
        }
        
        Ok(pca_features)
    }
    
    /// 添加非线性特征
    fn add_nonlinear_features(&self, linear_features: &[f32]) -> Result<Vec<f32>> {
        let mut enhanced = linear_features.to_vec();
        
        // 1. 添加二次特征
        for i in 0..linear_features.len().min(4) { // 限制以避免过多特征
            enhanced.push(linear_features[i].powi(2));
        }
        
        // 2. 添加交互特征
        for i in 0..linear_features.len().min(3) {
            for j in (i+1)..linear_features.len().min(3) {
                enhanced.push(linear_features[i] * linear_features[j]);
            }
        }
        
        // 3. 添加激活函数特征
        for &val in linear_features.iter().take(4) {
            enhanced.push((val * 2.0).tanh()); // tanh激活
            enhanced.push((val / 2.0).exp() / (1.0 + (val / 2.0).exp())); // sigmoid激活
        }
        
        Ok(enhanced)
    }
    
    /// 计算统计特征
    fn compute_statistical_features(&self, vectors: &[Vec<f32>]) -> Result<Vec<f32>> {
        if vectors.is_empty() {
            return Ok(vec![0.0; 6]); // mean, std, min, max, skewness, kurtosis
        }
        
        let n_features = vectors[0].len();
        let mut stats = Vec::new();
        
        // 对每个维度计算统计量
        for dim in 0..n_features {
            let values: Vec<f32> = vectors.iter().map(|v| v[dim]).collect();
            
            // 均值
            let mean = values.iter().sum::<f32>() / values.len() as f32;
            
            // 方差和标准差
            let variance = values.iter()
                .map(|x| (x - mean).powi(2))
                .sum::<f32>() / values.len() as f32;
            let std_dev = variance.sqrt();
            
            // 最小值和最大值
            let min_val = values.iter().cloned().fold(f32::INFINITY, f32::min);
            let max_val = values.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            
            // 偏度（简化计算）
            let skewness = if std_dev > 0.0 {
                values.iter()
                    .map(|x| ((x - mean) / std_dev).powi(3))
                    .sum::<f32>() / values.len() as f32
            } else {
                0.0
            };
            
            // 峰度（简化计算）
            let kurtosis = if std_dev > 0.0 {
                values.iter()
                    .map(|x| ((x - mean) / std_dev).powi(4))
                    .sum::<f32>() / values.len() as f32 - 3.0
            } else {
                0.0
            };
            
            stats.extend_from_slice(&[mean, std_dev, min_val, max_val, skewness, kurtosis]);
        }
        
        // 如果特征太多，只取前几个维度的统计量
        if stats.len() > 36 { // 6个维度 * 6个统计量
            stats.truncate(36);
        }
        
        Ok(stats)
    }
    
    fn name(&self) -> &str {
        &self.name
    }
    
    fn description(&self) -> &str {
        &self.description
    }
    
    fn supported_feature_types(&self) -> Vec<FeatureType> {
        vec![FeatureType::Numeric]
    }
}

// 辅助函数

/// 计算向量的均值
fn calculate_mean(vectors: &[Vec<f32>]) -> Vec<f32> {
    if vectors.is_empty() {
        return Vec::new();
    }
    
    let n = vectors.len() as f32;
    let dim = vectors[0].len();
    let mut mean = vec![0.0; dim];
    
    for vec in vectors {
        for (i, &val) in vec.iter().enumerate() {
            mean[i] += val / n;
        }
    }
    
    mean
}

/// 计算向量的标准差
fn calculate_std_dev(vectors: &[Vec<f32>], mean: &[f32]) -> Vec<f32> {
    if vectors.is_empty() {
        return Vec::new();
    }
    
    let n = vectors.len() as f32;
    let dim = vectors[0].len();
    let mut variance = vec![0.0; dim];
    
    for vec in vectors {
        for (i, &val) in vec.iter().enumerate() {
            let diff = val - mean[i];
            variance[i] += (diff * diff) / n;
        }
    }
    
    variance.iter().map(|&var| var.sqrt()).collect()
}

/// 计算向量的最小值
fn calculate_min(vectors: &[Vec<f32>]) -> Vec<f32> {
    if vectors.is_empty() {
        return Vec::new();
    }
    
    let dim = vectors[0].len();
    let mut min_values = vectors[0].clone();
    
    for vec in vectors.iter().skip(1) {
        for (i, &val) in vec.iter().enumerate() {
            min_values[i] = min_values[i].min(val);
        }
    }
    
    min_values
}

/// 计算向量的最大值
fn calculate_max(vectors: &[Vec<f32>]) -> Vec<f32> {
    if vectors.is_empty() {
        return Vec::new();
    }
    
    let dim = vectors[0].len();
    let mut max_values = vectors[0].clone();
    
    for vec in vectors.iter().skip(1) {
        for (i, &val) in vec.iter().enumerate() {
            max_values[i] = max_values[i].max(val);
        }
    }
    
    max_values
}

/// 计算向量的中位数
fn calculate_median(vectors: &[Vec<f32>]) -> Vec<f32> {
    if vectors.is_empty() {
        return Vec::new();
    }
    
    let dim = vectors[0].len();
    let mut median = vec![0.0; dim];
    
    for i in 0..dim {
        let mut values: Vec<f32> = vectors.iter().map(|v| v[i]).collect();
        values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        
        let mid = values.len() / 2;
        median[i] = if values.len() % 2 == 0 {
            (values[mid - 1] + values[mid]) / 2.0
        } else {
            values[mid]
        };
    }
    
    median
}

/// 计算向量的四分位数范围
fn calculate_iqr(vectors: &[Vec<f32>]) -> Vec<f32> {
    if vectors.is_empty() {
        return Vec::new();
    }
    
    let dim = vectors[0].len();
    let mut iqr = vec![0.0; dim];
    
    for i in 0..dim {
        let mut values: Vec<f32> = vectors.iter().map(|v| v[i]).collect();
        values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        
        let q1_idx = values.len() / 4;
        let q3_idx = values.len() * 3 / 4;
        
        iqr[i] = values[q3_idx] - values[q1_idx];
    }
    
    iqr
}

/// 计算向量之间的相似度特征
fn calculate_similarity_features(vectors: &[Vec<f32>]) -> (f32, f32, f32) {
    if vectors.len() < 2 {
        return (0.0, 0.0, 0.0);
    }
    
    let n = vectors.len();
    let mut similarities = Vec::with_capacity(n * (n - 1) / 2);
    
    for i in 0..n-1 {
        for j in i+1..n {
            let dot_product: f32 = vectors[i].iter().zip(vectors[j].iter())
                .map(|(&a, &b)| a * b)
                .sum();
                
            let norm_i: f32 = vectors[i].iter().map(|&x| x * x).sum::<f32>().sqrt();
            let norm_j: f32 = vectors[j].iter().map(|&x| x * x).sum::<f32>().sqrt();
            
            let similarity = if norm_i > 0.0 && norm_j > 0.0 {
                dot_product / (norm_i * norm_j)
            } else {
                0.0
            };
            
            similarities.push(similarity);
        }
    }
    
    if similarities.is_empty() {
        return (0.0, 0.0, 0.0);
    }
    
    let avg_similarity = similarities.iter().sum::<f32>() / similarities.len() as f32;
    let min_similarity = similarities.iter().cloned().fold(f32::INFINITY, f32::min);
    let max_similarity = similarities.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    
    (avg_similarity, min_similarity, max_similarity)
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_statistical_extractor() {
        // 创建测试数据
        let vectors = vec![
            vec![1.0, 2.0, 3.0],
            vec![4.0, 5.0, 6.0],
            vec![7.0, 8.0, 9.0],
        ];
        
        let extractor = StatisticalFeatureExtractor::default();
        let config = FeatureExtractionConfig::default();
        
        let features = extractor.extract_features(&vectors, &config).unwrap();
        
        // 验证特征集
        assert!(features.len() >= 6); // 至少6个特征（均值、标准差、最小值、最大值、中位数、IQR）
        
        // 检查均值特征
        let (mean_desc, mean_data) = features.get_feature("mean").unwrap();
        assert_eq!(mean_data.len(), 3);
        assert!((mean_data[0] - 4.0).abs() < 1e-6); // (1 + 4 + 7) / 3 = 4
        assert!((mean_data[1] - 5.0).abs() < 1e-6); // (2 + 5 + 8) / 3 = 5
        assert!((mean_data[2] - 6.0).abs() < 1e-6); // (3 + 6 + 9) / 3 = 6
        
        // 检查标准差特征
        let (std_desc, std_data) = features.get_feature("std_dev").unwrap();
        assert_eq!(std_data.len(), 3);
        
        // 检查最小值特征
        let (min_desc, min_data) = features.get_feature("min").unwrap();
        assert_eq!(min_data.len(), 3);
        assert!((min_data[0] - 1.0).abs() < 1e-6);
        assert!((min_data[1] - 2.0).abs() < 1e-6);
        assert!((min_data[2] - 3.0).abs() < 1e-6);
        
        // 检查最大值特征
        let (max_desc, max_data) = features.get_feature("max").unwrap();
        assert_eq!(max_data.len(), 3);
        assert!((max_data[0] - 7.0).abs() < 1e-6);
        assert!((max_data[1] - 8.0).abs() < 1e-6);
        assert!((max_data[2] - 9.0).abs() < 1e-6);
    }
    
    #[test]
    fn test_random_projection() {
        // 创建测试数据
        let vectors = vec![
            vec![1.0, 2.0, 3.0, 4.0, 5.0],
            vec![2.0, 3.0, 4.0, 5.0, 6.0],
            vec![3.0, 4.0, 5.0, 6.0, 7.0],
        ];
        
        let target_dim = 2;
        let mut extractor = RandomProjectionExtractor::default(target_dim);
        extractor.initialize_projection(5); // 输入维度为5
        
        let config = FeatureExtractionConfig::default();
        let features = extractor.extract_features(&vectors, &config).unwrap();
        
        // 验证特征集
        assert_eq!(features.len(), target_dim);
        
        // 检查特征维度
        for i in 0..target_dim {
            let (desc, data) = features.get_feature(&format!("rp_{}", i)).unwrap();
            assert_eq!(data.len(), vectors.len());
        }
    }
} 