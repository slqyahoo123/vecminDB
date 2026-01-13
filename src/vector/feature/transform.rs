// 向量变换器模块
//
// 提供对向量数据的各种变换功能

use rayon::prelude::*;

use crate::Result;
use crate::Error;
// 该模块以通用数组接口工作，不直接使用 Vector 类型

/// 向量变换器特性
pub trait VectorTransformer: Send + Sync {
    /// 对向量集合应用变换
    fn transform(&self, vectors: &[Vec<f32>]) -> Result<Vec<Vec<f32>>>;
    
    /// 对单个向量应用变换
    fn transform_one(&self, vector: &[f32]) -> Result<Vec<f32>>;
    
    /// 获取变换器名称
    fn name(&self) -> &str;
    
    /// 获取变换器描述
    fn description(&self) -> &str;
}

/// 向量归一化变换器
pub struct NormalizerTransformer {
    /// 变换器名称
    name: String,
    /// 变换器描述
    description: String,
    /// 归一化方式
    norm_type: NormType,
}

/// 归一化方式
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum NormType {
    /// L1范数(曼哈顿距离)
    L1,
    /// L2范数(欧几里得距离)
    L2,
    /// 最大值归一化
    Max,
    /// 均值归一化
    Mean,
}

impl NormalizerTransformer {
    /// 创建新的归一化变换器
    pub fn new(name: &str, description: &str, norm_type: NormType) -> Self {
        Self {
            name: name.to_string(),
            description: description.to_string(),
            norm_type,
        }
    }
    
    /// 创建默认的L2归一化变换器
    pub fn default() -> Self {
        Self::new(
            "l2_normalizer",
            "L2 normalization transformer",
            NormType::L2
        )
    }
    
    /// 计算向量的范数
    fn compute_norm(&self, vector: &[f32]) -> f32 {
        match self.norm_type {
            NormType::L1 => vector.iter().fold(0.0, |acc, &val| acc + val.abs()),
            NormType::L2 => vector.iter().fold(0.0, |acc, &val| acc + val * val).sqrt(),
            NormType::Max => vector.iter().fold(0.0, |acc, &val| acc.max(val.abs())),
            NormType::Mean => {
                let sum = vector.iter().fold(0.0, |acc, &val| acc + val.abs());
                if vector.len() > 0 {
                    sum / vector.len() as f32
                } else {
                    0.0
                }
            }
        }
    }
}

impl VectorTransformer for NormalizerTransformer {
    fn transform(&self, vectors: &[Vec<f32>]) -> Result<Vec<Vec<f32>>> {
        if vectors.is_empty() {
            return Ok(Vec::new());
        }
        
        // 验证所有向量维度一致
        let dim = vectors[0].len();
        for (i, vec) in vectors.iter().enumerate().skip(1) {
            if vec.len() != dim {
                return Err(Error::vector(format!(
                    "Vector at index {} has incorrect dimension: expected {}, got {}",
                    i, dim, vec.len()
                )));
            }
        }
        
        // 并行归一化每个向量
        let normalized = vectors.par_iter()
            .map(|vec| self.transform_one(vec))
            .collect::<Result<Vec<Vec<f32>>>>()?;
        
        Ok(normalized)
    }
    
    fn transform_one(&self, vector: &[f32]) -> Result<Vec<f32>> {
        if vector.is_empty() {
            return Ok(Vec::new());
        }
        
        // 计算范数
        let norm = self.compute_norm(vector);
        
        // 避免除以零
        if norm > 1e-10 {
            // 归一化
            let normalized = vector.iter().map(|&val| val / norm).collect();
            Ok(normalized)
        } else {
            // 如果范数接近零，返回原向量的拷贝
            Ok(vector.to_vec())
        }
    }
    
    fn name(&self) -> &str {
        &self.name
    }
    
    fn description(&self) -> &str {
        &self.description
    }
}

/// 标准化变换器(Z-Score标准化)
pub struct StandardScalerTransformer {
    /// 变换器名称
    name: String,
    /// 变换器描述
    description: String,
    /// 均值向量
    mean: Option<Vec<f32>>,
    /// 标准差向量
    std_dev: Option<Vec<f32>>,
    /// 是否使用中心化
    with_mean: bool,
    /// 是否使用标准差缩放
    with_std: bool,
}

impl StandardScalerTransformer {
    /// 创建新的标准化变换器
    pub fn new(
        name: &str, 
        description: &str, 
        with_mean: bool, 
        with_std: bool
    ) -> Self {
        Self {
            name: name.to_string(),
            description: description.to_string(),
            mean: None,
            std_dev: None,
            with_mean,
            with_std,
        }
    }
    
    /// 创建默认的标准化变换器
    pub fn default() -> Self {
        Self::new(
            "standard_scaler",
            "Z-Score standardization transformer",
            true,
            true
        )
    }
    
    /// 拟合数据，计算均值和标准差
    pub fn fit(&mut self, vectors: &[Vec<f32>]) -> Result<()> {
        if vectors.is_empty() {
            return Err(Error::vector("Cannot fit transformer on empty vector list".to_string()));
        }
        
        let dim = vectors[0].len();
        // 验证所有向量维度一致
        for (i, vec) in vectors.iter().enumerate().skip(1) {
            if vec.len() != dim {
                return Err(Error::vector(format!(
                    "Vector at index {} has incorrect dimension: expected {}, got {}",
                    i, dim, vec.len()
                )));
            }
        }
        
        // 计算均值
        if self.with_mean {
            let mut mean = vec![0.0; dim];
            let n = vectors.len() as f32;
            
            for vec in vectors {
                for (i, &val) in vec.iter().enumerate() {
                    mean[i] += val / n;
                }
            }
            
            self.mean = Some(mean);
        }
        
        // 计算标准差
        if self.with_std {
            let mut variance = vec![0.0; dim];
            let n = vectors.len() as f32;
            // 使用本地所有权以避免悬垂引用
            let mean_vec: Vec<f32> = if let Some(ref mean) = self.mean {
                mean.clone()
            } else {
                // 如果没有计算均值，临时计算一个
                let mut temp_mean = vec![0.0; dim];
                for vec in vectors {
                    for (i, &val) in vec.iter().enumerate() {
                        temp_mean[i] += val / n;
                    }
                }
                temp_mean
            };
            
            for vec in vectors {
                for (i, &val) in vec.iter().enumerate() {
                    let diff = val - mean_vec[i];
                    variance[i] += (diff * diff) / n;
                }
            }
            
            // 避免除以零
            for var in &mut variance {
                if *var < 1e-10 {
                    *var = 1.0;
                }
            }
            
            let std_dev = variance.iter().map(|&var| var.sqrt()).collect();
            self.std_dev = Some(std_dev);
        }
        
        Ok(())
    }
}

impl VectorTransformer for StandardScalerTransformer {
    fn transform(&self, vectors: &[Vec<f32>]) -> Result<Vec<Vec<f32>>> {
        if vectors.is_empty() {
            return Ok(Vec::new());
        }
        
        // 验证所有向量维度一致
        let dim = vectors[0].len();
        for (i, vec) in vectors.iter().enumerate().skip(1) {
            if vec.len() != dim {
                return Err(Error::vector(format!(
                    "Vector at index {} has incorrect dimension: expected {}, got {}",
                    i, dim, vec.len()
                )));
            }
        }
        
        // 验证是否已经拟合
        if self.with_mean && self.mean.is_none() || self.with_std && self.std_dev.is_none() {
            return Err(Error::vector("StandardScalerTransformer not fitted".to_string()));
        }
        
        // 并行标准化每个向量
        let transformed = vectors.par_iter()
            .map(|vec| self.transform_one(vec))
            .collect::<Result<Vec<Vec<f32>>>>()?;
        
        Ok(transformed)
    }
    
    fn transform_one(&self, vector: &[f32]) -> Result<Vec<f32>> {
        if vector.is_empty() {
            return Ok(Vec::new());
        }
        
        // 验证向量维度
        let expected_dim = if let Some(ref mean) = self.mean {
            mean.len()
        } else if let Some(ref std_dev) = self.std_dev {
            std_dev.len()
        } else {
            // 尚未拟合
            return Err(Error::vector("StandardScalerTransformer not fitted".to_string()));
        };
        
        if vector.len() != expected_dim {
            return Err(Error::vector(format!(
                "Vector dimension mismatch: expected {}, got {}",
                expected_dim, vector.len()
            )));
        }
        
        // 应用变换
        let mut result = vector.to_vec();
        
        // 中心化
        if self.with_mean {
            if let Some(ref mean) = self.mean {
                for (i, val) in result.iter_mut().enumerate() {
                    *val -= mean[i];
                }
            }
        }
        
        // 缩放
        if self.with_std {
            if let Some(ref std_dev) = self.std_dev {
                for (i, val) in result.iter_mut().enumerate() {
                    if std_dev[i] > 0.0 {
                        *val /= std_dev[i];
                    }
                }
            }
        }
        
        Ok(result)
    }
    
    fn name(&self) -> &str {
        &self.name
    }
    
    fn description(&self) -> &str {
        &self.description
    }
}

/// 缩放变换器(Min-Max缩放)
pub struct MinMaxScalerTransformer {
    /// 变换器名称
    name: String,
    /// 变换器描述
    description: String,
    /// 最小值向量
    min_values: Option<Vec<f32>>,
    /// 最大值向量
    max_values: Option<Vec<f32>>,
    /// 目标范围下限
    feature_range_min: f32,
    /// 目标范围上限
    feature_range_max: f32,
}

impl MinMaxScalerTransformer {
    /// 创建新的Min-Max缩放变换器
    pub fn new(
        name: &str, 
        description: &str, 
        feature_range_min: f32,
        feature_range_max: f32
    ) -> Self {
        Self {
            name: name.to_string(),
            description: description.to_string(),
            min_values: None,
            max_values: None,
            feature_range_min,
            feature_range_max,
        }
    }
    
    /// 创建默认的Min-Max缩放变换器(缩放到[0,1]范围)
    pub fn default() -> Self {
        Self::new(
            "minmax_scaler",
            "Min-Max scaling transformer",
            0.0,
            1.0
        )
    }
    
    /// 拟合数据，计算每个特征的最小值和最大值
    pub fn fit(&mut self, vectors: &[Vec<f32>]) -> Result<()> {
        if vectors.is_empty() {
            return Err(Error::vector("Cannot fit transformer on empty vector list".to_string()));
        }
        
        let dim = vectors[0].len();
        // 验证所有向量维度一致
        for (i, vec) in vectors.iter().enumerate().skip(1) {
            if vec.len() != dim {
                return Err(Error::vector(format!(
                    "Vector at index {} has incorrect dimension: expected {}, got {}",
                    i, dim, vec.len()
                )));
            }
        }
        
        // 找出每个维度的最小值和最大值
        let mut min_values = vectors[0].clone();
        let mut max_values = vectors[0].clone();
        
        for vec in vectors.iter().skip(1) {
            for (i, &val) in vec.iter().enumerate() {
                min_values[i] = min_values[i].min(val);
                max_values[i] = max_values[i].max(val);
            }
        }
        
        self.min_values = Some(min_values);
        self.max_values = Some(max_values);
        
        Ok(())
    }
}

impl VectorTransformer for MinMaxScalerTransformer {
    fn transform(&self, vectors: &[Vec<f32>]) -> Result<Vec<Vec<f32>>> {
        if vectors.is_empty() {
            return Ok(Vec::new());
        }
        
        // 验证所有向量维度一致
        let dim = vectors[0].len();
        for (i, vec) in vectors.iter().enumerate().skip(1) {
            if vec.len() != dim {
                return Err(Error::vector(format!(
                    "Vector at index {} has incorrect dimension: expected {}, got {}",
                    i, dim, vec.len()
                )));
            }
        }
        
        // 验证是否已经拟合
        if self.min_values.is_none() || self.max_values.is_none() {
            return Err(Error::vector("MinMaxScalerTransformer not fitted".to_string()));
        }
        
        // 并行转换每个向量
        let transformed = vectors.par_iter()
            .map(|vec| self.transform_one(vec))
            .collect::<Result<Vec<Vec<f32>>>>()?;
        
        Ok(transformed)
    }
    
    fn transform_one(&self, vector: &[f32]) -> Result<Vec<f32>> {
        if vector.is_empty() {
            return Ok(Vec::new());
        }
        
        // 获取最小值和最大值
        let min_values = self.min_values.as_ref()
            .ok_or_else(|| Error::vector("MinMaxScalerTransformer not fitted".to_string()))?;
            
        let max_values = self.max_values.as_ref()
            .ok_or_else(|| Error::vector("MinMaxScalerTransformer not fitted".to_string()))?;
        
        // 验证向量维度
        if vector.len() != min_values.len() {
            return Err(Error::vector(format!(
                "Vector dimension mismatch: expected {}, got {}",
                min_values.len(), vector.len()
            )));
        }
        
        // 计算缩放范围
        let scale = self.feature_range_max - self.feature_range_min;
        
        // 应用变换
        let mut result = Vec::with_capacity(vector.len());
        
        for i in 0..vector.len() {
            let range = max_values[i] - min_values[i];
            
            let scaled = if range > 1e-10 {
                // 标准缩放公式: (x - min) / (max - min) * scale + min_range
                (vector[i] - min_values[i]) / range * scale + self.feature_range_min
            } else {
                // 如果范围接近零，直接使用特征范围的中间值
                (self.feature_range_min + self.feature_range_max) / 2.0
            };
            
            result.push(scaled);
        }
        
        Ok(result)
    }
    
    fn name(&self) -> &str {
        &self.name
    }
    
    fn description(&self) -> &str {
        &self.description
    }
}

/// 加权变换器
pub struct WeightedTransformer {
    /// 变换器名称
    name: String,
    /// 变换器描述
    description: String,
    /// 权重向量
    weights: Vec<f32>,
}

impl WeightedTransformer {
    /// 创建新的加权变换器
    pub fn new(name: &str, description: &str, weights: Vec<f32>) -> Self {
        Self {
            name: name.to_string(),
            description: description.to_string(),
            weights,
        }
    }
    
    /// 创建均匀权重的变换器
    pub fn uniform(dim: usize) -> Self {
        Self::new(
            "uniform_weighted",
            "Uniform weighted transformer",
            vec![1.0; dim]
        )
    }
}

impl VectorTransformer for WeightedTransformer {
    fn transform(&self, vectors: &[Vec<f32>]) -> Result<Vec<Vec<f32>>> {
        if vectors.is_empty() {
            return Ok(Vec::new());
        }
        
        // 验证所有向量维度一致
        let dim = vectors[0].len();
        for (i, vec) in vectors.iter().enumerate().skip(1) {
            if vec.len() != dim {
                return Err(Error::vector(format!(
                    "Vector at index {} has incorrect dimension: expected {}, got {}",
                    i, dim, vec.len()
                )));
            }
        }
        
        // 验证权重向量维度
        if self.weights.len() != dim {
            return Err(Error::vector(format!(
                "Weight vector dimension mismatch: expected {}, got {}",
                dim, self.weights.len()
            )));
        }
        
        // 并行转换每个向量
        let transformed = vectors.par_iter()
            .map(|vec| self.transform_one(vec))
            .collect::<Result<Vec<Vec<f32>>>>()?;
        
        Ok(transformed)
    }
    
    fn transform_one(&self, vector: &[f32]) -> Result<Vec<f32>> {
        if vector.is_empty() {
            return Ok(Vec::new());
        }
        
        // 验证向量维度
        if vector.len() != self.weights.len() {
            return Err(Error::vector(format!(
                "Vector dimension mismatch: expected {}, got {}",
                self.weights.len(), vector.len()
            )));
        }
        
        // 应用权重
        let weighted = vector.iter()
            .zip(self.weights.iter())
            .map(|(&val, &weight)| val * weight)
            .collect();
            
        Ok(weighted)
    }
    
    fn name(&self) -> &str {
        &self.name
    }
    
    fn description(&self) -> &str {
        &self.description
    }
}

/// 复合变换器，按顺序应用多个变换器
pub struct PipelineTransformer {
    /// 变换器名称
    name: String,
    /// 变换器描述
    description: String,
    /// 变换器列表
    transformers: Vec<Box<dyn VectorTransformer>>,
}

impl PipelineTransformer {
    /// 创建新的复合变换器
    pub fn new(name: &str, description: &str) -> Self {
        Self {
            name: name.to_string(),
            description: description.to_string(),
            transformers: Vec::new(),
        }
    }
    
    /// 添加变换器
    pub fn add_transformer<T: VectorTransformer + 'static>(&mut self, transformer: T) {
        self.transformers.push(Box::new(transformer));
    }
}

impl VectorTransformer for PipelineTransformer {
    fn transform(&self, vectors: &[Vec<f32>]) -> Result<Vec<Vec<f32>>> {
        if vectors.is_empty() {
            return Ok(Vec::new());
        }
        
        let mut result = vectors.to_vec();
        
        // 依次应用每个变换器
        for transformer in &self.transformers {
            result = transformer.transform(&result)?;
        }
        
        Ok(result)
    }
    
    fn transform_one(&self, vector: &[f32]) -> Result<Vec<f32>> {
        if vector.is_empty() {
            return Ok(Vec::new());
        }
        
        let mut result = vector.to_vec();
        
        // 依次应用每个变换器
        for transformer in &self.transformers {
            result = transformer.transform_one(&result)?;
        }
        
        Ok(result)
    }
    
    fn name(&self) -> &str {
        &self.name
    }
    
    fn description(&self) -> &str {
        &self.description
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_normalizer_l2() {
        // 创建测试数据
        let vectors = vec![
            vec![3.0, 4.0],
            vec![1.0, 2.0],
            vec![5.0, 12.0],
        ];
        
        let transformer = NormalizerTransformer::default();
        let normalized = transformer.transform(&vectors).unwrap();
        
        assert_eq!(normalized.len(), vectors.len());
        
        // 检查第一个向量 [3,4] -> [3/5, 4/5]
        assert!((normalized[0][0] - 0.6).abs() < 1e-6);
        assert!((normalized[0][1] - 0.8).abs() < 1e-6);
        
        // 检查第二个向量 [1,2] -> [1/sqrt(5), 2/sqrt(5)]
        let norm = 5.0_f32.sqrt();
        assert!((normalized[1][0] - 1.0/norm).abs() < 1e-6);
        assert!((normalized[1][1] - 2.0/norm).abs() < 1e-6);
        
        // 检查第三个向量 [5,12] -> [5/13, 12/13]
        assert!((normalized[2][0] - 5.0/13.0).abs() < 1e-6);
        assert!((normalized[2][1] - 12.0/13.0).abs() < 1e-6);
    }
    
    #[test]
    fn test_standard_scaler() {
        // 创建测试数据
        let vectors = vec![
            vec![1.0, 2.0, 3.0],
            vec![4.0, 5.0, 6.0],
            vec![7.0, 8.0, 9.0],
        ];
        
        let mut transformer = StandardScalerTransformer::default();
        transformer.fit(&vectors).unwrap();
        
        let transformed = transformer.transform(&vectors).unwrap();
        
        assert_eq!(transformed.len(), vectors.len());
        
        // 验证变换后的均值接近0，标准差接近1
        let mut means = vec![0.0; 3];
        let mut variances = vec![0.0; 3];
        let n = transformed.len() as f32;
        
        for vec in &transformed {
            for (i, &val) in vec.iter().enumerate() {
                means[i] += val / n;
            }
        }
        
        for vec in &transformed {
            for (i, &val) in vec.iter().enumerate() {
                let diff = val - means[i];
                variances[i] += (diff * diff) / n;
            }
        }
        
        // 验证均值接近0
        for mean in &means {
            assert!(mean.abs() < 1e-6);
        }
        
        // 验证方差接近1
        for var in &variances {
            assert!((var - 1.0).abs() < 1e-6);
        }
    }
    
    #[test]
    fn test_minmax_scaler() {
        // 创建测试数据
        let vectors = vec![
            vec![1.0, 2.0, 3.0],
            vec![4.0, 5.0, 6.0],
            vec![7.0, 8.0, 9.0],
        ];
        
        let mut transformer = MinMaxScalerTransformer::default();
        transformer.fit(&vectors).unwrap();
        
        let transformed = transformer.transform(&vectors).unwrap();
        
        assert_eq!(transformed.len(), vectors.len());
        
        // 检查每个维度是否都缩放到[0,1]范围
        for i in 0..3 {
            // 第一个向量的值应该是最小的 -> 0.0
            assert!(transformed[0][i].abs() < 1e-6);
            
            // 最后一个向量的值应该是最大的 -> 1.0
            assert!((transformed[2][i] - 1.0).abs() < 1e-6);
            
            // 中间向量应该在0.5附近
            assert!((transformed[1][i] - 0.5).abs() < 1e-6);
        }
    }
    
    #[test]
    fn test_weighted_transformer() {
        // 创建测试数据
        let vectors = vec![
            vec![1.0, 2.0, 3.0],
            vec![4.0, 5.0, 6.0],
        ];
        
        let weights = vec![0.5, 1.0, 2.0];
        let transformer = WeightedTransformer::new("test", "test", weights);
        
        let transformed = transformer.transform(&vectors).unwrap();
        
        assert_eq!(transformed.len(), vectors.len());
        
        // 检查权重是否正确应用
        assert!((transformed[0][0] - 0.5).abs() < 1e-6); // 1.0 * 0.5
        assert!((transformed[0][1] - 2.0).abs() < 1e-6); // 2.0 * 1.0
        assert!((transformed[0][2] - 6.0).abs() < 1e-6); // 3.0 * 2.0
        
        assert!((transformed[1][0] - 2.0).abs() < 1e-6); // 4.0 * 0.5
        assert!((transformed[1][1] - 5.0).abs() < 1e-6); // 5.0 * 1.0
        assert!((transformed[1][2] - 12.0).abs() < 1e-6); // 6.0 * 2.0
    }
    
    #[test]
    fn test_pipeline_transformer() {
        // 创建测试数据
        let vectors = vec![
            vec![1.0, 2.0],
            vec![4.0, 5.0],
        ];
        
        // 创建一个变换管道：先缩放到[0,1]，再应用权重，最后归一化
        let mut minmax = MinMaxScalerTransformer::default();
        minmax.fit(&vectors).unwrap();
        
        let weights = vec![0.5, 2.0];
        let weighted = WeightedTransformer::new("weighted", "weighted", weights);
        
        let normalizer = NormalizerTransformer::default();
        
        let mut pipeline = PipelineTransformer::new("pipeline", "pipeline");
        pipeline.add_transformer(minmax);
        pipeline.add_transformer(weighted);
        pipeline.add_transformer(normalizer);
        
        let transformed = pipeline.transform(&vectors).unwrap();
        
        assert_eq!(transformed.len(), vectors.len());
        
        // 验证管道变换结果
        // 第一个向量 [1,2] -> [0,0] (MinMax) -> [0,0] (Weight) -> [0,0] (或不确定，因为归一化零向量不确定)
        // 第二个向量 [4,5] -> [1,1] (MinMax) -> [0.5,2] (Weight) -> [0.5/sqrt(0.5^2+2^2), 2/sqrt(0.5^2+2^2)]
        
        // 只验证第二个向量
        let norm = (0.5_f32.powi(2) + 2.0_f32.powi(2)).sqrt();
        assert!((transformed[1][0] - 0.5/norm).abs() < 1e-6);
        assert!((transformed[1][1] - 2.0/norm).abs() < 1e-6);
    }
} 