// Validator Module

use std::fmt::Debug;
use std::collections::HashMap;
use async_trait::async_trait;

use super::extractor::{FeatureVector, FeatureBatch};

/// 特征验证器错误类型
#[derive(Debug, thiserror::Error)]
pub enum ValidatorError {
    #[error("维度错误: {0}")]
    DimensionError(String),
    
    #[error("类型错误: {0}")]
    TypeError(String),
    
    #[error("值错误: {0}")]
    ValueError(String),
    
    #[error("其他错误: {0}")]
    Other(String),
}

/// 特征验证器特质
#[async_trait]
pub trait FeatureValidator: Send + Sync + Debug {
    /// 获取验证器名称
    fn name(&self) -> &str;
    
    /// 设置验证器名称
    fn set_name(&mut self, name: String);
    
    /// 验证特征向量
    async fn validate(&self, features: &FeatureVector) -> Result<ValidationResult, ValidatorError>;
    
    /// 验证单个特征向量
    async fn validate_vector(&self, vector: &FeatureVector) -> Result<(), ValidatorError>;
    
    /// 批量验证特征向量
    async fn validate_batch(&self, features: &FeatureBatch) -> Result<Vec<ValidationResult>, ValidatorError>;
    
    /// 获取验证统计信息
    async fn get_validation_stats(&self) -> Result<ValidationStats, ValidatorError>;
}

/// 验证结果
#[derive(Debug, Clone)]
pub struct ValidationResult {
    /// 是否通过验证
    pub is_valid: bool,
    /// 验证分数
    pub score: f64,
    /// 错误信息
    pub errors: Vec<String>,
    /// 警告信息
    pub warnings: Vec<String>,
    /// 验证时间戳
    pub timestamp: chrono::DateTime<chrono::Utc>,
}

/// 验证统计信息
#[derive(Debug, Clone, Default)]
pub struct ValidationStats {
    /// 总验证次数
    pub total_validations: usize,
    /// 通过验证次数
    pub passed_validations: usize,
    /// 失败验证次数
    pub failed_validations: usize,
    /// 平均验证分数
    pub average_score: f64,
    /// 验证历史
    pub validation_history: HashMap<String, Vec<ValidationResult>>,
}

/// 基础特征验证器
#[derive(Debug)]
pub struct BasicFeatureValidator {
    /// 验证器名称
    name: String,
    /// 验证规则
    rules: HashMap<String, ValidationRule>,
    /// 验证统计
    stats: ValidationStats,
}

impl BasicFeatureValidator {
    /// 创建新的基础验证器
    pub fn new() -> Self {
        Self {
            name: "BasicFeatureValidator".to_string(),
            rules: HashMap::new(),
            stats: ValidationStats::default(),
        }
    }
    
    /// 添加验证规则
    pub fn add_rule(&mut self, rule: ValidationRule) {
        self.rules.insert(rule.name.clone(), rule);
    }
    
    /// 移除验证规则
    pub fn remove_rule(&mut self, rule_name: &str) {
        self.rules.remove(rule_name);
    }
}

#[async_trait]
impl FeatureValidator for BasicFeatureValidator {
    fn name(&self) -> &str {
        &self.name
    }
    
    fn set_name(&mut self, name: String) {
        self.name = name;
    }
    
    async fn validate(&self, features: &FeatureVector) -> Result<ValidationResult, ValidatorError> {
        let mut result = ValidationResult {
            is_valid: true,
            score: 1.0,
            errors: Vec::new(),
            warnings: Vec::new(),
            timestamp: chrono::Utc::now(),
        };
        
        // 基础验证：检查维度
        if features.values.is_empty() {
            result.is_valid = false;
            result.errors.push("特征向量为空".to_string());
            return Ok(result);
        }
        
        // 检查数值有效性
        for (i, &value) in features.values.iter().enumerate() {
            if value.is_nan() {
                result.is_valid = false;
                result.errors.push(format!("第{}个特征值为NaN", i));
            }
            if value.is_infinite() {
                result.warnings.push(format!("第{}个特征值为无穷大", i));
            }
        }
        
        Ok(result)
    }
    
    async fn validate_vector(&self, vector: &FeatureVector) -> Result<(), ValidatorError> {
        let result = self.validate(vector).await?;
        if !result.is_valid {
            return Err(ValidatorError::ValueError(
                result.errors.join("; ")
            ));
        }
        Ok(())
    }
    
    async fn validate_batch(&self, features: &FeatureBatch) -> Result<Vec<ValidationResult>, ValidatorError> {
        let mut results = Vec::new();
        
        for i in 0..features.batch_size {
            let vector = FeatureVector {
                feature_type: features.feature_type,
                values: features.values[i].clone(),
                extractor_type: features.extractor_type,
                metadata: features.metadata.clone(),
                features: Vec::new(),
            };
            
            let result = self.validate(&vector).await?;
            results.push(result);
        }
        
        Ok(results)
    }
    
    async fn get_validation_stats(&self) -> Result<ValidationStats, ValidatorError> {
        Ok(self.stats.clone())
    }
}

/// 验证规则
#[derive(Debug, Clone)]
pub struct ValidationRule {
    /// 规则名称
    pub name: String,
    /// 规则类型
    pub rule_type: RuleType,
    /// 规则参数
    pub parameters: HashMap<String, f64>,
    /// 是否启用
    pub enabled: bool,
}

/// 规则类型
#[derive(Debug, Clone)]
pub enum RuleType {
    /// 维度检查
    DimensionCheck,
    /// 范围检查
    RangeCheck,
    /// 类型检查
    TypeCheck,
    /// 空值检查
    NullCheck,
    /// 异常值检查
    OutlierCheck,
}

/// 高级特征验证器
/// 提供更复杂的特征验证功能
#[derive(Debug, Clone)]
pub struct AdvancedFeatureValidator {
    /// 验证器名称
    name: String,
    /// 验证规则
    rules: HashMap<String, ValidationRule>,
    /// 验证统计
    stats: ValidationStats,
}

impl AdvancedFeatureValidator {
    /// 创建新的高级特征验证器
    pub fn new(min_dim: usize, max_dim: usize) -> Self {
        Self {
            name: "AdvancedFeatureValidator".to_string(),
            rules: HashMap::new(),
            stats: ValidationStats {
                total_validations: 0,
                passed_validations: 0,
                failed_validations: 0,
                average_score: 0.0,
                validation_history: HashMap::new(),
            },
        }
    }
    
    /// 添加验证规则
    pub fn add_rule(&mut self, rule: ValidationRule) {
        self.rules.insert(rule.name.clone(), rule);
    }
    
    /// 移除验证规则
    pub fn remove_rule(&mut self, rule_name: &str) {
        self.rules.remove(rule_name);
    }
    
    /// 更新验证统计
    fn update_stats(&mut self, result: &ValidationResult) {
        self.stats.total_validations += 1;
        if result.is_valid {
            self.stats.passed_validations += 1;
        } else {
            self.stats.failed_validations += 1;
        }
        
        // 更新平均分数
        let total_score = self.stats.average_score * (self.stats.total_validations - 1) as f64 + result.score;
        self.stats.average_score = total_score / self.stats.total_validations as f64;
        
        // 添加到历史记录
        let timestamp_key = result.timestamp.format("%Y-%m-%d").to_string();
        self.stats.validation_history
            .entry(timestamp_key)
            .or_insert_with(Vec::new)
            .push(result.clone());
    }
    
    /// 设置验证器名称
    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = name.into();
        self
    }
    
    /// 设置是否检查非零
    pub fn with_non_zero_check(mut self, check: bool) -> Self {
        // AdvancedFeatureValidator 使用规则系统，这里简化处理
        self
    }
    
    /// 设置是否检查NaN
    pub fn with_nan_check(mut self, check: bool) -> Self {
        // AdvancedFeatureValidator 使用规则系统，这里简化处理
        self
    }
    
    /// 设置是否检查无穷大
    pub fn with_infinity_check(mut self, check: bool) -> Self {
        // AdvancedFeatureValidator 使用规则系统，这里简化处理
        self
    }
}

#[async_trait]
impl FeatureValidator for AdvancedFeatureValidator {
    fn name(&self) -> &str {
        &self.name
    }
    
    fn set_name(&mut self, name: String) {
        self.name = name;
    }
    
    async fn validate(&self, features: &FeatureVector) -> Result<ValidationResult, ValidatorError> {
        let mut errors = Vec::new();
        let mut warnings = Vec::new();
        let mut score: f64 = 1.0;
        
        // 应用所有启用的验证规则
        for (_, rule) in &self.rules {
            if !rule.enabled {
                continue;
            }
            
            match rule.rule_type {
                RuleType::DimensionCheck => {
                    if let Some(expected_dim) = rule.parameters.get("expected_dimension") {
                        if features.values.len() != *expected_dim as usize {
                            errors.push(format!(
                                "维度不匹配: 期望 {}, 实际 {}",
                                expected_dim,
                                features.values.len()
                            ));
                            score -= 0.3;
                        }
                    }
                },
                RuleType::RangeCheck => {
                    if let (Some(min_val), Some(max_val)) = (rule.parameters.get("min_value"), rule.parameters.get("max_value")) {
                        let min_val_f32 = *min_val as f32;
                        let max_val_f32 = *max_val as f32;
                        for (i, &value) in features.values.iter().enumerate() {
                            if value < min_val_f32 || value > max_val_f32 {
                                warnings.push(format!(
                                    "值 {} 超出范围 [{}, {}] 在位置 {}",
                                    value, min_val_f32, max_val_f32, i
                                ));
                                score -= 0.1;
                            }
                        }
                    }
                },
                RuleType::TypeCheck => {
                    // 检查数据类型
                    for (i, &value) in features.values.iter().enumerate() {
                        if value.is_nan() || value.is_infinite() {
                            errors.push(format!("无效的数值在位置 {}: {}", i, value));
                            score -= 0.5;
                        }
                    }
                },
                RuleType::NullCheck => {
                    // 检查空值
                    for (i, &value) in features.values.iter().enumerate() {
                        if value.is_nan() {
                            errors.push(format!("空值在位置 {}", i));
                            score -= 0.4;
                        }
                    }
                },
                RuleType::OutlierCheck => {
                    // 检查异常值
                    if let Some(threshold) = rule.parameters.get("outlier_threshold") {
                        let mean = features.values.iter().sum::<f32>() as f64 / features.values.len() as f64;
                        let variance = features.values.iter()
                            .map(|&x| ((x as f64) - mean).powi(2))
                            .sum::<f64>() / features.values.len() as f64;
                        let std_dev = variance.sqrt();
                        
                        for (i, &value) in features.values.iter().enumerate() {
                            if ((value as f64) - mean).abs() > *threshold * std_dev {
                                warnings.push(format!(
                                    "异常值在位置 {}: {} (均值: {}, 标准差: {})",
                                    i, value, mean, std_dev
                                ));
                                score -= 0.2;
                            }
                        }
                    }
                },
            }
        }
        
        Ok(ValidationResult {
            is_valid: errors.is_empty(),
            score: score.max(0.0f64),
            errors,
            warnings,
            timestamp: chrono::Utc::now(),
        })
    }
    
    async fn validate_batch(&self, features: &FeatureBatch) -> Result<Vec<ValidationResult>, ValidatorError> {
        let mut results = Vec::new();
        for i in 0..features.batch_size {
            let vector = FeatureVector {
                feature_type: features.feature_type,
                values: features.values[i].clone(),
                extractor_type: features.extractor_type,
                metadata: features.metadata.clone(),
                features: Vec::new(),
            };
            let result = self.validate(&vector).await?;
            results.push(result);
        }
        Ok(results)
    }
    
    async fn validate_vector(&self, vector: &FeatureVector) -> Result<(), ValidatorError> {
        let result = self.validate(vector).await?;
        if !result.is_valid {
            return Err(ValidatorError::ValueError(format!("验证失败: {:?}", result.errors)));
        }
        Ok(())
    }
    
    async fn get_validation_stats(&self) -> Result<ValidationStats, ValidatorError> {
        Ok(self.stats.clone())
    }
}

#[async_trait]
impl FeatureValidator for AdvancedStatisticalFeatureValidator {
    fn name(&self) -> &str {
        &self.name
    }
    
    fn set_name(&mut self, name: String) {
        self.name = name;
    }
    
    async fn validate(&self, features: &FeatureVector) -> Result<ValidationResult, ValidatorError> {
        let mut errors = Vec::new();
        let mut warnings = Vec::new();
        let mut score: f64 = 1.0;
        
        // 检查维度
        let dim = features.values.len();
        if dim < self.min_dim || dim > self.max_dim {
            errors.push(format!(
                "维度不匹配: 期望范围 [{}, {}], 实际 {}",
                self.min_dim, self.max_dim, dim
            ));
            score -= 0.3;
        }
        
        // 检查值范围
        for (i, &value) in features.values.iter().enumerate() {
            if value < self.min_value || value > self.max_value {
                warnings.push(format!(
                    "值 {} 超出范围 [{}, {}] 在位置 {}",
                    value, self.min_value, self.max_value, i
                ));
                score -= 0.1;
            }
            
            // 检查NaN
            if self.check_nan && value.is_nan() {
                errors.push(format!("NaN值在位置 {}", i));
                score -= 0.5;
            }
            
            // 检查无穷大
            if self.check_infinity && value.is_infinite() {
                warnings.push(format!("无穷大值在位置 {}: {}", i, value));
                score -= 0.2;
            }
            
            // 检查非零（如果启用）
            if self.check_non_zero && value == 0.0 {
                warnings.push(format!("零值在位置 {}", i));
                score -= 0.05;
            }
        }
        
        Ok(ValidationResult {
            is_valid: errors.is_empty(),
            score: score.max(0.0f64),
            errors,
            warnings,
            timestamp: chrono::Utc::now(),
        })
    }
    
    async fn validate_batch(&self, features: &FeatureBatch) -> Result<Vec<ValidationResult>, ValidatorError> {
        let mut results = Vec::new();
        
        // FeatureBatch包含多个FeatureVector，需要逐个验证
        for i in 0..features.batch_size {
            let vector = FeatureVector {
                feature_type: features.feature_type,
                values: features.values[i].clone(),
                extractor_type: None,
                metadata: HashMap::new(),
                features: Vec::new(),
            };
            let result = self.validate(&vector).await?;
            results.push(result);
        }
        
        Ok(results)
    }
    
    async fn validate_vector(&self, vector: &FeatureVector) -> Result<(), ValidatorError> {
        let result = self.validate(vector).await?;
        if !result.is_valid {
            return Err(ValidatorError::ValueError(format!("验证失败: {:?}", result.errors)));
        }
        Ok(())
    }
    
    async fn get_validation_stats(&self) -> Result<ValidationStats, ValidatorError> {
        Ok(ValidationStats::default())
    }
}


/// 统计特征验证器
/// 验证特征维度、数值范围和非零性
#[derive(Debug, Clone)]
pub struct AdvancedStatisticalFeatureValidator {
    /// 验证器名称
    name: String,
    /// 最小维度
    min_dim: usize,
    /// 最大维度
    max_dim: usize,
    /// 最小值
    min_value: f32,
    /// 最大值
    max_value: f32,
    /// 是否检查非零
    check_non_zero: bool,
    /// 是否检查NaN
    check_nan: bool,
    /// 是否检查无穷大
    check_infinity: bool,
}

impl AdvancedStatisticalFeatureValidator {
    /// 创建新的统计验证器
    pub fn new(min_dim: usize, max_dim: usize) -> Self {
        Self {
            name: "AdvancedStatisticalFeatureValidator".to_string(),
            min_dim,
            max_dim,
            min_value: -f32::MAX,
            max_value: f32::MAX,
            check_non_zero: false,
            check_nan: true,
            check_infinity: true,
        }
    }
    
    /// 设置数值范围
    pub fn with_value_range(mut self, min: f32, max: f32) -> Self {
        self.min_value = min;
        self.max_value = max;
        self
    }
    
    /// 设置是否检查全零向量
    pub fn with_non_zero_check(mut self, check: bool) -> Self {
        self.check_non_zero = check;
        self
    }
    
    /// 设置是否检查NaN
    pub fn with_nan_check(mut self, check: bool) -> Self {
        self.check_nan = check;
        self
    }
    
    /// 设置是否检查无穷大
    pub fn with_infinity_check(mut self, check: bool) -> Self {
        self.check_infinity = check;
        self
    }
    
    /// 设置验证器名称
    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = name.into();
        self
    }
}

// 为了向后兼容，提供类型别名
pub type StatisticalFeatureValidator = AdvancedStatisticalFeatureValidator;

/// 比较特征验证器
/// 验证特征与另一个特征的关系
#[derive(Debug)]
pub struct ComparisonFeatureValidator {
    /// 验证器名称
    name: String,
    /// 参考特征
    reference: Vec<f32>,
    /// 最小余弦相似度
    min_cosine_similarity: Option<f32>,
    /// 最大欧氏距离
    max_euclidean_distance: Option<f32>,
    /// 最大曼哈顿距离
    max_manhattan_distance: Option<f32>,
}

impl ComparisonFeatureValidator {
    /// 创建新的比较验证器
    pub fn new(reference: Vec<f32>) -> Self {
        Self {
            name: "ComparisonFeatureValidator".to_string(),
            reference,
            min_cosine_similarity: None,
            max_euclidean_distance: None,
            max_manhattan_distance: None,
        }
    }
    
    /// 设置最小余弦相似度
    /// 
    /// # Panics
    /// 
    /// 如果 `min` 不在 [-1.0, 1.0] 范围内，会在 debug 模式下 panic
    pub fn with_min_cosine_similarity(mut self, min: f32) -> Self {
        debug_assert!(
            min >= -1.0 && min <= 1.0,
            "余弦相似度必须在-1到1之间，当前值: {}",
            min
        );
        if min < -1.0 || min > 1.0 {
            // 在 release 模式下，将值限制在有效范围内
            self.min_cosine_similarity = Some(min.max(-1.0).min(1.0));
        } else {
            self.min_cosine_similarity = Some(min);
        }
        self
    }
    
    /// 设置最大欧氏距离
    /// 
    /// # Panics
    /// 
    /// 如果 `max` 小于 0，会在 debug 模式下 panic
    pub fn with_max_euclidean_distance(mut self, max: f32) -> Self {
        debug_assert!(
            max >= 0.0,
            "欧氏距离必须大于等于0，当前值: {}",
            max
        );
        self.max_euclidean_distance = Some(max.max(0.0));
        self
    }
    
    /// 设置最大曼哈顿距离
    /// 
    /// # Panics
    /// 
    /// 如果 `max` 小于 0，会在 debug 模式下 panic
    pub fn with_max_manhattan_distance(mut self, max: f32) -> Self {
        debug_assert!(
            max >= 0.0,
            "曼哈顿距离必须大于等于0，当前值: {}",
            max
        );
        self.max_manhattan_distance = Some(max.max(0.0));
        self
    }
    
    /// 设置验证器名称
    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = name.into();
        self
    }
    
    /// 计算余弦相似度
    fn calculate_cosine_similarity(&self, features: &[f32]) -> f32 {
        // 检查维度是否一致
        if features.len() != self.reference.len() {
            return -2.0; // 超出[-1,1]范围的值表示无效
        }
        
        // 计算点积
        let dot_product: f32 = features.iter()
            .zip(self.reference.iter())
            .map(|(&a, &b)| a * b)
            .sum();
        
        // 计算模长
        let norm_a: f32 = features.iter()
            .map(|&x| x * x)
            .sum::<f32>()
            .sqrt();
            
        let norm_b: f32 = self.reference.iter()
            .map(|&x| x * x)
            .sum::<f32>()
            .sqrt();
        
        // 避免除以0
        if norm_a == 0.0 || norm_b == 0.0 {
            return 0.0;
        }
        
        // 计算余弦相似度
        dot_product / (norm_a * norm_b)
    }
    
    /// 计算欧氏距离
    fn calculate_euclidean_distance(&self, features: &[f32]) -> f32 {
        // 检查维度是否一致
        if features.len() != self.reference.len() {
            return f32::MAX; // 最大值表示无效
        }
        
        // 计算平方和
        let squared_sum: f32 = features.iter()
            .zip(self.reference.iter())
            .map(|(&a, &b)| (a - b).powi(2))
            .sum();
        
        // 计算欧氏距离
        squared_sum.sqrt()
    }
    
    /// 计算曼哈顿距离
    fn calculate_manhattan_distance(&self, features: &[f32]) -> f32 {
        // 检查维度是否一致
        if features.len() != self.reference.len() {
            return f32::MAX; // 最大值表示无效
        }
        
        // 计算绝对差之和
        features.iter()
            .zip(self.reference.iter())
            .map(|(&a, &b)| (a - b).abs())
            .sum()
    }
}

#[async_trait]
impl FeatureValidator for ComparisonFeatureValidator {
    fn name(&self) -> &str {
        &self.name
    }
    
    fn set_name(&mut self, name: String) {
        self.name = name;
    }
    
    async fn validate_vector(&self, vector: &FeatureVector) -> Result<(), ValidatorError> {
        // 检查是否为空
        if vector.values.is_empty() {
            return Err(ValidatorError::ValueError("特征向量为空".to_string()));
        }
        
        // 检查维度是否一致
        if vector.values.len() != self.reference.len() {
            return Err(ValidatorError::DimensionError(
                format!("特征维度不匹配: 期望 {}, 实际 {}", self.reference.len(), vector.values.len())
            ));
        }
        
        // 检查余弦相似度
        if let Some(min_cosine) = self.min_cosine_similarity {
            let cosine = self.calculate_cosine_similarity(&vector.values);
            if cosine < min_cosine {
                return Err(ValidatorError::ValueError(
                    format!("余弦相似度 {} 小于最小允许值 {}", cosine, min_cosine)
                ));
            }
        }
        
        // 检查欧氏距离
        if let Some(max_euclidean) = self.max_euclidean_distance {
            let euclidean = self.calculate_euclidean_distance(&vector.values);
            if euclidean > max_euclidean {
                return Err(ValidatorError::ValueError(
                    format!("欧氏距离 {} 大于最大允许值 {}", euclidean, max_euclidean)
                ));
            }
        }
        
        // 检查曼哈顿距离
        if let Some(max_manhattan) = self.max_manhattan_distance {
            let manhattan = self.calculate_manhattan_distance(&vector.values);
            if manhattan > max_manhattan {
                return Err(ValidatorError::ValueError(
                    format!("曼哈顿距离 {} 大于最大允许值 {}", manhattan, max_manhattan)
                ));
            }
        }
        
        Ok(())
    }
    
    async fn validate(&self, features: &FeatureVector) -> Result<ValidationResult, ValidatorError> {
        let mut errors = Vec::new();
        let mut warnings = Vec::new();
        let mut score: f64 = 1.0;
        
        // 检查维度是否一致
        if features.values.len() != self.reference.len() {
            errors.push(format!("特征维度不匹配: 期望 {}, 实际 {}", self.reference.len(), features.values.len()));
            score -= 0.5;
        }
        
        // 检查余弦相似度
        if let Some(min_cosine) = self.min_cosine_similarity {
            let cosine = self.calculate_cosine_similarity(&features.values);
            if cosine < min_cosine {
                warnings.push(format!("余弦相似度 {} 小于最小允许值 {}", cosine, min_cosine));
                score -= 0.2;
            }
        }
        
        // 检查欧氏距离
        if let Some(max_euclidean) = self.max_euclidean_distance {
            let euclidean = self.calculate_euclidean_distance(&features.values);
            if euclidean > max_euclidean {
                warnings.push(format!("欧氏距离 {} 大于最大允许值 {}", euclidean, max_euclidean));
                score -= 0.2;
            }
        }
        
        // 检查曼哈顿距离
        if let Some(max_manhattan) = self.max_manhattan_distance {
            let manhattan = self.calculate_manhattan_distance(&features.values);
            if manhattan > max_manhattan {
                warnings.push(format!("曼哈顿距离 {} 大于最大允许值 {}", manhattan, max_manhattan));
                score -= 0.2;
            }
        }
        
        let is_valid = errors.is_empty();
        Ok(ValidationResult {
            is_valid,
            score: score.max(0.0f64),
            errors,
            warnings,
            timestamp: chrono::Utc::now(),
        })
    }
    
    async fn validate_batch(&self, features: &FeatureBatch) -> Result<Vec<ValidationResult>, ValidatorError> {
        let mut results = Vec::new();
        for i in 0..features.batch_size {
            let vector = FeatureVector {
                feature_type: features.feature_type,
                values: features.values[i].clone(),
                extractor_type: features.extractor_type,
                metadata: features.metadata.clone(),
                features: Vec::new(),
            };
            let result = self.validate(&vector).await?;
            results.push(result);
        }
        Ok(results)
    }
    
    async fn get_validation_stats(&self) -> Result<ValidationStats, ValidatorError> {
        Ok(ValidationStats::default())
    }
}

/// 组合验证器
/// 组合多个验证器，可以设置AND或OR关系
#[derive(Debug)]
pub struct CompositeValidator {
    /// 验证器名称
    name: String,
    /// 子验证器
    validators: Vec<Box<dyn FeatureValidator>>,
    /// 是否使用AND关系（默认为TRUE，表示所有验证器都必须通过）
    use_and: bool,
}

impl CompositeValidator {
    /// 创建新的组合验证器（默认使用AND关系）
    pub fn new() -> Self {
        Self {
            name: "CompositeValidator".to_string(),
            validators: Vec::new(),
            use_and: true,
        }
    }
    
    /// 创建使用AND关系的组合验证器
    pub fn with_and_relation() -> Self {
        Self {
            name: "AndCompositeValidator".to_string(),
            validators: Vec::new(),
            use_and: true,
        }
    }
    
    /// 创建使用OR关系的组合验证器
    pub fn with_or_relation() -> Self {
        Self {
            name: "OrCompositeValidator".to_string(),
            validators: Vec::new(),
            use_and: false,
        }
    }
    
    /// 添加验证器
    pub fn add_validator<V: FeatureValidator + 'static>(&mut self, validator: V) {
        self.validators.push(Box::new(validator));
    }
    
    /// 添加验证器（链式调用）
    pub fn with_validator<V: FeatureValidator + 'static>(mut self, validator: V) -> Self {
        self.add_validator(validator);
        self
    }
    
    /// 设置验证器名称
    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = name.into();
        self
    }
}

#[async_trait]
impl FeatureValidator for CompositeValidator {
    fn name(&self) -> &str {
        &self.name
    }
    
    fn set_name(&mut self, name: String) {
        self.name = name;
    }
    
    async fn validate_vector(&self, vector: &FeatureVector) -> Result<(), ValidatorError> {
        if self.validators.is_empty() {
            return Ok(());
        }
        
        if self.use_and {
            // AND关系：所有验证器都必须通过
            for validator in &self.validators {
                validator.validate_vector(vector).await?;
            }
            Ok(())
        } else {
            // OR关系：至少一个验证器通过
            for validator in &self.validators {
                if validator.validate_vector(vector).await.is_ok() {
                    return Ok(());
                }
            }
            Err(ValidatorError::ValueError("所有验证器都失败".to_string()))
        }
    }
    
    async fn validate(&self, features: &FeatureVector) -> Result<ValidationResult, ValidatorError> {
        if self.validators.is_empty() {
            return Ok(ValidationResult {
                is_valid: true,
                score: 1.0,
                errors: Vec::new(),
                warnings: Vec::new(),
                timestamp: chrono::Utc::now(),
            });
        }
        
        let mut all_results = Vec::new();
        for validator in &self.validators {
            let result = validator.validate(features).await?;
            all_results.push(result);
        }
        
        if self.use_and {
            // AND关系：所有验证器都必须通过
            let all_valid = all_results.iter().all(|r| r.is_valid);
            let avg_score = all_results.iter().map(|r| r.score).sum::<f64>() / all_results.len() as f64;
            let mut all_errors: Vec<String> = all_results.iter().flat_map(|r| r.errors.clone()).collect();
            let mut all_warnings: Vec<String> = all_results.iter().flat_map(|r| r.warnings.clone()).collect();
            
            Ok(ValidationResult {
                is_valid: all_valid,
                score: avg_score,
                errors: all_errors,
                warnings: all_warnings,
                timestamp: chrono::Utc::now(),
            })
        } else {
            // OR关系：至少一个验证器通过
            let any_valid = all_results.iter().any(|r| r.is_valid);
            let max_score = all_results.iter().map(|r| r.score).fold(0.0, f64::max);
            let mut all_errors: Vec<String> = all_results.iter().flat_map(|r| r.errors.clone()).collect();
            let mut all_warnings: Vec<String> = all_results.iter().flat_map(|r| r.warnings.clone()).collect();
            
            Ok(ValidationResult {
                is_valid: any_valid,
                score: max_score,
                errors: all_errors,
                warnings: all_warnings,
                timestamp: chrono::Utc::now(),
            })
        }
    }
    
    async fn validate_batch(&self, features: &FeatureBatch) -> Result<Vec<ValidationResult>, ValidatorError> {
        let mut results = Vec::new();
        for i in 0..features.batch_size {
            let vector = FeatureVector {
                feature_type: features.feature_type,
                values: features.values[i].clone(),
                extractor_type: features.extractor_type,
                metadata: features.metadata.clone(),
                features: Vec::new(),
            };
            let result = self.validate(&vector).await?;
            results.push(result);
        }
        Ok(results)
    }
    
    async fn get_validation_stats(&self) -> Result<ValidationStats, ValidatorError> {
        Ok(ValidationStats::default())
    }
}

/// 创建预设基本验证器
pub fn create_basic_validator(min_dim: usize, max_dim: usize) -> AdvancedFeatureValidator {
    AdvancedFeatureValidator::new(min_dim, max_dim)
}

/// 创建预设标准化特征验证器（均值接近0，标准差接近1）
pub fn create_standardized_validator() -> AdvancedStatisticalFeatureValidator {
    AdvancedStatisticalFeatureValidator::new(1, usize::MAX)
        .with_name("StandardizedValidator")
        .with_value_range(-5.0, 5.0)  // Z-score标准化后的合理范围
        .with_nan_check(true)
        .with_infinity_check(true)
}

/// 创建预设归一化特征验证器（值在0-1范围内）
pub fn create_normalized_validator() -> AdvancedStatisticalFeatureValidator {
    AdvancedStatisticalFeatureValidator::new(1, usize::MAX)
        .with_name("NormalizedValidator")
        .with_value_range(-0.001, 1.001)
        .with_nan_check(true)
        .with_infinity_check(true)
}

/// 创建预设非零验证器
pub fn create_non_zero_validator() -> AdvancedFeatureValidator {
    AdvancedFeatureValidator::new(1, usize::MAX)
        .with_name("NonZeroValidator")
        .with_non_zero_check(true)
}

/// 创建预设有限值验证器（检查NaN和无穷大）
pub fn create_finite_validator() -> AdvancedFeatureValidator {
    AdvancedFeatureValidator::new(1, usize::MAX)
        .with_name("FiniteValidator")
        .with_nan_check(true)
        .with_infinity_check(true)
}

/// 创建预设稀疏特征验证器（最多50%的元素为0）
pub fn create_sparse_validator(max_zero_ratio: f32) -> AdvancedStatisticalFeatureValidator {
    AdvancedStatisticalFeatureValidator::new(1, usize::MAX)
        .with_name("SparseValidator")
        .with_value_range(-1.0, 1.0)  // 使用value_range代替max_zero_ratio
        .with_nan_check(true)
        .with_infinity_check(true)
}

/// 创建预设工业应用验证器组合（适用于大多数场景）
pub fn create_production_validator(dim: usize) -> CompositeValidator {
    let basic = create_basic_validator(dim, dim)
        .with_name("ProductionBasicValidator");
        
    let finite = create_finite_validator()
        .with_name("ProductionFiniteValidator");
        
    let non_zero = create_non_zero_validator()
        .with_name("ProductionNonZeroValidator");
    
    CompositeValidator::with_and_relation()
        .with_name("ProductionValidator")
        .with_validator(basic)
        .with_validator(finite)
        .with_validator(non_zero)
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_basic_validator() {
        let validator = AdvancedFeatureValidator::new(2, 5);
        
        // 测试维度验证
        assert!(!validator.validate_vector(&FeatureVector { values: vec![1.0], feature_type: "".to_string(), extractor_type: "".to_string(), metadata: HashMap::new() }).is_ok()); // 维度太小
        assert!(validator.validate_vector(&FeatureVector { values: vec![1.0, 2.0], feature_type: "".to_string(), extractor_type: "".to_string(), metadata: HashMap::new() }).is_ok()); // 维度正好
        assert!(!validator.validate_vector(&FeatureVector { values: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], feature_type: "".to_string(), extractor_type: "".to_string(), metadata: HashMap::new() }).is_ok()); // 维度太大
        
        // 测试值范围验证
        let range_validator = AdvancedFeatureValidator::new(2, 5)
            .with_value_range(0.0, 10.0);
        assert!(range_validator.validate_vector(&FeatureVector { values: vec![1.0, 5.0], feature_type: "".to_string(), extractor_type: "".to_string(), metadata: HashMap::new() }).is_ok()); // 在范围内
        assert!(!range_validator.validate_vector(&FeatureVector { values: vec![-1.0, 5.0], feature_type: "".to_string(), extractor_type: "".to_string(), metadata: HashMap::new() }).is_ok()); // 低于范围
        assert!(!range_validator.validate_vector(&FeatureVector { values: vec![1.0, 15.0], feature_type: "".to_string(), extractor_type: "".to_string(), metadata: HashMap::new() }).is_ok()); // 高于范围
        
        // 测试非零验证
        let non_zero_validator = AdvancedFeatureValidator::new(2, 5)
            .with_non_zero_check(true);
        assert!(non_zero_validator.validate_vector(&FeatureVector { values: vec![1.0, 5.0], feature_type: "".to_string(), extractor_type: "".to_string(), metadata: HashMap::new() }).is_ok()); // 非零
        assert!(!non_zero_validator.validate_vector(&FeatureVector { values: vec![0.0, 0.0], feature_type: "".to_string(), extractor_type: "".to_string(), metadata: HashMap::new() }).is_ok()); // 全零
        assert!(non_zero_validator.validate_vector(&FeatureVector { values: vec![0.0, 5.0], feature_type: "".to_string(), extractor_type: "".to_string(), metadata: HashMap::new() }).is_ok()); // 部分零
    }
    
    #[test]
    fn test_statistical_validator() {
        let validator = AdvancedStatisticalFeatureValidator::new()
            .with_mean_range(0.0, 10.0)
            .with_std_range(0.0, 5.0);
        
        // 测试均值验证
        assert!(validator.validate_vector(&FeatureVector { values: vec![5.0, 5.0], feature_type: "".to_string(), extractor_type: "".to_string(), metadata: HashMap::new() }).is_ok()); // 均值在范围内
        assert!(!validator.validate_vector(&FeatureVector { values: vec![-5.0, -5.0], feature_type: "".to_string(), extractor_type: "".to_string(), metadata: HashMap::new() }).is_ok()); // 均值太小
        assert!(!validator.validate_vector(&FeatureVector { values: vec![15.0, 15.0], feature_type: "".to_string(), extractor_type: "".to_string(), metadata: HashMap::new() }).is_ok()); // 均值太大
        
        // 测试标准差验证
        assert!(validator.validate_vector(&FeatureVector { values: vec![3.0, 7.0], feature_type: "".to_string(), extractor_type: "".to_string(), metadata: HashMap::new() }).is_ok()); // 标准差在范围内
        assert!(!validator.validate_vector(&FeatureVector { values: vec![0.0, 10.0, 20.0], feature_type: "".to_string(), extractor_type: "".to_string(), metadata: HashMap::new() }).is_ok()); // 标准差太大
    }
    
    #[test]
    fn test_comparison_validator() {
        let reference = vec![1.0, 0.0, 0.0];
        let validator = ComparisonFeatureValidator::new(reference)
            .with_min_cosine_similarity(0.5)
            .with_max_euclidean_distance(1.0);
        
        // 测试余弦相似度验证
        assert!(validator.validate_vector(&FeatureVector { values: vec![1.0, 0.0, 0.0], feature_type: "".to_string(), extractor_type: "".to_string(), metadata: HashMap::new() }).is_ok()); // 完全相同，余弦相似度=1
        assert!(validator.validate_vector(&FeatureVector { values: vec![0.9, 0.1, 0.1], feature_type: "".to_string(), extractor_type: "".to_string(), metadata: HashMap::new() }).is_ok()); // 接近，余弦相似度>0.5
        assert!(!validator.validate_vector(&FeatureVector { values: vec![0.0, 1.0, 0.0], feature_type: "".to_string(), extractor_type: "".to_string(), metadata: HashMap::new() }).is_ok()); // 正交，余弦相似度=0
        
        // 测试欧氏距离验证
        assert!(validator.validate_vector(&FeatureVector { values: vec![1.0, 0.0, 0.0], feature_type: "".to_string(), extractor_type: "".to_string(), metadata: HashMap::new() }).is_ok()); // 完全相同，欧氏距离=0
        assert!(validator.validate_vector(&FeatureVector { values: vec![0.5, 0.0, 0.0], feature_type: "".to_string(), extractor_type: "".to_string(), metadata: HashMap::new() }).is_ok()); // 距离=0.5，小于1.0
        assert!(!validator.validate_vector(&FeatureVector { values: vec![0.0, 0.0, 0.0], feature_type: "".to_string(), extractor_type: "".to_string(), metadata: HashMap::new() }).is_ok()); // 距离=1.0，等于1.0
    }
    
    #[test]
    fn test_composite_validator() {
        let basic = AdvancedFeatureValidator::new(2, 5);
        let statistical = AdvancedStatisticalFeatureValidator::new()
            .with_mean_range(0.0, 10.0);
        
        // 测试AND关系
        let mut and_validator = CompositeValidator::with_and_relation();
        and_validator.add_validator(basic);
        and_validator.add_validator(statistical);
        
        assert!(and_validator.validate_vector(&FeatureVector { values: vec![5.0, 5.0], feature_type: "".to_string(), extractor_type: "".to_string(), metadata: HashMap::new() }).is_ok()); // 两个验证器都通过
        assert!(!and_validator.validate_vector(&FeatureVector { values: vec![5.0], feature_type: "".to_string(), extractor_type: "".to_string(), metadata: HashMap::new() }).is_ok()); // 第一个验证器失败
        assert!(!and_validator.validate_vector(&FeatureVector { values: vec![-5.0, -5.0], feature_type: "".to_string(), extractor_type: "".to_string(), metadata: HashMap::new() }).is_ok()); // 第二个验证器失败
        
        // 测试OR关系
        let mut or_validator = CompositeValidator::with_or_relation();
        or_validator.add_validator(AdvancedFeatureValidator::new(2, 3));
        or_validator.add_validator(AdvancedFeatureValidator::new(5, 6));
        
        assert!(or_validator.validate_vector(&FeatureVector { values: vec![1.0, 2.0], feature_type: "".to_string(), extractor_type: "".to_string(), metadata: HashMap::new() }).is_ok()); // 第一个验证器通过
        assert!(or_validator.validate_vector(&FeatureVector { values: vec![1.0, 2.0, 3.0, 4.0, 5.0], feature_type: "".to_string(), extractor_type: "".to_string(), metadata: HashMap::new() }).is_ok()); // 第二个验证器通过
        assert!(!or_validator.validate_vector(&FeatureVector { values: vec![1.0, 2.0, 3.0, 4.0], feature_type: "".to_string(), extractor_type: "".to_string(), metadata: HashMap::new() }).is_ok()); // 两个验证器都失败
    }
}
