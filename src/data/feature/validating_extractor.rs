// Validating Extractor Module
// 该模块提供特征提取器的验证包装功能，确保提取的特征符合预期标准

use std::fmt::{Debug, Formatter, Result as FmtResult};
use std::time::Instant;
use async_trait::async_trait;
use futures::{stream, StreamExt};
use std::collections::HashMap;

use super::extractor::{
    FeatureExtractor, ExtractorError, InputData, ExtractorContext, 
    FeatureVector, FeatureBatch, ExtractorConfig
};
use super::validator::{FeatureValidator, ValidatorError};
use super::types::{ExtractorType, FeatureType};
use tracing::{debug, error, info, warn, instrument, span, Level};

/// 验证错误类型
#[derive(Debug, thiserror::Error)]
pub enum ValidationError {
    #[error("验证失败: {0}")]
    Failed(String),
    
    #[error("验证器错误: {0}")]
    ValidatorError(#[from] ValidatorError),
    
    #[error("提取器错误: {0}")]
    ExtractorError(#[from] ExtractorError),
}

/// 验证统计信息
#[derive(Debug, Clone, Default)]
pub struct ValidationStats {
    /// 验证总数
    pub total_validations: usize,
    /// 通过验证数
    pub passed_validations: usize,
    /// 失败验证数
    pub failed_validations: usize,
    /// 平均验证时间(毫秒)
    pub avg_validation_time_ms: f64,
    /// 最近验证结果
    pub last_results: HashMap<String, bool>,
}

/// 验证包装特征提取器
/// 将一个特征提取器与多个验证器组合，在提取特征后进行验证
pub struct ValidatingFeatureExtractor {
    /// 内部特征提取器
    inner: Box<dyn FeatureExtractor>,
    
    /// 特征验证器列表
    validators: Vec<Box<dyn FeatureValidator>>,
    
    /// 验证配置
    config: ValidationConfig,
    
    /// 验证统计
    stats: ValidationStats,
}

/// 验证配置
#[derive(Debug, Clone)]
pub struct ValidationConfig {
    /// 是否启用验证
    pub enabled: bool,
    
    /// 默认为严格模式(全部验证器都必须通过)
    pub strict_mode: bool,
    
    /// 是否并行验证
    pub parallel_validation: bool,
    
    /// 最大并行验证数
    pub max_parallel_validations: usize,
    
    /// 是否记录详细日志
    pub verbose_logging: bool,
    
    /// 是否收集统计信息
    pub collect_stats: bool,
    
    /// 验证超时(毫秒)
    pub timeout_ms: u64,
}

impl Default for ValidationConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            strict_mode: true,
            parallel_validation: false,
            max_parallel_validations: 4,
            verbose_logging: false,
            collect_stats: true,
            timeout_ms: 5000,
        }
    }
}

impl ValidatingFeatureExtractor {
    /// 创建新的验证包装提取器
    ///
    /// # 参数
    /// * `inner` - 内部特征提取器
    /// * `validators` - 特征验证器列表
    ///
    /// # 返回
    /// 返回验证包装特征提取器实例
    #[instrument(skip(inner, validators), level = "debug")]
    pub fn new(
        inner: Box<dyn FeatureExtractor>,
        validators: Vec<Box<dyn FeatureValidator>>
    ) -> Self {
        info!(
            "创建验证包装提取器，提取器类型: {:?}，验证器数量: {}",
            inner.extractor_type(),
            validators.len()
        );
        
        Self {
            inner,
            validators,
            config: ValidationConfig::default(),
            stats: ValidationStats::default(),
        }
    }
    
    /// 使用自定义配置创建验证包装提取器
    ///
    /// # 参数
    /// * `inner` - 内部特征提取器
    /// * `validators` - 特征验证器列表
    /// * `config` - 验证配置
    ///
    /// # 返回
    /// 返回验证包装特征提取器实例
    #[instrument(skip(inner, validators), level = "debug")]
    pub fn with_config(
        inner: Box<dyn FeatureExtractor>,
        validators: Vec<Box<dyn FeatureValidator>>,
        config: ValidationConfig
    ) -> Self {
        info!(
            "创建配置化验证包装提取器，提取器类型: {:?}，验证器数量: {}，严格模式: {}",
            inner.extractor_type(),
            validators.len(),
            config.strict_mode
        );
        
        Self {
            inner,
            validators,
            config,
            stats: ValidationStats::default(),
        }
    }
    
    /// 添加验证器
    ///
    /// # 参数
    /// * `validator` - 要添加的特征验证器
    #[instrument(skip(self, validator), level = "debug")]
    pub fn add_validator(&mut self, validator: Box<dyn FeatureValidator>) {
        debug!("添加验证器: {}", validator.name());
        self.validators.push(validator);
    }
    
    /// 设置验证配置
    ///
    /// # 参数
    /// * `config` - 新的验证配置
    pub fn set_config(&mut self, config: ValidationConfig) {
        debug!("更新验证配置");
        self.config = config;
    }
    
    /// 获取验证统计信息
    pub fn get_stats(&self) -> &ValidationStats {
        &self.stats
    }
    
    /// 重置验证统计信息
    pub fn reset_stats(&mut self) {
        debug!("重置验证统计信息");
        self.stats = ValidationStats::default();
    }
    
    /// 启用验证
    pub fn enable_validation(&mut self) {
        debug!("启用验证");
        self.config.enabled = true;
    }
    
    /// 禁用验证
    pub fn disable_validation(&mut self) {
        debug!("禁用验证");
        self.config.enabled = false;
    }
    
    /// 验证特征向量
    ///
    /// # 参数
    /// * `vector` - 要验证的特征向量
    ///
    /// # 返回
    /// 成功或验证错误
    #[instrument(skip(self, vector), level = "debug")]
    async fn validate(&self, vector: &FeatureVector) -> Result<(), ExtractorError> {
        if !self.config.enabled {
            debug!("验证已禁用，跳过验证");
            return Ok(());
        }
        
        if self.validators.is_empty() {
            debug!("无验证器可用，跳过验证");
            return Ok(());
        }
        
        let start_time = Instant::now();
        let mut validation_errors = Vec::new();
        
        // 严格模式下所有验证器都必须通过
        if self.config.strict_mode {
            if self.config.parallel_validation && self.validators.len() > 1 {
                // 并行验证
                let results = stream::iter(&self.validators)
                    .map(|validator| async move {
                        let result = validator.validate_vector(vector).await;
                        (validator.name(), result)
                    })
                    .buffer_unordered(self.config.max_parallel_validations)
                    .collect::<Vec<_>>()
                    .await;
                
                for (name, result) in results {
                    match result {
                        Ok(_) => {
                            debug!("验证器 {} 验证通过", name);
                        }
                        Err(err) => {
                            error!("验证器 {} 验证失败: {}", name, err);
                            validation_errors.push(format!("验证器 {} 验证失败: {}", name, err));
                        }
                    }
                }
            } else {
                // 串行验证
                for validator in &self.validators {
                    match validator.validate_vector(vector).await {
                        Ok(_) => {
                            debug!("验证器 {} 验证通过", validator.name());
                        }
                        Err(err) => {
                            error!("验证器 {} 验证失败: {}", validator.name(), err);
                            validation_errors.push(format!("验证器 {} 验证失败: {}", validator.name(), err));
                            
                            // 严格模式下遇到错误立即返回
                            break;
                        }
                    }
                }
            }
        } else {
            // 非严格模式下至少一个验证器通过即可
            let mut passed = false;
            
            for validator in &self.validators {
                match validator.validate_vector(vector).await {
                    Ok(_) => {
                        debug!("验证器 {} 验证通过", validator.name());
                        passed = true;
                        break;
                    }
                    Err(err) => {
                        debug!("验证器 {} 验证失败: {}", validator.name(), err);
                        validation_errors.push(format!("验证器 {} 验证失败: {}", validator.name(), err));
                    }
                }
            }
            
            if passed {
                validation_errors.clear();
            }
        }
        
        // 更新统计信息
        if self.config.collect_stats {
            let elapsed = start_time.elapsed().as_millis() as f64;
            
            // 使用内部可变性更新统计信息
            let stats = &mut *unsafe { &mut *((&self.stats) as *const ValidationStats as *mut ValidationStats) };
            stats.total_validations += 1;
            
            if validation_errors.is_empty() {
                stats.passed_validations += 1;
            } else {
                stats.failed_validations += 1;
            }
            
            // 更新平均验证时间
            let total_time = stats.avg_validation_time_ms * (stats.total_validations - 1) as f64 + elapsed;
            stats.avg_validation_time_ms = total_time / stats.total_validations as f64;
            
            // 更新最近验证结果
            for validator in &self.validators {
                stats.last_results.insert(
                    validator.name().to_string(), 
                    !validation_errors.iter().any(|e| e.contains(validator.name()))
                );
            }
        }
        
        // 处理验证错误
        if !validation_errors.is_empty() {
            return Err(ExtractorError::Validation(
                validation_errors.join("; ")
            ));
        }
        
        Ok(())
    }
    
    /// 验证特征批次
    ///
    /// # 参数
    /// * `batch` - 要验证的特征批次
    ///
    /// # 返回
    /// 成功或验证错误
    #[instrument(skip(self, batch), level = "debug")]
    async fn validate_batch(&self, batch: &FeatureBatch) -> Result<(), ExtractorError> {
        if !self.config.enabled {
            debug!("验证已禁用，跳过批量验证");
            return Ok(());
        }
        
        if self.validators.is_empty() {
            debug!("无验证器可用，跳过批量验证");
            return Ok(());
        }
        
        let start_time = Instant::now();
        
        for validator in &self.validators {
            if let Err(err) = validator.validate_batch(batch).await {
                error!("验证器 {} 批量验证失败: {}", validator.name(), err);
                return Err(ExtractorError::Validation(
                    format!("验证器 {} 批量验证失败: {}", validator.name(), err)
                ));
            }
        }
        
        if self.config.verbose_logging {
            let elapsed = start_time.elapsed().as_millis();
            info!(
                "批量验证完成，批次大小: {}，耗时: {}ms",
                batch.batch_size,
                elapsed
            );
        }
        
        Ok(())
    }
    
    /// 获取内部提取器
    pub fn inner(&self) -> &dyn FeatureExtractor {
        self.inner.as_ref()
    }
    
    /// 获取验证器列表
    pub fn validators(&self) -> &[Box<dyn FeatureValidator>] {
        &self.validators
    }
}

impl Debug for ValidatingFeatureExtractor {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        f.debug_struct("ValidatingFeatureExtractor")
            .field("inner", &self.inner)
            .field("validators_count", &self.validators.len())
            .field("config", &self.config)
            .finish()
    }
}

#[async_trait]
impl FeatureExtractor for ValidatingFeatureExtractor {
    /// 获取提取器类型
    fn extractor_type(&self) -> ExtractorType {
        self.inner.extractor_type()
    }
    
    /// 获取提取器配置
    fn config(&self) -> &ExtractorConfig {
        self.inner.config()
    }
    
    /// 检查输入数据是否兼容
    fn is_compatible(&self, input: &InputData) -> bool {
        self.inner.is_compatible(input)
    }
    
    /// 提取特征
    ///
    /// # 参数
    /// * `input` - 输入数据
    /// * `context` - 提取上下文(可选)
    ///
    /// # 返回
    /// 提取的特征向量或错误
    #[instrument(skip(self, input, context), level = "debug")]
    async fn extract(&self, input: InputData, context: Option<ExtractorContext>) -> Result<FeatureVector, ExtractorError> {
        let span = span!(Level::DEBUG, "extract_and_validate");
        let _enter = span.enter();
        
        // 首先验证输入数据兼容性
        if !self.is_compatible(&input) {
            return Err(ExtractorError::InvalidInput(
                format!("输入数据类型不兼容提取器: {:?}", input)
            ));
        }
        
        debug!("开始从输入提取特征");
        
        // 使用内部提取器提取特征
        let vector = match self.inner.extract(input, context).await {
            Ok(v) => v,
            Err(e) => {
                error!("内部提取器提取失败: {}", e);
                return Err(e);
            }
        };
        
        debug!("特征提取成功，开始验证");
        
        // 验证特征
        self.validate(&vector).await?;
        
        debug!("验证通过，返回特征向量");
        
        Ok(vector)
    }
    
    /// 批量提取特征
    ///
    /// # 参数
    /// * `inputs` - 输入数据列表
    /// * `context` - 提取上下文(可选)
    ///
    /// # 返回
    /// 批量特征或错误
    #[instrument(skip(self, inputs, context), level = "debug", fields(batch_size = inputs.len()))]
    async fn batch_extract(&self, inputs: Vec<InputData>, context: Option<ExtractorContext>) -> Result<FeatureBatch, ExtractorError> {
        let span = span!(Level::DEBUG, "batch_extract_and_validate");
        let _enter = span.enter();
        
        if inputs.is_empty() {
            warn!("批量提取收到空输入列表");
            return Err(ExtractorError::InvalidInput("输入列表为空".to_string()));
        }
        
        // 检查所有输入兼容性
        for (i, input) in inputs.iter().enumerate() {
            if !self.is_compatible(input) {
                return Err(ExtractorError::InvalidInput(
                    format!("位置 {} 的输入数据类型不兼容提取器: {:?}", i, input)
                ));
            }
        }
        
        info!("开始批量提取特征，批次大小: {}", inputs.len());
        
        // 使用内部提取器批量提取特征
        let batch = match self.inner.batch_extract(inputs, context).await {
            Ok(b) => b,
            Err(e) => {
                error!("内部提取器批量提取失败: {}", e);
                return Err(e);
            }
        };
        
        debug!("批量特征提取成功，开始验证批次");
        
        // 验证特征批次
        self.validate_batch(&batch).await?;
        
        debug!("批次验证通过，返回特征批次");
        
        Ok(batch)
    }
    
    /// 获取输出特征类型
    fn output_feature_type(&self) -> FeatureType {
        self.inner.output_feature_type()
    }
    
    /// 获取输出维度
    fn output_dimension(&self) -> Option<usize> {
        self.inner.output_dimension()
    }
    
    /// 转换为Any类型
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

/// 创建验证特征提取器
///
/// # 参数
/// * `inner` - 内部特征提取器
/// * `validators` - 验证器列表
///
/// # 返回
/// 返回特征提取器实例
pub fn create_validating_extractor(
    inner: Box<dyn FeatureExtractor>,
    validators: Vec<Box<dyn FeatureValidator>>,
) -> Box<dyn FeatureExtractor> {
    Box::new(ValidatingFeatureExtractor::new(inner, validators))
}

/// 创建带基本验证的特征提取器
///
/// # 参数
/// * `inner` - 内部特征提取器
/// * `min_dim` - 最小维度
/// * `max_dim` - 最大维度
///
/// # 返回
/// 返回特征提取器实例
pub fn create_basic_validating_extractor(
    inner: Box<dyn FeatureExtractor>,
    min_dim: usize,
    max_dim: usize,
) -> Box<dyn FeatureExtractor> {
    let validator = super::validator::create_basic_validator(min_dim, max_dim);
    create_validating_extractor(inner, vec![Box::new(validator)])
}

/// 创建带生产级验证的特征提取器
///
/// # 参数
/// * `inner` - 内部特征提取器
/// * `expected_dim` - 期望维度
///
/// # 返回
/// 返回特征提取器实例
pub fn create_production_validating_extractor(
    inner: Box<dyn FeatureExtractor>,
    expected_dim: usize,
) -> Box<dyn FeatureExtractor> {
    let validator = super::validator::create_production_validator(expected_dim);
    create_validating_extractor(inner, vec![Box::new(validator)])
}

/// 创建并行验证特征提取器
///
/// # 参数
/// * `inner` - 内部特征提取器
/// * `validators` - 验证器列表
/// * `max_parallel` - 最大并行验证数
///
/// # 返回
/// 返回特征提取器实例
pub fn create_parallel_validating_extractor(
    inner: Box<dyn FeatureExtractor>,
    validators: Vec<Box<dyn FeatureValidator>>,
    max_parallel: usize,
) -> Box<dyn FeatureExtractor> {
    let config = ValidationConfig {
        parallel_validation: true,
        max_parallel_validations: max_parallel,
        ..ValidationConfig::default()
    };
    
    Box::new(ValidatingFeatureExtractor::with_config(inner, validators, config))
}

/// 创建非严格模式验证特征提取器(只要有一个验证器通过即可)
///
/// # 参数
/// * `inner` - 内部特征提取器
/// * `validators` - 验证器列表
///
/// # 返回
/// 返回特征提取器实例
pub fn create_non_strict_validating_extractor(
    inner: Box<dyn FeatureExtractor>,
    validators: Vec<Box<dyn FeatureValidator>>,
) -> Box<dyn FeatureExtractor> {
    let config = ValidationConfig {
        strict_mode: false,
        ..ValidationConfig::default()
    };
    
    Box::new(ValidatingFeatureExtractor::with_config(inner, validators, config))
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::extractor::DummyExtractor;
    use super::super::validator::{BasicFeatureValidator, StatisticalFeatureValidator};
    use std::collections::HashMap;
    
    #[tokio::test]
    async fn test_validating_extractor() {
        // 创建基础提取器
        let config = ExtractorConfig::new(ExtractorType::Text);
        let inner = Box::new(DummyExtractor::new(config));
        
        // 创建验证器
        let validator = BasicFeatureValidator::new(1, 3)
            .with_value_range(0.0, 10.0);
        
        // 创建验证特征提取器
        let extractor = ValidatingFeatureExtractor::new(
            inner,
            vec![Box::new(validator)]
        );
        
        // 测试成功情况
        let input = InputData::Text("valid".to_string());
        let result = extractor.extract(input, None).await;
        assert!(result.is_ok());
        
        // 测试失败情况
        let config = ExtractorConfig::new(ExtractorType::Text);
        let inner = Box::new(DummyExtractor::new(config.clone())
            .with_test_values(vec![-1.0, 2.0, 3.0, 4.0])); // 超出范围的值和维度
        
        let extractor = ValidatingFeatureExtractor::new(
            inner,
            vec![Box::new(BasicFeatureValidator::new(1, 3)
                .with_value_range(0.0, 10.0))]
        );
        
        let input = InputData::Text("invalid".to_string());
        let result = extractor.extract(input, None).await;
        assert!(result.is_err());
    }
    
    #[tokio::test]
    async fn test_validating_extractor_with_config() {
        // 创建配置
        let config = ValidationConfig {
            enabled: true,
            strict_mode: false, // 非严格模式
            ..ValidationConfig::default()
        };
        
        // 创建基础提取器
        let ext_config = ExtractorConfig::new(ExtractorType::Text);
        let inner = Box::new(DummyExtractor::new(ext_config));
        
        // 创建两个验证器，一个会通过，一个会失败
        let validator1 = BasicFeatureValidator::new(1, 10) // 这个会通过
            .with_name("验证器1");
            
        let validator2 = BasicFeatureValidator::new(5, 10) // 这个会失败
            .with_name("验证器2");
        
        // 创建验证特征提取器
        let extractor = ValidatingFeatureExtractor::with_config(
            inner,
            vec![Box::new(validator1), Box::new(validator2)],
            config
        );
        
        // 测试非严格模式
        let input = InputData::Text("test".to_string());
        let result = extractor.extract(input, None).await;
        
        // 非严格模式下，只要有一个验证器通过就应该成功
        assert!(result.is_ok());
    }
    
    #[tokio::test]
    async fn test_parallel_validation() {
        // 创建配置
        let config = ValidationConfig {
            enabled: true,
            parallel_validation: true,
            max_parallel_validations: 4,
            ..ValidationConfig::default()
        };
        
        // 创建基础提取器
        let ext_config = ExtractorConfig::new(ExtractorType::Text);
        let inner = Box::new(DummyExtractor::new(ext_config));
        
        // 创建多个验证器
        let validators: Vec<Box<dyn FeatureValidator>> = (0..5).map(|i| {
            Box::new(BasicFeatureValidator::new(1, 10)
                .with_name(format!("验证器{}", i)))
        }).collect();
        
        // 创建验证特征提取器
        let extractor = ValidatingFeatureExtractor::with_config(
            inner,
            validators,
            config
        );
        
        // 测试并行验证
        let input = InputData::Text("test".to_string());
        let result = extractor.extract(input, None).await;
        
        assert!(result.is_ok());
    }
    
    #[tokio::test]
    async fn test_validation_disabled() {
        // 创建配置
        let mut config = ValidationConfig::default();
        config.enabled = false;
        
        // 创建基础提取器
        let ext_config = ExtractorConfig::new(ExtractorType::Text);
        let inner = Box::new(DummyExtractor::new(ext_config));
        
        // 创建一个会失败的验证器
        let validator = BasicFeatureValidator::new(100, 200); // 维度要求不合理
        
        // 创建验证特征提取器
        let extractor = ValidatingFeatureExtractor::with_config(
            inner,
            vec![Box::new(validator)],
            config
        );
        
        // 测试禁用验证
        let input = InputData::Text("test".to_string());
        let result = extractor.extract(input, None).await;
        
        // 验证禁用时，即使有不合理的验证器也应该通过
        assert!(result.is_ok());
    }
    
    #[tokio::test]
    async fn test_batch_extract() {
        // 创建基础提取器
        let config = ExtractorConfig::new(ExtractorType::Text);
        let inner = Box::new(DummyExtractor::new(config));
        
        // 创建验证器
        let validator = BasicFeatureValidator::new(1, 3);
        
        // 创建验证特征提取器
        let extractor = ValidatingFeatureExtractor::new(
            inner,
            vec![Box::new(validator)]
        );
        
        // 测试批量提取
        let inputs = vec![
            InputData::Text("test1".to_string()),
            InputData::Text("test2".to_string()),
            InputData::Text("test3".to_string()),
        ];
        
        let result = extractor.batch_extract(inputs, None).await;
        assert!(result.is_ok());
        
        let batch = result.unwrap();
        assert_eq!(batch.batch_size, 3);
    }
    
    #[tokio::test]
    async fn test_validation_stats() {
        // 创建配置
        let config = ValidationConfig {
            collect_stats: true,
            ..ValidationConfig::default()
        };
        
        // 创建基础提取器
        let ext_config = ExtractorConfig::new(ExtractorType::Text);
        let inner = Box::new(DummyExtractor::new(ext_config));
        
        // 创建验证器
        let validator = BasicFeatureValidator::new(1, 3);
        
        // 创建验证特征提取器
        let extractor = ValidatingFeatureExtractor::with_config(
            inner,
            vec![Box::new(validator)],
            config
        );
        
        // 执行几次提取
        for i in 0..5 {
            let input = InputData::Text(format!("test{}", i));
            let _ = extractor.extract(input, None).await;
        }
        
        // 检查统计信息
        let stats = extractor.get_stats();
        assert_eq!(stats.total_validations, 5);
        assert_eq!(stats.passed_validations, 5);
        assert_eq!(stats.failed_validations, 0);
        assert!(stats.avg_validation_time_ms > 0.0);
    }
    
    #[tokio::test]
    async fn test_empty_validators() {
        // 创建基础提取器
        let config = ExtractorConfig::new(ExtractorType::Text);
        let inner = Box::new(DummyExtractor::new(config));
        
        // 创建没有验证器的提取器
        let extractor = ValidatingFeatureExtractor::new(
            inner,
            vec![]
        );
        
        // 没有验证器时应该正常提取
        let input = InputData::Text("test".to_string());
        let result = extractor.extract(input, None).await;
        assert!(result.is_ok());
    }
    
    #[tokio::test]
    async fn test_helper_functions() {
        // 测试基本验证提取器创建函数
        let config = ExtractorConfig::new(ExtractorType::Text);
        let inner = Box::new(DummyExtractor::new(config));
        
        let extractor = create_basic_validating_extractor(inner, 1, 5);
        assert!(extractor.output_dimension().is_some());
        
        // 测试生产级验证提取器创建函数
        let config = ExtractorConfig::new(ExtractorType::Text);
        let inner = Box::new(DummyExtractor::new(config));
        
        let extractor = create_production_validating_extractor(inner, 3);
        assert!(extractor.output_dimension().is_some());
        
        // 测试并行验证提取器创建函数
        let config = ExtractorConfig::new(ExtractorType::Text);
        let inner = Box::new(DummyExtractor::new(config));
        
        let validators = vec![
            Box::new(BasicFeatureValidator::new(1, 10)),
            Box::new(StatisticalFeatureValidator::new()),
        ];
        
        let extractor = create_parallel_validating_extractor(inner, validators, 2);
        assert!(extractor.output_dimension().is_some());
    }
}
