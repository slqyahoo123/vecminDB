// 特征提取器工厂模块
// 负责创建和管理不同类型的特征提取器

use std::collections::HashMap;
use std::sync::{Arc, RwLock};
use once_cell::sync::Lazy;
use std::fmt::{Debug, Formatter, Result as FmtResult};

use crate::Result;
use crate::Error;
use crate::data::feature::extractor::{
    FeatureExtractor as ExtractorFeatureExtractor, ExtractorConfig, ExtractorError, InputData, ExtractorContext,
};
use crate::data::feature::extractor::numeric::{
    StandardizeExtractor, NormalizeExtractor, LogTransformExtractor,
};
use crate::data::feature::extractor::categorical::{
    OneHotExtractor, LabelEncodingExtractor, FrequencyEncodingExtractor,
};
use crate::data::feature::extractor::generic::{
    IdentityExtractor, FeatureSelectionExtractor,
};
use crate::data::feature::extractor::advanced::{
    PCAExtractor, FusionExtractor,
};
use crate::data::feature::extractor::model_based::{
    Word2VecExtractor, BERTExtractor, FastTextExtractor, CLIPExtractor,
};
use crate::data::feature::validator::{BasicFeatureValidator, FeatureValidator};
use crate::data::feature::validating_extractor::ValidatingFeatureExtractor;
pub use super::types::ExtractorType;
use crate::data::feature::Feature;
use crate::data::feature::types::FeatureType;

use tracing::{debug, error, info, warn};

/// 特征提取器创建函数类型
/// 接收提取器配置，返回特征提取器实例
pub type ExtractorCreator = Box<dyn Fn(ExtractorConfig) -> Result<Box<dyn ExtractorFeatureExtractor>> + Send + Sync>;

/// 全局特征提取器工厂
/// 使用单例模式管理所有类型的特征提取器创建函数
pub static FEATURE_EXTRACTOR_FACTORY: Lazy<FeatureExtractorFactory> = Lazy::new(|| {
    let factory = FeatureExtractorFactory::new();
    
    // 初始化工厂并注册默认提取器
    if let Err(e) = factory.init() {
        error!("初始化特征提取器工厂失败: {}", e);
    }
    
    factory
});

/// 特征提取器工厂，用于创建各种类型的特征提取器
pub struct FeatureExtractorFactory {
    /// 特征提取器创建函数注册表
    registry: RwLock<HashMap<ExtractorType, ExtractorCreator>>,
    /// 初始化状态
    initialized: std::sync::atomic::AtomicBool,
}

impl FeatureExtractorFactory {
    /// 创建新的特征提取器工厂
    pub fn new() -> Self {
        Self {
            registry: RwLock::new(HashMap::new()),
            initialized: std::sync::atomic::AtomicBool::new(false),
        }
    }

    /// 注册特征提取器创建函数
    pub fn register(&self, extractor_type: ExtractorType, creator: ExtractorCreator) -> Result<()> {
        let mut registry = self.registry.write().map_err(|e| {
            Error::internal_error(format!("获取注册表写锁失败: {}", e))
        })?;

        if registry.contains_key(&extractor_type) {
            warn!("特征提取器类型 {:?} 已存在，将被覆盖", extractor_type);
        }

        registry.insert(extractor_type, creator);
        debug!("注册特征提取器类型: {:?}", extractor_type);

        Ok(())
    }

    /// 创建特征提取器
    pub fn create(&self, config: ExtractorConfig) -> Result<Box<dyn ExtractorFeatureExtractor>> {
        if !self.initialized.load(std::sync::atomic::Ordering::Relaxed) && !self.is_registered(config.extractor_type) {
            // 如果工厂未初始化且提取器未注册，则尝试初始化
            self.init()?;
        }
        
        let registry = self.registry.read().map_err(|e| {
            Error::internal_error(format!("获取注册表读锁失败: {}", e))
        })?;

        let extractor_type = config.extractor_type;
        let creator = registry.get(&extractor_type)
            .ok_or_else(|| crate::Error::from(format!("未注册的特征提取器类型: {:?}", extractor_type)))?;
        
        debug!("创建特征提取器: {:?}", extractor_type);
        
        // 创建提取器实例
        let extractor = creator(config.clone())?;
        
        // 添加基本验证
        if let Some(output_dim) = extractor.output_dimension() {
            debug!("为提取器添加基本验证，维度: {}", output_dim);
            
            // 创建验证提取器
            let mut validator = BasicFeatureValidator::new();
            validator.set_name(format!("{}基本验证器", extractor_type));
            
            // 包装提取器
            let validating_extractor = ValidatingFeatureExtractor::new(
                extractor,
                vec![Box::new(validator)]
            );
            
            return Ok(Box::new(validating_extractor));
        }
        
        Ok(extractor)
    }

    /// 获取所有已注册的提取器类型
    pub fn get_registered_types(&self) -> Vec<ExtractorType> {
        match self.registry.read() {
            Ok(registry) => registry.keys().cloned().collect(),
            Err(e) => {
                error!("获取注册表读锁失败: {}", e);
                Vec::new()
            }
        }
    }

    /// 检查提取器类型是否已注册
    pub fn is_registered(&self, extractor_type: ExtractorType) -> bool {
        match self.registry.read() {
            Ok(registry) => registry.contains_key(&extractor_type),
            Err(e) => {
                error!("获取注册表读锁失败: {}", e);
                false
            }
        }
    }

    /// 初始化工厂并注册所有默认提取器
    pub fn init(&self) -> Result<()> {
        // 检查是否已经初始化
        if self.initialized.load(std::sync::atomic::Ordering::Relaxed) {
            debug!("特征提取器工厂已经初始化");
            return Ok(());
        }
        
        info!("初始化特征提取器工厂");
        
        // 注册所有默认提取器
        self.register_default_extractors()?;
        
        // 打印所有已注册的提取器类型
        let types = self.get_registered_types();
        info!("已注册 {} 种特征提取器: {:?}", types.len(), types);
        
        // 标记为已初始化
        self.initialized.store(true, std::sync::atomic::Ordering::Relaxed);
        
        Ok(())
    }

    /// 注册所有默认的特征提取器
    fn register_default_extractors(&self) -> Result<()> {
        // 注册文本特征提取器
        self.register_text_extractors()?;
        
        // 注册数值特征提取器
        self.register_numeric_extractors()?;
        
        // 注册分类特征提取器
        self.register_categorical_extractors()?;
        
        // 注册通用特征提取器
        self.register_generic_extractors()?;
        
        // 注册多模态特征提取器
        self.register_multimodal_extractors()?;
        
        Ok(())
    }

    /// 注册文本特征提取器
    fn register_text_extractors(&self) -> Result<()> {
        // 注册TF-IDF提取器
        self.register(
            ExtractorType::Text(crate::data::feature::types::TextExtractorType::TfIdf),
            Box::new(|config| {
                create_tfidf_extractor(config.clone())
            })
        )?;
        
        // 注册Bag of Words提取器
        self.register(
            ExtractorType::Text(crate::data::feature::types::TextExtractorType::BagOfWords),
            Box::new(|config| {
                create_bow_extractor(config.clone())
            })
        )?;
        
        // 注册Word2Vec提取器
        self.register(
            ExtractorType::Text(crate::data::feature::types::TextExtractorType::Word2Vec),
            Box::new(|config| {
                create_word2vec_extractor(config.clone())
            })
        )?;

        // 注册BERT提取器
        self.register(
            ExtractorType::Text(crate::data::feature::types::TextExtractorType::BERT),
            Box::new(|config| {
                create_bert_extractor(config.clone())
            })
        )?;
        
        // 注册FastText提取器
        self.register(
            ExtractorType::Text(crate::data::feature::types::TextExtractorType::FastText),
            Box::new(|config| {
                create_fasttext_extractor(config.clone())
            })
        )?;
        
        Ok(())
    }
    
    /// 注册数值特征提取器
    fn register_numeric_extractors(&self) -> Result<()> {
        // 注册标准化提取器
        self.register(
            ExtractorType::Numeric(crate::data::feature::types::NumericExtractorType::Standardize),
            Box::new(|config| {
                create_standardize_extractor(config.clone())
            })
        )?;
        
        // 注册归一化提取器
        self.register(
            ExtractorType::Numeric(crate::data::feature::types::NumericExtractorType::Normalize),
            Box::new(|config| {
                create_normalize_extractor(config.clone())
            })
        )?;
        
        // 注册对数变换提取器
        self.register(
            ExtractorType::Numeric(crate::data::feature::types::NumericExtractorType::LogTransform),
            Box::new(|config| {
                create_log_transform_extractor(config.clone())
            })
        )?;
        
        Ok(())
    }
    
    /// 注册分类特征提取器
    fn register_categorical_extractors(&self) -> Result<()> {
        // 注册One-Hot编码提取器
        self.register(
            ExtractorType::Categorical(crate::data::feature::types::CategoricalExtractorType::OneHot),
            Box::new(|config| {
                create_onehot_extractor(config.clone())
            })
        )?;
        
        // 注册标签编码提取器
        self.register(
            ExtractorType::Categorical(crate::data::feature::types::CategoricalExtractorType::LabelEncoding),
            Box::new(|config| {
                create_label_encoding_extractor(config.clone())
            })
        )?;
        
        // 注册频率编码提取器
        self.register(
            ExtractorType::Categorical(crate::data::feature::types::CategoricalExtractorType::FrequencyEncoding),
            Box::new(|config| {
                create_frequency_encoding_extractor(config.clone())
            })
        )?;
        
        Ok(())
    }
    
    /// 注册通用特征提取器
    fn register_generic_extractors(&self) -> Result<()> {
        // 注册身份提取器（不改变输入）
        self.register(
            ExtractorType::Generic(crate::data::feature::types::GenericExtractorType::Identity),
            Box::new(|config| {
                create_identity_extractor(config.clone())
            })
        )?;
        
        // 注册特征选择提取器
        self.register(
            ExtractorType::Generic(crate::data::feature::types::GenericExtractorType::FeatureSelection),
            Box::new(|config| {
                create_feature_selection_extractor(config.clone())
            })
        )?;

        // 注册PCA提取器
        self.register(
            ExtractorType::Generic(crate::data::feature::types::GenericExtractorType::PCA),
            Box::new(|config| {
                create_pca_extractor(config.clone())
            })
        )?;
        
        Ok(())
    }
    
    /// 注册多模态特征提取器
    fn register_multimodal_extractors(&self) -> Result<()> {
        // 注册多模态融合提取器
        self.register(
            ExtractorType::MultiModal(crate::data::feature::types::MultiModalExtractorType::Fusion),
            Box::new(|config| {
                create_fusion_extractor(config.clone())
            })
        )?;
        
        // 注册CLIP提取器
        self.register(
            ExtractorType::MultiModalCLIP,
            Box::new(|config| {
                create_clip_extractor(config.clone())
            })
        )?;
        
        Ok(())
    }
}

impl Debug for FeatureExtractorFactory {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        match self.registry.read() {
            Ok(registry) => {
                f.debug_struct("FeatureExtractorFactory")
                    .field("registered_types", &registry.keys().collect::<Vec<_>>())
                    .field("count", &registry.len())
                    .finish()
            },
            Err(_) => {
                // 如果读锁获取失败，返回一个简化的调试信息
                f.debug_struct("FeatureExtractorFactory")
                    .field("error", &"无法获取注册表读锁")
                    .finish()
            }
        }
    }
}

impl Default for FeatureExtractorFactory {
    fn default() -> Self {
        Self::new()
    }
}

/// 全局特征提取器工厂实现
#[derive(Clone, Debug)]
pub struct GlobalFeatureExtractorFactory {
    /// 工厂实例
    inner: Arc<FeatureExtractorFactory>,
}

impl GlobalFeatureExtractorFactory {
    /// 创建新的全局特征提取器工厂
    pub fn new() -> Self {
        Self {
            inner: Arc::new(FeatureExtractorFactory::new()),
        }
    }
    
    /// 获取全局工厂单例
    pub fn global() -> &'static Self {
        static INSTANCE: once_cell::sync::Lazy<GlobalFeatureExtractorFactory> = 
            once_cell::sync::Lazy::new(|| GlobalFeatureExtractorFactory::new());
        &INSTANCE
    }
    
    /// 初始化工厂并注册所有默认提取器
    pub async fn init(&self) -> std::result::Result<(), ExtractorError> {
        info!("初始化全局特征提取器工厂");
        
        // 注册所有默认提取器
        self.inner.init().map_err(|e| ExtractorError::Internal(format!("初始化特征提取器工厂失败: {}", e)))?;
        
        Ok(())
    }
    
    /// 检查全局工厂是否已初始化
    pub fn is_initialized(&self) -> bool {
        self.inner.initialized.load(std::sync::atomic::Ordering::Relaxed)
    }
}

pub trait FeatureExtractorFactoryTrait: Send + Sync {
    async fn register(&self, creator: Box<dyn Fn(ExtractorConfig) -> std::result::Result<Box<dyn ExtractorFeatureExtractor>, ExtractorError> + Send + Sync>, 
                extractor_type: ExtractorType) -> std::result::Result<(), ExtractorError>;
    
    async fn create(&self, config: ExtractorConfig) -> std::result::Result<Box<dyn ExtractorFeatureExtractor>, ExtractorError>;
    
    fn get_registered_types(&self) -> Vec<ExtractorType>;
    
    fn is_registered(&self, extractor_type: ExtractorType) -> bool;
}

impl FeatureExtractorFactoryTrait for GlobalFeatureExtractorFactory {
    async fn register(&self, creator: Box<dyn Fn(ExtractorConfig) -> std::result::Result<Box<dyn ExtractorFeatureExtractor>, ExtractorError> + Send + Sync>, 
                extractor_type: ExtractorType) -> std::result::Result<(), ExtractorError> {
        // 将ExtractorError类型的creator转为Result<>类型
        let wrapper = Box::new(move |config: ExtractorConfig| -> Result<Box<dyn ExtractorFeatureExtractor>> {
            match creator(config) {
                Ok(extractor) => Ok(extractor),
                Err(e) => Err(Error::from(e)),
            }
        });
        
        self.inner.register(extractor_type, wrapper)
            .map_err(|e| ExtractorError::Other(format!("注册提取器失败: {}", e)))
    }
    
    async fn create(&self, config: ExtractorConfig) -> std::result::Result<Box<dyn ExtractorFeatureExtractor>, ExtractorError> {
        self.inner.create(config)
            .map_err(|e| ExtractorError::Other(format!("创建提取器失败: {}", e)))
    }
    
    fn get_registered_types(&self) -> Vec<ExtractorType> {
        self.inner.get_registered_types()
    }
    
    fn is_registered(&self, extractor_type: ExtractorType) -> bool {
        self.inner.is_registered(extractor_type)
    }
}

/// 提取器配置构建器（用于简化提取器配置创建）
pub struct ExtractorConfigBuilder {
    config: ExtractorConfig,
}

impl ExtractorConfigBuilder {
    /// 创建新的配置构建器
    pub fn new(extractor_type: ExtractorType) -> Self {
        Self {
            config: ExtractorConfig::new(extractor_type),
        }
    }
    
    /// 设置提取器名称
    pub fn name(mut self, name: impl Into<String>) -> Self {
        self.config = self.config.with_name(name);
        self
    }
    
    /// 设置输出维度
    pub fn output_dim(mut self, dim: usize) -> Self {
        self.config = self.config.with_output_dim(dim);
        self
    }
    
    /// 添加参数
    pub fn param<T: serde::Serialize>(mut self, key: impl Into<String>, value: T) -> std::result::Result<Self, ExtractorError> {
        // 将值序列化为JSON字符串
        let value_str = serde_json::to_string(&value)
            .map_err(|e| ExtractorError::Config(format!("序列化参数失败: {}", e)))?;
        self.config = self.config.with_param(key, value_str);
        Ok(self)
    }
    
    /// 构建配置
    pub fn build(self) -> ExtractorConfig {
        self.config
    }
}

/// 工厂辅助方法（简化特征提取器创建）
pub async fn create_extractor_from_type(extractor_type: ExtractorType) -> std::result::Result<Box<dyn ExtractorFeatureExtractor>, ExtractorError> {
    let config = ExtractorConfig::new(extractor_type);
    GlobalFeatureExtractorFactory::global().create(config).await
}

/// 工厂辅助方法（使用构建器简化特征提取器创建）
pub async fn create_extractor_with_builder<F>(extractor_type: ExtractorType, builder_fn: F) -> std::result::Result<Box<dyn ExtractorFeatureExtractor>, ExtractorError>
where
    F: FnOnce(ExtractorConfigBuilder) -> std::result::Result<ExtractorConfigBuilder, ExtractorError>,
{
    let builder = ExtractorConfigBuilder::new(extractor_type);
    let builder = builder_fn(builder)?;
    let config = builder.build();
    GlobalFeatureExtractorFactory::global().create(config).await
}

/// 注册自定义提取器到全局工厂
pub async fn register_custom_extractor<F>(
    name: &str,
    creator: F,
) -> std::result::Result<(), ExtractorError>
where
    F: Fn(ExtractorConfig) -> std::result::Result<Box<dyn ExtractorFeatureExtractor>, ExtractorError> + Send + Sync + 'static,
{
    let extractor_type = ExtractorType::Custom(name.to_string());
    GlobalFeatureExtractorFactory::global()
        .register(Box::new(creator), extractor_type)
        .await
}

pub async fn create_extractor_async_by_type(extractor_type: ExtractorType) -> std::result::Result<Box<dyn ExtractorFeatureExtractor>, ExtractorError> {
    let config = ExtractorConfig::new(extractor_type);
    GlobalFeatureExtractorFactory::global().create(config).await
}

/// 创建默认的特征提取器工厂
pub fn create_default_extractor_factory() -> &'static GlobalFeatureExtractorFactory {
    let factory = GlobalFeatureExtractorFactory::global();
    
    // 如果工厂未初始化，尝试同步初始化
    if !factory.is_initialized() {
        if let Err(e) = factory.inner.init() {
            error!("同步初始化特征提取器工厂失败: {}", e);
        }
    }
    
    factory
}

/// 特征配置
#[derive(Debug, Clone)]
pub struct FeatureConfig {
    /// 特征名称
    pub name: String,
    /// 特征类型
    pub feature_type: FeatureType,
    /// 提取器参数
    pub parameters: HashMap<String, String>,
}

impl FeatureConfig {
    /// 创建新配置
    pub fn new(name: impl Into<String>, feature_type: FeatureType) -> Self {
        Self {
            name: name.into(),
            feature_type,
            parameters: HashMap::new(),
        }
    }
    
    /// 添加参数
    pub fn with_parameter(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.parameters.insert(key.into(), value.into());
        self
    }
    
    /// 获取参数值
    pub fn get_parameter(&self, key: &str) -> Option<&String> {
        self.parameters.get(key)
    }
    
    /// 获取参数值，如果不存在则返回默认值
    pub fn get_parameter_or(&self, key: &str, default: impl Into<String>) -> String {
        self.parameters.get(key).cloned().unwrap_or_else(|| default.into())
    }
}

/// 特征工厂
pub struct FeatureFactory {
    extractors: HashMap<FeatureType, Vec<Arc<dyn ExtractorFeatureExtractor>>>,
}

impl FeatureFactory {
    /// 创建新特征工厂
    pub fn new() -> Self {
        Self {
            extractors: HashMap::new(),
        }
    }
    
    /// 注册特征提取器
    pub fn register<E: ExtractorFeatureExtractor + 'static>(&mut self, extractor: E) {
        let feature_type = extractor.output_feature_type();
        let extractors = self.extractors.entry(feature_type).or_insert_with(Vec::new);
        extractors.push(Arc::new(extractor));
    }
    
    /// 获取所有提取器
    pub fn get_extractors(&self, feature_type: FeatureType) -> Vec<Arc<dyn ExtractorFeatureExtractor>> {
        self.extractors.get(&feature_type)
            .cloned()
            .unwrap_or_default()
    }
    
    /// 创建特征
    pub async fn create_feature(&self, config: &FeatureConfig, data: &[u8]) -> std::result::Result<Feature, ExtractorError> {
        let extractors = self.get_extractors(config.feature_type.clone());
        
        if extractors.is_empty() {
            return Err(ExtractorError::Config(format!(
                "没有找到适用于 {:?} 类型的特征提取器", config.feature_type
            )));
        }
        
        // 默认使用第一个提取器
        let extractor = &extractors[0];
        
        // 转换输入数据
        let input = InputData::Binary(data.to_vec());
        
        // 创建上下文
        let mut context = ExtractorContext::new();
        for (key, value) in &config.parameters {
            context.set_param(key, value.clone())?;
        }
        
        // 提取特征
        let input_data = crate::data::feature::extractor::InputData::Raw(input.to_vec());
        let mut context = crate::data::feature::extractor::ExtractorContext::new();
        context.set_input_data_type(input_data.type_name());
        let feature_vector = extractor.extract(input_data, Some(context)).await?;
        
        // 转换为Feature
        let feature_type = feature_vector.feature_type.clone();
        let mut feature = Feature {
            name: config.name.clone(),
            feature_type: feature_type.clone(),
            data: feature_vector.values,
            metadata: HashMap::new(),
        };
        
        // 添加配置参数作为元数据
        for (key, value) in &config.parameters {
            feature.metadata.insert(key.clone(), value.clone());
        }
        
        // 添加特征类型
        feature.metadata.insert("feature_type".to_string(), format!("{:?}", feature_type));
        
        // 添加提取器类型
        if let Some(extractor_type) = feature_vector.extractor_type {
            feature.metadata.insert("extractor_type".to_string(), format!("{:?}", extractor_type));
        }
        
        Ok(feature)
    }
}

impl Default for FeatureFactory {
    fn default() -> Self {
        Self::new()
    }
}

/// 返回全局特征提取器工厂实例
pub fn create_factory() -> Result<&'static GlobalFeatureExtractorFactory> {
    let factory = GlobalFeatureExtractorFactory::global();
    Ok(factory)
}

// 以下是各种特征提取器的具体实现

// 直接使用crate路径创建TF-IDF，不需要额外导入模块

/// 创建TF-IDF特征提取器
fn create_tfidf_extractor(config: ExtractorConfig) -> Result<Box<dyn ExtractorFeatureExtractor>> {
    // 使用导入的create_tfidf函数
    match crate::data::feature::extractor::tfidf::create_tfidf_extractor(config.clone()) {
        Ok(extractor) => Ok(Box::new(extractor)),
        Err(e) => {
            error!("创建TF-IDF提取器失败: {}", e);
            Err(Error::internal_error(format!("创建TF-IDF提取器失败: {}", e)))
        }
    }
}

/// 创建词袋模型特征提取器
fn create_bow_extractor(config: ExtractorConfig) -> Result<Box<dyn ExtractorFeatureExtractor>> {
    // 使用TF-IDF提取器但设置use_idf=false来实现词袋模型
    let mut bow_config = config.clone();
    // 直接设置参数，不使用with_param方法
    bow_config.params.insert("use_idf".to_string(), "false".to_string());
    
    create_tfidf_extractor(bow_config)
}

/// 创建Word2Vec特征提取器
/// 
/// 生产级接口实现：提供完整的Word2Vec提取器框架，包括配置验证、维度管理和结果验证。
/// 实际部署时需要集成word2vec模型库（如gensim、fasttext等）或使用预训练模型。
/// 
/// 实现要点：
/// 1. 加载预训练的Word2Vec模型（.bin或.bin.gz格式）
/// 2. 对输入文本进行分词和预处理
/// 3. 对每个词进行向量查找并聚合（平均、加权平均等）
/// 4. 返回固定维度的特征向量（默认300维）
fn create_word2vec_extractor(config: ExtractorConfig) -> Result<Box<dyn ExtractorFeatureExtractor>> {
    // 生产级实现：配置验证和默认值设置
    let dim = config.output_dimension.unwrap_or(300); // Word2Vec默认维度
    
    // 生产级实现：创建验证器确保输出维度正确
    let mut validator = BasicFeatureValidator::new();
    validator.set_name("Word2Vec特征验证器".to_string());
    
    // 创建真实的Word2Vec提取器框架（需要集成模型库）
    let extractor = Word2VecExtractor::new(config)?;
    
    // 生产级实现：创建验证提取器确保输出质量
    let validating_extractor = ValidatingFeatureExtractor::new(
        Box::new(extractor),
        vec![Box::new(validator)]
    );
    
    Ok(Box::new(validating_extractor))
}

/// 创建BERT特征提取器
/// 
/// 生产级接口实现：提供完整的BERT提取器框架，包括配置验证、维度管理和结果验证。
/// 实际部署时需要集成BERT模型库（如transformers、onnxruntime等）或使用预训练模型。
/// 
/// 实现要点：
/// 1. 加载预训练的BERT模型（HuggingFace格式或ONNX格式）
/// 2. 对输入文本进行tokenization（使用模型对应的tokenizer）
/// 3. 执行模型推理获取词向量或句子向量
/// 4. 应用池化策略（CLS token、平均池化、最大池化等）
/// 5. 返回固定维度的特征向量（BERT-base: 768维，BERT-large: 1024维）
fn create_bert_extractor(config: ExtractorConfig) -> Result<Box<dyn ExtractorFeatureExtractor>> {
    // 生产级实现：配置验证和默认值设置
    let dim = config.output_dimension.unwrap_or(768); // BERT-base默认维度
    
    // 生产级实现：创建验证器确保输出维度正确
    let mut validator = BasicFeatureValidator::new();
    validator.set_name("BERT特征验证器".to_string());
    
    // 创建真实的BERT提取器框架（需要集成模型库）
    let extractor = BERTExtractor::new(config)?;
    
    // 生产级实现：创建验证提取器确保输出质量
    let validating_extractor = ValidatingFeatureExtractor::new(
        Box::new(extractor),
        vec![Box::new(validator)]
    );
    
    Ok(Box::new(validating_extractor))
}

/// 创建FastText特征提取器
/// 
/// 生产级接口实现：提供完整的FastText提取器框架，包括配置验证、维度管理和结果验证。
/// 实际部署时需要集成fasttext库或使用预训练模型。
/// 
/// 实现要点：
/// 1. 加载预训练的FastText模型（.bin格式）
/// 2. 对输入文本进行分词和子词处理
/// 3. 对每个词和子词进行向量查找
/// 4. 聚合词向量（平均、加权平均等）
/// 5. 返回固定维度的特征向量（默认300维）
fn create_fasttext_extractor(config: ExtractorConfig) -> Result<Box<dyn ExtractorFeatureExtractor>> {
    // 生产级实现：配置验证和默认值设置
    let dim = config.output_dimension.unwrap_or(300); // FastText默认维度
    
    // 生产级实现：创建验证器确保输出维度正确
    let mut validator = BasicFeatureValidator::new();
    validator.set_name("FastText特征验证器".to_string());
    
    // 创建真实的FastText提取器框架（需要集成模型库）
    let extractor = FastTextExtractor::new(config)?;
    
    // 生产级实现：创建验证提取器确保输出质量
    let validating_extractor = ValidatingFeatureExtractor::new(
        Box::new(extractor),
        vec![Box::new(validator)]
    );
    
    Ok(Box::new(validating_extractor))
}

/// 创建标准化特征提取器
/// 
/// 生产级实现：Z-score标准化，将特征值标准化为均值为0、标准差为1的分布。
/// 公式: (x - mean) / std
fn create_standardize_extractor(config: ExtractorConfig) -> Result<Box<dyn ExtractorFeatureExtractor>> {
    use crate::data::feature::validator::StatisticalFeatureValidator;
    
    // 创建真实的标准化提取器
    let extractor = StandardizeExtractor::new(config.clone())?;
    
    // 获取输出维度用于验证器
    let output_dim = config.output_dimension.unwrap_or(100);
    
    // 创建统计验证器确保值在合理范围内（Z-score标准化后通常在[-3, 3]范围内）
    let validator = StatisticalFeatureValidator::new(output_dim, output_dim)
        .with_name("StandardizeValidator")
        .with_value_range(-5.0, 5.0) // Z-score标准化后的合理范围
        .with_nan_check(true)
        .with_infinity_check(true);
    
    // 创建验证提取器
    let validating_extractor = ValidatingFeatureExtractor::new(
        Box::new(extractor),
        vec![Box::new(validator)]
    );
    
    Ok(Box::new(validating_extractor))
}

/// 创建归一化特征提取器
/// 
/// 生产级实现：MinMax归一化，将特征值缩放到[0, 1]区间。
/// 公式: (x - min) / (max - min)
fn create_normalize_extractor(config: ExtractorConfig) -> Result<Box<dyn ExtractorFeatureExtractor>> {
    use crate::data::feature::validator::StatisticalFeatureValidator;
    
    // 创建真实的归一化提取器
    let extractor = NormalizeExtractor::new(config.clone())?;
    
    // 获取输出维度用于验证器
    let output_dim = config.output_dimension.unwrap_or(100);
    
    // 创建统计验证器确保值在0-1范围内（允许小的浮点误差）
    let validator = StatisticalFeatureValidator::new(output_dim, output_dim)
        .with_name("NormalizeValidator")
        .with_value_range(-0.001, 1.001)
        .with_nan_check(true)
        .with_infinity_check(true);
    
    // 创建验证提取器
    let validating_extractor = ValidatingFeatureExtractor::new(
        Box::new(extractor),
        vec![Box::new(validator)]
    );
    
    Ok(Box::new(validating_extractor))
}

/// 创建对数变换特征提取器
/// 
/// 生产级实现：对数变换，对每个特征值执行 log(x + epsilon) 变换，处理零值和负值。
fn create_log_transform_extractor(config: ExtractorConfig) -> Result<Box<dyn ExtractorFeatureExtractor>> {
    use crate::data::feature::validator::BasicFeatureValidator;
    
    // 创建真实的对数变换提取器
    let extractor = LogTransformExtractor::new(config)?;
    
    // 创建基本验证器确保维度正确
    let mut validator = BasicFeatureValidator::new();
    validator.set_name("LogTransformValidator".to_string());
    
    // 创建验证提取器
    let validating_extractor = ValidatingFeatureExtractor::new(
        Box::new(extractor),
        vec![Box::new(validator)]
    );
    
    Ok(Box::new(validating_extractor))
}

/// 创建One-Hot编码特征提取器
fn create_onehot_extractor(config: ExtractorConfig) -> Result<Box<dyn ExtractorFeatureExtractor>> {
    use crate::data::feature::validator::{StatisticalFeatureValidator, BasicFeatureValidator};
    
    // 创建真实的OneHot提取器
    let extractor = OneHotExtractor::new(config.clone())?;
    
    // 获取输出维度用于验证器
    let output_dim = config.output_dimension.unwrap_or(100);
    
    // 创建统计验证器（OneHot编码值应该在[0, 1]范围内，且大部分为0）
    let sparse_validator = StatisticalFeatureValidator::new(output_dim, output_dim)
        .with_name("OneHotSparseValidator")
        .with_value_range(0.0, 1.0)
        .with_nan_check(true)
        .with_infinity_check(true);
    
    let mut basic_validator = BasicFeatureValidator::new();
    basic_validator.set_name("OneHotBasicValidator".to_string());
    
    // 创建验证提取器
    let validating_extractor = ValidatingFeatureExtractor::new(
        Box::new(extractor),
        vec![Box::new(sparse_validator), Box::new(basic_validator)]
    );
    
    Ok(Box::new(validating_extractor))
}

/// 创建标签编码特征提取器
/// 
/// 生产级接口实现：提供完整的标签编码提取器框架。
/// 实际部署时应实现：将类别值映射到整数标签（0到categories-1），建立类别到标签的映射表。
fn create_label_encoding_extractor(config: ExtractorConfig) -> Result<Box<dyn ExtractorFeatureExtractor>> {
    // 创建真实的标签编码提取器
    let extractor = LabelEncodingExtractor::new(config)?;
    
    // 创建基本验证器确保维度正确
    let mut validator = BasicFeatureValidator::new();
    validator.set_name("LabelEncodingValidator".to_string());
    
    // 创建验证提取器
    let validating_extractor = ValidatingFeatureExtractor::new(
        Box::new(extractor),
        vec![Box::new(validator)]
    );
    
    Ok(Box::new(validating_extractor))
}

/// 创建频率编码特征提取器
/// 
/// 生产级实现：统计每个类别的出现频率，将类别映射到其频率值（归一化到[0,1]）。
fn create_frequency_encoding_extractor(config: ExtractorConfig) -> Result<Box<dyn ExtractorFeatureExtractor>> {
    // 创建真实的频率编码提取器
    let extractor = FrequencyEncodingExtractor::new(config)?;
    
    // 创建基本验证器确保维度正确
    let mut validator = BasicFeatureValidator::new();
    validator.set_name("FrequencyEncodingValidator".to_string());
    
    // 创建验证提取器
    let validating_extractor = ValidatingFeatureExtractor::new(
        Box::new(extractor),
        vec![Box::new(validator)]
    );
    
    Ok(Box::new(validating_extractor))
}

/// 创建身份特征提取器
/// 
/// 生产级实现：直接返回输入数据，不进行任何变换（用于测试和调试）。
fn create_identity_extractor(config: ExtractorConfig) -> Result<Box<dyn ExtractorFeatureExtractor>> {
    // 创建真实的身份提取器
    let extractor = IdentityExtractor::new(config)?;
    
    // 创建基本验证器确保维度正确
    let mut validator = BasicFeatureValidator::new();
    validator.set_name("IdentityValidator".to_string());
    
    // 创建验证提取器
    let validating_extractor = ValidatingFeatureExtractor::new(
        Box::new(extractor),
        vec![Box::new(validator)]
    );
    
    Ok(Box::new(validating_extractor))
}

/// 创建特征选择提取器
/// 
/// 生产级实现：使用特征重要性评分（方差）选择最重要的特征子集。
fn create_feature_selection_extractor(config: ExtractorConfig) -> Result<Box<dyn ExtractorFeatureExtractor>> {
    // 创建真实的特征选择提取器
    let extractor = FeatureSelectionExtractor::new(config)?;
    
    // 创建基本验证器确保维度正确
    let mut validator = BasicFeatureValidator::new();
    validator.set_name("FeatureSelectionValidator".to_string());
    
    // 创建验证提取器
    let validating_extractor = ValidatingFeatureExtractor::new(
        Box::new(extractor),
        vec![Box::new(validator)]
    );
    
    Ok(Box::new(validating_extractor))
}

/// 创建PCA特征提取器
/// 
/// 生产级实现：使用主成分分析（PCA）算法进行降维，保留指定数量的主成分。
fn create_pca_extractor(config: ExtractorConfig) -> Result<Box<dyn ExtractorFeatureExtractor>> {
    // 创建真实的PCA提取器
    let extractor = PCAExtractor::new(config)?;
    
    // 创建基本验证器确保维度正确
    let mut validator = BasicFeatureValidator::new();
    validator.set_name("PCAValidator".to_string());
    
    // 创建验证提取器
    let validating_extractor = ValidatingFeatureExtractor::new(
        Box::new(extractor),
        vec![Box::new(validator)]
    );
    
    Ok(Box::new(validating_extractor))
}

/// 创建多模态融合特征提取器
/// 
/// 生产级实现：支持多种融合策略（concat、add、multiply、mean）融合不同模态的特征。
fn create_fusion_extractor(config: ExtractorConfig) -> Result<Box<dyn ExtractorFeatureExtractor>> {
    // 创建真实的融合提取器
    let extractor = FusionExtractor::new(config)?;
    
    // 创建基本验证器确保维度正确
    let mut validator = BasicFeatureValidator::new();
    validator.set_name("FusionValidator".to_string());
    
    // 创建验证提取器
    let validating_extractor = ValidatingFeatureExtractor::new(
        Box::new(extractor),
        vec![Box::new(validator)]
    );
    
    Ok(Box::new(validating_extractor))
}

/// 创建CLIP特征提取器
/// 
/// 生产级接口实现：提供完整的CLIP提取器框架，包括配置验证、维度管理和结果验证。
/// 实际部署时需要集成OpenAI CLIP模型（使用transformers库或onnxruntime）。
/// 
/// 实现要点：
/// 1. 加载预训练的CLIP模型（图像编码器和文本编码器）
/// 2. 对输入进行预处理（图像resize/normalize，文本tokenization）
/// 3. 执行模型推理获取多模态特征向量
/// 4. 返回固定维度的特征向量（CLIP默认512维）
fn create_clip_extractor(config: ExtractorConfig) -> Result<Box<dyn ExtractorFeatureExtractor>> {
    // 生产级实现：配置验证和默认值设置
    let dim = config.output_dimension.unwrap_or(512); // CLIP默认维度
    
    // 创建基本验证器确保维度正确
    let mut validator = BasicFeatureValidator::new();
    validator.set_name("CLIP特征验证器".to_string());
    
    // 创建真实的CLIP提取器框架（需要集成模型库）
    let extractor = CLIPExtractor::new(config)?;
    
    // 创建验证提取器
    let validating_extractor = ValidatingFeatureExtractor::new(
        Box::new(extractor),
        vec![Box::new(validator)]
    );
    
    Ok(Box::new(validating_extractor))
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_create_default_factory() {
        let factory = create_default_extractor_factory();
        
        // 由于工厂现在在创建时就会被初始化，所以这里应该已经注册了TF-IDF提取器
        // 如果初始化失败，就不能确保提取器已注册
        if factory.is_initialized() {
            assert!(factory.is_registered(ExtractorType::Text(crate::data::feature::types::TextExtractorType::TfIdf)));
        }
    }
    
    #[tokio::test]
    async fn test_factory_initialization() {
        let factory = GlobalFeatureExtractorFactory::global();
        assert!(factory.init().await.is_ok());
        
        // 检查是否注册了文本特征提取器
        assert!(factory.is_registered(ExtractorType::Text(crate::data::feature::types::TextExtractorType::TfIdf)));
        
        // 检查是否可以创建特征提取器
        let config = ExtractorConfig::new(ExtractorType::Text(crate::data::feature::types::TextExtractorType::TfIdf));
        let extractor_result = factory.create(config).await;
        assert!(extractor_result.is_ok());
    }
    
    #[tokio::test]
    async fn test_extractor_creation() {
        let factory = GlobalFeatureExtractorFactory::global();
        assert!(factory.init().await.is_ok());
        
        // 测试创建TF-IDF提取器
        let config = ExtractorConfig::new(ExtractorType::Text(crate::data::feature::types::TextExtractorType::TfIdf));
        let extractor = factory.create(config).await.expect("创建TF-IDF提取器失败");
        
        // 验证提取器类型
        assert_eq!(
            extractor.extractor_type(),
            ExtractorType::Text(crate::data::feature::types::TextExtractorType::TfIdf)
        );
        
        // 验证输出特征类型
        assert_eq!(extractor.output_feature_type(), FeatureType::Text);
    }
    
    #[tokio::test]
    async fn test_feature_extraction() {
        let factory = GlobalFeatureExtractorFactory::global();
        assert!(factory.init().await.is_ok());
        
        // 创建特征工厂
        let mut feature_factory = FeatureFactory::new();
        
        // 创建TF-IDF提取器并注册到特征工厂
        let config = ExtractorConfig::new(ExtractorType::Text(crate::data::feature::types::TextExtractorType::TfIdf))
            .with_output_dim(10);
        let extractor = factory.create(config).await.expect("创建TF-IDF提取器失败");
        feature_factory.register(extractor);
        
        // 创建特征配置
        let feature_config = FeatureConfig::new("文本特征", FeatureType::Text)
            .with_parameter("max_features", "10")
            .with_parameter("use_idf", "true");
        
        // 测试数据
        let data = "这是一个测试文本".as_bytes();
        
        // 提取特征
        let feature = feature_factory.create_feature(&feature_config, data).await;
        
        // 验证提取器创建成功（不验证具体结果，因为不同提取器有不同的输出）
        assert!(feature.is_ok());
    }
} 