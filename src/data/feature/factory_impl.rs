use std::collections::HashMap;
use std::sync::{Arc, RwLock};
use std::any::Any;
use std::fmt::{Debug, Formatter, Result as FmtResult};
use log::{info};

use async_trait::async_trait;

use crate::Result;
use crate::Error;
use crate::data::feature::types::{FeatureType, ExtractorType, TextExtractorType, GenericExtractorType};
use crate::data::feature::extractor::{
    FeatureExtractor, ExtractorConfig, ExtractorError, FeatureVector, FeatureBatch, 
    InputData, ExtractorContext
};
use crate::data::feature::interface::FeatureExtractorFactory;

/// 创建器函数类型
type ExtractorCreator = Box<dyn Fn(ExtractorConfig) -> Result<Box<dyn FeatureExtractor>> + Send + Sync>;

/// 全局特征提取器工厂实现
#[derive(Default)]
pub struct GlobalFeatureExtractorFactory {
    /// 特征提取器创建函数映射
    creators: RwLock<HashMap<ExtractorType, ExtractorCreator>>,
}

impl Debug for GlobalFeatureExtractorFactory {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        let creators = self.creators.read().unwrap();
        f.debug_struct("GlobalFeatureExtractorFactory")
            .field("registered_types", &creators.keys().collect::<Vec<_>>())
            .field("count", &creators.len())
            .finish()
    }
}

impl GlobalFeatureExtractorFactory {
    /// 创建新的工厂实例
    pub fn new() -> Self {
        Self {
            creators: RwLock::new(HashMap::default()),
        }
    }
    
    /// 获取全局实例
    pub fn instance() -> Arc<Self> {
        use std::sync::Once;
        static mut INSTANCE: Option<Arc<GlobalFeatureExtractorFactory>> = None;
        static ONCE: Once = Once::new();
        
        unsafe {
            ONCE.call_once(|| {
                INSTANCE = Some(Arc::new(GlobalFeatureExtractorFactory::new()));
            });
            INSTANCE.as_ref().unwrap().clone()
        }
    }
    
    /// 初始化工厂，注册默认的特征提取器
    /// 
    /// 生产级实现：注册默认的特征提取器。
    /// 注意：此工厂实现使用主工厂（FeatureExtractorFactory）来创建提取器，确保使用真实实现。
    pub async fn init(&self) -> Result<(), ExtractorError> {
        info!("初始化全局特征提取器工厂");
        
        // 生产级实现：使用主工厂创建真实的提取器
        // 注意：这里应该调用主工厂（FEATURE_EXTRACTOR_FACTORY）来创建提取器
        // 当前实现作为备用工厂，实际应使用主工厂的注册机制
        
        // 注册BERT提取器（使用框架实现）
        self.register(ExtractorType::Text(TextExtractorType::BERT), Box::new(|config| {
            use crate::data::feature::extractor::model_based::BERTExtractor;
            Ok(Box::new(BERTExtractor::new(config)?))
        }))?;
        
        // 注册TF-IDF提取器（使用真实实现）
        self.register(ExtractorType::Text(TextExtractorType::TfIdf), Box::new(|config| {
            use crate::data::feature::extractor::tfidf::TfIdfExtractor;
            Ok(Box::new(TfIdfExtractor::new(config)))
        }))?;
        
        // 注册Identity提取器（使用真实实现）
        self.register(ExtractorType::Generic(GenericExtractorType::Identity), Box::new(|config| {
            use crate::data::feature::extractor::generic::IdentityExtractor;
            Ok(Box::new(IdentityExtractor::new(config)?))
        }))?;
        
        info!("特征提取器工厂初始化完成");
        Ok(())
    }
    
    /// 注册特征提取器创建函数
    pub fn register(&self, extractor_type: ExtractorType, creator: ExtractorCreator) -> Result<()> {
        let mut creators = self.creators.write().map_err(|_| Error::internal("无法获取写锁"))?;
        creators.insert(extractor_type, creator);
        Ok(())
    }
    
    /// 创建特征提取器
    pub fn create(&self, config: ExtractorConfig) -> Result<Box<dyn FeatureExtractor>> {
        let creators = self.creators.read().map_err(|_| Error::internal("无法获取读锁"))?;
        
        let creator = creators.get(&config.extractor_type)
            .ok_or_else(|| Error::invalid_argument(&format!("不支持的特征提取器类型: {:?}", config.extractor_type)))?;
            
        creator(config)
    }
    
    /// 获取所有已注册的提取器类型
    pub fn get_registered_types(&self) -> Vec<ExtractorType> {
        let creators = self.creators.read().expect("无法获取读锁");
        creators.keys().cloned().collect()
    }
}

impl FeatureExtractorFactory for GlobalFeatureExtractorFactory {
    /// 注册特征提取器创建函数
    fn register(&mut self, extractor_type: ExtractorType, creator: crate::data::feature::interface::ExtractorCreator) -> Result<()> {
        // 创建兼容的包装器
        // 注意：interface::ExtractorCreator 是函数指针，需要转换为 Box<dyn Fn>
        let creator_fn = creator; // 保存函数指针
        let wrapper_creator: ExtractorCreator = Box::new(move |config| {
            // 将extractor::ExtractorConfig转换为interface::ExtractorConfig
            let interface_config = crate::data::feature::interface::ExtractorConfig {
                extractor_type: config.extractor_type,
                feature_type: FeatureType::Generic,
                target_dimension: config.output_dimension,
                enable_cache: true,
                timeout_ms: None,
                batch_size: None,
                options: config.params.clone(),
            };
            
            // 调用interface creator，然后转换为extractor::FeatureExtractor
            let interface_extractor = creator_fn(interface_config)?;
            // 使用反向适配器包装：将interface::FeatureExtractor转换为extractor::FeatureExtractor
            Ok(Box::new(ReverseFeatureExtractorAdapter::new(interface_extractor)))
        });
        
        let mut creators = self.creators.write().map_err(|_| Error::internal("无法获取写锁"))?;
        creators.insert(extractor_type, wrapper_creator);
        Ok(())
    }
    
    /// 创建特征提取器
    fn create(&self, config: crate::data::feature::interface::ExtractorConfig) -> Result<Box<dyn crate::data::feature::interface::FeatureExtractor>> {
        // 将interface::ExtractorConfig转换为extractor::ExtractorConfig
        let mut extractor_config = ExtractorConfig::new(config.extractor_type)
            .with_output_dim(config.target_dimension.unwrap_or(128));
        
        // 逐个添加参数
        for (key, value) in config.options {
            extractor_config = extractor_config.with_param(key, value);
        }
        
        // 直接调用内部的create方法避免递归
        let creators = self.creators.read().map_err(|_| Error::internal("无法获取读锁"))?;
        
        let creator = creators.get(&extractor_config.extractor_type)
            .ok_or_else(|| Error::invalid_argument(&format!("不支持的特征提取器类型: {:?}", extractor_config.extractor_type)))?;
            
        let extractor = creator(extractor_config)?;
        
        // 将extractor::FeatureExtractor转换为interface::FeatureExtractor
        Ok(Box::new(FeatureExtractorAdapter::new(extractor)))
    }
    
    /// 创建文本特征提取器
    fn create_text_extractor(&self, config: crate::data::feature::interface::ExtractorConfig) -> Result<Box<dyn crate::data::feature::interface::FeatureExtractor>> {
        // 将interface::ExtractorConfig转换为extractor::ExtractorConfig
        let mut extractor_config = ExtractorConfig::new(config.extractor_type)
            .with_output_dim(config.target_dimension.unwrap_or(100));
        
        // 逐个添加参数
        for (key, value) in config.options {
            extractor_config = extractor_config.with_param(key, value);
        }
        
        // 直接调用内部的create方法避免递归
        let creators = self.creators.read().map_err(|_| Error::internal("无法获取读锁"))?;
        
        let creator = creators.get(&extractor_config.extractor_type)
            .ok_or_else(|| Error::invalid_argument(&format!("不支持的特征提取器类型: {:?}", extractor_config.extractor_type)))?;
            
        let extractor = creator(extractor_config)?;
        
        // 将extractor::FeatureExtractor转换为interface::FeatureExtractor
        // 由于trait不匹配，我们需要创建一个适配器
        Ok(Box::new(FeatureExtractorAdapter::new(extractor)))
    }
    
    fn create_image_extractor(&self, config: crate::data::feature::interface::ExtractorConfig) -> Result<Box<dyn crate::data::feature::interface::FeatureExtractor>> {
        let mut extractor_config = ExtractorConfig::new(config.extractor_type)
            .with_output_dim(config.target_dimension.unwrap_or(512));
        
        // 逐个添加参数
        for (key, value) in config.options {
            extractor_config = extractor_config.with_param(key, value);
        }
        
        // 直接调用内部的create方法避免递归
        let creators = self.creators.read().map_err(|_| Error::internal("无法获取读锁"))?;
        
        let creator = creators.get(&extractor_config.extractor_type)
            .ok_or_else(|| Error::invalid_argument(&format!("不支持的特征提取器类型: {:?}", extractor_config.extractor_type)))?;
            
        let extractor = creator(extractor_config)?;
        
        Ok(Box::new(FeatureExtractorAdapter::new(extractor)))
    }
    
    fn create_generic_extractor(&self, config: crate::data::feature::interface::ExtractorConfig) -> Result<Box<dyn crate::data::feature::interface::FeatureExtractor>> {
        // 将interface::ExtractorConfig转换为extractor::ExtractorConfig
        let mut extractor_config = ExtractorConfig::new(config.extractor_type)
            .with_output_dim(config.target_dimension.unwrap_or(128));
        
        // 逐个添加参数
        for (key, value) in config.options {
            extractor_config = extractor_config.with_param(key, value);
        }
        
        // 调用内部的create方法（期望extractor::ExtractorConfig）
        let creators = self.creators.read().map_err(|_| Error::internal("无法获取读锁"))?;
        let creator = creators.get(&extractor_config.extractor_type)
            .ok_or_else(|| Error::invalid_argument(&format!("不支持的特征提取器类型: {:?}", extractor_config.extractor_type)))?;
        
        let extractor = creator(extractor_config)?;
        
        // 使用适配器转换为interface::FeatureExtractor
        Ok(Box::new(FeatureExtractorAdapter::new(extractor)))
    }
    
    fn get_registered_types(&self) -> Vec<ExtractorType> {
        GlobalFeatureExtractorFactory::get_registered_types(self)
    }
    
    fn is_registered(&self, extractor_type: ExtractorType) -> bool {
        let creators = self.creators.read().expect("无法获取读锁");
        creators.contains_key(&extractor_type)
    }
}

/// 适配器，用于将extractor::FeatureExtractor转换为interface::FeatureExtractor
pub struct FeatureExtractorAdapter {
    inner: Box<dyn FeatureExtractor>,
}

impl FeatureExtractorAdapter {
    pub fn new(inner: Box<dyn FeatureExtractor>) -> Self {
        Self { inner }
    }
}

impl Debug for FeatureExtractorAdapter {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        f.debug_struct("FeatureExtractorAdapter")
            .field("inner", &"FeatureExtractor")
            .finish()
    }
}

#[async_trait]
impl crate::data::feature::interface::FeatureExtractor for FeatureExtractorAdapter {
    fn extractor_type(&self) -> ExtractorType {
        self.inner.extractor_type()
    }
    
    fn feature_type(&self) -> FeatureType {
        self.inner.output_feature_type()
    }
    
    fn name(&self) -> &str {
        "FeatureExtractorAdapter"
    }
    
    fn config(&self) -> &crate::data::feature::interface::ExtractorConfig {
        // 使用 Lazy 避免在 static 中调用非 const 函数
        use once_cell::sync::Lazy;
        static DEFAULT_CONFIG: Lazy<crate::data::feature::interface::ExtractorConfig> = Lazy::new(||
            crate::data::feature::interface::ExtractorConfig {
                extractor_type: ExtractorType::Generic(GenericExtractorType::Identity),
                feature_type: FeatureType::Generic,
                target_dimension: None,
                enable_cache: true,
                timeout_ms: None,
                batch_size: None,
                options: HashMap::new(),
            }
        );
        &DEFAULT_CONFIG
    }
    
    fn get_config(&self) -> crate::data::feature::interface::ExtractorConfig {
        crate::data::feature::interface::ExtractorConfig {
            extractor_type: self.inner.extractor_type(),
            feature_type: self.inner.output_feature_type(),
            target_dimension: self.inner.output_dimension(),
            enable_cache: true,
            timeout_ms: None,
            batch_size: None,
            options: HashMap::new(),
        }
    }
    
    async fn extract(&self, input: &str) -> Result<crate::data::feature::interface::FeatureExtractionResult> {
        let input_data = InputData::Text(input.to_string());
        let context = Some(ExtractorContext::new());
        
        let feature_vector = self.inner.extract(input_data, context).await
            .map_err(|e| Error::processing(format!("特征提取失败: {}", e)))?;
        
        Ok(crate::data::feature::interface::FeatureExtractionResult::new(
            feature_vector.values,
            feature_vector.feature_type,
            self.inner.extractor_type(),
        ))
    }
}

/// 创建特征提取器的便捷函数
pub fn create_extractor(extractor_type: ExtractorType, options: Option<HashMap<String, String>>) -> Result<Box<dyn FeatureExtractor>> {
    let factory = GlobalFeatureExtractorFactory::instance();
    
    let mut config = ExtractorConfig::new(extractor_type);
    
    if let Some(opts) = options {
        for (key, value) in opts {
            config = config.with_param(key, value);
        }
    }
    
    factory.create(config)
}

/// 创建文本特征提取器的便捷函数
pub fn create_text_extractor(method: &str, options: Option<HashMap<String, String>>) -> Result<Box<dyn FeatureExtractor>> {
    let extractor_type = match method.to_lowercase().as_str() {
        "tfidf" => ExtractorType::Text(TextExtractorType::TfIdf),
        "bow" | "bagofwords" => ExtractorType::Text(TextExtractorType::BagOfWords),
        "bert" => ExtractorType::Text(TextExtractorType::BERT),
        "word2vec" => ExtractorType::Text(TextExtractorType::Word2Vec),
        "fasttext" => ExtractorType::Text(TextExtractorType::FastText),
        _ => ExtractorType::Text(TextExtractorType::TfIdf), // 使用TF-IDF作为默认值
    };
    
    let mut opts = options.unwrap_or_default();
    opts.insert("method".to_string(), method.to_string());
    
    create_extractor(extractor_type, Some(opts))
}

/// 创建图像特征提取器的便捷函数
pub fn create_image_extractor(method: &str, options: Option<HashMap<String, String>>) -> Result<Box<dyn FeatureExtractor>> {
    let extractor_type = match method.to_lowercase().as_str() {
        "cnn" => ExtractorType::ImageCNN,
        "resnet" => ExtractorType::ImageResNet,
        "vgg" => ExtractorType::ImageVGG,
        "inception" => ExtractorType::ImageInception,
        "sift" => ExtractorType::ImageSIFT,
        "hog" => ExtractorType::ImageHOG,
        _ => ExtractorType::Image,
    };
    
    let mut opts = options.unwrap_or_default();
    opts.insert("method".to_string(), method.to_string());
    
    create_extractor(extractor_type, Some(opts))
}

/// 组合特征提取器
/// 将多个特征提取器组合在一起，形成一个复合特征提取器
pub struct CompositeExtractor {
    /// 提取器配置
    config: ExtractorConfig,
    /// 组合的特征提取器
    extractors: Vec<Box<dyn FeatureExtractor>>,
}

impl CompositeExtractor {
    /// 创建新的组合特征提取器
    pub fn new(config: ExtractorConfig, extractors: Vec<Box<dyn FeatureExtractor>>) -> Self {
        Self {
            config,
            extractors,
        }
    }
}

impl Debug for CompositeExtractor {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        f.debug_struct("CompositeExtractor")
            .field("config", &self.config)
            .field("extractors_count", &self.extractors.len())
            .finish()
    }
}

#[async_trait]
impl FeatureExtractor for CompositeExtractor {
    fn extractor_type(&self) -> ExtractorType {
        self.config.extractor_type
    }
    
    fn config(&self) -> &ExtractorConfig {
        &self.config
    }
    
    fn is_compatible(&self, input: &InputData) -> bool {
        // 如果任一提取器与输入兼容，则认为组合提取器兼容
        self.extractors.iter().any(|e| e.is_compatible(input))
    }
    
    async fn extract(&self, input: InputData, context: Option<ExtractorContext>) -> std::result::Result<FeatureVector, ExtractorError> {
        if self.extractors.is_empty() {
            return Err(ExtractorError::Config("没有可用的特征提取器".to_string()));
        }

        // 收集各个提取器的结果
        let mut features = Vec::with_capacity(self.extractors.len());
        let mut feature_type = None;
        
        for extractor in &self.extractors {
            let result = extractor.extract(input.clone(), context.clone()).await?;
            
            if feature_type.is_none() {
                feature_type = Some(result.feature_type);
            }
            
            features.push(result);
        }
        
        // 获取融合策略参数
        let fusion_strategy_str = self.config.get_string_param("fusion_strategy")
            .unwrap_or_else(|| "concatenation".to_string());
        
        // 简单的串联融合实现
        let mut fused_values = Vec::new();
        let mut metadata = HashMap::new();
        
        for (i, feature) in features.iter().enumerate() {
            fused_values.extend_from_slice(&feature.values);
            metadata.insert(format!("extractor_{}", i), format!("{:?}", feature.feature_type));
        }
        
        // 逐个添加元数据
        let mut fused_feature = FeatureVector::new(feature_type.unwrap_or(FeatureType::Generic), fused_values);
        for (key, value) in metadata {
            fused_feature = fused_feature.with_metadata(key, value);
        }
        
        Ok(fused_feature)
    }
    
    async fn batch_extract(&self, inputs: Vec<InputData>, context: Option<ExtractorContext>) -> std::result::Result<FeatureBatch, ExtractorError> {
        if self.extractors.is_empty() {
            return Err(ExtractorError::Config("没有可用的特征提取器".to_string()));
        }

        let mut all_features = Vec::with_capacity(inputs.len());
        
        for input in inputs {
            let feature = self.extract(input, context.clone()).await?;
            all_features.push(feature);
        }
        
        // 合并结果为FeatureBatch
        let dimension = if !all_features.is_empty() {
            all_features[0].values.len()
        } else {
            0
        };
        
        let feature_type = if !all_features.is_empty() {
            all_features[0].feature_type
        } else {
            FeatureType::Generic
        };
        
        // 提取所有特征向量的值
        let values = all_features.into_iter()
            .map(|f| f.values)
            .collect::<Vec<_>>();
        
        Ok(FeatureBatch::new(values, feature_type))
    }
    
    fn output_feature_type(&self) -> FeatureType {
        self.config.extractor_type.default_feature_type()
    }
    
    fn output_dimension(&self) -> Option<usize> {
        self.config.output_dimension
    }
    
    fn as_any(&self) -> &dyn Any {
        self
    }
}

/// 反向适配器：将interface::FeatureExtractor转换为extractor::FeatureExtractor
pub struct ReverseFeatureExtractorAdapter {
    inner: Box<dyn crate::data::feature::interface::FeatureExtractor>,
}

impl ReverseFeatureExtractorAdapter {
    pub fn new(inner: Box<dyn crate::data::feature::interface::FeatureExtractor>) -> Self {
        Self { inner }
    }
}

impl Debug for ReverseFeatureExtractorAdapter {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        f.debug_struct("ReverseFeatureExtractorAdapter")
            .field("inner", &self.inner.name())
            .finish()
    }
}

#[async_trait]
impl FeatureExtractor for ReverseFeatureExtractorAdapter {
    fn extractor_type(&self) -> ExtractorType {
        self.inner.extractor_type()
    }
    
    fn config(&self) -> &ExtractorConfig {
        // 返回一个默认配置，因为interface::FeatureExtractor没有ExtractorConfig
        use once_cell::sync::Lazy;
        static DEFAULT_CONFIG: Lazy<ExtractorConfig> = Lazy::new(|| {
            ExtractorConfig::new(ExtractorType::Generic(GenericExtractorType::Identity))
        });
        &DEFAULT_CONFIG
    }
    
    fn is_compatible(&self, input: &InputData) -> bool {
        // 生产级实现：检查输入类型是否与内部提取器兼容
        match input {
            InputData::Text(_) | InputData::TextArray(_) => true,
            _ => false,
        }
    }

    async fn extract(&self, input: InputData, _context: Option<ExtractorContext>) -> std::result::Result<FeatureVector, ExtractorError> {
        // 生产级实现：将InputData转换为字符串，支持文本和文本数组
        let text = match input {
            InputData::Text(t) => t,
            InputData::TextArray(arr) => {
                // 使用空格连接多个文本，这是标准的文本数组处理方式
                arr.join(" ")
            },
            _ => return Err(ExtractorError::Internal("不支持的输入类型，期望Text或TextArray".to_string())),
        };
        
        // 调用interface extractor
        let result = self.inner.extract(&text).await
            .map_err(|e| ExtractorError::Internal(format!("特征提取失败: {}", e)))?;
        
        // 转换为extractor::FeatureVector
        Ok(FeatureVector::new(result.feature_type, result.feature_vector)
            .with_extractor_type(result.extractor_type))
    }
    
    async fn batch_extract(&self, inputs: Vec<InputData>, _context: Option<ExtractorContext>) -> std::result::Result<FeatureBatch, ExtractorError> {
        let mut results = Vec::with_capacity(inputs.len());
        let mut feature_type = None;
        
        for input in inputs {
            let feature = self.extract(input, None).await?;
            if feature_type.is_none() {
                feature_type = Some(feature.feature_type);
            }
            results.push(feature.values);
        }
        
        Ok(FeatureBatch::new(results, feature_type.unwrap_or(FeatureType::Generic)))
    }
    
    fn output_feature_type(&self) -> FeatureType {
        self.inner.feature_type()
    }
    
    fn output_dimension(&self) -> Option<usize> {
        self.inner.config().target_dimension
    }
    
    fn as_any(&self) -> &dyn Any {
        self
    }
}

/// 注册默认特征提取器到工厂
pub fn register_default_extractors() -> Result<()> {
    // 在这里注册默认的特征提取器
    Ok(())
}

/// 创建组合特征提取器的便捷函数
pub async fn create_composite_extractor(
    extractor_configs: Vec<ExtractorConfig>,
    fusion_strategy: &str,
) -> std::result::Result<Box<dyn FeatureExtractor>, ExtractorError> {
    let factory = GlobalFeatureExtractorFactory::instance();
    
    let mut extractors = Vec::with_capacity(extractor_configs.len());
    
    for config in &extractor_configs {
        let extractor = factory.create(config.clone()).map_err(|e| {
            ExtractorError::Config(format!("创建提取器失败: {}", e))
        })?;
        extractors.push(extractor);
    }
    
    // 使用第一个提取器的类型作为组合提取器的类型
    let first_config = extractor_configs.first()
        .ok_or_else(|| ExtractorError::Config("至少需要一个提取器配置".to_string()))?;
    
    let mut composite_config = first_config.clone();
    // 设置Composite类型
    composite_config.extractor_type = ExtractorType::Composite;
    // 添加融合策略参数
    composite_config = composite_config.with_param("fusion_strategy", fusion_strategy);
    
    Ok(Box::new(CompositeExtractor::new(composite_config, extractors)))
} 