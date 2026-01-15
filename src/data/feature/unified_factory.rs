// 统一特征提取工厂
// 整合所有特征提取器的工厂实现

use std::collections::HashMap;
use std::sync::{Arc, RwLock};
use once_cell::sync::Lazy;
use tracing::{info, debug};

use crate::Result;
use crate::Error;
use crate::data::feature::interface::{FeatureExtractor as UnifiedFeatureExtractor, FeatureExtractorFactory};
use crate::data::feature::types::ExtractorType;
use crate::data::feature::config::ExtractorConfig;
use crate::data::feature::extractor::FeatureExtractor;
use crate::data::feature::factory_impl::GlobalFeatureExtractorFactory;
use crate::data::feature::extractor::{
    ExtractorError, InputData, ExtractorContext, FeatureVector, FeatureBatch,
};

// 全局特征提取器工厂
static GLOBAL_EXTRACTOR_FACTORY: Lazy<RwLock<Option<Arc<UnifiedFeatureExtractorFactory>>>> = Lazy::new(|| {
    RwLock::new(None)
});

/// 创建默认的特征提取器工厂
pub fn create_default_extractor_factory() -> Arc<UnifiedFeatureExtractorFactory> {
    // 检查全局工厂是否已初始化
    let read_guard = GLOBAL_EXTRACTOR_FACTORY.read().unwrap();
    if let Some(factory) = &*read_guard {
        return factory.clone();
    }
    drop(read_guard);
    
    // 如果未初始化，创建默认工厂
    let factory = UnifiedFeatureExtractorFactory::new();
    
    // 在这里注册默认的特征提取器工厂
    // 将在后续实现中添加
    
    let factory_arc = Arc::new(factory);
    
    // 保存到全局变量
    let mut write_guard = GLOBAL_EXTRACTOR_FACTORY.write().unwrap();
    *write_guard = Some(factory_arc.clone());
    
    factory_arc
}

/// 获取全局特征提取器工厂
pub fn get_global_extractor_factory() -> Option<Arc<UnifiedFeatureExtractorFactory>> {
    GLOBAL_EXTRACTOR_FACTORY.read().unwrap().clone()
}

/// 注册特征提取器到全局工厂
pub fn register_extractor_factory<F>(extractor_type: ExtractorType, factory: F) -> Result<()>
where
    F: FeatureExtractorFactory + 'static,
{
    let global_factory = match get_global_extractor_factory() {
        Some(f) => f,
        None => create_default_extractor_factory(),
    };
    
    global_factory.register_factory(extractor_type, Box::new(factory))
}

/// 根据配置创建特征提取器
pub fn create_extractor(config: &ExtractorConfig) -> Result<Box<dyn UnifiedFeatureExtractor>> {
    let factory = match get_global_extractor_factory() {
        Some(f) => f,
        None => create_default_extractor_factory(),
    };
    
    factory.create(config)
}

/// 统一特征提取器工厂
pub struct UnifiedFeatureExtractorFactory {
    factories: RwLock<HashMap<ExtractorType, Box<dyn FeatureExtractorFactory>>>,
}

impl UnifiedFeatureExtractorFactory {
    /// 创建新的统一特征提取器工厂
    pub fn new() -> Self {
        Self {
            factories: RwLock::new(HashMap::new()),
        }
    }
    
    /// 注册特征提取器工厂
    pub fn register_factory(&self, extractor_type: ExtractorType, factory: Box<dyn FeatureExtractorFactory>) -> Result<()> {
        let mut factories = match self.factories.write() {
            Ok(guard) => guard,
            Err(poisoned) => poisoned.into_inner(),
        };
        
        if factories.contains_key(&extractor_type) {
            return Err(Error::already_exists(&format!("特征提取器工厂已存在：{}", extractor_type)));
        }
        
        factories.insert(extractor_type, factory);
        Ok(())
    }
    
    /// 创建特征提取器
    pub fn create(&self, config: &ExtractorConfig) -> Result<Box<dyn UnifiedFeatureExtractor>> {
        use crate::data::feature::interface::ExtractorConfig as InterfaceExtractorConfig;
        use crate::data::feature::types::FeatureType;
        
        let factories = match self.factories.read() {
            Ok(guard) => guard,
            Err(poisoned) => poisoned.into_inner(),
        };
        
        let factory = factories.get(&config.extractor_type)
            .ok_or_else(|| Error::invalid_argument(&format!("未找到特征提取器工厂：{}", config.extractor_type)))?;
        
        // 将 config::ExtractorConfig 转换为 interface::ExtractorConfig
        let interface_config = InterfaceExtractorConfig {
            extractor_type: config.extractor_type.clone(),
            feature_type: FeatureType::Generic, // 默认类型
            target_dimension: Some(config.dimension),
            enable_cache: true,
            timeout_ms: None,
            batch_size: None,
            options: config.params.iter()
                .map(|(k, v)| {
                    let value = match v {
                        serde_json::Value::String(s) => s.clone(),
                        _ => v.to_string(),
                    };
                    (k.clone(), value)
                })
                .collect(),
        };
        
        factory.create(interface_config)
    }
    
    /// 获取支持的特征提取器类型
    pub fn get_supported_types(&self) -> Vec<ExtractorType> {
        let factories = match self.factories.read() {
            Ok(guard) => guard,
            Err(poisoned) => poisoned.into_inner(),
        };
        
        factories.keys().cloned().collect()
    }
    
    /// 检查是否支持指定类型的提取器
    pub fn supports_extractor_type(&self, extractor_type: ExtractorType) -> bool {
        let factories = match self.factories.read() {
            Ok(guard) => guard,
            Err(poisoned) => poisoned.into_inner(),
        };
        
        factories.contains_key(&extractor_type)
    }
    
    /// 获取工厂数量
    pub fn factory_count(&self) -> usize {
        let factories = match self.factories.read() {
            Ok(guard) => guard,
            Err(poisoned) => poisoned.into_inner(),
        };
        
        factories.len()
    }
}

impl Default for UnifiedFeatureExtractorFactory {
    fn default() -> Self {
        Self::new()
    }
}

/// 特征提取器管理器
/// 管理所有类型的特征提取器，提供统一的创建和使用接口
pub struct FeatureExtractorManager {
    /// 特征提取器工厂
    factory: Arc<GlobalFeatureExtractorFactory>,
    /// 缓存的特征提取器
    extractors: RwLock<HashMap<String, Box<dyn FeatureExtractor>>>,
}

impl FeatureExtractorManager {
    /// 创建新的特征提取器管理器
    pub fn new() -> Self {
        Self {
            factory: GlobalFeatureExtractorFactory::instance(),
            extractors: RwLock::new(HashMap::new()),
        }
    }
    
    /// 获取全局实例
    pub fn instance() -> Arc<Self> {
        static INSTANCE: Lazy<Arc<FeatureExtractorManager>> = Lazy::new(|| {
            Arc::new(FeatureExtractorManager::new())
        });
        INSTANCE.clone()
    }
    
    /// 初始化管理器
    pub async fn init(&self) -> std::result::Result<(), ExtractorError> {
        info!("初始化特征提取器管理器");
        
        // 确保工厂已初始化
        self.factory.init().await?;
        
        info!("特征提取器管理器初始化完成");
        Ok(())
    }
    
    /// 获取特征提取器实例（带缓存）
    pub async fn get_extractor(&self, config: crate::data::feature::extractor::ExtractorConfig) -> std::result::Result<Box<dyn FeatureExtractor>, ExtractorError> {
        let extractor_id = self.generate_extractor_id(&config);
        
        // 检查缓存，但不直接返回，而是重新创建实例
        {
            let cache = self.extractors.read().map_err(|e| {
                ExtractorError::Internal(format!("无法获取提取器缓存读锁: {}", e))
            })?;
            
            if cache.contains_key(&extractor_id) {
                debug!("提取器类型已缓存，重新创建实例: {}", extractor_id);
                // 直接创建新实例，避免克隆问题
                return self.factory.create(config).map_err(|e| {
                    ExtractorError::Config(format!("创建提取器失败: {}", e))
                });
            }
        }
        
        // 从工厂创建新的提取器
        let extractor = self.factory.create(config.clone()).map_err(|e| {
            ExtractorError::Config(format!("创建提取器失败: {}", e))
        })?;
        
        // 创建一个用于缓存的标记实例
        {
            let mut cache = self.extractors.write().map_err(|e| {
                ExtractorError::Internal(format!("无法获取提取器缓存写锁: {}", e))
            })?;
            
            // 存储一个产品级的提取器实例
            let cache_extractor = self.factory.create(config.clone()).map_err(|e| {
                ExtractorError::Config(format!("创建缓存提取器失败: {}", e))
            })?;
            cache.insert(extractor_id.clone(), cache_extractor);
        }
        
        debug!("创建并缓存特征提取器: {}", extractor_id);
        Ok(extractor)
    }
    
    /// 创建组合特征提取器
    pub async fn create_composite_extractor(
        &self,
        configs: Vec<crate::data::feature::extractor::ExtractorConfig>,
        fusion_strategy: &str,
    ) -> Result<Box<dyn FeatureExtractor>> {
        crate::data::feature::factory_impl::create_composite_extractor(configs, fusion_strategy)
            .await
            .map_err(|e| e.into())
    }
    
    /// 生成提取器唯一标识符
    fn generate_extractor_id(&self, config: &crate::data::feature::extractor::ExtractorConfig) -> String {
        let name = &config.name;
        let extractor_type = format!("{:?}", config.extractor_type);
        
        // 简单地组合信息生成ID
        format!("{}:{}", name, extractor_type)
    }
    
    /// 清除缓存
    pub fn clear_cache(&self) -> Result<(), ExtractorError> {
        let mut cache = self.extractors.write().map_err(|e| {
            ExtractorError::Internal(format!("无法获取提取器缓存写锁: {}", e))
        })?;
        
        cache.clear();
        debug!("已清除特征提取器缓存");
        
        Ok(())
    }
    
    /// 从输入数据中提取特征
    pub async fn extract_features(
        &self, 
        config: crate::data::feature::extractor::ExtractorConfig, 
        input: InputData,
        context: Option<ExtractorContext>,
    ) -> Result<FeatureVector> {
        let extractor = self.get_extractor(config).await.map_err(|e| e.into())?;
        extractor.extract(input, context).await.map_err(|e| e.into())
    }
    
    /// 批量提取特征
    pub async fn batch_extract_features(
        &self,
        config: crate::data::feature::extractor::ExtractorConfig,
        inputs: Vec<InputData>,
        context: Option<ExtractorContext>,
    ) -> Result<FeatureBatch> {
        let extractor = self.get_extractor(config).await.map_err(|e| e.into())?;
        extractor.batch_extract(inputs, context).await.map_err(|e| e.into())
    }
    
    /// 提取特征并计算相似度
    pub async fn compute_similarity(
        &self,
        config: crate::data::feature::extractor::ExtractorConfig,
        input1: InputData,
        input2: InputData,
        context: Option<ExtractorContext>,
    ) -> Result<f32> {
        let extractor = self.get_extractor(config).await.map_err(|e| e.into())?;
        
        let feature1 = extractor.extract(input1, context.clone()).await.map_err(|e| e.into())?;
        let feature2 = extractor.extract(input2, context).await.map_err(|e| e.into())?;
        
        // 计算余弦相似度
        let dot_product: f32 = feature1.values.iter()
            .zip(feature2.values.iter())
            .map(|(&a, &b)| a * b)
            .sum();
            
        let norm1: f32 = feature1.values.iter()
            .map(|&a| a * a)
            .sum::<f32>()
            .sqrt();
            
        let norm2: f32 = feature2.values.iter()
            .map(|&a| a * a)
            .sum::<f32>()
            .sqrt();
        
        if norm1 == 0.0 || norm2 == 0.0 {
            return Ok(0.0);
        }
        
        Ok(dot_product / (norm1 * norm2))
    }
}

impl Default for FeatureExtractorManager {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_create_default_factory() {
        let factory = create_default_extractor_factory();
        assert_eq!(factory.factory_count(), 0);
    }
    
    #[tokio::test]
    async fn test_manager_init() {
        let manager = FeatureExtractorManager::new();
        assert!(manager.init().await.is_ok());
    }
    
    #[tokio::test]
    async fn test_manager_clear_cache() {
        let manager = FeatureExtractorManager::new();
        assert!(manager.clear_cache().is_ok());
    }
} 