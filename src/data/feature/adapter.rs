use crate::data::feature::types::FeatureType;
use crate::error::Result;

/// 特征适配器特性，用于在不同特征类型间进行转换
pub trait FeatureAdapter: Send + Sync {
    /// 源特征类型
    fn source_type(&self) -> FeatureType;
    
    /// 目标特征类型
    fn target_type(&self) -> FeatureType;
    
    /// 将源特征转换为目标特征
    fn adapt(&self, source: &[f32]) -> Result<Vec<f32>>;
}

/// 通用特征适配器
pub struct GenericAdapter {
    source: FeatureType,
    target: FeatureType,
    adapter_fn: Box<dyn Fn(&[f32]) -> Result<Vec<f32>> + Send + Sync>,
}

impl GenericAdapter {
    /// 创建新的通用适配器
    pub fn new<F>(source: FeatureType, target: FeatureType, adapter_fn: F) -> Self
    where
        F: Fn(&[f32]) -> Result<Vec<f32>> + Send + Sync + 'static,
    {
        Self {
            source,
            target,
            adapter_fn: Box::new(adapter_fn),
        }
    }
}

impl FeatureAdapter for GenericAdapter {
    fn source_type(&self) -> FeatureType {
        self.source
    }
    
    fn target_type(&self) -> FeatureType {
        self.target
    }
    
    fn adapt(&self, source: &[f32]) -> Result<Vec<f32>> {
        (self.adapter_fn)(source)
    }
}

/// 特征适配器工厂
pub struct AdapterFactory {
    adapters: Vec<Box<dyn FeatureAdapter>>,
}

impl AdapterFactory {
    /// 创建新的适配器工厂
    pub fn new() -> Self {
        Self {
            adapters: Vec::new(),
        }
    }
    
    /// 注册适配器
    pub fn register<A: FeatureAdapter + 'static>(&mut self, adapter: A) {
        self.adapters.push(Box::new(adapter));
    }
    
    /// 查找适配器
    pub fn find_adapter(&self, source: FeatureType, target: FeatureType) -> Option<&dyn FeatureAdapter> {
        self.adapters.iter()
            .find(|a| a.source_type() == source && a.target_type() == target)
            .map(|a| a.as_ref())
    }
}

impl Default for AdapterFactory {
    fn default() -> Self {
        Self::new()
    }
} 