pub mod text;

use std::collections::HashMap;
use crate::data::pipeline::ProcessorFactory;
use crate::error::Result;
use crate::data::processor::factory_adapter::LegacyProcessorFactoryAdapter;

/// 注册所有处理器
pub fn register_all_processors(registry: &mut HashMap<String, Box<dyn ProcessorFactory>>) -> Result<()> {
    // 注册文本处理器
    text::register_processors(registry);
    
    // 注册适配的旧系统处理器
    register_legacy_processors(registry)?;
    
    Ok(())
}

/// 注册旧系统处理器
#[allow(deprecated)]
fn register_legacy_processors(registry: &mut HashMap<String, Box<dyn ProcessorFactory>>) -> Result<()> {
    // 注册normalize处理器
    registry.insert(
        "legacy_normalize".to_string(),
        Box::new(LegacyProcessorFactoryAdapter::new("normalize"))
    );
    
    // 注册tokenize处理器
    registry.insert(
        "legacy_tokenize".to_string(),
        Box::new(LegacyProcessorFactoryAdapter::new("tokenize"))
    );
    
    // 注册encode处理器
    registry.insert(
        "legacy_encode".to_string(),
        Box::new(LegacyProcessorFactoryAdapter::new("encode"))
    );
    
    // 注册transform处理器
    registry.insert(
        "legacy_transform".to_string(),
        Box::new(LegacyProcessorFactoryAdapter::new("transform"))
    );
    
    // 注册filter处理器
    registry.insert(
        "legacy_filter".to_string(),
        Box::new(LegacyProcessorFactoryAdapter::new("filter"))
    );
    
    // 注册augment处理器
    registry.insert(
        "legacy_augment".to_string(),
        Box::new(LegacyProcessorFactoryAdapter::new("augment"))
    );
    
    Ok(())
} 