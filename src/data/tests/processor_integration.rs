//! 处理器融合测试
//! 测试旧系统处理器通过适配器在新系统中的使用

#[cfg(test)]
#[allow(deprecated)]
mod tests {
    use std::collections::HashMap;
    use crate::data::pipeline::{ProcessorContext, ProcessorResult, Processor};
    use crate::data::processor::adapter::LegacyProcessorAdapter;
    use crate::data::processor::factory_adapter::LegacyProcessorFactoryAdapter;
    use crate::data::DataValue;
    use crate::error::Result;
    
    #[test]
    fn test_legacy_adapter_basic() -> Result<()> {
        // 创建一个简单的测试上下文
        let mut data = HashMap::new();
        data.insert("field1".to_string(), DataValue::String("Hello World".to_string()));
        data.insert("field2".to_string(), DataValue::Number(42.0));
        
        let mut metadata = HashMap::new();
        metadata.insert("test".to_string(), "value".to_string());
        
        let context = ProcessorContext { data, metadata };
        
        // 创建一个转换处理器适配器
        let mut config = HashMap::new();
        config.insert("field".to_string(), "field1".to_string());
        config.insert("transform_type".to_string(), "upper".to_string());
        
        let factory = LegacyProcessorFactoryAdapter::new("transform");
        let mut processor = factory.create(config)?;
        
        // 执行处理并验证结果
        let result = processor.process(context)?;
        
        assert!(result.data.contains_key("field1"), "结果应该包含field1字段");
        
        if let Some(DataValue::String(value)) = result.data.get("field1") {
            assert_eq!(value, "HELLO WORLD", "字段应该被转换为大写");
        } else {
            panic!("field1字段应该是字符串类型");
        }
        
        Ok(())
    }
    
    #[test]
    fn test_factory_adapter_integration() -> Result<()> {
        // 测试工厂适配器创建处理器并注册
        let mut registry = HashMap::new();
        
        // 注册几个旧系统处理器
        registry.insert(
            "legacy_normalize".to_string(),
            Box::new(LegacyProcessorFactoryAdapter::new("normalize"))
        );
        
        registry.insert(
            "legacy_tokenize".to_string(),
            Box::new(LegacyProcessorFactoryAdapter::new("tokenize"))
        );
        
        // 验证是否能够从注册表中获取处理器
        assert!(registry.contains_key("legacy_normalize"), "注册表应该包含legacy_normalize");
        assert!(registry.contains_key("legacy_tokenize"), "注册表应该包含legacy_tokenize");
        
        // 测试创建处理器实例
        let params = HashMap::new();
        let factory = registry.get("legacy_normalize").unwrap();
        let processor = factory.create(params)?;
        
        assert!(processor.get_stats().processor_name.contains("legacy"), 
               "处理器名称应该包含legacy前缀");
        
        Ok(())
    }
    
    #[test]
    fn test_processor_stats() -> Result<()> {
        // 测试处理器统计信息
        let mut config = HashMap::new();
        config.insert("field".to_string(), "text".to_string());
        config.insert("transform_type".to_string(), "upper".to_string());
        
        let mut processor = LegacyProcessorAdapter::new(
            "test_processor", 
            "transform", 
            config
        );
        
        // 验证初始统计信息
        assert_eq!(processor.get_stats().processed_count, 0, "初始处理计数应该为0");
        
        // 创建测试上下文
        let mut data = HashMap::new();
        data.insert("text".to_string(), DataValue::String("test".to_string()));
        
        let context = ProcessorContext { 
            data, 
            metadata: HashMap::new() 
        };
        
        // 执行处理
        let _ = processor.process(context)?;
        
        // 验证处理后的统计信息
        assert_eq!(processor.get_stats().processed_count, 1, "处理计数应该为1");
        
        // 重置统计信息
        processor.reset_stats();
        
        // 验证重置后的统计信息
        assert_eq!(processor.get_stats().processed_count, 0, "重置后处理计数应该为0");
        
        Ok(())
    }
} 