use crate::data::text_features::{MixedFeatureExtractor, MixedFeatureConfig, TextFeatureConfig, TextFeatureMethod};
use serde_json::Value;
use std::time::Instant;

/// 生成测试数据
fn generate_test_data(count: usize) -> Vec<Value> {
    let mut data = Vec::with_capacity(count);
    
    // 产品数据
    for i in 0..count {
        let mut obj = serde_json::Map::new();
        
        // 标题字段 (短文本)
        obj.insert("title".to_string(), Value::String(format!(
            "产品 #{} - 高性能智能设备", i
        )));
        
        // 描述字段 (长文本)
        obj.insert("description".to_string(), Value::String(format!(
            "这是一款高性能智能设备，具有多种功能和特性。它采用了最新的技术和材料，\
            提供卓越的用户体验和性能表现。产品支持多种连接方式，包括WiFi、蓝牙和USB。\
            它的电池续航时间长达10小时，满足用户的日常使用需求。此外，产品还具有防水防尘功能，\
            适合各种环境下使用。产品的设计简洁美观，符合现代审美标准。我们提供12个月的质保服务，\
            确保用户能够放心使用。产品 #{} 是您理想的选择。", i
        )));
        
        // 关键词字段
        obj.insert("keywords".to_string(), Value::String(format!(
            "智能设备 高性能 便携 耐用 时尚 #{}", i
        )));
        
        // 产品代码
        obj.insert("product_code".to_string(), Value::String(format!(
            "PRD{:06}", i
        )));
        
        // 价格
        obj.insert("price".to_string(), Value::Number(serde_json::Number::from_f64(
            100.0 + (i as f64 % 10.0) * 50.0
        ).unwrap()));
        
        // 库存
        obj.insert("stock".to_string(), Value::Number(serde_json::Number::from(
            10 + i % 90
        )));
        
        // 类别
        let categories = ["电子产品", "智能家居", "办公设备", "娱乐设备", "健康设备"];
        obj.insert("category".to_string(), Value::String(
            categories[i % categories.len()].to_string()
        ));
        
        data.push(Value::Object(obj));
    }
    
    data
}

#[test]
fn test_improved_algorithm_selection() {
    // 生成测试数据
    let data = generate_test_data(50);
    
    // 创建特征提取器配置
    let config = MixedFeatureConfig {
        text_config: TextFeatureConfig {
            method: TextFeatureMethod::Auto,
            max_features: 100,
            case_sensitive: false,
            stop_words: None,
            custom_params: None,
            enable_cache: true,
            cache_size_limit: 1000,
            parallel_processing: true,
            batch_size: 10,
        },
        normalize_numeric: true,
        one_hot_categorical: true,
        auto_detect_types: true,
        auto_select_methods: true,
        max_dimensions: 200,
    };
    
    // 创建特征提取器
    let mut extractor = MixedFeatureExtractor::from_config(config);
    
    // 初始化提取器
    extractor.detect_field_types(&data).unwrap();
    extractor.compute_numeric_stats(&data).unwrap();
    extractor.collect_categorical_values(&data).unwrap();
    
    // 测试全局方法选择
    println!("\n=== 测试全局方法选择 ===");
    let start_time = Instant::now();
    let global_method = extractor.auto_select_best_method(&data).unwrap();
    let global_duration = start_time.elapsed();
    println!("全局最佳方法: {:?}, 耗时: {:?}", global_method, global_duration);
    
    // 测试字段特定方法选择
    println!("\n=== 测试字段特定方法选择 ===");
    let start_time = Instant::now();
    let field_methods = extractor.auto_select_best_method_combination(&data).unwrap();
    let field_duration = start_time.elapsed();
    println!("字段特定方法选择耗时: {:?}", field_duration);
    
    // 验证字段特定方法选择结果
    assert!(!field_methods.is_empty(), "字段特定方法选择应该返回非空结果");
    
    // 测试使用全局方法提取特征
    println!("\n=== 测试使用全局方法提取特征 ===");
    extractor.text_extractor.config.method = global_method;
    let start_time = Instant::now();
    let global_features = extractor.batch_process_json(&data, true).unwrap();
    let global_extract_duration = start_time.elapsed();
    println!("使用全局方法提取特征耗时: {:?}", global_extract_duration);
    
    // 测试使用字段特定方法提取特征
    println!("\n=== 测试使用字段特定方法提取特征 ===");
    let start_time = Instant::now();
    let field_features = extractor.batch_process_json_with_field_methods(&data, true, &field_methods).unwrap();
    let field_extract_duration = start_time.elapsed();
    println!("使用字段特定方法提取特征耗时: {:?}", field_extract_duration);
    
    // 验证特征提取结果
    assert_eq!(global_features.len(), data.len(), "全局方法应该为每个数据项提取特征");
    assert_eq!(field_features.len(), data.len(), "字段特定方法应该为每个数据项提取特征");
    
    // 测试缓存机制
    println!("\n=== 测试缓存机制 ===");
    let start_time = Instant::now();
    let cached_method = extractor.auto_select_best_method(&data).unwrap();
    let cached_duration = start_time.elapsed();
    println!("使用缓存的方法选择耗时: {:?}", cached_duration);
    assert_eq!(cached_method, global_method, "缓存的方法应该与之前选择的方法相同");
    assert!(cached_duration < global_duration, "使用缓存应该比首次选择更快");
    
    // 测试字段特定方法缓存
    let start_time = Instant::now();
    let cached_field_methods = extractor.auto_select_best_method_combination(&data).unwrap();
    let cached_field_duration = start_time.elapsed();
    println!("使用缓存的字段特定方法选择耗时: {:?}", cached_field_duration);
    assert_eq!(cached_field_methods.len(), field_methods.len(), "缓存的字段特定方法数量应该相同");
    assert!(cached_field_duration < field_duration, "使用缓存应该比首次选择更快");
    
    // 测试自动选择最佳方法的批处理
    println!("\n=== 测试自动选择最佳方法的批处理 ===");
    let mut auto_extractor = MixedFeatureExtractor::from_config(config);
    auto_extractor.detect_field_types(&data).unwrap();
    auto_extractor.compute_numeric_stats(&data).unwrap();
    auto_extractor.collect_categorical_values(&data).unwrap();
    
    let start_time = Instant::now();
    let auto_features = auto_extractor.batch_process_json(&data, true).unwrap();
    let auto_duration = start_time.elapsed();
    println!("自动选择最佳方法的批处理耗时: {:?}", auto_duration);
    
    // 验证自动选择结果
    assert_eq!(auto_features.len(), data.len(), "自动选择应该为每个数据项提取特征");
}

#[test]
fn test_domain_specific_rules() {
    // 创建不同领域的测试数据
    let mut data = Vec::new();
    
    // 代码示例
    let mut code_obj = serde_json::Map::new();
    code_obj.insert("code".to_string(), Value::String(
        "function test() { return 'Hello World'; }".to_string()
    ));
    data.push(Value::Object(code_obj));
    
    // 标题示例
    let mut title_obj = serde_json::Map::new();
    title_obj.insert("title".to_string(), Value::String(
        "这是一个测试标题".to_string()
    ));
    data.push(Value::Object(title_obj));
    
    // 描述示例
    let mut desc_obj = serde_json::Map::new();
    desc_obj.insert("description".to_string(), Value::String(
        "这是一段较长的描述文本，包含了多个句子和段落。这段文本的目的是测试TF-IDF方法对长文本的处理效果。\
        长文本通常包含更多的词汇和更复杂的语义结构，需要更高级的特征提取方法来处理。".to_string()
    ));
    data.push(Value::Object(desc_obj));
    
    // 关键词示例
    let mut keyword_obj = serde_json::Map::new();
    keyword_obj.insert("keywords".to_string(), Value::String(
        "测试 关键词 特征提取 算法".to_string()
    ));
    data.push(Value::Object(keyword_obj));
    
    // 创建特征提取器
    let mut extractor = MixedFeatureExtractor::default();
    
    // 测试领域特定规则
    println!("\n=== 测试领域特定规则 ===");
    let field_methods = extractor.auto_select_best_method_combination(&data).unwrap();
    
    // 验证领域特定规则结果
    assert!(!field_methods.is_empty(), "领域特定规则应该返回非空结果");
    
    // 验证代码字段使用字符级特征
    if let Some(method) = field_methods.get("code") {
        assert_eq!(*method, TextFeatureMethod::CharacterLevel, "代码字段应该使用字符级特征");
    }
    
    // 验证标题字段使用词袋模型
    if let Some(method) = field_methods.get("title") {
        assert_eq!(*method, TextFeatureMethod::BagOfWords, "标题字段应该使用词袋模型");
    }
    
    // 验证描述字段使用TF-IDF
    if let Some(method) = field_methods.get("description") {
        assert_eq!(*method, TextFeatureMethod::TfIdf, "描述字段应该使用TF-IDF");
    }
    
    // 验证关键词字段使用词频统计
    if let Some(method) = field_methods.get("keywords") {
        assert_eq!(*method, TextFeatureMethod::WordFrequency, "关键词字段应该使用词频统计");
    }
} 