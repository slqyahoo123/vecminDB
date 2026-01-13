use crate::data::text_features::{MixedFeatureExtractor, MixedFeatureConfig, TextFeatureConfig, TextFeatureMethod, DataCharacteristics};
use crate::error::Result;
use serde_json::Value;
use rand;
use serde_json::json;
use crate::data::EvaluationWeights;
use std::collections::HashMap;
use std::collections::HashSet;

/// 混合数据类型特征提取示例
pub fn mixed_feature_extraction_example() -> Result<()> {
    // 创建示例数据
    let data = r#"[
        {
            "id": 1,
            "title": "智能手机新品发布",
            "description": "这款新型号智能手机配备了高性能处理器和先进的摄像系统，支持5G网络和快速充电技术。",
            "price": 4999.0,
            "category": "电子产品",
            "rating": 4.8,
            "in_stock": true
        },
        {
            "id": 2,
            "title": "无线蓝牙耳机",
            "description": "高品质无线蓝牙耳机，提供出色的音质和舒适的佩戴体验，电池续航时间长达24小时。",
            "price": 899.0,
            "category": "电子配件",
            "rating": 4.5,
            "in_stock": true
        },
        {
            "id": 3,
            "title": "智能手表",
            "description": "多功能智能手表，支持心率监测、睡眠追踪和运动记录，防水设计适合各种场景使用。",
            "price": 1299.0,
            "category": "电子配件",
            "rating": 4.2,
            "in_stock": false
        }
    ]"#;
    
    // 解析JSON数据
    let json_data: Vec<Value> = serde_json::from_str(data)?;
    
    // 创建自动特征提取器
    let config = MixedFeatureConfig {
        text_config: TextFeatureConfig {
            name: None,
            method: TextFeatureMethod::AutoSelect,
            max_features: 100,
            min_df: 1,
            max_df: 1.0,
            enable_cache: true,
            parallel: true,
            stemming: Some(false),
            remove_punctuation: Some(true),
            remove_digits: Some(false),
            remove_html: Some(true),
            remove_urls: true,
            remove_emails: true,
            remove_emojis: true,
            use_ngrams: true,
            ngram_range: (1, 3),
            use_char_ngrams: true,
            char_ngram_range: (2, 5),
            use_pos_tagging: false,
            use_ner: false,
            use_embeddings: false,
            embedding_model_path: None,
            use_custom_tokenizer: false,
            custom_tokenizer_path: None,
            use_custom_extractor: false,
            custom_extractor_path: None,
            use_custom_preprocessor: false,
            custom_preprocessor_path: None,
            use_custom_postprocessor: false,
            custom_postprocessor_path: None,
            cache_size_limit: 1000,
            case_sensitive: Some(false),
            use_idf: true,
            smooth_idf: true,
            sublinear_tf: false,
            binary: false,
            stop_words: None,
            remove_stopwords: Some(false),
            max_length: None,
            vector_size: Some(768),
            extra_params: HashMap::new(),
            use_preprocessing: false,
        },
        normalize_numeric: true,
        one_hot_categorical: true,
        auto_detect_types: true,
        auto_select_methods: true,
        max_dimensions: 200,
    };
    
    let mut extractor = MixedFeatureExtractor::from_config_result(config)
        .expect("Failed to create extractor");
    
    // 初始化提取器
    extractor.detect_field_types(&json_data)?;
    
    // 获取数值字段统计信息
    extractor.compute_numeric_stats(&json_data)?;
    
    // 获取类别字段值
    extractor.collect_categorical_values(&json_data)?;
    
    // 打印字段类型
    println!("字段类型:");
    for (field, field_type) in &extractor.field_types {
        println!("  {}: {:?}", field, field_type);
    }
    
    // 自动选择最佳方法
    let best_method = extractor.auto_select_best_method_adaptive(&json_data)?;
    println!("自动选择的最佳方法: {:?}", best_method);
    
    // 提取特征
    let features = extractor.batch_process_json(&json_data, true)?;
    
    // 输出特征向量
    for (i, feature_vec) in features.iter().enumerate() {
        println!("商品 #{}: 特征向量维度 = {}", i + 1, feature_vec.len());
        
        // 打印前10个特征值
        print!("前10个特征值: [");
        for (j, &value) in feature_vec.iter().take(10).enumerate() {
            if j > 0 {
                print!(", ");
            }
            print!("{:.4}", value);
        }
        println!("...]");
    }
    
    // 打印特征向量的维度
    println!("\n特征向量维度:");
    for (i, feature_vec) in features.iter().enumerate().take(5) {
        println!("样本 {}: {} 维", i, feature_vec.len());
    }
    
    println!("\n特征提取完成!");
    
    Ok(())
}

/// 特征重要性评估示例
pub fn feature_importance_example() -> crate::error::Result<()> {
    println!("开始特征重要性评估示例...");
    
    // 创建示例数据
    let mut data = Vec::new();
    let mut labels = Vec::new();
    
    // 生成50个样本数据
    for i in 0..50 {
        // 创建一个JSON对象
        let mut obj = serde_json::Map::new();
        
        // 添加数值字段 - 与标签相关性高
        let age = 20.0 + (i as f64 / 2.0);
        obj.insert("age".to_string(), json!(age));
        
        // 添加数值字段 - 与标签相关性低
        let random_value = rand::random::<f64>() * 100.0;
        obj.insert("random_value".to_string(), json!(random_value));
        
        // 添加文本字段 - 与标签相关性中等
        let sentiment = if i % 3 == 0 {
            "非常满意，产品质量很好"
        } else if i % 3 == 1 {
            "一般，有待改进"
        } else {
            "不满意，质量太差"
        };
        obj.insert("comment".to_string(), json!(sentiment));
        
        // 添加类别字段 - 与标签相关性高
        let category = if i % 5 < 2 { "A" } else if i % 5 < 4 { "B" } else { "C" };
        obj.insert("category".to_string(), json!(category));
        
        // 生成标签 - 主要由age和category决定
        let label = if age > 35.0 && category == "A" {
            1.0
        } else if age > 30.0 && (category == "A" || category == "B") {
            0.8
        } else if age > 25.0 {
            0.5
        } else {
            0.2
        };
        
        data.push(serde_json::Value::Object(obj));
        labels.push(label);
    }
    
    // 创建特征提取器
    let mut extractor = MixedFeatureExtractor::default();
    
    // 初始化提取器
    extractor.detect_field_types(&data)?;
    extractor.compute_numeric_stats(&data)?;
    extractor.collect_categorical_values(&data)?;
    
    // 提取特征
    println!("提取特征...");
    let features = extractor.batch_process_json(&data, true)?;
    
    // 打印特征向量的维度
    println!("\n特征向量维度:");
    for (i, feature_vec) in features.iter().enumerate().take(5) {
        println!("样本 {}: {} 维", i, feature_vec.len());
    }
    
    println!("\n特征提取完成!");
    
    Ok(())
}

/// 自适应权重调整示例
pub fn adaptive_weight_adjustment_example() -> Result<()> {
    println!("自适应权重调整示例");
    println!("===================");

    // 创建测试数据
    let mut data = Vec::new();
    
    // 长文本数据
    for i in 0..5 {
        let long_text = format!("这是一个非常长的文本，包含了大量的词汇和信息。这是为了测试自适应权重调整功能而创建的。这个文本需要足够长，以便能够测试出不同特征提取方法的性能差异。我们希望看到基于数据特性的权重调整能够选择最合适的特征提取方法。这是第{}个长文本样本。", i);
        data.push(json!({
            "id": i,
            "title": format!("长文本样本 {}", i),
            "content": long_text,
            "category": "test",
            "score": i as f64 * 1.5
        }));
    }
    
    // 短文本数据
    for i in 5..10 {
        let short_text = format!("这是第{}个短文本样本。", i);
        data.push(json!({
            "id": i,
            "title": format!("短文本样本 {}", i),
            "content": short_text,
            "category": "test",
            "score": i as f64 * 0.5
        }));
    }
    
    // 数值型数据
    for i in 10..15 {
        data.push(json!({
            "id": i,
            "value1": i as f64 * 1.1,
            "value2": i as f64 * 0.9,
            "value3": i as f64 * 1.5,
            "category": "numeric"
        }));
    }
    
    // 类别型数据
    for i in 15..20 {
        data.push(json!({
            "id": i,
            "category1": format!("cat_{}", i % 3),
            "category2": format!("type_{}", i % 4),
            "category3": format!("group_{}", i % 2),
            "type": "categorical"
        }));
    }
    
    // 创建特征提取器
    let config = TextFeatureConfig {
        method: TextFeatureMethod::TfIdf,
        max_features: 100,
        stop_words: None,
        min_df: 1,
        max_df: 0.95,
        enable_cache: true,
        parallel: true,
        bert_model_path: Some("models/bert-base-uncased".to_string()),
        bert_layers: Some(vec![1, 2, 3]),
        transformer_model_path: Some("models/transformer-base".to_string()),
        pooling_strategy: Some("mean".to_string()),
        stemming: Some(false),
        remove_punctuation: Some(true),
        lowercase: true,
        remove_numbers: false,
        remove_html: Some(true),
        remove_urls: true,
        remove_emails: true,
        remove_emojis: true,
        use_ngrams: true,
        ngram_range: (1, 3),
        use_char_ngrams: true,
        char_ngram_range: (2, 5),
        use_pos_tagging: false,
        use_ner: false,
        use_embeddings: false,
        embedding_model_path: None,
        use_custom_tokenizer: false,
        custom_tokenizer_path: None,
        use_custom_extractor: false,
        custom_extractor_path: None,
        use_custom_preprocessor: false,
        custom_preprocessor_path: None,
        use_custom_postprocessor: false,
        custom_postprocessor_path: None,
        cache_size_limit: 1000,
        case_sensitive: Some(false),
        use_idf: true,
        smooth_idf: true,
        sublinear_tf: false,
        binary: false,
    };
    let mut extractor = MixedFeatureExtractor::new(
        config,
        vec!["score".to_string(), "value1".to_string(), "value2".to_string(), "value3".to_string()],
        vec!["content".to_string(), "title".to_string()],
        vec!["category".to_string(), "category1".to_string(), "category2".to_string(), "category3".to_string()]
    );
    
    // 分析数据特性
    println!("\n1. 分析数据特性");
    println!("-----------------");
    let characteristics = extractor.analyze_data(&data)?;
    
    print_characteristics(&characteristics);
    
    // 调整权重
    println!("\n2. 根据数据特性调整权重");
    println!("-----------------");
    let weights = extractor.adjust_weights(&vec![
        "text".to_string(),
        "numeric".to_string(),
        "categorical".to_string()
    ])?;
    
    println!("调整后的评估权重:");
    println!("  - 文本相似度权重: {:.2}", weights.text_similarity);
    println!("  - 数值相似度权重: {:.2}", weights.numeric_similarity);
    println!("  - 类别相似度权重: {:.2}", weights.categorical_similarity);
    
    // 使用固定权重评估各种方法
    println!("\n3. 使用固定权重评估各种方法");
    println!("-----------------");
    let default_weights = EvaluationWeights::default();
    
    let methods = vec![
        TextFeatureMethod::BagOfWords,
        TextFeatureMethod::TfIdf,
        TextFeatureMethod::WordFrequency,
        TextFeatureMethod::CharacterLevel,
    ];
    
    let mut fixed_results = Vec::new();
    for method in &methods {
        let score = extractor.evaluate_method_performance_with_weights(&data, method.clone(), &default_weights)?;
        fixed_results.push((method.clone(), score));
    }
    
    // 找出得分最高的方法
    let mut best_fixed_method = TextFeatureMethod::BagOfWords;
    let mut best_fixed_score = 0.0;
    
    for (method, score) in &fixed_results {
        if *score > best_fixed_score {
            best_fixed_score = *score;
            best_fixed_method = method.clone();
        }
    }
    
    println!("使用固定权重的最佳方法: {:?}, 得分: {:.2}", best_fixed_method, best_fixed_score);
    
    // 使用自适应权重评估各种方法
    println!("\n4. 使用自适应权重评估各种方法");
    println!("-----------------");
    
    let mut adaptive_results = Vec::new();
    for method in &methods {
        let score = extractor.evaluate_method_performance_with_weights(&data, method.clone(), &weights)?;
        adaptive_results.push((method.clone(), score));
    }
    
    // 找出得分最高的方法
    let mut best_adaptive_method = TextFeatureMethod::BagOfWords;
    let mut best_adaptive_score = 0.0;
    
    for (method, score) in &adaptive_results {
        if *score > best_adaptive_score {
            best_adaptive_score = *score;
            best_adaptive_method = method.clone();
        }
    }
    
    println!("使用自适应权重的最佳方法: {:?}, 得分: {:.2}", best_adaptive_method, best_adaptive_score);
    
    // 比较结果
    println!("\n5. 比较结果");
    println!("-----------------");
    println!("固定权重选择的方法: {:?}", best_fixed_method);
    println!("自适应权重选择的方法: {:?}", best_adaptive_method);
    
    if best_fixed_method == best_adaptive_method {
        println!("两种方法选择了相同的特征提取方法。");
    } else {
        println!("两种方法选择了不同的特征提取方法！");
        println!("这表明自适应权重调整能够根据数据特性选择更合适的方法。");
    }
    
    // 使用自动选择方法
    println!("\n6. 使用自动选择方法");
    println!("-----------------");
    let best_method = extractor.auto_select_best_method_adaptive(&data)?;
    println!("自动选择的最佳方法: {:?}", best_method);
    
    Ok(())
}

fn print_characteristics(characteristics: &DataCharacteristics) {
    println!("数据特征分析结果:");
    println!("  - 数据类型: {}", characteristics.data_type);
    println!("  - 平均文本长度: {:.2}", characteristics.avg_text_length);
    println!("  - 词汇量大小: {}", characteristics.vocabulary_size);
    println!("  - 数值特征数量: {}", characteristics.numeric_feature_count);
    println!("  - 分类特征数量: {}", characteristics.categorical_feature_count);
    println!("  - 总样本数: {}", characteristics.sample_count);
    
    // 打印文本字段统计
    if !characteristics.text_fields.is_empty() {
        println!("  文本字段统计:");
        for (field, stats) in &characteristics.text_fields {
            println!("    {}: 平均长度={:.2}, 平均词数={:.2}, 特殊字符比例={:.2}",
                field, stats.avg_length, stats.avg_word_count, stats.special_char_ratio);
        }
    }
    
    // 打印数值字段统计
    if !characteristics.numeric_fields.is_empty() {
        println!("  数值字段统计:");
        for (field, stats) in &characteristics.numeric_fields {
            println!("    {}: 最小值={:.2}, 最大值={:.2}, 平均值={:.2}, 标准差={:.2}",
                field, stats.min, stats.max, stats.mean, stats.std);
        }
    }
    
    // 打印类别字段统计
    if !characteristics.categorical_fields.is_empty() {
        println!("  类别字段统计:");
        for (field, stats) in &characteristics.categorical_fields {
            println!("    {}: 基数={}", field, stats.category_count);
        }
    }
}

/// 运行示例
pub fn run_examples() -> crate::error::Result<()> {
    println!("\n运行混合数据类型特征提取示例...");
    mixed_feature_extraction_example()?;
    
    println!("\n运行特征重要性评估示例...");
    feature_importance_example()?;
    
    println!("\n运行自适应权重调整示例...");
    adaptive_weight_adjustment_example()?;
    
    println!("所有示例运行完成！");
    Ok(())
}

fn run_mixed_feature_example() -> Result<()> {
    // 创建一个简单的特征提取配置
    let config = MixedFeatureConfig {
        enable_parallel: false,
        text_config: TextFeatureConfig {
            name: "default".to_string(),
            remove_digits: true,
            remove_stopwords: true,
            max_length: 1000,
            vector_dim: 100,
            bert_model_path: Some("models/bert-base".to_string()),
            aggregation: Some("mean".to_string()),
            use_stemming: Some(true),
            language: Some("english".to_string()),
            max_vocab_size: Some(10000),
            min_word_freq: Some(2),
        },
        normalize_numeric: true,
        one_hot_categorical: true,
        auto_detect_types: true,
        auto_select_methods: true,
        max_dimensions: 200,
    };

    // ... existing code ...
    Ok(())
}

fn run_text_feature_example() -> Result<()> {
    // 创建文本特征配置
    let config = TextFeatureConfig {
        method: TextFeatureMethod::TfIdf,
        max_features: 100,
        stop_words: None,
        min_df: 1,
        max_df: 0.95,
        enable_cache: true,
        parallel: true,
        bert_model_path: Some("models/bert-base-uncased".to_string()),
        bert_layers: Some(vec![1, 2, 3]),
        transformer_model_path: Some("models/transformer-base".to_string()),
        pooling_strategy: Some("mean".to_string()),
        stemming: Some(false),
        remove_punctuation: Some(true),
        lowercase: true,
        remove_numbers: false,
        remove_html: Some(true),
        remove_urls: true,
        remove_emails: true,
        remove_emojis: true,
        use_ngrams: true,
        ngram_range: (1, 2),
        use_char_ngrams: true,
        char_ngram_range: (2, 4),
        use_pos_tagging: false,
        use_ner: false,
        use_embeddings: false,
        embedding_model_path: None,
        use_custom_tokenizer: false,
        custom_tokenizer_path: None,
        use_custom_extractor: false,
        custom_extractor_path: None,
        use_custom_preprocessor: false,
        custom_preprocessor_path: None,
        use_custom_postprocessor: false,
        custom_postprocessor_path: None,
        cache_size_limit: 1000,
        case_sensitive: Some(false),
        use_idf: true,
        smooth_idf: true,
        sublinear_tf: false,
        binary: false,
    };

    // ... existing code ...
    Ok(())
} 