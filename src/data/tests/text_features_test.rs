#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::text_features::*;
    use crate::error::Result;
    use serde_json::Value;
    use std::collections::HashMap;
    use serde_json::json;

    // 生成测试数据
    fn generate_test_data() -> Vec<Value> {
        vec![
            serde_json::json!({
                "title": "智能手机新品发布",
                "description": "这款新型号智能手机配备了高性能处理器和先进的摄像系统，支持5G网络和快速充电技术。",
                "price": 4999.0,
                "category": "电子产品",
                "rating": 4.8,
                "in_stock": true
            }),
            serde_json::json!({
                "title": "无线蓝牙耳机",
                "description": "高品质无线蓝牙耳机，提供出色的音质和舒适的佩戴体验，电池续航时间长达24小时。",
                "price": 899.0,
                "category": "电子配件",
                "rating": 4.5,
                "in_stock": true
            }),
            serde_json::json!({
                "title": "智能手表",
                "description": "多功能智能手表，支持心率监测、睡眠追踪和运动记录，防水设计适合各种场景使用。",
                "price": 1299.0,
                "category": "电子配件",
                "rating": 4.2,
                "in_stock": false
            })
        ]
    }

    // 生成多模态测试数据
    fn generate_multimodal_test_data() -> Vec<Value> {
        vec![
            serde_json::json!({
                "title": "智能手机新品发布",
                "description": "这款新型号智能手机配备了高性能处理器和先进的摄像系统，支持5G网络和快速充电技术。",
                "price": 4999.0,
                "category": "电子产品",
                "image": "base64encodedimage", // 假设的Base64编码图像
                "time_series": [1.0, 2.0, 3.0, 4.0, 5.0] // 假设的时间序列数据
            }),
            serde_json::json!({
                "title": "无线蓝牙耳机",
                "description": "高品质无线蓝牙耳机，提供出色的音质和舒适的佩戴体验，电池续航时间长达24小时。",
                "price": 899.0,
                "category": "电子配件",
                "audio": "base64encodedaudio" // 假设的Base64编码音频
            })
        ]
    }

    #[test]
    fn test_multi_objective_optimization() -> Result<()> {
        let data = generate_test_data();
        let extractor = TextFeatureExtractor::default();
        
        // 测试默认权重
        let weights = MultiObjectiveWeights::default();
        assert_eq!(weights.accuracy, 0.6);
        assert_eq!(weights.efficiency, 0.2);
        assert_eq!(weights.interpretability, 0.2);
        
        // 测试自定义权重
        let custom_weights = MultiObjectiveWeights::new(0.5, 0.3, 0.2);
        assert_eq!(custom_weights.accuracy, 0.5);
        assert_eq!(custom_weights.efficiency, 0.3);
        assert_eq!(custom_weights.interpretability, 0.2);
        
        // 测试多目标优化方法选择
        let best_method = extractor.multi_objective_method_selection(&data, &weights)?;
        println!("多目标优化选择的最佳方法: {:?}", best_method);
        
        // 验证方法是否有效
        match best_method {
            TextFeatureMethod::BagOfWords | 
            TextFeatureMethod::TfIdf | 
            TextFeatureMethod::WordFrequency | 
            TextFeatureMethod::CharacterLevel => assert!(true),
            _ => assert!(false, "选择了不支持的方法"),
        }
        
        Ok(())
    }

    #[test]
    fn test_advanced_feature_representation() -> Result<()> {
        let mut extractor = TextFeatureExtractor::default();
        let text = "这是一个测试文本，用于测试高级特征表示方法。";
        
        // 测试词向量嵌入
        let word_embedding = AdvancedFeatureRepresentation::WordEmbedding {
            dimension: 100,
            model_path: None,
            fine_tune: false,
        };
        let features = extractor.extract_advanced_features(text, &word_embedding)?;
        assert_eq!(features.len(), 100);
        
        // 测试文档向量
        let doc_embedding = AdvancedFeatureRepresentation::DocEmbedding {
            aggregation: EmbeddingAggregation::Mean,
            dimension: 150,
        };
        let features = extractor.extract_advanced_features(text, &doc_embedding)?;
        assert_eq!(features.len(), 150);
        
        // 测试主题模型
        let topic_model = AdvancedFeatureRepresentation::TopicModel {
            num_topics: 20,
            model_type: TopicModelType::LDA,
        };
        let features = extractor.extract_advanced_features(text, &topic_model)?;
        assert_eq!(features.len(), 20);
        
        // 测试自动选择最佳高级特征表示方法
        let data = generate_test_data();
        let best_representation = extractor.auto_select_advanced_representation(&data)?;
        println!("自动选择的最佳高级特征表示方法: {:?}", best_representation);
        
        Ok(())
    }

    #[test]
    fn test_feature_fusion_strategy() -> Result<()> {
        let extractor = MixedFeatureExtractor::default();
        
        // 创建测试特征
        let mut features = HashMap::new();
        features.insert("text".to_string(), vec![0.1, 0.2, 0.3]);
        features.insert("numeric".to_string(), vec![0.4, 0.5]);
        features.insert("categorical".to_string(), vec![0.6, 0.7, 0.8, 0.9]);
        
        // 测试简单拼接
        let concatenation = FeatureFusionStrategy::Concatenation;
        let result = extractor.fuse_features(&features, &concatenation)?;
        assert_eq!(result.len(), 9); // 3 + 2 + 4
        
        // 测试加权融合
        let mut weights = HashMap::new();
        weights.insert("text".to_string(), 0.5);
        weights.insert("numeric".to_string(), 0.3);
        weights.insert("categorical".to_string(), 0.2);
        let weighted_fusion = FeatureFusionStrategy::WeightedFusion { weights };
        let result = extractor.fuse_features(&features, &weighted_fusion)?;
        assert_eq!(result.len(), 4); // max(3, 2, 4)
        
        // 测试注意力融合
        let attention_fusion = FeatureFusionStrategy::AttentionFusion {
            num_heads: 2,
            attention_dim: 32,
        };
        let result = extractor.fuse_features(&features, &attention_fusion)?;
        assert_eq!(result.len(), 4); // max(3, 2, 4)
        
        // 测试自动选择融合策略
        let data = generate_test_data();
        let best_strategy = extractor.auto_select_fusion_strategy(&data)?;
        println!("自动选择的最佳融合策略: {:?}", best_strategy);
        
        Ok(())
    }

    #[test]
    fn test_context_aware_features() -> Result<()> {
        let mut extractor = TextFeatureExtractor::default();
        let texts = vec![
            "这是第一个测试文本。".to_string(),
            "这是第二个测试文本，它与第一个有一定的关联。".to_string(),
            "这是第三个测试文本，它与前两个都有关联。".to_string(),
        ];
        
        // 测试默认上下文配置
        let default_config = ContextAwareConfig::default();
        assert_eq!(default_config.window_size, 5);
        assert!(default_config.bidirectional);
        
        // 测试提取上下文感知特征
        let features = extractor.extract_context_aware_features(&texts, &default_config)?;
        assert_eq!(features.len(), texts.len());
        assert_eq!(features[0].len(), default_config.hidden_dim);
        
        // 测试自动选择上下文配置
        let best_config = extractor.auto_select_context_config(&texts)?;
        println!("自动选择的最佳上下文配置: {:?}", best_config);
        
        // 测试带上下文的文档特征提取
        let doc_features = extractor.extract_document_features_with_context(&texts)?;
        assert_eq!(doc_features.len(), texts.len());
        
        Ok(())
    }

    #[test]
    fn test_multimodal_feature_extraction() -> Result<()> {
        let data = generate_multimodal_test_data();
        let mut extractor = MultiModalFeatureExtractor::default();
        
        // 测试提取多模态特征
        let features = extractor.extract_features(&data[0])?;
        assert!(!features.is_empty());
        
        // 测试批量处理
        let batch_features = extractor.batch_process(&data)?;
        assert_eq!(batch_features.len(), data.len());
        
        // 测试自动选择多模态配置
        let best_config = extractor.auto_select_multimodal_config(&data)?;
        println!("自动选择的最佳多模态配置: {:?}", best_config);
        
        Ok(())
    }

    #[test]
    fn test_incremental_learning() -> Result<()> {
        let mut extractor = TextFeatureExtractor::default();
        let initial_texts = vec![
            "这是初始文本一。".to_string(),
            "这是初始文本二。".to_string(),
        ];
        
        // 初始处理
        let initial_features = extractor.batch_process(&initial_texts)?;
        assert_eq!(initial_features.len(), initial_texts.len());
        
        // 增量更新
        let new_texts = vec![
            "这是新文本一。".to_string(),
            "这是新文本二。".to_string(),
        ];
        extractor.incremental_update(&new_texts)?;
        
        // 验证更新后的特征
        let updated_features = extractor.batch_process(&initial_texts)?;
        assert_eq!(updated_features.len(), initial_texts.len());
        
        // 验证模型状态保存和加载
        let state = extractor.save_state()?;
        let mut new_extractor = TextFeatureExtractor::default();
        new_extractor.load_state(&state)?;
        
        // 验证加载后的特征
        let loaded_features = new_extractor.batch_process(&initial_texts)?;
        assert_eq!(loaded_features.len(), initial_texts.len());
        
        Ok(())
    }

    #[test]
    fn test_adaptive_weight_adjustment() -> Result<()> {
        let mut extractor = MixedFeatureExtractor::default();
        let data = generate_test_data();
        
        // 分析数据特性
        let characteristics = extractor.text_extractor.analyze_data_characteristics(&data)?;
        assert!(characteristics.avg_text_length > 0.0);
        assert!(characteristics.avg_word_count > 0.0);
        
        // 调整权重
        let weights = extractor.text_extractor.adjust_weights(&vec![
            "text".to_string(),
            "numeric".to_string(),
            "categorical".to_string()
        ])?;
        
        // 验证权重调整
        assert_eq!(weights.text_similarity + weights.numeric_similarity + weights.categorical_similarity, 1.0);
        
        // 使用自适应权重评估方法
        let methods = vec![
            TextFeatureMethod::BagOfWords,
            TextFeatureMethod::TfIdf,
            TextFeatureMethod::WordFrequency,
            TextFeatureMethod::CharacterLevel,
        ];
        
        let mut scores = Vec::new();
        for method in &methods {
            let score = extractor.text_extractor.evaluate_method_performance_with_weights(
                &data, method.clone(), &weights
            )?;
            scores.push((method.clone(), score));
        }
        
        // 验证评分
        assert!(!scores.is_empty());
        
        // 自动选择最佳方法
        let best_method = extractor.text_extractor.auto_select_best_method_adaptive(&data)?;
        println!("自适应选择的最佳方法: {:?}", best_method);
        
        Ok(())
    }

    #[test]
    fn test_text_feature_extraction_tfidf() {
        // 创建TF-IDF特征提取配置
        let config = TextFeatureConfig {
            method: TextFeatureMethod::TfIdf,
            max_features: 100,
            stop_words: None,
            min_df: 1,
            max_df: 1.0,
            enable_cache: false,
            parallel: false,
            use_stemming: false,
            remove_punctuation: true,
            lowercase: true,
            remove_numbers: false,
            remove_html: true,
            remove_urls: true,
            remove_emails: true,
            remove_emojis: true,
            use_ngrams: false,
            ngram_range: (1, 1),
            use_char_ngrams: false,
            char_ngram_range: (1, 3),
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
            case_sensitive: false,
            use_idf: true,
            smooth_idf: true,
            sublinear_tf: false,
            binary: false,
        };
        
        // 创建特征提取器
        let mut extractor = TextFeatureExtractor::new(config);
        
        // 测试文本
        let texts = vec![
            "这是第一个测试文档，用于测试TF-IDF特征提取。",
            "这是第二个测试文档，内容与第一个有所不同。",
            "第三个文档包含一些独特的词汇和表达方式。"
        ];
        
        // 批量处理文本
        let features = extractor.batch_process(&texts).unwrap();
        
        // 验证结果
        assert_eq!(features.len(), texts.len());
        for feature_vec in &features {
            assert_eq!(feature_vec.len(), 100);
        }
        
        // 验证不同文档的特征向量应该不同
        assert_ne!(features[0], features[1]);
        assert_ne!(features[0], features[2]);
        assert_ne!(features[1], features[2]);
    }

    #[test]
    fn test_text_feature_extraction_bow() {
        // 创建词袋模型特征提取配置
        let config = TextFeatureConfig {
            method: TextFeatureMethod::BagOfWords,
            max_features: 100,
            case_sensitive: false,
            stop_words: None,
            enable_cache: true,
            cache_size_limit: 1000,
            parallel: true,
            min_df: 1,
            max_df: 1.0,
            use_stemming: false,
            remove_punctuation: false,
            lowercase: false,
            remove_numbers: false,
            remove_html: false,
            remove_urls: false,
            remove_emails: false,
            remove_emojis: false,
            use_ngrams: false,
            ngram_range: (1, 3),
            use_char_ngrams: false,
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
            use_idf: false,
            smooth_idf: false,
            sublinear_tf: false,
            binary: false,
        };
        
        // 创建特征提取器
        let mut extractor = TextFeatureExtractor::new(config);
        
        // 测试文本
        let text = "这是一个测试文档，用于验证词袋模型特征提取功能。";
        
        // 提取特征
        let features = extractor.extract_features(text).unwrap();
        
        // 验证结果
        assert_eq!(features.len(), 100);
        
        // 再次提取相同文本的特征，结果应该一致
        let features2 = extractor.extract_features(text).unwrap();
        assert_eq!(features, features2);
    }

    #[test]
    fn test_mixed_feature_extractor() {
        // 创建混合特征提取器配置
        let text_config = TextFeatureConfig {
            method: TextFeatureMethod::TfIdf,
            max_features: 20,
            ..Default::default()
        };
        
        let config = MixedFeatureConfig {
            text_config,
            normalize_numeric: true,
            one_hot_categorical: true,
            auto_detect_types: false,
            auto_select_methods: false,
            max_dimensions: 100,
        };
        
        // 创建混合特征提取器
        let mut extractor = MixedFeatureExtractor::new(
            config.text_config.clone(),
            vec!["age".to_string(), "income".to_string()],
            vec!["description".to_string()],
            vec!["education".to_string(), "occupation".to_string()]
        );
        
        // 创建测试数据
        let data = vec![
            json!({
                "description": "这是一个测试文档",
                "age": 30,
                "income": 80000,
                "education": "大学",
                "occupation": "工程师"
            }),
            json!({
                "description": "另一个测试文档",
                "age": 35,
                "income": 90000,
                "education": "研究生",
                "occupation": "数据科学家"
            })
        ];
        
        // 计算数值统计
        extractor.compute_numeric_stats(&data).unwrap();
        
        // 收集类别值
        extractor.collect_categorical_values(&data).unwrap();
        
        // 提取特征
        let features = extractor.batch_process_json(&data, false).unwrap();
        
        // 验证结果
        assert_eq!(features.len(), data.len());
        
        // 验证特征维度不超过最大限制
        for feature_vec in &features {
            assert!(feature_vec.len() <= config.max_dimensions);
        }
    }

    #[test]
    fn test_method_selector() {
        // 创建方法选择器
        let mut selector = MethodSelector::new();
        selector.configure_extractor(
            vec!["description".to_string()],
            vec!["age".to_string(), "income".to_string()],
            vec!["education".to_string(), "occupation".to_string()]
        );
        
        // 创建测试数据
        let data = vec![
            json!({
                "description": "这是第一个测试文档，用于测试方法选择器。",
                "age": 30,
                "income": 80000,
                "education": "大学",
                "occupation": "工程师"
            }),
            json!({
                "description": "这是第二个测试文档，内容与第一个有所不同。",
                "age": 35,
                "income": 90000,
                "education": "研究生",
                "occupation": "数据科学家"
            }),
            json!({
                "description": "第三个文档包含一些独特的词汇和表达方式。",
                "age": 40,
                "income": 100000,
                "education": "博士",
                "occupation": "研究员"
            })
        ];
        
        // 选择最佳方法
        let method = selector.select_best_method(&data).unwrap();
        
        // 验证选择的方法是有效的
        assert!(matches!(
            method,
            TextFeatureMethod::TfIdf | 
            TextFeatureMethod::BagOfWords | 
            TextFeatureMethod::WordFrequency | 
            TextFeatureMethod::CharacterLevel |
            TextFeatureMethod::BERT
        ));
        
        // 比较不同方法的性能
        let performance = selector.compare_methods(&data).unwrap();
        
        // 验证性能比较结果
        assert!(!performance.is_empty());
        for (_, score) in &performance {
            assert!(*score >= 0.0 && *score <= 1.0);
        }
    }

    #[test]
    fn test_data_characteristics_analysis() {
        // 创建混合特征提取器
        let mut extractor = MixedFeatureExtractor::new_default();
        extractor.set_text_fields(vec!["description".to_string()])
                .set_numeric_fields(vec!["age".to_string(), "income".to_string()])
                .set_categorical_fields(vec!["education".to_string(), "occupation".to_string()]);
        
        // 创建测试数据
        let data = vec![
            json!({
                "description": "这是第一个测试文档，用于测试数据特征分析。",
                "age": 30,
                "income": 80000,
                "education": "大学",
                "occupation": "工程师"
            }),
            json!({
                "description": "这是第二个测试文档，内容与第一个有所不同。",
                "age": 35,
                "income": 90000,
                "education": "研究生",
                "occupation": "数据科学家"
            }),
            json!({
                "description": "第三个文档包含一些独特的词汇和表达方式。",
                "age": 40,
                "income": 100000,
                "education": "博士",
                "occupation": "研究员"
            })
        ];
        
        // 分析数据特征
        let characteristics = extractor.analyze_data_characteristics(&data).unwrap();
        
        // 验证结果
        assert_eq!(characteristics.total_samples, data.len());
        
        // 验证文本字段统计
        assert!(characteristics.text_fields.contains_key("description"));
        let text_stats = characteristics.text_fields.get("description").unwrap();
        assert!(text_stats.avg_length > 0.0);
        
        // 验证数值字段统计
        assert!(characteristics.numeric_fields.contains_key("age"));
        assert!(characteristics.numeric_fields.contains_key("income"));
        let age_stats = characteristics.numeric_fields.get("age").unwrap();
        assert_eq!(age_stats.min, 30.0);
        assert_eq!(age_stats.max, 40.0);
        
        // 验证类别字段统计
        assert!(characteristics.categorical_fields.contains_key("education"));
        assert!(characteristics.categorical_fields.contains_key("occupation"));
        let education_stats = characteristics.categorical_fields.get("education").unwrap();
        assert_eq!(education_stats.cardinality, 3);
    }
} 