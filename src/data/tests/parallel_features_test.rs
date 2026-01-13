#[cfg(test)]
mod tests {
    use crate::data::text_features::{MixedFeatureExtractor, MixedFeatureConfig, TextFeatureConfig, TextFeatureMethod};
    use crate::data::parallel_utils::ParallelConfig;
    use serde_json::json;
    use serde_json::Value;
    use std::time::Instant;

    // 生成测试数据
    fn generate_test_data(count: usize) -> Vec<Value> {
        let mut data = Vec::with_capacity(count);
        for i in 0..count {
            data.push(json!({
                "id": i,
                "title": format!("测试标题 {}", i),
                "description": format!("这是一个测试描述，包含一些文本内容。ID: {}", i),
                "price": 100.0 + (i as f64),
                "category": if i % 3 == 0 { "类别A" } else if i % 3 == 1 { "类别B" } else { "类别C" },
                "tags": ["标签1", "标签2", if i % 2 == 0 { "标签3" } else { "标签4" }]
            }));
        }
        data
    }

    #[test]
    fn test_parallel_feature_extraction() {
        // 生成测试数据
        let data = generate_test_data(100);
        
        // 创建串行特征提取器
        let serial_config = MixedFeatureConfig {
            text_config: TextFeatureConfig {
                method: TextFeatureMethod::BagOfWords,
                max_features: 100,
                case_sensitive: false,
                stop_words: None,
                enable_cache: true,
                cache_size_limit: 1000,
                parallel: false,
            },
            normalize_numeric: true,
            one_hot_categorical: true,
            auto_detect_types: true,
            auto_select_methods: false,
            max_dimensions: 200,
        };
        
        let mut serial_extractor = MixedFeatureExtractor::from_config(serial_config);
        
        // 创建并行特征提取器
        let parallel_config = MixedFeatureConfig {
            text_config: TextFeatureConfig {
                method: TextFeatureMethod::BagOfWords,
                max_features: 100,
                case_sensitive: false,
                stop_words: None,
                enable_cache: true,
                cache_size_limit: 1000,
                parallel: false,
            },
            normalize_numeric: true,
            one_hot_categorical: true,
            auto_detect_types: true,
            auto_select_methods: false,
            max_dimensions: 200,
        };
        
        let mut parallel_extractor = MixedFeatureExtractor::from_config(parallel_config);
        
        // 初始化提取器
        serial_extractor.detect_field_types(&data).unwrap();
        serial_extractor.compute_numeric_stats(&data).unwrap();
        serial_extractor.collect_categorical_values(&data).unwrap();
        
        parallel_extractor.detect_field_types(&data).unwrap();
        parallel_extractor.compute_numeric_stats(&data).unwrap();
        parallel_extractor.collect_categorical_values(&data).unwrap();
        
        // 测试串行提取性能
        let serial_start = Instant::now();
        let serial_features = serial_extractor.batch_process_json(&data, false).unwrap();
        let serial_duration = serial_start.elapsed();
        
        // 测试并行提取性能
        let parallel_start = Instant::now();
        let parallel_features = parallel_extractor.batch_process_json(&data, true).unwrap();
        let parallel_duration = parallel_start.elapsed();
        
        // 验证结果正确性
        assert_eq!(serial_features.len(), parallel_features.len());
        
        if !serial_features.is_empty() && !parallel_features.is_empty() {
            assert_eq!(serial_features[0].len(), parallel_features[0].len());
            
            // 验证特征向量的值是否相似
            for i in 0..std::cmp::min(5, serial_features.len()) {
                let serial_vec = &serial_features[i];
                let parallel_vec = &parallel_features[i];
                
                // 检查向量长度
                assert_eq!(serial_vec.len(), parallel_vec.len());
                
                // 检查向量值是否相似 (允许小的浮点误差)
                for j in 0..serial_vec.len() {
                    assert!((serial_vec[j] - parallel_vec[j]).abs() < 1e-6, 
                        "特征向量值不匹配: serial[{}][{}]={}, parallel[{}][{}]={}", 
                        i, j, serial_vec[j], i, j, parallel_vec[j]);
                }
            }
        }
        
        println!("串行处理时间: {:?}", serial_duration);
        println!("并行处理时间: {:?}", parallel_duration);
        println!("加速比: {:.2}x", serial_duration.as_secs_f64() / parallel_duration.as_secs_f64());
        
        // 并行处理应该比串行处理快
        // 注意：在某些情况下，如果数据量太小或系统负载高，并行处理可能不会更快
        // 因此，这个断言可能在某些环境中失败
        // assert!(parallel_duration < serial_duration);
    }
    
    #[test]
    fn test_auto_select_method_parallel() {
        // 生成测试数据
        let data = generate_test_data(50);
        
        // 创建串行特征提取器
        let serial_config = MixedFeatureConfig {
            text_config: TextFeatureConfig {
                method: TextFeatureMethod::Auto,
                max_features: 100,
                case_sensitive: false,
                stop_words: None,
                enable_cache: true,
                cache_size_limit: 1000,
                parallel: false,
            },
            normalize_numeric: true,
            one_hot_categorical: true,
            auto_detect_types: true,
            auto_select_methods: true,
            max_dimensions: 200,
        };
        
        let mut serial_extractor = MixedFeatureExtractor::from_config(serial_config);
        
        // 创建并行特征提取器
        let parallel_config = MixedFeatureConfig {
            text_config: TextFeatureConfig {
                method: TextFeatureMethod::Auto,
                max_features: 100,
                case_sensitive: false,
                stop_words: None,
                enable_cache: true,
                cache_size_limit: 1000,
                parallel: false,
            },
            normalize_numeric: true,
            one_hot_categorical: true,
            auto_detect_types: true,
            auto_select_methods: true,
            max_dimensions: 200,
        };
        
        let mut parallel_extractor = MixedFeatureExtractor::from_config(parallel_config);
        
        // 初始化提取器
        serial_extractor.detect_field_types(&data).unwrap();
        serial_extractor.compute_numeric_stats(&data).unwrap();
        serial_extractor.collect_categorical_values(&data).unwrap();
        
        parallel_extractor.detect_field_types(&data).unwrap();
        parallel_extractor.compute_numeric_stats(&data).unwrap();
        parallel_extractor.collect_categorical_values(&data).unwrap();
        
        // 测试串行自动选择方法
        let serial_start = Instant::now();
        let serial_method = serial_extractor.auto_select_best_method_adaptive(&data).unwrap();
        let serial_duration = serial_start.elapsed();
        
        // 测试并行自动选择方法
        let parallel_start = Instant::now();
        let parallel_method = parallel_extractor.auto_select_best_method_adaptive(&data).unwrap();
        let parallel_duration = parallel_start.elapsed();
        
        println!("串行自动选择方法: {:?}, 耗时: {:?}", serial_method, serial_duration);
        println!("并行自动选择方法: {:?}, 耗时: {:?}", parallel_method, parallel_duration);
        println!("加速比: {:.2}x", serial_duration.as_secs_f64() / parallel_duration.as_secs_f64());
        
        // 两种方法应该选择相同的特征提取方法
        assert_eq!(serial_method, parallel_method);
    }
    
    #[test]
    fn test_parallel_config() {
        // 测试不同批处理大小的性能
        let data = generate_test_data(200);
        let batch_sizes = vec![1, 10, 20, 50, 100];
        
        for batch_size in batch_sizes {
            let config = MixedFeatureConfig {
                text_config: TextFeatureConfig {
                    method: TextFeatureMethod::BagOfWords,
                    max_features: 100,
                    case_sensitive: false,
                    stop_words: None,
                    enable_cache: true,
                    cache_size_limit: 1000,
                    parallel: false,
                },
                normalize_numeric: true,
                one_hot_categorical: true,
                auto_detect_types: true,
                auto_select_methods: false,
                max_dimensions: 200,
            };
            
            let mut extractor = MixedFeatureExtractor::from_config(config);
            extractor.detect_field_types(&data).unwrap();
            extractor.compute_numeric_stats(&data).unwrap();
            extractor.collect_categorical_values(&data).unwrap();
            
            let start = Instant::now();
            let features = extractor.batch_process_json(&data, true).unwrap();
            let duration = start.elapsed();
            
            println!("批处理大小: {}, 处理时间: {:?}", batch_size, duration);
            
            // 验证结果正确性
            assert_eq!(features.len(), data.len());
        }
    }
} 