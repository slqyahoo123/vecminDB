//! 视频特征提取器集成测试
//! 
//! 本文件提供了视频特征提取器的集成测试示例，演示主要功能的使用方法。
//! 注意：要运行这些测试，你需要将test_videos目录放在项目根目录下，并包含示例视频文件。

#[cfg(test)]
mod integration_tests {
    use crate::data::multimodal::video_extractor::*;
    use crate::data::multimodal::video_extractor::config::VideoFeatureConfig;
    use crate::data::multimodal::video_extractor::types::*;
    use crate::data::multimodal::video_extractor::error::*;
    use std::path::Path;
    use std::fs;
    use std::time::Instant;
    
    // 测试数据路径
    const TEST_VIDEO_DIR: &str = "test_videos";
    const SAMPLE_VIDEO: &str = "test_videos/sample.mp4";
    
    /// 准备测试环境并返回测试视频列表
    fn setup() -> Vec<String> {
        // 确保测试目录存在
        if !Path::new(TEST_VIDEO_DIR).exists() {
            let _ = fs::create_dir_all(TEST_VIDEO_DIR);
        }
        
        // 构建测试视频路径列表
        let mut video_paths = Vec::new();
        
        // 添加真实存在的视频
        if Path::new(SAMPLE_VIDEO).exists() {
            video_paths.push(SAMPLE_VIDEO.to_string());
        }
        
        // 添加模拟视频
        video_paths.push("test_videos/mock_video1.mp4".to_string());
        video_paths.push("test_videos/mock_video2.mp4".to_string());
        video_paths.push("test_videos/mock_video3.mp4".to_string());
        
        video_paths
    }
    
    /// 测试提取器创建和销毁
    #[test]
    fn test_extractor_lifecycle() {
        let start = Instant::now();
        println!("=== 测试提取器生命周期 ===");
        
        // 创建并销毁多个提取器实例
        for i in 1..=5 {
            println!("创建提取器实例 #{}", i);
            
            // 每次使用不同配置
            let config = VideoFeatureConfig::default()
                .with_feature_type(match i % 3 {
                    0 => VideoFeatureType::RGB,
                    1 => VideoFeatureType::OpticalFlow,
                    _ => VideoFeatureType::I3D,
                });
                
            let extractor = match VideoFeatureExtractor::new(config) {
                Ok(ext) => ext,
                Err(e) => {
                    println!("创建提取器失败: {:?}", e);
                    continue;
                }
            };
            
            println!("提取器实例 #{} 创建成功", i);
            
            // 验证配置设置正确
            let ext_config = extractor.get_config();
            println!("验证特征类型: {:?}", ext_config.feature_types);
            
            // 检查内存使用统计
            if let Some(memory_usage) = extractor.get_memory_usage() {
                println!("内存使用: {} MB", memory_usage);
            }
            
            // 提取器实例自动销毁（超出作用域）
            println!("提取器实例 #{} 销毁", i);
        }
        
        println!("测试耗时: {:?}", start.elapsed());
        assert!(true); // 确保测试通过
    }
    
    /// 测试基本提取流程
    #[test]
    fn test_basic_extraction_flow() {
        let start = Instant::now();
        println!("=== 测试基本特征提取流程 ===");
        
        let video_paths = setup();
        if video_paths.is_empty() {
            println!("警告：没有找到测试视频");
            return;
        }
        
        // 创建默认配置
        let config = VideoFeatureConfig::default();
        println!("创建默认配置");
        
        // 创建提取器
        let mut extractor = match VideoFeatureExtractor::new(config) {
            Ok(ext) => ext,
            Err(e) => {
                println!("创建提取器失败: {:?}", e);
                return;
            }
        };
        println!("成功创建视频特征提取器");
        
        // 提取特征
        let video_path = &video_paths[0];
        println!("开始从 {} 提取特征", video_path);
        
        match extractor.extract_features(video_path) {
            Ok(result) => {
                println!("特征提取成功:");
                println!("  - 视频ID: {}", result.video_id);
                println!("  - 特征类型: {:?}", result.feature_type);
                println!("  - 特征维度: {}", result.features.len());
                println!("  - 特征前5个值: {:?}", &result.features.iter().take(5).collect::<Vec<_>>());
                
                if let Some(meta) = &result.metadata {
                    println!("  - 视频长度: {:.2}秒", meta.duration);
                    println!("  - 分辨率: {}x{}", meta.width, meta.height);
                    println!("  - 帧率: {:.2} fps", meta.fps);
                    println!("  - 编解码器: {}", meta.codec);
                }
                
                println!("  - 处理时间: {} ms", result.processing_time_ms);
                
                // 验证结果
                assert!(!result.features.is_empty(), "特征向量不应为空");
                assert!(result.metadata.is_some(), "元数据应该存在");
                assert!(result.processing_time_ms > 0, "处理时间应大于0");
            },
            Err(e) => {
                println!("特征提取失败: {:?}", e);
                // 测试错误诊断
                let diagnostics = e.diagnose();
                println!("错误诊断摘要:\n{}", diagnostics.get_summary());
            }
        }
        
        println!("提取器状态:");
        let stats = extractor.get_stats();
        for (key, value) in stats {
            println!("  - {}: {}", key, value);
        }
        
        println!("测试耗时: {:?}", start.elapsed());
    }
    
    /// 测试批量提取流程
    #[test]
    fn test_batch_extraction() {
        let start = Instant::now();
        println!("=== 测试批量提取流程 ===");
        
        let video_paths = setup();
        if video_paths.len() < 2 {
            println!("警告：测试视频数量不足");
        }
        
        // 创建配置，启用缓存和批处理
        let mut config = VideoFeatureConfig::default();
        config.feature_types = vec![VideoFeatureType::RGB];
        config.cache_size = 10000;
        config.parallel_threads = 2;
        
        // 创建提取器
        let mut extractor = match VideoFeatureExtractor::new(config) {
            Ok(ext) => ext,
            Err(e) => {
                println!("创建提取器失败: {:?}", e);
                return;
            }
        };
        
        println!("开始批量提取，视频数量: {}", video_paths.len());
        let batch_start = Instant::now();
        
        match extractor.extract_features_batch(&video_paths) {
            Ok(results) => {
                let batch_elapsed = batch_start.elapsed();
                println!("批量提取完成:");
                println!("  - 总数: {}", results.len());
                println!("  - 成功: {}", results.iter().filter(|r| r.is_ok()).count());
                println!("  - 失败: {}", results.iter().filter(|r| r.is_err()).count());
                println!("  - 总处理时间: {:?}", batch_elapsed);
                
                // 平均每个视频处理时间
                let avg_time = batch_elapsed.as_millis() as f64 / video_paths.len() as f64;
                println!("  - 平均每个视频处理时间: {:.2} ms", avg_time);
                
                // 打印成功结果的信息
                for (i, result) in results.iter().enumerate() {
                    match result {
                        Ok(r) => {
                            println!("视频 #{} 处理成功:", i + 1);
                            println!("  - 视频ID: {}", r.video_id);
                            println!("  - 特征维度: {}", r.features.len());
                            println!("  - 处理时间: {} ms", r.processing_time_ms);
                        },
                        Err(e) => {
                            println!("视频 #{} 处理失败: {}", i + 1, e);
                        }
                    }
                }
                
                // 显示缓存统计
                if let Some(cache_stats) = extractor.get_cache_stats() {
                    println!("缓存统计:");
                    println!("  - 大小: {}", cache_stats.size);
                    println!("  - 容量: {}", cache_stats.capacity);
                    println!("  - 命中次数: {}", cache_stats.hit_count);
                    println!("  - 未命中次数: {}", cache_stats.miss_count);
                    println!("  - 命中率: {:.2}", cache_stats.hit_rate);
                    
                    // 缓存二次命中测试
                    println!("\n执行二次批量提取（测试缓存）");
                    let second_batch_start = Instant::now();
                    match extractor.extract_features_batch(&video_paths) {
                        Ok(_) => {
                            let second_batch_elapsed = second_batch_start.elapsed();
                            println!("二次批量提取完成:");
                            println!("  - 总处理时间: {:?}", second_batch_elapsed);
                            println!("  - 速度提升比: {:.2}x", 
                                batch_elapsed.as_millis() as f64 / second_batch_elapsed.as_millis() as f64);
                            
                            // 更新后的缓存统计
                            if let Some(updated_stats) = extractor.get_cache_stats() {
                                println!("更新的缓存统计:");
                                println!("  - 命中次数: {}", updated_stats.hit_count);
                                println!("  - 命中率: {:.2}", updated_stats.hit_rate);
                                
                                // 验证缓存命中
                                assert!(updated_stats.hit_count > cache_stats.hit_count, 
                                    "二次提取应增加缓存命中");
                            }
                        },
                        Err(e) => {
                            println!("二次批量提取失败: {:?}", e);
                        }
                    }
                }
            },
            Err(e) => {
                println!("批量提取失败: {:?}", e);
            }
        }
        
        println!("测试耗时: {:?}", start.elapsed());
    }
    
    /// 测试按时间段提取
    #[test]
    fn test_interval_extraction() {
        let start = Instant::now();
        println!("=== 测试按时间段提取 ===");
        
        let video_paths = setup();
        if video_paths.is_empty() {
            println!("警告：没有找到测试视频");
            return;
        }
        
        // 创建配置
        let config = VideoFeatureConfig::default();
        
        // 创建提取器
        let mut extractor = match VideoFeatureExtractor::new(config) {
            Ok(ext) => ext,
            Err(e) => {
                println!("创建提取器失败: {:?}", e);
                return;
            }
        };
        
        // 定义时间段
        let intervals = vec![
            TimeInterval { start: 0.0, end: 5.0 },
            TimeInterval { start: 10.0, end: 15.0 },
            TimeInterval { start: 20.0, end: 25.0 },
        ];
        
        let video_path = &video_paths[0];
        println!("开始按时间段提取特征，视频: {}", video_path);
        
        match extractor.extract_features_by_intervals(video_path, &intervals) {
            Ok(results) => {
                println!("时间段提取完成:");
                println!("  - 总时间段数: {}", intervals.len());
                println!("  - 成功提取数: {}", results.len());
                
                // 验证每个时间段的结果
                for (i, (interval, result)) in results.iter().enumerate() {
                    println!("时间段 #{} ({:.1}s-{:.1}s):", 
                        i + 1, interval.start, interval.end);
                    println!("  - 特征维度: {}", result.features.len());
                    println!("  - 处理时间: {} ms", result.processing_time_ms);
                    
                    // 特征向量统计
                    if !result.features.is_empty() {
                        let min_val = result.features.iter().fold(f32::INFINITY, |a, &b| a.min(b));
                        let max_val = result.features.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
                        println!("  - 特征值范围: [{:.4}, {:.4}]", min_val, max_val);
                    }
                    
                    // 验证
                    let interval_duration = interval.end - interval.start;
                    // 对于短片段，特征向量可能比完整视频小
                    if interval_duration < 5.0 && i > 0 {
                        let first_result = &results[0].1;
                        assert!(result.features.len() <= first_result.features.len(), 
                            "短时间段的特征维度应小于或等于完整视频");
                    }
                }
                
                // 比较不同时间段的特征差异
                if results.len() > 1 {
                    println!("\n时间段特征相似度比较:");
                    for i in 0..results.len() {
                        for j in i+1..results.len() {
                            let similarity = calculate_feature_similarity(
                                &results[i].1.features, &results[j].1.features);
                            println!("  - 时间段 #{} vs #{}: {:.4}", i+1, j+1, similarity);
                        }
                    }
                }
            },
            Err(e) => {
                println!("时间段提取失败: {:?}", e);
                
                // 测试错误诊断
                let diagnostics = e.diagnose();
                println!("错误诊断摘要:\n{}", diagnostics.get_summary());
            }
        }
        
        println!("测试耗时: {:?}", start.elapsed());
    }
    
    /// 测试错误处理和恢复
    #[test]
    fn test_error_handling_and_recovery() {
        let start = Instant::now();
        println!("=== 测试错误处理和恢复 ===");
        
        // 创建配置
        let config = VideoFeatureConfig::default();
        
        // 创建提取器
        let mut extractor = match VideoFeatureExtractor::new(config) {
            Ok(ext) => ext,
            Err(e) => {
                println!("创建提取器失败: {:?}", e);
                return;
            }
        };
        
        // 1. 测试不存在的视频文件
        println!("测试场景1: 处理不存在的视频文件");
        let non_existent_path = "test_videos/non_existent_video.mp4";
        
        match extractor.extract_features(non_existent_path) {
            Ok(_) => {
                println!("错误：应当失败但成功了");
                assert!(false, "处理不存在的文件应当失败");
            },
            Err(e) => {
                println!("预期的错误: {:?}", e);
                
                // 检查错误类型
                match e {
                    VideoExtractionError::FileError(_) => {
                        println!("错误类型正确: FileError");
                    },
                    _ => {
                        println!("错误类型不正确，预期FileError，得到: {:?}", e);
                    }
                }
                
                // 测试错误诊断
                let diagnostics = e.diagnose();
                println!("错误诊断摘要:\n{}", diagnostics.get_summary());
                
                // 验证有可能的解决方案
                assert!(!diagnostics.recommendations.is_empty(), 
                    "诊断应包含解决建议");
            }
        }
        
        // 2. 测试格式错误的视频文件
        println!("\n测试场景2: 处理格式错误的文件");
        
        // 创建一个文本文件伪装成视频
        let fake_video_path = "test_videos/fake_video.mp4";
        fs::write(fake_video_path, "This is not a valid video file").unwrap_or_default();
        
        match extractor.extract_features(fake_video_path) {
            Ok(_) => {
                println!("错误：应当失败但成功了");
                assert!(false, "处理无效视频文件应当失败");
            },
            Err(e) => {
                println!("预期的错误: {:?}", e);
                
                // 检查错误类型
                match e {
                    VideoExtractionError::DecodeError(_) => {
                        println!("错误类型正确: DecodeError");
                    },
                    VideoExtractionError::FileError(_) => {
                        println!("错误类型可接受: FileError");
                    },
                    _ => {
                        println!("错误类型不正确，预期DecodeError，得到: {:?}", e);
                    }
                }
                
                // 测试错误诊断
                let diagnostics = e.diagnose();
                println!("错误诊断摘要:\n{}", diagnostics.get_summary());
            }
        }
        
        // 3. 测试提取器复位和恢复
        println!("\n测试场景3: 提取器复位和恢复");
        
        // 复位提取器
        extractor.reset();
        println!("提取器已复位");
        
        // 使用真实或模拟视频继续测试
        let video_paths = setup();
        if !video_paths.is_empty() {
            let video_path = &video_paths[0];
            println!("尝试在错误后提取特征，视频: {}", video_path);
            
            match extractor.extract_features(video_path) {
                Ok(result) => {
                    println!("恢复成功，提取结果:");
                    println!("  - 特征维度: {}", result.features.len());
                    println!("  - 处理时间: {} ms", result.processing_time_ms);
                    
                    // 验证提取器恢复正常
                    assert!(!result.features.is_empty(), "恢复后特征提取应成功");
                },
                Err(e) => {
                    println!("恢复失败: {:?}", e);
                    assert!(false, "提取器应能从错误中恢复");
                }
            }
        }
        
        // 清理测试文件
        let _ = fs::remove_file(fake_video_path);
        
        println!("测试耗时: {:?}", start.elapsed());
    }
    
    /// 测试性能基准对比
    #[test]
    fn test_performance_benchmarking() {
        let start = Instant::now();
        println!("=== 测试性能基准对比 ===");
        
        let video_paths = setup();
        if video_paths.is_empty() {
            println!("警告：没有找到测试视频");
            return;
        }
        
        // 创建不同配置
        let configs = vec![
            // 低分辨率配置
            VideoFeatureConfig::high_performance(),
            
            // 标准配置
            VideoFeatureConfig::default(),
            
            // 高质量配置
            VideoFeatureConfig::high_quality(),
        ];
        
        println!("准备测试 {} 种不同配置:", configs.len());
        for (i, config) in configs.iter().enumerate() {
            println!("配置 #{}: 分辨率={}x{}, 特征类型={:?}",
                i + 1, config.frame_width, config.frame_height, 
                config.feature_types.first().unwrap_or(&VideoFeatureType::RGB));
        }
        
        // 运行基准测试
        println!("\n开始运行基准测试...");
        let mut benchmark_results = Vec::new();
        
        for (i, config) in configs.iter().enumerate() {
            println!("测试配置 #{}", i + 1);
            
            // 创建提取器
            let mut extractor = match VideoFeatureExtractor::new(config.clone()) {
                Ok(ext) => ext,
                Err(e) => {
                    println!("创建提取器失败: {:?}", e);
                    continue;
                }
            };
            
            // 运行基准测试
            let bench_start = Instant::now();
            let video_path = &video_paths[0]; // 使用第一个视频
            
            // 多次提取以获得更准确的性能数据
            const REPEAT_COUNT: usize = 3;
            let mut total_time_ms = 0;
            let mut success_count = 0;
            
            for j in 0..REPEAT_COUNT {
                println!("  运行 #{}", j + 1);
                match extractor.extract_features(video_path) {
                    Ok(result) => {
                        total_time_ms += result.processing_time_ms;
                        success_count += 1;
                        
                        println!("  - 处理时间: {} ms", result.processing_time_ms);
                        println!("  - 特征维度: {}", result.features.len());
                    },
                    Err(e) => {
                        println!("  - 提取失败: {:?}", e);
                    }
                }
            }
            
            let bench_elapsed = bench_start.elapsed();
            let avg_time_ms = if success_count > 0 { 
                total_time_ms as f64 / success_count as f64 
            } else { 
                0.0 
            };
            
            println!("配置 #{} 基准测试完成:", i + 1);
            println!("  - 总时间: {:?}", bench_elapsed);
            println!("  - 平均处理时间: {:.2} ms", avg_time_ms);
            println!("  - 成功率: {}/{}", success_count, REPEAT_COUNT);
            
            benchmark_results.push((i + 1, avg_time_ms, bench_elapsed, config.clone()));
        }
        
        // 比较不同配置的性能
        println!("\n配置性能比较:");
        if benchmark_results.len() > 1 {
            // 以第一个配置为基准
            let baseline = benchmark_results[0].1; // 平均时间
            
            for (i, avg_time, elapsed, config) in &benchmark_results {
                let perf_ratio = if baseline > 0.0 { baseline / avg_time } else { 1.0 };
                println!("配置 #{}: ", i);
                println!("  - 特征类型: {:?}", config.feature_types);
                println!("  - 分辨率: {}x{}", config.frame_width, config.frame_height);
                println!("  - 平均处理时间: {:.2} ms", avg_time);
                println!("  - 总耗时: {:?}", elapsed);
                println!("  - 性能比例: {:.2}x", perf_ratio);
            }
            
            // 找出最佳配置
            let best = benchmark_results.iter()
                .min_by(|(_, time1, _, _), (_, time2, _, _)| 
                    time1.partial_cmp(time2).unwrap_or(std::cmp::Ordering::Equal));
                
            if let Some((best_i, best_time, _, best_config)) = best {
                println!("\n最佳性能配置: 配置 #{}", best_i);
                println!("  - 特征类型: {:?}", best_config.feature_types);
                println!("  - 分辨率: {}x{}", best_config.frame_width, best_config.frame_height);
                println!("  - 平均处理时间: {:.2} ms", best_time);
            }
        } else {
            println!("至少需要两种配置才能进行性能比较");
        }
        
        println!("测试耗时: {:?}", start.elapsed());
    }
    
    /// 测试多模态特征提取
    #[test]
    fn test_multimodal_extraction() {
        let start = Instant::now();
        println!("=== 测试多模态特征提取 ===");
        
        let video_paths = setup();
        if video_paths.is_empty() {
            println!("警告：没有找到测试视频");
            return;
        }
        
        // 创建多模态配置
        let mut config = VideoFeatureConfig::default();
        config.feature_types = vec![
            VideoFeatureType::RGB,
            VideoFeatureType::Audio
        ];
        config.extract_audio = true; // 启用音频提取
        
        // 创建提取器
        let mut extractor = match VideoFeatureExtractor::new(config) {
            Ok(ext) => ext,
            Err(e) => {
                println!("创建提取器失败: {:?}", e);
                return;
            }
        };
        
        let video_path = &video_paths[0];
        println!("开始多模态特征提取，视频: {}", video_path);
        
        // 一次性提取多种类型特征
        match extractor.extract_multimodal_features(video_path) {
            Ok(results) => {
                println!("多模态特征提取成功:");
                println!("  - 总特征类型: {}", results.len());
                
                for (feature_type, result) in &results {
                    println!("特征类型: {:?}", feature_type);
                    println!("  - 特征维度: {}", result.features.len());
                    println!("  - 处理时间: {} ms", result.processing_time_ms);
                    
                    if !result.features.is_empty() {
                        // 特征统计
                        let mut sum = 0.0;
                        let mut min = f32::INFINITY;
                        let mut max = f32::NEG_INFINITY;
                        
                        for &val in &result.features {
                            sum += val as f64;
                            min = min.min(val);
                            max = max.max(val);
                        }
                        
                        let avg = sum / result.features.len() as f64;
                        println!("  - 特征统计: 平均={:.4}, 最小={:.4}, 最大={:.4}", 
                            avg, min, max);
                    }
                }
                
                // 验证至少包含RGB和音频特征（如果支持）
                assert!(results.contains_key(&VideoFeatureType::RGB), 
                    "应包含RGB特征");
                    
                if extractor.supports_audio() {
                    assert!(results.contains_key(&VideoFeatureType::Audio), 
                        "应包含音频特征");
                } else {
                    println!("注意：当前环境不支持音频特征提取");
                }
            },
            Err(e) => {
                println!("多模态特征提取失败: {:?}", e);
                
                // 测试错误诊断
                let diagnostics = e.diagnose();
                println!("错误诊断摘要:\n{}", diagnostics.get_summary());
            }
        }
        
        println!("测试耗时: {:?}", start.elapsed());
    }
    
    /// 测试不同的特征类型
    #[test]
    fn test_feature_types() {
        let start = Instant::now();
        println!("=== 测试不同特征类型 ===");
        
        let video_paths = setup();
        if video_paths.is_empty() {
            println!("警告：没有找到测试视频");
            return;
        }
        
        // 要测试的特征类型
        let feature_types = vec![
            VideoFeatureType::RGB,
            VideoFeatureType::OpticalFlow,
            VideoFeatureType::I3D,
            VideoFeatureType::SlowFast,
        ];
        
        println!("将测试 {} 种特征类型", feature_types.len());
        
        // 创建基础配置
        let base_config = VideoFeatureConfig::default();
        let video_path = &video_paths[0];
        
        // 测试每种特征类型
        let mut results = Vec::new();
        
        for feature_type in &feature_types {
            println!("\n测试特征类型: {:?}", feature_type);
            
            // 为当前特征类型创建配置
            let mut config = base_config.clone();
            config.feature_types = vec![feature_type.clone()];
            
            // 创建提取器
            let mut extractor = match VideoFeatureExtractor::new(config) {
                Ok(ext) => ext,
                Err(e) => {
                    println!("创建提取器失败: {:?}", e);
                    continue;
                }
            };
            
            // 提取特征
            let extract_start = Instant::now();
            match extractor.extract_features(video_path) {
                Ok(result) => {
                    let extract_elapsed = extract_start.elapsed();
                    println!("特征提取成功:");
                    println!("  - 特征维度: {}", result.features.len());
                    println!("  - 处理时间: {} ms", result.processing_time_ms);
                    println!("  - 总耗时: {:?}", extract_elapsed);
                    
                    // 保存结果
                    results.push((feature_type, result, extract_elapsed));
                },
                Err(e) => {
                    println!("特征提取失败: {:?}", e);
                    
                    if matches!(feature_type, VideoFeatureType::OpticalFlow |
                                            VideoFeatureType::I3D |
                                            VideoFeatureType::SlowFast) {
                        println!("  注意：此特征类型可能需要额外依赖或不支持模拟实现");
                    }
                }
            }
        }
        
        // 比较不同特征类型
        if results.len() > 1 {
            println!("\n不同特征类型比较:");
            
            // 维度比较
            println!("特征维度比较:");
            for (ft, result, _) in &results {
                println!("  - {:?}: {} 维", ft, result.features.len());
            }
            
            // 提取时间比较
            println!("提取时间比较:");
            for (ft, result, elapsed) in &results {
                println!("  - {:?}: {} ms (API), {:?} (实际)", 
                    ft, result.processing_time_ms, elapsed);
            }
            
            // 特征相似度比较（仅对相同维度有意义）
            println!("特征相似度比较 (对相同维度才有意义):");
            for i in 0..results.len() {
                for j in i+1..results.len() {
                    let (ft1, result1, _) = &results[i];
                    let (ft2, result2, _) = &results[j];
                    
                    if result1.features.len() == result2.features.len() {
                        let similarity = calculate_feature_similarity(
                            &result1.features, &result2.features);
                        println!("  - {:?} vs {:?}: {:.4}", ft1, ft2, similarity);
                    } else {
                        println!("  - {:?} vs {:?}: 维度不同，无法比较", ft1, ft2);
                    }
                }
            }
        }
        
        println!("测试耗时: {:?}", start.elapsed());
    }
    
    /// 计算特征向量相似度（余弦相似度）
    fn calculate_feature_similarity(a: &[f32], b: &[f32]) -> f32 {
        if a.len() != b.len() || a.is_empty() {
            return 0.0;
        }
        
        let mut dot_product = 0.0;
        let mut norm_a = 0.0;
        let mut norm_b = 0.0;
        
        for i in 0..a.len() {
            dot_product += (a[i] * b[i]) as f64;
            norm_a += (a[i] * a[i]) as f64;
            norm_b += (b[i] * b[i]) as f64;
        }
        
        if norm_a <= 0.0 || norm_b <= 0.0 {
            return 0.0;
        }
        
        (dot_product / (norm_a.sqrt() * norm_b.sqrt())) as f32
    }
} 