//! 视频特征提取器示例
//!
//! 本文件提供了视频特征提取器的使用示例，包括基本使用、批量处理、错误处理等

use std::path::Path;
use std::time::Instant;

use super::VideoFeatureExtractor;
use super::config::VideoFeatureConfig;
use super::types::VideoFeatureType;
use super::error::VideoExtractionError;

/// 基本使用示例
pub fn basic_usage_example() {
    println!("=== 基本使用示例 ===");
    
    // 示例视频路径 - 请确保此路径存在一个有效的视频文件
    let video_path = "examples/videos/sample.mp4";
    
    if !Path::new(video_path).exists() {
        println!("示例视频不存在: {}，将使用模拟模式", video_path);
    }
    
    // 创建默认配置
    let config = VideoFeatureConfig::default();
    println!("使用默认配置: 分辨率={}x{}, 特征类型={:?}",
        config.frame_width, config.frame_height, config.feature_types);
    
    // 创建提取器
    match VideoFeatureExtractor::new(config) {
        Ok(mut extractor) => {
            println!("成功创建提取器");
            
            // 提取特征
            println!("开始提取特征...");
            let start = Instant::now();
            
            match extractor.extract_features(video_path) {
                Ok(result) => {
                    let elapsed = start.elapsed();
                    println!("提取完成，耗时: {:?}", elapsed);
                    println!("特征结果:");
                    println!("- 视频ID: {}", result.video_id);
                    println!("- 特征类型: {:?}", result.feature_type);
                    println!("- 特征维度: {}", result.features.len());
                    
                    if !result.features.is_empty() {
                        println!("- 特征前5个值: {:?}", &result.features.iter().take(5).collect::<Vec<_>>());
                    }
                    
                    if let Some(meta) = result.metadata {
                        println!("- 视频长度: {:.2}秒", meta.duration);
                        println!("- 分辨率: {}x{}", meta.width, meta.height);
                        println!("- 帧率: {:.2} fps", meta.fps);
                    }
                    
                    println!("- 处理时间: {} ms", result.processing_time_ms);
                },
                Err(e) => {
                    println!("提取失败: {:?}", e);
                    
                    // 使用错误诊断
                    let diagnostics = e.diagnose();
                    println!("错误诊断:");
                    println!("{}", diagnostics.get_summary());
                }
            }
        },
        Err(e) => {
            println!("创建提取器失败: {:?}", e);
        }
    }
    
    println!("示例结束\n");
}

/// 配置选项示例
pub fn configuration_example() {
    println!("=== 配置选项示例 ===");
    
    // 示例视频路径
    let video_path = "examples/videos/sample.mp4";
    
    // 创建不同的配置
    let config1 = VideoFeatureConfig::high_performance();
    let config2 = VideoFeatureConfig::high_quality();
    let config3 = VideoFeatureConfig::balanced();
    
    println!("预设配置比较:");
    println!("1. 高性能配置: 分辨率={}x{}, 特征类型={:?}", 
        config1.frame_width, config1.frame_height, config1.feature_types);
    println!("2. 高质量配置: 分辨率={}x{}, 特征类型={:?}", 
        config2.frame_width, config2.frame_height, config2.feature_types);
    println!("3. 平衡配置: 分辨率={}x{}, 特征类型={:?}", 
        config3.frame_width, config3.frame_height, config3.feature_types);
    
    // 创建自定义配置
    let custom_config = VideoFeatureConfig::default()
        .with_resolution(320, 240)
        .with_fps(15)
        .with_feature_type(VideoFeatureType::RGB)
        .with_threads(4)
        .with_cache_enabled(true, 1000);
    
    println!("\n自定义配置: 分辨率={}x{}, 特征类型={:?}, 线程数={}",
        custom_config.frame_width, custom_config.frame_height, 
        custom_config.feature_types, custom_config.parallel_threads);
    
    // 使用自定义配置创建提取器
    match VideoFeatureExtractor::new(custom_config) {
        Ok(mut extractor) => {
            println!("使用自定义配置创建提取器成功");
            
            // 获取当前配置
            let current_config = extractor.get_config();
            println!("当前配置:");
            println!("- 分辨率: {}x{}", current_config.frame_width, current_config.frame_height);
            println!("- 特征类型: {:?}", current_config.feature_types);
            println!("- 线程数: {}", current_config.parallel_threads);
            println!("- 缓存大小: {}", current_config.cache_size);
            
            // 修改配置
            println!("\n修改配置...");
            
            let new_config = VideoFeatureConfig::default()
                .with_resolution(640, 480)
                .with_feature_type(VideoFeatureType::OpticalFlow);
                
            if let Err(e) = extractor.update_config(new_config) {
                println!("更新配置失败: {:?}", e);
            } else {
                println!("配置已更新");
                
                // 获取更新后的配置
                let updated_config = extractor.get_config();
                println!("更新后配置:");
                println!("- 分辨率: {}x{}", updated_config.frame_width, updated_config.frame_height);
                println!("- 特征类型: {:?}", updated_config.feature_types);
            }
            
            // 使用当前配置提取特征
            if Path::new(video_path).exists() {
                println!("\n使用当前配置提取特征...");
                match extractor.extract_features(video_path) {
                    Ok(result) => {
                        println!("提取成功:");
                        println!("- 特征维度: {}", result.features.len());
                        println!("- 处理时间: {} ms", result.processing_time_ms);
                    },
                    Err(e) => {
                        println!("提取失败: {:?}", e);
                    }
                }
            }
        },
        Err(e) => {
            println!("创建提取器失败: {:?}", e);
        }
    }
    
    println!("示例结束\n");
}

/// 批量处理示例
pub fn batch_processing_example() {
    println!("=== 批量处理示例 ===");
    
    // 准备视频路径列表
    let video_paths = vec![
        "examples/videos/sample1.mp4".to_string(),
        "examples/videos/sample2.mp4".to_string(),
        "examples/videos/sample3.mp4".to_string(),
    ];
    
    println!("准备处理 {} 个视频", video_paths.len());
    
    // 创建配置，启用缓存和多线程
    let config = VideoFeatureConfig::default()
        .with_threads(4)
        .with_cache_enabled(true, 10000);
    
    // 创建提取器
    match VideoFeatureExtractor::new(config) {
        Ok(mut extractor) => {
            println!("成功创建提取器");
            
            // 批量提取特征
            println!("开始批量提取...");
            let start = Instant::now();
            
            match extractor.extract_features_batch(&video_paths) {
                Ok(results) => {
                    let elapsed = start.elapsed();
                    println!("批量提取完成，耗时: {:?}", elapsed);
                    println!("结果统计:");
                    println!("- 总数: {}", results.len());
                    println!("- 成功: {}", results.iter().filter(|r| r.is_ok()).count());
                    println!("- 失败: {}", results.iter().filter(|r| r.is_err()).count());
                    
                    // 打印各个结果
                    for (i, result) in results.iter().enumerate() {
                        match result {
                            Ok(r) => {
                                println!("\n视频 #{}: {}", i+1, video_paths[i]);
                                println!("- 视频ID: {}", r.video_id);
                                println!("- 特征维度: {}", r.features.len());
                                println!("- 处理时间: {} ms", r.processing_time_ms);
                            },
                            Err(e) => {
                                println!("\n视频 #{}: {} 处理失败", i+1, video_paths[i]);
                                println!("- 错误: {:?}", e);
                            }
                        }
                    }
                    
                    // 显示缓存统计
                    if let Some(cache_stats) = extractor.get_cache_stats() {
                        println!("\n缓存统计:");
                        println!("- 大小: {}", cache_stats.size);
                        println!("- 容量: {}", cache_stats.capacity);
                        println!("- 命中率: {:.2}", cache_stats.hit_rate);
                    }
                },
                Err(e) => {
                    println!("批量提取失败: {:?}", e);
                }
            }
            
            // 进度回调示例
            println!("\n使用进度回调的批量提取:");
            let mut last_progress = 0;
            let progress_callback = |progress: f32| {
                let current = (progress * 100.0) as i32;
                if current > last_progress {
                    println!("处理进度: {}%", current);
                    last_progress = current;
                }
                true // 继续处理
            };
            
            match extractor.extract_features_batch_with_progress(&video_paths, progress_callback) {
                Ok(results) => {
                    println!("带进度回调的批量提取完成，成功: {}/{}", 
                        results.iter().filter(|r| r.is_ok()).count(), results.len());
                },
                Err(e) => {
                    println!("带进度回调的批量提取失败: {:?}", e);
                }
            }
        },
        Err(e) => {
            println!("创建提取器失败: {:?}", e);
        }
    }
    
    println!("示例结束\n");
}

/// 错误处理示例
pub fn error_handling_example() {
    println!("=== 错误处理示例 ===");
    
    // 创建配置
    let config = VideoFeatureConfig::default();
    
    // 创建提取器
    match VideoFeatureExtractor::new(config) {
        Ok(mut extractor) => {
            println!("成功创建提取器");
            
            // 1. 处理不存在的视频
            let non_existent_path = "non_existent_video.mp4";
            println!("\n场景1: 处理不存在的视频文件");
            
            match extractor.extract_features(non_existent_path) {
                Ok(_) => {
                    println!("提取成功（意外结果）");
                },
                Err(e) => {
                    println!("提取失败（预期结果）: {:?}", e);
                    
                    // 错误类型检查
                    if let VideoExtractionError::FileError(_) = e {
                        println!("正确识别为文件错误");
                    } else {
                        println!("错误类型识别不正确");
                    }
                    
                    // 使用错误诊断
                    let diagnostics = e.diagnose();
                    println!("错误诊断:");
                    println!("{}", diagnostics.get_summary());
                    
                    // 检查解决方案
                    if !diagnostics.recommendations.is_empty() {
                        println!("推荐解决方案:");
                        for (i, rec) in diagnostics.recommendations.iter().enumerate() {
                            println!("{}. {}", i+1, rec);
                        }
                    }
                }
            }
            
            // 2. 处理损坏的视频文件
            let corrupt_video_path = "examples/corrupt_video.mp4";
            println!("\n场景2: 处理损坏的视频文件");
            
            // 创建一个非视频文件
            let _ = std::fs::write(corrupt_video_path, b"这不是有效的视频文件");
            
            match extractor.extract_features(corrupt_video_path) {
                Ok(_) => {
                    println!("提取成功（意外结果）");
                },
                Err(e) => {
                    println!("提取失败（预期结果）: {:?}", e);
                    
                    // 错误类型检查
                    if let VideoExtractionError::DecodeError(_) = e {
                        println!("正确识别为解码错误");
                    } else {
                        println!("错误类型识别不正确");
                    }
                    
                    // 使用错误诊断
                    let diagnostics = e.diagnose();
                    println!("错误诊断:");
                    println!("{}", diagnostics.get_summary());
                }
            }
            
            // 清理
            let _ = std::fs::remove_file(corrupt_video_path);
            
            // 3. 异常恢复
            println!("\n场景3: 错误后恢复");
            
            // 复位提取器
            extractor.reset();
            println!("提取器已复位");
            
            // 使用有效视频继续
            let valid_video = "examples/videos/sample.mp4";
            if Path::new(valid_video).exists() {
                println!("使用有效视频继续提取");
                
                match extractor.extract_features(valid_video) {
                    Ok(_) => {
                        println!("提取成功，提取器已成功恢复");
                    },
                    Err(e) => {
                        println!("提取失败，提取器恢复失败: {:?}", e);
                    }
                }
            } else {
                println!("示例视频不存在，跳过恢复测试");
            }
        },
        Err(e) => {
            println!("创建提取器失败: {:?}", e);
        }
    }
    
    println!("示例结束\n");
}

/// 特征类型示例
pub fn feature_types_example() {
    println!("=== 特征类型示例 ===");
    
    // 示例视频路径
    let video_path = "examples/videos/sample.mp4";
    
    if !Path::new(video_path).exists() {
        println!("示例视频不存在，将使用模拟模式");
    }
    
    // 测试不同的特征类型
    let feature_types = vec![
        VideoFeatureType::RGB,
        VideoFeatureType::OpticalFlow,
        VideoFeatureType::I3D,
        VideoFeatureType::SlowFast,
        VideoFeatureType::Audio,
    ];
    
    println!("将测试 {} 种特征类型", feature_types.len());
    
    // 基础配置
    let base_config = VideoFeatureConfig::default();
    
    // 测试每种特征类型
    for feature_type in &feature_types {
        println!("\n特征类型: {:?}", feature_type);
        
        // 创建特定配置
        let mut config = base_config.clone();
        config.feature_types = vec![feature_type.clone()];
        
        // 创建提取器
        match VideoFeatureExtractor::new(config) {
            Ok(mut extractor) => {
                println!("成功创建提取器");
                
                // 提取特征
                let start = Instant::now();
                match extractor.extract_features(video_path) {
                    Ok(result) => {
                        let elapsed = start.elapsed();
                        println!("提取成功:");
                        println!("- 特征维度: {}", result.features.len());
                        println!("- 处理时间: {} ms", result.processing_time_ms);
                        println!("- 总耗时: {:?}", elapsed);
                        
                        if !result.features.is_empty() {
                            // 计算简单统计
                            let mut sum = 0.0;
                            let mut min = f32::INFINITY;
                            let mut max = f32::NEG_INFINITY;
                            
                            for &val in &result.features {
                                sum += val as f64;
                                min = min.min(val);
                                max = max.max(val);
                            }
                            
                            let avg = sum / result.features.len() as f64;
                            println!("- 特征统计: 平均={:.4}, 最小={:.4}, 最大={:.4}", 
                                avg, min, max);
                        }
                    },
                    Err(e) => {
                        println!("提取失败: {:?}", e);
                        
                        // 对于高级特征类型，可能需要额外依赖
                        if matches!(feature_type, VideoFeatureType::OpticalFlow | 
                                               VideoFeatureType::I3D | 
                                               VideoFeatureType::SlowFast) {
                            println!("注意: 此特征类型可能需要额外依赖或不在模拟模式下支持");
                        }
                    }
                }
            },
            Err(e) => {
                println!("创建提取器失败: {:?}", e);
            }
        }
    }
    
    println!("示例结束\n");
}

/// 性能基准测试示例
pub fn benchmark_example() {
    println!("=== 性能基准测试示例 ===");
    
    // 示例视频路径
    let video_paths = vec![
        "examples/videos/sample1.mp4".to_string(),
        "examples/videos/sample2.mp4".to_string(),
    ];
    
    // 创建默认配置
    let config = VideoFeatureConfig::default();
    
    // 创建提取器
    match VideoFeatureExtractor::new(config) {
        Ok(mut extractor) => {
            println!("成功创建提取器");
            
            // 运行基准测试
            println!("开始基准测试...");
            match extractor.run_benchmark(&video_paths, 3) {
                Ok(benchmark) => {
                    println!("基准测试完成");
                    println!("结果:");
                    println!("- 特征类型: {:?}", benchmark.feature_type);
                    println!("- 处理速度: {:.2} MB/s", benchmark.processing_speed_mbps);
                    println!("- 平均处理时间: {} ms/视频", benchmark.avg_processing_time_ms);
                    println!("- 峰值内存: {} MB", benchmark.peak_memory_mb);
                },
                Err(e) => {
                    println!("基准测试失败: {:?}", e);
                }
            }
        },
        Err(e) => {
            println!("创建提取器失败: {:?}", e);
        }
    }
    
    println!("示例结束\n");
}

/// 特征导出示例
pub fn export_features_example() {
    println!("=== 特征导出示例 ===");
    
    // 示例视频路径
    let video_path = "examples/videos/sample.mp4";
    
    // 创建默认配置
    let config = VideoFeatureConfig::default();
    
    // 创建提取器
    match VideoFeatureExtractor::new(config) {
        Ok(mut extractor) => {
            println!("成功创建提取器");
            
            // 提取特征
            match extractor.extract_features(video_path) {
                Ok(result) => {
                    println!("特征提取成功");
                    
                    // 输出目录
                    let output_dir = std::env::temp_dir().join("video_features_export");
                    std::fs::create_dir_all(&output_dir)
                        .expect("创建输出目录失败");
                    
                    println!("导出目录: {}", output_dir.display());
                    
                    // 获取支持的导出格式
                    let supported_formats = extractor.get_available_export_formats();
                    println!("支持的导出格式:");
                    for format in &supported_formats {
                        println!("- {:?}", format);
                    }
                    
                    // 使用CSV格式导出
                    let csv_path = output_dir.join("feature.csv");
                    match extractor.export_features(&result, &csv_path, export::ExportFormat::CSV) {
                        Ok(path) => {
                            println!("CSV导出成功: {}", path.display());
                        },
                        Err(e) => {
                            println!("CSV导出失败: {:?}", e);
                        }
                    }
                    
                    // 使用JSON格式导出
                    let json_path = output_dir.join("feature.json");
                    match extractor.export_features(&result, &json_path, export::ExportFormat::JSON) {
                        Ok(path) => {
                            println!("JSON导出成功: {}", path.display());
                        },
                        Err(e) => {
                            println!("JSON导出失败: {:?}", e);
                        }
                    }
                    
                    // 创建自定义导出选项
                    let options = extractor.create_export_options(
                        export::ExportFormat::Binary,
                        true,   // 包含元数据
                        true,   // 包含处理信息
                        true    // 压缩
                    );
                    
                    // 使用自定义选项导出为二进制格式
                    let bin_path = output_dir.join("feature.bin");
                    match extractor.export_features_with_options(&result, &bin_path, &options) {
                        Ok(path) => {
                            println!("二进制导出成功: {}", path.display());
                        },
                        Err(e) => {
                            println!("二进制导出失败: {:?}", e);
                        }
                    }
                    
                    println!("清理导出文件...");
                    let _ = std::fs::remove_file(csv_path);
                    let _ = std::fs::remove_file(json_path);
                    let _ = std::fs::remove_file(bin_path.with_extension("bin.gz")); // 压缩后的文件
                    let _ = std::fs::remove_dir(&output_dir);
                },
                Err(e) => {
                    println!("特征提取失败: {:?}", e);
                }
            }
        },
        Err(e) => {
            println!("创建提取器失败: {:?}", e);
        }
    }
    
    println!("示例结束\n");
}

/// 批量导出示例
pub fn batch_export_example() {
    println!("=== 批量导出示例 ===");
    
    // 示例视频路径
    let video_paths = vec![
        "examples/videos/sample1.mp4".to_string(),
        "examples/videos/sample2.mp4".to_string(),
        "examples/videos/sample3.mp4".to_string(),
    ];
    
    // 创建默认配置
    let config = VideoFeatureConfig::default();
    
    // 创建提取器
    match VideoFeatureExtractor::new(config) {
        Ok(mut extractor) => {
            println!("成功创建提取器");
            
            // 输出目录
            let output_dir = std::env::temp_dir().join("batch_export_test");
            std::fs::create_dir_all(&output_dir)
                .expect("创建输出目录失败");
                
            println!("导出目录: {}", output_dir.display());
            
            // 直接提取并导出
            println!("开始批量提取并导出...");
            match extractor.extract_and_export_batch(&video_paths, &output_dir, export::ExportFormat::JSON) {
                Ok(paths) => {
                    println!("批量导出成功，生成了 {} 个文件:", paths.len());
                    for (i, path) in paths.iter().enumerate() {
                        println!("- 文件 #{}: {}", i+1, path.display());
                    }
                    
                    // 清理导出的文件
                    println!("清理导出文件...");
                    for path in &paths {
                        let _ = std::fs::remove_file(path);
                    }
                },
                Err(e) => {
                    println!("批量导出失败: {:?}", e);
                }
            }
            
            let _ = std::fs::remove_dir(&output_dir);
        },
        Err(e) => {
            println!("创建提取器失败: {:?}", e);
        }
    }
    
    println!("示例结束\n");
}

/// 运行所有示例
pub fn run_all_examples() {
    println!("======== 视频特征提取器示例 ========\n");
    
    basic_usage_example();
    configuration_example();
    batch_processing_example();
    error_handling_example();
    feature_types_example();
    benchmark_example();
    export_features_example();
    batch_export_example();
    
    println!("======== 示例结束 ========");
} 