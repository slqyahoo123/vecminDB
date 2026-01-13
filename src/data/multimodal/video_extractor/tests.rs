//! 视频特征提取器测试模块
//! 
//! 本模块包含视频特征提取器的单元测试，验证各个功能模块的正确性

#[cfg(test)]
mod tests {
    use super::super::*;
    use super::super::types::*;
    use super::super::config::*;
    use super::super::error::*;
    // 未使用导入移除

    /// 测试默认配置创建
    #[test]
    fn test_default_config() {
        let config = VideoFeatureConfig::default();
        assert_eq!(config.frame_width, 224);
        assert_eq!(config.frame_height, 224);
        assert_eq!(config.fps, 15);
        assert!(config.feature_types.contains(&VideoFeatureType::RGB));
    }

    /// 测试配置构建器模式
    #[test]
    fn test_config_builder() {
        let mut config = VideoFeatureConfig::default();
        config.frame_width = 320;
        config.frame_height = 240;
        config.fps = 25;
        config.feature_types = vec![VideoFeatureType::OpticalFlow];
        config.set_string_param("feature_dimension", "512".to_string());
        
        assert_eq!(config.frame_width, 320);
        assert_eq!(config.frame_height, 240);
        assert_eq!(config.fps, 25);
        assert!(config.feature_types.contains(&VideoFeatureType::OpticalFlow));
        assert_eq!(config.get_usize_param("feature_dimension"), Some(512));
    }

    /// 测试视频元数据提取（模拟）
    #[test]
    fn test_video_metadata_extraction() {
        use super::super::processing::extract_video_metadata;
        
        // 使用不存在的文件测试错误处理
        let result = extract_video_metadata("non_existent_video.mp4");
        assert!(result.is_err());
        
        // 通过测试生成的元数据检查字段是否完整
        // 注意：这里假设内部实现会为测试路径生成模拟数据
        if let Ok(metadata) = extract_video_metadata("test_video.mp4") {
            assert!(metadata.width > 0);
            assert!(metadata.height > 0);
            assert!(metadata.duration_seconds > 0.0);
            assert!(metadata.frame_count > 0);
            assert!(metadata.fps > 0.0);
            assert!(!metadata.video_id.is_empty());
        }
    }

    /// 测试视频特征提取器创建
    #[test]
    fn test_extractor_creation() {
        let config = VideoFeatureConfig::default();
        let extractor = VideoFeatureExtractor::new(config);
        assert!(extractor.is_ok());
    }

    /// 测试视频ID生成
    #[test]
    fn test_video_id_generation() {
        use super::super::processing::generate_video_id;
        
        let id1 = generate_video_id("path/to/video1.mp4");
        let id2 = generate_video_id("path/to/video2.mp4");
        let id1_again = generate_video_id("path/to/video1.mp4");
        
        // 相同路径生成相同ID
        assert_eq!(id1, id1_again);
        // 不同路径生成不同ID
        assert_ne!(id1, id2);
    }

    /// 测试缓存功能
    #[test]
    fn test_feature_cache() {
        use super::super::cache::FeatureCache;
        use super::super::types::{VideoFeature, VideoFeatureType};
        
        let mut cache = FeatureCache::new(1024 * 1024); // 1MB缓存
        let video_id = "test-video-id";
        let feature_type = VideoFeatureType::RGB;
        
        // 创建测试特征数据
        let feature_data = vec![0.1, 0.2, 0.3, 0.4, 0.5];
        let feature_result = VideoFeatureResult {
            feature_type,
            features: feature_data.clone(),
            metadata: None,
            processing_info: None,
            dimensions: feature_data.len(),
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs(),
        };
        
        // 测试缓存添加
        let cache_key = format!("{}_{}", video_id, feature_type);
        cache.put(&cache_key, feature_result.clone());
        
        // 测试缓存检索
        let cached_feature = cache.get(&cache_key);
        assert!(cached_feature.is_some());
        
        // 验证缓存的数据正确
        if let Some(cached) = cached_feature {
            assert_eq!(cached.feature_type, feature_type);
            assert_eq!(cached.features, feature_data);
        }
        
        // 测试缓存删除
        cache.remove(&cache_key);
        let cached_feature = cache.get(video_id, feature_type, None);
        assert!(cached_feature.is_none());
    }

    /// 测试批处理计划
    #[test]
    fn test_batch_processing_plan() {
        use super::super::batch::BatchProcessingPlan;
        
        let video_ids = vec![
            "video1.mp4".to_string(),
            "video2.mp4".to_string(),
            "video3.mp4".to_string(),
            "video4.mp4".to_string(),
        ];
        
        let plan = BatchProcessingPlan::new(
            video_ids.clone(),
            2, // batch_size
            None,
        );
        
        // 验证基本属性
        assert_eq!(plan.get_batch_size(), 2);
        assert_eq!(plan.get_batch_count(), 2);
        
        // 验证批次数量
        assert_eq!(plan.get_batch_count(), 2); // 4个视频，每批2个
        
        // 验证总视频数
        assert_eq!(plan.get_total_videos(), 4);
        
        // 测试获取特定批次的视频ID
        let batch1 = plan.get_video_ids_for_batch(0);
        assert_eq!(batch1.len(), 2);
        assert_eq!(batch1[0], "video1.mp4");
        assert_eq!(batch1[1], "video2.mp4");
        
        let batch2 = plan.get_video_ids_for_batch(1);
        assert_eq!(batch2.len(), 2);
        assert_eq!(batch2[0], "video3.mp4");
        assert_eq!(batch2[1], "video4.mp4");
    }

    /// 测试性能基准测试
    #[test]
    fn test_performance_benchmark() {
        use super::super::benchmark::{PerformanceBenchmark, BenchmarkComparison};
        use super::super::types::VideoFeatureType;
        
        // 创建基准测试实例
        let benchmark1 = PerformanceBenchmark::new(
            VideoFeatureType::RGB,
            100, // video_count
            1024 * 1024 * 1024, // 1GB total_size_bytes
            10000.0, // 10000ms total processing_time_ms
            512.0, // 512MB memory_mb
            4, // 4 threads
            ModelType::ResNet50, // model_type
        );
        
        // 测试添加自定义指标
        let mut benchmark2 = benchmark1.clone();
        benchmark2.add_custom_metric("accuracy", 0.95);
        
        // 验证基准测试属性
        assert_eq!(benchmark1.feature_type, VideoFeatureType::RGB);
        assert_eq!(benchmark1.video_count, 100);
        assert_eq!(benchmark1.processing_speed_mbps, 50.0);
        
        // 测试性能比较
        let comparison = BenchmarkComparison::new(benchmark1.clone(), benchmark2.clone());
        assert_eq!(comparison.speed_change_percent(), 0.0); // 相同速度
        assert_eq!(comparison.memory_change_percent(), 0.0); // 相同内存
        
        // 测试性能评分
        let score1 = benchmark1.calculate_performance_score(0.7, 0.3);
        assert!(score1 > 0.0);
    }

    /// 测试错误处理
    #[test]
    fn test_error_handling() {
        // 测试从String创建错误
        let err_str = "测试错误信息".to_string();
        let video_err: VideoExtractionError = err_str.clone().into();
        match video_err {
            VideoExtractionError::GenericError(msg) => assert_eq!(msg, err_str),
            _ => panic!("错误类型不匹配"),
        }
        
        // 测试从IO错误创建错误
        use std::io::{Error, ErrorKind};
        let io_err = Error::new(ErrorKind::NotFound, "文件不存在");
        let video_err: VideoExtractionError = io_err.into();
        match video_err {
            VideoExtractionError::FileError(_) => {} // 正确转换
            _ => panic!("错误类型不匹配"),
        }
    }
    
    /// 测试导出格式和选项
    #[test]
    fn test_export_formats_and_options() {
        use super::super::export::{ExportFormat, ExportOptions, get_available_export_formats, 
                                   is_format_supported, create_export_options};
        
        // 测试可用格式获取
        let formats = get_available_export_formats();
        assert!(formats.contains(&ExportFormat::CSV));
        assert!(formats.contains(&ExportFormat::JSON));
        assert!(formats.contains(&ExportFormat::Binary));
        
        // 测试格式支持检查
        assert!(is_format_supported(ExportFormat::CSV));
        assert!(is_format_supported(ExportFormat::JSON));
        assert!(is_format_supported(ExportFormat::Binary));
        
        // 测试创建导出选项
        let options = create_export_options(
            ExportFormat::CSV,
            true,  // include_metadata
            false, // include_processing_info
            false  // compress
        );
        
        assert_eq!(options.format, ExportFormat::CSV);
        assert_eq!(options.include_metadata, true);
        assert_eq!(options.include_processing_info, false);
        assert_eq!(options.compress, false);
        assert_eq!(options.batch_size, None);
        assert!(options.custom_options.is_empty());
    }
    
    /// 测试特征导出功能
    #[test]
    fn test_feature_export() {
        use super::super::export::{export_features, ExportFormat, ExportOptions};
        use super::super::types::{VideoFeatureResult, VideoFeatureType, VideoMetadata};
        use std::collections::HashMap;
        use std::fs;
        use std::path::PathBuf;
        
        // 创建临时目录
        let temp_dir = std::env::temp_dir().join("video_feature_export_test");
        fs::create_dir_all(&temp_dir).expect("创建临时目录失败");
        
        // 清理之前的测试文件
        let _ = fs::remove_file(temp_dir.join("test_export.csv"));
        let _ = fs::remove_file(temp_dir.join("test_export.json"));
        
        // 创建测试用特征数据
        let test_result = VideoFeatureResult {
            video_id: "test-video-123".to_string(),
            feature_type: VideoFeatureType::RGB,
            dimension: 5,
            features: vec![0.1, 0.2, 0.3, 0.4, 0.5],
            metadata: VideoMetadata {
                filename: Some("test_video.mp4".to_string()),
                width: Some(1280),
                height: Some(720),
                duration: Some(10.5),
                fps: Some(30.0),
                codec: Some("h264".to_string()),
                bitrate: Some(5000000),
                audio_codec: Some("aac".to_string()),
                audio_channels: Some(2),
                audio_sample_rate: Some(44100),
                created_at: None,
                file_size: 1024 * 1024,  // 1MB
                id: "test-video-123".to_string(),
            },
            processing_time: 0.5,
            created_at: 0.0,
        };
        
        // 测试CSV导出
        let csv_options = ExportOptions {
            format: ExportFormat::CSV,
            include_metadata: true,
            include_processing_info: true,
            compress: false,
            batch_size: None,
            custom_options: HashMap::new(),
        };
        
        let csv_path = temp_dir.join("test_export.csv");
        let csv_result = export_features(vec![&test_result], &csv_path, csv_options);
        assert!(csv_result.is_ok(), "CSV导出失败: {:?}", csv_result.err());
        assert!(csv_path.exists(), "CSV文件未创建");
        
        // 测试JSON导出
        let json_options = ExportOptions {
            format: ExportFormat::JSON,
            include_metadata: true,
            include_processing_info: false,
            compress: false,
            batch_size: None,
            custom_options: HashMap::new(),
        };
        
        let json_path = temp_dir.join("test_export.json");
        let json_result = export_features(vec![&test_result], &json_path, json_options);
        assert!(json_result.is_ok(), "JSON导出失败: {:?}", json_result.err());
        assert!(json_path.exists(), "JSON文件未创建");
        
        // 清理测试文件
        let _ = fs::remove_file(&csv_path);
        let _ = fs::remove_file(&json_path);
        let _ = fs::remove_dir(&temp_dir);
    }
    
    /// 测试批量导出功能
    #[test]
    fn test_batch_export() {
        use super::super::export::{export_features_batch, ExportFormat, ExportOptions};
        use super::super::types::{VideoFeatureResult, VideoFeatureType, VideoMetadata};
        use std::collections::HashMap;
        use std::fs;
        use std::path::PathBuf;
        
        // 创建临时目录
        let temp_dir = std::env::temp_dir().join("video_feature_batch_export_test");
        fs::create_dir_all(&temp_dir).expect("创建临时目录失败");
        
        // 创建多个测试用特征数据
        let test_results = vec![
            VideoFeatureResult {
                video_id: "test-video-1".to_string(),
                feature_type: VideoFeatureType::RGB,
                dimension: 3,
                features: vec![0.1, 0.2, 0.3],
                metadata: VideoMetadata {
                    filename: Some("test_video1.mp4".to_string()),
                    width: Some(1280),
                    height: Some(720),
                    duration: Some(5.0),
                    fps: Some(30.0),
                    codec: Some("h264".to_string()),
                    bitrate: Some(5000000),
                    audio_codec: Some("aac".to_string()),
                    audio_channels: Some(2),
                    audio_sample_rate: Some(44100),
                    created_at: None,
                    file_size: 1024 * 1024,
                    id: "test-video-1".to_string(),
                },
                processing_time: 0.3,
                created_at: 0.0,
            },
            VideoFeatureResult {
                video_id: "test-video-2".to_string(),
                feature_type: VideoFeatureType::RGB,
                dimension: 3,
                features: vec![0.4, 0.5, 0.6],
                metadata: VideoMetadata {
                    filename: Some("test_video2.mp4".to_string()),
                    width: Some(1920),
                    height: Some(1080),
                    duration: Some(8.0),
                    fps: Some(30.0),
                    codec: Some("h264".to_string()),
                    bitrate: Some(8000000),
                    audio_codec: Some("aac".to_string()),
                    audio_channels: Some(2),
                    audio_sample_rate: Some(44100),
                    created_at: None,
                    file_size: 2 * 1024 * 1024,
                    id: "test-video-2".to_string(),
                },
                processing_time: 0.4,
                created_at: 0.0,
            },
        ];
        
        // 设置导出选项
        let options = ExportOptions {
            format: ExportFormat::JSON,
            include_metadata: true,
            include_processing_info: true,
            compress: false,
            batch_size: Some(1),  // 每批一个文件
            custom_options: HashMap::new(),
        };
        
        // 执行批量导出
        let export_result = export_features_batch(&test_results, &temp_dir, &options);
        assert!(export_result.is_ok(), "批量导出失败: {:?}", export_result.err());
        
        // 验证结果
        if let Ok(output_paths) = export_result {
            assert_eq!(output_paths.len(), 2, "应该生成2个文件");
            
            // 检查文件是否存在
            for path in &output_paths {
                assert!(path.exists(), "导出文件不存在: {:?}", path);
            }
            
            // 清理文件
            for path in &output_paths {
                let _ = fs::remove_file(path);
            }
        }
        
        // 清理目录
        let _ = fs::remove_dir(&temp_dir);
    }
} 