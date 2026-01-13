//! 光流特征提取器实现
//! 
//! 本模块提供视频光流特征提取的具体实现

use super::VideoFeatureExtractor;
use super::ConfigOption;
use super::ConfigOptionType;
use crate::data::multimodal::video_extractor::types::*;
use crate::data::multimodal::video_extractor::config::VideoFeatureConfig;
use crate::data::multimodal::video_extractor::error::VideoExtractionError;
use crate::data::multimodal::video_extractor::processing;
use crate::data::multimodal::video_extractor::processing::FeatureExtractor;
use std::path::Path;
use std::collections::HashMap;
use log::{info, error, debug};

/// 光流特征提取器，实现VideoFeatureExtractor特性
pub struct OpticalFlowExtractor {
    config: VideoFeatureConfig,
    name: String,
    description: String,
    is_initialized: bool,
    processor: processing::OpticalFlowExtractor,
}

impl OpticalFlowExtractor {
    /// 创建新的光流特征提取器
    pub fn new(config: &VideoFeatureConfig) -> Self {
        Self {
            config: config.clone(),
            name: "光流特征提取器".to_string(),
            description: "提取视频的光流特征，捕捉动态信息".to_string(),
            is_initialized: false,
            processor: processing::OpticalFlowExtractor::new(config),
        }
    }
    
    /// 从现有处理器创建
    pub fn from_processor(processor: processing::OpticalFlowExtractor, config: &VideoFeatureConfig) -> Self {
        Self {
            config: config.clone(),
            name: "光流特征提取器".to_string(),
            description: "提取视频的光流特征，捕捉动态信息".to_string(),
            is_initialized: true,
            processor,
        }
    }
    
    /// 获取配置
    pub fn get_config(&self) -> &VideoFeatureConfig {
        &self.config
    }
}

impl VideoFeatureExtractor for OpticalFlowExtractor {
    fn name(&self) -> &str {
        &self.name
    }
    
    fn description(&self) -> &str {
        &self.description
    }
    
    fn supported_features(&self) -> Vec<VideoFeatureType> {
        vec![VideoFeatureType::OpticalFlow]
    }
    
    fn extract_features(&self, video_path: &Path, config: &VideoFeatureConfig) -> Result<VideoFeature, VideoExtractionError> {
        if !self.is_initialized {
            return Err(VideoExtractionError::ModelError(
                "光流提取器未初始化".to_string()
            ));
        }
        
        // 使用processing中的处理函数进行实际特征提取
        let path_str = video_path.to_str().ok_or_else(|| 
            VideoExtractionError::InputError("无法转换视频路径为字符串".to_string())
        )?;
        
        // 创建一个光流特定的配置
        let mut optical_flow_config = config.clone();
        optical_flow_config.feature_types = vec![VideoFeatureType::OpticalFlow];
        
        let result = processing::extract_features_from_file(path_str, &optical_flow_config)?;
        
        // 将VideoFeatureResult转换为VideoFeature
        Ok(VideoFeature {
            feature_type: VideoFeatureType::OpticalFlow,
            features: result.features.clone(),
            metadata: result.metadata.clone(),
            dimensions: result.dimensions,
            timestamp: result.timestamp,
        })
    }
    
    fn is_available(&self) -> bool {
        self.is_initialized
    }
    
    fn initialize(&mut self) -> Result<(), VideoExtractionError> {
        info!("初始化光流特征提取器");
        if !self.is_initialized {
            match self.processor.initialize() {
                Ok(_) => {
                    self.is_initialized = true;
                    Ok(())
                },
                Err(e) => {
                    error!("光流特征提取器初始化失败: {}", e);
                    Err(e)
                }
            }
        } else {
            debug!("光流特征提取器已经初始化");
            Ok(())
        }
    }
    
    fn release(&mut self) -> Result<(), VideoExtractionError> {
        info!("释放光流特征提取器资源");
        let result = self.processor.release();
        self.is_initialized = false;
        result
    }
    
    fn get_config_options(&self) -> HashMap<String, ConfigOption> {
        let mut options = HashMap::new();
        
        options.insert("flow_algorithm".to_string(), ConfigOption {
            name: "flow_algorithm".to_string(),
            description: "光流计算算法".to_string(),
            option_type: ConfigOptionType::Enum,
            default_value: Some("farneback".to_string()),
            allowed_values: Some(vec![
                "farneback".to_string(), 
                "lucas_kanade".to_string(), 
                "dense".to_string(), 
                "deep_flow".to_string()
            ]),
        });
        
        options.insert("feature_dim".to_string(), ConfigOption {
            name: "feature_dim".to_string(),
            description: "光流特征维度".to_string(),
            option_type: ConfigOptionType::Integer,
            default_value: Some("256".to_string()),
            allowed_values: None,
        });
        
        options.insert("stacked_frames".to_string(), ConfigOption {
            name: "stacked_frames".to_string(),
            description: "堆叠帧数".to_string(),
            option_type: ConfigOptionType::Integer,
            default_value: Some("5".to_string()),
            allowed_values: None,
        });
        
        options.insert("flow_histogram_bins".to_string(), ConfigOption {
            name: "flow_histogram_bins".to_string(),
            description: "光流方向直方图分箱数".to_string(),
            option_type: ConfigOptionType::Integer,
            default_value: Some("8".to_string()),
            allowed_values: None,
        });
        
        options
    }
    
    fn batch_extract(&self, video_paths: &[std::path::PathBuf], config: &VideoFeatureConfig) 
        -> Result<HashMap<std::path::PathBuf, Result<VideoFeature, VideoExtractionError>>, VideoExtractionError> {
        if !self.is_initialized {
            return Err(VideoExtractionError::ModelError(
                "光流提取器未初始化".to_string()
            ));
        }
        
        let paths_str: Vec<String> = video_paths.iter()
            .filter_map(|p| p.to_str().map(|s| s.to_string()))
            .collect();
        
        // 创建一个光流特定的配置
        let mut optical_flow_config = config.clone();
        optical_flow_config.feature_types = vec![VideoFeatureType::OpticalFlow];
        
        // 使用批处理函数
        let batch_results = processing::extract_features_batch(&paths_str, &optical_flow_config)?;
        
        // 将结果转换回原始格式
        let mut results = HashMap::new();
        for (i, result) in batch_results.into_iter().enumerate() {
            if i < video_paths.len() {
                let path = video_paths[i].clone();
                let feature_result = result.map(|r| VideoFeature {
                    feature_type: VideoFeatureType::OpticalFlow,
                    features: r.features,
                    metadata: r.metadata,
                    dimensions: r.dimensions,
                    timestamp: r.timestamp,
                });
                results.insert(path, feature_result);
            }
        }
        
        Ok(results)
    }
} 