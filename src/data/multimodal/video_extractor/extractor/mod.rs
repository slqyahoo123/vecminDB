//! 视频特征提取器模块
//! 
//! 本模块提供视频特征提取的核心接口和管理功能，定义了统一的特征提取器接口规范

mod rgb;
mod optical_flow;
mod composite;
mod factory;
mod benchmark;
mod manager;

// 重新导出所有公共组件
pub use rgb::RGBExtractor;
pub use optical_flow::OpticalFlowExtractor;
pub use composite::CompositeExtractor;
pub use factory::ExtractorFactory;
// 需要时由上层模块显式导入 benchmark 与 manager 内的导出，避免在本模块内产生未使用导出

use crate::data::multimodal::video_extractor::types::*;
use crate::data::multimodal::video_extractor::config::VideoFeatureConfig;
use crate::data::multimodal::video_extractor::error::VideoExtractionError;
use std::path::Path;
use std::collections::HashMap;

/// 视频特征提取器特性，所有特征提取器的通用接口
pub trait VideoFeatureExtractor: Send + Sync {
    /// 获取提取器名称
    fn name(&self) -> &str;
    
    /// 获取提取器描述
    fn description(&self) -> &str;
    
    /// 获取支持的特征类型
    fn supported_features(&self) -> Vec<VideoFeatureType>;
    
    /// 提取视频特征
    fn extract_features(&self, video_path: &Path, config: &VideoFeatureConfig) -> Result<VideoFeature, VideoExtractionError>;
    
    /// 检查提取器是否可用
    fn is_available(&self) -> bool;
    
    /// 批量提取特征（默认实现，可以被特定提取器优化）
    fn batch_extract(&self, video_paths: &[std::path::PathBuf], config: &VideoFeatureConfig) 
        -> Result<HashMap<std::path::PathBuf, Result<VideoFeature, VideoExtractionError>>, VideoExtractionError> {
        use rayon::prelude::*;
        use log::debug;

        debug!("使用默认批量提取实现处理{}个视频", video_paths.len());
        let results: HashMap<_, _> = video_paths.par_iter()
            .map(|path| {
                let result = self.extract_features(path, config);
                (path.clone(), result)
            })
            .collect();
        
        Ok(results)
    }
    
    /// 初始化提取器
    fn initialize(&mut self) -> Result<(), VideoExtractionError> {
        use log::debug;
        debug!("执行{}提取器的默认初始化", self.name());
        Ok(())
    }
    
    /// 释放资源
    fn release(&mut self) -> Result<(), VideoExtractionError> {
        use log::debug;
        debug!("执行{}提取器的默认资源释放", self.name());
        Ok(())
    }
    
    /// 获取提取器配置选项
    fn get_config_options(&self) -> HashMap<String, ConfigOption> {
        HashMap::new()
    }
}

/// 配置选项定义
#[derive(Debug, Clone)]
pub struct ConfigOption {
    pub name: String,
    pub description: String,
    pub option_type: ConfigOptionType,
    pub default_value: Option<String>,
    pub allowed_values: Option<Vec<String>>,
}

/// 配置选项类型
#[derive(Debug, Clone)]
pub enum ConfigOptionType {
    String,
    Integer,
    Float,
    Boolean,
    Enum,
} 