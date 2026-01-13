//! 特征提取器接口
//! 
//! 定义视频特征提取器的基本接口

use crate::data::multimodal::video_extractor::types::*;
use crate::data::multimodal::video_extractor::error::VideoExtractionError;

/// 特征提取器接口
pub trait FeatureExtractor: Send + Sync {
    /// 初始化特征提取器
    fn initialize(&mut self) -> Result<(), VideoExtractionError>;
    
    /// 处理单个视频帧
    fn process_frame(&mut self, frame: &VideoFrame) -> Result<Vec<f32>, VideoExtractionError>;
    
    /// 处理一批视频帧
    fn process_batch(&mut self, frames: &[VideoFrame]) -> Result<Vec<Vec<f32>>, VideoExtractionError> {
        let mut results = Vec::with_capacity(frames.len());
        for frame in frames {
            let features = self.process_frame(frame)?;
            results.push(features);
        }
        Ok(results)
    }
    
    /// 获取特征维度
    fn get_feature_dim(&self) -> usize;
    
    /// 获取特征类型
    fn get_feature_type(&self) -> VideoFeatureType;
    
    /// 执行时间池化
    fn temporal_pooling(&self, features: &[Vec<f32>], pooling_type: &TemporalPoolingType) 
        -> Result<Vec<f32>, VideoExtractionError> {
        match pooling_type {
            TemporalPoolingType::Mean => Ok(super::pooling::temporal_mean_pooling(features)),
            TemporalPoolingType::Max => Ok(super::pooling::temporal_max_pooling(features)),
            TemporalPoolingType::Attention => Err(VideoExtractionError::NotImplementedError(
                "注意力池化需要在特定提取器中实现".to_string()
            )),
            TemporalPoolingType::Custom(_) => Err(VideoExtractionError::NotImplementedError(
                "自定义池化需要在特定提取器中实现".to_string()
            )),
        }
    }
    
    /// 释放资源
    fn release(&mut self) -> Result<(), VideoExtractionError> {
        Ok(())
    }
} 