//! 特征池化模块
//! 
//! 本模块提供视频特征的时间和空间池化功能，用于将高维特征聚合为更紧凑的表示

use crate::data::multimodal::video_extractor::error::VideoExtractionError;
use log::{debug, warn};

/// 时间平均池化，对多帧特征计算平均值
pub fn temporal_mean_pooling(features: &[Vec<f32>]) -> Vec<f32> {
    if features.is_empty() {
        return Vec::new();
    }
    
    // 假设所有特征向量长度相同
    let dim = features[0].len();
    let mut result = vec![0.0; dim];
    
    // 计算每个维度的平均值
    for feature in features {
        for (i, &val) in feature.iter().enumerate() {
            if i < dim {
                result[i] += val;
            }
        }
    }
    
    // 归一化
    let count = features.len() as f32;
    for val in &mut result {
        *val /= count;
    }
    
    debug!("执行时间平均池化: {}帧输入 -> {}维输出", features.len(), result.len());
    result
}

/// 时间最大池化，对多帧特征取最大值
pub fn temporal_max_pooling(features: &[Vec<f32>]) -> Vec<f32> {
    if features.is_empty() {
        return Vec::new();
    }
    
    // 假设所有特征向量长度相同
    let dim = features[0].len();
    let mut result = vec![f32::NEG_INFINITY; dim];
    
    // 计算每个维度的最大值
    for feature in features {
        for (i, &val) in feature.iter().enumerate() {
            if i < dim {
                result[i] = result[i].max(val);
            }
        }
    }
    
    debug!("执行时间最大池化: {}帧输入 -> {}维输出", features.len(), result.len());
    result
}

/// 空间平均池化，将特征图在空间维度上平均
pub fn spatial_average_pooling(feature_map: &[Vec<f32>], height: usize, width: usize) -> Result<Vec<f32>, VideoExtractionError> {
    if feature_map.is_empty() || height == 0 || width == 0 {
        return Err(VideoExtractionError::ProcessingError(
            "空间平均池化：无效的输入特征图".to_string()
        ));
    }
    
    if feature_map.len() != height * width {
        return Err(VideoExtractionError::ProcessingError(
            format!("空间平均池化：特征图尺寸不匹配，期望{}x{}={}, 实际{}", 
                    height, width, height * width, feature_map.len())
        ));
    }
    
    // 假设所有特征向量长度相同
    let channels = feature_map[0].len();
    let mut result = vec![0.0; channels];
    
    // 计算每个通道的平均值
    for feature_vector in feature_map {
        for (c, &val) in feature_vector.iter().enumerate() {
            if c < channels {
                result[c] += val / (height * width) as f32;
            }
        }
    }
    
    debug!("执行空间平均池化: {}x{} 特征图 -> {}维输出", height, width, result.len());
    Ok(result)
}

/// 空间最大池化，取特征图在空间维度上的最大值
pub fn spatial_max_pooling(feature_map: &[Vec<f32>], height: usize, width: usize) -> Result<Vec<f32>, VideoExtractionError> {
    if feature_map.is_empty() || height == 0 || width == 0 {
        return Err(VideoExtractionError::ProcessingError(
            "空间最大池化：无效的输入特征图".to_string()
        ));
    }
    
    if feature_map.len() != height * width {
        return Err(VideoExtractionError::ProcessingError(
            format!("空间最大池化：特征图尺寸不匹配，期望{}x{}={}, 实际{}", 
                    height, width, height * width, feature_map.len())
        ));
    }
    
    // 假设所有特征向量长度相同
    let channels = feature_map[0].len();
    let mut result = vec![f32::NEG_INFINITY; channels];
    
    // 计算每个通道的最大值
    for feature_vector in feature_map {
        for (c, &val) in feature_vector.iter().enumerate() {
            if c < channels {
                result[c] = result[c].max(val);
            }
        }
    }
    
    debug!("执行空间最大池化: {}x{} 特征图 -> {}维输出", height, width, result.len());
    Ok(result)
}

/// ROI池化，对感兴趣区域进行池化
pub fn roi_pooling(feature_map: &[Vec<f32>], height: usize, width: usize, 
                  rois: &[Rect], output_size: (usize, usize)) -> Result<Vec<Vec<f32>>, VideoExtractionError> {
    if feature_map.is_empty() || height == 0 || width == 0 || rois.is_empty() {
        return Err(VideoExtractionError::ProcessingError(
            "ROI池化：无效的输入参数".to_string()
        ));
    }
    
    if feature_map.len() != height * width {
        return Err(VideoExtractionError::ProcessingError(
            format!("ROI池化：特征图尺寸不匹配，期望{}x{}={}, 实际{}", 
                    height, width, height * width, feature_map.len())
        ));
    }
    
    let (out_height, out_width) = output_size;
    if out_height == 0 || out_width == 0 {
        return Err(VideoExtractionError::ProcessingError(
            format!("ROI池化：无效的输出尺寸：({}, {})", out_height, out_width)
        ));
    }
    
    // 假设所有特征向量长度相同
    let channels = feature_map[0].len();
    let mut roi_features = Vec::with_capacity(rois.len());
    
    // 对每个ROI进行池化
    for roi in rois {
        if !roi.is_valid(width, height) {
            warn!("ROI超出边界，跳过: {:?}", roi);
            continue;
        }
        
        let mut roi_feature = Vec::with_capacity(out_height * out_width * channels);
        
        // 计算步长
        let bin_width = roi.width as f32 / out_width as f32;
        let bin_height = roi.height as f32 / out_height as f32;
        
        // 对每个输出位置进行池化
        for oy in 0..out_height {
            for ox in 0..out_width {
                // 计算输入区域
                let x_start = roi.x + (ox as f32 * bin_width) as usize;
                let y_start = roi.y + (oy as f32 * bin_height) as usize;
                let x_end = (roi.x + ((ox + 1) as f32 * bin_width) as usize).min(roi.x + roi.width);
                let y_end = (roi.y + ((oy + 1) as f32 * bin_height) as usize).min(roi.y + roi.height);
                
                // 初始化该位置的特征向量
                let mut bin_feature = vec![0.0; channels];
                let mut count = 0;
                
                // 计算区域内的平均特征
                for y in y_start..y_end {
                    for x in x_start..x_end {
                        let idx = y * width + x;
                        if idx < feature_map.len() {
                            for (c, &val) in feature_map[idx].iter().enumerate() {
                                if c < channels {
                                    bin_feature[c] += val;
                                }
                            }
                            count += 1;
                        }
                    }
                }
                
                // 归一化
                if count > 0 {
                    for val in &mut bin_feature {
                        *val /= count as f32;
                    }
                }
                
                // 添加到结果
                roi_feature.push(bin_feature);
            }
        }
        
        roi_features.push(roi_feature.concat());
    }
    
    debug!("执行ROI池化: {} ROIs -> {}个{}维特征向量", 
           rois.len(), roi_features.len(), 
           if roi_features.is_empty() { 0 } else { roi_features[0].len() });
    
    Ok(roi_features)
}

/// 加权时间池化，根据权重聚合多帧特征
pub fn weighted_temporal_pooling(features: &[Vec<f32>], weights: &[f32]) -> Result<Vec<f32>, VideoExtractionError> {
    if features.is_empty() {
        return Err(VideoExtractionError::ProcessingError(
            "加权时间池化：特征为空".to_string()
        ));
    }
    
    if weights.len() != features.len() {
        return Err(VideoExtractionError::ProcessingError(
            format!("加权时间池化：权重数({})与特征数({})不匹配", weights.len(), features.len())
        ));
    }
    
    // 计算权重和
    let weight_sum: f32 = weights.iter().sum();
    if weight_sum <= f32::EPSILON {
        return Err(VideoExtractionError::ProcessingError(
            format!("加权时间池化：权重和太小 ({:e})", weight_sum)
        ));
    }
    
    // 假设所有特征向量长度相同
    let dim = features[0].len();
    let mut result = vec![0.0; dim];
    
    // 应用加权平均
    for (feature, &weight) in features.iter().zip(weights.iter()) {
        let norm_weight = weight / weight_sum;
        for (i, &val) in feature.iter().enumerate() {
            if i < dim {
                result[i] += val * norm_weight;
            }
        }
    }
    
    debug!("执行加权时间池化: {}帧输入 -> {}维输出", features.len(), result.len());
    Ok(result)
}

/// 空间金字塔池化，多尺度空间池化
pub fn spatial_pyramid_pooling(feature_map: &[Vec<f32>], height: usize, width: usize, 
                              levels: &[usize]) -> Result<Vec<f32>, VideoExtractionError> {
    if feature_map.is_empty() || height == 0 || width == 0 || levels.is_empty() {
        return Err(VideoExtractionError::ProcessingError(
            "空间金字塔池化：无效的输入参数".to_string()
        ));
    }
    
    if feature_map.len() != height * width {
        return Err(VideoExtractionError::ProcessingError(
            format!("空间金字塔池化：特征图尺寸不匹配，期望{}x{}={}, 实际{}", 
                    height, width, height * width, feature_map.len())
        ));
    }
    
    // 假设所有特征向量长度相同
    let channels = feature_map[0].len();
    let mut result = Vec::new();
    
    // 对每个层级进行池化
    for &level in levels {
        if level == 0 {
            warn!("空间金字塔池化：跳过层级0");
            continue;
        }
        
        // 计算网格尺寸
        let grid_h = height / level;
        let grid_w = width / level;
        
        // 确保网格尺寸不为0
        if grid_h == 0 || grid_w == 0 {
            warn!("空间金字塔池化：层级{}的网格尺寸过小，跳过", level);
            continue;
        }
        
        // 对每个网格进行池化
        for gy in 0..level {
            for gx in 0..level {
                let y_start = gy * grid_h;
                let x_start = gx * grid_w;
                let y_end = ((gy + 1) * grid_h).min(height);
                let x_end = ((gx + 1) * grid_w).min(width);
                
                // 初始化该网格的平均特征
                let mut grid_feature = vec![0.0; channels];
                let mut count = 0;
                
                // 计算网格内的平均特征
                for y in y_start..y_end {
                    for x in x_start..x_end {
                        let idx = y * width + x;
                        if idx < feature_map.len() {
                            for (c, &val) in feature_map[idx].iter().enumerate() {
                                if c < channels {
                                    grid_feature[c] += val;
                                }
                            }
                            count += 1;
                        }
                    }
                }
                
                // 归一化
                if count > 0 {
                    for val in &mut grid_feature {
                        *val /= count as f32;
                    }
                }
                
                // 添加到结果
                result.extend_from_slice(&grid_feature);
            }
        }
    }
    
    debug!("执行空间金字塔池化: {:?}层级 -> {}维输出", levels, result.len());
    Ok(result)
}

/// 矩形区域定义，用于ROI池化
#[derive(Debug, Clone, Copy)]
pub struct Rect {
    /// X坐标
    pub x: usize,
    /// Y坐标
    pub y: usize,
    /// 宽度
    pub width: usize,
    /// 高度
    pub height: usize,
}

impl Rect {
    /// 创建新的矩形区域
    pub fn new(x: usize, y: usize, width: usize, height: usize) -> Self {
        Self { x, y, width, height }
    }
    
    /// 检查矩形是否有效
    pub fn is_valid(&self, max_width: usize, max_height: usize) -> bool {
        // 检查矩形是否在图像范围内
        if self.x >= max_width || self.y >= max_height {
            return false;
        }
        
        // 检查矩形是否有效大小
        if self.width == 0 || self.height == 0 {
            return false;
        }
        
        // 检查矩形是否超出边界
        if self.x + self.width > max_width || self.y + self.height > max_height {
            return false;
        }
        
        true
    }
    
    /// 获取矩形面积
    pub fn area(&self) -> usize {
        self.width * self.height
    }
    
    /// 计算与另一个矩形的交集
    pub fn intersection(&self, other: &Rect) -> Option<Rect> {
        let x1 = self.x.max(other.x);
        let y1 = self.y.max(other.y);
        let x2 = (self.x + self.width).min(other.x + other.width);
        let y2 = (self.y + self.height).min(other.y + other.height);
        
        if x1 >= x2 || y1 >= y2 {
            return None; // 无交集
        }
        
        Some(Rect {
            x: x1,
            y: y1,
            width: x2 - x1,
            height: y2 - y1,
        })
    }
} 