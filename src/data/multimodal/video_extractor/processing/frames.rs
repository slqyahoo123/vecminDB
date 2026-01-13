//! 视频帧处理模块
//! 
//! 本模块提供视频帧的处理、转换和模拟生成功能

use crate::data::multimodal::video_extractor::types::*;
use crate::data::multimodal::video_extractor::config::VideoFeatureConfig;
use crate::data::multimodal::video_extractor::error::VideoExtractionError;
use log::{debug, info};
use std::time::Instant;

/// 预处理视频帧，应用各种转换和规范化操作
pub fn preprocess_frame(frame: &VideoFrame, config: &VideoFeatureConfig) -> Result<VideoFrame, VideoExtractionError> {
    let start_time = Instant::now();
    debug!("预处理视频帧：{}x{}x{}", frame.width, frame.height, frame.channels);
    
    // 检查输入帧是否有效
    if frame.data.is_empty() {
        return Err(VideoExtractionError::InputError(
            "无效的视频帧数据".to_string()
        ));
    }
    
    // 1. 调整大小（如果需要）
    let mut processed_frame = frame.clone();
    if frame.width != config.frame_width || frame.height != config.frame_height {
        processed_frame = resize_frame(&processed_frame, config.frame_width, config.frame_height)?;
        debug!("调整帧大小至 {}x{}", config.frame_width, config.frame_height);
    }
    
    // 2. 通道转换（如果需要）- 使用自定义参数
    if let Some(target_channels) = config.get_usize_param("frame_channels") {
        if processed_frame.channels != target_channels {
            processed_frame = convert_channels(&processed_frame, target_channels)?;
            debug!("转换帧通道数至 {}", target_channels);
        }
    }
    
    // 3. 颜色空间转换（如果需要）- 使用自定义参数
    if let Some(color_conversion) = config.get_string_param("color_conversion") {
        processed_frame = convert_color_space(&processed_frame, &color_conversion)?;
        debug!("应用颜色空间转换: {}", color_conversion);
    }
    
    // 4. 应用滤波器（如果配置）- 使用自定义参数
    if config.get_bool_param("apply_filters").unwrap_or(false) {
        processed_frame = apply_filters(&processed_frame, config)?;
        debug!("应用图像滤波器");
    }
    
    // 记录处理时间
    let elapsed = start_time.elapsed();
    debug!("帧预处理完成，耗时：{:?}", elapsed);
    
    Ok(processed_frame)
}

/// 生成模拟视频帧，用于测试和开发
pub fn simulate_video_frames(frame_count: usize, width: usize, height: usize) -> Result<Vec<VideoFrame>, VideoExtractionError> {
    info!("生成{}个模拟视频帧，尺寸：{}x{}", frame_count, width, height);
    
    // 检查参数有效性
    if frame_count == 0 || width == 0 || height == 0 {
        return Err(VideoExtractionError::InputError(
            format!("无效的模拟参数: frame_count={}, width={}, height={}", frame_count, width, height)
        ));
    }
    
    let mut frames = Vec::with_capacity(frame_count);
    let channels = 3; // RGB格式
    
    // 生成模拟帧
    for i in 0..frame_count {
        // 创建模拟图像数据
        let timestamp = i as f64 / 30.0; // 假设30 FPS
        let frame = generate_simulated_frame(width, height, channels, timestamp, i)?;
        frames.push(frame);
    }
    
    debug!("已生成{}个模拟帧", frames.len());
    
    Ok(frames)
}

/// 根据时间间隔生成模拟视频帧
pub fn simulate_interval_frames(
    frame_count: usize,
    width: usize,
    height: usize,
    start_time: f64,
    end_time: f64
) -> Result<Vec<VideoFrame>, VideoExtractionError> {
    info!("生成时间间隔[{:.2}-{:.2}]内的{}个模拟帧，尺寸：{}x{}", 
        start_time, end_time, frame_count, width, height);
    
    // 检查参数有效性
    if frame_count == 0 || width == 0 || height == 0 || start_time >= end_time {
        return Err(VideoExtractionError::InputError(
            format!("无效的模拟参数: frame_count={}, width={}, height={}, start_time={}, end_time={}", 
                frame_count, width, height, start_time, end_time)
        ));
    }
    
    let mut frames = Vec::with_capacity(frame_count);
    let channels = 3; // RGB格式
    let duration = end_time - start_time;
    
    // 生成指定间隔内的模拟帧
    for i in 0..frame_count {
        // 在指定区间内平均分布时间戳
        let progress = if frame_count > 1 { i as f64 / (frame_count - 1) as f64 } else { 0.0 };
        let timestamp = start_time + progress * duration;
        
        // 根据时间戳生成帧
        let frame = generate_simulated_frame(width, height, channels, timestamp, i)?;
        frames.push(frame);
    }
    
    debug!("已生成{}个间隔模拟帧", frames.len());
    
    Ok(frames)
}

/// 应用图像滤波器
fn apply_filters(frame: &VideoFrame, config: &VideoFeatureConfig) -> Result<VideoFrame, VideoExtractionError> {
    // 应用各种图像处理滤波器
    let mut result = frame.clone();
    
    // 使用高斯模糊滤波（简化示例）
    if config.get_bool_param("apply_gaussian_blur").unwrap_or(false) {
        result = apply_gaussian_blur(&result, 3)?; // 半径为3
    }
    
    // 使用锐化滤波
    if config.get_bool_param("apply_sharpen").unwrap_or(false) {
        result = apply_sharpen_filter(&result)?;
    }
    
    // 使用中值滤波（去除噪点）
    if config.get_bool_param("apply_median_filter").unwrap_or(false) {
        result = apply_median_filter(&result, 3)?; // 3x3核
    }
    
    Ok(result)
}

/// 应用高斯模糊滤波
fn apply_gaussian_blur(frame: &VideoFrame, radius: usize) -> Result<VideoFrame, VideoExtractionError> {
    let width = frame.width;
    let height = frame.height;
    let channels = frame.channels;
    
    // 简化实现：使用简单的均值滤波代替高斯滤波
    let mut result = VideoFrame::new(width, height, channels, frame.timestamp);
    
    for y in 0..height {
        for x in 0..width {
            for c in 0..channels {
                let mut sum = 0;
                let mut count = 0;
                
                // 简单的半径周围平均
                let radius_i = radius as isize;
                for dy in -radius_i..=radius_i {
                    for dx in -radius_i..=radius_i {
                        let nx = x as isize + dx;
                        let ny = y as isize + dy;
                        
                        if nx >= 0 && nx < width as isize && ny >= 0 && ny < height as isize {
                            let idx = ((ny as usize) * width + (nx as usize)) * channels + c;
                            if idx < frame.data.len() {
                                sum += frame.data[idx] as u32;
                                count += 1;
                            }
                        }
                    }
                }
                
                let avg = if count > 0 { (sum / count) as u8 } else { 0 };
                
                let dst_idx = (y * width + x) * channels + c;
                if dst_idx < result.data.len() {
                    result.data[dst_idx] = avg;
                }
            }
        }
    }
    
    Ok(result)
}

/// 应用锐化滤波
fn apply_sharpen_filter(frame: &VideoFrame) -> Result<VideoFrame, VideoExtractionError> {
    let width = frame.width;
    let height = frame.height;
    let channels = frame.channels;
    
    // 锐化核
    // [ 0 -1  0 ]
    // [-1  5 -1 ]
    // [ 0 -1  0 ]
    let kernel = [
        0, -1, 0,
        -1, 5, -1,
        0, -1, 0
    ];
    
    let mut result = VideoFrame::new(width, height, channels, frame.timestamp);
    
    for c in 0..channels {
        for y in 1..height-1 {
            for x in 1..width-1 {
                let mut sum = 0i32;
                
                // 应用卷积核
                sum += (frame.data[((y-1) * width + x) * channels + c] as i32) * kernel[1];  // 上
                sum += (frame.data[(y * width + (x-1)) * channels + c] as i32) * kernel[3];  // 左
                sum += (frame.data[(y * width + x) * channels + c] as i32) * kernel[4];      // 中心
                sum += (frame.data[(y * width + (x+1)) * channels + c] as i32) * kernel[5];  // 右
                sum += (frame.data[((y+1) * width + x) * channels + c] as i32) * kernel[7];  // 下
                
                // 限制到有效范围
                sum = sum.clamp(0, 255);
                
                let dst_idx = (y * width + x) * channels + c;
                if dst_idx < result.data.len() {
                    result.data[dst_idx] = sum as u8;
                }
            }
        }
        
        // 处理边界
        for y in 0..height {
            for x in [0, width-1] {
                if x < width {
                    let src_idx = (y * width + x) * channels + c;
                    let dst_idx = (y * width + x) * channels + c;
                    
                    if src_idx < frame.data.len() && dst_idx < result.data.len() {
                        result.data[dst_idx] = frame.data[src_idx];
                    }
                }
            }
        }
        
        for x in 0..width {
            for y in [0, height-1] {
                if y < height {
                    let src_idx = (y * width + x) * channels + c;
                    let dst_idx = (y * width + x) * channels + c;
                    
                    if src_idx < frame.data.len() && dst_idx < result.data.len() {
                        result.data[dst_idx] = frame.data[src_idx];
                    }
                }
            }
        }
    }
    
    Ok(result)
}

/// 应用中值滤波器
fn apply_median_filter(frame: &VideoFrame, size: usize) -> Result<VideoFrame, VideoExtractionError> {
    if size % 2 == 0 {
        return Err(VideoExtractionError::ProcessingError(
            format!("中值滤波窗口大小必须是奇数: {}", size)
        ));
    }
    
    let width = frame.width;
    let height = frame.height;
    let channels = frame.channels;
    
    let mut result = VideoFrame::new(width, height, channels, frame.timestamp);
    let radius = size / 2;
    
    for c in 0..channels {
        for y in 0..height {
            for x in 0..width {
                // 收集窗口内的像素
                let mut window = Vec::with_capacity(size * size);
                
                for dy in -(radius as isize)..=(radius as isize) {
                    for dx in -(radius as isize)..=(radius as isize) {
                        let nx = (x as isize + dx).clamp(0, width as isize - 1) as usize;
                        let ny = (y as isize + dy).clamp(0, height as isize - 1) as usize;
                        
                        let idx = (ny * width + nx) * channels + c;
                        if idx < frame.data.len() {
                            window.push(frame.data[idx]);
                        }
                    }
                }
                
                // 对窗口内的像素排序并取中值
                window.sort_unstable();
                let median = if !window.is_empty() {
                    window[window.len() / 2]
                } else {
                    0
                };
                
                let dst_idx = (y * width + x) * channels + c;
                if dst_idx < result.data.len() {
                    result.data[dst_idx] = median;
                }
            }
        }
    }
    
    Ok(result)
}

/// 调整视频帧大小
fn resize_frame(frame: &VideoFrame, target_width: usize, target_height: usize) -> Result<VideoFrame, VideoExtractionError> {
    let src_width = frame.width;
    let src_height = frame.height;
    let channels = frame.channels;
    
    if target_width == 0 || target_height == 0 {
        return Err(VideoExtractionError::ProcessingError(
            format!("无效的目标尺寸: {}x{}", target_width, target_height)
        ));
    }
    
    if src_width == target_width && src_height == target_height {
        return Ok(frame.clone());
    }
    
    let mut result = VideoFrame::new(target_width, target_height, channels, frame.timestamp);
    
    // 双线性插值缩放
    let x_ratio = src_width as f32 / target_width as f32;
    let y_ratio = src_height as f32 / target_height as f32;
    
    for y in 0..target_height {
        for x in 0..target_width {
            let src_x = (x as f32 * x_ratio).floor() as usize;
            let src_y = (y as f32 * y_ratio).floor() as usize;
            
            let src_x_next = (src_x + 1).min(src_width - 1);
            let src_y_next = (src_y + 1).min(src_height - 1);
            
            let x_diff = (x as f32 * x_ratio) - src_x as f32;
            let y_diff = (y as f32 * y_ratio) - src_y as f32;
            
            for c in 0..channels {
                // 获取四个相邻点的颜色
                let top_left = frame.data[(src_y * src_width + src_x) * channels + c] as f32;
                let top_right = frame.data[(src_y * src_width + src_x_next) * channels + c] as f32;
                let bottom_left = frame.data[(src_y_next * src_width + src_x) * channels + c] as f32;
                let bottom_right = frame.data[(src_y_next * src_width + src_x_next) * channels + c] as f32;
                
                // 双线性插值
                let top = top_left * (1.0 - x_diff) + top_right * x_diff;
                let bottom = bottom_left * (1.0 - x_diff) + bottom_right * x_diff;
                let value = top * (1.0 - y_diff) + bottom * y_diff;
                
                // 存储结果
                let dst_idx = (y * target_width + x) * channels + c;
                result.data[dst_idx] = value.clamp(0.0, 255.0) as u8;
            }
        }
    }
    
    Ok(result)
}

/// 转换通道数
fn convert_channels(frame: &VideoFrame, target_channels: usize) -> Result<VideoFrame, VideoExtractionError> {
    let width = frame.width;
    let height = frame.height;
    let src_channels = frame.channels;
    
    if src_channels == target_channels {
        return Ok(frame.clone());
    }
    
    let mut converted = VideoFrame::new(width, height, target_channels, frame.timestamp);
    
    match (src_channels, target_channels) {
        (3, 1) => {
            // RGB到灰度
            for y in 0..height {
                for x in 0..width {
                    let src_idx = (y * width + x) * 3;
                    let dst_idx = y * width + x;
                    
                    if src_idx + 2 < frame.data.len() && dst_idx < converted.data.len() {
                        // 使用加权方式转换为灰度
                        let r = frame.data[src_idx] as f32 * 0.299;
                        let g = frame.data[src_idx + 1] as f32 * 0.587;
                        let b = frame.data[src_idx + 2] as f32 * 0.114;
                        
                        converted.data[dst_idx] = (r + g + b).round() as u8;
                    }
                }
            }
        },
        (1, 3) => {
            // 灰度到RGB（复制相同值）
            for y in 0..height {
                for x in 0..width {
                    let src_idx = y * width + x;
                    let dst_idx = (y * width + x) * 3;
                    
                    if src_idx < frame.data.len() && dst_idx + 2 < converted.data.len() {
                        let gray = frame.data[src_idx];
                        converted.data[dst_idx] = gray;     // R
                        converted.data[dst_idx + 1] = gray; // G
                        converted.data[dst_idx + 2] = gray; // B
                    }
                }
            }
        },
        (4, 3) => {
            // RGBA到RGB (丢弃Alpha通道)
            for y in 0..height {
                for x in 0..width {
                    let src_idx = (y * width + x) * 4;
                    let dst_idx = (y * width + x) * 3;
                    
                    if src_idx + 3 < frame.data.len() && dst_idx + 2 < converted.data.len() {
                        converted.data[dst_idx] = frame.data[src_idx];         // R
                        converted.data[dst_idx + 1] = frame.data[src_idx + 1]; // G
                        converted.data[dst_idx + 2] = frame.data[src_idx + 2]; // B
                    }
                }
            }
        },
        (3, 4) => {
            // RGB到RGBA (添加Alpha=255)
            for y in 0..height {
                for x in 0..width {
                    let src_idx = (y * width + x) * 3;
                    let dst_idx = (y * width + x) * 4;
                    
                    if src_idx + 2 < frame.data.len() && dst_idx + 3 < converted.data.len() {
                        converted.data[dst_idx] = frame.data[src_idx];         // R
                        converted.data[dst_idx + 1] = frame.data[src_idx + 1]; // G
                        converted.data[dst_idx + 2] = frame.data[src_idx + 2]; // B
                        converted.data[dst_idx + 3] = 255;                     // A (不透明)
                    }
                }
            }
        },
        _ => {
            return Err(VideoExtractionError::ProcessingError(
                format!("不支持的通道转换: {} -> {}", src_channels, target_channels)
            ));
        }
    }
    
    Ok(converted)
}

/// 转换颜色空间
fn convert_color_space(frame: &VideoFrame, conversion_type: &str) -> Result<VideoFrame, VideoExtractionError> {
    let width = frame.width;
    let height = frame.height;
    let channels = frame.channels;
    
    match conversion_type {
        "RGB2HSV" => {
            if channels != 3 {
                return Err(VideoExtractionError::ProcessingError(
                    format!("RGB到HSV转换需要3通道输入，当前: {}", channels)
                ));
            }
            
            let mut converted = VideoFrame::new(width, height, 3, frame.timestamp);
            
            for y in 0..height {
                for x in 0..width {
                    let idx = (y * width + x) * 3;
                    
                    if idx + 2 < frame.data.len() && idx + 2 < converted.data.len() {
                        let r = frame.data[idx] as f32 / 255.0;
                        let g = frame.data[idx + 1] as f32 / 255.0;
                        let b = frame.data[idx + 2] as f32 / 255.0;
                        
                        let max_val = r.max(g).max(b);
                        let min_val = r.min(g).min(b);
                        let delta = max_val - min_val;
                        
                        // 计算HSV
                        let mut h = 0.0;
                        if delta > 0.0 {
                            if max_val == r {
                                h = ((g - b) / delta).rem_euclid(6.0);
                            } else if max_val == g {
                                h = (b - r) / delta + 2.0;
                            } else {
                                h = (r - g) / delta + 4.0;
                            }
                            h *= 60.0; // 转换为角度
                        }
                        
                        let s = if max_val > 0.0 { delta / max_val } else { 0.0 };
                        let v = max_val;
                        
                        // 存储HSV (缩放到0-255)
                        converted.data[idx] = (h / 2.0).round() as u8;         // H (0-180)
                        converted.data[idx + 1] = (s * 255.0).round() as u8;   // S (0-255)
                        converted.data[idx + 2] = (v * 255.0).round() as u8;   // V (0-255)
                    }
                }
            }
            
            Ok(converted)
        },
        "RGB2YUV" => {
            if channels != 3 {
                return Err(VideoExtractionError::ProcessingError(
                    format!("RGB到YUV转换需要3通道输入，当前: {}", channels)
                ));
            }
            
            let mut converted = VideoFrame::new(width, height, 3, frame.timestamp);
            
            for y in 0..height {
                for x in 0..width {
                    let idx = (y * width + x) * 3;
                    
                    if idx + 2 < frame.data.len() && idx + 2 < converted.data.len() {
                        let r = frame.data[idx] as f32;
                        let g = frame.data[idx + 1] as f32;
                        let b = frame.data[idx + 2] as f32;
                        
                        // RGB到YUV转换公式
                        let y_val = (0.299 * r + 0.587 * g + 0.114 * b).round() as u8;
                        let u_val = ((-0.169 * r - 0.331 * g + 0.5 * b) + 128.0).round() as u8;
                        let v_val = ((0.5 * r - 0.419 * g - 0.081 * b) + 128.0).round() as u8;
                        
                        converted.data[idx] = y_val;     // Y
                        converted.data[idx + 1] = u_val; // U
                        converted.data[idx + 2] = v_val; // V
                    }
                }
            }
            
            Ok(converted)
        },
        _ => {
            Err(VideoExtractionError::ProcessingError(
                format!("不支持的颜色空间转换: {}", conversion_type)
            ))
        }
    }
}

/// 生成模拟视频帧，用于测试
fn generate_simulated_frame(width: usize, height: usize, channels: usize, timestamp: f64, frame_index: usize) -> Result<VideoFrame, VideoExtractionError> {
    // 确认参数有效性
    if width == 0 || height == 0 || channels == 0 || channels > 4 {
        return Err(VideoExtractionError::ProcessingError(
            format!("无效的模拟帧参数: width={}, height={}, channels={}", width, height, channels)
        ));
    }
    
    let mut frame = VideoFrame::new(width, height, channels, timestamp);
    
    // 生成一个基于索引的伪随机数，使同一帧索引每次都生成相同的图像
    let mut seed = frame_index as u64;
    
    // 简单的伪随机数生成器
    let mut rand = || {
        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (seed >> 32) as u32
    };
    
    // 为了让生成的帧更有真实感，我们提供几种不同的模拟场景
    // 根据frame_index选择不同的场景类型
    let scene_type = frame_index % 5;
    
    match scene_type {
        0 => {
            // 场景1: 渐变背景上的移动物体
            let progress = (timestamp * 0.5).sin() * 0.5 + 0.5; // 0.0 - 1.0范围内变化
            let obj_x = (width as f64 * progress) as usize;
            let obj_y = height / 2;
            let obj_radius = height as f64 * 0.15;
            
            for y in 0..height {
                for x in 0..width {
                    // 计算到物体中心的距离
                    let dx = x as f64 - obj_x as f64;
                    let dy = y as f64 - obj_y as f64;
                    let distance = (dx * dx + dy * dy).sqrt();
                    
                    // 生成渐变背景
                    let background_r = (x * 255 / width) as u8;
                    let background_g = (y * 255 / height) as u8;
                    let background_b = ((x + y) * 127 / (width + height)) as u8;
                    
                    // 确定索引
                    let idx = (y * width + x) * channels;
                    
                    // 绘制物体
                    if distance < obj_radius {
                        // 物体内部
                        if channels >= 3 {
                            frame.data[idx] = 255;     // 红色
                            frame.data[idx + 1] = 0;    // 绿色
                            frame.data[idx + 2] = 0;    // 蓝色
                            
                            if channels == 4 {
                                frame.data[idx + 3] = 255; // 完全不透明
                            }
                        } else {
                            // 灰度图像
                            frame.data[idx] = 255;
                        }
                    } else {
                        // 背景
                        if channels >= 3 {
                            frame.data[idx] = background_r;
                            frame.data[idx + 1] = background_g;
                            frame.data[idx + 2] = background_b;
                            
                            if channels == 4 {
                                frame.data[idx + 3] = 255; // 完全不透明
                            }
                        } else {
                            // 灰度图像 - 转换RGB到灰度
                            frame.data[idx] = (0.299 * background_r as f64 + 
                                              0.587 * background_g as f64 + 
                                              0.114 * background_b as f64) as u8;
                        }
                    }
                }
            }
        },
        1 => {
            // 场景2: 棋盘格
            let square_size = width.min(height) / 8;
            
            for y in 0..height {
                for x in 0..width {
                    let is_white = ((x / square_size) + (y / square_size)) % 2 == 0;
                    let idx = (y * width + x) * channels;
                    
                    if is_white {
                        // 白色格子
                        if channels >= 3 {
                            frame.data[idx] = 255;     // 红色
                            frame.data[idx + 1] = 255;   // 绿色
                            frame.data[idx + 2] = 255;   // 蓝色
                            
                            if channels == 4 {
                                frame.data[idx + 3] = 255; // 完全不透明
                            }
                        } else {
                            // 灰度图像
                            frame.data[idx] = 255;
                        }
                    } else {
                        // 黑色格子
                        if channels >= 3 {
                            frame.data[idx] = 0;       // 红色
                            frame.data[idx + 1] = 0;     // 绿色
                            frame.data[idx + 2] = 0;     // 蓝色
                            
                            if channels == 4 {
                                frame.data[idx + 3] = 255; // 完全不透明
                            }
                        } else {
                            // 灰度图像
                            frame.data[idx] = 0;
                        }
                    }
                }
            }
        },
        2 => {
            // 场景3: 噪声纹理
            for y in 0..height {
                for x in 0..width {
                    let idx = (y * width + x) * channels;
                    
                    if channels >= 3 {
                        frame.data[idx] = (rand() % 256) as u8;     // 红色
                        frame.data[idx + 1] = (rand() % 256) as u8;   // 绿色
                        frame.data[idx + 2] = (rand() % 256) as u8;   // 蓝色
                        
                        if channels == 4 {
                            frame.data[idx + 3] = 255; // 完全不透明
                        }
                    } else {
                        // 灰度图像
                        frame.data[idx] = (rand() % 256) as u8;
                    }
                }
            }
        },
        3 => {
            // 场景4: 彩色条纹
            let stripe_width = width / 7;
            
            for y in 0..height {
                for x in 0..width {
                    let stripe = x / stripe_width;
                    let idx = (y * width + x) * channels;
                    
                    if channels >= 3 {
                        match stripe % 7 {
                            0 => { // 红色
                                frame.data[idx] = 255;
                                frame.data[idx + 1] = 0;
                                frame.data[idx + 2] = 0;
                            },
                            1 => { // 橙色
                                frame.data[idx] = 255;
                                frame.data[idx + 1] = 127;
                                frame.data[idx + 2] = 0;
                            },
                            2 => { // 黄色
                                frame.data[idx] = 255;
                                frame.data[idx + 1] = 255;
                                frame.data[idx + 2] = 0;
                            },
                            3 => { // 绿色
                                frame.data[idx] = 0;
                                frame.data[idx + 1] = 255;
                                frame.data[idx + 2] = 0;
                            },
                            4 => { // 蓝色
                                frame.data[idx] = 0;
                                frame.data[idx + 1] = 0;
                                frame.data[idx + 2] = 255;
                            },
                            5 => { // 靛蓝
                                frame.data[idx] = 75;
                                frame.data[idx + 1] = 0;
                                frame.data[idx + 2] = 130;
                            },
                            6 => { // 紫色
                                frame.data[idx] = 148;
                                frame.data[idx + 1] = 0;
                                frame.data[idx + 2] = 211;
                            },
                            _ => {}
                        }
                        
                        if channels == 4 {
                            frame.data[idx + 3] = 255; // 完全不透明
                        }
                    } else {
                        // 灰度图像 - 灰度条纹
                        frame.data[idx] = match stripe % 7 {
                            0 => 255,   // 白色
                            1 => 220,   // 浅灰
                            2 => 180,
                            3 => 150,
                            4 => 100,
                            5 => 50,
                            6 => 0,     // 黑色
                            _ => 128,   // 中灰
                        };
                    }
                }
            }
        },
        4 => {
            // 场景5: 波纹效果
            for y in 0..height {
                for x in 0..width {
                    // 计算到中心的距离
                    let center_x = width as f64 / 2.0;
                    let center_y = height as f64 / 2.0;
                    let dx = x as f64 - center_x;
                    let dy = y as f64 - center_y;
                    let distance = (dx * dx + dy * dy).sqrt();
                    
                    // 创建波纹效果
                    let wave = (distance / 10.0 + timestamp * 3.0).sin() * 0.5 + 0.5; // 0-1范围
                    
                    let idx = (y * width + x) * channels;
                    
                    if channels >= 3 {
                        frame.data[idx] = (wave * 255.0) as u8;       // 红色
                        frame.data[idx + 1] = ((1.0 - wave) * 255.0) as u8; // 绿色
                        frame.data[idx + 2] = ((wave * 0.5) * 255.0) as u8; // 蓝色
                        
                        if channels == 4 {
                            frame.data[idx + 3] = 255; // 完全不透明
                        }
                    } else {
                        // 灰度图像
                        frame.data[idx] = (wave * 255.0) as u8;
                    }
                }
            }
        },
        _ => {
            // 默认场景: 纯色随机块
            let block_size = width.min(height) / 20;
            
            for y in 0..height {
                for x in 0..width {
                    let block_x = x / block_size;
                    let block_y = y / block_size;
                    
                    // 使用块坐标生成伪随机颜色
                    let mut block_seed = (block_x * 31337 + block_y * 7919 + frame_index * 104729) as u64;
                    block_seed = block_seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                    
                    let idx = (y * width + x) * channels;
                    
                    if channels >= 3 {
                        frame.data[idx] = ((block_seed >> 32) % 256) as u8;     // 红色
                        frame.data[idx + 1] = ((block_seed >> 40) % 256) as u8;   // 绿色
                        frame.data[idx + 2] = ((block_seed >> 48) % 256) as u8;   // 蓝色
                        
                        if channels == 4 {
                            frame.data[idx + 3] = 255; // 完全不透明
                        }
                    } else {
                        // 灰度图像
                        frame.data[idx] = ((block_seed >> 32) % 256) as u8;
                    }
                }
            }
        }
    }
    
    Ok(frame)
} 