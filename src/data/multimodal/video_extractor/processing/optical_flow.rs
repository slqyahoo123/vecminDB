//! 光流特征提取器实现
//! 
//! 本模块提供从视频帧序列中提取光流特征的核心算法

use super::FeatureExtractor;
use crate::data::multimodal::video_extractor::types::*;
use crate::data::multimodal::video_extractor::config::VideoFeatureConfig;
use crate::data::multimodal::video_extractor::error::VideoExtractionError;
use std::collections::VecDeque;
use log::{info, debug, warn};

/// 光流特征提取器，负责从连续视频帧中提取运动信息
pub struct OpticalFlowExtractor {
    /// 特征维度
    feature_dim: usize,
    /// 光流算法类型
    flow_algorithm: String,
    /// 堆叠帧数
    stacked_frames: usize,
    /// 特征类型
    feature_type: VideoFeatureType,
    /// 归一化类型
    normalization: Option<NormalizationType>,
    /// 是否已初始化
    initialized: bool,
    /// 帧缓冲，用于存储连续帧
    frame_buffer: VecDeque<VideoFrame>,
    /// 配置项
    config: VideoFeatureConfig,
    /// 光流方向分箱数
    flow_histogram_bins: usize,
}

impl OpticalFlowExtractor {
    /// 创建新的光流特征提取器
    pub fn new(config: &VideoFeatureConfig) -> Self {
        // 从配置中提取参数
        let feature_dim = config.get_usize_param("feature_dimension").unwrap_or(256);
        let flow_algorithm = config.get_string_param("flow_algorithm").unwrap_or_else(|| "farneback".to_string());
        let stacked_frames = config.get_usize_param("stacked_frames").unwrap_or(5);
        let flow_histogram_bins = config.get_usize_param("flow_histogram_bins").unwrap_or(8);
        
        Self {
            feature_dim,
            flow_algorithm,
            stacked_frames,
            feature_type: VideoFeatureType::OpticalFlow,
            normalization: config.normalization,
            initialized: false,
            frame_buffer: VecDeque::with_capacity(stacked_frames + 1),
            config: config.clone(),
            flow_histogram_bins,
        }
    }
    
    /// 计算两帧之间的光流
    fn compute_optical_flow(&self, prev_frame: &VideoFrame, curr_frame: &VideoFrame) 
        -> Result<(Vec<f32>, Vec<f32>), VideoExtractionError> {
        
        // 确保两帧尺寸相同
        if prev_frame.width != curr_frame.width || prev_frame.height != curr_frame.height {
            return Err(VideoExtractionError::ProcessingError(
                format!("帧尺寸不匹配：({}, {}) vs ({}, {})",
                    prev_frame.width, prev_frame.height,
                    curr_frame.width, curr_frame.height)
            ));
        }
        
        let width = prev_frame.width;
        let height = prev_frame.height;
        let pixel_count = width * height;
        
        // 将两帧转换为灰度图
        let prev_gray = Self::to_grayscale(prev_frame)?;
        let curr_gray = Self::to_grayscale(curr_frame)?;
        
        match self.flow_algorithm.as_str() {
            "farneback" => {
                // 简化实现：模拟Farneback光流算法的输出
                // 实际应用中应该使用OpenCV或其他图像处理库
                
                // 分配光流向量内存
                let mut flow_x = vec![0.0f32; pixel_count];
                let mut flow_y = vec![0.0f32; pixel_count];
                
                // 计算简化版的光流：基于相邻像素的差异
                for y in 1..height-1 {
                    for x in 1..width-1 {
                        let idx = y * width + x;
                        let prev_val = prev_gray[idx];
                        let curr_val = curr_gray[idx];
                        
                        // 计算水平和垂直梯度
                        let dx = (curr_gray[idx+1] as f32 - curr_gray[idx-1] as f32) / 2.0;
                        let dy = (curr_gray[idx+width] as f32 - curr_gray[idx-width] as f32) / 2.0;
                        
                        // 计算时间梯度
                        let dt = curr_val as f32 - prev_val as f32;
                        
                        // 简化的运动估计
                        if dx.abs() > 1e-6 && dy.abs() > 1e-6 {
                            flow_x[idx] = -dt / dx;
                            flow_y[idx] = -dt / dy;
                        }
                        
                        // 限制光流大小
                        flow_x[idx] = flow_x[idx].clamp(-20.0, 20.0);
                        flow_y[idx] = flow_y[idx].clamp(-20.0, 20.0);
                    }
                }
                
                Ok((flow_x, flow_y))
            },
            "lucas_kanade" => {
                // 简化实现：模拟Lucas-Kanade光流算法
                // 实际实现应该使用OpenCV等库的密集光流算法
                
                // 分配光流向量内存
                let mut flow_x = vec![0.0f32; pixel_count];
                let mut flow_y = vec![0.0f32; pixel_count];
                
                // 窗口大小
                let window_size = 3;
                let half_window = window_size / 2;
                
                // 对每一点计算光流
                for y in (half_window as usize)..(height - half_window as usize) {
                    for x in (half_window as usize)..(width - half_window as usize) {
                        
                        // 模拟Lucas-Kanade算法的块匹配
                        let mut best_dx = 0.0f32;
                        let mut best_dy = 0.0f32;
                        let mut min_diff = f32::MAX;
                        
                        // 搜索范围
                        for dy in -3..=3 {
                            for dx in -3..=3 {
                                let mut sum_diff = 0.0f32;
                                
                                // 计算块差异
                                for wy in -half_window..=half_window {
                                    for wx in -half_window..=half_window {
                                        let prev_y = (y as i32 + wy) as usize;
                                        let prev_x = (x as i32 + wx) as usize;
                                        let curr_y = (y as i32 + wy + dy) as usize;
                                        let curr_x = (x as i32 + wx + dx) as usize;
                                        
                                        if prev_y < height && prev_x < width && 
                                           curr_y < height && curr_x < width {
                                            let prev_idx = prev_y * width + prev_x;
                                            let curr_idx = curr_y * width + curr_x;
                                            
                                            if prev_idx < prev_gray.len() && curr_idx < curr_gray.len() {
                                                let diff = (prev_gray[prev_idx] as f32 - curr_gray[curr_idx] as f32).abs();
                                                sum_diff += diff;
                                            }
                                        }
                                    }
                                }
                                
                                if sum_diff < min_diff {
                                    min_diff = sum_diff;
                                    best_dx = dx as f32;
                                    best_dy = dy as f32;
                                }
                            }
                        }
                        
                        let idx = y * width + x;
                        flow_x[idx] = best_dx;
                        flow_y[idx] = best_dy;
                    }
                }
                
                Ok((flow_x, flow_y))
            },
            "dense" => {
                // 另一种密集光流算法的简化实现
                let mut flow_x = vec![0.0f32; pixel_count];
                let mut flow_y = vec![0.0f32; pixel_count];
                
                // 简化实现：使用帧差法估计运动
                for y in 0..height {
                    for x in 0..width {
                        let idx = y * width + x;
                        
                        // 在附近区域搜索最佳匹配
                        let mut best_match_x = x;
                        let mut best_match_y = y;
                        let mut min_diff = u32::MAX;
                        
                        // 搜索范围
                        let search_range = 5;
                        
                        for sy in y.saturating_sub(search_range)..=(y + search_range).min(height - 1) {
                            for sx in x.saturating_sub(search_range)..=(x + search_range).min(width - 1) {
                                let search_idx = sy * width + sx;
                                
                                if idx < prev_gray.len() && search_idx < curr_gray.len() {
                                    let diff = (prev_gray[idx] as i32 - curr_gray[search_idx] as i32).abs() as u32;
                                    
                                    if diff < min_diff {
                                        min_diff = diff;
                                        best_match_x = sx;
                                        best_match_y = sy;
                                    }
                                }
                            }
                        }
                        
                        flow_x[idx] = (best_match_x as i32 - x as i32) as f32;
                        flow_y[idx] = (best_match_y as i32 - y as i32) as f32;
                    }
                }
                
                Ok((flow_x, flow_y))
            },
            _ => Err(VideoExtractionError::ProcessingError(
                format!("不支持的光流算法: {}", self.flow_algorithm)
            )),
        }
    }
    
    /// 使用深度学习模型计算光流
    fn compute_deep_flow(
        &self,
        prev_gray: &[u8],
        curr_gray: &[u8],
        width: usize,
        height: usize
    ) -> Result<(Vec<f32>, Vec<f32>), VideoExtractionError> {
        debug!("使用深度学习方法计算光流，图像尺寸: {}x{}", width, height);
        
        #[cfg(feature = "tensorflow")]
        {
            use tensorflow::{Graph, ImportGraphDefOptions, Session, SessionOptions, Tensor};
            
            // 检查TensorFlow模型是否存在
            let model_dir = std::env::var("MODEL_DIR").unwrap_or_else(|_| "./models".to_string());
            let model_path = format!("{}/deep_flow.pb", model_dir);
            
            if !std::path::Path::new(&model_path).exists() {
                return Err(VideoExtractionError::ProcessingError(
                    format!("DeepFlow模型文件不存在: {}", model_path)
                ));
            }
            
            // 加载图形
            let mut graph = Graph::new();
            let model_data = std::fs::read(&model_path).map_err(|e| VideoExtractionError::ProcessingError(
                format!("读取DeepFlow模型失败: {}", e)
            ))?;
            
            graph.import_graph_def(&model_data, &ImportGraphDefOptions::new())
                .map_err(|e| VideoExtractionError::ProcessingError(
                    format!("导入TensorFlow图形定义失败: {}", e)
                ))?;
            
            // 创建会话
            let session = Session::new(&SessionOptions::new(), &graph)
                .map_err(|e| VideoExtractionError::ProcessingError(
                    format!("创建TensorFlow会话失败: {}", e)
                ))?;
            
            // 准备输入张量 - 前一帧和当前帧
            let prev_tensor = self.prepare_input_tensor(prev_gray, width, height)?;
            let curr_tensor = self.prepare_input_tensor(curr_gray, width, height)?;
            
            // 运行推理
            let mut args = tensorflow::SessionRunArgs::new();
            args.add_feed(&graph.operation_by_name_required("prev_frame").map_err(|e| {
                VideoExtractionError::ProcessingError(format!("找不到模型输入节点'prev_frame': {}", e))
            })?, 0, &prev_tensor);
            args.add_feed(&graph.operation_by_name_required("curr_frame").map_err(|e| {
                VideoExtractionError::ProcessingError(format!("找不到模型输入节点'curr_frame': {}", e))
            })?, 0, &curr_tensor);
            
            // 获取输出
            let flow_x_fetch = args.request_fetch(&graph.operation_by_name_required("flow_x").map_err(|e| {
                VideoExtractionError::ProcessingError(format!("找不到模型输出节点'flow_x': {}", e))
            })?, 0);
            let flow_y_fetch = args.request_fetch(&graph.operation_by_name_required("flow_y").map_err(|e| {
                VideoExtractionError::ProcessingError(format!("找不到模型输出节点'flow_y': {}", e))
            })?, 0);
            
            // 执行会话
            session.run(&mut args).map_err(|e| VideoExtractionError::ProcessingError(
                format!("执行TensorFlow推理失败: {}", e)
            ))?;
            
            // 获取结果
            let flow_x_tensor: Tensor<f32> = args.fetch(flow_x_fetch).map_err(|e| {
                VideoExtractionError::ProcessingError(format!("获取flow_x输出失败: {}", e))
            })?;
            let flow_y_tensor: Tensor<f32> = args.fetch(flow_y_fetch).map_err(|e| {
                VideoExtractionError::ProcessingError(format!("获取flow_y输出失败: {}", e))
            })?;
            
            // 转换为Vec
            let pixel_count = width * height;
            let mut flow_x = vec![0.0f32; pixel_count];
            let mut flow_y = vec![0.0f32; pixel_count];
            
            // 假设输出形状为 [height, width, 1]
            for y in 0..height {
                for x in 0..width {
                    let idx = y * width + x;
                    flow_x[idx] = flow_x_tensor[y][x][0];
                    flow_y[idx] = flow_y_tensor[y][x][0];
                }
            }
            
            debug!("深度光流计算完成，结果大小: {}x{}", width, height);
            Ok((flow_x, flow_y))
        }
        
        #[cfg(feature = "opencv")]
        {
            use opencv::{
                core::{Mat, MatTraitConst, Size, CV_8UC1},
                imgproc::{filter_2d, get_gaussian_kernel, BORDER_DEFAULT},
                optflow::{calc_optical_flow_farneback, OPTFLOW_FARNEBACK_GAUSSIAN},
                prelude::*,
            };
            
            // 准备输入图像
            let mut prev_mat = unsafe {
                Mat::new_rows_cols_with_data(
                    height as i32,
                    width as i32,
                    CV_8UC1,
                    prev_gray.as_ptr() as *mut std::ffi::c_void,
                    width as usize,
                ).map_err(|e| VideoExtractionError::ProcessingError(
                    format!("创建OpenCV矩阵失败: {}", e)
                ))?
            };
            
            let mut curr_mat = unsafe {
                Mat::new_rows_cols_with_data(
                    height as i32,
                    width as i32,
                    CV_8UC1,
                    curr_gray.as_ptr() as *mut std::ffi::c_void,
                    width as usize,
                ).map_err(|e| VideoExtractionError::ProcessingError(
                    format!("创建OpenCV矩阵失败: {}", e)
                ))?
            };
            
            // 创建输出流场
            let mut flow = Mat::new_rows_cols(
                height as i32,
                width as i32,
                opencv::core::CV_32FC2
            ).map_err(|e| VideoExtractionError::ProcessingError(
                format!("创建OpenCV流场矩阵失败: {}", e)
            ))?;
            
            // 计算光流
            // 参数：
            // 1. 金字塔层级 = 3
            // 2. 窗口大小 = 15
            // 3. 迭代次数 = 3
            // 4. 多项式展开邻域大小 = 5
            // 5. Sigma = 1.2
            // 6. 流动参数 = 0.5
            calc_optical_flow_farneback(
                &prev_mat,
                &curr_mat,
                &mut flow,
                0.5,  // 金字塔尺度
                3,    // 金字塔层级
                15,   // 窗口大小
                3,    // 迭代次数
                5,    // 多项式展开邻域大小
                1.2,  // 高斯标准差
                OPTFLOW_FARNEBACK_GAUSSIAN // 使用高斯核
            ).map_err(|e| VideoExtractionError::ProcessingError(
                format!("计算光流失败: {}", e)
            ))?;
            
            // 提取x和y分量
            let pixel_count = width * height;
            let mut flow_x = vec![0.0f32; pixel_count];
            let mut flow_y = vec![0.0f32; pixel_count];
            
            for y in 0..height {
                for x in 0..width {
                    let vector = flow.at_2d::<opencv::core::Vec2f>(y as i32, x as i32)
                        .map_err(|e| VideoExtractionError::ProcessingError(
                            format!("访问流场数据失败: {}", e)
                        ))?;
                    
                    let idx = y * width + x;
                    flow_x[idx] = vector[0];
                    flow_y[idx] = vector[1];
                }
            }
            
            debug!("OpenCV光流计算完成，结果大小: {}x{}", width, height);
            Ok((flow_x, flow_y))
        }
        
        #[cfg(not(any(feature = "tensorflow", feature = "opencv")))]
        {
            warn!("未启用TensorFlow或OpenCV特性，使用简化的光流计算");
            
            // 使用简化版的稀疏Lucas-Kanade算法作为后备实现
            let pixel_count = width * height;
            let mut flow_x = vec![0.0f32; pixel_count];
            let mut flow_y = vec![0.0f32; pixel_count];
            
            // 采样点间隔
            let step = 8;
            let window_size = 15;
            let half_window = window_size / 2;
            
            // 对稀疏网格点计算光流
            for y in (half_window..height-half_window).step_by(step) {
                for x in (half_window..width-half_window).step_by(step) {
                    // 获取当前块的最佳匹配位置
                    let (best_dx, best_dy) = self.find_best_match(
                        prev_gray, curr_gray, x, y, width, height, window_size
                    );
                    
                    // 设置当前网格点的光流向量
                    let idx = y * width + x;
                    flow_x[idx] = best_dx;
                    flow_y[idx] = best_dy;
                }
            }
            
            // 插值填充其他点
            for y in 0..height {
                for x in 0..width {
                    let idx = y * width + x;
                    
                    // 如果这个点已经计算过，跳过
                    if (y % step == half_window && x % step == half_window) && 
                       y >= half_window && y < height - half_window &&
                       x >= half_window && x < width - half_window {
                        continue;
                    }
                    
                    // 找到最近的四个采样点
                    let grid_x = (x / step) * step + half_window;
                    let grid_y = (y / step) * step + half_window;
                    
                    // 确保网格点在有效范围内
                    let grid_x = grid_x.clamp(half_window, width - half_window - 1);
                    let grid_y = grid_y.clamp(half_window, height - half_window - 1);
                    
                    // 简单的就近插值
                    let grid_idx = grid_y * width + grid_x;
                    flow_x[idx] = flow_x[grid_idx];
                    flow_y[idx] = flow_y[grid_idx];
                }
            }
            
            debug!("简化光流计算完成，结果大小: {}x{}", width, height);
            Ok((flow_x, flow_y))
        }
    }
    
    /// 准备输入张量
    fn prepare_input_tensor(&self, gray_image: &[u8], width: usize, height: usize) -> Result<Vec<f32>, VideoExtractionError> {
        // 创建归一化的浮点数向量
        let mut tensor_data = Vec::with_capacity(width * height);
        
        // 填充数据 - 归一化到[0,1]范围
        for y in 0..height {
            for x in 0..width {
                let idx = y * width + x;
                if idx < gray_image.len() {
                    tensor_data.push(gray_image[idx] as f32 / 255.0);
                }
            }
        }
        
        Ok(tensor_data)
    }
    
    /// 提取光流特征
    fn extract_flow_features(&self, flow_x: &[f32], flow_y: &[f32], width: usize, height: usize) 
        -> Result<Vec<f32>, VideoExtractionError> {
        debug!("提取光流特征，图像尺寸: {}x{}", width, height);
        
        // 提取不同的光流特征
        let mut features = Vec::with_capacity(self.feature_dim);
        
        // 根据配置选择不同的特征提取方法
        let extraction_method = self.config.get_string_param("flow_feature_method")
            .unwrap_or_else(|| "combined".to_string());
        
        match extraction_method.as_str() {
            "histogram" => {
                // 提取光流方向和幅度直方图
                let (directions, magnitudes) = self.compute_flow_direction_magnitude(flow_x, flow_y, width, height)?;
                
                // 计算方向直方图
                let direction_hist = self.compute_direction_histogram(&directions, &magnitudes, width, height)?;
                features.extend_from_slice(&direction_hist);
                
                // 计算幅度直方图
                let magnitude_hist = self.compute_magnitude_histogram(&magnitudes, width, height)?;
                features.extend_from_slice(&magnitude_hist);
            },
            "hof" => {
                // 提取光流直方图特征（Histogram of Optical Flow）
                features.extend_from_slice(&self.compute_hof(flow_x, flow_y, width, height)?);
            },
            "mbh" => {
                // 提取运动边界直方图（Motion Boundary Histogram）
                let mbh_x = self.compute_mbh(&flow_x, width, height)?;
                let mbh_y = self.compute_mbh(&flow_y, width, height)?;
                features.extend_from_slice(&mbh_x);
                features.extend_from_slice(&mbh_y);
            },
            "statistical" => {
                // 提取统计特征
                features.extend_from_slice(&self.compute_flow_statistics(flow_x, flow_y, width, height)?);
            },
            "combined" | _ => {
                // 结合多种特征
                // 1. 光流直方图特征
                let hof = self.compute_hof(flow_x, flow_y, width, height)?;
                
                // 2. 运动边界直方图
                let mbh_x = self.compute_mbh(&flow_x, width, height)?;
                let mbh_y = self.compute_mbh(&flow_y, width, height)?;
                
                // 3. 统计特征
                let stats = self.compute_flow_statistics(flow_x, flow_y, width, height)?;
                
                // 结合所有特征
                features.extend_from_slice(&hof);
                features.extend_from_slice(&mbh_x);
                features.extend_from_slice(&mbh_y);
                features.extend_from_slice(&stats);
            }
        }
        
        // 确保特征维度符合要求
        if features.len() > self.feature_dim {
            // 截断多余的特征
            features.truncate(self.feature_dim);
        } else if features.len() < self.feature_dim {
            // 填充不足的特征
            features.resize(self.feature_dim, 0.0);
        }
        
        // 应用特征归一化
        if let Some(norm_type) = &self.normalization {
            match norm_type {
                NormalizationType::L2 => {
                    // L2归一化
                    let mut norm = 0.0;
                    for &val in &features {
                        norm += val * val;
                    }
                    norm = norm.sqrt();
                    
                    if norm > 1e-10 {
                        for val in &mut features {
                            *val /= norm;
                        }
                    }
                },
                NormalizationType::L2 => {
                    // L1归一化
                    let mut sum = 0.0;
                    for &val in &features {
                        sum += val.abs();
                    }
                    
                    if sum > 1e-10 {
                        for val in &mut features {
                            *val /= sum;
                        }
                    }
                },
                NormalizationType::MinMax => {
                    // Min-Max归一化
                    if !features.is_empty() {
                        let min = *features.iter().min_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                            .unwrap_or(&0.0);
                        let max = *features.iter().max_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                            .unwrap_or(&1.0);
                        
                        if (max - min).abs() > 1e-10 {
                            for val in &mut features {
                                *val = (*val - min) / (max - min);
                            }
                        }
                    }
                },
            }
        }
        
        debug!("光流特征提取完成，特征维度: {}", features.len());
        Ok(features)
    }
    
    /// 计算光流方向和幅度
    fn compute_flow_direction_magnitude(
        &self, 
        flow_x: &[f32], 
        flow_y: &[f32], 
        width: usize, 
        height: usize
    ) -> Result<(Vec<f32>, Vec<f32>), VideoExtractionError> {
        let pixel_count = width * height;
        let mut directions = vec![0.0; pixel_count];
        let mut magnitudes = vec![0.0; pixel_count];
        
        for i in 0..pixel_count {
            let dx = flow_x[i];
            let dy = flow_y[i];
            
            magnitudes[i] = (dx * dx + dy * dy).sqrt();
            directions[i] = dy.atan2(dx); // 范围: [-pi, pi]
        }
        
        Ok((directions, magnitudes))
    }
    
    /// 计算方向直方图
    fn compute_direction_histogram(
        &self,
        directions: &[f32],
        magnitudes: &[f32],
        width: usize,
        height: usize
    ) -> Result<Vec<f32>, VideoExtractionError> {
        // 使用配置的直方图分箱数，默认为8
        let num_bins = self.flow_histogram_bins;
        let mut histogram = vec![0.0; num_bins];
        
        // 每个bin的范围
        let bin_size = 2.0 * std::f32::consts::PI / num_bins as f32;
        
        // 计算直方图
        for i in 0..directions.len() {
            // 将方向转换到[0, 2*pi]范围
            let angle = if directions[i] < 0.0 {
                directions[i] + 2.0 * std::f32::consts::PI
            } else {
                directions[i]
            };
            
            // 确定bin索引
            let bin_idx = ((angle / bin_size) as usize) % num_bins;
            
            // 使用幅度作为权重
            histogram[bin_idx] += magnitudes[i];
        }
        
        // 归一化直方图
        let sum: f32 = histogram.iter().sum();
        if sum > 0.0 {
            for val in &mut histogram {
                *val /= sum;
            }
        }
        
        Ok(histogram)
    }
    
    /// 计算幅度直方图
    fn compute_magnitude_histogram(
        &self,
        magnitudes: &[f32],
        width: usize,
        height: usize
    ) -> Result<Vec<f32>, VideoExtractionError> {
        // 使用配置的直方图分箱数，默认为8
        let num_bins = self.flow_histogram_bins;
        let mut histogram = vec![0.0; num_bins];
        
        // 确定幅度范围
        let max_magnitude = magnitudes.iter()
            .fold(0.0f32, |max, &val| f32::max(max, val));
        
        if max_magnitude <= 0.0 {
            return Ok(histogram);
        }
        
        // 每个bin的大小
        let bin_size = max_magnitude / num_bins as f32;
        
        // 计算直方图
        for &magnitude in magnitudes {
            if magnitude > 0.0 {
                // 确定bin索引
                let bin_idx = ((magnitude / bin_size) as usize).min(num_bins - 1);
                histogram[bin_idx] += 1.0;
            }
        }
        
        // 归一化直方图
        let sum: f32 = histogram.iter().sum();
        if sum > 0.0 {
            for val in &mut histogram {
                *val /= sum;
            }
        }
        
        Ok(histogram)
    }
    
    /// 计算光流直方图特征（Histogram of Optical Flow）
    fn compute_hof(
        &self,
        flow_x: &[f32],
        flow_y: &[f32],
        width: usize,
        height: usize
    ) -> Result<Vec<f32>, VideoExtractionError> {
        // 划分图像为网格区域，默认为4x4网格
        let grid_x = 4;
        let grid_y = 4;
        let num_bins = self.flow_histogram_bins;
        let mut features = Vec::with_capacity(grid_x * grid_y * num_bins);
        
        // 计算每个网格的宽高
        let cell_width = width / grid_x;
        let cell_height = height / grid_y;
        
        // 处理每个网格
        for gy in 0..grid_y {
            for gx in 0..grid_x {
                // 当前网格的边界
                let start_x = gx * cell_width;
                let start_y = gy * cell_height;
                let end_x = ((gx + 1) * cell_width).min(width);
                let end_y = ((gy + 1) * cell_height).min(height);
                
                // 收集当前网格内的光流
                let mut cell_flow_x = Vec::new();
                let mut cell_flow_y = Vec::new();
                
                for y in start_y..end_y {
                    for x in start_x..end_x {
                        let idx = y * width + x;
                        if idx < flow_x.len() && idx < flow_y.len() {
                            cell_flow_x.push(flow_x[idx]);
                            cell_flow_y.push(flow_y[idx]);
                        }
                    }
                }
                
                // 计算当前网格的光流直方图
                let (directions, magnitudes) = self.compute_flow_direction_magnitude(
                    &cell_flow_x, &cell_flow_y, cell_flow_x.len(), 1
                )?;
                let hist = self.compute_direction_histogram(&directions, &magnitudes, cell_flow_x.len(), 1)?;
                
                // 添加到特征向量
                features.extend_from_slice(&hist);
            }
        }
        
        Ok(features)
    }
    
    /// 计算运动边界直方图（Motion Boundary Histogram）
    fn compute_mbh(
        &self,
        flow: &[f32],
        width: usize,
        height: usize
    ) -> Result<Vec<f32>, VideoExtractionError> {
        // 计算流场的梯度
        let mut gradient_x = vec![0.0f32; width * height];
        let mut gradient_y = vec![0.0f32; width * height];
        
        // 使用简单的有限差分计算梯度
        for y in 1..height-1 {
            for x in 1..width-1 {
                let idx = y * width + x;
                
                // X方向梯度
                gradient_x[idx] = (flow[idx+1] - flow[idx-1]) / 2.0;
                
                // Y方向梯度
                gradient_y[idx] = (flow[idx+width] - flow[idx-width]) / 2.0;
            }
        }
        
        // 划分图像为网格区域，默认为4x4网格
        let grid_x = 4;
        let grid_y = 4;
        let num_bins = self.flow_histogram_bins;
        let mut features = Vec::with_capacity(grid_x * grid_y * num_bins);
        
        // 计算每个网格的宽高
        let cell_width = width / grid_x;
        let cell_height = height / grid_y;
        
        // 处理每个网格
        for gy in 0..grid_y {
            for gx in 0..grid_x {
                // 当前网格的边界
                let start_x = gx * cell_width;
                let start_y = gy * cell_height;
                let end_x = ((gx + 1) * cell_width).min(width);
                let end_y = ((gy + 1) * cell_height).min(height);
                
                // 收集当前网格内的梯度
                let mut cell_grad_x = Vec::new();
                let mut cell_grad_y = Vec::new();
                
                for y in start_y..end_y {
                    for x in start_x..end_x {
                        let idx = y * width + x;
                        if idx < gradient_x.len() && idx < gradient_y.len() {
                            cell_grad_x.push(gradient_x[idx]);
                            cell_grad_y.push(gradient_y[idx]);
                        }
                    }
                }
                
                // 计算当前网格的梯度方向和幅度
                let (directions, magnitudes) = self.compute_flow_direction_magnitude(
                    &cell_grad_x, &cell_grad_y, cell_grad_x.len(), 1
                )?;
                
                // 计算方向直方图
                let hist = self.compute_direction_histogram(&directions, &magnitudes, cell_grad_x.len(), 1)?;
                
                // 添加到特征向量
                features.extend_from_slice(&hist);
            }
        }
        
        Ok(features)
    }
    
    /// 计算光流统计特征
    fn compute_flow_statistics(
        &self,
        flow_x: &[f32],
        flow_y: &[f32],
        width: usize,
        height: usize
    ) -> Result<Vec<f32>, VideoExtractionError> {
        let mut features = Vec::with_capacity(10);
        
        // 计算光流的方向和幅度
        let (directions, magnitudes) = self.compute_flow_direction_magnitude(flow_x, flow_y, width, height)?;
        
        // 计算X和Y方向流的统计量
        features.push(flow_x.iter().sum::<f32>() / flow_x.len() as f32); // 平均X流
        features.push(flow_y.iter().sum::<f32>() / flow_y.len() as f32); // 平均Y流
        
        // 计算X和Y方向流的标准差
        let mean_x = features[0];
        let mean_y = features[1];
        let var_x = flow_x.iter().map(|&v| (v - mean_x).powi(2)).sum::<f32>() / flow_x.len() as f32;
        let var_y = flow_y.iter().map(|&v| (v - mean_y).powi(2)).sum::<f32>() / flow_y.len() as f32;
        features.push(var_x.sqrt()); // X流标准差
        features.push(var_y.sqrt()); // Y流标准差
        
        // 计算流幅度的统计量
        let mean_mag = magnitudes.iter().sum::<f32>() / magnitudes.len() as f32;
        features.push(mean_mag); // 平均幅度
        
        let var_mag = magnitudes.iter().map(|&v| (v - mean_mag).powi(2)).sum::<f32>() / magnitudes.len() as f32;
        features.push(var_mag.sqrt()); // 幅度标准差
        
        // 计算最大和最小幅度
        let max_mag = magnitudes.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        let min_mag = magnitudes.iter().fold(f32::INFINITY, |a, &b| a.min(b));
        features.push(max_mag); // 最大幅度
        features.push(min_mag); // 最小幅度
        
        // 计算方向统计量
        let mean_dir = directions.iter().sum::<f32>() / directions.len() as f32;
        features.push(mean_dir); // 平均方向
        
        // 方向的周期性需要特殊处理
        let sin_sum = directions.iter().map(|&d| d.sin()).sum::<f32>();
        let cos_sum = directions.iter().map(|&d| d.cos()).sum::<f32>();
        let direction_consistency = (sin_sum.powi(2) + cos_sum.powi(2)).sqrt() / directions.len() as f32;
        features.push(direction_consistency); // 方向一致性
        
        Ok(features)
    }
    
    /// 转换帧为灰度图
    fn to_grayscale(frame: &VideoFrame) -> Result<Vec<u8>, VideoExtractionError> {
        let width = frame.width;
        let height = frame.height;
        let channels = frame.channels;
        
        if frame.data.is_empty() {
            return Err(VideoExtractionError::ProcessingError(
                "帧数据为空，无法转换为灰度图".to_string()
            ));
        }
        
        // 分配灰度图内存
        let mut gray = vec![0u8; width * height];
        
        match channels {
            1 => {
                // 已经是灰度图
                gray.copy_from_slice(&frame.data);
            },
            3 => {
                // RGB转灰度
                for y in 0..height {
                    for x in 0..width {
                        let src_idx = (y * width + x) * 3;
                        let dst_idx = y * width + x;
                        
                        if src_idx + 2 < frame.data.len() && dst_idx < gray.len() {
                            // 使用加权公式: Y = 0.299*R + 0.587*G + 0.114*B
                            let r = frame.data[src_idx] as f32;
                            let g = frame.data[src_idx + 1] as f32;
                            let b = frame.data[src_idx + 2] as f32;
                            
                            let y = (0.299 * r + 0.587 * g + 0.114 * b).round() as u8;
                            gray[dst_idx] = y;
                        }
                    }
                }
            },
            4 => {
                // RGBA转灰度
                for y in 0..height {
                    for x in 0..width {
                        let src_idx = (y * width + x) * 4;
                        let dst_idx = y * width + x;
                        
                        if src_idx + 2 < frame.data.len() && dst_idx < gray.len() {
                            // 使用加权公式: Y = 0.299*R + 0.587*G + 0.114*B (忽略Alpha通道)
                            let r = frame.data[src_idx] as f32;
                            let g = frame.data[src_idx + 1] as f32;
                            let b = frame.data[src_idx + 2] as f32;
                            
                            let y = (0.299 * r + 0.587 * g + 0.114 * b).round() as u8;
                            gray[dst_idx] = y;
                        }
                    }
                }
            },
            _ => {
                return Err(VideoExtractionError::ProcessingError(
                    format!("不支持的通道数: {}", channels)
                ));
            }
        }
        
        Ok(gray)
    }
    
    /// 在两帧之间查找最佳匹配位置
    /// 
    /// 用于简化的光流计算，在当前位置周围搜索最佳匹配
    fn find_best_match(
        &self,
        prev_gray: &[u8],
        curr_gray: &[u8],
        center_x: usize,
        center_y: usize,
        width: usize,
        height: usize,
        window_size: usize
    ) -> (f32, f32) {
        let half_window = window_size / 2;
        let mut best_dx = 0.0f32;
        let mut best_dy = 0.0f32;
        let mut min_ssd = f32::MAX; // 平方差之和
        
        // 搜索范围
        let search_range = 7;
        
        // 在搜索范围内寻找最佳匹配
        for dy in -search_range..=search_range {
            for dx in -search_range..=search_range {
                // 新中心位置
                let new_center_x = (center_x as i32 + dx).clamp(half_window as i32, (width - half_window) as i32) as usize;
                let new_center_y = (center_y as i32 + dy).clamp(half_window as i32, (height - half_window) as i32) as usize;
                
                // 计算窗口内的误差平方和
                let mut ssd = 0.0f32;
                let half_window_i32 = half_window as i32;
                
                for wy in -half_window_i32..=half_window_i32 {
                    for wx in -half_window_i32..=half_window_i32 {
                        let prev_x = (center_x as i32 + wx) as usize;
                        let prev_y = (center_y as i32 + wy) as usize;
                        let curr_x = (new_center_x as i32 + wx) as usize;
                        let curr_y = (new_center_y as i32 + wy) as usize;
                        
                        if prev_x < width && prev_y < height && curr_x < width && curr_y < height {
                            let prev_idx = prev_y * width + prev_x;
                            let curr_idx = curr_y * width + curr_x;
                            
                            if prev_idx < prev_gray.len() && curr_idx < curr_gray.len() {
                                let diff = (prev_gray[prev_idx] as i32 - curr_gray[curr_idx] as i32) as f32;
                                ssd += diff * diff;
                            }
                        }
                    }
                }
                
                // 使用高斯加权，使中心的匹配更重要
                let distance = (dx * dx + dy * dy) as f32;
                let weight = (-distance / (2.0 * 3.0 * 3.0)).exp();
                ssd *= weight;
                
                // 更新最佳匹配
                if ssd < min_ssd {
                    min_ssd = ssd;
                    best_dx = dx as f32;
                    best_dy = dy as f32;
                }
            }
        }
        
        (best_dx, best_dy)
    }
}

impl FeatureExtractor for OpticalFlowExtractor {
    fn initialize(&mut self) -> Result<(), VideoExtractionError> {
        if self.initialized {
            debug!("光流特征提取器已初始化，跳过");
            return Ok(());
        }
        
        info!("初始化光流特征提取器: {}, 堆叠帧数: {}, 特征维度: {}", 
            self.flow_algorithm, self.stacked_frames, self.feature_dim);
        
        // 清空帧缓冲
        self.frame_buffer.clear();
        
        self.initialized = true;
        info!("光流特征提取器初始化完成");
        
        Ok(())
    }
    
    fn process_frame(&mut self, frame: &VideoFrame) -> Result<Vec<f32>, VideoExtractionError> {
        if !self.initialized {
            return Err(VideoExtractionError::ModelError(
                "光流特征提取器尚未初始化".to_string()
            ));
        }
        
        // 添加当前帧到缓冲区
        self.frame_buffer.push_back(frame.clone());
        
        // 如果缓冲区中的帧数不足，返回零向量
        if self.frame_buffer.len() < 2 {
            debug!("缓冲区帧数不足，无法计算光流。当前帧数: {}", self.frame_buffer.len());
            return Ok(vec![0.0; self.feature_dim]);
        }
        
        // 保持缓冲区大小
        while self.frame_buffer.len() > self.stacked_frames + 1 {
            self.frame_buffer.pop_front();
        }
        
        // 获取当前帧和前一帧
        let prev_frame = self.frame_buffer[self.frame_buffer.len() - 2].clone();
        let curr_frame = self.frame_buffer[self.frame_buffer.len() - 1].clone();
        
        // 计算光流
        let (flow_x, flow_y) = self.compute_optical_flow(&prev_frame, &curr_frame)?;
        
        // 提取特征
        let features = self.extract_flow_features(&flow_x, &flow_y, curr_frame.width, curr_frame.height)?;
        
        debug!("成功计算帧间光流，输出特征维度: {}", features.len());
        
        Ok(features)
    }
    
    fn get_feature_dim(&self) -> usize {
        self.feature_dim
    }
    
    fn get_feature_type(&self) -> VideoFeatureType {
        self.feature_type
    }
    
    fn temporal_pooling(&self, features: &[Vec<f32>], pooling_type: &TemporalPoolingType) -> Result<Vec<f32>, VideoExtractionError> {
        if features.is_empty() {
            return Err(VideoExtractionError::ProcessingError(
                "没有特征可以池化".to_string()
            ));
        }
        
        // 使用共享的池化函数
        match pooling_type {
            TemporalPoolingType::Mean => Ok(super::pooling::temporal_mean_pooling(features)),
            TemporalPoolingType::Max => Ok(super::pooling::temporal_max_pooling(features)),
            TemporalPoolingType::Attention => {
                // 简化版的注意力池化
                let mut attention_weights = vec![0.0; features.len()];
                let mut max_weight: f32 = 0.0;
                
                // 计算每个特征向量的范数作为注意力权重
                for (i, feature) in features.iter().enumerate() {
                    let norm: f32 = feature.iter().map(|x| x * x).sum::<f32>().sqrt();
                    attention_weights[i] = norm;
                    max_weight = max_weight.max(norm);
                }
                
                // 归一化权重
                if max_weight > 0.0 {
                    for weight in &mut attention_weights {
                        *weight /= max_weight;
                    }
                } else {
                    // 如果所有权重都为0，使用均匀权重
                    for weight in &mut attention_weights {
                        *weight = 1.0 / attention_weights.len() as f32;
                    }
                }
                
                // 应用注意力池化
                let dim = features[0].len();
                let mut pooled = vec![0.0; dim];
                
                for (i, feature) in features.iter().enumerate() {
                    if i < attention_weights.len() {
                        let weight = attention_weights[i];
                        for j in 0..dim {
                            if j < feature.len() {
                                pooled[j] += feature[j] * weight;
                            }
                        }
                    }
                }
                
                Ok(pooled)
            },
            TemporalPoolingType::Custom(_) => {
                Err(VideoExtractionError::NotImplementedError(
                    "自定义时间池化尚未实现".to_string()
                ))
            },
        }
    }
    
    fn release(&mut self) -> Result<(), VideoExtractionError> {
        if self.initialized {
            info!("释放光流特征提取器资源");
            // 清除缓冲区
            self.frame_buffer.clear();
            self.initialized = false;
        }
        
        Ok(())
    }
    
    fn process_batch(&mut self, frames: &[VideoFrame]) -> Result<Vec<Vec<f32>>, VideoExtractionError> {
        if !self.initialized {
            return Err(VideoExtractionError::ModelError(
                "光流特征提取器尚未初始化".to_string()
            ));
        }
        
        if frames.len() < 2 {
            return Err(VideoExtractionError::InputError(
                format!("光流特征提取需要至少2帧，当前: {}", frames.len())
            ));
        }
        
        debug!("光流特征提取器批量处理{}个帧", frames.len());
        
        let mut all_features = Vec::with_capacity(frames.len());
        
        // 首帧返回零向量
        all_features.push(vec![0.0; self.feature_dim]);
        
        // 计算每对相邻帧之间的光流
        for i in 1..frames.len() {
            let prev_frame = &frames[i-1];
            let curr_frame = &frames[i];
            
            // 计算光流
            let (flow_x, flow_y) = self.compute_optical_flow(prev_frame, curr_frame)?;
            
            // 提取特征
            let features = self.extract_flow_features(&flow_x, &flow_y, curr_frame.width, curr_frame.height)?;
            
            all_features.push(features);
        }
        
        Ok(all_features)
    }
} 