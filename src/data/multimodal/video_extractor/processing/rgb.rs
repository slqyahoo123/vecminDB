//! RGB特征提取器实现
//! 
//! 本模块提供RGB特征提取的核心算法实现

use super::FeatureExtractor;
use crate::data::multimodal::video_extractor::types::*;
use crate::data::multimodal::video_extractor::config::VideoFeatureConfig;
use crate::data::multimodal::video_extractor::error::VideoExtractionError;
use std::collections::HashMap;
use log::{info, debug, warn};

/// RGB特征提取器，负责从视频帧中提取RGB特征
pub struct RGBFeatureExtractor {
    /// 特征维度
    feature_dim: usize,
    /// 模型类型
    model_type: String,
    /// 是否使用预训练模型
    use_pretrained: bool,
    /// 特征类型
    feature_type: VideoFeatureType,
    /// 归一化类型
    normalization: Option<NormalizationType>,
    /// 空间池化类型
    spatial_pooling: Option<SpatialPoolingType>,
    /// 是否已初始化
    initialized: bool,
    /// 已加载的模型权重（可选，用于自定义或预训练模型）
    model_weights: Option<HashMap<String, Vec<f32>>>,
    /// 配置项
    config: VideoFeatureConfig,
}

impl RGBFeatureExtractor {
    /// 创建新的RGB特征提取器
    pub fn new(config: &VideoFeatureConfig) -> Self {
        let feature_dim = config.get_usize_param("feature_dimension").unwrap_or(512);
        let model_type = format!("{:?}", config.model_type);
        let use_pretrained = config.use_pretrained;
        
        Self {
            feature_dim,
            model_type,
            use_pretrained,
            feature_type: VideoFeatureType::RGB,
            normalization: config.normalization,
            spatial_pooling: Some(config.spatial_pooling),
            initialized: false,
            model_weights: None,
            config: config.clone(),
        }
    }
    
    /// 加载模型权重
    fn load_model_weights(&mut self) -> Result<(), VideoExtractionError> {
        debug!("加载RGB特征提取器模型权重: {}", self.model_type);
        
        // 在实际应用中，需要根据模型类型加载不同的预训练模型
        let model_dir = std::env::var("MODEL_DIR").unwrap_or_else(|_| "./models".to_string());
        let model_path = match self.model_type.as_str() {
            "resnet" => format!("{}/resnet50_rgb.onnx", model_dir),
            "vgg" => format!("{}/vgg16_rgb.onnx", model_dir),
            "mobilenet" => format!("{}/mobilenet_v2_rgb.onnx", model_dir),
            custom if custom.starts_with("custom_") => {
                // 自定义模型处理
                if let Some(custom_path) = self.config.custom_params.get("custom_model_path") {
                    custom_path.clone()
                } else {
                    return Err(VideoExtractionError::ModelError(
                        format!("自定义模型类型需要指定custom_model_path参数: {}", self.model_type)
                    ));
                }
            },
            _ => {
                return Err(VideoExtractionError::ModelError(
                    format!("不支持的RGB模型类型: {}", self.model_type)
                ));
            }
        };
        
        // 检查模型文件是否存在
        if !std::path::Path::new(&model_path).exists() {
            // 尝试下载模型（实际应用中可能需要更复杂的下载逻辑）
            if self.use_pretrained {
                debug!("模型文件不存在，尝试下载预训练模型: {}", model_path);
                self.download_pretrained_model(&model_path)?;
            } else {
                return Err(VideoExtractionError::ModelError(
                    format!("模型文件不存在: {}", model_path)
                ));
            }
        }
        
        // 实际加载模型权重
        // 在生产环境中通常会使用ONNX Runtime、TensorRT或其他优化过的推理引擎
        debug!("从文件加载模型权重: {}", model_path);
        
        // 使用ONNX Runtime或类似库加载模型（伪代码，实际实现需要添加具体依赖）
        // 注意：onnx 特性未在 Cargo.toml 中定义，已注释掉相关代码
        /* #[cfg(feature = "onnx")]
        {
            use onnxruntime::{environment::Environment, session::Session, GraphOptimizationLevel};
            
            // 创建ONNX环境
            let environment = Environment::builder()
                .with_name("rgb_feature_extractor")
                .build()?;
            
            // 设置ONNX会话选项
            let session_options = onnxruntime::SessionOptions::new()?;
            session_options.set_graph_optimization_level(GraphOptimizationLevel::All)?;
            
            // 启用GPU加速（如果可用）
            if self.config.use_gpu.unwrap_or(false) {
                session_options.enable_cuda()?;
            }
            
            // 加载模型
            let session = Session::new(&environment, &model_path, &session_options)?;
            
            // 存储会话和相关信息
            let mut weights = HashMap::new();
            weights.insert("onnx_session".to_string(), Box::new(session));
            
            self.model_weights = Some(weights);
            debug!("ONNX模型加载完成: {}", model_path);
        } */
        
        // 如果没有启用ONNX特性，使用简化的权重模拟
        // #[cfg(not(feature = "onnx"))]
        {
            let mut weights = HashMap::new();
            
            match self.model_type.as_str() {
                "resnet" => {
                    // 实际应用中应加载真实权重，这里使用模拟数据
                    weights.insert("conv1.weight".to_string(), vec![0.1; 64 * 3 * 7 * 7]);
                    weights.insert("fc.weight".to_string(), vec![0.01; self.feature_dim * 2048]);
                    // 添加其他必要的层权重...
                    warn!("使用模拟的ResNet权重，建议启用'onnx'特性以使用真实模型");
                },
                "vgg" => {
                    weights.insert("features.0.weight".to_string(), vec![0.1; 64 * 3 * 3 * 3]);
                    weights.insert("classifier.6.weight".to_string(), vec![0.01; self.feature_dim * 4096]);
                    warn!("使用模拟的VGG权重，建议启用'onnx'特性以使用真实模型");
                },
                "mobilenet" => {
                    weights.insert("conv1.weight".to_string(), vec![0.1; 32 * 3 * 3 * 3]);
                    weights.insert("fc.weight".to_string(), vec![0.01; self.feature_dim * 1280]);
                    warn!("使用模拟的MobileNet权重，建议启用'onnx'特性以使用真实模型");
                },
                _ => {
                    return Err(VideoExtractionError::ModelError(
                        format!("不支持的RGB模型类型: {}", self.model_type)
                    ));
                }
            }
            
            self.model_weights = Some(weights);
            debug!("模拟模型权重加载完成（非生产级实现）");
        }
        
        Ok(())
    }
    
    /// 下载预训练模型（在生产环境中，这应该连接到模型仓库或CDN）
    fn download_pretrained_model(&self, model_path: &str) -> Result<(), VideoExtractionError> {
        // 创建模型目录
        if let Some(parent) = std::path::Path::new(model_path).parent() {
            std::fs::create_dir_all(parent).map_err(|e| {
                VideoExtractionError::ModelError(
                    format!("无法创建模型目录: {}", e)
                )
            })?;
        }
        
        let model_type = self.model_type.as_str();
        let model_url = match model_type {
            "resnet" => "https://github.com/onnx/models/raw/main/vision/classification/resnet/model/resnet50-v1-7.onnx",
            "vgg" => "https://github.com/onnx/models/raw/main/vision/classification/vgg/model/vgg16-7.onnx",
            "mobilenet" => "https://github.com/onnx/models/raw/main/vision/classification/mobilenet/model/mobilenetv2-7.onnx",
            _ => {
                return Err(VideoExtractionError::ModelError(
                    format!("不支持自动下载此模型类型: {}", model_type)
                ));
            }
        };
        
        info!("正在下载预训练模型: {} -> {}", model_url, model_path);
        
        // 在实际应用中，这里应该有一个健壮的下载实现，包括：
        // - 断点续传
        // - 哈希验证
        // - 进度报告
        // - 超时和重试逻辑
        // 为简洁起见，这里使用一个基本实现
        
        #[cfg(feature = "reqwest")]
        {
            use std::io::copy;
            use std::fs::File;
            
            // 创建HTTP客户端
            let client = reqwest::blocking::Client::builder()
                .timeout(std::time::Duration::from_secs(600))
                .build()
                .map_err(|e| {
                    VideoExtractionError::ModelError(
                        format!("创建HTTP客户端失败: {}", e)
                    )
                })?;
            
            // 下载模型
            let mut response = client.get(model_url)
                .send()
                .map_err(|e| {
                    VideoExtractionError::ModelError(
                        format!("下载模型失败: {}", e)
                    )
                })?;
            
            // 检查响应状态
            if !response.status().is_success() {
                return Err(VideoExtractionError::InitializationError(
                    format!("下载模型失败，HTTP状态: {}", response.status())
                ));
            }
            
            // 保存文件
            let mut file = File::create(model_path).map_err(|e| {
                VideoExtractionError::InitializationError(
                    format!("创建模型文件失败: {}", e)
                )
            })?;
            
            copy(&mut response, &mut file).map_err(|e| {
                VideoExtractionError::InitializationError(
                    format!("保存模型文件失败: {}", e)
                )
            })?;
            
            debug!("预训练模型下载完成: {}", model_path);
        }
        
        #[cfg(not(feature = "reqwest"))]
        {
            return Err(VideoExtractionError::FeatureNotEnabled(
                "自动下载模型功能需要启用'reqwest'特性".to_string()
            ));
        }
        
        #[cfg(feature = "reqwest")]
        {
            Ok(())
        }
    }
    
    /// 提取RGB特征，使用深度学习模型进行特征提取
    fn extract_rgb_features(&self, processed_frame: &[f32]) -> Result<Vec<f32>, VideoExtractionError> {
        // 确保模型已加载
        if self.model_weights.is_none() {
            return Err(VideoExtractionError::ProcessingError(
                "模型权重尚未加载".to_string()
            ));
        }
        
        // 在实际生产环境中，这里应该使用加载的深度学习模型进行特征提取
        // 我们将实现基于配置的多种特征提取方式
        
        // 注意：onnx 和 opencv 特性未在 Cargo.toml 中定义，已注释掉相关代码
        /* #[cfg(feature = "onnx")]
        {
            // 使用ONNX Runtime进行推理
            let weights = self.model_weights.as_ref().unwrap();
            
            if let Some(session) = weights.get("onnx_session").and_then(|s| s.downcast_ref::<onnxruntime::session::Session>()) {
                return self.extract_features_onnx(session, processed_frame);
            }
        }
        
        // 如果没有启用ONNX或无法使用ONNX会话，回退到基于OpenCV的特征提取
        #[cfg(feature = "opencv")]
        {
            return self.extract_features_opencv(processed_frame);
        } */
        
        // 最后回退到简化的CPU实现
        self.extract_features_cpu(processed_frame)
    }
    
    /// 使用ONNX Runtime提取特征（高性能实现）
    // 注意：onnx 特性未在 Cargo.toml 中定义，已注释掉相关代码
    /* #[cfg(feature = "onnx")]
    fn extract_features_onnx(
        &self, 
        session: &onnxruntime::session::Session, 
        processed_frame: &[f32]
    ) -> Result<Vec<f32>, VideoExtractionError> {
        use ndarray::{Array, ArrayD, IxDyn};
        use onnxruntime::{tensor::OrtOwnedTensor, value::Value};
        
        debug!("使用ONNX Runtime提取RGB特征");
        
        // 1. 准备输入数据
        // 假设processed_frame是[0,1]范围内的浮点数，需要进行预处理
        
        // 对于ImageNet预训练模型，标准化参数
        let mean = [0.485, 0.456, 0.406]; // RGB均值
        let std = [0.229, 0.224, 0.225];  // RGB标准差
        
        let mut normalized_data = Vec::with_capacity(processed_frame.len());
        let channels = 3; // RGB
        let pixel_count = processed_frame.len() / channels;
        
        // 执行图像预处理：标准化
        for c in 0..channels {
            for i in 0..pixel_count {
                let pixel_value = processed_frame[i * channels + c];
                // 标准化处理: (pixel - mean) / std
                let norm_value = (pixel_value - mean[c]) / std[c];
                normalized_data.push(norm_value);
            }
        }
        
        // 2. 构建输入tensor
        // 假设输入是批量大小为1，3通道，高度和宽度由处理后的帧决定
        let height = (pixel_count as f64).sqrt().round() as usize;
        let width = pixel_count / height;
        
        // 创建适合ONNX的输入张量
        // NCHW格式: [batch_size, channels, height, width]
        let input_dimensions = vec![1, channels, height, width];
        let input_array = Array::from_shape_vec(
            IxDyn(&input_dimensions), 
            normalized_data
        ).map_err(|e| {
            VideoExtractionError::ProcessingError(
                format!("创建输入张量失败: {}", e)
            )
        })?;
        
        // 创建输入值映射
        let input_name = session.inputs[0].name.clone();
        let inputs = vec![Value::from_array(session.allocator(), &input_array).map_err(|e| {
            VideoExtractionError::ProcessingError(
                format!("转换输入张量失败: {}", e)
            )
        })?];
        
        // 3. 运行模型推理
        let outputs: Vec<OrtOwnedTensor<f32, _>> = session.run(inputs).map_err(|e| {
            VideoExtractionError::ProcessingError(
                format!("模型推理失败: {}", e)
            )
        })?;
        
        // 4. 处理输出
        // 假设输出是特征向量，通常是最后一个全连接层的输出
        // 将输出张量转换为Vec<f32>
        let features: Vec<f32> = outputs[0].view().iter().copied().collect();
        
        // 5. 如果需要，调整特征维度
        let mut final_features = features;
        if final_features.len() != self.feature_dim {
            debug!("调整特征维度: {} -> {}", final_features.len(), self.feature_dim);
            final_features = self.resize_feature_vector(&final_features, self.feature_dim);
        }
        
        // 6. 应用选择的归一化方法
        let normalized_features = self.apply_normalization(&final_features)?;
        
        debug!("ONNX RGB特征提取成功，维度: {}", normalized_features.len());
        Ok(normalized_features)
    } */
    
    /// 使用OpenCV提取RGB特征（中等性能实现）
    // 注意：opencv 特性未在 Cargo.toml 中定义，已注释掉相关代码
    /* #[cfg(feature = "opencv")]
    fn extract_features_opencv(&self, processed_frame: &[f32]) -> Result<Vec<f32>, VideoExtractionError> {
        use opencv::{
            core::{Mat, MatTraitConst, CV_32F, Size, Scalar, Vector, NORM_L2},
            imgproc::{calcHist, color_convert, COLOR_RGB2HSV, COLOR_RGB2GRAY},
            prelude::*,
        };
        
        debug!("使用OpenCV提取RGB特征");
        
        if processed_frame.len() < 12 {
            return Err(VideoExtractionError::ProcessingError(
                format!("帧数据太小，无法提取特征: {}", processed_frame.len())
            ));
        }
        
        // 1. 将处理后的帧数据转换为OpenCV Mat
        let height = (processed_frame.len() / 3 as f64).sqrt().round() as i32;
        let width = (processed_frame.len() / 3) as i32 / height;
        
        // 创建Mat并复制数据
        let mut frame_mat = unsafe {
            Mat::new_rows_cols(height, width, CV_32F).unwrap()
        };
        
        // 将数据复制到Mat中（假设processed_frame是RGB交错格式）
        unsafe {
            std::ptr::copy_nonoverlapping(
                processed_frame.as_ptr(), 
                frame_mat.data_mut() as *mut f32, 
                processed_frame.len()
            );
        }
        
        // 2. 提取多种特征并合并
        let mut all_features = Vec::new();
        
        // 2.1 提取颜色直方图特征
        let color_histograms = self.extract_color_histograms(&frame_mat)?;
        all_features.extend_from_slice(&color_histograms);
        
        // 2.2 提取纹理特征
        let texture_features = self.extract_texture_features(&frame_mat)?;
        all_features.extend_from_slice(&texture_features);
        
        // 2.3 提取边缘特征
        let edge_features = self.extract_edge_features(&frame_mat)?;
        all_features.extend_from_slice(&edge_features);
        
        // 3. 调整最终特征维度
        let mut final_features = all_features;
        if final_features.len() != self.feature_dim {
            debug!("调整特征维度: {} -> {}", final_features.len(), self.feature_dim);
            final_features = self.resize_feature_vector(&final_features, self.feature_dim);
        }
        
        // 4. 应用归一化方法
        let normalized_features = self.apply_normalization(&final_features)?;
        
        debug!("OpenCV RGB特征提取成功，维度: {}", normalized_features.len());
        Ok(normalized_features)
    } */
    
    /// 在CPU上提取RGB特征（基本实现）
    fn extract_features_cpu(&self, processed_frame: &[f32]) -> Result<Vec<f32>, VideoExtractionError> {
        debug!("使用CPU提取RGB特征");
        
        if processed_frame.len() < 12 {
            return Err(VideoExtractionError::ProcessingError(
                format!("帧数据太小，无法提取特征: {}", processed_frame.len())
            ));
        }
        
        // 1. 计算颜色直方图特征
        let mut color_histograms = Vec::with_capacity(768); // 3通道 * 256bins
        
        // 简化的特征提取: 对每个通道计算直方图
        let pixel_count = processed_frame.len() / 3;
        
        // 对于每个通道 (R, G, B)
        for c in 0..3 {
            let mut histogram = vec![0f32; 256];
            
            // 为每个通道计算直方图
            for i in 0..pixel_count {
                let idx = i * 3 + c;
                if idx < processed_frame.len() {
                    let pixel_value = (processed_frame[idx] * 255.0).clamp(0.0, 255.0) as usize;
                    if pixel_value < 256 {
                        histogram[pixel_value] += 1.0;
                    }
                }
            }
            
            // 归一化直方图
            let sum: f32 = histogram.iter().sum();
            if sum > 0.0 {
                for bin in &mut histogram {
                    *bin /= sum;
                }
            }
            
            color_histograms.extend_from_slice(&histogram);
        }
        
        // 2. 计算简化的纹理特征（梯度统计）
        let height = (pixel_count as f32).sqrt() as usize;
        let width = if height > 0 { pixel_count / height } else { 0 };
        
        let mut texture_features = Vec::new();
        if width > 2 && height > 2 {
            // 计算简化的水平和垂直梯度直方图
            let hist_bins = 32;
            let mut gradient_hist_h = vec![0f32; hist_bins];
            let mut gradient_hist_v = vec![0f32; hist_bins];
            
            // 计算梯度
            for y in 1..height-1 {
                for x in 1..width-1 {
                    // 计算中心像素在RGB平面的梯度
                    let center_idx = (y * width + x) * 3;
                    let right_idx = (y * width + (x+1)) * 3;
                    let bottom_idx = ((y+1) * width + x) * 3;
                    
                    if center_idx+2 < processed_frame.len() && 
                       right_idx+2 < processed_frame.len() && 
                       bottom_idx+2 < processed_frame.len() {
                        
                        // 计算水平梯度（右 - 中）
                        let grad_h = (0..3).map(|c| {
                            processed_frame[right_idx+c] - processed_frame[center_idx+c]
                        }).sum::<f32>() / 3.0;
                        
                        // 计算垂直梯度（下 - 中）
                        let grad_v = (0..3).map(|c| {
                            processed_frame[bottom_idx+c] - processed_frame[center_idx+c]
                        }).sum::<f32>() / 3.0;
                        
                        // 映射梯度到直方图bin
                        let bin_h = ((grad_h + 1.0) * 0.5 * (hist_bins as f32 - 1.0)) as usize;
                        let bin_v = ((grad_v + 1.0) * 0.5 * (hist_bins as f32 - 1.0)) as usize;
                        
                        let bin_h = bin_h.min(hist_bins - 1);
                        let bin_v = bin_v.min(hist_bins - 1);
                        
                        gradient_hist_h[bin_h] += 1.0;
                        gradient_hist_v[bin_v] += 1.0;
                    }
                }
            }
            
            // 归一化梯度直方图
            let sum_h: f32 = gradient_hist_h.iter().sum();
            let sum_v: f32 = gradient_hist_v.iter().sum();
            
            if sum_h > 0.0 {
                for bin in &mut gradient_hist_h {
                    *bin /= sum_h;
                }
            }
            
            if sum_v > 0.0 {
                for bin in &mut gradient_hist_v {
                    *bin /= sum_v;
                }
            }
            
            texture_features.extend_from_slice(&gradient_hist_h);
            texture_features.extend_from_slice(&gradient_hist_v);
            
            // 添加一些统计特征
            // 1. 颜色协方差矩阵（3x3=9个值）
            let mut color_means = [0f32; 3];
            for c in 0..3 {
                for i in 0..pixel_count {
                    let idx = i * 3 + c;
                    if idx < processed_frame.len() {
                        color_means[c] += processed_frame[idx];
                    }
                }
                color_means[c] /= pixel_count as f32;
            }
            
            let mut color_cov = [[0f32; 3]; 3];
            for i in 0..pixel_count {
                for c1 in 0..3 {
                    let idx1 = i * 3 + c1;
                    if idx1 < processed_frame.len() {
                        for c2 in 0..3 {
                            let idx2 = i * 3 + c2;
                            if idx2 < processed_frame.len() {
                                color_cov[c1][c2] += (processed_frame[idx1] - color_means[c1]) 
                                                   * (processed_frame[idx2] - color_means[c2]);
                            }
                        }
                    }
                }
            }
            
            for c1 in 0..3 {
                for c2 in 0..3 {
                    color_cov[c1][c2] /= (pixel_count - 1) as f32;
                    texture_features.push(color_cov[c1][c2]);
                }
            }
        }
        
        // 3. 合并特征并调整维度
        let mut combined_features = Vec::with_capacity(self.feature_dim);
        combined_features.extend_from_slice(&color_histograms[0..color_histograms.len().min(512)]);
        combined_features.extend_from_slice(&texture_features[0..texture_features.len().min(256)]);
        
        // 调整至目标特征维度
        let mut final_features = self.resize_feature_vector(&combined_features, self.feature_dim);
        
        // 4. 应用归一化方法
        final_features = self.apply_normalization(&final_features)?;
        
        debug!("CPU RGB特征提取成功，维度: {}", final_features.len());
        Ok(final_features)
    }
    
    /// 将特征向量调整为指定维度
    fn resize_feature_vector(&self, features: &[f32], target_dim: usize) -> Vec<f32> {
        let src_dim = features.len();
        
        if src_dim == target_dim {
            return features.to_vec();
        } else if src_dim == 0 {
            return vec![0.0; target_dim];
        } else if src_dim > target_dim {
            // 降维：选择均匀间隔的特征
            let mut result = Vec::with_capacity(target_dim);
            let step = (src_dim as f32) / (target_dim as f32);
            
            for i in 0..target_dim {
                let src_idx = (i as f32 * step) as usize;
                result.push(features[src_idx.min(src_dim - 1)]);
            }
            
            result
        } else {
            // 升维：重复最后的特征或使用插值
            let mut result = features.to_vec();
            result.resize(target_dim, 0.0);
            
            // 可以使用简单的线性插值填充额外的值
            let step = (src_dim as f32) / ((target_dim - src_dim) as f32);
            for i in 0..(target_dim - src_dim) {
                let src_idx = (i as f32 * step) as usize;
                result[src_dim + i] = if src_idx + 1 < src_dim {
                    let t = (i as f32 * step) - (src_idx as f32);
                    features[src_idx] * (1.0 - t) + features[src_idx + 1] * t
                } else {
                    features[src_dim - 1]
                };
            }
            
            result
        }
    }
    
    /// 应用指定的归一化方法
    fn apply_normalization(&self, features: &[f32]) -> Result<Vec<f32>, VideoExtractionError> {
        let mut result = features.to_vec();
        
        if let Some(norm_type) = &self.normalization {
            match norm_type {
                NormalizationType::L2 => {
                    // 计算L2范数
                    let norm_sq: f32 = result.iter().map(|v| v * v).sum();
                    let norm = norm_sq.sqrt();
                    
                    if norm > 1e-10 {
                        for v in &mut result {
                            *v /= norm;
                        }
                    }
                },
                NormalizationType::ZScore => {
                    // 计算均值和标准差
                    let mean: f32 = result.iter().sum::<f32>() / result.len() as f32;
                    let var: f32 = result.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / result.len() as f32;
                    let std_dev = var.sqrt();
                    
                    if std_dev > 1e-10 {
                        for v in &mut result {
                            *v = (*v - mean) / std_dev;
                        }
                    }
                },
                NormalizationType::MinMax => {
                    // 最大最小归一化
                    if let (Some(min), Some(max)) = (
                        result.iter().copied().reduce(f32::min),
                        result.iter().copied().reduce(f32::max)
                    ) {
                        let range = max - min;
                        if range > 1e-10 {
                            for v in &mut result {
                                *v = (*v - min) / range;
                            }
                        }
                    }
                },
                NormalizationType::None => {
                    // 不执行归一化
                }
            }
        }
        
        Ok(result)
    }
    
    // 注意：opencv 特性未在 Cargo.toml 中定义，已注释掉相关代码
    /* #[cfg(feature = "opencv")]
    fn extract_color_histograms(&self, frame_mat: &opencv::core::Mat) -> Result<Vec<f32>, VideoExtractionError> {
        use opencv::{
            core::{Mat, MatTraitConst, Vector, _InputArray, _OutputArray, NORM_MINMAX},
            imgproc::{calc_hist, COLOR_BGR2HSV},
            prelude::*,
        };
        
        // 转换为HSV颜色空间
        let mut hsv_mat = Mat::default();
        opencv::imgproc::cvt_color(frame_mat, &mut hsv_mat, COLOR_BGR2HSV, 0)?;
        
        // 设置直方图参数
        let channels = Vector::from_slice(&[0, 1, 2]); // H, S, V通道
        let h_bins = 30; // H通道bins
        let s_bins = 32; // S通道bins
        let v_bins = 32; // V通道bins
        let hist_size = Vector::from_slice(&[h_bins, s_bins, v_bins]);
        
        // H: 0-180, S: 0-255, V: 0-255
        let h_ranges = Vector::from_slice(&[0f32, 180f32]);
        let s_ranges = Vector::from_slice(&[0f32, 256f32]);
        let v_ranges = Vector::from_slice(&[0f32, 256f32]);
        let ranges = Vector::from_slice(&[h_ranges[0], h_ranges[1], s_ranges[0], s_ranges[1], v_ranges[0], v_ranges[1]]);
        
        // 计算直方图
        let mut hist = Mat::default();
        calc_hist(
            &[hsv_mat], 
            &channels, 
            &Mat::default(), 
            &mut hist, 
            &hist_size, 
            &ranges, 
            false,
        )?;
        
        // 归一化直方图
        opencv::core::normalize(&hist, &mut hist, 0.0, 1.0, NORM_MINMAX, -1, &Mat::default())?;
        
        // 将直方图转换为特征向量
        let mut features = Vec::with_capacity((h_bins * s_bins * v_bins) as usize);
        
        for h in 0..h_bins {
            for s in 0..s_bins {
                for v in 0..v_bins {
                    let value = *hist.at_3d::<f32>(h, s, v)?;
                    features.push(value);
                }
            }
        }
        
        Ok(features)
    } */
    
    /* #[cfg(feature = "opencv")]
    fn extract_texture_features(&self, frame_mat: &opencv::core::Mat) -> Result<Vec<f32>, VideoExtractionError> {
        use opencv::{
            core::{Mat, Point, BORDER_DEFAULT},
            imgproc::{Sobel, CV_32F, THRESH_BINARY},
            prelude::*,
        };
        
        // 转换为灰度图
        let mut gray_mat = Mat::default();
        opencv::imgproc::cvt_color(frame_mat, &mut gray_mat, opencv::imgproc::COLOR_BGR2GRAY, 0)?;
        
        // 计算Sobel梯度
        let mut grad_x = Mat::default();
        let mut grad_y = Mat::default();
        
        Sobel(&gray_mat, &mut grad_x, CV_32F, 1, 0, 3, 1.0, 0.0, BORDER_DEFAULT)?;
        Sobel(&gray_mat, &mut grad_y, CV_32F, 0, 1, 3, 1.0, 0.0, BORDER_DEFAULT)?;
        
        // 计算梯度幅值和方向
        let mut magnitude = Mat::default();
        let mut angle = Mat::default();
        
        opencv::core::cartToPolar(&grad_x, &grad_y, &mut magnitude, &mut angle, true)?;
        
        // 计算方向直方图（梯度方向的分布）
        let bins = 36; // 每10度一个bin
        let mut angle_hist = vec![0f32; bins];
        
        let rows = magnitude.rows();
        let cols = magnitude.cols();
        
        for y in 0..rows {
            for x in 0..cols {
                let mag = *magnitude.at_2d::<f32>(y, x)?;
                let ang = *angle.at_2d::<f32>(y, x)?;
                
                // 只考虑梯度大小足够大的像素
                if mag > 10.0 {
                    let bin = ((ang / 10.0) as usize) % bins;
                    angle_hist[bin] += mag; // 加权直方图
                }
            }
        }
        
        // 归一化直方图
        let sum: f32 = angle_hist.iter().sum();
        if sum > 0.0 {
            for bin in &mut angle_hist {
                *bin /= sum;
            }
        }
        
        // 计算Haralick纹理特征（共生矩阵特征）
        let haralick_features = self.compute_haralick_features(&gray_mat)?;
        
        // 合并所有纹理特征
        let mut features = Vec::with_capacity(bins + haralick_features.len());
        features.extend_from_slice(&angle_hist);
        features.extend_from_slice(&haralick_features);
        
        Ok(features)
    } */
    
    /* #[cfg(feature = "opencv")]
    fn compute_haralick_features(&self, gray_mat: &opencv::core::Mat) -> Result<Vec<f32>, VideoExtractionError> {
        // 简化的Haralick纹理特征计算
        // 在实际应用中，可以使用更完整的实现或第三方库
        
        let rows = gray_mat.rows();
        let cols = gray_mat.cols();
        
        // 量化灰度级
        let gray_levels = 8;
        let mut quantized = opencv::core::Mat::default();
        gray_mat.convert_to(&mut quantized, opencv::core::CV_8U, 255.0/gray_levels, 0.0)?;
        
        // 计算共生矩阵（水平方向，距离=1）
        let mut cooccurrence = vec![vec![0f32; gray_levels]; gray_levels];
        
        for y in 0..rows {
            for x in 0..cols-1 {
                let i = *quantized.at_2d::<u8>(y, x)? as usize;
                let j = *quantized.at_2d::<u8>(y, x+1)? as usize;
                
                if i < gray_levels && j < gray_levels {
                    cooccurrence[i][j] += 1.0;
                    cooccurrence[j][i] += 1.0; // 对称矩阵
                }
            }
        }
        
        // 归一化共生矩阵
        let sum: f32 = cooccurrence.iter().flat_map(|row| row.iter()).sum();
        if sum > 0.0 {
            for i in 0..gray_levels {
                for j in 0..gray_levels {
                    cooccurrence[i][j] /= sum;
                }
            }
        }
        
        // 计算常用的Haralick特征（简化版）
        let mut features = Vec::with_capacity(5);
        
        // 1. 角二阶矩（Energy）
        let mut energy = 0.0;
        for i in 0..gray_levels {
            for j in 0..gray_levels {
                energy += cooccurrence[i][j] * cooccurrence[i][j];
            }
        }
        features.push(energy);
        
        // 2. 对比度（Contrast）
        let mut contrast = 0.0;
        for i in 0..gray_levels {
            for j in 0..gray_levels {
                contrast += (i as f32 - j as f32).powi(2) * cooccurrence[i][j];
            }
        }
        features.push(contrast);
        
        // 3. 相关性（Correlation）
        let mut mean_i = 0.0;
        let mut mean_j = 0.0;
        for i in 0..gray_levels {
            for j in 0..gray_levels {
                mean_i += i as f32 * cooccurrence[i][j];
                mean_j += j as f32 * cooccurrence[i][j];
            }
        }
        
        let mut var_i = 0.0;
        let mut var_j = 0.0;
        for i in 0..gray_levels {
            for j in 0..gray_levels {
                var_i += (i as f32 - mean_i).powi(2) * cooccurrence[i][j];
                var_j += (j as f32 - mean_j).powi(2) * cooccurrence[i][j];
            }
        }
        
        let mut correlation = 0.0;
        if var_i > 0.0 && var_j > 0.0 {
            for i in 0..gray_levels {
                for j in 0..gray_levels {
                    correlation += (i as f32 - mean_i) * (j as f32 - mean_j) * cooccurrence[i][j] / (var_i * var_j).sqrt();
                }
            }
        }
        features.push(correlation);
        
        // 4. 熵（Entropy）
        let mut entropy = 0.0;
        for i in 0..gray_levels {
            for j in 0..gray_levels {
                if cooccurrence[i][j] > 0.0 {
                    entropy -= cooccurrence[i][j] * cooccurrence[i][j].ln();
                }
            }
        }
        features.push(entropy);
        
        // 5. 同质性（Homogeneity）
        let mut homogeneity = 0.0;
        for i in 0..gray_levels {
            for j in 0..gray_levels {
                homogeneity += cooccurrence[i][j] / (1.0 + (i as f32 - j as f32).powi(2));
            }
        }
        features.push(homogeneity);
        
        Ok(features)
    } */
    
    /* #[cfg(feature = "opencv")]
    fn extract_edge_features(&self, frame_mat: &opencv::core::Mat) -> Result<Vec<f32>, VideoExtractionError> {
        use opencv::{
            core::{Mat, Point, Size, BORDER_DEFAULT},
            imgproc::{Canny, GaussianBlur, CV_8U},
            prelude::*,
        };
        
        // 转换为灰度图
        let mut gray_mat = Mat::default();
        opencv::imgproc::cvt_color(frame_mat, &mut gray_mat, opencv::imgproc::COLOR_BGR2GRAY, 0)?;
        
        // 高斯模糊以减少噪声
        let mut blurred = Mat::default();
        GaussianBlur(&gray_mat, &mut blurred, Size::new(5, 5), 0.0, 0.0, BORDER_DEFAULT)?;
        
        // Canny边缘检测
        let mut edges = Mat::default();
        Canny(&blurred, &mut edges, 50.0, 150.0, 3, false)?;
        
        // 提取边缘特征
        let rows = edges.rows();
        let cols = edges.cols();
        
        // 计算水平和垂直区域的边缘密度
        let regions_h = 4;
        let regions_v = 4;
        let region_height = rows / regions_v;
        let region_width = cols / regions_h;
        
        let mut edge_density = vec![0f32; regions_h * regions_v];
        
        for y in 0..rows {
            for x in 0..cols {
                let edge_value = *edges.at_2d::<u8>(y, x)?;
                if edge_value > 0 {
                    let region_y = y / region_height;
                    let region_x = x / region_width;
                    let region_idx = region_y * regions_h + region_x;
                    if region_idx < edge_density.len() {
                        edge_density[region_idx] += 1.0;
                    }
                }
            }
        }
        
        // 归一化边缘密度
        for i in 0..regions_h * regions_v {
            let region_y = i / regions_h;
            let region_x = i % regions_h;
            
            let y_start = region_y * region_height;
            let x_start = region_x * region_width;
            
            let y_end = (y_start + region_height).min(rows as usize);
            let x_end = (x_start + region_width).min(cols as usize);
            
            let region_area = (y_end - y_start) * (x_end - x_start);
            if region_area > 0 {
                edge_density[i] /= region_area as f32;
            }
        }
        
        // 计算边缘方向特征
        let mut grad_x = Mat::default();
        let mut grad_y = Mat::default();
        
        opencv::imgproc::Sobel(&gray_mat, &mut grad_x, CV_8U, 1, 0, 3, 1.0, 0.0, BORDER_DEFAULT)?;
        opencv::imgproc::Sobel(&gray_mat, &mut grad_y, CV_8U, 0, 1, 3, 1.0, 0.0, BORDER_DEFAULT)?;
        
        // 计算梯度方向直方图
        let direction_bins = 8;
        let mut direction_hist = vec![0f32; direction_bins];
        
        for y in 0..rows {
            for x in 0..cols {
                if *edges.at_2d::<u8>(y, x)? > 0 {
                    let gx = *grad_x.at_2d::<u8>(y, x)? as f32 - 128.0;
                    let gy = *grad_y.at_2d::<u8>(y, x)? as f32 - 128.0;
                    
                    let angle = gy.atan2(gx); // 弧度，范围 [-pi, pi]
                    let angle_norm = (angle + std::f32::consts::PI) / (2.0 * std::f32::consts::PI); // 归一化到 [0, 1]
                    
                    let bin = (angle_norm * direction_bins as f32) as usize % direction_bins;
                    direction_hist[bin] += 1.0;
                }
            }
        }
        
        // 归一化方向直方图
        let sum: f32 = direction_hist.iter().sum();
        if sum > 0.0 {
            for bin in &mut direction_hist {
                *bin /= sum;
            }
        }
        
        // 合并所有边缘特征
        let mut features = Vec::with_capacity(edge_density.len() + direction_hist.len());
        features.extend_from_slice(&edge_density);
        features.extend_from_slice(&direction_hist);
        
        Ok(features)
    } */
}

impl FeatureExtractor for RGBFeatureExtractor {
    fn initialize(&mut self) -> Result<(), VideoExtractionError> {
        if self.initialized {
            debug!("RGB特征提取器已经初始化，跳过");
            return Ok(());
        }
        
        info!("正在初始化RGB特征提取器：{}，特征维度：{}", self.model_type, self.feature_dim);
        
        // 加载模型权重
        self.load_model_weights()?;
        
        self.initialized = true;
        info!("RGB特征提取器初始化完成");
        
        Ok(())
    }
    
    fn process_frame(&mut self, frame: &VideoFrame) -> Result<Vec<f32>, VideoExtractionError> {
        if !self.initialized {
            return Err(VideoExtractionError::ModelError(
                "RGB特征提取器尚未初始化".to_string()
            ));
        }
        
        // 确保帧数据有效
        if frame.data.is_empty() {
            return Err(VideoExtractionError::InputError(
                "无效的视频帧数据".to_string()
            ));
        }
        
        // 1. 预处理帧数据：转换为浮点数[0,1]范围
        let mut processed_data = Vec::with_capacity(frame.data.len());
        for &pixel in &frame.data {
            processed_data.push(pixel as f32 / 255.0);
        }
        
        // 2. 提取RGB特征
        let features = self.extract_rgb_features(&processed_data)?;
        
        debug!("成功从帧提取RGB特征，维度: {}", features.len());
        
        Ok(features)
    }
    
    fn get_feature_dim(&self) -> usize {
        self.feature_dim
    }
    
    fn get_feature_type(&self) -> VideoFeatureType {
        self.feature_type
    }
    
    fn release(&mut self) -> Result<(), VideoExtractionError> {
        if self.initialized {
            info!("释放RGB特征提取器资源");
            // 清除模型权重释放内存
            self.model_weights = None;
            self.initialized = false;
        }
        
        Ok(())
    }
    
    fn process_batch(&mut self, frames: &[VideoFrame]) -> Result<Vec<Vec<f32>>, VideoExtractionError> {
        if !self.initialized {
            return Err(VideoExtractionError::ModelError(
                "RGB特征提取器尚未初始化".to_string()
            ));
        }
        
        if frames.is_empty() {
            return Err(VideoExtractionError::InputError(
                "帧批处理：输入帧数组为空".to_string()
            ));
        }
        
        debug!("RGB特征提取器批量处理{}个帧", frames.len());
        
        // 使用rayon并行处理提高效率
        use rayon::prelude::*;
        
        let results: Result<Vec<_>, _> = frames.par_iter()
            .map(|frame| self.process_frame(frame))
            .collect();
            
        results
    }
    
    fn temporal_pooling(&self, features: &[Vec<f32>], pooling_type: &TemporalPoolingType) -> Result<Vec<f32>, VideoExtractionError> {
        if features.is_empty() {
            return Err(VideoExtractionError::ProcessingError(
                "没有特征可以池化".to_string()
            ));
        }
        
        match pooling_type {
            TemporalPoolingType::Mean => {
                // 均值池化
                Ok(super::pooling::temporal_mean_pooling(features))
            },
            TemporalPoolingType::Max => {
                // 最大值池化
                Ok(super::pooling::temporal_max_pooling(features))
            },
            TemporalPoolingType:: Attention => {
                // 基于 L2 范数的注意力池化实现：根据特征能量自适应分配权重
                let weights: Vec<f32> = features.iter()
                    .map(|feature| {
                        let norm_sq: f32 = feature.iter().map(|v| v * v).sum();
                        norm_sq.sqrt()
                    })
                    .collect();
                
                // 归一化权重，避免数值不稳定
                let sum_weights: f32 = weights.iter().sum();
                let norm_weights = if sum_weights > 1e-6 {
                    weights.iter().map(|w| w / sum_weights).collect()
                } else {
                    vec![1.0 / weights.len() as f32; weights.len()]
                };
                
                // 加权求和
                let dim = features[0].len();
                let mut pooled = vec![0.0; dim];
                
                for (i, feature) in features.iter().enumerate() {
                    let weight = if i < norm_weights.len() { norm_weights[i] } else { 0.0 };
                    for (j, &val) in feature.iter().enumerate() {
                        if j < dim {
                            pooled[j] += val * weight;
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
} 