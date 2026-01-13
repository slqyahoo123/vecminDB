//! 组合特征提取器实现
//! 
//! 本模块提供组合多种提取器的复合提取器实现

use super::VideoFeatureExtractor;
use super::ConfigOption;
use super::ConfigOptionType;
use crate::data::multimodal::video_extractor::types::*;
use crate::data::multimodal::video_extractor::config::VideoFeatureConfig;
use crate::data::multimodal::video_extractor::error::VideoExtractionError;
use std::path::Path;
use std::collections::HashMap;
use log::{info, warn, error, debug};

/// 组合提取器，将多个提取器组合到一起
pub struct CompositeExtractor {
    name: String,
    description: String,
    extractors: Vec<Box<dyn VideoFeatureExtractor + Send + Sync>>,
    config: VideoFeatureConfig,
    aggregation_strategy: FeatureAggregationStrategy,
    projection_matrices: HashMap<(usize, usize), Vec<Vec<f32>>>, // 投影矩阵缓存，键为(输出维度, 输入维度)
}

impl CompositeExtractor {
    /// 创建新的组合提取器
    pub fn new(name: &str, description: &str, extractors: Vec<Box<dyn VideoFeatureExtractor + Send + Sync>>, config: &VideoFeatureConfig) -> Self {
        Self {
            name: name.to_string(),
            description: description.to_string(),
            extractors,
            config: config.clone(),
            aggregation_strategy: FeatureAggregationStrategy::Concatenate,
            projection_matrices: HashMap::new(),
        }
    }
    
    /// 添加提取器
    pub fn add_extractor(&mut self, extractor: Box<dyn VideoFeatureExtractor + Send + Sync>) {
        self.extractors.push(extractor);
    }
    
    /// 设置特征聚合策略
    pub fn set_aggregation_strategy(&mut self, strategy: FeatureAggregationStrategy) {
        self.aggregation_strategy = strategy;
    }
    
    /// 获取特征聚合策略
    pub fn get_aggregation_strategy(&self) -> FeatureAggregationStrategy {
        self.aggregation_strategy
    }
    
    /// 获取当前配置
    pub fn get_config(&self) -> &VideoFeatureConfig {
        &self.config
    }
    
    /// 获取提取器列表
    pub fn get_extractors(&self) -> &Vec<Box<dyn VideoFeatureExtractor + Send + Sync>> {
        &self.extractors
    }
    
    /// 加载投影矩阵
    fn load_projection_matrix(&mut self, output_dim: usize, input_dim: usize) -> Result<&Vec<Vec<f32>>, VideoExtractionError> {
        // 先检查缓存中是否已存在
        let key = (output_dim, input_dim);
        if !self.projection_matrices.contains_key(&key) {
            // 尝试从文件加载
            if let Some(matrix) = self.load_matrix_from_file(output_dim, input_dim)? {
                self.projection_matrices.insert(key, matrix);
            } else {
                // 如果没有预训练文件，则生成默认矩阵
                let default_matrix = self.generate_default_matrix(output_dim, input_dim);
                self.projection_matrices.insert(key, default_matrix);
            }
        }
        
        Ok(self.projection_matrices.get(&key).unwrap())
    }
    
    /// 从文件加载投影矩阵
    fn load_matrix_from_file(&self, output_dim: usize, input_dim: usize) -> Result<Option<Vec<Vec<f32>>>, VideoExtractionError> {
        // 构建矩阵文件路径
        let matrix_file = self.get_matrix_file_path(output_dim, input_dim);
        if let Ok(file) = std::fs::File::open(matrix_file) {
            info!("加载投影矩阵 {}x{}", output_dim, input_dim);
            
            // 尝试解析文件内容
            match self.parse_matrix_file(file, output_dim, input_dim) {
                Ok(matrix) => Ok(Some(matrix)),
                Err(e) => {
                    warn!("解析投影矩阵文件错误: {}", e);
                    Ok(None)
                }
            }
        } else {
            // 文件不存在，返回None
            debug!("找不到投影矩阵文件 {}x{}", output_dim, input_dim);
            Ok(None)
        }
    }
    
    /// 解析矩阵文件
    fn parse_matrix_file(&self, file: std::fs::File, output_dim: usize, input_dim: usize) -> Result<Vec<Vec<f32>>, VideoExtractionError> {
        use std::io::{BufRead, BufReader};
        
        let reader = BufReader::new(file);
        let mut matrix = vec![vec![0.0; input_dim]; output_dim];
        
        for (i, line) in reader.lines().enumerate() {
            if i >= output_dim {
                break;
            }
            
            let line = line.map_err(|e| VideoExtractionError::FileError(format!("读取矩阵文件行失败: {}", e)))?;
            let values: Result<Vec<f32>, _> = line.split_whitespace()
                .map(|s| s.parse::<f32>())
                .collect();
                
            let values = values.map_err(|e| VideoExtractionError::FileError(format!("解析矩阵值失败: {}", e)))?;
            
            // 确保值的数量正确
            if values.len() != input_dim {
                return Err(VideoExtractionError::FileError(
                    format!("矩阵行{}的值数量不正确: 预期 {}, 实际 {}", i, input_dim, values.len())
                ));
            }
            
            matrix[i] = values;
        }
        
        Ok(matrix)
    }
    
    /// 获取矩阵文件路径
    fn get_matrix_file_path(&self, output_dim: usize, input_dim: usize) -> std::path::PathBuf {
        // 从配置中获取矩阵文件目录，默认为"models/projection_matrices"
        let matrix_dir = self.config.custom_params
            .get("projection_matrix_dir")
            .cloned()
            .unwrap_or_else(|| "models/projection_matrices".to_string());
            
        // 构建文件名，格式为"matrix_{output_dim}x{input_dim}.txt"
        let file_name = format!("matrix_{}x{}.txt", output_dim, input_dim);
        std::path::Path::new(&matrix_dir).join(file_name)
    }
    
    /// 生成默认投影矩阵
    fn generate_default_matrix(&self, output_dim: usize, input_dim: usize) -> Vec<Vec<f32>> {
        info!("生成默认投影矩阵 {}x{}", output_dim, input_dim);
        
        let mut matrix = vec![vec![0.0; input_dim]; output_dim];
        let features_count = input_dim / output_dim;
        
        // 如果输入维度大于输出维度，则尝试生成更智能的降维矩阵
        if features_count > 0 {
            for i in 0..output_dim {
                for feat_idx in 0..features_count {
                    let feature_start = feat_idx * output_dim;
                    
                    // 为同一维度设置较高权重
                    if feature_start + i < input_dim {
                        matrix[i][feature_start + i] = 1.0 / features_count as f32;
                    }
                    
                    // 为相邻维度设置较低权重
                    for offset in 1..=2 {
                        if i >= offset && feature_start + (i - offset) < input_dim {
                            matrix[i][feature_start + (i - offset)] = 0.2 / features_count as f32;
                        }
                        
                        if i + offset < output_dim && feature_start + (i + offset) < input_dim {
                            matrix[i][feature_start + (i + offset)] = 0.2 / features_count as f32;
                        }
                    }
                }
            }
        } else {
            // 如果输入维度小于输出维度，生成一个较为均匀的上采样矩阵
            let repeat_factor = (output_dim as f32 / input_dim as f32).ceil() as usize;
            
            for i in 0..output_dim {
                let source_idx = (i / repeat_factor) % input_dim;
                matrix[i][source_idx] = 1.0;
            }
        }
        
        // 对每行进行L2归一化
        for i in 0..output_dim {
            let norm_sq: f32 = matrix[i].iter().map(|&x| x * x).sum();
            let norm = norm_sq.sqrt();
            
            if norm > 1e-6 {
                for j in 0..input_dim {
                    matrix[i][j] /= norm;
                }
            }
        }
        
        matrix
    }
    
    /// 聚合特征
    fn aggregate_features(&self, features: &[VideoFeature]) -> Result<Vec<f32>, VideoExtractionError> {
        if features.is_empty() {
            return Err(VideoExtractionError::ProcessingError(
                "没有特征可以聚合".to_string()
            ));
        }
        
        match self.aggregation_strategy {
            FeatureAggregationStrategy::Concatenate => {
                // 简单拼接所有特征
                let mut result = Vec::new();
                for feature in features {
                    result.extend_from_slice(&feature.features);
                }
                Ok(result)
            },
            
            FeatureAggregationStrategy::Average => {
                // 确保所有特征维度相同
                let dim = features[0].dimensions;
                for feature in features.iter().skip(1) {
                    if feature.dimensions != dim {
                        return Err(VideoExtractionError::ProcessingError(
                            format!("特征维度不一致：{}和{}", dim, feature.dimensions)
                        ));
                    }
                }
                
                // 计算平均值
                let mut result = vec![0.0; dim];
                for feature in features {
                    for (i, val) in feature.features.iter().enumerate() {
                        result[i] += val;
                    }
                }
                
                // 归一化
                let n = features.len() as f32;
                for val in &mut result {
                    *val /= n;
                }
                
                Ok(result)
            },
            
            FeatureAggregationStrategy::WeightedAverage => {
                // 实现加权平均聚合策略
                if features.is_empty() {
                    return Err(VideoExtractionError::ProcessingError(
                        "没有可用的特征进行加权平均".to_string()
                    ));
                }
                
                // 获取特征维度
                let dim = features[0].features.len();
                let mut result = vec![0.0; dim];
                
                // 计算每个提取器的权重
                let mut weights = Vec::with_capacity(features.len());
                let mut total_weight = 0.0;
                
                for (i, feature) in features.iter().enumerate() {
                    // 根据特征类型和质量计算权重
                    let weight = match feature.feature_type {
                        VideoFeatureType::RGB => 1.0,            // RGB特征基础权重
                        VideoFeatureType::OpticalFlow => 1.2,    // 光流特征权重稍高
                        VideoFeatureType::Audio => 0.8,          // 音频特征权重稍低
                        VideoFeatureType::Generic => 0.9,        // 通用特征中等权重
                        VideoFeatureType::I3D => 1.1,            // I3D特征权重较高
                        VideoFeatureType::SlowFast => 1.1,       // SlowFast特征权重较高
                        VideoFeatureType::Custom(_) => 0.8,      // 自定义特征中等权重
                    };
                    
                    // 根据元数据调整权重
                    let adjusted_weight = if let Some(ref metadata) = feature.metadata {
                        // 根据分辨率调整权重
                        let resolution_factor = match (metadata.width, metadata.height) {
                            (w, h) if w >= 1920 && h >= 1080 => 1.2,  // 高清视频权重更高
                            (w, h) if w >= 1280 && h >= 720 => 1.1,   // 标清视频
                            _ => 0.9,                                 // 低分辨率视频
                        };
                        
                        // 根据视频质量调整权重（使用分辨率和帧率作为质量指标）
                        let quality_factor = if metadata.fps >= 30.0 && metadata.width >= 1280 {
                            1.0
                        } else if metadata.fps >= 24.0 && metadata.width >= 720 {
                            0.9
                        } else {
                            0.8
                        };
                        
                        // 综合权重
                        weight * resolution_factor * quality_factor
                    } else {
                        weight
                    };
                    
                    weights.push(adjusted_weight);
                    total_weight += adjusted_weight;
                }
                
                // 归一化权重
                if total_weight > 0.0 {
                    for w in &mut weights {
                        *w /= total_weight;
                    }
                } else {
                    // 如果总权重为0，则平均分配
                    let avg_weight = 1.0 / features.len() as f32;
                    weights = vec![avg_weight; features.len()];
                }
                
                // 加权平均计算
                for (feature, weight) in features.iter().zip(weights.iter()) {
                    for (i, &val) in feature.features.iter().enumerate() {
                        if i < result.len() {
                            result[i] += val * weight;
                        }
                    }
                }
                
                debug!("使用加权平均聚合策略合并了{}个特征，权重：{:?}", features.len(), weights);
                Ok(result)
            },
            
            FeatureAggregationStrategy::Voting => {
                // 实现特征投票聚合策略
                if features.is_empty() {
                    return Err(VideoExtractionError::ProcessingError(
                        "没有可用的特征进行投票聚合".to_string()
                    ));
                }
                
                // 获取特征维度
                let dim = features[0].features.len();
                
                // 针对不同维度的特征，先进行维度统一化处理
                let normalized_features = self.normalize_feature_dimensions(features, dim)?;
                
                // 对于投票策略，我们需要将特征值转换为离散类别
                // 1. 计算每个特征向量的阈值（可以是均值、中位数等）
                let mut thresholds = Vec::with_capacity(normalized_features.len());
                
                for feature in &normalized_features {
                    // 使用百分位数作为阈值，更稳健
                    let mut values = feature.features.clone();
                    values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                    
                    // 使用第75百分位数作为高活跃阈值
                    let high_idx = (values.len() as f32 * 0.75) as usize;
                    let high_threshold = values.get(high_idx.min(values.len() - 1)).unwrap_or(&0.0);
                    
                    // 使用第25百分位数作为低活跃阈值
                    let low_idx = (values.len() as f32 * 0.25) as usize;
                    let low_threshold = values.get(low_idx.min(values.len() - 1)).unwrap_or(&0.0);
                    
                    thresholds.push((*low_threshold, *high_threshold));
                }
                
                // 2. 投票：使用三态投票机制（高、中、低活跃度）
                let mut high_votes = vec![0; dim];
                let mut low_votes = vec![0; dim];
                
                for (feature_idx, feature) in normalized_features.iter().enumerate() {
                    if feature_idx < thresholds.len() {
                        let (low_threshold, high_threshold) = thresholds[feature_idx];
                        
                        for (i, &val) in feature.features.iter().enumerate() {
                            if i < high_votes.len() {
                                if val >= high_threshold {
                                    high_votes[i] += 1;  // 高活跃投票
                                } else if val <= low_threshold {
                                    low_votes[i] += 1;   // 低活跃投票
                                }
                                // 中活跃不计票
                            }
                        }
                    }
                }
                
                // 3. 根据投票结果生成最终特征向量
                // 计算投票阈值：超过半数特征表示同意
                let feature_count = normalized_features.len();
                let high_threshold = feature_count / 3;  // 至少1/3的特征认为是高活跃
                let low_threshold = feature_count / 3;   // 至少1/3的特征认为是低活跃
                
                let mut result = vec![0.0; dim];
                
                for i in 0..dim {
                    // 如果高活跃票数超过阈值，设为1.0
                    if high_votes[i] >= high_threshold {
                        result[i] = 1.0;
                    } 
                    // 如果低活跃票数超过阈值，设为-1.0（表示抑制）
                    else if low_votes[i] >= low_threshold {
                        result[i] = -1.0;
                    }
                    // 其余情况，维持中性状态0.0
                }
                
                // 4. 应用后处理
                // 标准化结果，使最大绝对值为1
                let max_abs = result.iter().map(|&v: &f32| v.abs()).fold(f32::MIN, f32::max);
                if max_abs > 1e-6 {
                    for val in &mut result {
                        *val /= max_abs;
                    }
                }
                
                // 如果配置了，可以将结果重新映射到[0,1]范围
                let use_positive_only = self.config.custom_params
                    .get("voting_use_positive_only")
                    .map(|v| v == "true" || v == "1")
                    .unwrap_or(false);
                if use_positive_only {
                    for val in &mut result {
                        *val = (*val + 1.0) / 2.0;
                    }
                }
                
                debug!("使用投票聚合策略合并了{}个特征", normalized_features.len());
                Ok(result)
            },
            
            FeatureAggregationStrategy::Learned => {
                // 实现学习型聚合策略
                if features.is_empty() {
                    return Err(VideoExtractionError::ProcessingError(
                        "没有可用的特征进行学习聚合".to_string()
                    ));
                }
                
                // 获取特征维度
                let dim = features[0].features.len();
                
                // 首先，将所有特征连接成一个大向量
                let mut combined_features = Vec::with_capacity(features.len() * dim);
                for feature in features {
                    combined_features.extend_from_slice(&feature.features);
                }
                
                // 加载预训练的投影矩阵
                let projection_matrix = self.load_projection_matrix(dim, combined_features.len())?;
                
                // 应用投影矩阵
                let mut result = vec![0.0; dim];
                for i in 0..dim {
                    let mut sum = 0.0;
                    for j in 0..combined_features.len() {
                        sum += projection_matrix[i][j] * combined_features[j];
                    }
                    result[i] = sum;
                }
                
                // 应用非线性激活函数(ReLU)
                for val in &mut result {
                    *val = val.max(0.0);
                }
                
                // 应用L2归一化
                let norm_sq: f32 = result.iter().map(|v| v * v).sum();
                let norm = norm_sq.sqrt();
                if norm > 1e-6 {
                    for val in &mut result {
                        *val /= norm;
                    }
                }
                
                debug!("使用学习型聚合策略合并了{}个特征到{}维向量", features.len(), dim);
                Ok(result)
            },
        }
    }
    
    /// 将特征维度标准化，确保所有特征具有相同的维度
    fn normalize_feature_dimensions(&self, features: &[VideoFeature], target_dim: usize) -> Result<Vec<VideoFeature>, VideoExtractionError> {
        let mut normalized = Vec::with_capacity(features.len());
        
        for feature in features {
            if feature.features.len() == target_dim {
                // 维度已经匹配，直接添加
                normalized.push(feature.clone());
            } else {
                // 需要调整维度
                let mut adjusted = feature.clone();
                
                if feature.features.len() < target_dim {
                    // 维度扩展 - 用0填充
                    adjusted.features.resize(target_dim, 0.0);
                } else {
                    // 维度压缩 - 截断或平均池化
                    let use_pooling = self.config.custom_params
                        .get("use_pooling_for_dimension_reduction")
                        .map(|v| v != "false" && v != "0")
                        .unwrap_or(true);
                    if use_pooling {
                        // 使用平均池化进行降维
                        adjusted.features = self.dimension_reduction_pooling(&feature.features, target_dim);
                    } else {
                        // 简单截断
                        adjusted.features.truncate(target_dim);
                    }
                }
                
                adjusted.dimensions = target_dim;
                normalized.push(adjusted);
            }
        }
        
        Ok(normalized)
    }
    
    /// 使用平均池化进行特征降维
    fn dimension_reduction_pooling(&self, features: &[f32], target_dim: usize) -> Vec<f32> {
        if features.len() <= target_dim {
            return features.to_vec();
        }
        
        let mut result = vec![0.0; target_dim];
        let window_size = (features.len() as f32 / target_dim as f32).ceil() as usize;
        
        for i in 0..target_dim {
            let start = i * window_size;
            let end = ((i + 1) * window_size).min(features.len());
            
            if start < end {
                let window = &features[start..end];
                let sum: f32 = window.iter().sum();
                result[i] = sum / window.len() as f32;
            }
        }
        
        result
    }
}

impl VideoFeatureExtractor for CompositeExtractor {
    fn name(&self) -> &str {
        &self.name
    }
    
    fn description(&self) -> &str {
        &self.description
    }
    
    fn supported_features(&self) -> Vec<VideoFeatureType> {
        let mut features = Vec::new();
        for extractor in &self.extractors {
            features.extend(extractor.supported_features());
        }
        features.sort();
        features.dedup();
        features
    }
    
    fn extract_features(&self, video_path: &Path, config: &VideoFeatureConfig) -> Result<VideoFeature, VideoExtractionError> {
        if self.extractors.is_empty() {
            return Err(VideoExtractionError::ProcessingError("组合提取器中没有提取器".to_string()));
        }
        
        let mut features = Vec::new();
        let mut metadata = None;
        
        // 从所有提取器获取特征
        for extractor in &self.extractors {
            match extractor.extract_features(video_path, config) {
                Ok(feature) => {
                    // 保存第一个成功提取的元数据
                    if metadata.is_none() && feature.metadata.is_some() {
                        metadata = feature.metadata.clone();
                    }
                    features.push(feature);
                },
                Err(e) => {
                    warn!("提取器{}提取特征失败: {}", extractor.name(), e);
                }
            }
        }
        
        if features.is_empty() {
            return Err(VideoExtractionError::ProcessingError(
                "所有提取器都失败了".to_string()
            ));
        }
        
        // 聚合特征
        let aggregated = self.aggregate_features(&features)?;
        
        // 创建结果
        Ok(VideoFeature {
            feature_type: VideoFeatureType::Generic,
            features: aggregated,
            metadata,
            dimensions: aggregated.len(),
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs(),
        })
    }
    
    fn is_available(&self) -> bool {
        !self.extractors.is_empty() && self.extractors.iter().all(|e| e.is_available())
    }
    
    fn initialize(&mut self) -> Result<(), VideoExtractionError> {
        let mut errors = Vec::new();
        
        for extractor in &mut self.extractors {
            if let Err(e) = extractor.initialize() {
                error!("初始化提取器{}失败: {}", extractor.name(), e);
                errors.push(e);
            }
        }
        
        if !errors.is_empty() {
            Err(VideoExtractionError::ProcessingError(
                format!("初始化组合提取器时有{}个错误", errors.len())
            ))
        } else {
            Ok(())
        }
    }
    
    fn release(&mut self) -> Result<(), VideoExtractionError> {
        let mut errors = Vec::new();
        
        for extractor in &mut self.extractors {
            if let Err(e) = extractor.release() {
                warn!("释放提取器{}资源失败: {}", extractor.name(), e);
                errors.push(e);
            }
        }
        
        if errors.is_empty() {
            Ok(())
        } else {
            Err(VideoExtractionError::ProcessingError(
                format!("释放资源时发生{}个错误", errors.len())
            ))
        }
    }
    
    fn get_config_options(&self) -> HashMap<String, ConfigOption> {
        let mut options = HashMap::new();
        
        options.insert("aggregation_strategy".to_string(), ConfigOption {
            name: "aggregation_strategy".to_string(),
            description: "特征聚合策略".to_string(),
            option_type: ConfigOptionType::Enum,
            default_value: Some("concatenate".to_string()),
            allowed_values: Some(vec![
                "concatenate".to_string(),
                "average".to_string(),
                "weighted_average".to_string(),
                "voting".to_string(),
                "learned".to_string(),
            ]),
        });
        
        // 合并所有提取器的配置选项
        for extractor in &self.extractors {
            let ext_name = extractor.name();
            let ext_options = extractor.get_config_options();
            
            for (key, option) in ext_options {
                let prefixed_key = format!("{}:{}", ext_name, key);
                options.insert(prefixed_key, option);
            }
        }
        
        options
    }
    
    fn batch_extract(&self, video_paths: &[std::path::PathBuf], config: &VideoFeatureConfig) 
        -> Result<HashMap<std::path::PathBuf, Result<VideoFeature, VideoExtractionError>>, VideoExtractionError> {
        if self.extractors.is_empty() {
            return Err(VideoExtractionError::ProcessingError("组合提取器中没有提取器".to_string()));
        }
        
        // 从每个提取器获取结果
        let mut all_results: Vec<HashMap<std::path::PathBuf, Result<VideoFeature, VideoExtractionError>>> = Vec::new();
        
        for extractor in &self.extractors {
            match extractor.batch_extract(video_paths, config) {
                Ok(results) => all_results.push(results),
                Err(e) => {
                    warn!("提取器{}批量提取失败: {}", extractor.name(), e);
                }
            }
        }
        
        if all_results.is_empty() {
            return Err(VideoExtractionError::ProcessingError(
                "所有提取器的批量提取都失败了".to_string()
            ));
        }
        
        // 聚合结果
        let mut final_results = HashMap::new();
        
        for path in video_paths {
            let mut path_features = Vec::new();
            let mut metadata = None;
            
            // 收集每个提取器对当前路径的结果
            for results in &all_results {
                if let Some(result) = results.get(path) {
                    if let Ok(feature) = result {
                        if metadata.is_none() && feature.metadata.is_some() {
                            metadata = feature.metadata.clone();
                        }
                        path_features.push(feature.clone());
                    }
                }
            }
            
            // 如果有至少一个结果，则聚合
            if !path_features.is_empty() {
                match self.aggregate_features(&path_features) {
                    Ok(aggregated) => {
                        let feature = VideoFeature {
                            feature_type: VideoFeatureType::Generic,
                            features: aggregated,
                            metadata,
                            dimensions: aggregated.len(),
                            timestamp: std::time::SystemTime::now()
                                .duration_since(std::time::UNIX_EPOCH)
                                .unwrap_or_default()
                                .as_secs(),
                        };
                        final_results.insert(path.clone(), Ok(feature));
                    },
                    Err(e) => {
                        final_results.insert(path.clone(), Err(e));
                    }
                }
            } else {
                final_results.insert(
                    path.clone(), 
                    Err(VideoExtractionError::ProcessingError(
                        "没有提取器成功处理此视频".to_string()
                    ))
                );
            }
        }
        
        Ok(final_results)
    }
} 