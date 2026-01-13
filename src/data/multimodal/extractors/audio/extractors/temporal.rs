//! 时域特征提取器实现
//!
//! 该模块提供音频信号的时域特征提取功能，包括波形统计特征、
//! 如均值、方差、峰度、偏度等，以及时域的波形分析特征。

use ndarray::Array2;
use crate::{Result, Error};
#[cfg(feature = "multimodal")]
use crate::data::multimodal::extractors::audio::extractors::frame_signal;

/// 时域特征提取器
pub struct TemporalFeaturesExtractor {
    /// 帧长度
    frame_size: usize,
    /// 帧移步长
    hop_size: usize,
    /// 采样率
    sample_rate: u32,
    /// 是否归一化
    normalize: bool,
    /// 提取的特征类型
    features: Vec<TemporalFeatureType>,
}

/// 时域特征类型
pub enum TemporalFeatureType {
    /// 均值
    Mean,
    /// 标准差
    StdDev,
    /// 峰度 (四阶矩)
    Kurtosis,
    /// 偏度 (三阶矩)
    Skewness,
    /// 过零率
    ZeroCrossingRate,
    /// 最大值
    Max,
    /// 最小值
    Min,
    /// 动态范围
    DynamicRange,
    /// 均方根 (RMS)
    RootMeanSquare,
    /// 自相关
    Autocorrelation,
    /// 振幅包络
    AmplitudeEnvelope,
    /// 峰值因子 (Crest Factor)
    CrestFactor,
}

impl Default for TemporalFeaturesExtractor {
    fn default() -> Self {
        Self {
            frame_size: 512,
            hop_size: 256,
            sample_rate: 44100,
            normalize: true,
            features: vec![
                TemporalFeatureType::Mean,
                TemporalFeatureType::StdDev,
                TemporalFeatureType::Kurtosis,
                TemporalFeatureType::Skewness,
                TemporalFeatureType::ZeroCrossingRate,
                TemporalFeatureType::Max,
                TemporalFeatureType::Min,
                TemporalFeatureType::DynamicRange,
                TemporalFeatureType::RootMeanSquare,
            ],
        }
    }
}

impl TemporalFeaturesExtractor {
    /// 创建新的时域特征提取器
    pub fn new(
        frame_size: usize,
        hop_size: usize,
        sample_rate: u32,
        normalize: bool,
        features: Vec<TemporalFeatureType>,
    ) -> Self {
        Self {
            frame_size,
            hop_size,
            sample_rate,
            normalize,
            features,
        }
    }
    
    /// 计算过零率
    fn calculate_zcr(&self, frame: &[f32]) -> f32 {
        let mut crossings = 0;
        
        for i in 1..frame.len() {
            if (frame[i] >= 0.0 && frame[i-1] < 0.0) || 
               (frame[i] < 0.0 && frame[i-1] >= 0.0) {
                crossings += 1;
            }
        }
        
        crossings as f32 / (frame.len() as f32 - 1.0)
    }
    
    /// 计算均方根
    fn calculate_rms(&self, frame: &[f32]) -> f32 {
        let sum_squares: f32 = frame.iter().map(|&x| x * x).sum();
        (sum_squares / frame.len() as f32).sqrt()
    }
    
    /// 计算峰度
    fn calculate_kurtosis(&self, frame: &[f32]) -> f32 {
        let n = frame.len() as f32;
        let mean = frame.iter().sum::<f32>() / n;
        
        let mut variance = 0.0;
        let mut sum_fourth_power = 0.0;
        
        for &x in frame {
            let deviation = x - mean;
            let deviation_squared = deviation * deviation;
            variance += deviation_squared;
            sum_fourth_power += deviation_squared * deviation_squared;
        }
        
        variance /= n;
        
        if variance > 1e-10 {
            (sum_fourth_power / n) / (variance * variance) - 3.0
        } else {
            0.0
        }
    }
    
    /// 计算偏度
    fn calculate_skewness(&self, frame: &[f32]) -> f32 {
        let n = frame.len() as f32;
        let mean = frame.iter().sum::<f32>() / n;
        
        let mut variance = 0.0;
        let mut sum_cubed = 0.0;
        
        for &x in frame {
            let deviation = x - mean;
            variance += deviation * deviation;
            sum_cubed += deviation * deviation * deviation;
        }
        
        variance /= n;
        
        if variance > 1e-10 {
            (sum_cubed / n) / variance.powf(1.5)
        } else {
            0.0
        }
    }
    
    /// 计算自相关
    fn calculate_autocorrelation(&self, frame: &[f32], lag: usize) -> f32 {
        let n = frame.len();
        if lag >= n {
            return 0.0;
        }
        
        let mean = frame.iter().sum::<f32>() / n as f32;
        let mut numerator = 0.0;
        let mut denominator = 0.0;
        
        for i in 0..(n - lag) {
            let x1 = frame[i] - mean;
            let x2 = frame[i + lag] - mean;
            numerator += x1 * x2;
            denominator += x1 * x1;
        }
        
        if denominator > 1e-10 {
            numerator / denominator
        } else {
            0.0
        }
    }
    
    /// 计算振幅包络
    fn calculate_amplitude_envelope(&self, frame: &[f32], window_size: usize) -> Vec<f32> {
        if window_size == 0 || frame.is_empty() {
            return vec![];
        }
        
        let n = frame.len();
        let n_windows = ((n as f32) / (window_size as f32)).ceil() as usize;
        let mut envelope = Vec::with_capacity(n_windows);
        
        for i in 0..n_windows {
            let start = i * window_size;
            let end = std::cmp::min(start + window_size, n);
            if start >= n {
                break;
            }
            
            let mut max_amp: f32 = 0.0;
            for j in start..end {
                let abs_val: f32 = frame[j].abs();
                max_amp = max_amp.max(abs_val);
            }
            
            envelope.push(max_amp);
        }
        
        envelope
    }
    
    /// 计算峰值因子
    fn calculate_crest_factor(&self, frame: &[f32]) -> f32 {
        let rms = self.calculate_rms(frame);
        if rms > 1e-10 {
            frame.iter().map(|&x| x.abs()).fold(0.0f32, |a, b| a.max(b)) / rms
        } else {
            0.0
        }
    }
    
    /// 提取时域特征
    #[cfg(feature = "multimodal")]
    pub fn extract(&self, samples: &[f32]) -> Result<Array2<f32>> {
        // 1. 分帧
        let frames = frame_signal(samples, self.frame_size, self.hop_size);
        if frames.is_empty() {
            return Err(Error::data("音频太短，无法提取足够的帧".to_string()));
        }
        
        // 2. 确定特征维度（每种特征一个维度）
        let num_features = self.features.len();
        let mut result = Array2::zeros((frames.len(), num_features));
        
        // 3. 对每一帧提取特征
        for (i, frame) in frames.iter().enumerate() {
            let mean = frame.iter().sum::<f32>() / frame.len() as f32;
            
            // 计算标准差
            let squared_diff_sum = frame.iter().map(|&x| (x - mean).powi(2)).sum::<f32>();
            let std_dev = (squared_diff_sum / frame.len() as f32).sqrt();
            
            // 提取各种特征
            for (j, feature) in self.features.iter().enumerate() {
                let value = match feature {
                    TemporalFeatureType::Mean => mean,
                    TemporalFeatureType::StdDev => std_dev,
                    TemporalFeatureType::Kurtosis => self.calculate_kurtosis(frame),
                    TemporalFeatureType::Skewness => self.calculate_skewness(frame),
                    TemporalFeatureType::ZeroCrossingRate => self.calculate_zcr(frame),
                    TemporalFeatureType::Max => frame.iter().fold(f32::MIN, |a, &b| a.max(b)),
                    TemporalFeatureType::Min => frame.iter().fold(f32::MAX, |a, &b| a.min(b)),
                    TemporalFeatureType::DynamicRange => {
                        let max = frame.iter().fold(f32::MIN, |a, &b| a.max(b));
                        let min = frame.iter().fold(f32::MAX, |a, &b| a.min(b));
                        max - min
                    },
                    TemporalFeatureType::RootMeanSquare => self.calculate_rms(frame),
                    TemporalFeatureType::Autocorrelation => self.calculate_autocorrelation(frame, 1),
                    TemporalFeatureType::AmplitudeEnvelope => {
                        // 对于AmplitudeEnvelope，我们只取平均振幅
                        let envelope = self.calculate_amplitude_envelope(frame, 16);
                        envelope.iter().sum::<f32>() / envelope.len().max(1) as f32
                    },
                    TemporalFeatureType::CrestFactor => self.calculate_crest_factor(frame),
                };
                
                result[[i, j]] = value;
            }
        }
        
        // 4. 如果需要，进行归一化
        if self.normalize {
            // 对每列特征单独归一化
            for j in 0..num_features {
                let column = result.column(j);
                
                // 找出最大和最小值
                let mut min_val = f32::MAX;
                let mut max_val = f32::MIN;
                
                for &val in column {
                    if val < min_val {
                        min_val = val;
                    }
                    if val > max_val {
                        max_val = val;
                    }
                }
                
                // 执行最大-最小归一化
                let range = max_val - min_val;
                if range > 1e-10 {
                    for i in 0..frames.len() {
                        result[[i, j]] = (result[[i, j]] - min_val) / range;
                    }
                }
            }
        }
        
        Ok(result)
    }
} 