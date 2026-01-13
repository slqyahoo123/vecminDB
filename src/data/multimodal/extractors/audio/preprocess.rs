// Copyright (c) 2023 Vecmind Labs. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//! 音频预处理工具模块
//!
//! 提供音频信号预处理功能，包括：
//! - 归一化
//! - 预加重处理
//! - 信号分帧
//! - 特征增强

use ndarray::{Array1, Array2, s};
use crate::error::Result;
use crate::data::multimodal::extractors::audio::error::AudioError;

/// 将音频数据归一化到 [-1.0, 1.0] 范围
///
/// # 参数
/// * `audio_data` - 要归一化的音频数据
///
/// # 返回值
/// * `Array1<f64>` - 归一化后的音频数据
///
/// # 示例
/// ```
/// use ndarray::Array1;
/// use vecmind::data::multimodal::extractors::audio::preprocess::normalize_audio;
///
/// let audio_data = Array1::from_vec(vec![0.5, -0.3, 0.8, -0.7]);
/// let normalized = normalize_audio(&audio_data);
/// ```
pub fn normalize_audio(audio_data: &Array1<f64>) -> Array1<f64> {
    if audio_data.is_empty() {
        return Array1::zeros(0);
    }

    let max_abs = audio_data.iter()
        .map(|&x| x.abs())
        .fold(f64::MIN, |a, b| a.max(b));

    if max_abs == 0.0 {
        return audio_data.clone();
    }

    audio_data.mapv(|x| x / max_abs)
}

/// 对音频数据进行预加重处理
///
/// 预加重通过应用高通滤波器增强高频部分，通常在提取MFCC特征前使用
///
/// # 参数
/// * `audio_data` - 输入音频数据
/// * `coeff` - 预加重系数，通常为0.95-0.97
///
/// # 返回值
/// * `Array1<f64>` - 预加重处理后的音频数据
pub fn preemphasis(audio_data: &Array1<f64>, coeff: f64) -> Array1<f64> {
    if audio_data.is_empty() {
        return Array1::zeros(0);
    }

    let mut result = Array1::zeros(audio_data.len());
    result[0] = audio_data[0];

    for i in 1..audio_data.len() {
        result[i] = audio_data[i] - coeff * audio_data[i - 1];
    }

    result
}

/// 预加重处理（f32版本）
///
/// # 参数
/// * `signal` - 输入音频信号
/// * `coeff` - 预加重系数
///
/// # 返回值
/// * 预加重处理后的信号
pub fn preemphasis_f32(signal: &[f32], coeff: f32) -> Vec<f32> {
    if signal.is_empty() {
        return Vec::new();
    }
    
    let mut result = Vec::with_capacity(signal.len());
    result.push(signal[0]);
    
    for i in 1..signal.len() {
        result.push(signal[i] - coeff * signal[i-1]);
    }
    
    result
}

/// 将音频信号分帧
///
/// # 参数
/// * `signal` - 输入音频信号
/// * `frame_length` - 每帧的样本数
/// * `hop_length` - 相邻帧之间的样本数
///
/// # 返回值
/// * `Array2<f64>` - 分帧后的音频数据，每行是一帧
pub fn frame_signal(signal: &Array1<f64>, frame_length: usize, hop_length: usize) -> Array2<f64> {
    if signal.is_empty() || frame_length == 0 {
        return Array2::zeros((0, 0));
    }

    let signal_length = signal.len();
    let num_frames = if signal_length <= frame_length {
        1
    } else {
        1 + (signal_length - frame_length) / hop_length
    };

    let mut frames = Array2::zeros((num_frames, frame_length));
    
    for i in 0..num_frames {
        let start = i * hop_length;
        let end = start + frame_length;
        
        if end <= signal_length {
            frames.slice_mut(s![i, ..]).assign(&signal.slice(s![start..end]));
        } else {
            let valid_samples = signal_length - start;
            frames.slice_mut(s![i, ..valid_samples]).assign(&signal.slice(s![start..]));
            // 不足的部分填充0
        }
    }
    
    frames
}

/// 将信号分帧(f32版本)
///
/// # 参数
/// * `signal` - 输入信号
/// * `frame_length` - 帧长度
/// * `hop_length` - 帧移
///
/// # 返回值
/// * 分帧后的二维数组
pub fn frame_signal_f32(signal: &[f32], frame_length: usize, hop_length: usize) -> Result<Array2<f32>> {
    if signal.is_empty() {
        return Err(AudioError::InvalidInput("输入信号为空".to_string()).into());
    }
    
    if frame_length == 0 {
        return Err(AudioError::InvalidParameter("帧长度不能为0".to_string()).into());
    }
    
    if hop_length == 0 {
        return Err(AudioError::InvalidParameter("帧移不能为0".to_string()).into());
    }
    
    let signal_length = signal.len();
    let n_frames = (signal_length - frame_length) / hop_length + 1;
    
    if n_frames == 0 {
        return Err(AudioError::ProcessingError("无法生成帧，信号长度过短".to_string()).into());
    }
    
    let mut frames = Array2::zeros((n_frames, frame_length));
    
    for i in 0..n_frames {
        let start = i * hop_length;
        let end = start + frame_length;
        
        if end > signal_length {
            // 如果帧超出信号范围，做零填充
            let valid_length = signal_length - start;
            for j in 0..valid_length {
                frames[[i, j]] = signal[start + j];
            }
            // 其余部分已经是0
        } else {
            for j in 0..frame_length {
                frames[[i, j]] = signal[start + j];
            }
        }
    }
    
    Ok(frames)
}

/// 增强特征(增加delta特征等)
///
/// # 参数
/// * `features` - 输入特征矩阵
/// * `normalize` - 是否进行归一化
///
/// # 返回值
/// * 处理后的特征矩阵
pub fn enhance_features(features: &mut Array2<f32>, normalize: bool) -> Result<()> {
    if features.is_empty() {
        return Err(AudioError::InvalidInput("输入特征矩阵为空".to_string()).into());
    }
    
    // 如果需要归一化
    if normalize {
        let (n_features, n_frames) = features.dim();
        
        // 对每个特征维度进行归一化
        for i in 0..n_features {
            let mut row = features.slice_mut(s![i, ..]);
            
            // 计算均值
            let mean = row.sum() / n_frames as f32;
            
            // 计算标准差
            let std_dev = (row.mapv(|x| (x - mean).powi(2)).sum() / n_frames as f32).sqrt();
            
            if std_dev > 1e-10 {
                // 标准化: (x - mean) / std
                row.mapv_inplace(|x| (x - mean) / std_dev);
            }
        }
    }
    
    Ok(())
}

/// 将幅度转换为分贝
///
/// # 参数
/// * `magnitude` - 幅度值
/// * `min_db` - 最小dB值，用于剪裁
///
/// # 返回值
/// * 对应的分贝值
pub fn amplitude_to_db(magnitude: f32, min_db: f32) -> f32 {
    if magnitude < 1e-10 {
        return min_db;
    }
    
    let db = 20.0 * magnitude.log10();
    db.max(min_db)
}

/// 将线性幅度转换为分贝表示
///
/// # 参数
/// * `magnitude` - 线性幅度值
/// * `ref_value` - 参考值，默认为1.0
/// * `min_db` - 最小dB值
///
/// # 返回值
/// * 分贝值
pub fn linear_to_db(magnitude: f64, ref_value: Option<f64>, min_db: Option<f64>) -> f64 {
    let ref_val = ref_value.unwrap_or(1.0);
    let min_db_val = min_db.unwrap_or(-80.0);
    
    if magnitude < 1e-10 {
        return min_db_val;
    }
    
    let db = 20.0 * (magnitude / ref_val).log10();
    db.max(min_db_val)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    
    #[test]
    fn test_normalize_audio() {
        let data = Array1::from_vec(vec![1.0, -2.0, 3.0, -4.0]);
        let normalized = normalize_audio(&data);
        assert_relative_eq!(normalized[3], -1.0);
        assert_relative_eq!(normalized[2], 0.75);
    }
    
    #[test]
    fn test_preemphasis() {
        let data = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0]);
        let processed = preemphasis(&data, 0.97);
        assert_relative_eq!(processed[0], 1.0);
        assert_relative_eq!(processed[1], 2.0 - 0.97 * 1.0);
    }
    
    #[test]
    fn test_frame_signal() {
        let data = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
        let frames = frame_signal(&data, 4, 2);
        assert_eq!(frames.shape(), &[3, 4]);
        assert_relative_eq!(frames[[0, 0]], 1.0);
        assert_relative_eq!(frames[[1, 0]], 3.0);
        assert_relative_eq!(frames[[2, 0]], 5.0);
    }
    
    #[test]
    fn test_amplitude_to_db() {
        assert_relative_eq!(amplitude_to_db(1.0, -80.0), 0.0);
        assert_relative_eq!(amplitude_to_db(10.0, -80.0), 20.0);
        assert_relative_eq!(amplitude_to_db(0.1, -80.0), -20.0);
        assert_eq!(amplitude_to_db(0.0, -80.0), -80.0);
    }
} 