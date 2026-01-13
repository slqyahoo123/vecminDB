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

//! 音频时域分析模块
//!
//! 提供各种时域分析功能，包括零交叉率、信号能量、
//! 均方根、峰值因子、峰度、偏度等

use ndarray::Array1;

/// 计算零交叉率
///
/// # 参数
/// * `frame` - 输入音频帧
///
/// # 返回值
/// 返回零交叉率
pub fn zero_crossing_rate(frame: &[f64]) -> f64 {
    if frame.len() < 2 {
        return 0.0;
    }
    
    let mut crossings = 0;
    
    for i in 1..frame.len() {
        if frame[i-1] * frame[i] < 0.0 {
            crossings += 1;
        }
    }
    
    crossings as f64 / (frame.len() as f64 - 1.0)
}

/// 计算信号能量
///
/// # 参数
/// * `frame` - 输入音频帧
/// * `use_db` - 是否使用分贝刻度
/// * `min_db` - 最小分贝值（仅当use_db为true时有效）
///
/// # 返回值
/// 返回信号能量
pub fn signal_energy(frame: &[f64], use_db: bool, min_db: Option<f64>) -> f64 {
    if frame.is_empty() {
        return 0.0;
    }
    
    let energy: f64 = frame.iter().map(|&x| x.powi(2)).sum();
    
    if use_db {
        let min_db_val = min_db.unwrap_or(-80.0);
        if energy > 0.0 {
            let db = 10.0 * energy.log10();
            db.max(min_db_val)
        } else {
            min_db_val
        }
    } else {
        energy
    }
}

/// 计算均方根值
///
/// # 参数
/// * `frame` - 输入音频帧
///
/// # 返回值
/// 返回均方根值
pub fn root_mean_square(frame: &[f64]) -> f64 {
    if frame.is_empty() {
        return 0.0;
    }
    
    let mean_square: f64 = frame.iter().map(|&x| x.powi(2)).sum::<f64>() / frame.len() as f64;
    mean_square.sqrt()
}

/// 计算峰值因子
///
/// # 参数
/// * `frame` - 输入音频帧
///
/// # 返回值
/// 返回峰值因子
pub fn crest_factor(frame: &[f64]) -> f64 {
    if frame.is_empty() {
        return 0.0;
    }
    
    let rms = root_mean_square(frame);
    if rms <= 1e-10 {
        return 0.0;
    }
    
    let peak: f64 = frame.iter().fold(0.0_f64, |max, &x| max.max(x.abs()));
    peak / rms
}

/// 计算峰度
///
/// # 参数
/// * `frame` - 输入音频帧
///
/// # 返回值
/// 返回峰度值
pub fn kurtosis(frame: &[f64]) -> f64 {
    if frame.len() < 2 {
        return 0.0;
    }
    
    let n = frame.len() as f64;
    let mean: f64 = frame.iter().sum::<f64>() / n;
    
    let variance = frame.iter()
        .map(|&x| (x - mean).powi(2))
        .sum::<f64>() / n;
    
    if variance < 1e-10 {
        return 0.0;
    }
    
    let sigma = variance.sqrt();
    
    let kurt = frame.iter()
        .map(|&x| (x - mean).powi(4))
        .sum::<f64>() / (n * sigma.powi(4)) - 3.0;
    
    kurt
}

/// 计算偏度
///
/// # 参数
/// * `frame` - 输入音频帧
///
/// # 返回值
/// 返回偏度值
pub fn skewness(frame: &[f64]) -> f64 {
    if frame.len() < 2 {
        return 0.0;
    }
    
    let n = frame.len() as f64;
    let mean: f64 = frame.iter().sum::<f64>() / n;
    
    let variance = frame.iter()
        .map(|&x| (x - mean).powi(2))
        .sum::<f64>() / n;
    
    if variance < 1e-10 {
        return 0.0;
    }
    
    let sigma = variance.sqrt();
    
    let skew = frame.iter()
        .map(|&x| (x - mean).powi(3))
        .sum::<f64>() / (n * sigma.powi(3));
    
    skew
}

/// 计算信号包络
///
/// # 参数
/// * `frame` - 输入音频帧
/// * `window_size` - 窗口大小（用于移动平均）
///
/// # 返回值
/// 返回信号包络
pub fn envelope(frame: &[f64], window_size: usize) -> Array1<f64> {
    if frame.is_empty() {
        return Array1::zeros(0);
    }
    
    let n = frame.len();
    
    // 计算信号的绝对值
    let abs_signal: Vec<f64> = frame.iter().map(|&x| x.abs()).collect();
    
    // 移动平均
    let win_size = window_size.min(n);
    let mut env = Vec::with_capacity(n);
    
    for i in 0..n {
        let start = if i < win_size / 2 {
            0
        } else {
            i - win_size / 2
        };
        
        let end = (i + win_size / 2 + 1).min(n);
        
        let window_mean: f64 = abs_signal[start..end].iter().sum::<f64>() / (end - start) as f64;
        env.push(window_mean);
    }
    
    Array1::from(env)
}

/// 计算时域特征
///
/// # 参数
/// * `frame` - 输入音频帧
///
/// # 返回值
/// 返回包含多个时域特征的向量
pub fn temporal_features(frame: &[f64]) -> Array1<f64> {
    // 提取7个时域特征：
    // 1. 均值
    // 2. 标准差
    // 3. 均方根值
    // 4. 零交叉率
    // 5. 峰度
    // 6. 偏度
    // 7. 峰值因子
    let mut features = Vec::with_capacity(7);
    
    if frame.is_empty() {
        return Array1::zeros(7);
    }
    
    // 1. 均值
    let mean: f64 = frame.iter().sum::<f64>() / frame.len() as f64;
    features.push(mean);
    
    // 2. 标准差
    let variance = frame.iter()
        .map(|&x| (x - mean).powi(2))
        .sum::<f64>() / frame.len() as f64;
    let std_dev = variance.sqrt();
    features.push(std_dev);
    
    // 3. 均方根值
    let rms = root_mean_square(frame);
    features.push(rms);
    
    // 4. 零交叉率
    let zcr = zero_crossing_rate(frame);
    features.push(zcr);
    
    // 5. 峰度
    let kurt = kurtosis(frame);
    features.push(kurt);
    
    // 6. 偏度
    let skew = skewness(frame);
    features.push(skew);
    
    // 7. 峰值因子
    let crest = crest_factor(frame);
    features.push(crest);
    
    Array1::from(features)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::multimodal::extractors::audio::generators::generate_sine_wave;
    use approx::assert_relative_eq;
    
    #[test]
    fn test_zero_crossing_rate() {
        // 生成1秒、440Hz的正弦波，采样率8000Hz
        let frequency = 440.0;
        let duration = 1.0;
        let sample_rate = 8000;
        let signal = generate_sine_wave(frequency, duration, sample_rate, None, None);
        
        // 计算零交叉率
        let zcr = zero_crossing_rate(&signal);
        
        // 理论上，一个正弦波的零交叉率应该是2*f/fs
        let expected_zcr = 2.0 * frequency / sample_rate as f64;
        
        // 允许一定误差
        assert_relative_eq!(zcr, expected_zcr, epsilon = 0.01);
    }
    
    #[test]
    fn test_signal_energy() {
        // 创建测试信号
        let signal = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        
        // 计算能量
        let energy = signal_energy(&signal, false, None);
        
        // 能量应为平方和
        let expected_energy = signal.iter().map(|&x| x.powi(2)).sum::<f64>();
        
        assert_relative_eq!(energy, expected_energy, epsilon = 1e-6);
        
        // 测试dB刻度
        let energy_db = signal_energy(&signal, true, None);
        let expected_db = 10.0 * expected_energy.log10();
        
        assert_relative_eq!(energy_db, expected_db, epsilon = 1e-6);
    }
    
    #[test]
    fn test_root_mean_square() {
        // 创建测试信号
        let signal = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        
        // 计算RMS
        let rms = root_mean_square(&signal);
        
        // 计算期望RMS
        let mean_square = signal.iter().map(|&x| x.powi(2)).sum::<f64>() / signal.len() as f64;
        let expected_rms = mean_square.sqrt();
        
        assert_relative_eq!(rms, expected_rms, epsilon = 1e-6);
    }
    
    #[test]
    fn test_temporal_features() {
        // 生成测试信号
        let frequency = 440.0;
        let duration = 1.0;
        let sample_rate = 8000;
        let signal = generate_sine_wave(frequency, duration, sample_rate, None, None);
        
        // 提取时域特征
        let features = temporal_features(&signal);
        
        // 正弦波的特征：均值应接近0，RMS应接近0.707
        assert_relative_eq!(features[0], 0.0, epsilon = 0.01); // 均值
        assert_relative_eq!(features[2], 1.0 / 2.0_f64.sqrt(), epsilon = 0.01); // RMS = 1/sqrt(2) 对于幅值为1的正弦波
        
        // 应该有7个特征
        assert_eq!(features.len(), 7);
    }
} 