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

//! 音频滤波器工具模块
//!
//! 提供各种窗函数和滤波器实现，包括：
//! - 窗函数 (汉宁窗、汉明窗、布莱克曼窗等)
//! - 梅尔滤波器组
//! - 色度滤波器组

use ndarray::{Array1, Array2};
use std::f64::consts::PI;
use crate::error::Result;
use crate::data::multimodal::extractors::audio::conversion::{hz_to_mel, mel_to_hz};

/// 应用窗函数到输入信号
/// 
/// # 参数
/// * `frame` - 输入音频帧
/// * `window_type` - 窗函数类型："hann", "hamming", "blackman", "rectangular"
/// 
/// # 返回值
/// * `Array1<f64>` - 应用窗函数后的音频帧
pub fn apply_window(frame: &Array1<f64>, window_type: &str) -> Array1<f64> {
    let n = frame.len();
    if n == 0 {
        return Array1::zeros(0);
    }

    let window = match window_type.to_lowercase().as_str() {
        "hann" => {
            Array1::from_iter((0..n).map(|i| 0.5 * (1.0 - (2.0 * PI * i as f64 / (n as f64 - 1.0)).cos())))
        },
        "hamming" => {
            Array1::from_iter((0..n).map(|i| 0.54 - 0.46 * (2.0 * PI * i as f64 / (n as f64 - 1.0)).cos()))
        },
        "blackman" => {
            Array1::from_iter((0..n).map(|i| {
                let x = i as f64 / (n as f64 - 1.0);
                0.42 - 0.5 * (2.0 * PI * x).cos() + 0.08 * (4.0 * PI * x).cos()
            }))
        },
        "rectangular" | _ => {
            Array1::ones(n)
        }
    };

    frame * &window
}

/// 应用窗函数到输入信号(f32版本)
///
/// # 参数
/// * `frame` - 输入音频帧
/// * `window_type` - 窗函数类型
///
/// # 返回值
/// * 应用窗函数后的音频帧
pub fn apply_window_f32(frame: &[f32], window_type: &str) -> Vec<f32> {
    if frame.is_empty() {
        return vec![];
    }
    
    let n = frame.len();
    let window = match window_type.to_lowercase().as_str() {
        "hann" => hanning_window(n),
        "hamming" => hamming_window(n),
        "blackman" => blackman_window(n),
        "rectangular" | _ => vec![1.0; n],
    };
    
    frame.iter()
        .zip(window.iter())
        .map(|(&s, &w)| s * w)
        .collect()
}

/// 创建汉宁窗
///
/// # 参数
/// * `size` - 窗函数长度
///
/// # 返回值
/// * 窗函数系数
pub fn hanning_window(size: usize) -> Vec<f32> {
    if size <= 1 {
        return vec![1.0; size];
    }
    
    let size_minus_1 = (size - 1) as f32;
    (0..size)
        .map(|i| 0.5 * (1.0 - (2.0 * std::f32::consts::PI * i as f32 / size_minus_1).cos()))
        .collect()
}

/// 创建汉明窗
///
/// # 参数
/// * `size` - 窗函数长度
///
/// # 返回值
/// * 窗函数系数
pub fn hamming_window(size: usize) -> Vec<f32> {
    if size <= 1 {
        return vec![1.0; size];
    }
    
    let size_minus_1 = (size - 1) as f32;
    (0..size)
        .map(|i| 0.54 - 0.46 * (2.0 * std::f32::consts::PI * i as f32 / size_minus_1).cos())
        .collect()
}

/// 创建布莱克曼窗
///
/// # 参数
/// * `size` - 窗函数长度
///
/// # 返回值
/// * 窗函数系数
pub fn blackman_window(size: usize) -> Vec<f32> {
    if size <= 1 {
        return vec![1.0; size];
    }
    
    let size_minus_1 = (size - 1) as f32;
    (0..size)
        .map(|i| {
            let x = i as f32 / size_minus_1;
            0.42 - 0.5 * (2.0 * std::f32::consts::PI * x).cos() + 
                  0.08 * (4.0 * std::f32::consts::PI * x).cos()
        })
        .collect()
}

/// 计算梅尔滤波器组
///
/// # 参数
/// * `n_mels` - 梅尔滤波器数量
/// * `n_fft` - FFT大小
/// * `sample_rate` - 采样率
/// * `fmin` - 最低频率
/// * `fmax` - 最高频率
///
/// # 返回值
/// * `Array2<f64>` - 梅尔滤波器组矩阵
pub fn mel_filterbank(n_mels: usize, n_fft: usize, sample_rate: u32, 
                      fmin: f64, fmax: f64) -> Array2<f64> {
    // 检查参数
    if n_mels == 0 || n_fft == 0 || sample_rate == 0 {
        return Array2::zeros((0, 0));
    }
    
    let fmax_actual = if fmax > 0.0 { fmax } else { sample_rate as f64 / 2.0 };
    
    // 将线性频率转换为梅尔频率
    let min_mel = 1127.0 * (1.0 + fmin / 700.0).ln();
    let max_mel = 1127.0 * (1.0 + fmax_actual / 700.0).ln();
    
    // 在梅尔刻度上均匀分布滤波器
    let mel_points: Vec<f64> = (0..=n_mels+1)
        .map(|i| min_mel + (max_mel - min_mel) * i as f64 / (n_mels as f64 + 1.0))
        .collect();
    
    // 将梅尔刻度转换回赫兹
    let hz_points: Vec<f64> = mel_points
        .iter()
        .map(|&mel| 700.0 * ((mel / 1127.0).exp() - 1.0))
        .collect();
    
    // 将赫兹转换为FFT bin
    let bin_indices: Vec<usize> = hz_points
        .iter()
        .map(|&hz| ((n_fft as f64 + 1.0) * hz / sample_rate as f64).round() as usize)
        .map(|idx| idx.min(n_fft / 2))
        .collect();
    
    // 创建滤波器组
    let mut filterbank = Array2::zeros((n_mels, n_fft / 2 + 1));
    
    // 构建三角滤波器
    for i in 0..n_mels {
        for j in bin_indices[i]..=bin_indices[i+2] {
            if j >= bin_indices[i+2] {
                break;
            }
            
            if j < bin_indices[i+1] {
                // 上升边
                filterbank[[i, j]] = (j as f64 - bin_indices[i] as f64) / 
                                      (bin_indices[i+1] as f64 - bin_indices[i] as f64);
            } else {
                // 下降边
                filterbank[[i, j]] = (bin_indices[i+2] as f64 - j as f64) / 
                                     (bin_indices[i+2] as f64 - bin_indices[i+1] as f64);
            }
        }
        
        // 归一化滤波器（可选）
        let filter_norm: f64 = filterbank.slice(ndarray::s![i, ..]).sum();
        if filter_norm > 0.0 {
            for j in 0..=n_fft/2 {
                filterbank[[i, j]] /= filter_norm;
            }
        }
    }
    
    filterbank
}

/// 创建梅尔滤波器组(f32版本)
///
/// # 参数
/// * `n_freqs` - 频率点数
/// * `n_mels` - 梅尔滤波器数量
/// * `sample_rate` - 采样率
/// * `f_min` - 最低频率
/// * `f_max` - 最高频率
///
/// # 返回值
/// * 梅尔滤波器组矩阵
pub fn mel_filter_bank(
    n_freqs: usize, 
    n_mels: usize, 
    sample_rate: f32, 
    f_min: f32, 
    f_max: f32
) -> Result<Array2<f32>> {
    if n_freqs == 0 || n_mels == 0 || sample_rate <= 0.0 {
        return Err(crate::error::Error::data("无效的参数".to_string()));
    }
    
    // 设置频率上下限
    let fmax = if f_max <= 0.0 { sample_rate / 2.0 } else { f_max };
    let fmin = if f_min < 0.0 { 0.0 } else { f_min };
    
    if fmax <= fmin {
        return Err(crate::error::Error::data("最大频率必须大于最小频率".to_string()));
    }
    
    // 创建梅尔刻度上的均匀点
    let min_mel = hz_to_mel(fmin as f64);
    let max_mel = hz_to_mel(fmax as f64);
    
    let mut mel_points = Vec::with_capacity(n_mels + 2);
    for i in 0..=n_mels+1 {
        let mel = min_mel + i as f64 * (max_mel - min_mel) / (n_mels as f64 + 1.0);
        mel_points.push(mel);
    }
    
    // 转换回Hz
    let mut hz_points = Vec::with_capacity(mel_points.len());
    for &mel in &mel_points {
        hz_points.push(mel_to_hz(mel));
    }
    
    // 转换为FFT bin
    let mut bin = Vec::with_capacity(hz_points.len());
    for &hz in &hz_points {
        let bin_idx = (n_freqs as f64 * hz / (sample_rate as f64 / 2.0)).floor() as usize;
        bin.push(bin_idx.min(n_freqs - 1));
    }
    
    // 创建滤波器组
    let mut weights = Array2::zeros((n_mels, n_freqs));
    
    for i in 0..n_mels {
        for j in bin[i]..=bin[i+2] {
            if j >= n_freqs {
                break;
            }
            
            if j <= bin[i+1] {
                // 上升边
                if bin[i+1] > bin[i] {
                    weights[[i, j]] = (j as f32 - bin[i] as f32) / (bin[i+1] as f32 - bin[i] as f32);
                }
            } else {
                // 下降边
                if bin[i+2] > bin[i+1] {
                    weights[[i, j]] = (bin[i+2] as f32 - j as f32) / (bin[i+2] as f32 - bin[i+1] as f32);
                }
            }
        }
        
        // 归一化（可选）
        let sum: f32 = weights.row(i).sum();
        if sum > 0.0 {
            for j in 0..n_freqs {
                weights[[i, j]] /= sum;
            }
        }
    }
    
    Ok(weights)
}

/// 创建色度滤波器组
///
/// # 参数
/// * `n_freqs` - 频率点数
/// * `n_chroma` - 色度数量 (通常为12)
/// * `sample_rate` - 采样率
///
/// # 返回值
/// * 色度滤波器组矩阵
pub fn create_chroma_filterbank(
    n_freqs: usize, 
    n_chroma: usize, 
    sample_rate: u32
) -> Result<Array2<f32>> {
    if n_freqs == 0 || n_chroma == 0 || sample_rate == 0 {
        return Err(crate::error::Error::data("无效的参数".to_string()));
    }
    
    // 创建频率数组 (Hz)
    let mut frequencies = Vec::with_capacity(n_freqs);
    for i in 0..n_freqs {
        let freq = i as f32 * sample_rate as f32 / (2.0 * (n_freqs - 1) as f32);
        frequencies.push(freq);
    }
    
    // 创建色度滤波器组
    let mut chroma_filters = Array2::zeros((n_chroma, n_freqs));
    
    // 参考频率 (C0 = 16.35Hz)
    let ref_freq = 16.35;
    
    // 对每个频率点
    for i in 0..n_freqs {
        let freq = frequencies[i];
        if freq <= 0.0 {
            continue;
        }
        
        // 计算音高类别 (色度) - 0=C, 1=C#, ..., 11=B
        let cents = 1200.0 * (freq / ref_freq).log2();
        let chroma = ((cents % 1200.0) / 100.0).round() as usize % n_chroma;
        
        // 使用高斯滤波器对每个色度赋权
        for j in 0..n_chroma {
            let mut distance = (j as isize - chroma as isize).abs();
            // 环绕处理 (距离不超过半个八度)
            if distance > n_chroma as isize / 2 {
                distance = n_chroma as isize - distance;
            }
            
            // 高斯权重 (sigma=1)
            let weight = (-0.5 * (distance as f32).powi(2)).exp();
            chroma_filters[[j, i]] += weight;
        }
    }
    
    // 归一化
    for i in 0..n_chroma {
        let sum = chroma_filters.row(i).sum();
        if sum > 0.0 {
            for j in 0..n_freqs {
                chroma_filters[[i, j]] /= sum;
            }
        }
    }
    
    Ok(chroma_filters)
}

/// 预增强处理，增强高频部分
pub fn preemphasis(signal: &[f32], coef: f32) -> Vec<f32> {
    if signal.is_empty() {
        return Vec::new();
    }
    
    let mut result = Vec::with_capacity(signal.len());
    result.push(signal[0]); // 第一个样本保持不变
    
    for i in 1..signal.len() {
        result.push(signal[i] - coef * signal[i-1]);
    }
    
    result
}

/// 归一化音频信号
pub fn normalize_audio(signal: &[f32]) -> Vec<f32> {
    if signal.is_empty() {
        return Vec::new();
    }
    
    // 查找最大绝对值
    let max_abs = signal.iter()
        .map(|&x| x.abs())
        .fold(0.0f32, |a, b| a.max(b));
    
    if max_abs == 0.0 {
        return signal.to_vec();
    }
    
    // 归一化
    signal.iter().map(|&x| x / max_abs).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    
    #[test]
    fn test_apply_window() {
        let frame = Array1::from_vec(vec![1.0, 1.0, 1.0, 1.0]);
        
        // 汉宁窗
        let windowed = apply_window(&frame, "hann");
        assert_eq!(windowed.len(), 4);
        assert_relative_eq!(windowed[0], 0.0, epsilon = 1e-6);
        assert_relative_eq!(windowed[2], 1.0, epsilon = 1e-6);
        
        // 汉明窗
        let windowed = apply_window(&frame, "hamming");
        assert_eq!(windowed.len(), 4);
        assert_relative_eq!(windowed[0], 0.08, epsilon = 1e-2);
    }
    
    #[test]
    fn test_mel_filterbank() {
        let n_mels = 10;
        let n_fft = 512;
        let sample_rate = 16000;
        
        let filterbank = mel_filterbank(n_mels, n_fft, sample_rate, 0.0, 8000.0);
        
        assert_eq!(filterbank.shape(), &[n_mels, n_fft / 2 + 1]);
        
        // 检查每个滤波器的和是否接近1.0
        for i in 0..n_mels {
            let sum = filterbank.slice(ndarray::s![i, ..]).sum();
            assert!(sum > 0.0 && sum <= 1.01, "Filter {} sum is {}", i, sum);
        }
    }
    
    #[test]
    fn test_hanning_window() {
        let window = hanning_window(5);
        assert_eq!(window.len(), 5);
        assert_relative_eq!(window[0], 0.0, epsilon = 1e-6);
        assert_relative_eq!(window[2], 1.0, epsilon = 1e-6);
        assert_relative_eq!(window[4], 0.0, epsilon = 1e-6);
    }
} 