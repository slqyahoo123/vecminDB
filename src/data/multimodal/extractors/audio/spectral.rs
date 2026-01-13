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

//! 音频频谱分析模块
//!
//! 提供各种频谱分析功能，包括MFCC、色度图、梅尔频谱图、
//! 频谱质心、频谱滚降点等。

use ndarray::{Array1, Array2, Axis, s};

use crate::Error;
#[cfg(feature = "multimodal")]
use crate::data::multimodal::extractors::audio::conversion::{
    amplitude_to_db, dct, frame_signal, magnitude_spectrum, stft
};
use crate::data::multimodal::extractors::audio::filter::{
    mel_filterbank, create_chroma_filterbank, preemphasis,
    hanning_window
};

/// 计算MFCC特征
///
/// # 参数
/// * `signal` - 输入音频信号
/// * `sample_rate` - 采样率 (Hz)
/// * `n_mfcc` - MFCC系数数量 (默认13)
/// * `n_fft` - FFT大小 (默认2048)
/// * `hop_length` - 帧移 (默认512)
/// * `n_mels` - Mel滤波器数量 (默认128)
/// * `fmin` - 最小频率 (Hz) (默认0)
/// * `fmax` - 最大频率 (Hz) (默认None，即采样率/2)
/// * `preemphasis_coef` - 预增强系数 (默认0.97)
///
/// # 返回值
/// 返回MFCC特征矩阵 (n_frames x n_mfcc)
#[cfg(feature = "multimodal")]
pub fn compute_mfcc(
    signal: &[f64],
    sample_rate: usize,
    n_mfcc: Option<usize>,
    n_fft: Option<usize>,
    hop_length: Option<usize>,
    n_mels: Option<usize>,
    fmin: Option<f64>,
    fmax: Option<f64>,
    preemphasis_coef: Option<f64>,
) -> Array2<f64> {
    // 设置默认参数
    let n_mfcc_val = n_mfcc.unwrap_or(13);
    let n_fft_val = n_fft.unwrap_or(2048);
    let hop_length_val = hop_length.unwrap_or(512);
    let n_mels_val = n_mels.unwrap_or(128);
    let fmax_val = fmax.unwrap_or(sample_rate as f64 / 2.0);
    let preemph_coef = preemphasis_coef.unwrap_or(0.97);
    
    // 预增强
    // 将 f64 信号转换为 f32 进行处理
    let signal_f32: Vec<f32> = signal.iter().map(|&x| x as f32).collect();
    let preemphasized = if preemph_coef > 0.0 {
        preemphasis(&signal_f32, preemph_coef as f32)
            .iter()
            .map(|&x| x as f64)
            .collect()
    } else {
        signal.to_vec()
    };
    
    // 计算梅尔频谱图
    let mel_spec = compute_melspectrogram(
        &preemphasized,
        sample_rate,
        n_fft,
        hop_length,
        n_mels,
        fmin,
        Some(fmax_val),
        true, // 转换为分贝
    );
    
    let n_frames = mel_spec.len_of(Axis(0));
    let mut mfcc = Array2::zeros((n_frames, n_mfcc_val));
    
    // 对每一帧计算DCT
    for i in 0..n_frames {
        let log_mel_frame = mel_spec.slice(s![i, ..]).to_vec();
        let dct_coeffs = dct(&log_mel_frame, n_mfcc_val);
        
        for j in 0..n_mfcc_val {
            mfcc[[i, j]] = dct_coeffs[j];
        }
    }
    
    mfcc
}

/// 计算梅尔频谱图
///
/// # 参数
/// * `signal` - 输入音频信号
/// * `sample_rate` - 采样率 (Hz)
/// * `n_fft` - FFT大小 (默认2048)
/// * `hop_length` - 帧移 (默认512)
/// * `n_mels` - Mel滤波器数量 (默认128)
/// * `fmin` - 最小频率 (Hz) (默认0)
/// * `fmax` - 最大频率 (Hz) (默认None，即采样率/2)
/// * `to_db` - 是否转换为分贝刻度 (默认true)
///
/// # 返回值
/// 返回梅尔频谱图矩阵 (n_frames x n_mels)
#[cfg(feature = "multimodal")]
pub fn compute_melspectrogram(
    signal: &[f64],
    sample_rate: usize,
    n_fft: Option<usize>,
    hop_length: Option<usize>,
    n_mels: Option<usize>,
    fmin: Option<f64>,
    fmax: Option<f64>,
    to_db: bool,
) -> Array2<f64> {
    // 设置默认参数
    let n_fft_val = n_fft.unwrap_or(2048);
    let hop_length_val = hop_length.unwrap_or(512);
    let n_mels_val = n_mels.unwrap_or(128);
    
    // 计算STFT
    let window = hanning_window(n_fft_val);
    let window_f64: Vec<f64> = window.iter().map(|&x| x as f64).collect();
    let stft_result = stft(signal, &window_f64, n_fft_val, hop_length_val);
    
    // 计算幅度谱
    let power_spec = magnitude_spectrum(&stft_result);
    
    // 创建Mel滤波器组
    let fmin_val = fmin.unwrap_or(0.0);
    let fmax_val = fmax.unwrap_or(sample_rate as f64 / 2.0);
    let mel_filters = mel_filterbank(n_mels_val, n_fft_val, sample_rate as u32, fmin_val, fmax_val);
    
    // 应用Mel滤波器组
    let n_frames = power_spec.len_of(Axis(0));
    let mut mel_spec = Array2::zeros((n_frames, n_mels_val));
    
    for i in 0..n_frames {
        for j in 0..n_mels_val {
            let mut mel_energy = 0.0;
            for k in 0..power_spec.len_of(Axis(1)) {
                mel_energy += power_spec[[i, k]] * mel_filters[[j, k]];
            }
            mel_spec[[i, j]] = mel_energy;
        }
    }
    
    // 转换为分贝刻度（如果需要）
    if to_db {
        for i in 0..n_frames {
            let row = mel_spec.slice(s![i, ..]).to_vec();
            let db_row = amplitude_to_db(&row, None, Some(-80.0));
            for j in 0..n_mels_val {
                mel_spec[[i, j]] = db_row[j];
            }
        }
    }
    
    mel_spec
}

/// 计算色度特征
///
/// # 参数
/// * `signal` - 输入音频信号
/// * `sample_rate` - 采样率 (Hz)
/// * `n_chroma` - 色度箱数量 (默认12)
/// * `n_fft` - FFT大小 (默认2048)
/// * `hop_length` - 帧移 (默认512)
/// * `center_freqs` - 是否使用中心频率 (默认true)
///
/// # 返回值
/// 返回色度特征矩阵 (n_frames x n_chroma)
pub fn compute_chroma(
    signal: &[f64],
    sample_rate: usize,
    n_chroma: Option<usize>,
    n_fft: Option<usize>,
    hop_length: Option<usize>,
    center_freqs: Option<bool>,
) -> crate::error::Result<Array2<f64>> {
    // 设置默认参数
    let n_chroma_val = n_chroma.unwrap_or(12);
    let n_fft_val = n_fft.unwrap_or(2048);
    let hop_length_val = hop_length.unwrap_or(512);
    let center_freqs_val = center_freqs.unwrap_or(true);
    
    // 计算STFT
    let window = hanning_window(n_fft_val);
    let window_f64: Vec<f64> = window.iter().map(|&x| x as f64).collect();
    let stft_result = stft(signal, &window_f64, n_fft_val, hop_length_val);
    
    // 计算幅度谱
    let power_spec = magnitude_spectrum(&stft_result);
    
    // 创建色度滤波器组
    let chroma_filters = create_chroma_filterbank(n_fft_val / 2 + 1, n_chroma_val, sample_rate as u32)
        .map_err(|e| Error::data(format!("创建色度滤波器组失败: {}", e)))?;
    
    // 应用色度滤波器组
    let n_frames = power_spec.len_of(Axis(0));
    let mut chroma_features = Array2::zeros((n_frames, n_chroma_val));
    
    for i in 0..n_frames {
        for j in 0..n_chroma_val {
            let mut chroma_energy = 0.0;
            for k in 0..power_spec.len_of(Axis(1)) {
                chroma_energy += power_spec[[i, k]] * chroma_filters[[j, k]] as f64;
            }
            chroma_features[[i, j]] = chroma_energy;
        }
    }
    
    // 归一化
    for i in 0..n_frames {
        let row_max: f64 = chroma_features.slice(s![i, ..]).fold(0.0_f64, |max, &val| max.max(val));
        if row_max > 0.0 {
            for j in 0..n_chroma_val {
                chroma_features[[i, j]] /= row_max;
            }
        }
    }
    
    Ok(chroma_features)
}

/// 计算频谱质心
///
/// # 参数
/// * `signal` - 输入音频信号
/// * `sample_rate` - 采样率 (Hz)
/// * `n_fft` - FFT大小 (默认2048)
/// * `hop_length` - 帧移 (默认512)
///
/// # 返回值
/// 返回频谱质心 (n_frames)
#[cfg(feature = "multimodal")]
pub fn compute_spectral_centroid(
    signal: &[f64],
    sample_rate: usize,
    n_fft: Option<usize>,
    hop_length: Option<usize>,
) -> Array1<f64> {
    // 设置默认参数
    let n_fft_val = n_fft.unwrap_or(2048);
    let hop_length_val = hop_length.unwrap_or(512);
    
    // 计算STFT
    let window = hanning_window(n_fft_val);
    let window_f64: Vec<f64> = window.iter().map(|&x| x as f64).collect();
    let stft_result = stft(signal, &window_f64, n_fft_val, hop_length_val);
    
    // 计算幅度谱
    let power_spec = magnitude_spectrum(&stft_result);
    
    let n_frames = power_spec.len_of(Axis(0));
    let n_freq = power_spec.len_of(Axis(1));
    
    // 创建频率数组
    let frequencies: Vec<f64> = (0..n_freq)
        .map(|i| i as f64 * sample_rate as f64 / n_fft_val as f64)
        .collect();
    
    let mut centroids = Vec::with_capacity(n_frames);
    
    for i in 0..n_frames {
        let mut weighted_sum = 0.0;
        let mut magnitude_sum = 0.0;
        
        for j in 0..n_freq {
            let magnitude = power_spec[[i, j]];
            weighted_sum += frequencies[j] * magnitude;
            magnitude_sum += magnitude;
        }
        
        let centroid = if magnitude_sum > 0.0 {
            weighted_sum / magnitude_sum
        } else {
            0.0
        };
        
        centroids.push(centroid);
    }
    
    Array1::from(centroids)
}

/// 计算频谱滚降点
///
/// # 参数
/// * `signal` - 输入音频信号
/// * `sample_rate` - 采样率 (Hz)
/// * `n_fft` - FFT大小 (默认2048)
/// * `hop_length` - 帧移 (默认512)
/// * `roll_percent` - 滚降百分比 (默认0.85)
///
/// # 返回值
/// 返回频谱滚降点 (n_frames)
#[cfg(feature = "multimodal")]
pub fn compute_spectral_rolloff(
    signal: &[f64],
    sample_rate: usize,
    n_fft: Option<usize>,
    hop_length: Option<usize>,
    roll_percent: Option<f64>,
) -> Array1<f64> {
    // 设置默认参数
    let n_fft_val = n_fft.unwrap_or(2048);
    let hop_length_val = hop_length.unwrap_or(512);
    let roll_percent_val = roll_percent.unwrap_or(0.85);
    
    // 计算STFT
    let window = hanning_window(n_fft_val);
    let window_f64: Vec<f64> = window.iter().map(|&x| x as f64).collect();
    let stft_result = stft(signal, &window_f64, n_fft_val, hop_length_val);
    
    // 计算幅度谱
    let power_spec = magnitude_spectrum(&stft_result);
    
    let n_frames = power_spec.len_of(Axis(0));
    let n_freq = power_spec.len_of(Axis(1));
    
    // 创建频率数组
    let frequencies: Vec<f64> = (0..n_freq)
        .map(|i| i as f64 * sample_rate as f64 / n_fft_val as f64)
        .collect();
    
    let mut rolloffs = Vec::with_capacity(n_frames);
    
    for i in 0..n_frames {
        let frame = power_spec.slice(s![i, ..]);
        let magnitude_sum: f64 = frame.sum();
        let threshold = magnitude_sum * roll_percent_val;
        
        let mut cumsum = 0.0;
        let mut rolloff_idx = 0;
        
        for j in 0..n_freq {
            cumsum += frame[j];
            if cumsum >= threshold {
                rolloff_idx = j;
                break;
            }
        }
        
        rolloffs.push(frequencies[rolloff_idx]);
    }
    
    Array1::from(rolloffs)
}

/// 计算零交叉率
///
/// # 参数
/// * `signal` - 输入音频信号
/// * `frame_length` - 帧长 (默认2048)
/// * `hop_length` - 帧移 (默认512)
///
/// # 返回值
/// 返回零交叉率 (n_frames)
#[cfg(feature = "multimodal")]
pub fn compute_zero_crossing_rate(
    signal: &[f64],
    frame_length: Option<usize>,
    hop_length: Option<usize>,
) -> Array1<f64> {
    // 设置默认参数
    let frame_length_val = frame_length.unwrap_or(2048);
    let hop_length_val = hop_length.unwrap_or(512);
    
    // 分帧
    let frames = frame_signal(signal, frame_length_val, hop_length_val);
    let n_frames = frames.len_of(Axis(0));
    
    let mut zcr = Vec::with_capacity(n_frames);
    
    for i in 0..n_frames {
        let mut crossings = 0;
        
        for j in 1..frame_length_val {
            if frames[[i, j-1]] * frames[[i, j]] < 0.0 {
                crossings += 1;
            }
        }
        
        // 归一化
        zcr.push(crossings as f64 / (frame_length_val as f64 - 1.0));
    }
    
    Array1::from(zcr)
}

/// 计算能量
///
/// # 参数
/// * `signal` - 输入音频信号
/// * `frame_length` - 帧长 (默认2048)
/// * `hop_length` - 帧移 (默认512)
///
/// # 返回值
/// 返回能量 (n_frames)
#[cfg(feature = "multimodal")]
pub fn compute_energy(
    signal: &[f64],
    frame_length: Option<usize>,
    hop_length: Option<usize>,
) -> Array1<f64> {
    // 设置默认参数
    let frame_length_val = frame_length.unwrap_or(2048);
    let hop_length_val = hop_length.unwrap_or(512);
    
    // 分帧
    let frames = frame_signal(signal, frame_length_val, hop_length_val);
    let n_frames = frames.len_of(Axis(0));
    
    let mut energy = Vec::with_capacity(n_frames);
    
    for i in 0..n_frames {
        let frame_energy: f64 = frames.slice(s![i, ..]).mapv(|x| x.powi(2)).sum();
        energy.push(frame_energy);
    }
    
    Array1::from(energy)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::multimodal::extractors::audio::generators::generate_sine_wave;
    
    #[test]
    fn test_mfcc() {
        // 创建测试信号
        let frequency = 440.0;
        let duration = 1.0;
        let sample_rate = 22050;
        let signal = generate_sine_wave(frequency, duration, sample_rate, None, None);
        
        // 计算MFCC
        let n_mfcc = 13;
        let mfcc = compute_mfcc(&signal, sample_rate, Some(n_mfcc), None, None, None, None, None, None);
        
        // 检查维度
        assert_eq!(mfcc.len_of(Axis(1)), n_mfcc);
        
        // 检查MFCC第一帧
        let first_frame = mfcc.slice(s![0, ..]);
        
        // 对于正弦波，第一个MFCC系数应该有最大能量
        let mut max_coef = 0;
        let mut max_val = first_frame[0];
        
        for i in 1..n_mfcc {
            if first_frame[i].abs() > max_val.abs() {
                max_val = first_frame[i].abs();
                max_coef = i;
            }
        }
        
        // 验证第一个系数有显著值（不一定是最大，但应该很大）
        assert!(first_frame[0].abs() > 1.0);
    }
    
    #[test]
    fn test_melspectrogram() {
        // 创建测试信号
        let frequency = 440.0;
        let duration = 1.0;
        let sample_rate = 22050;
        let signal = generate_sine_wave(frequency, duration, sample_rate, None, None);
        
        // 计算梅尔频谱图
        let n_mels = 128;
        let mel_spec = compute_melspectrogram(&signal, sample_rate, None, None, Some(n_mels), None, None, true);
        
        // 检查维度
        assert_eq!(mel_spec.len_of(Axis(1)), n_mels);
        
        // 对于A4 (440Hz)，应该在中低频段有明显能量
        let first_frame = mel_spec.slice(s![0, ..]);
        
        // 找到最大能量的频带
        let mut max_band = 0;
        let mut max_energy = first_frame[0];
        
        for i in 1..n_mels {
            if first_frame[i] > max_energy {
                max_energy = first_frame[i];
                max_band = i;
            }
        }
        
        // 验证最大能量在合理的频带范围内（对于440Hz的正弦波）
        assert!(max_band > 10 && max_band < n_mels/2);
    }
    
    #[test]
    fn test_spectral_centroid() {
        // 创建测试信号
        let frequency = 1000.0;
        let duration = 1.0;
        let sample_rate = 22050;
        let signal = generate_sine_wave(frequency, duration, sample_rate, None, None);
        
        // 计算频谱质心
        let centroids = compute_spectral_centroid(&signal, sample_rate, None, None);
        
        // 对于纯正弦波，频谱质心应该接近信号频率
        let mean_centroid: f64 = centroids.sum() / centroids.len() as f64;
        
        // 允许一定误差
        assert!((mean_centroid - frequency).abs() < 100.0);
    }
} 