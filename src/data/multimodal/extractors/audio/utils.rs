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

//! 音频处理工具模块
//! 
//! 提供各种音频特征提取和处理的通用工具函数，包括:
//! - 频谱分析工具
//! - 音频变换函数
//! - 统计计算工具
//! - 预处理函数
//! - 特征提取辅助函数

use crate::data::multimodal::extractors::audio::error::AudioError;
use ndarray::{Array1, Array2, s};
#[cfg(feature = "multimodal")]
use rustfft::num_complex::Complex64;
#[cfg(feature = "multimodal")]
use realfft::{RealFftPlanner, RealToComplex};
use std::f64::consts::PI;
#[cfg(feature = "multimodal")]
use rustfft::{FftPlanner, num_complex::Complex};
use crate::error::Result;

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
/// use vecmind::data::multimodal::extractors::audio::utils::normalize_audio;
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

/// 计算快速傅里叶变换（FFT）
/// 
/// # 参数
/// * `signal` - 输入信号
/// * `n_fft` - FFT点数，如果为None则使用信号长度的下一个2的幂
/// 
/// # 返回值
/// * `Result<Vec<Complex64>, AudioError>` - FFT结果或错误
#[cfg(feature = "multimodal")]
pub fn compute_fft(signal: &Array1<f64>, n_fft: Option<usize>) -> Result<Vec<Complex64>, AudioError> {
    if signal.is_empty() {
        return Ok(Vec::new());
    }

    let length = signal.len();
    let n_fft = n_fft.unwrap_or_else(|| {
        // 找到下一个2的幂
        let mut n = 1;
        while n < length {
            n *= 2;
        }
        n
    });

    // 准备输入数据
    let mut real_input = if length < n_fft {
        let mut padded = Vec::with_capacity(n_fft);
        padded.extend(signal.iter());
        padded.resize(n_fft, 0.0);
        padded
    } else {
        signal.iter().take(n_fft).cloned().collect()
    };

    // 创建FFT规划器
    let mut planner = RealFftPlanner::<f64>::new();
    let r2c = planner.plan_fft_forward(n_fft);
    
    // 准备输出空间
    let mut spectrum = r2c.make_output_vec();
    
    // 计算FFT
    match r2c.process(&mut real_input, &mut spectrum) {
        Ok(_) => Ok(spectrum),
        Err(e) => Err(AudioError::ProcessingError(format!("FFT计算失败: {}", e)))
    }
}

/// 计算功率谱
/// 
/// # 参数
/// * `spectrum` - 复数频谱
/// * `normalize` - 是否归一化
/// 
/// # 返回值
/// * `Array1<f64>` - 功率谱
#[cfg(feature = "multimodal")]
pub fn compute_power_spectrum(spectrum: &[Complex64], normalize: bool) -> Array1<f64> {
    if spectrum.is_empty() {
        return Array1::zeros(0);
    }

    let power = Array1::from_iter(
        spectrum.iter().map(|&c| (c.norm_sqr()))
    );

    if normalize {
        let sum = power.sum();
        if sum > 0.0 {
            return power.mapv(|x| x / sum);
        }
    }

    power
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
    if n_mels == 0 || n_fft == 0 || sample_rate == 0 {
        return Array2::zeros((0, 0));
    }

    // 将线性频率转换为梅尔频率
    let hz_to_mel = |hz: f64| 2595.0 * (1.0 + hz / 700.0).log10();
    
    // 将梅尔频率转换为线性频率
    let mel_to_hz = |mel: f64| 700.0 * (10.0_f64.powf(mel / 2595.0) - 1.0);
    
    let fmax = if fmax > 0.0 { fmax } else { sample_rate as f64 / 2.0 };
    
    // 计算梅尔刻度上的等间隔点
    let mel_min = hz_to_mel(fmin);
    let mel_max = hz_to_mel(fmax);
    let mel_points = Array1::linspace(mel_min, mel_max, n_mels + 2);
    
    // 转换回线性频率
    let hz_points = mel_points.mapv(mel_to_hz);
    
    // 转换为FFT bin下标
    let bin = hz_points.mapv(|hz| {
        (n_fft as f64 * hz / sample_rate as f64).round().max(0.0)
            .min((n_fft / 2) as f64) as usize
    });
    
    // 创建滤波器组
    let mut filters = Array2::zeros((n_mels, n_fft / 2 + 1));
    
    for i in 0..n_mels {
        for j in bin[i]..=bin[i+2] {
            if j <= bin[i+1] {
                // 上升段
                filters[[i, j]] = (j as f64 - bin[i] as f64) / (bin[i+1] as f64 - bin[i] as f64);
            } else {
                // 下降段
                filters[[i, j]] = (bin[i+2] as f64 - j as f64) / (bin[i+2] as f64 - bin[i+1] as f64);
            }
        }
    }
    
    // 归一化滤波器组
    for i in 0..n_mels {
        let row_sum = filters.row(i).sum();
        if row_sum > 0.0 {
            filters.row_mut(i).mapv_inplace(|x| x / row_sum);
        }
    }
    
    filters
}

/// 计算频谱质心
/// 
/// # 参数
/// * `power_spectrum` - 功率谱
/// * `freqs` - 频率数组
/// 
/// # 返回值
/// * `f64` - 频谱质心
pub fn spectral_centroid(power_spectrum: &Array1<f64>, freqs: &Array1<f64>) -> f64 {
    if power_spectrum.is_empty() || freqs.is_empty() || power_spectrum.len() != freqs.len() {
        return 0.0;
    }

    let total_power = power_spectrum.sum();
    if total_power == 0.0 {
        return 0.0;
    }

    let weighted_sum = power_spectrum.iter()
        .zip(freqs.iter())
        .map(|(&p, &f)| p * f)
        .sum::<f64>();

    weighted_sum / total_power
}

/// 计算色谱图
/// 
/// # 参数
/// * `power_spectrum` - 功率谱
/// * `sample_rate` - 采样率
/// * `n_chroma` - 色谱数量（通常为12，代表12个音高类）
/// * `n_fft` - FFT大小
/// 
/// # 返回值
/// * `Array1<f64>` - 色谱特征
pub fn compute_chroma(power_spectrum: &Array1<f64>, sample_rate: u32, n_chroma: usize, n_fft: usize) -> Array1<f64> {
    if power_spectrum.is_empty() || n_chroma == 0 || n_fft == 0 {
        return Array1::zeros(0);
    }

    // 创建频率到色谱的映射
    let mut chroma_map = Array2::zeros((n_chroma, power_spectrum.len()));
    
    // 计算每个FFT bin对应的频率
    let freqs = Array1::linspace(0.0, sample_rate as f64 / 2.0, power_spectrum.len());
    
    // 参考频率 C4 (MIDI note 60, 261.63 Hz)
    let ref_freq = 440.0 * 2.0_f64.powf((60.0 - 69.0) / 12.0);
    
    for (i, &freq) in freqs.iter().enumerate() {
        if freq > 0.0 {
            // 将频率映射到色谱
            let cents = 1200.0 * (freq / ref_freq).log2();
            let chroma_bin = (cents.rem_euclid(1200.0) / 100.0).floor() as usize % n_chroma;
            chroma_map[[chroma_bin, i]] = 1.0;
        }
    }
    
    // 计算色谱图
    let chroma_features = chroma_map.dot(power_spectrum);
    
    // 归一化
    let chroma_sum = chroma_features.sum();
    if chroma_sum > 0.0 {
        chroma_features.mapv(|x| x / chroma_sum)
    } else {
        chroma_features
    }
}

/// 计算零交叉率
/// 
/// # 参数
/// * `signal` - 输入信号
/// 
/// # 返回值
/// * `f64` - 零交叉率
pub fn zero_crossing_rate(signal: &Array1<f64>) -> f64 {
    if signal.len() <= 1 {
        return 0.0;
    }

    let mut crossings = 0;
    for i in 1..signal.len() {
        if signal[i-1] * signal[i] < 0.0 {
            crossings += 1;
        }
    }

    crossings as f64 / (signal.len() as f64 - 1.0)
}

/// 计算信号能量
/// 
/// # 参数
/// * `signal` - 输入信号
/// * `use_db` - 是否返回分贝值
/// * `ref_value` - 参考值（用于dB计算）
/// * `min_db` - 最小dB值
/// 
/// # 返回值
/// * `f64` - 信号能量
pub fn signal_energy(signal: &Array1<f64>, use_db: bool, ref_value: f64, min_db: f64) -> f64 {
    if signal.is_empty() {
        return 0.0;
    }

    let energy = signal.iter().map(|&x| x * x).sum::<f64>() / signal.len() as f64;
    
    if use_db {
        if energy <= 0.0 {
            return min_db;
        }
        let db = 10.0 * (energy / ref_value.powi(2)).log10();
        db.max(min_db)
    } else {
        energy
    }
}

/// 计算谱平坦度
/// 
/// # 参数
/// * `power_spectrum` - 功率谱
/// 
/// # 返回值
/// * `f64` - 谱平坦度
pub fn spectral_flatness(power_spectrum: &Array1<f64>) -> f64 {
    if power_spectrum.is_empty() {
        return 0.0;
    }

    let n = power_spectrum.len() as f64;
    
    // 过滤掉零值，避免log计算问题
    let filtered: Vec<f64> = power_spectrum.iter()
        .filter(|&&x| x > 0.0)
        .cloned()
        .collect();
    
    if filtered.is_empty() {
        return 0.0;
    }
    
    let n_filtered = filtered.len() as f64;
    
    // 几何平均值
    let log_sum: f64 = filtered.iter().map(|&x| x.ln()).sum::<f64>();
    let geometric_mean = (log_sum / n_filtered).exp();
    
    // 算术平均值
    let arithmetic_mean = filtered.iter().sum::<f64>() / n_filtered;
    
    if arithmetic_mean == 0.0 {
        return 0.0;
    }
    
    // 谱平坦度 = 几何平均 / 算术平均
    geometric_mean / arithmetic_mean
}

/// 计算谱滚降点
/// 
/// # 参数
/// * `power_spectrum` - 功率谱
/// * `roll_percent` - 滚降百分比 (0.0-1.0)
/// 
/// # 返回值
/// * `usize` - 滚降点索引
pub fn spectral_rolloff(power_spectrum: &Array1<f64>, roll_percent: f64) -> usize {
    if power_spectrum.is_empty() {
        return 0;
    }

    let total_energy: f64 = power_spectrum.sum();
    if total_energy == 0.0 {
        return 0;
    }

    let threshold = roll_percent * total_energy;
    let mut cumulative_energy = 0.0;

    for (i, &energy) in power_spectrum.iter().enumerate() {
        cumulative_energy += energy;
        if cumulative_energy >= threshold {
            return i;
        }
    }

    power_spectrum.len() - 1
}

/// 计算信号的RMS (Root Mean Square)
/// 
/// # 参数
/// * `signal` - 输入信号
/// 
/// # 返回值
/// * `f64` - RMS值
pub fn root_mean_square(signal: &Array1<f64>) -> f64 {
    if signal.is_empty() {
        return 0.0;
    }

    let mean_square = signal.iter().map(|&x| x * x).sum::<f64>() / signal.len() as f64;
    mean_square.sqrt()
}

/// 计算信号的峰值因子 (Crest Factor)
/// 
/// # 参数
/// * `signal` - 输入信号
/// 
/// # 返回值
/// * `f64` - 峰值因子
pub fn crest_factor(signal: &Array1<f64>) -> f64 {
    if signal.is_empty() {
        return 0.0;
    }

    let peak = signal.iter().fold(0.0_f64, |max, &x| max.max(x.abs()));
    let rms = root_mean_square(signal);
    
    if rms == 0.0 {
        return 0.0;
    }
    
    peak / rms
}

/// 计算信号的峭度 (Kurtosis)
/// 
/// # 参数
/// * `signal` - 输入信号
/// 
/// # 返回值
/// * `f64` - 峭度值
pub fn kurtosis(signal: &Array1<f64>) -> f64 {
    if signal.len() < 4 {
        return 0.0;
    }

    let mean = signal.mean().unwrap_or(0.0);
    let n = signal.len() as f64;
    
    let m4 = signal.iter()
        .map(|&x| (x - mean).powi(4))
        .sum::<f64>() / n;
    
    let var_squared = signal.iter()
        .map(|&x| (x - mean).powi(2))
        .sum::<f64>() / n;
    
    if var_squared == 0.0 {
        return 0.0;
    }
    
    let var_squared = var_squared.powi(2);
    m4 / var_squared - 3.0 // 减去3使得正态分布的峭度为0
}

/// 计算信号的偏度 (Skewness)
/// 
/// # 参数
/// * `signal` - 输入信号
/// 
/// # 返回值
/// * `f64` - 偏度值
pub fn skewness(signal: &Array1<f64>) -> f64 {
    if signal.len() < 3 {
        return 0.0;
    }

    let mean = signal.mean().unwrap_or(0.0);
    let n = signal.len() as f64;
    
    let m3 = signal.iter()
        .map(|&x| (x - mean).powi(3))
        .sum::<f64>() / n;
    
    let std_dev = signal.iter()
        .map(|&x| (x - mean).powi(2))
        .sum::<f64>() / n;
    
    if std_dev == 0.0 {
        return 0.0;
    }
    
    let std_dev = std_dev.powf(1.5);
    m3 / std_dev
}

/// 计算梅尔频谱图
/// 
/// # 参数
/// * `signal` - 输入音频信号
/// * `sample_rate` - 采样率
/// * `n_fft` - FFT窗口大小
/// * `hop_length` - 帧移
/// * `n_mels` - 梅尔滤波器数量
/// 
/// # 返回
/// * 梅尔频谱图矩阵 (n_mels x n_frames)
pub fn melspectrogram(
    signal: &[f32], 
    sample_rate: u32, 
    n_fft: usize, 
    hop_length: usize, 
    n_mels: usize
) -> Result<Array2<f32>> {
    // 计算功率谱
    let stft_result = stft(signal, n_fft, hop_length)?;
    let power_spec = magnitude_spectrum(&stft_result, true)?;
    
    // 创建梅尔滤波器组
    let mel_filters = mel_filter_bank(n_fft / 2 + 1, n_mels, sample_rate as f32, 0.0, sample_rate as f32 / 2.0)?;
    
    // 应用梅尔滤波器
    let mut mel_spec = Array2::zeros((n_mels, power_spec.shape()[1]));
    for (i, filter) in mel_filters.outer_iter().enumerate() {
        for j in 0..power_spec.shape()[1] {
            let frame = power_spec.slice(s![.., j]);
            let filtered = (&filter * &frame).sum();
            mel_spec[[i, j]] = filtered;
        }
    }
    
    Ok(mel_spec)
}

/// 计算MFCC (梅尔频率倒谱系数)
/// 
/// # 参数
/// * `signal` - 输入音频信号
/// * `sample_rate` - 采样率
/// * `n_mfcc` - MFCC系数数量
/// * `n_fft` - FFT窗口大小
/// * `hop_length` - 帧移
/// * `n_mels` - 梅尔滤波器数量
/// 
/// # 返回
/// * MFCC系数矩阵 (n_mfcc x n_frames)
pub fn mfcc(
    signal: &[f32], 
    sample_rate: u32, 
    n_mfcc: usize, 
    n_fft: usize, 
    hop_length: usize, 
    n_mels: usize
) -> Result<Array2<f32>> {
    // 计算梅尔频谱图
    let mel_spec = melspectrogram(signal, sample_rate, n_fft, hop_length, n_mels)?;
    
    // 取对数
    let log_mel_spec = mel_spec.mapv(|x| if x > 0.0 { x.log10() } else { -80.0 });
    
    // 应用DCT
    let mut mfcc_features = Array2::zeros((n_mfcc, log_mel_spec.shape()[1]));
    for j in 0..log_mel_spec.shape()[1] {
        let frame = log_mel_spec.slice(s![.., j]);
        let coeffs = dct(&frame.to_vec(), n_mfcc)?;
        for i in 0..n_mfcc {
            mfcc_features[[i, j]] = coeffs[i];
        }
    }
    
    Ok(mfcc_features)
}

/// 计算离散余弦变换 (DCT)
/// 
/// # 参数
/// * `input` - 输入信号
/// * `n_coeffs` - 输出系数数量
/// 
/// # 返回
/// * DCT系数
pub fn dct(input: &[f32], n_coeffs: usize) -> Result<Vec<f32>> {
    let n = input.len();
    let mut coeffs = vec![0.0; n_coeffs];
    
    for k in 0..n_coeffs {
        let mut sum: f64 = 0.0;
        for i in 0..n {
            let angle = PI * (i as f64 + 0.5) * k as f64 / n as f64;
            sum += input[i] as f64 * angle.cos();
        }
        
        let scale = if k == 0 { (1.0 / n as f64).sqrt() } else { (2.0 / n as f64).sqrt() };
        coeffs[k] = (sum * scale) as f32;
    }
    
    Ok(coeffs)
}

/// 计算短时傅里叶变换 (STFT)
/// 
/// # 参数
/// * `signal` - 输入音频信号
/// * `n_fft` - FFT窗口大小
/// * `hop_length` - 帧移
/// 
/// # 返回
/// * 复数STFT矩阵 (n_fft/2+1 x n_frames)
#[cfg(feature = "multimodal")]
pub fn stft(signal: &[f32], n_fft: usize, hop_length: usize) -> Result<Array2<Complex<f32>>> {
    let n_samples = signal.len();
    let n_frames = 1 + (n_samples - n_fft) / hop_length;
    
    // 创建汉宁窗
    let window = hanning_window(n_fft);
    
    // 初始化STFT矩阵
    let mut stft_matrix = Array2::zeros((n_fft / 2 + 1, n_frames));
    
    // 创建FFT规划器
    let mut planner = FftPlanner::new();
    let fft = planner.plan_fft_forward(n_fft);
    
    for i in 0..n_frames {
        let start = i * hop_length;
        let end = start + n_fft;
        
        if end <= n_samples {
            // 应用窗函数
            let mut frame: Vec<Complex<f32>> = signal[start..end]
                .iter()
                .zip(window.iter())
                .map(|(&s, &w)| Complex::new(s * w, 0.0))
                .collect();
            
            // 补零至n_fft
            frame.resize(n_fft, Complex::new(0.0, 0.0));
            
            // 执行FFT
            fft.process(&mut frame);
            
            // 存储前一半结果（由于信号是实数，FFT结果是对称的）
            for j in 0..=n_fft/2 {
                stft_matrix[[j, i]] = frame[j];
            }
        }
    }
    
    Ok(stft_matrix)
}

/// 计算功率谱或幅值谱
/// 
/// # 参数
/// * `stft_matrix` - STFT矩阵
/// * `power` - 是否计算功率谱（平方幅值），否则返回幅值谱
/// 
/// # 返回
/// * 功率谱或幅值谱矩阵
#[cfg(feature = "multimodal")]
pub fn magnitude_spectrum(stft_matrix: &Array2<Complex<f32>>, power: bool) -> Result<Array2<f32>> {
    let (n_bins, n_frames) = (stft_matrix.shape()[0], stft_matrix.shape()[1]);
    let mut mag_spec = Array2::zeros((n_bins, n_frames));
    
    for i in 0..n_bins {
        for j in 0..n_frames {
            let complex_val = stft_matrix[[i, j]];
            let magnitude = complex_val.norm();
            mag_spec[[i, j]] = if power { magnitude * magnitude } else { magnitude };
        }
    }
    
    Ok(mag_spec)
}

/// 创建汉宁窗函数
/// 
/// # 参数
/// * `size` - 窗口大小
/// 
/// # 返回
/// * 汉宁窗系数
pub fn hanning_window(size: usize) -> Vec<f32> {
    (0..size)
        .map(|i| {
            let i_f64 = i as f64;
            let size_f64 = size as f64;
            0.5_f32 * (1.0 - (2.0 * PI * i_f64 / (size_f64 - 1.0)).cos() as f32)
        })
        .collect()
}

/// 创建梅尔滤波器组
/// 
/// # 参数
/// * `n_freqs` - 频率点数量
/// * `n_mels` - 梅尔滤波器数量
/// * `sample_rate` - 采样率
/// * `f_min` - 最低频率
/// * `f_max` - 最高频率
/// 
/// # 返回
/// * 梅尔滤波器矩阵
pub fn mel_filter_bank(
    n_freqs: usize, 
    n_mels: usize, 
    sample_rate: f32, 
    f_min: f32, 
    f_max: f32
) -> Result<Array2<f32>> {
    // 创建梅尔刻度
    let mel_min = hz_to_mel(f_min);
    let mel_max = hz_to_mel(f_max);
    let mel_points: Vec<f32> = (0..=n_mels+1)
        .map(|i| mel_min + (mel_max - mel_min) * i as f32 / (n_mels as f32 + 1.0))
        .collect();
    
    // 转换回频率
    let freq_points: Vec<f32> = mel_points.iter().map(|&m| mel_to_hz(m)).collect();
    
    // 将频率点转换为FFT bin
    let bin_points: Vec<usize> = freq_points
        .iter()
        .map(|&f| ((n_freqs - 1) as f32 * f / (sample_rate / 2.0)).round() as usize)
        .map(|bin| bin.min(n_freqs - 1))
        .collect();
    
    // 创建滤波器矩阵
    let mut filters = Array2::zeros((n_mels, n_freqs));
    
    for i in 0..n_mels {
        let left_bin = bin_points[i];
        let center_bin = bin_points[i + 1];
        let right_bin = bin_points[i + 2];
        
        // 左半三角
        for j in left_bin..=center_bin {
            if j < n_freqs && center_bin > left_bin {
                filters[[i, j]] = (j - left_bin) as f32 / (center_bin - left_bin) as f32;
            }
        }
        
        // 右半三角
        for j in center_bin..=right_bin {
            if j < n_freqs && right_bin > center_bin {
                filters[[i, j]] = 1.0 - (j - center_bin) as f32 / (right_bin - center_bin) as f32;
            }
        }
    }
    
    Ok(filters)
}

/// 赫兹到梅尔刻度转换
/// 
/// # 参数
/// * `freq` - 频率（赫兹）
/// 
/// # 返回
/// * 梅尔刻度值
pub fn hz_to_mel(freq: f32) -> f32 {
    1127.0 * (1.0 + freq / 700.0).ln()
}

/// 梅尔刻度到赫兹转换
/// 
/// # 参数
/// * `mel` - 梅尔刻度值
/// 
/// # 返回
/// * 频率（赫兹）
pub fn mel_to_hz(mel: f32) -> f32 {
    700.0 * ((mel / 1127.0).exp() - 1.0)
}

/// 计算色度特征 (Chroma)
/// 
/// # 参数
/// * `signal` - 输入音频信号
/// * `sample_rate` - 采样率
/// * `n_fft` - FFT窗口大小
/// * `hop_length` - 帧移
/// * `n_chroma` - 色度特征维度（通常为12）
/// 
/// # 返回
/// * 色度特征矩阵 (n_chroma x n_frames)
pub fn chroma_feature(
    signal: &[f32], 
    sample_rate: u32, 
    n_fft: usize, 
    hop_length: usize, 
    n_chroma: usize
) -> Result<Array2<f32>> {
    // 计算STFT
    let stft_result = stft(signal, n_fft, hop_length)?;
    let mag_spec = magnitude_spectrum(&stft_result, false)?;
    
    // FFT频率到色度映射
    let chroma_map = create_chroma_filterbank(n_fft / 2 + 1, n_chroma, sample_rate)?;
    
    // 应用色度滤波器
    let n_frames = mag_spec.shape()[1];
    let mut chroma_features = Array2::zeros((n_chroma, n_frames));
    
    for i in 0..n_frames {
        let frame = mag_spec.slice(s![.., i]);
        for j in 0..n_chroma {
            let filter = chroma_map.slice(s![j, ..]);
            chroma_features[[j, i]] = (&filter * &frame).sum();
        }
    }
    
    // 标准化每一帧的色度特征
    for i in 0..n_frames {
        let mut frame = chroma_features.slice_mut(s![.., i]);
        let max_val = frame.iter().fold(0.0f32, |a, &b| a.max(b));
        if max_val > 0.0 {
            frame.mapv_inplace(|x| x / max_val);
        }
    }
    
    Ok(chroma_features)
}

/// 创建色度滤波器组
/// 
/// # 参数
/// * `n_freqs` - 频率点数量
/// * `n_chroma` - 色度特征维度
/// * `sample_rate` - 采样率
/// 
/// # 返回
/// * 色度滤波器矩阵
pub fn create_chroma_filterbank(
    n_freqs: usize, 
    n_chroma: usize, 
    sample_rate: u32
) -> Result<Array2<f32>> {
    let mut chroma_filters = Array2::zeros((n_chroma, n_freqs));
    
    // 频率分辨率
    let fft_freqs: Vec<f32> = (0..n_freqs)
        .map(|i| i as f32 * sample_rate as f32 / (2.0 * (n_freqs - 1) as f32))
        .collect();
    
    // 标准 MIDI 音高 A4 = 440Hz (MIDI note 69)
    let a440 = 440.0;
    
    // 将频率转换为色度
    for i in 0..n_freqs {
        let freq = fft_freqs[i];
        if freq > 0.0 {
            // 转换到对数频率
            let log_freq = (12.0 * (freq / a440).log2()).round();
            
            // 映射到对应的色度
            let chroma_bin = ((log_freq % 12.0) + 12.0) % 12.0;
            
            // 如果n_chroma不是12，需要重新映射
            let bin = (chroma_bin * n_chroma as f32 / 12.0).round() as usize % n_chroma;
            
            chroma_filters[[bin, i]] += 1.0;
        }
    }
    
    // 标准化滤波器
    for i in 0..n_chroma {
        let mut filter = chroma_filters.slice_mut(s![i, ..]);
        let sum = filter.sum();
        if sum > 0.0 {
            filter.mapv_inplace(|x| x / sum);
        }
    }
    
    Ok(chroma_filters)
}

/// 计算频谱质心
/// 
/// # 参数
/// * `signal` - 输入音频信号
/// * `sample_rate` - 采样率
/// * `n_fft` - FFT窗口大小
/// * `hop_length` - 帧移
/// 
/// # 返回
/// * 频谱质心特征向量 (n_frames)
pub fn spectral_centroid_frames(
    signal: &[f32],
    sample_rate: u32,
    n_fft: usize,
    hop_length: usize,
    window: Option<&[f32]>
) -> Result<Vec<f64>> {
    // 计算STFT
    let stft_result = stft(signal, n_fft, hop_length)?;
    let mag_spec = magnitude_spectrum(&stft_result, false)?;
    
    let n_freqs = mag_spec.shape()[0];
    let n_frames = mag_spec.shape()[1];
    
    // 频率范围
    let freqs: Vec<f32> = (0..n_freqs)
        .map(|i| i as f32 * sample_rate as f32 / n_fft as f32)
        .collect();
    
    let mut centroids = Vec::with_capacity(n_frames);
    
    for i in 0..n_frames {
        let frame = mag_spec.slice(s![.., i]);
        let mut weighted_sum = 0.0;
        let mut norm = 0.0;
        
        for j in 0..n_freqs {
            weighted_sum += freqs[j] * frame[j];
            norm += frame[j];
        }
        
        centroids.push(if norm > 0.0 { weighted_sum / norm } else { 0.0 } as f64);
    }
    
    Ok(centroids)
}

/// 计算频谱滚降点
/// 
/// # 参数
/// * `signal` - 输入音频信号
/// * `sample_rate` - 采样率
/// * `n_fft` - FFT窗口大小
/// * `hop_length` - 帧移
/// * `roll_percent` - 滚降百分比 (0.0-1.0)
/// 
/// # 返回
/// * 频谱滚降特征向量 (n_frames)
pub fn spectral_rolloff_frames(
    signal: &[f32],
    sample_rate: u32,
    n_fft: usize,
    hop_length: usize,
    roll_percent: f64,
    window: Option<&[f32]>
) -> Result<Vec<f64>> {
    // 计算STFT
    let stft_result = stft(signal, n_fft, hop_length)?;
    let mag_spec = magnitude_spectrum(&stft_result, true)?;
    
    let n_freqs = mag_spec.shape()[0];
    let n_frames = mag_spec.shape()[1];
    
    // 频率范围
    let freqs: Vec<f32> = (0..n_freqs)
        .map(|i| i as f32 * sample_rate as f32 / n_fft as f32)
        .collect();
    
    let mut rolloffs = Vec::with_capacity(n_frames);
    
    for i in 0..n_frames {
        let frame = mag_spec.slice(s![.., i]);
        let energy_sum: f64 = frame.sum() as f64;
        
        if energy_sum > 0.0 {
            let threshold = roll_percent * energy_sum;
            let mut cumsum: f64 = 0.0;
            
            for j in 0..n_freqs {
                cumsum += frame[j] as f64;
                if cumsum >= threshold {
                    rolloffs.push(freqs[j] as f64);
                    break;
                }
            }
        }
    }
    
    Ok(rolloffs)
}

/// 计算过零率 (Zero Crossing Rate)
/// 
/// # 参数
/// * `signal` - 输入音频信号
/// * `frame_length` - 帧长度
/// * `hop_length` - 帧移
/// 
/// # 返回
/// * 过零率特征向量 (n_frames)
pub fn zero_crossing_rate_frames(
    signal: &[f32],
    frame_length: usize,
    hop_length: usize
) -> Result<Vec<f64>> {
    let n_samples = signal.len();
    let n_frames = 1 + (n_samples - frame_length) / hop_length;
    
    let mut zcr = Vec::with_capacity(n_frames);
    
    for i in 0..n_frames {
        let start = i * hop_length;
        let end = (start + frame_length).min(n_samples);
        
        let mut crossings = 0;
        for j in start+1..end {
            if (signal[j] >= 0.0 && signal[j-1] < 0.0) || 
               (signal[j] < 0.0 && signal[j-1] >= 0.0) {
                crossings += 1;
            }
        }
        
        zcr.push(crossings as f64 / (end - start) as f64);
    }
    
    Ok(zcr)
}

/// 计算RMS能量
/// 
/// # 参数
/// * `signal` - 输入音频信号
/// * `frame_length` - 帧长度
/// * `hop_length` - 帧移
/// 
/// # 返回
/// * RMS能量特征向量 (n_frames)
pub fn rms_energy(
    signal: &[f32], 
    frame_length: usize, 
    hop_length: usize
) -> Result<Array1<f32>> {
    let n_samples = signal.len();
    let n_frames = 1 + (n_samples - frame_length) / hop_length;
    
    let mut energy = Array1::zeros(n_frames);
    
    for i in 0..n_frames {
        let start = i * hop_length;
        let end = (start + frame_length).min(n_samples);
        
        let mut sum_squares = 0.0;
        for j in start..end {
            sum_squares += signal[j] * signal[j];
        }
        
        energy[i] = (sum_squares / (end - start) as f32).sqrt();
    }
    
    Ok(energy)
}

/// 计算声音包络
/// 
/// # 参数
/// * `signal` - 输入音频信号
/// * `frame_length` - 帧长度
/// * `hop_length` - 帧移
/// 
/// # 返回
/// * 包络特征向量 (n_frames)
pub fn envelope(
    signal: &[f32], 
    frame_length: usize, 
    hop_length: usize
) -> Result<Array1<f32>> {
    let n_samples = signal.len();
    let n_frames = 1 + (n_samples - frame_length) / hop_length;
    
    let mut env = Array1::zeros(n_frames);
    
    for i in 0..n_frames {
        let start = i * hop_length;
        let end = (start + frame_length).min(n_samples);
        
        let mut max_amp: f32 = 0.0;
        for j in start..end {
            max_amp = max_amp.max(signal[j].abs());
        }
        
        env[i] = max_amp;
    }
    
    Ok(env)
}

/// 计算时域特征 (均值、方差、RMS等)
/// 
/// # 参数
/// * `signal` - 输入音频信号
/// * `frame_length` - 帧长度
/// * `hop_length` - 帧移
/// 
/// # 返回
/// * 时域特征矩阵 (6 x n_frames)
pub fn temporal_features(
    signal: &[f32], 
    frame_length: usize, 
    hop_length: usize
) -> Result<Array2<f32>> {
    let n_samples = signal.len();
    let n_frames = 1 + (n_samples - frame_length) / hop_length;
    
    // 6个特征: 均值、标准差、RMS、ZCR、峰度、波形因子
    let mut features = Array2::zeros((6, n_frames));
    
    for i in 0..n_frames {
        let start = i * hop_length;
        let end = (start + frame_length).min(n_samples);
        let frame = &signal[start..end];
        
        // 1. 均值
        let mean: f32 = frame.iter().sum::<f32>() / frame.len() as f32;
        features[[0, i]] = mean;
        
        // 2. 标准差
        let variance: f32 = frame.iter()
            .map(|&x| (x - mean).powi(2))
            .sum::<f32>() / frame.len() as f32;
        features[[1, i]] = variance.sqrt();
        
        // 3. RMS
        let rms: f32 = (frame.iter()
            .map(|&x| x.powi(2))
            .sum::<f32>() / frame.len() as f32).sqrt();
        features[[2, i]] = rms;
        
        // 4. 过零率
        let mut zcr = 0;
        for j in 1..frame.len() {
            if (frame[j] >= 0.0 && frame[j-1] < 0.0) || 
               (frame[j] < 0.0 && frame[j-1] >= 0.0) {
                zcr += 1;
            }
        }
        features[[3, i]] = zcr as f32 / (frame.len() - 1) as f32;
        
        // 5. 峰度
        let kurtosis: f32 = if variance > 0.0 {
            frame.iter()
                .map(|&x| ((x - mean) / variance.sqrt()).powi(4))
                .sum::<f32>() / frame.len() as f32 - 3.0
        } else {
            0.0
        };
        features[[4, i]] = kurtosis;
        
        // 6. 波形因子 (Crest Factor) = 峰值/RMS
        let peak: f32 = frame.iter().map(|&x| x.abs()).fold(0.0, |a, b| a.max(b));
        features[[5, i]] = if rms > 0.0 { peak / rms } else { 0.0 };
    }
    
    Ok(features)
}

/// 将音频特征转换为dB刻度
/// 
/// # 参数
/// * `magnitude` - 线性幅度
/// * `min_db` - 最小dB阈值（低于此值将被剪裁）
/// 
/// # 返回
/// * dB刻度值
pub fn amplitude_to_db(magnitude: f32, min_db: f32) -> f32 {
    const REF_VALUE: f32 = 1.0;
    
    if magnitude > 0.0 {
        let db = 20.0 * (magnitude / REF_VALUE).log10();
        db.max(min_db)
    } else {
        min_db
    }
}

/// 在特征矩阵上应用增强（时间方向平滑和标准化）
/// 
/// # 参数
/// * `features` - 输入特征矩阵
/// * `normalize` - 是否进行标准化
/// 
/// # 返回
/// * 增强后的特征矩阵
pub fn enhance_features(features: &mut Array2<f32>, normalize: bool) -> Result<()> {
    let (n_features, n_frames) = features.dim();
    
    if n_frames > 2 {
        // 时间方向平滑（简单均值滤波）
        let mut smoothed = Array2::zeros((n_features, n_frames));
        for i in 0..n_features {
            for j in 1..n_frames-1 {
                smoothed[[i, j]] = (features[[i, j-1]] + features[[i, j]] + features[[i, j+1]]) / 3.0;
            }
            // 边界处理
            smoothed[[i, 0]] = (features[[i, 0]] + features[[i, 1]]) / 2.0;
            smoothed[[i, n_frames-1]] = (features[[i, n_frames-2]] + features[[i, n_frames-1]]) / 2.0;
        }
        *features = smoothed;
    }
    
    if normalize {
        // 特征维度标准化
        for i in 0..n_features {
            let mut row = features.slice_mut(s![i, ..]);
            let min_val = row.fold(f32::INFINITY, |a, &b| a.min(b));
            let max_val = row.fold(f32::NEG_INFINITY, |a, &b| a.max(b));
            
            let range = max_val - min_val;
            if range > 1e-10 {
                row.mapv_inplace(|x| (x - min_val) / range);
            } else {
                row.fill(0.5);
            }
        }
    }
    
    Ok(())
}

/// 生成测试用的正弦音频样本
/// 
/// # 参数
/// * `duration_secs` - 音频时长（秒）
/// * `sample_rate` - 采样率（Hz）
/// * `frequencies` - 频率组件列表（Hz）
/// * `amplitudes` - 对应的振幅列表
/// 
/// # 返回
/// * 合成的音频样本
pub fn generate_sine_wave(
    duration_secs: f32,
    sample_rate: u32,
    frequencies: &[f32],
    amplitudes: &[f32],
) -> Vec<f32> {
    let n_samples = (duration_secs * sample_rate as f32) as usize;
    let mut signal = vec![0.0; n_samples];
    
    // 确保频率和振幅列表长度匹配
    let min_len = frequencies.len().min(amplitudes.len());
    
    for i in 0..n_samples {
        let t = i as f32 / sample_rate as f32;
        
        for j in 0..min_len {
            let freq = frequencies[j];
            let amp = amplitudes[j];
            signal[i] += amp * (2.0 * PI * freq as f64 * t as f64).sin() as f32;
        }
    }
    
    // 标准化到[-1, 1]范围
    let max_amplitude = signal.iter().fold(0.0f32, |a, &b| a.max(b.abs()));
    if max_amplitude > 0.0 {
        for sample in &mut signal {
            *sample /= max_amplitude;
        }
    }
    
    signal
}

/// 添加白噪声到音频信号
/// 
/// # 参数
/// * `signal` - 输入音频信号
/// * `noise_level` - 噪声水平（0.0-1.0）
/// 
/// # 返回
/// * 添加噪声后的信号
pub fn add_noise(signal: &[f32], noise_level: f32) -> Vec<f32> {
    use rand::Rng;
    let mut rng = rand::thread_rng();
    
    signal.iter()
        .map(|&s| {
            let noise = (rng.gen::<f32>() * 2.0 - 1.0) * noise_level;
            (s + noise).max(-1.0).min(1.0)
        })
        .collect()
}

/// 生成音乐音阶测试信号
/// 
/// # 参数
/// * `duration_secs` - 音频时长（秒）
/// * `sample_rate` - 采样率（Hz）
/// * `base_note` - 基准音符频率（Hz，默认A4=440Hz）
/// 
/// # 返回
/// * 合成的音频样本（包含完整的音阶）
pub fn generate_musical_scale(
    duration_secs: f32,
    sample_rate: u32,
    base_note: f32,
) -> Vec<f32> {
    // C大调音阶的半音级数：C, D, E, F, G, A, B
    let semitones = [0, 2, 4, 5, 7, 9, 11, 12];
    let n_notes = semitones.len();
    
    let note_duration = duration_secs / n_notes as f32;
    let samples_per_note = (note_duration * sample_rate as f32) as usize;
    let total_samples = samples_per_note * n_notes;
    
    let mut signal = vec![0.0; total_samples];
    
    for (i, &semitone) in semitones.iter().enumerate() {
        // 计算音符频率: f = base_note * 2^(semitone/12)
        let freq = base_note * 2.0f32.powf(semitone as f32 / 12.0);
        
        let start_sample = i * samples_per_note;
        let end_sample = (i + 1) * samples_per_note;
        
        // 生成单个音符
        for j in start_sample..end_sample {
            let t = (j - start_sample) as f32 / sample_rate as f32;
            
            // 应用衰减包络，使声音自然
            let env = 0.5_f32 * (1.0 - ((t as f64 / note_duration as f64) * 2.0 * PI).cos() as f32);
            signal[j] = env * (2.0 * PI * freq as f64 * t as f64).sin() as f32;
        }
    }
    
    signal
}

/// 创建带时变特性的测试音频信号
/// 
/// # 参数
/// * `duration_secs` - 音频时长（秒）
/// * `sample_rate` - 采样率（Hz）
/// 
/// # 返回
/// * 合成的音频样本，包含不同频率部分和过渡
pub fn generate_test_audio(duration_secs: f32, sample_rate: u32) -> Vec<f32> {
    let n_samples = (duration_secs * sample_rate as f32) as usize;
    let mut signal = vec![0.0; n_samples];
    
    // 将持续时间分为多个段
    let segments = 3;
    let samples_per_segment = n_samples / segments;
    
    // 段1：单频正弦波
    for i in 0..samples_per_segment {
        let t = i as f32 / sample_rate as f32;
        signal[i] = (2.0 * PI * 440.0_f64 * t as f64).sin() as f32;
    }
    
    // 段2：频率扫描
    for i in 0..samples_per_segment {
        let segment_pos = i as f32 / samples_per_segment as f32; // 0到1
        let freq = 220.0 + 880.0 * segment_pos;
        let t = (samples_per_segment + i) as f32 / sample_rate as f32;
        signal[samples_per_segment + i] = (2.0 * PI * freq as f64 * t as f64).sin() as f32;
    }
    
    // 段3：多频正弦波叠加
    for i in 0..samples_per_segment {
        let t = (2 * samples_per_segment + i) as f32 / sample_rate as f32;
        signal[2 * samples_per_segment + i] = 
            0.5 * (2.0 * PI * 330.0_f64 * t as f64).sin() as f32 +
            0.3 * (2.0 * PI * 660.0_f64 * t as f64).sin() as f32 +
            0.2 * (2.0 * PI * 990.0_f64 * t as f64).sin() as f32;
    }
    
    // 应用淡入淡出以避免间断
    for i in 0..n_samples {
        // 在段与段之间应用淡入淡出
        if i % samples_per_segment < 1000 {
            let fade = (i % samples_per_segment) as f32 / 1000.0;
            signal[i] *= fade;
        } else if i % samples_per_segment >= samples_per_segment - 1000 {
            let fade = (samples_per_segment - (i % samples_per_segment)) as f32 / 1000.0;
            signal[i] *= fade;
        }
    }
    
    // 添加少量噪声使信号更真实
    add_noise(&signal, 0.05)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    // 未使用导入移除

    #[test]
    fn test_normalize_audio() {
        let audio = Array1::from_vec(vec![0.5, -1.0, 0.25, -0.75]);
        let normalized = normalize_audio(&audio);
        assert_relative_eq!(normalized[0], 0.5);
        assert_relative_eq!(normalized[1], -1.0);
        assert_relative_eq!(normalized[2], 0.25);
        assert_relative_eq!(normalized[3], -0.75);
    }

    #[test]
    fn test_preemphasis() {
        let audio = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0]);
        let preemphasized = preemphasis(&audio, 0.97);
        assert_relative_eq!(preemphasized[0], 1.0);
        assert_relative_eq!(preemphasized[1], 2.0 - 0.97 * 1.0);
        assert_relative_eq!(preemphasized[2], 3.0 - 0.97 * 2.0);
        assert_relative_eq!(preemphasized[3], 4.0 - 0.97 * 3.0);
    }

    #[test]
    fn test_apply_window() {
        let frame = Array1::ones(4);
        
        // 汉宁窗
        let hann = apply_window(&frame, "hann");
        assert_eq!(hann.len(), 4);
        assert_relative_eq!(hann[0], 0.0, epsilon = 1e-10);
        assert_relative_eq!(hann[2], 1.0);
        assert_relative_eq!(hann[3], 0.0, epsilon = 1e-10);
        
        // 矩形窗
        let rect = apply_window(&frame, "rectangular");
        for v in rect.iter() {
            assert_relative_eq!(*v, 1.0);
        }
    }

    #[test]
    fn test_frame_signal() {
        let signal = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        
        // 帧长3，步长2
        let frames = frame_signal(&signal, 3, 2);
        assert_eq!(frames.shape(), &[2, 3]);
        assert_relative_eq!(frames[[0, 0]], 1.0);
        assert_relative_eq!(frames[[0, 1]], 2.0);
        assert_relative_eq!(frames[[0, 2]], 3.0);
        assert_relative_eq!(frames[[1, 0]], 3.0);
        assert_relative_eq!(frames[[1, 1]], 4.0);
        assert_relative_eq!(frames[[1, 2]], 5.0);
    }

    #[test]
    fn test_zero_crossing_rate() {
        let signal = Array1::from_vec(vec![1.0, -1.0, 1.0, -1.0]);
        let zcr = zero_crossing_rate(&signal);
        assert_relative_eq!(zcr, 1.0);
        
        let signal2 = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0]);
        let zcr2 = zero_crossing_rate(&signal2);
        assert_relative_eq!(zcr2, 0.0);
    }

    #[test]
    fn test_signal_energy() {
        let signal = Array1::from_vec(vec![1.0, 1.0, 1.0, 1.0]);
        let energy = signal_energy(&signal, false, 1.0, -60.0);
        assert_relative_eq!(energy, 1.0);
        
        let energy_db = signal_energy(&signal, true, 1.0, -60.0);
        assert_relative_eq!(energy_db, 0.0);
    }

    #[test]
    fn test_root_mean_square() {
        let signal = Array1::from_vec(vec![1.0, 1.0, 1.0, 1.0]);
        let rms = root_mean_square(&signal);
        assert_relative_eq!(rms, 1.0);
        
        let signal2 = Array1::from_vec(vec![2.0, 2.0, 2.0, 2.0]);
        let rms2 = root_mean_square(&signal2);
        assert_relative_eq!(rms2, 2.0);
    }

    #[test]
    fn test_generate_sine_wave() {
        let sample_rate = 16000;
        let duration = 0.1;
        let signal = generate_sine_wave(duration, sample_rate, &[440.0], &[1.0]);
        
        // 验证样本数
        assert_eq!(signal.len(), (duration * sample_rate as f32) as usize);
        
        // 验证信号范围
        for sample in &signal {
            assert!(*sample >= -1.0 && *sample <= 1.0);
        }
    }
    
    #[test]
    fn test_mfcc_extraction() {
        let sample_rate = 16000;
        let duration = 0.5;
        let signal = generate_sine_wave(duration, sample_rate, &[440.0], &[1.0]);
        
        let n_mfcc = 13;
        let n_fft = 2048;
        let hop_length = 512;
        let n_mels = 40;
        
        let mfcc_features = mfcc(
            &signal, 
            sample_rate, 
            n_mfcc, 
            n_fft, 
            hop_length, 
            n_mels
        ).unwrap();
        
        // 验证MFCC特征维度
        assert_eq!(mfcc_features.shape()[0], n_mfcc);
        
        // 验证帧数
        let expected_frames = 1 + (signal.len() - n_fft) / hop_length;
        assert_eq!(mfcc_features.shape()[1], expected_frames);
    }
    
    #[test]
    fn test_chroma_feature() {
        let sample_rate = 22050;
        let duration = 0.3;
        
        // C大调和弦：C, E, G（261.63Hz, 329.63Hz, 392.00Hz）
        let frequencies = vec![261.63, 329.63, 392.00];
        let amplitudes = vec![0.8, 0.6, 0.7];
        
        let signal = generate_sine_wave(duration, sample_rate, &frequencies, &amplitudes);
        
        let n_chroma = 12;
        let n_fft = 2048;
        let hop_length = 512;
        
        let chroma_features = chroma_feature(
            &signal, 
            sample_rate, 
            n_fft, 
            hop_length, 
            n_chroma
        ).unwrap();
        
        // 验证色度特征维度
        assert_eq!(chroma_features.shape()[0], n_chroma);
        
        // C大调和弦应该在C, E, G对应的色度bin上有较高的激活度
        // C的色度索引是0，E是4，G是7
        let frame_idx = chroma_features.shape()[1] / 2; // 使用中间帧
        
        // 验证C, E, G的激活值比其他音符高
        let c_val = chroma_features[[0, frame_idx]];
        let e_val = chroma_features[[4, frame_idx]];
        let g_val = chroma_features[[7, frame_idx]];
        
        // 计算其他音符的平均值
        let mut other_sum = 0.0;
        let mut other_count = 0;
        
        for i in 0..n_chroma {
            if i != 0 && i != 4 && i != 7 {
                other_sum += chroma_features[[i, frame_idx]];
                other_count += 1;
            }
        }
        
        let other_avg = other_sum / other_count as f32;
        
        // C, E, G的激活度应该高于其他音符的平均值
        assert!(c_val > other_avg);
        assert!(e_val > other_avg);
        assert!(g_val > other_avg);
    }
    
    #[test]
    fn test_spectral_centroid() {
        let sample_rate = 22050;
        let duration = 0.2;
        
        // 低频信号
        let low_freq_signal = generate_sine_wave(duration, sample_rate, &[220.0], &[1.0]);
        
        // 高频信号
        let high_freq_signal = generate_sine_wave(duration, sample_rate, &[1760.0], &[1.0]);
        
        let n_fft = 2048;
        let hop_length = 512;
        
        let low_centroid = spectral_centroid_frames(&low_freq_signal, sample_rate, n_fft, hop_length, None).unwrap();
        let high_centroid = spectral_centroid_frames(&high_freq_signal, sample_rate, n_fft, hop_length, None).unwrap();
        
        // 高频信号的频谱质心应该高于低频信号
        let low_avg = low_centroid[low_centroid.len() / 2];
        let high_avg = high_centroid[high_centroid.len() / 2];
        
        assert!(high_avg > low_avg);
    }
} 