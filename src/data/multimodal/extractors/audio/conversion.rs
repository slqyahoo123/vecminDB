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

//! 音频信号转换相关功能
//!
//! 提供各种音频信号转换工具函数，如频率转换、分贝转换、
//! 信号分帧等。

use ndarray::{Array1, Array2, Axis};
#[cfg(feature = "multimodal")]
use rustfft::{FftPlanner, num_complex::Complex64};
use std::f64::consts::PI;

/// 将Hz频率转换为Mel刻度
///
/// # 参数
/// * `hz` - 赫兹频率
///
/// # 返回值
/// 返回对应的Mel刻度值
pub fn hz_to_mel(hz: f64) -> f64 {
    2595.0 * (1.0 + hz / 700.0).log10()
}

/// 将Mel刻度转换为Hz频率
///
/// # 参数
/// * `mel` - Mel刻度值
///
/// # 返回值
/// 返回对应的赫兹频率
pub fn mel_to_hz(mel: f64) -> f64 {
    700.0 * (10.0_f64.powf(mel / 2595.0) - 1.0)
}

/// 将幅度值转换为分贝
///
/// # 参数
/// * `amplitude` - 幅度值
/// * `ref_value` - 参考值，默认为1.0
/// * `min_db` - 最小分贝值，低于此值将被裁剪
///
/// # 返回值
/// 返回对应的分贝值
pub fn amplitude_to_db(
    amplitude: &[f64],
    ref_value: Option<f64>,
    min_db: Option<f64>,
) -> Array1<f64> {
    let reference = ref_value.unwrap_or(1.0);
    let min_db_val = min_db.unwrap_or(-80.0);
    
    let mut db_values = Vec::with_capacity(amplitude.len());
    
    for &amp in amplitude {
        if amp > 0.0 {
            let db = 20.0 * (amp / reference).log10();
            db_values.push(db.max(min_db_val));
        } else {
            db_values.push(min_db_val);
        }
    }
    
    Array1::from(db_values)
}

/// 将二维幅度谱转换为分贝
///
/// # 参数
/// * `spectrogram` - 二维幅度谱
/// * `ref_value` - 参考值，默认为1.0
/// * `min_db` - 最小分贝值，低于此值将被裁剪
///
/// # 返回值
/// 返回对应的分贝谱
pub fn spectrogram_to_db(
    spectrogram: &Array2<f64>,
    ref_value: Option<f64>,
    min_db: Option<f64>,
) -> Array2<f64> {
    let reference = ref_value.unwrap_or(1.0);
    let min_db_val = min_db.unwrap_or(-80.0);
    
    let mut db_spec = Array2::zeros(spectrogram.dim());
    
    for ((i, j), &amp) in spectrogram.indexed_iter() {
        if amp > 0.0 {
            let db = 20.0 * (amp / reference).log10();
            db_spec[[i, j]] = db.max(min_db_val);
        } else {
            db_spec[[i, j]] = min_db_val;
        }
    }
    
    db_spec
}

/// 将音频信号分帧
///
/// # 参数
/// * `signal` - 输入音频信号
/// * `frame_length` - 帧长度
/// * `hop_length` - 帧移样本数量
///
/// # 返回值
/// 返回分帧后的二维数组，每行为一帧
pub fn frame_signal(
    signal: &[f64],
    frame_length: usize,
    hop_length: usize,
) -> Array2<f64> {
    let signal_length = signal.len();
    let num_frames = 1 + (signal_length - frame_length) / hop_length;
    
    let mut frames = Array2::zeros((num_frames, frame_length));
    
    for i in 0..num_frames {
        let start = i * hop_length;
        let end = start + frame_length;
        
        if end <= signal_length {
            for j in 0..frame_length {
                frames[[i, j]] = signal[start + j];
            }
        }
    }
    
    frames
}

/// 离散傅里叶变换包装函数
///
/// # 参数
/// * `signal` - 输入信号
///
/// # 返回值
/// 返回复数形式的DFT结果
#[cfg(feature = "multimodal")]
pub fn compute_fft(signal: &[f64]) -> Vec<Complex64> {
    let mut planner = FftPlanner::new();
    let fft = planner.plan_fft_forward(signal.len());
    
    // 将实信号转换为复信号
    let mut complex_signal: Vec<Complex64> = signal
        .iter()
        .map(|&x| Complex64::new(x, 0.0))
        .collect();
    
    // 执行FFT
    fft.process(&mut complex_signal);
    
    complex_signal
}

/// 离散余弦变换
///
/// # 参数
/// * `signal` - 输入信号
/// * `n_coeffs` - 需要保留的系数数量
///
/// # 返回值
/// 返回DCT结果
pub fn dct(signal: &[f64], n_coeffs: usize) -> Array1<f64> {
    let n = signal.len();
    let mut result = Vec::with_capacity(n_coeffs);
    
    for k in 0..n_coeffs {
        let mut sum = 0.0;
        for n_idx in 0..n {
            sum += signal[n_idx] * ((PI * (n_idx as f64 + 0.5) * k as f64) / n as f64).cos();
        }
        
        let alpha = if k == 0 { (1.0 / n as f64).sqrt() } else { (2.0 / n as f64).sqrt() };
        result.push(alpha * sum);
    }
    
    Array1::from(result)
}

/// 短时傅里叶变换
///
/// # 参数
/// * `signal` - 输入信号
/// * `window` - 窗函数
/// * `n_fft` - FFT点数
/// * `hop_length` - 帧移样本数量
///
/// # 返回值
/// 返回STFT的复数结果
#[cfg(feature = "multimodal")]
pub fn stft(
    signal: &[f64],
    window: &[f64],
    n_fft: usize,
    hop_length: usize,
) -> Array2<Complex64> {
    let n_frames = 1 + (signal.len() - window.len()) / hop_length;
    let mut result = Array2::zeros((n_frames, n_fft / 2 + 1));
    
    for frame_idx in 0..n_frames {
        let start = frame_idx * hop_length;
        let end = start + window.len();
        
        if end <= signal.len() {
            // 应用窗函数
            let mut windowed_frame = Vec::with_capacity(n_fft);
            for i in 0..window.len() {
                windowed_frame.push(signal[start + i] * window[i]);
            }
            
            // 填充零
            windowed_frame.resize(n_fft, 0.0);
            
            // 计算FFT
            let fft_result = compute_fft(&windowed_frame);
            
            // 只保留正频率部分（含直流分量）
            for k in 0..=n_fft/2 {
                result[[frame_idx, k]] = fft_result[k];
            }
        }
    }
    
    result
}

/// 计算幅度谱
///
/// # 参数
/// * `stft_result` - STFT结果
///
/// # 返回值
/// 返回幅度谱
#[cfg(feature = "multimodal")]
pub fn magnitude_spectrum(stft_result: &Array2<Complex64>) -> Array2<f64> {
    let (n_frames, n_freq) = stft_result.dim();
    let mut magnitude = Array2::zeros((n_frames, n_freq));
    
    for i in 0..n_frames {
        for j in 0..n_freq {
            magnitude[[i, j]] = stft_result[[i, j]].norm();
        }
    }
    
    magnitude
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use crate::data::multimodal::extractors::audio::generators::generate_sine_wave;
    use crate::data::multimodal::extractors::audio::filter::hanning_window;
    
    #[test]
    fn test_hz_to_mel_conversion() {
        // 测试几个标准频率点
        assert_relative_eq!(hz_to_mel(0.0), 0.0, epsilon = 1e-6);
        assert_relative_eq!(hz_to_mel(1000.0), 1000.0.log10() * 2595.0, epsilon = 1e-6);
        
        // 测试转换的可逆性
        let freq = 440.0;
        let mel = hz_to_mel(freq);
        let back_to_hz = mel_to_hz(mel);
        assert_relative_eq!(freq, back_to_hz, epsilon = 1e-6);
    }
    
    #[test]
    fn test_amplitude_to_db() {
        // 测试基本转换
        let amplitudes = vec![1.0, 0.1, 0.01, 0.001];
        let db_values = amplitude_to_db(&amplitudes, None, None);
        
        // 验证转换结果
        assert_relative_eq!(db_values[0], 0.0, epsilon = 1e-6); // 1.0 -> 0 dB
        assert_relative_eq!(db_values[1], -20.0, epsilon = 1e-6); // 0.1 -> -20 dB
        assert_relative_eq!(db_values[2], -40.0, epsilon = 1e-6); // 0.01 -> -40 dB
        assert_relative_eq!(db_values[3], -60.0, epsilon = 1e-6); // 0.001 -> -60 dB
    }
    
    #[test]
    fn test_frame_signal() {
        // 创建测试信号
        let signal: Vec<f64> = (0..10).map(|x| x as f64).collect();
        
        // 参数设置
        let frame_length = 5;
        let hop_length = 2;
        
        // 分帧
        let frames = frame_signal(&signal, frame_length, hop_length);
        
        // 验证帧数
        assert_eq!(frames.len_of(Axis(0)), 3);
        
        // 验证第一帧的内容
        for i in 0..frame_length {
            assert_eq!(frames[[0, i]], i as f64);
        }
        
        // 验证第二帧的内容
        for i in 0..frame_length {
            assert_eq!(frames[[1, i]], (i + hop_length) as f64);
        }
    }
    
    #[test]
    fn test_stft_magnitude() {
        // 创建正弦波
        let freq = 100.0; // Hz
        let sample_rate = 1000; // Hz
        let duration = 1.0; // 秒
        let signal = generate_sine_wave(freq, duration, sample_rate);
        
        // STFT参数
        let window_size = 256;
        let hop_length = 128;
        let n_fft = 512;
        
        // 创建汉宁窗
        let window = hanning_window(window_size);
        
        // 执行STFT
        let stft_result = stft(&signal, &window, n_fft, hop_length);
        
        // 计算幅度谱
        let magnitude = magnitude_spectrum(&stft_result);
        
        // 检查维度
        let (n_frames, n_freq) = magnitude.dim();
        assert_eq!(n_freq, n_fft / 2 + 1);
        
        // 理论上最大幅度应该在100Hz对应的频率点附近
        let freq_bin = (freq / sample_rate as f64 * n_fft as f64).round() as usize;
        
        // 找到第一帧中幅度最大的频率点
        let mut max_bin = 0;
        let mut max_val = 0.0;
        
        for bin in 1..n_freq {
            if magnitude[[0, bin]] > max_val {
                max_val = magnitude[[0, bin]];
                max_bin = bin;
            }
        }
        
        // 检查最大幅度是否出现在预期的频率点附近
        assert!((max_bin as isize - freq_bin as isize).abs() <= 1);
    }
}

/// 使用轴操作进行音频数据转换
///
/// # 参数
/// * `audio_data` - 音频数据数组
/// * `axis` - 操作的轴
///
/// # 返回值
/// 返回转换后的音频数据
pub fn process_audio_with_axis(audio_data: &Array2<f64>, axis: Axis) -> Result<Array2<f64>, Box<dyn std::error::Error>> {
    match axis {
        Axis(0) => {
            // 沿第一个轴（行）进行处理
            let mut processed = Array2::zeros(audio_data.dim());
            for (i, row) in audio_data.axis_iter(axis).enumerate() {
                let mean = row.mean().unwrap_or(0.0);
                let std = row.std(1.0);
                if std > 0.0 {
                    for (j, &x) in row.iter().enumerate() {
                        processed[[i, j]] = (x - mean) / std;
                    }
                } else {
                    for (j, &x) in row.iter().enumerate() {
                        processed[[i, j]] = x;
                    }
                }
            }
            Ok(processed)
        },
        Axis(1) => {
            // 沿第二个轴（列）进行处理
            let mut processed = Array2::zeros(audio_data.dim());
            for (j, col) in audio_data.axis_iter(axis).enumerate() {
                let mean = col.mean().unwrap_or(0.0);
                let std = col.std(1.0);
                if std > 0.0 {
                    for (i, &x) in col.iter().enumerate() {
                        processed[[i, j]] = (x - mean) / std;
                    }
                } else {
                    for (i, &x) in col.iter().enumerate() {
                        processed[[i, j]] = x;
                    }
                }
            }
            Ok(processed)
        },
        _ => Err("不支持的轴操作".into())
    }
}

/// 使用轴操作进行音频数据分帧
///
/// # 参数
/// * `audio_data` - 音频数据数组
/// * `frame_size` - 帧大小
/// * `hop_size` - 跳跃大小
///
/// # 返回值
/// 返回分帧后的音频数据
pub fn frame_audio_with_axis(audio_data: &Array1<f64>, frame_size: usize, hop_size: usize) -> Result<Array2<f64>, Box<dyn std::error::Error>> {
    if audio_data.len() < frame_size {
        return Err("音频数据长度小于帧大小".into());
    }
    
    let num_frames = (audio_data.len() - frame_size) / hop_size + 1;
    let mut frames = Array2::zeros((num_frames, frame_size));
    
    for (i, frame) in frames.axis_iter_mut(Axis(0)).enumerate() {
        let start = i * hop_size;
        let end = start + frame_size;
        if end <= audio_data.len() {
            frame.assign(&audio_data.slice(ndarray::s![start..end]));
        }
    }
    
    Ok(frames)
}

/// 使用轴操作进行音频数据窗口化
///
/// # 参数
/// * `audio_data` - 音频数据数组
/// * `window_type` - 窗口类型
///
/// # 返回值
/// 返回窗口化后的音频数据
pub fn window_audio_with_axis(audio_data: &Array2<f64>, window_type: &str) -> Result<Array2<f64>, Box<dyn std::error::Error>> {
    let window = match window_type {
        "hann" => create_hann_window(audio_data.ncols()),
        "hamming" => create_hamming_window(audio_data.ncols()),
        "blackman" => create_blackman_window(audio_data.ncols()),
        _ => return Err("不支持的窗口类型".into())
    };
    
    let mut windowed = Array2::zeros(audio_data.dim());
    for (col_idx, frame) in audio_data.axis_iter(Axis(1)).enumerate() {
        let frame_owned = frame.to_owned();
        let multiplied = &frame_owned * &window;
        for (row_idx, &val) in multiplied.iter().enumerate() {
            windowed[[row_idx, col_idx]] = val;
        }
    }
    
    Ok(windowed)
}

/// 创建汉宁窗口
fn create_hann_window(size: usize) -> Array1<f64> {
    Array1::from_iter((0..size).map(|i| {
        0.5 * (1.0 - (2.0 * PI * i as f64 / (size - 1) as f64).cos())
    }))
}

/// 创建汉明窗口
fn create_hamming_window(size: usize) -> Array1<f64> {
    Array1::from_iter((0..size).map(|i| {
        0.54 - 0.46 * (2.0 * PI * i as f64 / (size - 1) as f64).cos()
    }))
}

/// 创建布莱克曼窗口
fn create_blackman_window(size: usize) -> Array1<f64> {
    Array1::from_iter((0..size).map(|i| {
        0.42 - 0.5 * (2.0 * PI * i as f64 / (size - 1) as f64).cos() 
            + 0.08 * (4.0 * PI * i as f64 / (size - 1) as f64).cos()
    }))
} 