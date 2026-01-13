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

//! 音频信号生成器模块
//!
//! 提供各种音频信号生成函数，主要用于测试和合成目的，
//! 包括正弦波、白噪声、冲击信号等。

use std::f64::consts::PI;
use rand::distributions::{Distribution, Uniform};
use rand_distr::Normal;
use rand::thread_rng;

/// 生成正弦波
///
/// # 参数
/// * `frequency` - 频率 (Hz)
/// * `duration` - 持续时间 (秒)
/// * `sample_rate` - 采样率 (Hz)
/// * `amplitude` - 振幅 (可选，默认为1.0)
/// * `phase` - 相位 (可选，默认为0.0，弧度)
///
/// # 返回值
/// 返回生成的正弦波信号
pub fn generate_sine_wave(
    frequency: f64,
    duration: f64,
    sample_rate: usize,
    amplitude: Option<f64>,
    phase: Option<f64>,
) -> Vec<f64> {
    let amp = amplitude.unwrap_or(1.0);
    let ph = phase.unwrap_or(0.0);
    let num_samples = (duration * sample_rate as f64) as usize;
    let mut signal = Vec::with_capacity(num_samples);
    
    for i in 0..num_samples {
        let t = i as f64 / sample_rate as f64;
        let value = amp * (2.0 * PI * frequency * t + ph).sin();
        signal.push(value);
    }
    
    signal
}

/// 生成方波
///
/// # 参数
/// * `frequency` - 频率 (Hz)
/// * `duration` - 持续时间 (秒)
/// * `sample_rate` - 采样率 (Hz)
/// * `amplitude` - 振幅 (可选，默认为1.0)
/// * `duty_cycle` - 占空比 (可选，默认为0.5，范围0.0-1.0)
///
/// # 返回值
/// 返回生成的方波信号
pub fn generate_square_wave(
    frequency: f64,
    duration: f64,
    sample_rate: usize,
    amplitude: Option<f64>,
    duty_cycle: Option<f64>,
) -> Vec<f64> {
    let amp = amplitude.unwrap_or(1.0);
    let duty = duty_cycle.unwrap_or(0.5).clamp(0.0, 1.0);
    let num_samples = (duration * sample_rate as f64) as usize;
    let period_samples = (sample_rate as f64 / frequency) as usize;
    let mut signal = Vec::with_capacity(num_samples);
    
    for i in 0..num_samples {
        let cycle_position = i % period_samples;
        let value = if (cycle_position as f64 / period_samples as f64) < duty { amp } else { -amp };
        signal.push(value);
    }
    
    signal
}

/// 生成锯齿波
///
/// # 参数
/// * `frequency` - 频率 (Hz)
/// * `duration` - 持续时间 (秒)
/// * `sample_rate` - 采样率 (Hz)
/// * `amplitude` - 振幅 (可选，默认为1.0)
/// * `ascending` - 是否为上升锯齿 (可选，默认为true)
///
/// # 返回值
/// 返回生成的锯齿波信号
pub fn generate_sawtooth_wave(
    frequency: f64,
    duration: f64,
    sample_rate: usize,
    amplitude: Option<f64>,
    ascending: Option<bool>,
) -> Vec<f64> {
    let amp = amplitude.unwrap_or(1.0);
    let is_ascending = ascending.unwrap_or(true);
    let num_samples = (duration * sample_rate as f64) as usize;
    let period_samples = (sample_rate as f64 / frequency) as usize;
    let mut signal = Vec::with_capacity(num_samples);
    
    for i in 0..num_samples {
        let cycle_position = i % period_samples;
        let normalized_position = cycle_position as f64 / period_samples as f64;
        let value = if is_ascending {
            amp * (2.0 * normalized_position - 1.0)
        } else {
            amp * (1.0 - 2.0 * normalized_position)
        };
        signal.push(value);
    }
    
    signal
}

/// 生成三角波
///
/// # 参数
/// * `frequency` - 频率 (Hz)
/// * `duration` - 持续时间 (秒)
/// * `sample_rate` - 采样率 (Hz)
/// * `amplitude` - 振幅 (可选，默认为1.0)
///
/// # 返回值
/// 返回生成的三角波信号
pub fn generate_triangle_wave(
    frequency: f64,
    duration: f64,
    sample_rate: usize,
    amplitude: Option<f64>,
) -> Vec<f64> {
    let amp = amplitude.unwrap_or(1.0);
    let num_samples = (duration * sample_rate as f64) as usize;
    let period_samples = (sample_rate as f64 / frequency) as usize;
    let mut signal = Vec::with_capacity(num_samples);
    
    for i in 0..num_samples {
        let cycle_position = i % period_samples;
        let normalized_position = cycle_position as f64 / period_samples as f64;
        
        let value = if normalized_position < 0.5 {
            amp * (4.0 * normalized_position - 1.0)
        } else {
            amp * (3.0 - 4.0 * normalized_position)
        };
        
        signal.push(value);
    }
    
    signal
}

/// 生成高斯白噪声
///
/// # 参数
/// * `duration` - 持续时间 (秒)
/// * `sample_rate` - 采样率 (Hz)
/// * `mean` - 均值 (可选，默认为0.0)
/// * `std_dev` - 标准差 (可选，默认为1.0)
///
/// # 返回值
/// 返回生成的高斯白噪声信号
pub fn generate_white_noise(
    duration: f64,
    sample_rate: usize,
    mean: Option<f64>,
    std_dev: Option<f64>,
) -> Vec<f64> {
    let mu = mean.unwrap_or(0.0);
    let sigma = std_dev.unwrap_or(1.0);
    let num_samples = (duration * sample_rate as f64) as usize;
    let normal = Normal::new(mu, sigma).unwrap();
    let mut rng = thread_rng();
    
    let mut signal = Vec::with_capacity(num_samples);
    for _ in 0..num_samples {
        signal.push(normal.sample(&mut rng));
    }
    
    signal
}

/// 生成均匀分布噪声
///
/// # 参数
/// * `duration` - 持续时间 (秒)
/// * `sample_rate` - 采样率 (Hz)
/// * `min` - 最小值 (可选，默认为-1.0)
/// * `max` - 最大值 (可选，默认为1.0)
///
/// # 返回值
/// 返回生成的均匀分布噪声信号
pub fn generate_uniform_noise(
    duration: f64,
    sample_rate: usize,
    min: Option<f64>,
    max: Option<f64>,
) -> Vec<f64> {
    let low = min.unwrap_or(-1.0);
    let high = max.unwrap_or(1.0);
    let num_samples = (duration * sample_rate as f64) as usize;
    let uniform = Uniform::new(low, high);
    let mut rng = thread_rng();
    
    let mut signal = Vec::with_capacity(num_samples);
    for _ in 0..num_samples {
        signal.push(uniform.sample(&mut rng));
    }
    
    signal
}

/// 生成冲击信号
///
/// # 参数
/// * `duration` - 持续时间 (秒)
/// * `sample_rate` - 采样率 (Hz)
/// * `impulse_positions` - 冲击位置列表 (秒)
/// * `amplitude` - 冲击振幅 (可选，默认为1.0)
///
/// # 返回值
/// 返回生成的冲击信号
pub fn generate_impulse(
    duration: f64,
    sample_rate: usize,
    impulse_positions: &[f64],
    amplitude: Option<f64>,
) -> Vec<f64> {
    let amp = amplitude.unwrap_or(1.0);
    let num_samples = (duration * sample_rate as f64) as usize;
    let mut signal = vec![0.0; num_samples];
    
    for &pos in impulse_positions {
        if pos >= 0.0 && pos < duration {
            let sample_index = (pos * sample_rate as f64) as usize;
            if sample_index < num_samples {
                signal[sample_index] = amp;
            }
        }
    }
    
    signal
}

/// 生成多音调信号
///
/// # 参数
/// * `frequencies` - 频率列表 (Hz)
/// * `amplitudes` - 振幅列表
/// * `duration` - 持续时间 (秒)
/// * `sample_rate` - 采样率 (Hz)
///
/// # 返回值
/// 返回生成的多音调信号
pub fn generate_multitone(
    frequencies: &[f64],
    amplitudes: &[f64],
    duration: f64,
    sample_rate: usize,
) -> Vec<f64> {
    assert_eq!(frequencies.len(), amplitudes.len(), "频率和振幅数组长度必须相同");
    
    let num_samples = (duration * sample_rate as f64) as usize;
    let mut signal = vec![0.0; num_samples];
    
    for i in 0..frequencies.len() {
        let freq = frequencies[i];
        let amp = amplitudes[i];
        
        for j in 0..num_samples {
            let t = j as f64 / sample_rate as f64;
            signal[j] += amp * (2.0 * PI * freq * t).sin();
        }
    }
    
    signal
}

/// 生成简单音频事件
///
/// # 参数
/// * `sample_rate` - 采样率 (Hz)
/// * `num_frames` - 帧数
/// * `hop_length` - 帧移
///
/// # 返回值
/// 返回一个包含信号变化事件的音频信号
pub fn generate_audio_events(
    sample_rate: usize,
    num_frames: usize,
    hop_length: usize,
) -> Vec<f64> {
    let duration = (num_frames * hop_length) as f64 / sample_rate as f64;
    let total_samples = (duration * sample_rate as f64) as usize;
    let mut signal = vec![0.0; total_samples];
    
    // 事件1：正弦波
    let event1_start = 0;
    let event1_end = (0.25 * total_samples as f64) as usize;
    let event1 = generate_sine_wave(440.0, (event1_end - event1_start) as f64 / sample_rate as f64, sample_rate, Some(0.5), None);
    
    // 事件2：方波
    let event2_start = (0.25 * total_samples as f64) as usize;
    let event2_end = (0.5 * total_samples as f64) as usize;
    let event2 = generate_square_wave(220.0, (event2_end - event2_start) as f64 / sample_rate as f64, sample_rate, Some(0.5), None);
    
    // 事件3：噪声
    let event3_start = (0.5 * total_samples as f64) as usize;
    let event3_end = (0.75 * total_samples as f64) as usize;
    let event3 = generate_white_noise((event3_end - event3_start) as f64 / sample_rate as f64, sample_rate, None, Some(0.5));
    
    // 事件4：锯齿波
    let event4_start = (0.75 * total_samples as f64) as usize;
    let event4_end = total_samples;
    let event4 = generate_sawtooth_wave(330.0, (event4_end - event4_start) as f64 / sample_rate as f64, sample_rate, Some(0.5), None);
    
    // 组合所有事件
    for i in event1_start..event1_end {
        signal[i] = event1[i - event1_start];
    }
    
    for i in event2_start..event2_end {
        signal[i] = event2[i - event2_start];
    }
    
    for i in event3_start..event3_end {
        signal[i] = event3[i - event3_start];
    }
    
    for i in event4_start..event4_end {
        signal[i] = event4[i - event4_start];
    }
    
    signal
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    
    #[test]
    fn test_sine_wave() {
        // 生成1秒、1kHz的正弦波，采样率8kHz
        let frequency = 1000.0;
        let duration = 1.0;
        let sample_rate = 8000;
        let sine = generate_sine_wave(frequency, duration, sample_rate, None, None);
        
        // 检查长度
        assert_eq!(sine.len(), (duration * sample_rate as f64) as usize);
        
        // 检查振幅
        let max_value = sine.iter().fold(0.0, |max, &x| max.max(x.abs()));
        assert_relative_eq!(max_value, 1.0, epsilon = 1e-6);
        
        // 检查周期性（第一个峰值应该在1/1000秒）
        let peak_index = sine
            .iter()
            .enumerate()
            .take((sample_rate / frequency) as usize)
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap()
            .0;
        
        let expected_peak = (sample_rate as f64 / (frequency * 4.0)).round() as usize; // 四分之一周期
        assert!((peak_index as isize - expected_peak as isize).abs() <= 1);
    }
    
    #[test]
    fn test_white_noise() {
        // 生成1秒白噪声，采样率8kHz
        let duration = 1.0;
        let sample_rate = 8000;
        let mean = 0.0;
        let std_dev = 1.0;
        let noise = generate_white_noise(duration, sample_rate, Some(mean), Some(std_dev));
        
        // 检查长度
        assert_eq!(noise.len(), (duration * sample_rate as f64) as usize);
        
        // 检查统计特性（大样本统计近似）
        let sum: f64 = noise.iter().sum();
        let mean_value = sum / noise.len() as f64;
        
        let variance_sum: f64 = noise.iter().map(|&x| (x - mean_value).powi(2)).sum();
        let variance = variance_sum / noise.len() as f64;
        let calc_std_dev = variance.sqrt();
        
        // 允许一定的统计误差
        assert_relative_eq!(mean_value, mean, epsilon = 0.1);
        assert_relative_eq!(calc_std_dev, std_dev, epsilon = 0.1);
    }
    
    #[test]
    fn test_audio_events() {
        let sample_rate = 16000;
        let num_frames = 100;
        let hop_length = 512;
        
        let events = generate_audio_events(sample_rate, num_frames, hop_length);
        
        // 检查长度
        let expected_length = num_frames * hop_length;
        assert_eq!(events.len(), expected_length);
        
        // 检查第一个事件段（正弦波）
        let first_section_end = (0.25 * events.len() as f64) as usize;
        let first_section = &events[0..first_section_end];
        
        // 检查最后一个事件段（锯齿波）
        let last_section_start = (0.75 * events.len() as f64) as usize;
        let last_section = &events[last_section_start..];
        
        // 验证不同事件段的信号特性不同
        let first_section_mean: f64 = first_section.iter().sum::<f64>() / first_section.len() as f64;
        let last_section_mean: f64 = last_section.iter().sum::<f64>() / last_section.len() as f64;
        
        // 由于信号特性不同，均值应有差异
        assert!(first_section_mean != last_section_mean);
    }
} 