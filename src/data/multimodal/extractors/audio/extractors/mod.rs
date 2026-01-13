//! 音频特征提取器实现模块
//!
//! 该模块包含各种音频特征提取算法的具体实现，
//! 包括MFCC、梅尔谱图、色度特征等。

pub mod mfcc;
pub mod mel;
pub mod spectral;
pub mod temporal;

// 重导出主要类型
pub use self::mfcc::MFCCExtractor;
pub use self::mel::MelSpectrogramExtractor;
pub use self::spectral::{
    ChromaExtractor,
    SpectralCentroidExtractor,
    SpectralRolloffExtractor,
    ZeroCrossingRateExtractor,
    EnergyExtractor,
};
pub use self::temporal::TemporalFeaturesExtractor;

// 工具函数
use ndarray::Array2;
#[cfg(feature = "multimodal")]
use rustfft::{FftPlanner, num_complex::Complex};

/// 应用快速傅里叶变换（FFT）到输入信号
#[cfg(feature = "multimodal")]
pub fn apply_fft(samples: &[f32], window_size: usize) -> Vec<Complex<f32>> {
    let mut planner = FftPlanner::new();
    let fft = planner.plan_fft_forward(window_size);
    
    // 准备FFT输入
    let mut fft_input: Vec<Complex<f32>> = samples.iter()
        .map(|&s| Complex::new(s, 0.0))
        .collect();
        
    // 如果输入长度不足window_size，则用零填充
    fft_input.resize(window_size, Complex::new(0.0, 0.0));
    
    // 执行FFT
    fft.process(&mut fft_input);
    
    fft_input
}

/// 计算功率谱
#[cfg(feature = "multimodal")]
pub fn compute_power_spectrum(fft_output: &[Complex<f32>]) -> Vec<f32> {
    fft_output.iter()
        .map(|c| c.norm_sqr())
        .collect()
}

/// 计算梅尔频率
pub fn hz_to_mel(hz: f32) -> f32 {
    2595.0 * (1.0 + hz / 700.0).log10()
}

/// 梅尔频率转赫兹
pub fn mel_to_hz(mel: f32) -> f32 {
    700.0 * (10.0_f32.powf(mel / 2595.0) - 1.0)
}

/// 生成梅尔滤波器组
pub fn create_mel_filterbank(n_filters: usize, fft_size: usize, sample_rate: u32) -> Array2<f32> {
    let nyquist = sample_rate as f32 / 2.0;
    let min_mel = hz_to_mel(0.0);
    let max_mel = hz_to_mel(nyquist);
    
    // 均匀间隔的梅尔点
    let mel_points: Vec<f32> = (0..=n_filters+1)
        .map(|i| min_mel + (max_mel - min_mel) * (i as f32) / (n_filters as f32 + 1.0))
        .collect();
    
    // 转换回赫兹
    let hz_points: Vec<f32> = mel_points.iter()
        .map(|&mel| mel_to_hz(mel))
        .collect();
    
    // 将赫兹点转换为FFT bin索引
    let bin_indices: Vec<usize> = hz_points.iter()
        .map(|&hz| ((fft_size as f32 + 1.0) * hz / sample_rate as f32).round() as usize)
        .map(|idx| idx.min(fft_size / 2))
        .collect();
    
    // 创建滤波器组
    let mut filterbank = Array2::zeros((n_filters, fft_size / 2 + 1));
    
    for i in 0..n_filters {
        for j in bin_indices[i]..=bin_indices[i+2] {
            if j >= fft_size / 2 + 1 {
                break;
            }
            
            if j >= bin_indices[i] && j <= bin_indices[i+1] {
                // 上升边
                filterbank[[i, j]] = (j as f32 - bin_indices[i] as f32) / 
                                     (bin_indices[i+1] as f32 - bin_indices[i] as f32);
            } else if j >= bin_indices[i+1] && j <= bin_indices[i+2] {
                // 下降边
                filterbank[[i, j]] = (bin_indices[i+2] as f32 - j as f32) / 
                                     (bin_indices[i+2] as f32 - bin_indices[i+1] as f32);
            }
        }
        
        // 归一化滤波器（可选）
        let sum: f32 = filterbank.row(i).sum();
        if sum > 0.0 {
            for j in 0..fft_size / 2 + 1 {
                filterbank[[i, j]] /= sum;
            }
        }
    }
    
    filterbank
}

/// 应用汉宁窗
#[cfg(feature = "multimodal")]
pub fn apply_hanning_window(samples: &[f32]) -> Vec<f32> {
    let n = samples.len();
    samples.iter()
        .enumerate()
        .map(|(i, &s)| {
            let window = 0.5 * (1.0 - (2.0 * std::f32::consts::PI * i as f32 / (n as f32 - 1.0)).cos());
            s * window
        })
        .collect()
}

/// 应用分帧处理
#[cfg(feature = "multimodal")]
pub fn frame_signal(signal: &[f32], frame_size: usize, hop_size: usize) -> Vec<Vec<f32>> {
    let n_frames = ((signal.len() - frame_size) as f32 / hop_size as f32).ceil() as usize + 1;
    let mut frames = Vec::with_capacity(n_frames);
    
    for i in 0..n_frames {
        let start = i * hop_size;
        if start >= signal.len() {
            break;
        }
        
        let end = (start + frame_size).min(signal.len());
        if end - start < frame_size / 2 {
            // 如果帧太短，则跳过
            break;
        }
        
        let mut frame = signal[start..end].to_vec();
        // 如果帧长度不足，用零填充
        frame.resize(frame_size, 0.0);
        frames.push(frame);
    }
    
    frames
} 