//! MFCC特征提取器实现
//! 
//! 梅尔频率倒谱系数(MFCC)是一种广泛用于语音和音频处理的特征。
//! 它通过模拟人类听觉系统来表示音频信号的短期能量谱。

use ndarray::{Array2, Axis};
use crate::Result;
#[cfg(feature = "multimodal")]
use super::{
    apply_fft, 
    compute_power_spectrum, 
    create_mel_filterbank, 
    apply_hanning_window, 
    frame_signal
};

/// MFCC特征提取器
pub struct MFCCExtractor {
    /// MFCC系数数量
    n_mfcc: usize,
    /// 梅尔滤波器数量
    n_mels: usize,
    /// FFT窗口大小
    n_fft: usize,
    /// 帧长度
    frame_size: usize,
    /// 帧移步长
    hop_size: usize,
    /// 是否包含能量
    include_energy: bool,
    /// 是否归一化
    normalize: bool,
    /// 使用DCT类型（1, 2, 3）
    dct_type: u8,
    /// 采样率
    sample_rate: u32,
    /// 预加重因子
    preemphasis: f32,
}

impl Default for MFCCExtractor {
    fn default() -> Self {
        Self {
            n_mfcc: 13,
            n_mels: 26,
            n_fft: 2048,
            frame_size: 512,
            hop_size: 256,
            include_energy: true,
            normalize: true,
            dct_type: 2,
            sample_rate: 44100,
            preemphasis: 0.97,
        }
    }
}

impl MFCCExtractor {
    /// 创建新的MFCC特征提取器
    pub fn new(
        n_mfcc: usize,
        n_mels: usize,
        n_fft: usize,
        frame_size: usize,
        hop_size: usize,
        include_energy: bool,
        normalize: bool,
        dct_type: u8,
        sample_rate: u32,
        preemphasis: f32,
    ) -> Self {
        Self {
            n_mfcc,
            n_mels,
            n_fft,
            frame_size,
            hop_size,
            include_energy,
            normalize,
            dct_type,
            sample_rate,
            preemphasis,
        }
    }
    
    /// 应用预加重滤波
    fn apply_preemphasis(&self, signal: &[f32]) -> Vec<f32> {
        if self.preemphasis <= 0.0 {
            return signal.to_vec();
        }
        
        let mut output = Vec::with_capacity(signal.len());
        output.push(signal[0]);
        
        for i in 1..signal.len() {
            output.push(signal[i] - self.preemphasis * signal[i-1]);
        }
        
        output
    }
    
    /// 应用离散余弦变换(DCT)
    fn apply_dct(&self, log_mel_spectrogram: &Array2<f32>) -> Array2<f32> {
        let n_frames = log_mel_spectrogram.shape()[0];
        let n_mels = log_mel_spectrogram.shape()[1];
        let mut dct_result = Array2::zeros((n_frames, self.n_mfcc));
        
        // 计算DCT变换矩阵
        let mut dct_matrix = Array2::zeros((self.n_mfcc, n_mels));
        
        // 选择DCT类型
        match self.dct_type {
            1 => {
                // DCT-I
                for i in 0..self.n_mfcc {
                    let factor = if i == 0 || i == n_mels - 1 { 1.0 } else { 2.0 };
                    let norm = (factor / (2.0 * (n_mels - 1) as f32)).sqrt();
                    
                    for j in 0..n_mels {
                        dct_matrix[[i, j]] = norm * 
                            (std::f32::consts::PI * i as f32 * j as f32 / (n_mels - 1) as f32).cos();
                    }
                }
            },
            2 => {
                // DCT-II（最常用）
                for i in 0..self.n_mfcc {
                    let norm = if i == 0 {
                        (1.0 / n_mels as f32).sqrt()
                    } else {
                        (2.0 / n_mels as f32).sqrt()
                    };
                    
                    for j in 0..n_mels {
                        dct_matrix[[i, j]] = norm * 
                            (std::f32::consts::PI * i as f32 * (j as f32 + 0.5) / n_mels as f32).cos();
                    }
                }
            },
            3 => {
                // DCT-III
                for i in 0..self.n_mfcc {
                    let norm = if i == 0 {
                        (1.0 / (2.0 * n_mels as f32)).sqrt()
                    } else {
                        (1.0 / n_mels as f32).sqrt()
                    };
                    
                    for j in 0..n_mels {
                        dct_matrix[[i, j]] = norm * 
                            (std::f32::consts::PI * (i as f32 + 0.5) * j as f32 / n_mels as f32).cos();
                    }
                }
            },
            _ => {
                // 默认使用DCT-II
                for i in 0..self.n_mfcc {
                    let norm = if i == 0 {
                        (1.0 / n_mels as f32).sqrt()
                    } else {
                        (2.0 / n_mels as f32).sqrt()
                    };
                    
                    for j in 0..n_mels {
                        dct_matrix[[i, j]] = norm * 
                            (std::f32::consts::PI * i as f32 * (j as f32 + 0.5) / n_mels as f32).cos();
                    }
                }
            }
        }
        
        // 应用DCT变换
        for (i, frame) in log_mel_spectrogram.axis_iter(Axis(0)).enumerate() {
            for j in 0..self.n_mfcc {
                let mut sum = 0.0;
                for k in 0..n_mels {
                    sum += frame[k] * dct_matrix[[j, k]];
                }
                dct_result[[i, j]] = sum;
            }
        }
        
        dct_result
    }
    
    /// 提取MFCC特征
    #[cfg(feature = "multimodal")]
    pub fn extract(&self, samples: &[f32]) -> Result<Array2<f32>> {
        // 1. 预加重
        let preemphasized = self.apply_preemphasis(samples);
        
        // 2. 分帧
        let frames = frame_signal(&preemphasized, self.frame_size, self.hop_size);
        
        // 3. 加窗
        let windowed_frames: Vec<Vec<f32>> = frames.iter()
            .map(|frame| apply_hanning_window(frame))
            .collect();
        
        // 4. 对每一帧应用短时傅里叶变换(STFT)
        let stft_frames: Vec<Vec<f32>> = windowed_frames.iter()
            .map(|frame| {
                let fft_result = apply_fft(frame, self.n_fft);
                let power_spectrum = compute_power_spectrum(&fft_result);
                // 只保留前一半（由于对称性）
                power_spectrum[..self.n_fft/2 + 1].to_vec()
            })
            .collect();
        
        // 5. 计算梅尔滤波器组
        let mel_filterbank = create_mel_filterbank(self.n_mels, self.n_fft, self.sample_rate);
        
        // 6. 将功率谱应用到梅尔滤波器组
        let mut mel_spectrogram = Array2::zeros((frames.len(), self.n_mels));
        
        for (i, spectrum) in stft_frames.iter().enumerate() {
            for j in 0..self.n_mels {
                let mut mel_energy = 0.0;
                for k in 0..spectrum.len() {
                    mel_energy += spectrum[k] * mel_filterbank[[j, k]];
                }
                // 避免log(0)
                mel_spectrogram[[i, j]] = mel_energy.max(1e-10);
            }
        }
        
        // 7. 取对数
        let log_mel_spectrogram = mel_spectrogram.mapv(|x| x.ln());
        
        // 8. 应用离散余弦变换
        let mut mfcc = self.apply_dct(&log_mel_spectrogram);
        
        // 9. 如果需要，添加能量
        if self.include_energy {
            // 计算每帧的能量
            let energies: Vec<f32> = frames.iter()
                .map(|frame| {
                    let energy = frame.iter().map(|&s| s * s).sum::<f32>();
                    energy.ln().max(-100.0) // 避免 -inf
                })
                .collect();
            
            // 将能量作为第一个MFCC系数
            for (i, &energy) in energies.iter().enumerate() {
                // 将所有系数右移，让能量成为第一个系数
                for j in (1..self.n_mfcc).rev() {
                    mfcc[[i, j]] = mfcc[[i, j-1]];
                }
                mfcc[[i, 0]] = energy;
            }
        }
        
        // 10. 如果需要，进行归一化
        if self.normalize {
            // 每列（每个系数）减去均值
            for j in 0..self.n_mfcc {
                let mean = mfcc.column(j).mean().unwrap_or(0.0);
                let std_dev = mfcc.column(j).std(0.0);
                
                if std_dev > 1e-10 {
                    for i in 0..mfcc.shape()[0] {
                        mfcc[[i, j]] = (mfcc[[i, j]] - mean) / std_dev;
                    }
                }
            }
        }
        
        Ok(mfcc)
    }
} 