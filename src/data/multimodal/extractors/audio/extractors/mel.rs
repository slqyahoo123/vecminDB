//! 梅尔谱图特征提取器实现
//! 
//! 梅尔谱图是一种音频特征，它模拟了人类听觉系统对不同频率的音频信号的响应。
//! 梅尔谱图通常用于语音识别、音乐分析和声音分类等应用。

#[cfg(feature = "multimodal")]
use ndarray::Array2;
#[cfg(feature = "multimodal")]
use crate::Result;
#[cfg(feature = "multimodal")]
use crate::Error;
#[cfg(feature = "multimodal")]
use super::{
    apply_fft, 
    compute_power_spectrum, 
    create_mel_filterbank, 
    apply_hanning_window, 
    frame_signal
};

/// 梅尔谱图特征提取器
pub struct MelSpectrogramExtractor {
    /// 梅尔滤波器数量
    n_mels: usize,
    /// FFT窗口大小
    n_fft: usize,
    /// 帧长度
    frame_size: usize,
    /// 帧移步长
    hop_size: usize,
    /// 是否归一化
    normalize: bool,
    /// 采样率
    sample_rate: u32,
    /// 预加重因子
    preemphasis: f32,
    /// 功率缩放因子
    power: f32,
    /// 是否应用对数变换
    log_mel: bool,
    /// 最小频率
    f_min: f32,
    /// 最大频率
    f_max: Option<f32>,
}

impl Default for MelSpectrogramExtractor {
    fn default() -> Self {
        Self {
            n_mels: 128,
            n_fft: 2048,
            frame_size: 512,
            hop_size: 256,
            normalize: true,
            sample_rate: 44100,
            preemphasis: 0.97,
            power: 2.0,  // 功率谱
            log_mel: true,
            f_min: 0.0,
            f_max: None, // 默认为Nyquist频率
        }
    }
}

impl MelSpectrogramExtractor {
    /// 创建新的梅尔谱图特征提取器
    pub fn new(
        n_mels: usize,
        n_fft: usize,
        frame_size: usize,
        hop_size: usize,
        normalize: bool,
        sample_rate: u32,
        preemphasis: f32,
        power: f32,
        log_mel: bool,
        f_min: f32,
        f_max: Option<f32>,
    ) -> Self {
        Self {
            n_mels,
            n_fft,
            frame_size,
            hop_size,
            normalize,
            sample_rate,
            preemphasis,
            power,
            log_mel,
            f_min,
            f_max,
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
    
    /// 提取梅尔谱图特征
    #[cfg(feature = "multimodal")]
    pub fn extract(&self, samples: &[f32]) -> Result<Array2<f32>> {
        // 1. 预加重
        let preemphasized = self.apply_preemphasis(samples);
        
        // 2. 分帧
        let frames = frame_signal(&preemphasized, self.frame_size, self.hop_size);
        if frames.is_empty() {
            return Err(Error::data("音频太短，无法提取足够的帧".to_string()));
        }
        
        // 3. 加窗
        let windowed_frames: Vec<Vec<f32>> = frames.iter()
            .map(|frame| apply_hanning_window(frame))
            .collect();
        
        // 4. 对每一帧应用短时傅里叶变换(STFT)
        let stft_frames: Vec<Vec<f32>> = windowed_frames.iter()
            .map(|frame| {
                let fft_result = apply_fft(frame, self.n_fft);
                
                // 计算功率谱或幅度谱
                let spectrum = if self.power == 2.0 {
                    // 功率谱
                    compute_power_spectrum(&fft_result)
                } else if self.power == 1.0 {
                    // 幅度谱
                    fft_result.iter().map(|c| c.norm()).collect()
                } else {
                    // 其他幂次
                    fft_result.iter().map(|c| c.norm().powf(self.power)).collect()
                };
                
                // 只保留前一半（由于对称性）
                spectrum[..self.n_fft/2 + 1].to_vec()
            })
            .collect();
        
        // 5. 创建梅尔滤波器组
        let mel_filterbank = create_mel_filterbank(self.n_mels, self.n_fft, self.sample_rate);
        
        // 6. 将谱图应用到梅尔滤波器组
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
        
        // 7. 如果需要，应用对数变换
        let result_spectrogram = if self.log_mel {
            mel_spectrogram.mapv(|x| x.ln())
        } else {
            mel_spectrogram
        };
        
        // 8. 如果需要，进行归一化
        let mut normalized_spectrogram = result_spectrogram;
        if self.normalize {
            // 全局最大-最小值归一化
            let mut min_val = f32::MAX;
            let mut max_val = f32::MIN;
            
            for &val in normalized_spectrogram.iter() {
                if val < min_val {
                    min_val = val;
                }
                if val > max_val {
                    max_val = val;
                }
            }
            
            let range = max_val - min_val;
            if range > 1e-10 {
                normalized_spectrogram.mapv_inplace(|x| (x - min_val) / range);
            }
        }
        
        Ok(normalized_spectrogram)
    }
} 