use serde::{Deserialize, Serialize};
use crate::data::multimodal::extractors::audio::types::AudioFeatureType;

/// 音频处理配置
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AudioProcessingConfig {
    /// 音频采样率，单位Hz
    pub sample_rate: usize,
    /// 音频通道数，1=单声道，2=立体声
    pub channels: usize,
    /// 帧长度，单位采样点数
    pub frame_length: usize,
    /// 帧步长，单位采样点数
    pub hop_length: usize,
    /// 要提取的特征类型
    pub feature_type: AudioFeatureType,
    /// FFT窗口大小
    pub n_fft: Option<usize>,
    /// Mel滤波器数量
    pub n_mels: Option<usize>,
    /// MFCC系数数量
    pub n_mfcc: Option<usize>,
    /// 色度特征的音乐音调数量
    pub n_chroma: Option<usize>,
    /// 谐波特征的迭代次数
    pub harmonic_iterations: Option<usize>,
    /// 特征规范化选项
    pub normalize: bool,
    /// 下采样到目标采样率
    pub target_sr: Option<usize>,
    /// 自定义提取器ID
    pub custom_extractor_id: Option<String>,
    /// 自定义提取器参数
    pub custom_params: Option<serde_json::Value>,
}

impl Default for AudioProcessingConfig {
    fn default() -> Self {
        Self {
            sample_rate: 44100,
            channels: 1,
            frame_length: 1024,
            hop_length: 512,
            feature_type: AudioFeatureType::MelSpectrogram,
            n_fft: Some(2048),
            n_mels: Some(128),
            n_mfcc: Some(20),
            n_chroma: Some(12),
            harmonic_iterations: Some(4),
            normalize: true,
            target_sr: None,
            custom_extractor_id: None,
            custom_params: None,
        }
    }
}

impl AudioProcessingConfig {
    /// 创建新的配置实例
    pub fn new(feature_type: AudioFeatureType) -> Self {
        Self {
            feature_type,
            ..Default::default()
        }
    }
    
    /// 设置采样率
    pub fn with_sample_rate(mut self, sample_rate: usize) -> Self {
        self.sample_rate = sample_rate;
        self
    }

    /// 设置通道数
    pub fn with_channels(mut self, channels: usize) -> Self {
        self.channels = channels;
        self
    }

    /// 设置帧长度
    pub fn with_frame_length(mut self, frame_length: usize) -> Self {
        self.frame_length = frame_length;
        self
    }

    /// 设置帧步长
    pub fn with_hop_length(mut self, hop_length: usize) -> Self {
        self.hop_length = hop_length;
        self
    }

    /// 设置FFT窗口大小
    pub fn with_n_fft(mut self, n_fft: usize) -> Self {
        self.n_fft = Some(n_fft);
        self
    }

    /// 设置Mel滤波器数量
    pub fn with_n_mels(mut self, n_mels: usize) -> Self {
        self.n_mels = Some(n_mels);
        self
    }

    /// 设置MFCC系数数量
    pub fn with_n_mfcc(mut self, n_mfcc: usize) -> Self {
        self.n_mfcc = Some(n_mfcc);
        self
    }

    /// 设置色度特征的音乐音调数量
    pub fn with_n_chroma(mut self, n_chroma: usize) -> Self {
        self.n_chroma = Some(n_chroma);
        self
    }

    /// 设置谐波特征的迭代次数
    pub fn with_harmonic_iterations(mut self, iterations: usize) -> Self {
        self.harmonic_iterations = Some(iterations);
        self
    }

    /// 设置是否规范化
    pub fn with_normalize(mut self, normalize: bool) -> Self {
        self.normalize = normalize;
        self
    }

    /// 设置目标采样率
    pub fn with_target_sr(mut self, target_sr: usize) -> Self {
        self.target_sr = Some(target_sr);
        self
    }

    /// 设置自定义提取器ID和参数
    pub fn with_custom_extractor(mut self, id: String, params: Option<serde_json::Value>) -> Self {
        self.custom_extractor_id = Some(id);
        self.custom_params = params;
        self
    }
    
    /// 获取特征维度
    pub fn get_feature_dimension(&self) -> usize {
        match self.feature_type {
            AudioFeatureType::MFCC => self.n_mfcc.unwrap_or(20) as usize,
            AudioFeatureType::MelSpectrogram => {
                if let Some(mel_config) = &self.n_mels {
                    *mel_config as usize
                } else {
                    128 // 默认值
                }
            },
            AudioFeatureType::Custom(_) => 1024, // 默认假设，实际由自定义提取器确定
            AudioFeatureType::Chroma | 
            AudioFeatureType::ChromaCQT | 
            AudioFeatureType::ChromaENS | 
            AudioFeatureType::ChromaCENS => self.n_chroma.unwrap_or(12),
            AudioFeatureType::SpectralCentroid => 1,
            AudioFeatureType::SpectralRolloff => 1,
            AudioFeatureType::ZeroCrossingRate => 1,
            AudioFeatureType::Energy => 1,
            AudioFeatureType::RMSEnergy => 1,
            AudioFeatureType::Waveform => self.frame_length,
            AudioFeatureType::SpectralBandwidth => 1,
            AudioFeatureType::SpectralContrast => 7, // 默认7个频带
            AudioFeatureType::SpectralFlatness => 1,
            AudioFeatureType::SpectralFlux => 1,
            AudioFeatureType::SpectralEntropy => 1,
            AudioFeatureType::HarmonicRatio => 1,
            AudioFeatureType::Pitch => 1,
            AudioFeatureType::LPC => 13,
            AudioFeatureType::LPCC => 13,
            AudioFeatureType::PerceptualWeights => 24,
        }
    }
} 