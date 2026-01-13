use serde::{Deserialize, Serialize};
use std::fmt;
use std::str::FromStr;

/// 音频特征类型枚举
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum AudioFeatureType {
    /// 梅尔频率倒谱系数，用于语音识别和音乐分类
    MFCC,
    /// 梅尔频谱图，将语音信号转换为基于人类听觉系统的视觉表示
    MelSpectrogram,
    /// 频谱质心，表示频谱的"重心"
    SpectralCentroid,
    /// 频谱带宽，表示频谱的分散程度
    SpectralBandwidth,
    /// 频谱衰减，表示频谱的衰减速率
    SpectralRolloff,
    /// 零交叉率，衡量信号从正值变为负值的频率
    ZeroCrossingRate,
    /// 色度特征，将频谱投影到12个半音上
    Chroma,
    /// 色度常量量化特征，加强不变性
    ChromaCQT,
    /// 色度能量归一化特征
    ChromaENS,
    /// 音调同步色度特征
    ChromaCENS,
    /// 调和噪声比，测量音频中和声与噪声的比率
    HarmonicRatio,
    /// 感知重量，关联到声音的"重量"感知
    PerceptualWeights,
    /// 声波特征，基于波形提取的特征
    Waveform,
    /// 能量特征，表示音频的能量分布
    Energy,
    /// RMS能量，音频的根均方能量
    RMSEnergy,
    /// 音调特征，提取音频的音调信息
    Pitch,
    /// 音谱平坦度，测量频谱的平滑程度
    SpectralFlatness,
    /// 音谱对比度，比较高低频能量
    SpectralContrast,
    /// 音谱通量，测量频谱的变化率
    SpectralFlux,
    /// 音谱熵，测量频谱中的信息量
    SpectralEntropy,
    /// 线性预测系数，用于语音编码和合成
    LPC,
    /// 线性预测倒谱系数
    LPCC,
    /// 自定义特征，允许用户定义自己的特征提取方法
    Custom(String),
}

impl Default for AudioFeatureType {
    fn default() -> Self {
        AudioFeatureType::MFCC
    }
}

impl fmt::Display for AudioFeatureType {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            AudioFeatureType::MFCC => write!(f, "MFCC"),
            AudioFeatureType::MelSpectrogram => write!(f, "MelSpectrogram"),
            AudioFeatureType::SpectralCentroid => write!(f, "SpectralCentroid"),
            AudioFeatureType::SpectralBandwidth => write!(f, "SpectralBandwidth"),
            AudioFeatureType::SpectralRolloff => write!(f, "SpectralRolloff"),
            AudioFeatureType::ZeroCrossingRate => write!(f, "ZeroCrossingRate"),
            AudioFeatureType::Chroma => write!(f, "Chroma"),
            AudioFeatureType::ChromaCQT => write!(f, "ChromaCQT"),
            AudioFeatureType::ChromaENS => write!(f, "ChromaENS"),
            AudioFeatureType::ChromaCENS => write!(f, "ChromaCENS"),
            AudioFeatureType::HarmonicRatio => write!(f, "HarmonicRatio"),
            AudioFeatureType::PerceptualWeights => write!(f, "PerceptualWeights"),
            AudioFeatureType::Waveform => write!(f, "Waveform"),
            AudioFeatureType::Energy => write!(f, "Energy"),
            AudioFeatureType::RMSEnergy => write!(f, "RMSEnergy"),
            AudioFeatureType::Pitch => write!(f, "Pitch"),
            AudioFeatureType::SpectralFlatness => write!(f, "SpectralFlatness"),
            AudioFeatureType::SpectralContrast => write!(f, "SpectralContrast"),
            AudioFeatureType::SpectralFlux => write!(f, "SpectralFlux"),
            AudioFeatureType::SpectralEntropy => write!(f, "SpectralEntropy"),
            AudioFeatureType::LPC => write!(f, "LPC"),
            AudioFeatureType::LPCC => write!(f, "LPCC"),
            AudioFeatureType::Custom(name) => write!(f, "Custom({})", name),
        }
    }
}

impl FromStr for AudioFeatureType {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_uppercase().as_str() {
            "MFCC" => Ok(AudioFeatureType::MFCC),
            "MELSPECTROGRAM" => Ok(AudioFeatureType::MelSpectrogram),
            "SPECTRALCENTROID" => Ok(AudioFeatureType::SpectralCentroid),
            "SPECTRALBANDWIDTH" => Ok(AudioFeatureType::SpectralBandwidth),
            "SPECTRALROLLOFF" => Ok(AudioFeatureType::SpectralRolloff),
            "ZEROCROSSINGRATE" => Ok(AudioFeatureType::ZeroCrossingRate),
            "CHROMA" => Ok(AudioFeatureType::Chroma),
            "CHROMACQT" => Ok(AudioFeatureType::ChromaCQT),
            "CHROMAENS" => Ok(AudioFeatureType::ChromaENS),
            "CHROMACENS" => Ok(AudioFeatureType::ChromaCENS),
            "HARMONICRATIO" => Ok(AudioFeatureType::HarmonicRatio),
            "PERCEPTUALWEIGHTS" => Ok(AudioFeatureType::PerceptualWeights),
            "WAVEFORM" => Ok(AudioFeatureType::Waveform),
            "ENERGY" => Ok(AudioFeatureType::Energy),
            "RMSENERGY" => Ok(AudioFeatureType::RMSEnergy),
            "PITCH" => Ok(AudioFeatureType::Pitch),
            "SPECTRALFLATNESS" => Ok(AudioFeatureType::SpectralFlatness),
            "SPECTRALCONTRAST" => Ok(AudioFeatureType::SpectralContrast),
            "SPECTRALFLUX" => Ok(AudioFeatureType::SpectralFlux),
            "SPECTRALENTROPY" => Ok(AudioFeatureType::SpectralEntropy),
            "LPC" => Ok(AudioFeatureType::LPC),
            "LPCC" => Ok(AudioFeatureType::LPCC),
            _ => {
                if s.to_uppercase().starts_with("CUSTOM(") && s.ends_with(')') {
                    let name = s[7..s.len()-1].to_string();
                    Ok(AudioFeatureType::Custom(name))
                } else {
                    Err(format!("未知的音频特征类型: {}", s))
                }
            }
        }
    }
}

/// 特征输出维度配置
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeatureDimensions {
    /// MFCC特征维度配置
    pub mfcc: usize,
    /// 梅尔频谱图特征维度配置
    pub mel_spectrogram: usize,
    /// 色度特征维度配置
    pub chroma: usize,
    /// 频谱特征维度配置
    pub spectral: usize,
    /// 时域特征维度配置
    pub temporal: usize,
    /// 音调特征维度配置
    pub pitch: usize,
    /// 自定义特征维度配置
    pub custom: usize,
}

impl Default for FeatureDimensions {
    fn default() -> Self {
        Self {
            mfcc: 20,           // 默认20维MFCC特征
            mel_spectrogram: 128, // 默认128个梅尔滤波器
            chroma: 12,         // 默认12个色度类别
            spectral: 7,        // 默认7个频谱特征
            temporal: 3,        // 默认3个时域特征
            pitch: 2,           // 默认2个音调特征
            custom: 64,         // 默认64维自定义特征
        }
    }
}

impl FeatureDimensions {
    /// 获取指定特征类型的输出维度
    pub fn get_dimension(&self, feature_type: &AudioFeatureType) -> usize {
        match feature_type {
            AudioFeatureType::MFCC => self.mfcc,
            AudioFeatureType::MelSpectrogram => self.mel_spectrogram,
            AudioFeatureType::Chroma | 
            AudioFeatureType::ChromaCQT | 
            AudioFeatureType::ChromaENS | 
            AudioFeatureType::ChromaCENS => self.chroma,
            AudioFeatureType::SpectralCentroid | 
            AudioFeatureType::SpectralBandwidth | 
            AudioFeatureType::SpectralRolloff |
            AudioFeatureType::SpectralFlatness |
            AudioFeatureType::SpectralContrast |
            AudioFeatureType::SpectralFlux |
            AudioFeatureType::SpectralEntropy => self.spectral,
            AudioFeatureType::ZeroCrossingRate | 
            AudioFeatureType::Energy | 
            AudioFeatureType::RMSEnergy => self.temporal,
            AudioFeatureType::Pitch |
            AudioFeatureType::HarmonicRatio => self.pitch,
            AudioFeatureType::LPC |
            AudioFeatureType::LPCC => 13, // 典型的LPC和LPCC维度
            AudioFeatureType::PerceptualWeights => 24, // 典型的感知权重维度
            AudioFeatureType::Waveform => 256, // 波形样本数
            AudioFeatureType::Custom(_) => self.custom,
        }
    }
} 