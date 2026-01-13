use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use crate::data::multimodal::extractors::audio::error::AudioError;
#[cfg(feature = "multimodal")]
use reqwest::blocking::Client;
use std::fs::File;
use std::io::Read;
use base64::{Engine as _, engine::general_purpose::STANDARD as BASE64};
use std::time::Duration;

/// 支持的音频文件扩展名
const SUPPORTED_AUDIO_EXTENSIONS: &[&str] = &[
    "wav", "mp3", "flac", "aac", "ogg", "m4a", "opus", "wma", "aiff"
];

/// 音频数据源的类型，用于表示和处理不同来源的音频数据
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AudioSource {
    /// 本地文件路径
    File(PathBuf),
    
    /// 远程URL链接
    URL(String),
    
    /// Base64编码的音频数据
    Base64(String),
    
    /// 原始音频数据（浮点数样本），包含采样率和通道数
    RawSamples {
        /// 音频样本数据，每个浮点数表示一个样本
        samples: Vec<f32>,
        /// 采样率（Hz）
        sample_rate: u32,
        /// 通道数（1=单声道，2=立体声）
        channels: u16,
    },
    
    /// 字节数组格式的音频数据
    Bytes(Vec<u8>),
    
    /// 从标准输入读取音频数据
    Stdin,
    
    /// 从麦克风实时读取
    Microphone {
        /// 采样率（Hz）
        sample_rate: u32,
        /// 录制持续时间（毫秒）
        duration_ms: u32,
        /// 通道数（1=单声道，2=立体声）
        channels: u16,
    },
}

impl AudioSource {
    /// 从文件路径创建音频源
    pub fn from_file<P: Into<PathBuf>>(path: P) -> Self {
        AudioSource::File(path.into())
    }
    
    /// 从URL创建音频源
    pub fn from_url<S: Into<String>>(url: S) -> Self {
        AudioSource::URL(url.into())
    }
    
    /// 从Base64编码的字符串创建音频源
    pub fn from_base64<S: Into<String>>(base64_str: S) -> Self {
        AudioSource::Base64(base64_str.into())
    }
    
    /// 从原始音频样本创建音频源
    pub fn from_raw_samples(samples: Vec<f32>, sample_rate: u32, channels: u16) -> Self {
        AudioSource::RawSamples { samples, sample_rate, channels }
    }
    
    /// 从字节数组创建音频源
    pub fn from_bytes(bytes: Vec<u8>) -> Self {
        AudioSource::Bytes(bytes)
    }
    
    /// 创建从标准输入读取的音频源
    pub fn from_stdin() -> Self {
        AudioSource::Stdin
    }
    
    /// 创建从麦克风读取的音频源
    pub fn from_microphone(sample_rate: u32, duration_ms: u32, channels: u16) -> Self {
        AudioSource::Microphone { sample_rate, duration_ms, channels }
    }
    
    /// 验证音频源是否有效
    pub fn validate(&self) -> Result<(), AudioError> {
        match self {
            AudioSource::File(path) => {
                if !path.exists() {
                    return Err(AudioError::FileNotFound(format!("文件不存在: {:?}", path)));
                }
                
                let extension = path.extension()
                    .and_then(|ext| ext.to_str())
                    .map(|s| s.to_lowercase());
                
                if let Some(ext) = extension {
                    if !SUPPORTED_AUDIO_EXTENSIONS.iter().any(|&valid_ext| valid_ext == ext) {
                        return Err(AudioError::UnsupportedFormat(format!("不支持的文件格式: {}", ext)));
                    }
                } else {
                    return Err(AudioError::UnsupportedFormat("无法确定文件扩展名".to_string()));
                }
                
                Ok(())
            },
            AudioSource::URL(url) => {
                if !url.starts_with("http://") && !url.starts_with("https://") {
                    return Err(AudioError::InvalidURL(format!("URL必须以http://或https://开头: {}", url)));
                }
                
                let lowercase_url = url.to_lowercase();
                let has_valid_extension = SUPPORTED_AUDIO_EXTENSIONS.iter()
                    .any(|&ext| lowercase_url.ends_with(&format!(".{}", ext)));
                    
                if !has_valid_extension {
                    return Err(AudioError::UnsupportedFormat("URL不包含有效的音频文件扩展名".to_string()));
                }
                
                Ok(())
            },
            AudioSource::Base64(data) => {
                if data.trim().is_empty() {
                    return Err(AudioError::EmptyData("Base64字符串为空".to_string()));
                }
                
                // 简单验证是否为有效的Base64字符串
                if data.len() % 4 != 0 || !data.chars().all(|c| {
                    c.is_ascii_alphanumeric() || c == '+' || c == '/' || c == '='
                }) {
                    return Err(AudioError::InvalidBase64("无效的Base64编码".to_string()));
                }
                
                Ok(())
            },
            AudioSource::RawSamples { samples, sample_rate, channels } => {
                if samples.is_empty() {
                    return Err(AudioError::EmptyData("音频样本为空".to_string()));
                }
                
                if *sample_rate == 0 {
                    return Err(AudioError::InvalidSampleRate("采样率不能为0".to_string()));
                }
                
                if *channels == 0 {
                    return Err(AudioError::InvalidChannelCount("通道数不能为0".to_string()));
                }
                
                Ok(())
            },
            AudioSource::Bytes(bytes) => {
                if bytes.is_empty() {
                    return Err(AudioError::EmptyData("字节数组为空".to_string()));
                }
                
                Ok(())
            },
            AudioSource::Stdin => {
                // 无需验证，在实际读取时才能确定是否有效
                Ok(())
            },
            AudioSource::Microphone { sample_rate, duration_ms, channels } => {
                if *sample_rate == 0 {
                    return Err(AudioError::InvalidSampleRate("采样率不能为0".to_string()));
                }
                
                if *duration_ms == 0 {
                    return Err(AudioError::InvalidConfig("录制时长不能为0".to_string()));
                }
                
                if *channels == 0 {
                    return Err(AudioError::InvalidChannelCount("通道数不能为0".to_string()));
                }
                
                Ok(())
            },
        }
    }
    
    /// 检查文件扩展名是否受支持
    pub fn is_supported_extension(extension: &str) -> bool {
        let ext = extension.to_lowercase();
        SUPPORTED_AUDIO_EXTENSIONS.iter().any(|&valid_ext| valid_ext == ext)
    }
    
    /// 获取源的描述信息
    pub fn get_description(&self) -> String {
        match self {
            AudioSource::File(path) => format!("文件: {:?}", path),
            AudioSource::URL(url) => format!("URL: {}", url),
            AudioSource::Base64(_) => "Base64编码音频".to_string(),
            AudioSource::RawSamples { sample_rate, channels, samples } => {
                format!("原始样本: {}Hz, {}通道, {}样本", sample_rate, channels, samples.len())
            },
            AudioSource::Bytes(bytes) => format!("字节数组: {}字节", bytes.len()),
            AudioSource::Stdin => "标准输入".to_string(),
            AudioSource::Microphone { sample_rate, duration_ms, channels } => {
                format!("麦克风: {}Hz, {}通道, {}毫秒", sample_rate, channels, duration_ms)
            },
        }
    }
    
    /// 获取源类型的名称
    pub fn get_type_name(&self) -> &'static str {
        match self {
            AudioSource::File(_) => "file",
            AudioSource::URL(_) => "url",
            AudioSource::Base64(_) => "base64",
            AudioSource::RawSamples { .. } => "raw_samples",
            AudioSource::Bytes(_) => "bytes",
            AudioSource::Stdin => "stdin",
            AudioSource::Microphone { .. } => "microphone",
        }
    }

    /// 从音频源读取原始字节数据
    pub fn load_bytes(&self) -> Result<Vec<u8>, AudioError> {
        match self {
            AudioSource::File(path) => {
                let mut file = File::open(path).map_err(|e| {
                    AudioError::IOError(e)
                })?;
                let mut buffer = Vec::new();
                file.read_to_end(&mut buffer).map_err(|e| {
                    AudioError::IOError(e)
                })?;
                Ok(buffer)
            },
            AudioSource::URL(url) => {
                let client = Client::builder()
                    .timeout(Duration::from_secs(30))
                    .build()
                    .map_err(|e| AudioError::NetworkError(e.to_string()))?;
                
                let response = client.get(url)
                    .send()
                    .map_err(|e| AudioError::NetworkError(e.to_string()))?;
                
                let status = response.status();
                if !status.is_success() {
                    return Err(AudioError::NetworkError(format!(
                        "HTTP请求失败: 状态码 {}", status
                    )));
                }
                
                let bytes = response.bytes()
                    .map_err(|e| AudioError::NetworkError(e.to_string()))?;
                
                Ok(bytes.to_vec())
            },
            AudioSource::Base64(encoded) => {
                let bytes = BASE64.decode(encoded.as_bytes())
                    .map_err(|e| AudioError::InvalidBase64(e.to_string()))?;
                Ok(bytes)
            },
            AudioSource::Bytes(bytes) => {
                Ok(bytes.clone())
            },
        }
    }
} 