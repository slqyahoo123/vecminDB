use std::fmt;
use std::io;
use std::error::Error;
#[cfg(feature = "multimodal")]
use realfft::FftError;

/// 音频处理过程中可能出现的错误
#[derive(Debug)]
pub enum AudioError {
    /// 音频文件未找到
    FileNotFound(String),
    /// 文件读取错误
    IOError(io::Error),
    /// 不支持的音频格式
    UnsupportedFormat(String),
    /// 解码错误
    DecodingError(String),
    /// 音频处理错误
    ProcessingError(String),
    /// 特征提取错误
    ExtractionError(String),
    /// 模型加载错误
    ModelLoadError(String),
    /// 音频数据为空
    EmptyData(String),
    /// 无效的配置
    InvalidConfig(String),
    /// 无效的URL
    InvalidURL(String),
    /// 网络错误
    NetworkError(String),
    /// 无效的Base64编码
    InvalidBase64(String),
    /// 采样率错误
    InvalidSampleRate(String),
    /// 通道数错误
    InvalidChannelCount(String),
    /// 不支持的操作
    UnsupportedOperation(String),
    /// 特征类型错误
    InvalidFeatureType(String),
    /// 特征维度错误
    InvalidDimension(String),
    /// 设备错误（如麦克风）
    DeviceError(String),
    /// 未实现的功能
    Unimplemented(String),
    /// 其他错误
    Other(String),
    /// FFT错误
    FftError(String),
}

impl fmt::Display for AudioError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            AudioError::FileNotFound(msg) => write!(f, "文件未找到: {}", msg),
            AudioError::IOError(err) => write!(f, "IO错误: {}", err),
            AudioError::UnsupportedFormat(msg) => write!(f, "不支持的格式: {}", msg),
            AudioError::DecodingError(msg) => write!(f, "解码错误: {}", msg),
            AudioError::ProcessingError(msg) => write!(f, "处理错误: {}", msg),
            AudioError::ExtractionError(msg) => write!(f, "特征提取错误: {}", msg),
            AudioError::ModelLoadError(msg) => write!(f, "模型加载错误: {}", msg),
            AudioError::EmptyData(msg) => write!(f, "空数据: {}", msg),
            AudioError::InvalidConfig(msg) => write!(f, "无效配置: {}", msg),
            AudioError::InvalidURL(msg) => write!(f, "无效URL: {}", msg),
            AudioError::NetworkError(msg) => write!(f, "网络错误: {}", msg),
            AudioError::InvalidBase64(msg) => write!(f, "无效Base64: {}", msg),
            AudioError::InvalidSampleRate(msg) => write!(f, "无效采样率: {}", msg),
            AudioError::InvalidChannelCount(msg) => write!(f, "无效通道数: {}", msg),
            AudioError::UnsupportedOperation(msg) => write!(f, "不支持的操作: {}", msg),
            AudioError::InvalidFeatureType(msg) => write!(f, "无效特征类型: {}", msg),
            AudioError::InvalidDimension(msg) => write!(f, "无效维度: {}", msg),
            AudioError::DeviceError(msg) => write!(f, "设备错误: {}", msg),
            AudioError::Unimplemented(msg) => write!(f, "未实现: {}", msg),
            AudioError::Other(msg) => write!(f, "其他错误: {}", msg),
            AudioError::FftError(msg) => write!(f, "FFT错误: {}", msg),
        }
    }
}

impl Error for AudioError {}

impl From<io::Error> for AudioError {
    fn from(err: io::Error) -> Self {
        AudioError::IOError(err)
    }
}

impl From<&str> for AudioError {
    fn from(msg: &str) -> Self {
        AudioError::Other(msg.to_string())
    }
}

impl From<String> for AudioError {
    fn from(msg: String) -> Self {
        AudioError::Other(msg)
    }
}

// 从其他可能的错误类型转换
impl From<base64::DecodeError> for AudioError {
    fn from(err: base64::DecodeError) -> Self {
        AudioError::InvalidBase64(err.to_string())
    }
}

impl From<reqwest::Error> for AudioError {
    fn from(err: reqwest::Error) -> Self {
        AudioError::NetworkError(err.to_string())
    }
}

#[cfg(feature = "multimodal")]
impl From<symphonia::core::errors::Error> for AudioError {
    fn from(err: symphonia::core::errors::Error) -> Self {
        AudioError::DecodingError(err.to_string())
    }
}

impl From<ndarray::ShapeError> for AudioError {
    fn from(err: ndarray::ShapeError) -> Self {
        AudioError::InvalidDimension(err.to_string())
    }
}

#[cfg(feature = "multimodal")]
impl From<FftError> for AudioError {
    fn from(err: FftError) -> Self {
        AudioError::FftError(format!("FFT error: {}", err))
    }
}

/// 判断错误是否为临时性错误，可以重试
pub fn is_temporary_error(err: &AudioError) -> bool {
    match err {
        AudioError::NetworkError(_) | 
        AudioError::IOError(_) | 
        AudioError::DeviceError(_) => true,
        AudioError::IOError(io_err) => {
            matches!(io_err.kind(), 
                io::ErrorKind::ConnectionRefused | 
                io::ErrorKind::ConnectionReset | 
                io::ErrorKind::ConnectionAborted | 
                io::ErrorKind::NotConnected | 
                io::ErrorKind::TimedOut |
                io::ErrorKind::Interrupted |
                io::ErrorKind::WouldBlock
            )
        },
        _ => false,
    }
}

/// 提供详细的错误上下文信息
pub fn error_with_context(error: AudioError, context: &str) -> AudioError {
    match error {
        AudioError::FileNotFound(msg) => AudioError::FileNotFound(format!("{} - {}", context, msg)),
        AudioError::IOError(err) => AudioError::IOError(io::Error::new(err.kind(), format!("{} - {}", context, err))),
        AudioError::UnsupportedFormat(msg) => AudioError::UnsupportedFormat(format!("{} - {}", context, msg)),
        AudioError::DecodingError(msg) => AudioError::DecodingError(format!("{} - {}", context, msg)),
        AudioError::ProcessingError(msg) => AudioError::ProcessingError(format!("{} - {}", context, msg)),
        AudioError::ExtractionError(msg) => AudioError::ExtractionError(format!("{} - {}", context, msg)),
        AudioError::ModelLoadError(msg) => AudioError::ModelLoadError(format!("{} - {}", context, msg)),
        AudioError::EmptyData(msg) => AudioError::EmptyData(format!("{} - {}", context, msg)),
        AudioError::InvalidConfig(msg) => AudioError::InvalidConfig(format!("{} - {}", context, msg)),
        AudioError::InvalidURL(msg) => AudioError::InvalidURL(format!("{} - {}", context, msg)),
        AudioError::NetworkError(msg) => AudioError::NetworkError(format!("{} - {}", context, msg)),
        AudioError::InvalidBase64(msg) => AudioError::InvalidBase64(format!("{} - {}", context, msg)),
        AudioError::InvalidSampleRate(msg) => AudioError::InvalidSampleRate(format!("{} - {}", context, msg)),
        AudioError::InvalidChannelCount(msg) => AudioError::InvalidChannelCount(format!("{} - {}", context, msg)),
        AudioError::UnsupportedOperation(msg) => AudioError::UnsupportedOperation(format!("{} - {}", context, msg)),
        AudioError::InvalidFeatureType(msg) => AudioError::InvalidFeatureType(format!("{} - {}", context, msg)),
        AudioError::InvalidDimension(msg) => AudioError::InvalidDimension(format!("{} - {}", context, msg)),
        AudioError::DeviceError(msg) => AudioError::DeviceError(format!("{} - {}", context, msg)),
        AudioError::Unimplemented(msg) => AudioError::Unimplemented(format!("{} - {}", context, msg)),
        AudioError::Other(msg) => AudioError::Other(format!("{} - {}", context, msg)),
        AudioError::FftError(msg) => AudioError::FftError(format!("{} - {}", context, msg)),
    }
} 