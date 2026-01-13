/// 视频提取错误类型
#[derive(Debug)]
pub enum VideoExtractionError {
    /// 视频文件读取错误
    FileError(String),
    /// 视频解码错误
    DecodeError(String),
    /// 特征提取错误
    ExtractionError(String),
    /// 配置错误
    ConfigError(String),
    /// 系统资源错误
    ResourceError(String),
    /// 模型加载错误
    ModelError(String),
    /// 未知错误
    Unknown(String),
    /// 通用错误
    GenericError(String),
}

impl fmt::Display for VideoExtractionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::FileError(msg) => write!(f, "文件错误: {}", msg),
            Self::DecodeError(msg) => write!(f, "解码错误: {}", msg),
            Self::ExtractionError(msg) => write!(f, "特征提取错误: {}", msg),
            Self::ConfigError(msg) => write!(f, "配置错误: {}", msg),
            Self::ResourceError(msg) => write!(f, "资源错误: {}", msg),
            Self::ModelError(msg) => write!(f, "模型错误: {}", msg),
            Self::Unknown(msg) => write!(f, "未知错误: {}", msg),
            Self::GenericError(msg) => write!(f, "通用错误: {}", msg),
        }
    }
}

impl std::error::Error for VideoExtractionError {}

/// 从标准IO错误转换
impl From<std::io::Error> for VideoExtractionError {
    fn from(err: std::io::Error) -> Self {
        VideoExtractionError::FileError(format!("IO错误: {}", err))
    }
}

/// 从字符串类型转换
impl From<String> for VideoExtractionError {
    fn from(err: String) -> Self {
        VideoExtractionError::Unknown(err)
    }
}

/// 从字符串常量转换
impl From<&str> for VideoExtractionError {
    fn from(err: &str) -> Self {
        VideoExtractionError::Unknown(err.to_string())
    }
}

/// 从自定义错误转换
impl<E: std::error::Error + Send + Sync + 'static> From<Box<E>> for VideoExtractionError {
    fn from(err: Box<E>) -> Self {
        VideoExtractionError::GenericError(err.to_string())
    }
}

/// 从标准库错误转换
impl From<Error> for VideoExtractionError {
    fn from(err: Error) -> Self {
        VideoExtractionError::GenericError(err.to_string())
    }
} 