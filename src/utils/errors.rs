use std::fmt;
use std::io;
use std::error::Error as StdError;

/// 项目通用错误类型
#[derive(Debug)]
pub enum Error {
    /// 数据错误
    Data(String),
    /// 模型错误
    Model(String),
    /// IO错误
    Io(io::Error),
    /// 序列化错误
    Serialization(String),
    /// 参数错误
    Parameter(String),
    /// 锁错误
    Lock(String),
    /// 验证错误
    Validation(String),
    /// 未找到错误
    NotFound(String),
    /// 其他错误
    Other(String),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::Data(msg) => write!(f, "Data error: {}", msg),
            Error::Model(msg) => write!(f, "Model error: {}", msg),
            Error::Io(err) => write!(f, "IO error: {}", err),
            Error::Serialization(msg) => write!(f, "Serialization error: {}", msg),
            Error::Parameter(msg) => write!(f, "Parameter error: {}", msg),
            Error::Lock(msg) => write!(f, "Lock error: {}", msg),
            Error::Validation(msg) => write!(f, "Validation error: {}", msg),
            Error::NotFound(msg) => write!(f, "Not found: {}", msg),
            Error::Other(msg) => write!(f, "Error: {}", msg),
        }
    }
}

impl StdError for Error {
    fn source(&self) -> Option<&(dyn StdError + 'static)> {
        match self {
            Error::Io(err) => Some(err),
            _ => None,
        }
    }
}

impl From<io::Error> for Error {
    fn from(err: io::Error) -> Self {
        Error::Io(err)
    }
}

impl From<serde_json::Error> for Error {
    fn from(err: serde_json::Error) -> Self {
        Error::Serialization(err.to_string())
    }
}

impl From<bincode::Error> for Error {
    fn from(err: bincode::Error) -> Self {
        Error::Serialization(err.to_string())
    }
}

impl Error {
    /// 创建数据错误
    pub fn data<T: Into<String>>(msg: T) -> Self {
        Error::Data(msg.into())
    }

    /// 创建模型错误
    pub fn model<T: Into<String>>(msg: T) -> Self {
        Error::Model(msg.into())
    }

    /// 创建参数错误
    pub fn parameter<T: Into<String>>(msg: T) -> Self {
        Error::Parameter(msg.into())
    }

    /// 创建锁错误
    pub fn lock<T: Into<String>>(msg: T) -> Self {
        Error::Lock(msg.into())
    }

    /// 创建验证错误
    pub fn validation<T: Into<String>>(msg: T) -> Self {
        Error::Validation(msg.into())
    }

    /// 创建未找到错误
    pub fn not_found<T: Into<String>>(msg: T) -> Self {
        Error::NotFound(msg.into())
    }

    /// 创建其他错误
    pub fn other<T: Into<String>>(msg: T) -> Self {
        Error::Other(msg.into())
    }
}

/// 项目通用结果类型
pub type Result<T> = std::result::Result<T, Error>; 