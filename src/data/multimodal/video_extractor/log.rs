//! 视频特征提取器日志模块
//!
//! 本模块定义了日志宏，用于调试和错误跟踪

use std::sync::atomic::{AtomicUsize, Ordering};
use std::fs::OpenOptions;
use std::io::Write;
use std::path::Path;
use std::sync::{Mutex, Once};
use std::time::{SystemTime, UNIX_EPOCH};
use std::collections::HashMap;
use std::fmt;
use chrono::{Local};
use serde_json;

/// 日志级别
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum LogLevel {
    Off = 0,
    Error = 1,
    Warn = 2,
    Info = 3,
    Debug = 4,
    Trace = 5,
}

impl fmt::Display for LogLevel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            LogLevel::Off => write!(f, "OFF"),
            LogLevel::Error => write!(f, "ERROR"),
            LogLevel::Warn => write!(f, "WARN"),
            LogLevel::Info => write!(f, "INFO"),
            LogLevel::Debug => write!(f, "DEBUG"),
            LogLevel::Trace => write!(f, "TRACE"),
        }
    }
}

impl From<&str> for LogLevel {
    fn from(s: &str) -> Self {
        match s.to_uppercase().as_str() {
            "OFF" => LogLevel::Off,
            "ERROR" => LogLevel::Error,
            "WARN" => LogLevel::Warn,
            "INFO" => LogLevel::Info,
            "DEBUG" => LogLevel::Debug,
            "TRACE" => LogLevel::Trace,
            _ => LogLevel::Info, // 默认级别
        }
    }
}

/// 日志输出目标
pub enum LogTarget {
    Console,
    File(String),
    Both(String),
    Custom(Box<dyn Fn(&str, LogLevel, &str) + Send + Sync>),
}

impl Clone for LogTarget {
    fn clone(&self) -> Self {
        match self {
            LogTarget::Console => LogTarget::Console,
            LogTarget::File(path) => LogTarget::File(path.clone()),
            LogTarget::Both(path) => LogTarget::Both(path.clone()),
            LogTarget::Custom(_) => LogTarget::Console, // Custom函数无法克隆，回退到Console
        }
    }
}

impl std::fmt::Debug for LogTarget {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            LogTarget::Console => write!(f, "LogTarget::Console"),
            LogTarget::File(path) => write!(f, "LogTarget::File({:?})", path),
            LogTarget::Both(path) => write!(f, "LogTarget::Both({:?})", path),
            LogTarget::Custom(_) => write!(f, "LogTarget::Custom(<function>)"),
        }
    }
}

/// 日志配置
#[derive(Debug, Clone)]
pub struct LogConfig {
    pub level: LogLevel,
    pub target: LogTarget,
    pub include_timestamp: bool,
    pub include_file_line: bool,
    pub structured: bool,
}

impl Default for LogConfig {
    fn default() -> Self {
        Self {
            level: LogLevel::Info,
            target: LogTarget::Console,
            include_timestamp: true,
            include_file_line: false,
            structured: false,
        }
    }
}

/// 全局日志级别
static LOG_LEVEL: AtomicUsize = AtomicUsize::new(LogLevel::Info as usize);

/// 全局日志配置
static mut LOG_CONFIG: Option<LogConfig> = None;
static LOG_INIT: Once = Once::new();

/// 日志文件句柄
static LOG_FILE: Mutex<Option<std::fs::File>> = Mutex::new(None);

/// 初始化日志系统
pub fn init(config: LogConfig) {
    LOG_LEVEL.store(config.level as usize, Ordering::SeqCst);
    
    // 准备文件输出
    match &config.target {
        LogTarget::File(path) | LogTarget::Both(path) => {
            let file = OpenOptions::new()
                .create(true)
                .append(true)
                .open(path);
                
            if let Ok(file) = file {
                let mut guard = LOG_FILE.lock().unwrap();
                *guard = Some(file);
            } else {
                eprintln!("无法打开日志文件: {}", path);
            }
        },
        _ => {}
    }
    
    // 设置全局配置
    LOG_INIT.call_once(|| {
        unsafe {
            LOG_CONFIG = Some(config);
        }
    });
}

/// 获取当前日志配置
pub fn get_config() -> LogConfig {
    unsafe {
        LOG_CONFIG.clone().unwrap_or_default()
    }
}

/// 设置日志级别
pub fn set_level(level: LogLevel) {
    LOG_LEVEL.store(level as usize, Ordering::SeqCst);
}

/// 获取当前日志级别
pub fn get_level() -> LogLevel {
    match LOG_LEVEL.load(Ordering::SeqCst) {
        0 => LogLevel::Off,
        1 => LogLevel::Error,
        2 => LogLevel::Warn,
        3 => LogLevel::Info,
        4 => LogLevel::Debug,
        5 => LogLevel::Trace,
        _ => LogLevel::Info,
    }
}

/// 检查是否启用指定级别日志
pub fn is_enabled(level: LogLevel) -> bool {
    level as usize <= LOG_LEVEL.load(Ordering::SeqCst)
}

/// 日志目标设置
pub fn set_target(target: LogTarget) {
    unsafe {
        if let Some(ref mut config) = LOG_CONFIG {
            config.target = target.clone();
        } else {
            let mut default_config = LogConfig::default();
            default_config.target = target.clone();
            LOG_CONFIG = Some(default_config);
        }
    }
    
    // 准备文件输出
    match &target {
        LogTarget::File(path) | LogTarget::Both(path) => {
            let file = OpenOptions::new()
                .create(true)
                .append(true)
                .open(path);
                
            if let Ok(file) = file {
                let mut guard = LOG_FILE.lock().unwrap();
                *guard = Some(file);
            } else {
                eprintln!("无法打开日志文件: {}", path);
            }
        },
        _ => {
            let mut guard = LOG_FILE.lock().unwrap();
            *guard = None;
        }
    }
}

/// 内部日志函数
pub fn log_internal(level: LogLevel, file: &str, line: u32, args: fmt::Arguments) {
    if !is_enabled(level) {
        return;
    }
    
    let config = unsafe { LOG_CONFIG.clone().unwrap_or_default() };
    
    let timestamp = if config.include_timestamp {
        let now = SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default();
        format!("{}.{:03} ", 
            Local::now().format("%Y-%m-%d %H:%M:%S"),
            now.subsec_millis())
    } else {
        String::new()
    };
    
    let location = if config.include_file_line {
        format!("{}:{} ", file, line)
    } else {
        String::new()
    };
    
    let message = format!("{}", args);
    
    let log_entry = if config.structured {
        // JSON格式日志
        let mut map = HashMap::new();
        map.insert("timestamp".to_string(), timestamp.trim().to_string());
        map.insert("level".to_string(), level.to_string());
        map.insert("file".to_string(), file.to_string());
        map.insert("line".to_string(), line.to_string());
        map.insert("message".to_string(), message);
        
        serde_json::to_string(&map).unwrap_or_else(|_| {
            format!("{timestamp}[{level}] {location}{message}")
        })
    } else {
        // 标准格式日志
        format!("{timestamp}[{level}] {location}{message}")
    };
    
    match &config.target {
        LogTarget::Console => {
            if level <= LogLevel::Error {
                eprintln!("{}", log_entry);
            } else {
                println!("{}", log_entry);
            }
        },
        LogTarget::File(_) => {
            let mut guard = LOG_FILE.lock().unwrap();
            if let Some(file) = guard.as_mut() {
                let _ = writeln!(file, "{}", log_entry);
                let _ = file.flush();
            }
        },
        LogTarget::Both(_) => {
            if level <= LogLevel::Error {
                eprintln!("{}", log_entry);
            } else {
                println!("{}", log_entry);
            }
            
            let mut guard = LOG_FILE.lock().unwrap();
            if let Some(file) = guard.as_mut() {
                let _ = writeln!(file, "{}", log_entry);
                let _ = file.flush();
            }
        },
        LogTarget::Custom(f) => {
            f(file, level, &message);
        }
    }
}

/// 创建结构化日志记录
pub fn structured_log(level: LogLevel, data: HashMap<String, String>) {
    if !is_enabled(level) {
        return;
    }
    
    let config = unsafe { LOG_CONFIG.clone().unwrap_or_default() };
    
    let mut log_data = data.clone();
    log_data.insert("level".to_string(), level.to_string());
    
    if config.include_timestamp {
        let now = SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default();
        log_data.insert("timestamp".to_string(), 
            format!("{}.{:03}", 
                Local::now().format("%Y-%m-%d %H:%M:%S"),
                now.subsec_millis()));
    }
    
    let log_entry = serde_json::to_string(&log_data).unwrap_or_else(|_| {
        format!("{:?}", log_data)
    });
    
    match &config.target {
        LogTarget::Console => {
            if level <= LogLevel::Error {
                eprintln!("{}", log_entry);
            } else {
                println!("{}", log_entry);
            }
        },
        LogTarget::File(_) => {
            let mut guard = LOG_FILE.lock().unwrap();
            if let Some(file) = guard.as_mut() {
                let _ = writeln!(file, "{}", log_entry);
                let _ = file.flush();
            }
        },
        LogTarget::Both(_) => {
            if level <= LogLevel::Error {
                eprintln!("{}", log_entry);
            } else {
                println!("{}", log_entry);
            }
            
            let mut guard = LOG_FILE.lock().unwrap();
            if let Some(file) = guard.as_mut() {
                let _ = writeln!(file, "{}", log_entry);
                let _ = file.flush();
            }
        },
        LogTarget::Custom(f) => {
            f("", level, &log_entry);
        }
    }
}

/// 旋转日志文件
pub fn rotate_log_file() -> std::io::Result<()> {
    let config = unsafe { LOG_CONFIG.clone().unwrap_or_default() };
    
    let path = match &config.target {
        LogTarget::File(path) | LogTarget::Both(path) => path,
        _ => return Ok(()),
    };
    
    let log_path = Path::new(path);
    if !log_path.exists() {
        return Ok(());
    }
    
    let file_name = log_path.file_name().unwrap().to_string_lossy();
    let parent = log_path.parent().unwrap_or_else(|| Path::new("."));
    
    let timestamp = Local::now().format("%Y%m%d_%H%M%S");
    let rotated_name = format!("{}_{}.log", file_name, timestamp);
    let rotated_path = parent.join(rotated_name);
    
    // 关闭当前日志文件
    {
        let mut guard = LOG_FILE.lock().unwrap();
        *guard = None;
    }
    
    // 重命名
    std::fs::rename(log_path, rotated_path)?;
    
    // 重新打开
    let file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)?;
        
    let mut guard = LOG_FILE.lock().unwrap();
    *guard = Some(file);
    
    Ok(())
}

// 定义宏
#[macro_export]
macro_rules! video_trace {
    ($($arg:tt)*) => {
        if crate::data::multimodal::video_extractor::log::is_enabled(crate::data::multimodal::video_extractor::log::LogLevel::Trace) {
            crate::data::multimodal::video_extractor::log::log_internal(
                crate::data::multimodal::video_extractor::log::LogLevel::Trace,
                file!(),
                line!(),
                format_args!($($arg)*)
            );
        }
    }
}

#[macro_export]
macro_rules! video_debug {
    ($($arg:tt)*) => {
        if crate::data::multimodal::video_extractor::log::is_enabled(crate::data::multimodal::video_extractor::log::LogLevel::Debug) {
            crate::data::multimodal::video_extractor::log::log_internal(
                crate::data::multimodal::video_extractor::log::LogLevel::Debug,
                file!(),
                line!(),
                format_args!($($arg)*)
            );
        }
    }
}

#[macro_export]
macro_rules! video_info {
    ($($arg:tt)*) => {
        if crate::data::multimodal::video_extractor::log::is_enabled(crate::data::multimodal::video_extractor::log::LogLevel::Info) {
            crate::data::multimodal::video_extractor::log::log_internal(
                crate::data::multimodal::video_extractor::log::LogLevel::Info,
                file!(),
                line!(),
                format_args!($($arg)*)
            );
        }
    }
}

#[macro_export]
macro_rules! video_warn {
    ($($arg:tt)*) => {
        if crate::data::multimodal::video_extractor::log::is_enabled(crate::data::multimodal::video_extractor::log::LogLevel::Warn) {
            crate::data::multimodal::video_extractor::log::log_internal(
                crate::data::multimodal::video_extractor::log::LogLevel::Warn,
                file!(),
                line!(),
                format_args!($($arg)*)
            );
        }
    }
}

#[macro_export]
macro_rules! video_error {
    ($($arg:tt)*) => {
        if crate::data::multimodal::video_extractor::log::is_enabled(crate::data::multimodal::video_extractor::log::LogLevel::Error) {
            crate::data::multimodal::video_extractor::log::log_internal(
                crate::data::multimodal::video_extractor::log::LogLevel::Error,
                file!(),
                line!(),
                format_args!($($arg)*)
            );
        }
    }
}

// 结构化日志宏
#[macro_export]
macro_rules! video_slog {
    ($level:expr, $($key:expr => $value:expr),*) => {
        if crate::data::multimodal::video_extractor::log::is_enabled($level) {
            let mut data = std::collections::HashMap::new();
            $(
                data.insert($key.to_string(), $value.to_string());
            )*
            crate::data::multimodal::video_extractor::log::structured_log($level, data);
        }
    }
} 