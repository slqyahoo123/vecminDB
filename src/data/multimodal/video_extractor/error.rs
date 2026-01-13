//! 视频特征提取器错误处理
//!
//! 本模块定义了视频特征提取过程中可能出现的各种错误类型

use std::fmt;
use crate::Error;
use std::collections::HashMap;
use std::path::Path;
use regex;
use crate::data::multimodal::video_extractor::util::estimate_available_memory;

/// 视频提取错误类型
#[derive(Debug, Clone)]
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
    /// 编解码错误
    CodecError(String),
    /// 处理错误
    ProcessingError(String),
    /// 输入错误
    InputError(String),
    /// 内存错误
    MemoryError(String),
    /// 缓存错误
    CacheError(String),
    /// 导出错误
    ExportError(String),
    /// 功能未实现
    NotImplementedError(String),
    /// 功能未启用
    FeatureNotEnabled(String),
    /// 系统错误
    SystemError(String),
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
            Self::CodecError(msg) => write!(f, "编解码错误: {}", msg),
            Self::ProcessingError(msg) => write!(f, "处理错误: {}", msg),
            Self::InputError(msg) => write!(f, "输入错误: {}", msg),
            Self::MemoryError(msg) => write!(f, "内存错误: {}", msg),
            Self::CacheError(msg) => write!(f, "缓存错误: {}", msg),
            Self::ExportError(msg) => write!(f, "导出错误: {}", msg),
            Self::NotImplementedError(msg) => write!(f, "功能未实现: {}", msg),
            Self::FeatureNotEnabled(msg) => write!(f, "功能未启用: {}", msg),
            Self::SystemError(msg) => write!(f, "系统错误: {}", msg),
        }
    }
}

impl std::error::Error for VideoExtractionError {}

/// 从IO错误转换
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

/// 从字符串引用错误转换
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

/// 视频提取错误诊断
#[derive(Debug, Clone)]
pub struct ErrorDiagnostics {
    /// 错误类型
    pub error_type: String,
    /// 错误消息
    pub message: String,
    /// 错误位置
    pub location: String,
    /// 可能的原因
    pub possible_causes: Vec<String>,
    /// 推荐解决方案
    pub recommendations: Vec<String>,
    /// 系统信息
    pub system_info: HashMap<String, String>,
    /// 测试结果
    pub test_results: HashMap<String, bool>,
}

impl ErrorDiagnostics {
    /// 创建新的错误诊断
    pub fn new(error: &VideoExtractionError) -> Self {
        let mut diagnostics = Self {
            error_type: format!("{:?}", error),
            message: error.to_string(),
            location: std::backtrace::Backtrace::capture().to_string(),
            possible_causes: Vec::new(),
            recommendations: Vec::new(),
            system_info: collect_system_info(),
            test_results: HashMap::new(),
        };
        
        diagnostics.analyze_error(error);
        diagnostics
    }
    
    /// 添加可能的原因
    pub fn add_cause(&mut self, cause: &str) {
        self.possible_causes.push(cause.to_string());
    }
    
    /// 添加推荐解决方案
    pub fn add_recommendation(&mut self, recommendation: &str) {
        self.recommendations.push(recommendation.to_string());
    }
    
    /// 添加测试结果
    pub fn add_test_result(&mut self, test_name: &str, result: bool) {
        self.test_results.insert(test_name.to_string(), result);
    }
    
    /// 分析错误并添加诊断信息
    fn analyze_error(&mut self, error: &VideoExtractionError) {
        match error {
            VideoExtractionError::FileError(msg) => {
                self.add_cause("文件不存在或无法访问");
                self.add_cause("文件格式不支持");
                self.add_cause("文件权限不足");
                self.add_cause("文件路径错误");
                
                self.add_recommendation("检查文件路径是否正确");
                self.add_recommendation("验证文件是否存在且可读");
                self.add_recommendation("确认文件格式是受支持的视频格式");
                self.add_recommendation("检查文件权限");
                
                // 运行文件存在测试
                if let Some(path) = extract_path_from_error(msg) {
                    let file_exists = Path::new(&path).exists();
                    self.add_test_result("文件存在", file_exists);
                    
                    if file_exists {
                        let file_readable = Path::new(&path).metadata().is_ok();
                        self.add_test_result("文件可读", file_readable);
                        
                        if let Ok(metadata) = Path::new(&path).metadata() {
                            let file_size = metadata.len();
                            self.add_test_result("文件非空", file_size > 0);
                        }
                    }
                }
            },
            VideoExtractionError::DecodeError(msg) => {
                self.add_cause("视频编解码器不支持");
                self.add_cause("视频文件已损坏");
                self.add_cause("视频格式错误");
                self.add_cause("系统缺少必要的编解码库");
                
                self.add_recommendation("尝试使用ffmpeg转换视频格式");
                self.add_recommendation("检查系统是否安装了必要的视频编解码库");
                self.add_recommendation("使用其他视频格式（如MP4、MKV）");
                self.add_recommendation("检查视频文件完整性");
                
                // 尝试运行ffmpeg检查
                if let Some(path) = extract_path_from_error(msg) {
                    let ffmpeg_check = std::process::Command::new("ffmpeg")
                        .args(&["-v", "error", "-i", &path, "-f", "null", "-"])
                        .output();
                    
                    match ffmpeg_check {
                        Ok(output) => {
                            let error_output = String::from_utf8_lossy(&output.stderr);
                            self.add_test_result("FFmpeg检查通过", error_output.is_empty());
                            if !error_output.is_empty() {
                                self.system_info.insert("FFmpeg错误".to_string(), error_output.to_string());
                            }
                        },
                        Err(_) => {
                            self.add_test_result("FFmpeg可用", false);
                        }
                    }
                }
            },
            VideoExtractionError::ExtractionError(msg) => {
                self.add_cause("特征提取算法错误");
                self.add_cause("视频帧处理失败");
                self.add_cause("内存不足");
                self.add_cause("特征类型与视频内容不匹配");
                
                self.add_recommendation("尝试使用不同的特征类型");
                self.add_recommendation("增加系统可用内存");
                self.add_recommendation("减小视频分辨率或帧率");
                self.add_recommendation("分批处理视频");
                
                // 检查可用内存
                self.system_info.insert("可用内存(MB)".to_string(), estimate_available_memory().to_string());
            },
            VideoExtractionError::ProcessingError(msg) => {
                self.add_cause("视频处理中断");
                self.add_cause("系统资源不足");
                self.add_cause("处理参数不适合当前视频");
                
                self.add_recommendation("使用更低的处理参数（分辨率、帧率）");
                self.add_recommendation("分段处理视频");
                self.add_recommendation("关闭其他占用资源的应用");
                self.add_recommendation("检查系统资源使用情况");
                
                // 检查CPU使用率
                if let Some(cpu_usage) = get_cpu_usage() {
                    self.system_info.insert("CPU使用率(%)".to_string(), cpu_usage.to_string());
                    self.add_test_result("CPU资源充足", cpu_usage < 90.0);
                }
            },
            VideoExtractionError::ConfigError(msg) => {
                self.add_cause("配置参数无效");
                self.add_cause("配置与当前系统不兼容");
                self.add_cause("特征类型与配置不匹配");
                
                self.add_recommendation("使用默认配置");
                self.add_recommendation("检查配置参数范围");
                self.add_recommendation("确保配置与特征类型匹配");
            },
            VideoExtractionError::SystemError(msg) => {
                self.add_cause("系统资源不足");
                self.add_cause("权限问题");
                self.add_cause("系统调用失败");
                
                self.add_recommendation("检查系统资源使用情况");
                self.add_recommendation("以管理员权限运行程序");
                self.add_recommendation("更新系统依赖库");
            },
            VideoExtractionError::FileError(msg) if msg.contains("元数据") || msg.contains("metadata") => {
                self.add_cause("视频元数据读取失败");
                self.add_cause("视频格式不支持");
                self.add_cause("元数据已损坏");
                
                self.add_recommendation("使用标准视频格式（MP4、MKV）");
                self.add_recommendation("使用ffmpeg修复视频元数据");
                self.add_recommendation("检查视频文件完整性");
            },
            VideoExtractionError::InputError(msg) => {
                self.add_cause("输入参数无效");
                self.add_cause("必要参数缺失");
                
                self.add_recommendation("检查输入参数");
                self.add_recommendation("参考文档中的参数说明");
            },
            VideoExtractionError::CacheError(msg) => {
                self.add_cause("缓存访问失败");
                self.add_cause("缓存已损坏");
                self.add_cause("缓存空间不足");
                
                self.add_recommendation("清除缓存并重试");
                self.add_recommendation("增加缓存大小配置");
                self.add_recommendation("检查磁盘空间");
            },
            VideoExtractionError::ExportError(msg) => {
                self.add_cause("导出功能未实现");
                self.add_cause("导出参数无效");
                
                self.add_recommendation("检查导出参数");
                self.add_recommendation("参考文档中的导出说明");
            },
            VideoExtractionError::NotImplementedError(msg) => {
                self.add_cause("功能未实现");
                self.add_cause("功能未启用");
                
                self.add_recommendation("检查功能实现情况");
                self.add_recommendation("参考文档中的功能说明");
            },
            VideoExtractionError::FeatureNotEnabled(msg) => {
                self.add_cause("功能未启用");
                self.add_cause("必要功能缺失");
                
                self.add_recommendation("检查功能启用情况");
                self.add_recommendation("参考文档中的功能说明");
            },
        }
    }
    
    /// 获取诊断摘要
    pub fn get_summary(&self) -> String {
        let mut summary = String::new();
        
        summary.push_str(&format!("错误类型: {}\n", self.error_type));
        summary.push_str(&format!("错误消息: {}\n\n", self.message));
        
        summary.push_str("可能的原因:\n");
        for (i, cause) in self.possible_causes.iter().enumerate() {
            summary.push_str(&format!("{}. {}\n", i + 1, cause));
        }
        summary.push_str("\n");
        
        summary.push_str("建议解决方案:\n");
        for (i, recommendation) in self.recommendations.iter().enumerate() {
            summary.push_str(&format!("{}. {}\n", i + 1, recommendation));
        }
        summary.push_str("\n");
        
        if !self.test_results.is_empty() {
            summary.push_str("测试结果:\n");
            for (test, result) in &self.test_results {
                let result_str = if *result { "通过" } else { "失败" };
                summary.push_str(&format!("- {}: {}\n", test, result_str));
            }
            summary.push_str("\n");
        }
        
        summary.push_str("系统信息:\n");
        for (key, value) in &self.system_info {
            summary.push_str(&format!("- {}: {}\n", key, value));
        }
        
        summary
    }
}

impl VideoExtractionError {
    /// 诊断错误
    pub fn diagnose(&self) -> ErrorDiagnostics {
        ErrorDiagnostics::new(self)
    }
}

/// 从错误消息中提取路径
fn extract_path_from_error(msg: &str) -> Option<String> {
    // 尝试匹配常见的路径格式
    let path_patterns = [
        r#"["']([^"']+\.(mp4|avi|mkv|mov|wmv|flv|webm))["']"#,
        r"/([^/]+\.(mp4|avi|mkv|mov|wmv|flv|webm))",
        r"\\([^\\]+\.(mp4|avi|mkv|mov|wmv|flv|webm))",
        r"路径[：:]\s*(.+)",
    ];
    
    for pattern in &path_patterns {
        if let Some(caps) = regex::Regex::new(pattern).ok()?.captures(msg) {
            if let Some(path) = caps.get(1) {
                return Some(path.as_str().to_string());
            }
        }
    }
    
    None
}

/// 收集系统信息
fn collect_system_info() -> HashMap<String, String> {
    let mut info = HashMap::new();
    
    // 操作系统信息
    info.insert("操作系统".to_string(), std::env::consts::OS.to_string());
    
    // 内存信息
    info.insert("可用内存(MB)".to_string(), estimate_available_memory().to_string());
    
    // 可执行文件路径
    if let Ok(exe_path) = std::env::current_exe() {
        info.insert("程序路径".to_string(), exe_path.to_string_lossy().to_string());
    }
    
    // FFmpeg版本
    if let Ok(output) = std::process::Command::new("ffmpeg").arg("-version").output() {
        let version = String::from_utf8_lossy(&output.stdout);
        if let Some(line) = version.lines().next() {
            info.insert("FFmpeg版本".to_string(), line.to_string());
        }
    } else {
        info.insert("FFmpeg可用".to_string(), "否".to_string());
    }
    
    // 当前工作目录
    if let Ok(cwd) = std::env::current_dir() {
        info.insert("当前目录".to_string(), cwd.to_string_lossy().to_string());
    }
    
    info
}

/// 获取当前CPU使用率
fn get_cpu_usage() -> Option<f64> {
    #[cfg(target_os = "linux")]
    {
        // 在Linux上读取/proc/stat获取CPU使用率
        // 需要两次读取来计算差值
        if let (Ok(stat1), Ok(_)) = (
            std::fs::read_to_string("/proc/stat"),
            std::thread::sleep(std::time::Duration::from_millis(100)),
        ) {
            if let Ok(stat2) = std::fs::read_to_string("/proc/stat") {
                if let (Some(cpu1), Some(cpu2)) = (
                    stat1.lines().next(),
                    stat2.lines().next(),
                ) {
                    let values1: Vec<u64> = cpu1.split_whitespace()
                        .skip(1)
                        .take(7)
                        .filter_map(|s| s.parse().ok())
                        .collect();
                    
                    let values2: Vec<u64> = cpu2.split_whitespace()
                        .skip(1)
                        .take(7)
                        .filter_map(|s| s.parse().ok())
                        .collect();
                    
                    if values1.len() == 7 && values2.len() == 7 {
                        let idle1 = values1[3];
                        let total1: u64 = values1.iter().sum();
                        
                        let idle2 = values2[3];
                        let total2: u64 = values2.iter().sum();
                        
                        let idle_diff = idle2 - idle1;
                        let total_diff = total2 - total1;
                        
                        if total_diff > 0 {
                            return Some(100.0 - (idle_diff as f64 * 100.0 / total_diff as f64));
                        }
                    }
                }
            }
        }
    }
    
    #[cfg(target_os = "windows")]
    {
        // 在Windows上使用wmic获取CPU使用率
        if let Ok(output) = std::process::Command::new("wmic")
            .args(&["cpu", "get", "LoadPercentage", "/Value"])
            .output() 
        {
            let output = String::from_utf8_lossy(&output.stdout);
            if let Some(value) = output.lines()
                .find(|l| l.starts_with("LoadPercentage="))
                .and_then(|l| l.split('=').nth(1))
                .and_then(|v| v.trim().parse::<f64>().ok()) 
            {
                return Some(value);
            }
        }
    }
    
    #[cfg(target_os = "macos")]
    {
        // 在macOS上使用top命令获取CPU使用率
        if let Ok(output) = std::process::Command::new("top")
            .args(&["-l", "1", "-n", "0"])
            .output() 
        {
            let output = String::from_utf8_lossy(&output.stdout);
            for line in output.lines() {
                if line.contains("CPU usage:") {
                    if let Some(idle_str) = line.split("% idle").next() {
                        if let Some(idle_pos) = idle_str.rfind(' ') {
                            let idle_substr = &idle_str[idle_pos + 1..];
                            if let Ok(idle) = idle_substr.parse::<f64>() {
                                return Some(100.0 - idle);
                            }
                        }
                    }
                    break;
                }
            }
        }
    }
    
    None
} 