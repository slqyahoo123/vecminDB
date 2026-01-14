//! 视频特征提取器工具模块
//!
//! 本模块提供了各种辅助功能，包括数据处理、文件操作和格式转换等

use std::path::Path;
use std::fs;
use std::fs::File;
use std::io::{Read, Write};
use std::time::{SystemTime, UNIX_EPOCH};
// use std::collections::HashMap; // not used directly here
use std::hash::{Hash, Hasher};
use std::collections::hash_map::DefaultHasher;
use log as std_log;

use super::config::VideoFeatureConfig;
use super::types::{VideoFeatureType};
use super::error::VideoExtractionError;
use crate::Result;

/// 获取当前时间戳（毫秒）
pub fn current_timestamp() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64
}

/// 估计可用内存
pub fn estimate_available_memory() -> usize {
    #[cfg(target_os = "linux")]
    {
        // 在Linux上读取/proc/meminfo获取可用内存
        match std::fs::read_to_string("/proc/meminfo") {
            Ok(content) => {
                for line in content.lines() {
                    if line.starts_with("MemAvailable:") {
                        if let Some(value) = line.split_whitespace().nth(1) {
                            if let Ok(kb) = value.parse::<usize>() {
                                return kb / 1024; // 转换为MB
                            }
                        }
                    }
                }
            },
            Err(_) => {}
        }
    }
    
    #[cfg(target_os = "windows")]
    {
        // 在Windows上尝试使用WMI获取可用内存
        if let Ok(output) = std::process::Command::new("wmic")
            .args(&["OS", "get", "FreePhysicalMemory", "/Value"])
            .output() 
        {
            let output = String::from_utf8_lossy(&output.stdout);
            if let Some(value) = output.lines()
                .find(|l| l.starts_with("FreePhysicalMemory="))
                .and_then(|l| l.split('=').nth(1))
                .and_then(|v| v.trim().parse::<usize>().ok()) 
            {
                return value / 1024; // 转换为MB
            }
        }
    }
    
    #[cfg(target_os = "macos")]
    {
        // 在macOS上尝试使用vm_stat获取可用内存
        if let Ok(output) = std::process::Command::new("vm_stat").output() {
            let output = String::from_utf8_lossy(&output.stdout);
            
            // 解析页面大小
            let page_size = output.lines()
                .find(|l| l.contains("page size of"))
                .and_then(|l| l.split("page size of").nth(1))
                .and_then(|v| v.trim().split(' ').next())
                .and_then(|v| v.parse::<usize>().ok())
                .unwrap_or(4096); // 默认4KB
            
            // 解析空闲页面数
            let free_pages = output.lines()
                .find(|l| l.starts_with("Pages free:"))
                .and_then(|l| l.split(':').nth(1))
                .and_then(|v| v.trim().split(' ').next())
                .and_then(|v| v.replace('.', "").parse::<usize>().ok())
                .unwrap_or(0);
            
            return (free_pages * page_size) / (1024 * 1024); // 转换为MB
        }
    }
    
    // 默认返回8GB（保守估计）
    8192
}

/// 估算视频特征提取的内存需求
pub fn estimate_memory_requirement(
    config: &VideoFeatureConfig,
    video_count: usize,
    avg_resolution: (usize, usize),
    avg_duration: f64
) -> usize {
    // 基础内存需求(MB)
    let base_memory = 200;
    
    // 每种特征类型的内存需求系数
    let feature_memory_factors: Vec<f64> = config.feature_types.iter()
        .map(|ft| match ft {
            VideoFeatureType::RGB => 1.0,
            VideoFeatureType::OpticalFlow => 1.5,
            VideoFeatureType::I3D => 2.0,
            VideoFeatureType::SlowFast => 2.5,
            VideoFeatureType::Audio => 0.8,
            VideoFeatureType::Custom(_) => 1.5,
        })
        .collect();
    
    // 特征内存需求
    let feature_memory = feature_memory_factors.iter().sum::<f64>() * 100.0;
    
    // 视频内存需求
    let pixels = avg_resolution.0 * avg_resolution.1;
    let frames = avg_duration * config.fps as f64;
    let video_memory = (pixels as f64 * frames * 3.0 * video_count as f64) / (1024.0 * 1024.0);
    
    // 总内存估计(MB)
    let total_memory = base_memory + feature_memory as usize + video_memory as usize;
    
    // 添加缓存内存
    let cache_memory = if config.memory_optimized {
        total_memory / 10 // 优化模式下降低缓存
    } else {
        total_memory / 4 // 正常模式下充分缓存
    };
    
    total_memory + cache_memory
}

/// 检查视频文件是否存在且可读
pub fn check_video_file(path: &str) -> bool {
    let path = Path::new(path);
    
    if !path.exists() {
        return false;
    }
    
    if !path.is_file() {
        return false;
    }
    
    // 尝试打开文件检查可读性
    match File::open(path) {
        Ok(_) => true,
        Err(_) => false
    }
}

/// 获取视频文件扩展名
pub fn get_video_extension(path: &str) -> Option<String> {
    Path::new(path)
        .extension()
        .and_then(|ext| ext.to_str())
        .map(|ext| ext.to_lowercase())
}

/// 检查是否为支持的视频格式
pub fn is_supported_video_format(path: &str) -> bool {
    let supported_extensions = vec!["mp4", "avi", "mkv", "mov", "wmv", "flv", "webm"];
    
    if let Some(extension) = get_video_extension(path) {
        supported_extensions.contains(&extension.as_str())
    } else {
        false
    }
}

/// 计算两个配置的相似度(0.0-1.0)
pub fn calculate_config_similarity(config1: &VideoFeatureConfig, config2: &VideoFeatureConfig) -> f32 {
    let mut similarity_scores = Vec::new();
    
    // 比较帧宽度
    let width_similarity = 1.0 - ((config1.frame_width as f32 - config2.frame_width as f32).abs() / 
                               (config1.frame_width as f32 + config2.frame_width as f32));
    similarity_scores.push(width_similarity);
    
    // 比较帧高度
    let height_similarity = 1.0 - ((config1.frame_height as f32 - config2.frame_height as f32).abs() / 
                                (config1.frame_height as f32 + config2.frame_height as f32));
    similarity_scores.push(height_similarity);
    
    // 比较帧率
    let fps_similarity = 1.0 - ((config1.fps as f32 - config2.fps as f32).abs() / 
                             (config1.fps as f32 + config2.fps as f32));
    similarity_scores.push(fps_similarity);
    
    // 比较特征类型数量的重叠度
    let mut common_feature_types = 0;
    for ft1 in &config1.feature_types {
        if config2.feature_types.contains(ft1) {
            common_feature_types += 1;
        }
    }
    
    let feature_types_similarity = 2.0 * common_feature_types as f32 / 
                                 (config1.feature_types.len() + config2.feature_types.len()) as f32;
    similarity_scores.push(feature_types_similarity);
    
    // 比较其他布尔属性
    let use_cache_similarity = if config1.use_cache == config2.use_cache { 1.0 } else { 0.0 };
    similarity_scores.push(use_cache_similarity);
    
    let memory_optimized_similarity = if config1.memory_optimized == config2.memory_optimized { 1.0 } else { 0.0 };
    similarity_scores.push(memory_optimized_similarity);
    
    // 计算加权平均相似度
    // 给特征类型更高的权重
    let weights = [0.15, 0.15, 0.2, 0.3, 0.1, 0.1];
    let weighted_sum: f32 = similarity_scores.iter().zip(weights.iter())
        .map(|(score, &weight)| score * weight)
        .sum();
    
    weighted_sum
}

/// 创建测试视频文件夹
pub fn create_test_video_directory() -> std::result::Result<String, VideoExtractionError> {
    std_log::info!("创建测试视频目录");
    
    // 创建测试目录
    let test_dir = format!("./test_videos_{}", current_timestamp());
    fs::create_dir_all(&test_dir)
        .map_err(|e| VideoExtractionError::FileError(format!("无法创建测试目录: {}", e)))?;
    
    Ok(test_dir)
}

/// 下载测试视频
pub fn download_test_video(url: &str, target_path: &str) -> std::result::Result<(), VideoExtractionError> {
    std_log::info!("下载测试视频: {} -> {}", url, target_path);
    
    // 检查目标路径的目录是否存在
    if let Some(parent) = Path::new(target_path).parent() {
        if !parent.exists() {
            fs::create_dir_all(parent)
                .map_err(|e| VideoExtractionError::FileError(format!("无法创建目录: {}", e)))?;
        }
    }
    
    // 使用适当的库下载视频
    // 为简单起见，此处使用模拟代码
    
    // 检查操作系统并使用合适的下载工具
    #[cfg(target_os = "windows")]
    {
        let status = std::process::Command::new("powershell")
            .args(&["-Command", &format!("Invoke-WebRequest -Uri '{}' -OutFile '{}'", url, target_path)])
            .status()
            .map_err(|e| VideoExtractionError::FileError(format!("无法执行下载命令: {}", e)))?;
        
        if !status.success() {
            return Err(VideoExtractionError::FileError(format!("下载命令执行失败，状态码: {:?}", status.code())));
        }
    }
    
    #[cfg(not(target_os = "windows"))]
    {
        let status = std::process::Command::new("curl")
            .args(&["-L", "-o", target_path, url])
            .status()
            .map_err(|e| VideoExtractionError::FileError(format!("无法执行下载命令: {}", e)))?;
        
        if !status.success() {
            return Err(VideoExtractionError::FileError(format!("下载命令执行失败，状态码: {:?}", status.code())));
        }
    }
    
    // 验证下载的文件
    if !check_video_file(target_path) {
        return Err(VideoExtractionError::FileError(format!("下载的文件无法访问或不存在: {}", target_path)));
    }
    
    Ok(())
}

/// 生成测试视频
pub fn generate_test_video(target_path: &str, duration: u32, width: u32, height: u32, fps: u32) -> std::result::Result<(), VideoExtractionError> {
    std_log::info!("生成测试视频: {}s, {}x{} @{}fps -> {}", duration, width, height, fps, target_path);
    
    // 检查目标路径的目录是否存在
    if let Some(parent) = Path::new(target_path).parent() {
        if !parent.exists() {
            fs::create_dir_all(parent)
                .map_err(|e| VideoExtractionError::FileError(format!("无法创建目录: {}", e)))?;
        }
    }
    
    // 使用ffmpeg生成测试视频
    let status = std::process::Command::new("ffmpeg")
        .args(&[
            "-y",
            "-f", "lavfi",
            "-i", &format!("testsrc=duration={}:size={}x{}:rate={}", duration, width, height, fps),
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            target_path
        ])
        .status()
        .map_err(|e| VideoExtractionError::FileError(format!("无法执行ffmpeg命令: {}", e)))?;
    
    if !status.success() {
        return Err(VideoExtractionError::FileError(format!("ffmpeg命令执行失败，状态码: {:?}", status.code())));
    }
    
    // 验证生成的文件
    if !check_video_file(target_path) {
        return Err(VideoExtractionError::FileError(format!("生成的文件无法访问或不存在: {}", target_path)));
    }
    
    Ok(())
}

/// 加载样本视频列表
pub fn load_sample_videos() -> std::result::Result<Vec<String>, VideoExtractionError> {
    std_log::info!("加载样本视频列表");
    
    // 检查常见位置的视频文件
    let sample_dirs = vec![
        "./test_videos",
        "./samples",
        "./data/samples",
        "./videos",
    ];
    
    let mut video_paths = Vec::new();
    let extensions = vec!["mp4", "avi", "mkv", "mov", "webm", "flv"];
    
    // 遍历所有可能的目录查找视频文件
    for dir in &sample_dirs {
        if !Path::new(dir).exists() {
            continue;
        }
        
        match fs::read_dir(dir) {
            Ok(entries) => {
                for entry in entries.filter_map(Result::ok) {
                    let path = entry.path();
                    if path.is_file() {
                        if let Some(extension) = path.extension().and_then(|e| e.to_str()) {
                            if extensions.contains(&extension.to_lowercase().as_str()) {
                                video_paths.push(path.to_string_lossy().to_string());
                            }
                        }
                    }
                }
            },
            Err(_) => continue,
        }
    }
    
    if video_paths.is_empty() {
        // 如果没有找到视频文件，创建一个测试视频
        let test_dir = create_test_video_directory()?;
        let test_video_path = format!("{}/test_video.mp4", test_dir);
        
        generate_test_video(&test_video_path, 10, 640, 480, 30)?;
        video_paths.push(test_video_path);
    }
    
    std_log::info!("找到 {} 个样本视频", video_paths.len());
    Ok(video_paths)
}

/// 计算哈希值
pub fn calculate_hash<T: AsRef<[u8]>>(data: T) -> String {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};
    
    let mut hasher = DefaultHasher::new();
    data.as_ref().hash(&mut hasher);
    format!("{:x}", hasher.finish())
}

/// 对字符串计算哈希值
pub fn hash_str(s: &str) -> u64 {
    let mut hasher = DefaultHasher::new();
    s.hash(&mut hasher);
    hasher.finish()
}

/// 保存特征数据到文件
pub fn save_features_to_file(features: &[f32], file_path: &str) -> std::result::Result<(), VideoExtractionError> {
    // 确保目录存在
    if let Some(parent) = Path::new(file_path).parent() {
        if !parent.exists() {
            fs::create_dir_all(parent)
                .map_err(|e| VideoExtractionError::FileError(format!("无法创建目录: {}", e)))?;
        }
    }
    
    // 将特征转换为字节
    let mut bytes = Vec::with_capacity(features.len() * 4);
    for &value in features {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    
    // 写入文件
    let mut file = File::create(file_path)
        .map_err(|e| VideoExtractionError::FileError(format!("无法创建文件: {}", e)))?;
    
    file.write_all(&bytes)
        .map_err(|e| VideoExtractionError::FileError(format!("无法写入文件: {}", e)))?;
    
    Ok(())
}

/// 从文件加载特征数据
pub fn load_features_from_file(file_path: &str) -> std::result::Result<Vec<f32>, VideoExtractionError> {
    // 检查文件是否存在
    if !Path::new(file_path).exists() {
        return Err(VideoExtractionError::FileError(format!("特征文件不存在: {}", file_path)));
    }
    
    // 读取文件内容
    let mut file = File::open(file_path)
        .map_err(|e| VideoExtractionError::FileError(format!("无法打开文件: {}", e)))?;
    
    let mut bytes = Vec::new();
    file.read_to_end(&mut bytes)
        .map_err(|e| VideoExtractionError::FileError(format!("无法读取文件: {}", e)))?;
    
    // 将字节转换为特征
    if bytes.len() % 4 != 0 {
        return Err(VideoExtractionError::FileError(format!("特征文件格式错误: {}", file_path)));
    }
    
    let mut features = Vec::with_capacity(bytes.len() / 4);
    for chunk in bytes.chunks_exact(4) {
        let value = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
        features.push(value);
    }
    
    Ok(features)
}

/// 格式化时间为可读字符串(毫秒 -> HH:MM:SS.mmm)
pub fn format_time(milliseconds: u64) -> String {
    let seconds = milliseconds / 1000;
    let minutes = seconds / 60;
    let hours = minutes / 60;
    
    let ms = milliseconds % 1000;
    let seconds = seconds % 60;
    let minutes = minutes % 60;
    
    format!("{:02}:{:02}:{:02}.{:03}", hours, minutes, seconds, ms)
}

/// 解析时间字符串为毫秒
pub fn parse_time(time_str: &str) -> std::result::Result<u64, VideoExtractionError> {
    // 支持多种格式:
    // - HH:MM:SS.mmm
    // - MM:SS.mmm
    // - SS.mmm
    // - SS
    
    let parts: Vec<&str> = time_str.split(':').collect();
    
    match parts.len() {
        3 => { // HH:MM:SS.mmm
            let hours: u64 = parts[0].parse()
                .map_err(|_| VideoExtractionError::ConfigError(format!("无效的小时格式: {}", parts[0])))?;
            
            let minutes: u64 = parts[1].parse()
                .map_err(|_| VideoExtractionError::ConfigError(format!("无效的分钟格式: {}", parts[1])))?;
            
            let seconds_parts: Vec<&str> = parts[2].split('.').collect();
            let seconds: u64 = seconds_parts[0].parse()
                .map_err(|_| VideoExtractionError::ConfigError(format!("无效的秒格式: {}", seconds_parts[0])))?;
            
            let ms: u64 = if seconds_parts.len() > 1 {
                let ms_str = format!("{:0<3}", seconds_parts[1]).chars().take(3).collect::<String>();
                ms_str.parse()
                    .map_err(|_| VideoExtractionError::ConfigError(format!("无效的毫秒格式: {}", seconds_parts[1])))?
            } else {
                0
            };
            
            Ok(hours * 3600000 + minutes * 60000 + seconds * 1000 + ms)
        },
        2 => { // MM:SS.mmm
            let minutes: u64 = parts[0].parse()
                .map_err(|_| VideoExtractionError::ConfigError(format!("无效的分钟格式: {}", parts[0])))?;
            
            let seconds_parts: Vec<&str> = parts[1].split('.').collect();
            let seconds: u64 = seconds_parts[0].parse()
                .map_err(|_| VideoExtractionError::ConfigError(format!("无效的秒格式: {}", seconds_parts[0])))?;
            
            let ms: u64 = if seconds_parts.len() > 1 {
                let ms_str = format!("{:0<3}", seconds_parts[1]).chars().take(3).collect::<String>();
                ms_str.parse()
                    .map_err(|_| VideoExtractionError::ConfigError(format!("无效的毫秒格式: {}", seconds_parts[1])))?
            } else {
                0
            };
            
            Ok(minutes * 60000 + seconds * 1000 + ms)
        },
        1 => { // SS.mmm 或 SS
            let seconds_parts: Vec<&str> = parts[0].split('.').collect();
            let seconds: u64 = seconds_parts[0].parse()
                .map_err(|_| VideoExtractionError::ConfigError(format!("无效的秒格式: {}", seconds_parts[0])))?;
            
            let ms: u64 = if seconds_parts.len() > 1 {
                let ms_str = format!("{:0<3}", seconds_parts[1]).chars().take(3).collect::<String>();
                ms_str.parse()
                    .map_err(|_| VideoExtractionError::ConfigError(format!("无效的毫秒格式: {}", seconds_parts[1])))?
            } else {
                0
            };
            
            Ok(seconds * 1000 + ms)
        },
        _ => Err(VideoExtractionError::ConfigError(format!("无效的时间格式: {}", time_str))),
    }
} 