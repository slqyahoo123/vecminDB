use crate::error::{Error, Result};
use crate::storage::constants::*;
use std::path::{Path, PathBuf};
use uuid::Uuid;
use chrono::{Utc, DateTime};
use std::io::{Read, Write};
use std::fs::{self, File, OpenOptions};
use std::time::{Duration, SystemTime};
use tracing::{debug, info, warn, error};

/// 将字符串转换为字节数组
#[inline]
pub fn str_to_bytes(s: &str) -> Vec<u8> {
    s.as_bytes().to_vec()
}

/// 将字节数组转换为字符串
#[inline]
pub fn bytes_to_str(bytes: &[u8]) -> Result<&str> {
    std::str::from_utf8(bytes).map_err(|e| Error::Internal(format!("无效的UTF-8序列: {}", e)))
}

/// 创建存储路径
pub fn make_storage_path<P: AsRef<Path>>(base_path: P, subdir: &str) -> Result<std::path::PathBuf> {
    let path = base_path.as_ref().join(subdir);
    ensure_dir_exists(&path)?;
    Ok(path)
}

/// 格式化模型键
#[inline]
pub fn format_model_key(model_id: &str, suffix: &str) -> String {
    format!("model:{}:{}", model_id, suffix)
}

/// 格式化数据键
#[inline]
pub fn format_data_key(data_id: &str, suffix: &str) -> String {
    format!("data:{}:{}", data_id, suffix)
}

/// 格式化用户键
#[inline]
pub fn format_user_key(user_id: &str, suffix: &str) -> String {
    format!("user:{}:{}", user_id, suffix)
}

/// 格式化会话键
#[inline]
pub fn format_session_key(session_id: &str, suffix: &str) -> String {
    format!("session:{}:{}", session_id, suffix)
}

/// 格式化权限键
#[inline]
pub fn format_permission_key(resource_type: &str, resource_id: &str, user_id: &str) -> String {
    format!("perm:{}:{}:{}", resource_type, resource_id, user_id)
}

/// 格式化算法键
#[inline]
pub fn format_algorithm_key(algo_id: &str, suffix: &str) -> String {
    format!("algo:{}:{}", algo_id, suffix)
}

/// 格式化版本键
#[inline]
pub fn format_version_key(version_type: &str, id: &str) -> String {
    format!("version:{}:{}", version_type, id)
}

/// 从键中提取ID
pub fn extract_id_from_key(key: &str, prefix: &str, suffix: &str) -> Option<String> {
    let prefix_pattern = format!("{}:", prefix);
    let suffix_pattern = format!(":{}", suffix);
    
    if key.starts_with(&prefix_pattern) && key.ends_with(&suffix_pattern) {
        let start = prefix_pattern.len();
        let end = key.len() - suffix_pattern.len();
        
        if start < end {
            Some(key[start..end].to_string())
        } else {
            None
        }
    } else {
        None
    }
}

/// 确保目录存在
pub fn ensure_dir_exists<P: AsRef<Path>>(path: P) -> Result<()> {
    let path = path.as_ref();
    if !path.exists() {
        fs::create_dir_all(path).map_err(|e| {
            error!("创建目录失败: {}, 错误: {}", path.display(), e);
            Error::Internal(format!("创建目录失败: {}", e))
        })?;
        debug!("创建目录: {}", path.display());
    }
    Ok(())
}

/// 生成唯一ID
pub fn generate_id() -> String {
    Uuid::new_v4().to_string()
}

/// 获取当前时间戳（毫秒）
pub fn current_timestamp_ms() -> i64 {
    Utc::now().timestamp_millis()
}

/// 将时间戳转换为DateTime
pub fn timestamp_to_datetime(timestamp_ms: i64) -> DateTime<Utc> {
    DateTime::from_timestamp_millis(timestamp_ms).unwrap_or_else(|| Utc::now())
}

/// 计算文件大小（以字节为单位）
pub fn calculate_file_size<P: AsRef<Path>>(path: P) -> Result<u64> {
    let metadata = fs::metadata(path.as_ref()).map_err(|e| {
        Error::Internal(format!("获取文件元信息失败: {}", e))
    })?;
    
    Ok(metadata.len())
}

/// 检查路径是否是一个文件
pub fn is_file<P: AsRef<Path>>(path: P) -> bool {
    path.as_ref().is_file()
}

/// 检查路径是否是一个目录
pub fn is_directory<P: AsRef<Path>>(path: P) -> bool {
    path.as_ref().is_dir()
}
