//! 文件操作工具模块
//! 
//! 提供文件读写、路径处理、文件监控等功能。

use std::fs::{self, File, OpenOptions};
use std::io::{self, Write, BufRead, BufReader, BufWriter};
use std::path::{Path, PathBuf};


/// 文件操作错误类型
#[derive(Debug, thiserror::Error)]
pub enum FileError {
    #[error("文件不存在: {0}")]
    FileNotFound(String),
    #[error("权限不足: {0}")]
    PermissionDenied(String),
    #[error("IO错误: {0}")]
    IoError(#[from] io::Error),
    #[error("路径无效: {0}")]
    InvalidPath(String),
}

/// 文件操作结果
pub type FileResult<T> = std::result::Result<T, FileError>;

/// 文件工具
pub struct FileUtils;

impl FileUtils {
    /// 读取文件内容为字符串
    pub fn read_to_string<P: AsRef<Path>>(path: P) -> FileResult<String> {
        let path = path.as_ref();
        
        if !path.exists() {
            return Err(FileError::FileNotFound(path.display().to_string()));
        }

        fs::read_to_string(path).map_err(FileError::IoError)
    }

    /// 读取文件内容为字节数组
    pub fn read_to_bytes<P: AsRef<Path>>(path: P) -> FileResult<Vec<u8>> {
        let path = path.as_ref();
        
        if !path.exists() {
            return Err(FileError::FileNotFound(path.display().to_string()));
        }

        fs::read(path).map_err(FileError::IoError)
    }

    /// 写入字符串到文件
    pub fn write_string<P: AsRef<Path>>(path: P, content: &str) -> FileResult<()> {
        let path = path.as_ref();
        
        // 确保父目录存在
        if let Some(parent) = path.parent() {
            Self::ensure_dir_exists(parent)?;
        }

        fs::write(path, content).map_err(FileError::IoError)
    }

    /// 写入字节数组到文件
    pub fn write_bytes<P: AsRef<Path>>(path: P, content: &[u8]) -> FileResult<()> {
        let path = path.as_ref();
        
        // 确保父目录存在
        if let Some(parent) = path.parent() {
            Self::ensure_dir_exists(parent)?;
        }

        fs::write(path, content).map_err(FileError::IoError)
    }

    /// 追加字符串到文件
    pub fn append_string<P: AsRef<Path>>(path: P, content: &str) -> FileResult<()> {
        let path = path.as_ref();
        
        let mut file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(path)
            .map_err(FileError::IoError)?;

        file.write_all(content.as_bytes()).map_err(FileError::IoError)
    }

    /// 逐行读取文件
    pub fn read_lines<P: AsRef<Path>>(path: P) -> FileResult<Vec<String>> {
        let path = path.as_ref();
        
        if !path.exists() {
            return Err(FileError::FileNotFound(path.display().to_string()));
        }

        let file = File::open(path).map_err(FileError::IoError)?;
        let reader = BufReader::new(file);
        
        let mut lines = Vec::new();
        for line in reader.lines() {
            lines.push(line.map_err(FileError::IoError)?);
        }
        
        Ok(lines)
    }

    /// 写入多行到文件
    pub fn write_lines<P: AsRef<Path>>(path: P, lines: &[String]) -> FileResult<()> {
        let path = path.as_ref();
        
        // 确保父目录存在
        if let Some(parent) = path.parent() {
            Self::ensure_dir_exists(parent)?;
        }

        let file = File::create(path).map_err(FileError::IoError)?;
        let mut writer = BufWriter::new(file);

        for line in lines {
            writeln!(writer, "{}", line).map_err(FileError::IoError)?;
        }

        writer.flush().map_err(FileError::IoError)
    }

    /// 复制文件
    pub fn copy_file<P: AsRef<Path>>(from: P, to: P) -> FileResult<u64> {
        let from = from.as_ref();
        let to = to.as_ref();
        
        if !from.exists() {
            return Err(FileError::FileNotFound(from.display().to_string()));
        }

        // 确保目标目录存在
        if let Some(parent) = to.parent() {
            Self::ensure_dir_exists(parent)?;
        }

        fs::copy(from, to).map_err(FileError::IoError)
    }

    /// 移动文件
    pub fn move_file<P: AsRef<Path>>(from: P, to: P) -> FileResult<()> {
        let from = from.as_ref();
        let to = to.as_ref();
        
        if !from.exists() {
            return Err(FileError::FileNotFound(from.display().to_string()));
        }

        // 确保目标目录存在
        if let Some(parent) = to.parent() {
            Self::ensure_dir_exists(parent)?;
        }

        fs::rename(from, to).map_err(FileError::IoError)
    }

    /// 删除文件
    pub fn delete_file<P: AsRef<Path>>(path: P) -> FileResult<()> {
        let path = path.as_ref();
        
        if !path.exists() {
            return Ok(()); // 文件不存在，认为删除成功
        }

        fs::remove_file(path).map_err(FileError::IoError)
    }

    /// 检查文件是否存在
    pub fn exists<P: AsRef<Path>>(path: P) -> bool {
        path.as_ref().exists()
    }

    /// 获取文件大小
    pub fn file_size<P: AsRef<Path>>(path: P) -> FileResult<u64> {
        let path = path.as_ref();
        
        if !path.exists() {
            return Err(FileError::FileNotFound(path.display().to_string()));
        }

        let metadata = fs::metadata(path).map_err(FileError::IoError)?;
        Ok(metadata.len())
    }

    /// 获取文件修改时间
    pub fn modified_time<P: AsRef<Path>>(path: P) -> FileResult<std::time::SystemTime> {
        let path = path.as_ref();
        
        if !path.exists() {
            return Err(FileError::FileNotFound(path.display().to_string()));
        }

        let metadata = fs::metadata(path).map_err(FileError::IoError)?;
        metadata.modified().map_err(FileError::IoError)
    }

    /// 创建目录（递归）
    pub fn ensure_dir_exists<P: AsRef<Path>>(path: P) -> FileResult<()> {
        let path = path.as_ref();
        
        if !path.exists() {
            fs::create_dir_all(path).map_err(FileError::IoError)?;
        }
        
        Ok(())
    }

    /// 删除目录（递归）
    pub fn delete_dir<P: AsRef<Path>>(path: P) -> FileResult<()> {
        let path = path.as_ref();
        
        if path.exists() {
            fs::remove_dir_all(path).map_err(FileError::IoError)?;
        }
        
        Ok(())
    }

    /// 列出目录内容
    pub fn list_dir<P: AsRef<Path>>(path: P) -> FileResult<Vec<PathBuf>> {
        let path = path.as_ref();
        
        if !path.exists() {
            return Err(FileError::FileNotFound(path.display().to_string()));
        }

        if !path.is_dir() {
            return Err(FileError::InvalidPath("不是目录".to_string()));
        }

        let mut entries = Vec::new();
        for entry in fs::read_dir(path).map_err(FileError::IoError)? {
            let entry = entry.map_err(FileError::IoError)?;
            entries.push(entry.path());
        }

        Ok(entries)
    }

    /// 查找文件（递归）
    pub fn find_files<P: AsRef<Path>>(
        dir: P,
        pattern: &str,
        recursive: bool,
    ) -> FileResult<Vec<PathBuf>> {
        let dir = dir.as_ref();
        
        if !dir.exists() {
            return Err(FileError::FileNotFound(dir.display().to_string()));
        }

        let mut found_files = Vec::new();
        Self::find_files_recursive(dir, pattern, recursive, &mut found_files)?;
        
        Ok(found_files)
    }

    fn find_files_recursive(
        dir: &Path,
        pattern: &str,
        recursive: bool,
        found_files: &mut Vec<PathBuf>,
    ) -> FileResult<()> {
        for entry in fs::read_dir(dir).map_err(FileError::IoError)? {
            let entry = entry.map_err(FileError::IoError)?;
            let path = entry.path();

            if path.is_file() {
                if let Some(file_name) = path.file_name().and_then(|n| n.to_str()) {
                    if file_name.contains(pattern) {
                        found_files.push(path);
                    }
                }
            } else if path.is_dir() && recursive {
                Self::find_files_recursive(&path, pattern, recursive, found_files)?;
            }
        }

        Ok(())
    }

    /// 清空目录内容
    pub fn clear_dir<P: AsRef<Path>>(path: P) -> FileResult<()> {
        let path = path.as_ref();
        
        if !path.exists() {
            return Ok(());
        }

        for entry in fs::read_dir(path).map_err(FileError::IoError)? {
            let entry = entry.map_err(FileError::IoError)?;
            let entry_path = entry.path();

            if entry_path.is_file() {
                fs::remove_file(entry_path).map_err(FileError::IoError)?;
            } else if entry_path.is_dir() {
                fs::remove_dir_all(entry_path).map_err(FileError::IoError)?;
            }
        }

        Ok(())
    }
}

/// 路径工具
pub struct PathUtils;

impl PathUtils {
    /// 规范化路径
    pub fn normalize<P: AsRef<Path>>(path: P) -> PathBuf {
        let path = path.as_ref();
        
        // 简单的路径规范化
        let mut components = Vec::new();
        
        for component in path.components() {
            match component {
                std::path::Component::ParentDir => {
                    if !components.is_empty() {
                        components.pop();
                    }
                }
                std::path::Component::CurDir => {
                    // 忽略当前目录
                }
                _ => components.push(component),
            }
        }

        components.iter().collect()
    }

    /// 获取相对路径
    pub fn relative_path<P: AsRef<Path>>(path: P, base: P) -> Option<PathBuf> {
        let path = path.as_ref();
        let base = base.as_ref();
        
        path.strip_prefix(base).ok().map(|p| p.to_path_buf())
    }

    /// 检查路径是否安全（防止路径遍历攻击）
    pub fn is_safe_path<P: AsRef<Path>>(path: P, base: P) -> bool {
        let path = path.as_ref();
        let base = base.as_ref();
        
        // 规范化路径
        let normalized_path = Self::normalize(path);
        let normalized_base = Self::normalize(base);
        
        // 检查是否在基础目录内
        normalized_path.starts_with(normalized_base)
    }

    /// 生成临时文件路径
    pub fn temp_file_path(prefix: &str, suffix: &str) -> PathBuf {
        let timestamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos();
        
        let filename = format!("{}_{}{}", prefix, timestamp, suffix);
        std::env::temp_dir().join(filename)
    }

    /// 获取文件扩展名
    pub fn get_extension<P: AsRef<Path>>(path: P) -> Option<String> {
        path.as_ref()
            .extension()
            .and_then(|ext| ext.to_str())
            .map(|s| s.to_lowercase())
    }

    /// 获取不带扩展名的文件名
    pub fn get_stem<P: AsRef<Path>>(path: P) -> Option<String> {
        path.as_ref()
            .file_stem()
            .and_then(|stem| stem.to_str())
            .map(|s| s.to_string())
    }
}

/// 文件类型检测
pub struct FileType;

impl FileType {
    /// 检测文件类型
    pub fn detect<P: AsRef<Path>>(path: P) -> FileResult<String> {
        let path = path.as_ref();
        
        if !path.exists() {
            return Err(FileError::FileNotFound(path.display().to_string()));
        }

        // 基于扩展名的简单检测
        match PathUtils::get_extension(path).as_deref() {
            Some("txt") | Some("log") => Ok("text".to_string()),
            Some("json") => Ok("json".to_string()),
            Some("yaml") | Some("yml") => Ok("yaml".to_string()),
            Some("toml") => Ok("toml".to_string()),
            Some("csv") => Ok("csv".to_string()),
            Some("jpg") | Some("jpeg") | Some("png") | Some("gif") => Ok("image".to_string()),
            Some("mp4") | Some("avi") | Some("mkv") => Ok("video".to_string()),
            Some("mp3") | Some("wav") | Some("flac") => Ok("audio".to_string()),
            Some("zip") | Some("tar") | Some("gz") => Ok("archive".to_string()),
            _ => Ok("unknown".to_string()),
        }
    }

    /// 检查是否为文本文件
    pub fn is_text_file<P: AsRef<Path>>(path: P) -> bool {
        matches!(
            PathUtils::get_extension(path).as_deref(),
            Some("txt") | Some("log") | Some("json") | Some("yaml") | 
            Some("yml") | Some("toml") | Some("csv") | Some("md") | 
            Some("rs") | Some("py") | Some("js") | Some("html") | Some("css")
        )
    }

    /// 检查是否为图像文件
    pub fn is_image_file<P: AsRef<Path>>(path: P) -> bool {
        matches!(
            PathUtils::get_extension(path).as_deref(),
            Some("jpg") | Some("jpeg") | Some("png") | Some("gif") | 
            Some("bmp") | Some("svg") | Some("webp")
        )
    }
} 