// Data Processor Utilities
// 数据处理器工具函数

use std::fs;
use std::path::Path;
use crate::Result;
use crate::Error;

/// 检查文件是否存在
pub fn file_exists(path: &str) -> bool {
    Path::new(path).exists()
}

/// 确保目录存在，如果不存在则创建
pub fn ensure_dir_exists(path: &str) -> Result<()> {
    let path = Path::new(path);
    if !path.exists() {
        fs::create_dir_all(path).map_err(|e| Error::io_error(&format!("无法创建目录: {}", e)))?;
    }
    Ok(())
}

/// 读取文件内容
pub fn read_file(path: &str) -> Result<String> {
    fs::read_to_string(path).map_err(|e| Error::io_error(&format!("无法读取文件: {}", e)))
}

/// 写入文件内容
pub fn write_file(path: &str, content: &str) -> Result<()> {
    fs::write(path, content).map_err(|e| Error::io_error(&format!("无法写入文件: {}", e)))
}

/// 获取文件大小
pub fn get_file_size(path: &str) -> Result<u64> {
    let metadata = fs::metadata(path).map_err(|e| Error::io_error(&format!("无法获取文件元数据: {}", e)))?;
    Ok(metadata.len())
}

/// 检查路径是否为目录
pub fn is_directory(path: &str) -> bool {
    Path::new(path).is_dir()
}

/// 检查路径是否为文件
pub fn is_file(path: &str) -> bool {
    Path::new(path).is_file()
}

/// 获取文件扩展名
pub fn get_file_extension(path: &str) -> Option<String> {
    Path::new(path)
        .extension()
        .and_then(|ext| ext.to_str())
        .map(|ext| ext.to_lowercase())
}

/// 构建路径
pub fn join_path(base: &str, parts: &[&str]) -> String {
    let mut path = Path::new(base).to_path_buf();
    for part in parts {
        path.push(part);
    }
    path.to_string_lossy().to_string()
}

/// 创建临时文件名
pub fn create_temp_filename(prefix: &str, extension: &str) -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    
    let timestamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis();
    
    format!("{}_{}.{}", prefix, timestamp, extension)
}

/// 安全删除文件
pub fn safe_delete_file(path: &str) -> Result<()> {
    if file_exists(path) {
        fs::remove_file(path).map_err(|e| Error::io_error(&format!("无法删除文件: {}", e)))?;
    }
    Ok(())
}

/// 安全删除目录
pub fn safe_delete_dir(path: &str) -> Result<()> {
    if is_directory(path) {
        fs::remove_dir_all(path).map_err(|e| Error::io_error(&format!("无法删除目录: {}", e)))?;
    }
    Ok(())
}

/// 复制文件
pub fn copy_file(src: &str, dst: &str) -> Result<()> {
    fs::copy(src, dst).map_err(|e| Error::io_error(&format!("无法复制文件: {}", e)))?;
    Ok(())
}

/// 移动文件
pub fn move_file(src: &str, dst: &str) -> Result<()> {
    fs::rename(src, dst).map_err(|e| Error::io_error(&format!("无法移动文件: {}", e)))?;
    Ok(())
}

/// 列出目录中的文件
pub fn list_files(dir_path: &str) -> Result<Vec<String>> {
    let dir = fs::read_dir(dir_path).map_err(|e| Error::io_error(&format!("无法读取目录: {}", e)))?;
    
    let mut files = Vec::new();
    for entry in dir {
        let entry = entry.map_err(|e| Error::io_error(&format!("无法读取目录项: {}", e)))?;
        if entry.file_type().map_err(|e| Error::io_error(&format!("无法获取文件类型: {}", e)))?.is_file() {
            if let Some(filename) = entry.file_name().to_str() {
                files.push(filename.to_string());
            }
        }
    }
    
    Ok(files)
}

/// 计算文件哈希值
pub fn calculate_file_hash(path: &str) -> Result<String> {
    use std::hash::{Hash, Hasher};
    use std::collections::hash_map::DefaultHasher;
    
    let content = fs::read(path).map_err(|e| Error::io_error(&format!("无法读取文件: {}", e)))?;
    
    let mut hasher = DefaultHasher::new();
    content.hash(&mut hasher);
    Ok(format!("{:x}", hasher.finish()))
}

/// 检查磁盘空间
pub fn check_disk_space(path: &str) -> Result<u64> {
    check_disk_space_detailed(path).map(|info| info.available)
}

/// 获取详细的磁盘空间信息
pub fn get_disk_space_info(path: &str) -> Result<DiskSpaceInfo> {
    check_disk_space_detailed(path)
}

/// 磁盘空间信息结构
#[derive(Debug, Clone)]
pub struct DiskSpaceInfo {
    /// 总空间大小（字节）
    pub total: u64,
    /// 可用空间大小（字节）
    pub available: u64,
    /// 已使用空间大小（字节）
    pub used: u64,
    /// 文件系统类型
    pub filesystem: String,
    /// 挂载点
    pub mount_point: String,
}

impl DiskSpaceInfo {
    /// 获取使用率百分比
    pub fn usage_percent(&self) -> f64 {
        if self.total > 0 {
            (self.used as f64 / self.total as f64) * 100.0
        } else {
            0.0
        }
    }
    
    /// 获取可用率百分比
    pub fn available_percent(&self) -> f64 {
        if self.total > 0 {
            (self.available as f64 / self.total as f64) * 100.0
        } else {
            0.0
        }
    }
    
    /// 检查是否空间不足
    pub fn is_low_space(&self, threshold_percent: f64) -> bool {
        self.available_percent() < threshold_percent
    }
    
    /// 检查是否有足够空间存储指定大小的数据
    pub fn has_space_for(&self, required_bytes: u64) -> bool {
        self.available >= required_bytes
    }
    
    /// 格式化磁盘空间信息为可读字符串
    pub fn format_readable(&self) -> String {
        format!(
            "总空间: {}, 已用: {} ({:.1}%), 可用: {} ({:.1}%), 文件系统: {}, 挂载点: {}",
            format_bytes(self.total),
            format_bytes(self.used),
            self.usage_percent(),
            format_bytes(self.available),
            self.available_percent(),
            self.filesystem,
            self.mount_point
        )
    }
}

/// 跨平台磁盘空间检查的内部实现
fn check_disk_space_detailed(path: &str) -> Result<DiskSpaceInfo> {
    let path = Path::new(path);
    
    // 确保路径存在
    if !path.exists() {
        return Err(Error::io_error(&format!("路径不存在: {}", path.display())));
    }
    
    // 获取实际路径（处理符号链接）
    let canonical_path = fs::canonicalize(path)
        .map_err(|e| Error::io_error(&format!("无法获取标准路径: {}", e)))?;
    
    #[cfg(unix)]
    {
        check_disk_space_unix(&canonical_path)
    }
    
    #[cfg(windows)]
    {
        check_disk_space_windows(&canonical_path)
    }
    
    #[cfg(not(any(unix, windows)))]
    {
        // 对于其他平台，提供基本实现
        check_disk_space_fallback(&canonical_path)
    }
}

/// Unix/Linux平台的磁盘空间检查
#[cfg(unix)]
fn check_disk_space_unix(path: &Path) -> Result<DiskSpaceInfo> {
    use std::ffi::CString;
    use std::mem;
    use std::os::unix::ffi::OsStrExt;
    
    #[repr(C)]
    struct StatVfs {
        f_bsize: u64,    // 文件系统块大小
        f_frsize: u64,   // 片段大小
        f_blocks: u64,   // 总块数
        f_bfree: u64,    // 空闲块数
        f_bavail: u64,   // 非超级用户可用块数
        f_files: u64,    // 总文件节点数
        f_ffree: u64,    // 空闲文件节点数
        f_favail: u64,   // 非超级用户可用文件节点数
        f_fsid: u64,     // 文件系统ID
        f_flag: u64,     // 挂载标志
        f_namemax: u64,  // 最大文件名长度
    }
    
    extern "C" {
        fn statvfs(path: *const i8, buf: *mut StatVfs) -> i32;
    }
    
    let path_cstr = CString::new(path.as_os_str().as_bytes())
        .map_err(|e| Error::io_error(&format!("路径转换失败: {}", e)))?;
    
    let mut statvfs_buf: StatVfs = unsafe { mem::zeroed() };
    let result = unsafe { statvfs(path_cstr.as_ptr(), &mut statvfs_buf) };
    
    if result != 0 {
        return Err(Error::io_error(&format!(
            "statvfs调用失败: {}",
            std::io::Error::last_os_error()
        )));
    }
    
    let block_size = statvfs_buf.f_frsize;
    let total = statvfs_buf.f_blocks * block_size;
    let available = statvfs_buf.f_bavail * block_size;
    let used = total - (statvfs_buf.f_bfree * block_size);
    
    // 获取文件系统类型和挂载点
    let (filesystem, mount_point) = get_mount_info_unix(path)?;
    
    Ok(DiskSpaceInfo {
        total,
        available,
        used,
        filesystem,
        mount_point,
    })
}

/// 获取Unix/Linux系统的挂载信息
#[cfg(unix)]
fn get_mount_info_unix(path: &Path) -> Result<(String, String)> {
    // 读取/proc/mounts文件来获取挂载信息
    let mounts_content = fs::read_to_string("/proc/mounts")
        .unwrap_or_else(|_| String::new());
    
    let path_str = path.to_string_lossy();
    let mut best_match = ("unknown".to_string(), "/".to_string());
    let mut best_match_len = 0;
    
    for line in mounts_content.lines() {
        let parts: Vec<&str> = line.split_whitespace().collect();
        if parts.len() >= 3 {
            let mount_point = parts[1];
            let filesystem = parts[2];
            
            // 找到最长匹配的挂载点
            if path_str.starts_with(mount_point) && mount_point.len() > best_match_len {
                best_match = (filesystem.to_string(), mount_point.to_string());
                best_match_len = mount_point.len();
            }
        }
    }
    
    Ok(best_match)
}

/// Windows平台的磁盘空间检查
#[cfg(windows)]
fn check_disk_space_windows(path: &Path) -> Result<DiskSpaceInfo> {
    use std::ffi::OsStr;
    use std::iter::once;
    use std::os::windows::ffi::OsStrExt;
    
    // 获取驱动器根路径
    let drive_letter = get_drive_letter_windows(path)?;
    let drive_path = format!("{}:\\", drive_letter);
    
    // 转换为宽字符
    let wide_path: Vec<u16> = OsStr::new(&drive_path)
        .encode_wide()
        .chain(once(0))
        .collect();
    
    let mut free_bytes_available = 0u64;
    let mut total_bytes = 0u64;
    let mut total_free_bytes = 0u64;
    
    extern "system" {
        fn GetDiskFreeSpaceExW(
            directory_name: *const u16,
            free_bytes_available: *mut u64,
            total_bytes: *mut u64,
            total_free_bytes: *mut u64,
        ) -> i32;
    }
    
    let result = unsafe {
        GetDiskFreeSpaceExW(
            wide_path.as_ptr(),
            &mut free_bytes_available,
            &mut total_bytes,
            &mut total_free_bytes,
        )
    };
    
    if result == 0 {
        return Err(Error::io_error(&format!(
            "GetDiskFreeSpaceExW调用失败: {}",
            std::io::Error::last_os_error()
        )));
    }
    
    let used = total_bytes - total_free_bytes;
    
    // 获取文件系统类型
    let filesystem = get_filesystem_type_windows(&drive_path)?;
    
    Ok(DiskSpaceInfo {
        total: total_bytes,
        available: free_bytes_available,
        used,
        filesystem,
        mount_point: drive_path,
    })
}

/// 获取Windows驱动器字母
#[cfg(windows)]
fn get_drive_letter_windows(path: &Path) -> Result<char> {
    let path_str = path.to_string_lossy();
    if path_str.len() >= 2 && path_str.chars().nth(1) == Some(':') {
        Ok(path_str.chars().next().unwrap().to_ascii_uppercase())
    } else {
        Err(Error::io_error("无法确定Windows驱动器字母".to_string()))
    }
}

/// 获取Windows文件系统类型
#[cfg(windows)]
fn get_filesystem_type_windows(drive_path: &str) -> Result<String> {
    use std::ffi::OsStr;
    use std::iter::once;
    use std::os::windows::ffi::OsStrExt;
    
    let wide_path: Vec<u16> = OsStr::new(drive_path)
        .encode_wide()
        .chain(once(0))
        .collect();
    
    let mut filesystem_name = vec![0u16; 32];
    let mut volume_name = vec![0u16; 64];
    let mut volume_serial = 0u32;
    let mut max_component_len = 0u32;
    let mut filesystem_flags = 0u32;
    
    extern "system" {
        fn GetVolumeInformationW(
            root_path_name: *const u16,
            volume_name_buffer: *mut u16,
            volume_name_size: u32,
            volume_serial_number: *mut u32,
            max_component_length: *mut u32,
            filesystem_flags: *mut u32,
            filesystem_name_buffer: *mut u16,
            filesystem_name_size: u32,
        ) -> i32;
    }
    
    let result = unsafe {
        GetVolumeInformationW(
            wide_path.as_ptr(),
            volume_name.as_mut_ptr(),
            volume_name.len() as u32,
            &mut volume_serial,
            &mut max_component_len,
            &mut filesystem_flags,
            filesystem_name.as_mut_ptr(),
            filesystem_name.len() as u32,
        )
    };
    
    if result != 0 {
        // 将宽字符转换为字符串
        let end = filesystem_name.iter().position(|&c| c == 0).unwrap_or(filesystem_name.len());
        let filesystem = String::from_utf16_lossy(&filesystem_name[..end]);
        Ok(filesystem)
    } else {
        Ok("UNKNOWN".to_string())
    }
}

/// 后备磁盘空间检查实现（适用于不支持的平台）
#[cfg(not(any(unix, windows)))]
fn check_disk_space_fallback(path: &Path) -> Result<DiskSpaceInfo> {
    // 对于不支持的平台，尝试通过文件系统操作估算
    let metadata = fs::metadata(path)
        .map_err(|e| Error::io_error(&format!("无法获取文件元数据: {}", e)))?;
    
    // 提供一个保守的估算
    let estimated_total = 100_000_000_000u64; // 100GB
    let estimated_available = estimated_total / 2; // 假设50%可用
    let estimated_used = estimated_total - estimated_available;
    
    Ok(DiskSpaceInfo {
        total: estimated_total,
        available: estimated_available,
        used: estimated_used,
        filesystem: "unknown".to_string(),
        mount_point: path.to_string_lossy().to_string(),
    })
}

/// 格式化字节大小为可读字符串
pub fn format_bytes(bytes: u64) -> String {
    const UNITS: &[&str] = &["B", "KB", "MB", "GB", "TB", "PB"];
    
    if bytes == 0 {
        return "0 B".to_string();
    }
    
    let mut size = bytes as f64;
    let mut unit_index = 0;
    
    while size >= 1024.0 && unit_index < UNITS.len() - 1 {
        size /= 1024.0;
        unit_index += 1;
    }
    
    if unit_index == 0 {
        format!("{} {}", bytes, UNITS[unit_index])
    } else {
        format!("{:.1} {}", size, UNITS[unit_index])
    }
}

/// 检查指定路径是否有足够的空间存储数据
pub fn check_sufficient_space(path: &str, required_bytes: u64) -> Result<bool> {
    let disk_info = check_disk_space_detailed(path)?;
    Ok(disk_info.has_space_for(required_bytes))
}

/// 获取路径的磁盘使用率
pub fn get_disk_usage_percent(path: &str) -> Result<f64> {
    let disk_info = check_disk_space_detailed(path)?;
    Ok(disk_info.usage_percent())
}

/// 监控磁盘空间并在空间不足时发出警告
pub fn monitor_disk_space(path: &str, warning_threshold_percent: f64) -> Result<DiskSpaceInfo> {
    let disk_info = check_disk_space_detailed(path)?;
    
    if disk_info.is_low_space(warning_threshold_percent) {
        eprintln!(
            "警告: 磁盘空间不足！路径: {}, 可用空间: {:.1}%, 阈值: {:.1}%",
            path,
            disk_info.available_percent(),
            warning_threshold_percent
        );
    }
    
    Ok(disk_info)
} 