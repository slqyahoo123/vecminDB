// 存储工具模块
// 提供通用的工具函数和辅助方法

use crate::{Error, Result};

/// 将字符串转换为字节数组
#[inline]
pub fn str_to_bytes(s: &str) -> Vec<u8> {
    s.as_bytes().to_vec()
}

/// 将字节数组转换为字符串
#[inline]
pub fn bytes_to_str(bytes: &[u8]) -> Result<&str> {
    std::str::from_utf8(bytes).map_err(|e| Error::invalid_data(e.to_string()))
}

/// 生成唯一ID
pub fn generate_id() -> String {
    uuid::Uuid::new_v4().to_string()
}

/// 计算数据的哈希值
pub fn calculate_hash(data: &[u8]) -> String {
    use sha2::{Sha256, Digest};
    let mut hasher = Sha256::new();
    hasher.update(data);
    let result = hasher.finalize();
    format!("{:x}", result)
}

/// 压缩数据
pub fn compress_data(data: &[u8]) -> Result<Vec<u8>> {
    use flate2::{Compression, write::ZlibEncoder};
    use std::io::Write;
    
    let mut encoder = ZlibEncoder::new(Vec::new(), Compression::default());
    encoder.write_all(data)?;
    Ok(encoder.finish()?)
}

/// 解压数据
pub fn decompress_data(data: &[u8]) -> Result<Vec<u8>> {
    use flate2::read::ZlibDecoder;
    use std::io::Read;
    
    let mut decoder = ZlibDecoder::new(data);
    let mut decompressed = Vec::new();
    decoder.read_to_end(&mut decompressed)?;
    Ok(decompressed)
}

/// 创建备份文件名
pub fn create_backup_filename() -> String {
    use chrono::Utc;
    format!("backup_{}.db", Utc::now().format("%Y%m%d%H%M%S"))
}

/// 获取文件大小
pub fn get_file_size(path: &std::path::Path) -> Result<u64> {
    use std::fs;
    let metadata = fs::metadata(path)?;
    Ok(metadata.len())
} 