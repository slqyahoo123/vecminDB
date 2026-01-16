//! 压缩工具模块 - 提供数据压缩和解压缩功能

use std::io::{self, Read, Write};
use flate2::Compression;
use flate2::read::{GzDecoder, ZlibDecoder};
use flate2::write::{GzEncoder, ZlibEncoder};
use brotli;
use zstd::stream::{encode_all, decode_all};
use std::path::Path;
use std::fs::File;
use std::io::{BufReader, BufWriter};

use crate::error::{Error, Result};

/// 压缩算法
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum CompressionAlgorithm {
    /// GZIP 压缩
    Gzip,
    /// Zlib 压缩
    Zlib,
    /// LZ4 压缩
    Lz4,
    /// Brotli 压缩
    Brotli,
    /// Zstandard 压缩
    Zstd,
}

/// 压缩级别
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum CompressionLevel {
    /// 无压缩
    None,
    /// 最快压缩（低压缩率）
    Fastest,
    /// 默认压缩（平衡速度和压缩率）
    Default,
    /// 最佳压缩（高压缩率，但较慢）
    Best,
    /// 自定义压缩级别 (0-10)
    Custom(u32),
}

impl CompressionLevel {
    /// 转换为 flate2 压缩级别
    fn to_flate2_level(&self) -> Compression {
        match self {
            CompressionLevel::None => Compression::none(),
            CompressionLevel::Fastest => Compression::fast(),
            CompressionLevel::Default => Compression::default(),
            CompressionLevel::Best => Compression::best(),
            CompressionLevel::Custom(level) => Compression::new(*level as u32),
        }
    }
    
    /// 转换为 brotli 压缩级别
    fn to_brotli_level(&self) -> u32 {
        match self {
            CompressionLevel::None => 0,
            CompressionLevel::Fastest => 1,
            CompressionLevel::Default => 4,
            CompressionLevel::Best => 11,
            CompressionLevel::Custom(level) => (*level).min(11),
        }
    }
    
    /// 转换为 zstd 压缩级别
    fn to_zstd_level(&self) -> i32 {
        match self {
            CompressionLevel::None => 1,
            CompressionLevel::Fastest => 1,
            CompressionLevel::Default => 3,
            CompressionLevel::Best => 19,
            CompressionLevel::Custom(level) => (*level as i32).min(19),
        }
    }
    
    /// 转换为 lz4 压缩级别
    fn to_lz4_level(&self) -> u32 {
        match self {
            CompressionLevel::None => 0,
            CompressionLevel::Fastest => 1,
            CompressionLevel::Default => 5,
            CompressionLevel::Best => 16,
            CompressionLevel::Custom(level) => (*level).min(16),
        }
    }
}

/// 压缩工具结构体
pub struct CompressionUtils;

impl CompressionUtils {
    /// 使用 GZIP 压缩数据
    pub fn compress_gzip(data: &[u8], level: CompressionLevel) -> Result<Vec<u8>> {
        let mut encoder = GzEncoder::new(Vec::new(), level.to_flate2_level());
        encoder.write_all(data).map_err(|e| Error::io_error(format!("GZIP压缩失败: {}", e)))?;
        encoder.finish().map_err(|e| Error::io_error(format!("完成GZIP压缩失败: {}", e)))
    }
    
    /// 解压 GZIP 数据
    pub fn decompress_gzip(data: &[u8]) -> Result<Vec<u8>> {
        let mut decoder = GzDecoder::new(data);
        let mut output = Vec::new();
        decoder.read_to_end(&mut output).map_err(|e| Error::io_error(format!("GZIP解压失败: {}", e)))?;
        Ok(output)
    }
    
    /// 使用 Zlib 压缩数据
    pub fn compress_zlib(data: &[u8], level: CompressionLevel) -> Result<Vec<u8>> {
        let mut encoder = ZlibEncoder::new(Vec::new(), level.to_flate2_level());
        encoder.write_all(data).map_err(|e| Error::io_error(format!("Zlib压缩失败: {}", e)))?;
        encoder.finish().map_err(|e| Error::io_error(format!("完成Zlib压缩失败: {}", e)))
    }
    
    /// 解压 Zlib 数据
    pub fn decompress_zlib(data: &[u8]) -> Result<Vec<u8>> {
        let mut decoder = ZlibDecoder::new(data);
        let mut output = Vec::new();
        decoder.read_to_end(&mut output).map_err(|e| Error::io_error(format!("Zlib解压失败: {}", e)))?;
        Ok(output)
    }
    
    /// 使用 Brotli 压缩数据
    pub fn compress_brotli(data: &[u8], level: CompressionLevel) -> Result<Vec<u8>> {
        let quality = level.to_brotli_level();
        
        // 使用 Brotli 压缩
        let mut output = Vec::new();
        let mut writer = brotli::CompressorWriter::new(&mut output, 4096, quality as u32, 22);
        writer.write_all(data).map_err(|e| Error::io_error(format!("Brotli压缩失败: {}", e)))?;
        writer.flush().map_err(|e| Error::io_error(format!("Brotli压缩刷新失败: {}", e)))?;
        drop(writer);
        Ok(output)
    }
    
    /// 解压 Brotli 数据
    pub fn decompress_brotli(data: &[u8]) -> Result<Vec<u8>> {
        let mut output = Vec::new();
        let mut reader = brotli::Decompressor::new(std::io::Cursor::new(data), 4096);
        reader.read_to_end(&mut output).map_err(|e| Error::io_error(format!("Brotli解压失败: {}", e)))?;
        Ok(output)
    }
    
    /// 使用 Zstandard 压缩数据
    pub fn compress_zstd(data: &[u8], level: CompressionLevel) -> Result<Vec<u8>> {
        let compression_level = level.to_zstd_level();
        encode_all(data, compression_level).map_err(|e| Error::io_error(format!("Zstd压缩失败: {}", e)))
    }
    
    /// 解压 Zstandard 数据
    pub fn decompress_zstd(data: &[u8]) -> Result<Vec<u8>> {
        decode_all(data).map_err(|e| Error::io_error(format!("Zstd解压失败: {}", e)))
    }
    
    /// 压缩 LZ4 数据（使用Deflate替代）
    pub fn compress_lz4(data: &[u8], level: CompressionLevel) -> Result<Vec<u8>> {
        // 使用Deflate压缩作为LZ4的替代方案
        let mut encoder = flate2::write::DeflateEncoder::new(Vec::new(), level.to_flate2_level());
        encoder.write_all(data)
            .map_err(|e| Error::io_error(format!("Deflate压缩失败: {}", e)))?;
        encoder.finish()
            .map_err(|e| Error::io_error(format!("完成Deflate压缩失败: {}", e)))
    }
    
    /// 解压 LZ4 数据（使用Deflate替代）
    pub fn decompress_lz4(data: &[u8], _decompressed_size: Option<usize>) -> Result<Vec<u8>> {
        // 使用Deflate解压作为LZ4的替代方案
        let mut decompressed = Vec::new();
        let mut decoder = flate2::read::DeflateDecoder::new(data);
        decoder.read_to_end(&mut decompressed)
            .map_err(|e| Error::io_error(format!("Deflate解压失败: {}", e)))?;
        Ok(decompressed)
    }
    
    /// 使用指定算法压缩数据
    pub fn compress(data: &[u8], algorithm: CompressionAlgorithm, level: CompressionLevel) -> Result<Vec<u8>> {
        match algorithm {
            CompressionAlgorithm::Gzip => Self::compress_gzip(data, level),
            CompressionAlgorithm::Zlib => Self::compress_zlib(data, level),
            CompressionAlgorithm::Brotli => Self::compress_brotli(data, level),
            CompressionAlgorithm::Zstd => Self::compress_zstd(data, level),
            CompressionAlgorithm::Lz4 => Self::compress_lz4(data, level),
        }
    }
    
    /// 使用指定算法解压数据
    pub fn decompress(data: &[u8], algorithm: CompressionAlgorithm, decompressed_size: Option<usize>) -> Result<Vec<u8>> {
        match algorithm {
            CompressionAlgorithm::Gzip => Self::decompress_gzip(data),
            CompressionAlgorithm::Zlib => Self::decompress_zlib(data),
            CompressionAlgorithm::Brotli => Self::decompress_brotli(data),
            CompressionAlgorithm::Zstd => Self::decompress_zstd(data),
            CompressionAlgorithm::Lz4 => Self::decompress_lz4(data, decompressed_size),
        }
    }
    
    /// 压缩文件
    pub fn compress_file<P: AsRef<Path>>(
        input_path: P, 
        output_path: P, 
        algorithm: CompressionAlgorithm,
        level: CompressionLevel
    ) -> Result<usize> {
        let input = File::open(&input_path)
            .map_err(|e| Error::io_error(format!("无法打开输入文件 {}: {}", input_path.as_ref().display(), e)))?;
        let output = File::create(&output_path)
            .map_err(|e| Error::io_error(format!("无法创建输出文件 {}: {}", output_path.as_ref().display(), e)))?;
        
        let input_size = input.metadata()
            .map_err(|e| Error::io_error(format!("无法获取输入文件大小: {}", e)))?
            .len() as usize;
            
        let mut reader = BufReader::new(input);
        let mut compressed_size = 0;
        
        match algorithm {
            CompressionAlgorithm::Gzip => {
                let mut encoder = GzEncoder::new(BufWriter::new(output), level.to_flate2_level());
                io::copy(&mut reader, &mut encoder)
                    .map_err(|e| Error::io_error(format!("GZIP压缩文件失败: {}", e)))?;
                    
                let writer = encoder.finish()
                    .map_err(|e| Error::io_error(format!("完成GZIP压缩失败: {}", e)))?;
                let file = writer.into_inner()
                    .map_err(|e| Error::io_error(format!("获取输出文件失败: {}", e)))?;
                compressed_size = file.metadata()
                    .map_err(|e| Error::io_error(format!("获取输出文件大小失败: {}", e)))?
                    .len() as usize;
            },
            CompressionAlgorithm::Zlib => {
                let mut encoder = ZlibEncoder::new(BufWriter::new(output), level.to_flate2_level());
                io::copy(&mut reader, &mut encoder)
                    .map_err(|e| Error::io_error(format!("Zlib压缩文件失败: {}", e)))?;
                    
                let writer = encoder.finish()
                    .map_err(|e| Error::io_error(format!("完成Zlib压缩失败: {}", e)))?;
                let file = writer.into_inner()
                    .map_err(|e| Error::io_error(format!("获取输出文件失败: {}", e)))?;
                compressed_size = file.metadata()
                    .map_err(|e| Error::io_error(format!("获取输出文件大小失败: {}", e)))?
                    .len() as usize;
            },
            _ => {
                // 对于其他算法，先读取整个文件，然后压缩
                let mut data = Vec::new();
                reader.read_to_end(&mut data)
                    .map_err(|e| Error::io_error(format!("读取输入文件失败: {}", e)))?;
                    
                let compressed = Self::compress(&data, algorithm, level)?;
                compressed_size = compressed.len();
                
                let mut writer = BufWriter::new(output);
                writer.write_all(&compressed)
                    .map_err(|e| Error::io_error(format!("写入压缩数据失败: {}", e)))?;
                    
                writer.flush()
                    .map_err(|e| Error::io_error(format!("刷新输出缓冲区失败: {}", e)))?;
            }
        }
        
        // 计算压缩比
        let compression_ratio = if input_size > 0 {
            (compressed_size as f64 / input_size as f64) * 100.0
        } else {
            0.0
        };
        
        log::debug!(
            "压缩文件: {} -> {}, 原始大小: {}, 压缩大小: {}, 压缩比: {:.2}%",
            input_path.as_ref().display(),
            output_path.as_ref().display(),
            input_size,
            compressed_size,
            compression_ratio
        );
        
        Ok(compressed_size)
    }
    
    /// 解压文件
    pub fn decompress_file<P: AsRef<Path>>(
        input_path: P, 
        output_path: P, 
        algorithm: CompressionAlgorithm
    ) -> Result<usize> {
        let input = File::open(&input_path)
            .map_err(|e| Error::io_error(format!("无法打开输入文件 {}: {}", input_path.as_ref().display(), e)))?;
        let output = File::create(&output_path)
            .map_err(|e| Error::io_error(format!("无法创建输出文件 {}: {}", output_path.as_ref().display(), e)))?;
        
        let input_size = input.metadata()
            .map_err(|e| Error::io_error(format!("无法获取输入文件大小: {}", e)))?
            .len() as usize;
            
        let mut reader = BufReader::new(input);
        let mut writer = BufWriter::new(output);
        let mut decompressed_size = 0;
        
        match algorithm {
            CompressionAlgorithm::Gzip => {
                let mut decoder = GzDecoder::new(reader);
                decompressed_size = io::copy(&mut decoder, &mut writer)
                    .map_err(|e| Error::io_error(format!("GZIP解压文件失败: {}", e)))? as usize;
            },
            CompressionAlgorithm::Zlib => {
                let mut decoder = ZlibDecoder::new(reader);
                decompressed_size = io::copy(&mut decoder, &mut writer)
                    .map_err(|e| Error::io_error(format!("Zlib解压文件失败: {}", e)))? as usize;
            },
            _ => {
                // 对于其他算法，先读取整个文件，然后解压
                let mut data = Vec::new();
                reader.read_to_end(&mut data)
                    .map_err(|e| Error::io_error(format!("读取输入文件失败: {}", e)))?;
                    
                let decompressed = Self::decompress(&data, algorithm, None)?;
                decompressed_size = decompressed.len();
                
                writer.write_all(&decompressed)
                    .map_err(|e| Error::io_error(format!("写入解压数据失败: {}", e)))?;
            }
        }
        
        writer.flush()
            .map_err(|e| Error::io_error(format!("刷新输出缓冲区失败: {}", e)))?;
        
        // 计算解压比
        let decompression_ratio = if input_size > 0 {
            (decompressed_size as f64 / input_size as f64) * 100.0
        } else {
            0.0
        };
        
        log::debug!(
            "解压文件: {} -> {}, 压缩大小: {}, 解压大小: {}, 解压比: {:.2}%",
            input_path.as_ref().display(),
            output_path.as_ref().display(),
            input_size,
            decompressed_size,
            decompression_ratio
        );
        
        Ok(decompressed_size)
    }
    
    /// 检测压缩算法（简单启发式方法）
    pub fn detect_compression_algorithm(data: &[u8]) -> Option<CompressionAlgorithm> {
        if data.len() < 2 {
            return None;
        }
        
        // GZIP 魔数: 0x1F 0x8B
        if data[0] == 0x1F && data[1] == 0x8B {
            return Some(CompressionAlgorithm::Gzip);
        }
        
        // Zlib 头: 0x78 + (0x01、0x9C、0xDA 之一)
        if data[0] == 0x78 && (data[1] == 0x01 || data[1] == 0x9C || data[1] == 0xDA) {
            return Some(CompressionAlgorithm::Zlib);
        }
        
        // Zstd 魔数: 0x28 0xB5 0x2F 0xFD
        if data.len() >= 4 && data[0] == 0x28 && data[1] == 0xB5 && data[2] == 0x2F && data[3] == 0xFD {
            return Some(CompressionAlgorithm::Zstd);
        }
        
        // Brotli 启发式检测（没有统一的魔数，但通常以0xCE开头）
        // 注意: 这不是100%可靠的检测方法
        if data.len() >= 2 && data[0] == 0xCE {
            return Some(CompressionAlgorithm::Brotli);
        }
        
        // LZ4 帧格式魔数: 0x04 0x22 0x4D 0x18
        if data.len() >= 4 && data[0] == 0x04 && data[1] == 0x22 && data[2] == 0x4D && data[3] == 0x18 {
            return Some(CompressionAlgorithm::Lz4);
        }
        
        None
    }
    
    /// 推荐算法（根据数据类型和大小）
    pub fn recommend_algorithm(size: usize, is_text: bool, prioritize_speed: bool) -> CompressionAlgorithm {
        if prioritize_speed {
            // 如果优先考虑速度
            if size < 100_000 { // 小于100KB
                CompressionAlgorithm::Lz4
            } else {
                if is_text {
                    CompressionAlgorithm::Zstd
                } else {
                    CompressionAlgorithm::Lz4
                }
            }
        } else {
            // 如果优先考虑压缩率
            if is_text {
                CompressionAlgorithm::Brotli 
            } else {
                if size > 1_000_000 { // 大于1MB
                    CompressionAlgorithm::Zstd
                } else {
                    CompressionAlgorithm::Gzip
                }
            }
        }
    }
    
    /// 递归压缩目录
    pub fn compress_directory<P: AsRef<Path>>(
        input_dir: P,
        output_file: P,
        algorithm: CompressionAlgorithm,
        level: CompressionLevel
    ) -> Result<usize> {
        #[cfg(any(feature = "tempfile", feature = "walkdir"))]
        use tar::{Builder, Header};
        
        // 创建临时TAR文件
        #[cfg(feature = "tempfile")]
        let temp_tar = tempfile::NamedTempFile::new()
            .map_err(|e| Error::io_error(format!("创建临时TAR文件失败: {}", e)))?;
        #[cfg(all(feature = "tempfile", feature = "walkdir"))]
        {
            // 创建TAR builder
            let mut builder = Builder::new(temp_tar.reopen()
                .map_err(|e| Error::io_error(format!("重新打开临时文件失败: {}", e)))?);
            
            // 添加目录中的所有文件到TAR
            for entry in walkdir::WalkDir::new(&input_dir)
                .min_depth(1)  // 跳过根目录本身
                .into_iter()
                .filter_map(Result::ok)
            {
                let path = entry.path();
                let relative_path = path.strip_prefix(&input_dir)
                    .map_err(|e| Error::io_error(format!("计算相对路径失败: {}", e)))?;
                    
                if path.is_file() {
                    let mut file = File::open(path)
                        .map_err(|e| Error::io_error(format!("打开文件失败 {}: {}", path.display(), e)))?;
                    
                    let mut header = Header::new_gnu();
                    header.set_path(relative_path)
                        .map_err(|e| Error::io_error(format!("设置TAR头路径失败: {}", e)))?;
                    header.set_size(path.metadata()
                        .map_err(|e| Error::io_error(format!("获取文件元数据失败: {}", e)))?
                        .len());
                    header.set_mode(0o644);
                    header.set_mtime(path.metadata()
                        .map_err(|e| Error::io_error(format!("获取文件修改时间失败: {}", e)))?
                        .modified()
                        .map_err(|e| Error::io_error(format!("获取文件修改时间失败: {}", e)))?
                        .duration_since(std::time::UNIX_EPOCH)
                        .map_err(|e| Error::io_error(format!("计算修改时间戳失败: {}", e)))?
                        .as_secs() as u64);
                    
                    builder.append_data(&mut header, relative_path, &mut file)
                        .map_err(|e| Error::io_error(format!("添加文件到TAR失败: {}", e)))?;
                } else if path.is_dir() {
                    builder.append_dir(relative_path, path)
                        .map_err(|e| Error::io_error(format!("添加目录到TAR失败: {}", e)))?;
                }
            }
            
            // 完成TAR文件并压缩
            builder.finish()
                .map_err(|e| Error::io_error(format!("完成TAR文件失败: {}", e)))?;
            
            let temp_path = temp_tar.path();
            let compressed_size = Self::compress_file(temp_path, &output_file, algorithm, level)?;
            Ok(compressed_size)
        }
        
        #[cfg(not(all(feature = "tempfile", feature = "walkdir")))]
        {
            return Err(Error::feature_not_enabled("tempfile and walkdir"));
        }
    }
    
    /// 解压归档到目录
    pub fn decompress_archive<P: AsRef<Path>>(
        input_file: P,
        output_dir: P,
        algorithm: CompressionAlgorithm
    ) -> Result<usize> {
        // 创建临时TAR文件并解压
        #[cfg(feature = "tempfile")]
        {
            let temp_tar = tempfile::NamedTempFile::new()
                .map_err(|e| Error::io_error(format!("创建临时TAR文件失败: {}", e)))?;
            let temp_path = temp_tar.path().to_path_buf();
            
            // 解压缩到临时TAR文件
            Self::decompress_file(&input_file, &temp_path, algorithm)?;
            
            // 确保输出目录存在
            fs::create_dir_all(&output_dir)
                .map_err(|e| Error::io_error(format!("创建输出目录失败 {}: {}", output_dir.as_ref().display(), e)))?;
            
            // 打开并解压TAR文件
            let tar_file = File::open(&temp_path)
                .map_err(|e| Error::io_error(format!("打开TAR文件失败: {}", e)))?;
            let mut archive = tar::Archive::new(tar_file);
            
            // 解压到目标目录
            archive.unpack(&output_dir)
                .map_err(|e| Error::io_error(format!("解压TAR文件失败: {}", e)))?;
            
            // 计算解压后的总大小
            let mut total_size = 0;
            #[cfg(feature = "walkdir")]
            for entry in walkdir::WalkDir::new(&output_dir)
                .into_iter()
                .filter_map(Result::ok)
            {
                if entry.path().is_file() {
                    total_size += entry.metadata()
                        .map_err(|e| Error::io_error(format!("获取文件元数据失败: {}", e)))?
                        .len() as usize;
                }
            }
            #[cfg(not(feature = "walkdir"))]
            {
                // 如果 walkdir 不可用，返回0
                total_size = 0;
            }
            
            Ok(total_size)
        }
        
        #[cfg(not(feature = "tempfile"))]
        {
            Err(Error::feature_not_enabled("tempfile"))
        }
    }
} 