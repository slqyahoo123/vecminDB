//! 压缩工具模块
//! 
//! 提供数据压缩/解压缩功能

use crate::Result;
use std::io::{Read, Write};

/// 压缩算法
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CompressionAlgorithm {
    /// 无压缩
    None,
    /// LZ4（快速）
    Lz4,
    /// Zstd（高压缩比）
    Zstd,
    /// Gzip（通用）
    Gzip,
}

/// 压缩器
pub struct Compressor {
    algorithm: CompressionAlgorithm,
}

impl Compressor {
    /// 创建新的压缩器
    pub fn new(algorithm: CompressionAlgorithm) -> Self {
        Self { algorithm }
    }

    /// 创建默认压缩器（使用LZ4）
    pub fn default() -> Self {
        Self::new(CompressionAlgorithm::Lz4)
    }

    /// 压缩数据
    pub fn compress(&self, data: &[u8]) -> Result<Vec<u8>> {
        match self.algorithm {
            CompressionAlgorithm::None => Ok(data.to_vec()),
            CompressionAlgorithm::Lz4 => self.compress_lz4(data),
            CompressionAlgorithm::Zstd => self.compress_zstd(data),
            CompressionAlgorithm::Gzip => self.compress_gzip(data),
        }
    }

    /// 解压缩数据
    pub fn decompress(&self, data: &[u8]) -> Result<Vec<u8>> {
        match self.algorithm {
            CompressionAlgorithm::None => Ok(data.to_vec()),
            CompressionAlgorithm::Lz4 => self.decompress_lz4(data),
            CompressionAlgorithm::Zstd => self.decompress_zstd(data),
            CompressionAlgorithm::Gzip => self.decompress_gzip(data),
        }
    }

    // LZ4压缩
    fn compress_lz4(&self, data: &[u8]) -> Result<Vec<u8>> {
        lz4_flex::compress_prepend_size(data)
            .map_err(|e| crate::Error::compression(format!("LZ4压缩失败: {}", e)))
            .map(|v| v)
    }

    fn decompress_lz4(&self, data: &[u8]) -> Result<Vec<u8>> {
        lz4_flex::decompress_size_prepended(data)
            .map_err(|e| crate::Error::compression(format!("LZ4解压缩失败: {}", e)))
    }

    // Zstd压缩
    fn compress_zstd(&self, data: &[u8]) -> Result<Vec<u8>> {
        zstd::encode_all(data, 3)
            .map_err(|e| crate::Error::compression(format!("Zstd压缩失败: {}", e)))
    }

    fn decompress_zstd(&self, data: &[u8]) -> Result<Vec<u8>> {
        zstd::decode_all(data)
            .map_err(|e| crate::Error::compression(format!("Zstd解压缩失败: {}", e)))
    }

    // Gzip压缩
    fn compress_gzip(&self, data: &[u8]) -> Result<Vec<u8>> {
        use flate2::write::GzEncoder;
        use flate2::Compression;

        let mut encoder = GzEncoder::new(Vec::new(), Compression::default());
        encoder.write_all(data)
            .map_err(|e| crate::Error::compression(format!("Gzip压缩失败: {}", e)))?;
        encoder.finish()
            .map_err(|e| crate::Error::compression(format!("Gzip压缩失败: {}", e)))
    }

    fn decompress_gzip(&self, data: &[u8]) -> Result<Vec<u8>> {
        use flate2::read::GzDecoder;

        let mut decoder = GzDecoder::new(data);
        let mut decompressed = Vec::new();
        decoder.read_to_end(&mut decompressed)
            .map_err(|e| crate::Error::compression(format!("Gzip解压缩失败: {}", e)))?;
        Ok(decompressed)
    }
}

/// 快速压缩函数（使用LZ4）
pub fn compress(data: &[u8]) -> Result<Vec<u8>> {
    Compressor::default().compress(data)
}

/// 快速解压缩函数（使用LZ4）
pub fn decompress(data: &[u8]) -> Result<Vec<u8>> {
    Compressor::default().decompress(data)
}

/// 估算压缩比
pub fn estimate_compression_ratio(data: &[u8], algorithm: CompressionAlgorithm) -> Result<f64> {
    let compressor = Compressor::new(algorithm);
    let compressed = compressor.compress(data)?;
    Ok(compressed.len() as f64 / data.len() as f64)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn generate_test_data(size: usize) -> Vec<u8> {
        // 生成可压缩的测试数据
        let mut data = Vec::with_capacity(size);
        for i in 0..size {
            data.push((i % 256) as u8);
        }
        data
    }

    #[test]
    fn test_lz4_compression() {
        let compressor = Compressor::new(CompressionAlgorithm::Lz4);
        let data = generate_test_data(1000);

        let compressed = compressor.compress(&data).unwrap();
        let decompressed = compressor.decompress(&compressed).unwrap();

        assert_eq!(data, decompressed);
        assert!(compressed.len() < data.len()); // 应该被压缩
    }

    #[test]
    fn test_zstd_compression() {
        let compressor = Compressor::new(CompressionAlgorithm::Zstd);
        let data = generate_test_data(1000);

        let compressed = compressor.compress(&data).unwrap();
        let decompressed = compressor.decompress(&compressed).unwrap();

        assert_eq!(data, decompressed);
    }

    #[test]
    fn test_gzip_compression() {
        let compressor = Compressor::new(CompressionAlgorithm::Gzip);
        let data = generate_test_data(1000);

        let compressed = compressor.compress(&data).unwrap();
        let decompressed = compressor.decompress(&compressed).unwrap();

        assert_eq!(data, decompressed);
    }

    #[test]
    fn test_no_compression() {
        let compressor = Compressor::new(CompressionAlgorithm::None);
        let data = generate_test_data(1000);

        let compressed = compressor.compress(&data).unwrap();
        let decompressed = compressor.decompress(&compressed).unwrap();

        assert_eq!(data, compressed);
        assert_eq!(data, decompressed);
    }

    #[test]
    fn test_quick_compress() {
        let data = generate_test_data(1000);

        let compressed = compress(&data).unwrap();
        let decompressed = decompress(&compressed).unwrap();

        assert_eq!(data, decompressed);
    }

    #[test]
    fn test_compression_ratio() {
        let data = generate_test_data(1000);

        let ratio_lz4 = estimate_compression_ratio(&data, CompressionAlgorithm::Lz4).unwrap();
        let ratio_zstd = estimate_compression_ratio(&data, CompressionAlgorithm::Zstd).unwrap();

        assert!(ratio_lz4 > 0.0 && ratio_lz4 < 1.0);
        assert!(ratio_zstd > 0.0 && ratio_zstd < 1.0);
    }
}



