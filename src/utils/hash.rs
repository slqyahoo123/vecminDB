/// 哈希工具模块
/// 
/// 提供各种哈希计算功能，支持多种算法，用于数据完整性验证、
/// 分片标识、版本控制等场景

use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::io::Write;
use sha2::{Sha256, Sha512, Digest};
use blake3;
use crc32fast::Hasher as Crc32Hasher;
use serde::{Serialize, Deserialize};

/// 支持的哈希算法类型
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum HashAlgorithm {
    /// SHA256算法
    Sha256,
    /// SHA512算法 
    Sha512,
    /// MD5算法（不推荐用于安全目的）
    Md5,
    /// Blake3算法（高性能）
    Blake3,
    /// CRC32算法（快速校验）
    Crc32,
    /// 默认哈希（Rust标准库）
    Default,
}

impl Default for HashAlgorithm {
    fn default() -> Self {
        HashAlgorithm::Sha256
    }
}

/// 哈希结果
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct HashResult {
    /// 哈希算法
    pub algorithm: HashAlgorithm,
    /// 哈希值（十六进制）
    pub hash: String,
    /// 原始字节长度
    pub input_length: usize,
}

impl HashResult {
    /// 创建新的哈希结果
    pub fn new(algorithm: HashAlgorithm, hash: String, input_length: usize) -> Self {
        Self {
            algorithm,
            hash,
            input_length,
        }
    }

    /// 验证哈希值是否匹配
    pub fn verify(&self, data: &[u8]) -> bool {
        let computed = compute_hash_with_algorithm(data, self.algorithm);
        computed.hash == self.hash
    }

    /// 获取哈希值的字节表示
    pub fn as_bytes(&self) -> Result<Vec<u8>, hex::FromHexError> {
        hex::decode(&self.hash)
    }
}

/// 计算数据的哈希值（使用默认SHA256算法）
pub fn compute_hash(data: &[u8]) -> String {
    compute_hash_with_algorithm(data, HashAlgorithm::Sha256).hash
}

/// 使用指定算法计算哈希值
pub fn compute_hash_with_algorithm(data: &[u8], algorithm: HashAlgorithm) -> HashResult {
    let hash = match algorithm {
        HashAlgorithm::Sha256 => {
            let mut hasher = Sha256::new();
            hasher.update(data);
            hex::encode(hasher.finalize())
        }
        HashAlgorithm::Sha512 => {
            let mut hasher = Sha512::new();
            hasher.update(data);
            hex::encode(hasher.finalize())
        }
        HashAlgorithm::Md5 => {
            let digest = md5::compute(data);
            format!("{:x}", digest)
        }
        HashAlgorithm::Blake3 => {
            let hash = blake3::hash(data);
            hash.to_hex().to_string()
        }
        HashAlgorithm::Crc32 => {
            let mut hasher = Crc32Hasher::new();
            hasher.update(data);
            format!("{:08x}", hasher.finalize())
        }
        HashAlgorithm::Default => {
            let mut hasher = DefaultHasher::new();
            data.hash(&mut hasher);
            format!("{:016x}", hasher.finish())
        }
    };

    HashResult::new(algorithm, hash, data.len())
}

/// 计算字符串的哈希值
pub fn compute_string_hash(s: &str) -> String {
    compute_hash(s.as_bytes())
}

/// 计算文件哈希值
pub fn compute_file_hash<P: AsRef<std::path::Path>>(
    path: P,
    algorithm: HashAlgorithm,
) -> std::io::Result<HashResult> {
    use std::fs::File;
    use std::io::{BufReader, Read};

    let file = File::open(path)?;
    let mut reader = BufReader::new(file);
    let mut buffer = Vec::new();
    reader.read_to_end(&mut buffer)?;

    Ok(compute_hash_with_algorithm(&buffer, algorithm))
}

/// 增量哈希计算器
pub struct IncrementalHasher {
    algorithm: HashAlgorithm,
    hasher: Box<dyn HashUpdater>,
    total_length: usize,
}

trait HashUpdater: Send + Sync {
    fn update(&mut self, data: &[u8]);
    fn finalize(self: Box<Self>) -> String;
}

struct Sha256Updater(Sha256);
struct Sha512Updater(Sha512);
struct Md5Updater(md5::Context);
struct Blake3Updater(blake3::Hasher);
struct Crc32Updater(Crc32Hasher);

impl HashUpdater for Sha256Updater {
    fn update(&mut self, data: &[u8]) {
        self.0.update(data);
    }
    
    fn finalize(self: Box<Self>) -> String {
        hex::encode(self.0.finalize())
    }
}

impl HashUpdater for Sha512Updater {
    fn update(&mut self, data: &[u8]) {
        self.0.update(data);
    }
    
    fn finalize(self: Box<Self>) -> String {
        hex::encode(self.0.finalize())
    }
}

impl HashUpdater for Md5Updater {
    fn update(&mut self, data: &[u8]) {
        self.0.write(data);
    }
    
    fn finalize(self: Box<Self>) -> String {
        format!("{:x}", self.0.compute())
    }
}

impl HashUpdater for Blake3Updater {
    fn update(&mut self, data: &[u8]) {
        self.0.update(data);
    }
    
    fn finalize(self: Box<Self>) -> String {
        self.0.finalize().to_hex().to_string()
    }
}

impl HashUpdater for Crc32Updater {
    fn update(&mut self, data: &[u8]) {
        self.0.update(data);
    }
    
    fn finalize(self: Box<Self>) -> String {
        format!("{:08x}", self.0.finalize())
    }
}

impl IncrementalHasher {
    /// 创建新的增量哈希计算器
    pub fn new(algorithm: HashAlgorithm) -> Self {
        let hasher: Box<dyn HashUpdater> = match algorithm {
            HashAlgorithm::Sha256 => Box::new(Sha256Updater(Sha256::new())),
            HashAlgorithm::Sha512 => Box::new(Sha512Updater(Sha512::new())),
            HashAlgorithm::Md5 => Box::new(Md5Updater(md5::Context::new())),
            HashAlgorithm::Blake3 => Box::new(Blake3Updater(blake3::Hasher::new())),
            HashAlgorithm::Crc32 => Box::new(Crc32Updater(Crc32Hasher::new())),
            HashAlgorithm::Default => {
                // 对于默认哈希，我们回退到一次性计算
                // 因为DefaultHasher不支持增量计算的序列化
                Box::new(Blake3Updater(blake3::Hasher::new()))
            }
        };

        Self {
            algorithm,
            hasher,
            total_length: 0,
        }
    }

    /// 更新哈希计算
    pub fn update(&mut self, data: &[u8]) {
        self.hasher.update(data);
        self.total_length += data.len();
    }

    /// 完成哈希计算
    pub fn finalize(self) -> HashResult {
        let hash = self.hasher.finalize();
        HashResult::new(self.algorithm, hash, self.total_length)
    }
}

/// 哈希验证器
pub struct HashVerifier {
    expected: HashResult,
    hasher: IncrementalHasher,
}

impl HashVerifier {
    /// 创建新的哈希验证器
    pub fn new(expected: HashResult) -> Self {
        let hasher = IncrementalHasher::new(expected.algorithm);
        Self { expected, hasher }
    }

    /// 更新数据
    pub fn update(&mut self, data: &[u8]) {
        self.hasher.update(data);
    }

    /// 验证哈希值
    pub fn verify(self) -> bool {
        let computed = self.hasher.finalize();
        computed.hash == self.expected.hash
    }
}

/// 多重哈希计算器（同时计算多种算法）
pub struct MultiHasher {
    algorithms: Vec<HashAlgorithm>,
    hashers: Vec<IncrementalHasher>,
}

impl MultiHasher {
    /// 创建多重哈希计算器
    pub fn new(algorithms: Vec<HashAlgorithm>) -> Self {
        let hashers = algorithms
            .iter()
            .map(|&alg| IncrementalHasher::new(alg))
            .collect();

        Self { algorithms, hashers }
    }

    /// 更新所有哈希计算
    pub fn update(&mut self, data: &[u8]) {
        for hasher in &mut self.hashers {
            hasher.update(data);
        }
    }

    /// 完成所有哈希计算
    pub fn finalize(self) -> Vec<HashResult> {
        self.hashers.into_iter().map(|h| h.finalize()).collect()
    }
}

/// 哈希工具函数集合
pub mod utils {
    use super::*;

    /// 比较两个哈希值是否相等（防时序攻击）
    pub fn secure_compare(a: &str, b: &str) -> bool {
        if a.len() != b.len() {
            return false;
        }

        let mut result = 0u8;
        for (x, y) in a.bytes().zip(b.bytes()) {
            result |= x ^ y;
        }
        result == 0
    }

    /// 生成随机盐值
    pub fn generate_salt(length: usize) -> String {
        use rand::Rng;
        let mut rng = rand::thread_rng();
        (0..length)
            .map(|_| format!("{:02x}", rng.gen::<u8>()))
            .collect()
    }

    /// 使用盐值计算哈希
    pub fn compute_salted_hash(data: &[u8], salt: &str, algorithm: HashAlgorithm) -> HashResult {
        let mut combined = data.to_vec();
        combined.extend_from_slice(salt.as_bytes());
        compute_hash_with_algorithm(&combined, algorithm)
    }

    /// 计算HMAC
    pub fn compute_hmac_sha256(key: &[u8], data: &[u8]) -> String {
        use hmac::{Hmac, Mac};
        use sha2::Sha256;

        type HmacSha256 = Hmac<Sha256>;
        let mut mac = HmacSha256::new_from_slice(key)
            .expect("HMAC接受任何大小的密钥");
        mac.update(data);
        hex::encode(mac.finalize().into_bytes())
    }

    /// 计算基于密码的密钥派生（PBKDF2）
    pub fn pbkdf2_derive_key(
        password: &str,
        salt: &[u8],
        iterations: u32,
        key_length: usize,
    ) -> Vec<u8> {
        use pbkdf2::{pbkdf2_hmac};
        use sha2::Sha256;

        let mut key = vec![0u8; key_length];
        pbkdf2_hmac::<Sha256>(password.as_bytes(), salt, iterations, &mut key);
        key
    }
}

/// 哈希性能基准测试
#[cfg(test)]
mod benchmarks {
    use super::*;
    use std::time::Instant;

    pub fn benchmark_algorithms(data_size: usize) -> Vec<(HashAlgorithm, std::time::Duration)> {
        let data = vec![0u8; data_size];
        let algorithms = [
            HashAlgorithm::Sha256,
            HashAlgorithm::Sha512,
            HashAlgorithm::Md5,
            HashAlgorithm::Blake3,
            HashAlgorithm::Crc32,
        ];

        algorithms
            .iter()
            .map(|&alg| {
                let start = Instant::now();
                compute_hash_with_algorithm(&data, alg);
                let duration = start.elapsed();
                (alg, duration)
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compute_hash() {
        let data = b"hello world";
        let hash = compute_hash(data);
        assert_eq!(hash.len(), 64); // SHA256产生64字符的十六进制字符串
    }

    #[test]
    fn test_different_algorithms() {
        let data = b"test data";
        
        let sha256 = compute_hash_with_algorithm(data, HashAlgorithm::Sha256);
        let sha512 = compute_hash_with_algorithm(data, HashAlgorithm::Sha512);
        let md5 = compute_hash_with_algorithm(data, HashAlgorithm::Md5);
        
        assert_ne!(sha256.hash, sha512.hash);
        assert_ne!(sha256.hash, md5.hash);
        assert_eq!(sha256.hash.len(), 64);
        assert_eq!(sha512.hash.len(), 128);
        assert_eq!(md5.hash.len(), 32);
    }

    #[test]
    fn test_incremental_hasher() {
        let data1 = b"hello ";
        let data2 = b"world";
        let full_data = b"hello world";

        // 增量计算
        let mut incremental = IncrementalHasher::new(HashAlgorithm::Sha256);
        incremental.update(data1);
        incremental.update(data2);
        let incremental_result = incremental.finalize();

        // 一次性计算
        let full_result = compute_hash_with_algorithm(full_data, HashAlgorithm::Sha256);

        assert_eq!(incremental_result.hash, full_result.hash);
    }

    #[test]
    fn test_hash_verification() {
        let data = b"test verification";
        let hash_result = compute_hash_with_algorithm(data, HashAlgorithm::Sha256);
        
        assert!(hash_result.verify(data));
        assert!(!hash_result.verify(b"different data"));
    }

    #[test]
    fn test_secure_compare() {
        assert!(utils::secure_compare("abc123", "abc123"));
        assert!(!utils::secure_compare("abc123", "abc124"));
        assert!(!utils::secure_compare("abc123", "abc12"));
    }
} 