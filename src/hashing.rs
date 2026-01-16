//! 哈希算法模块
//! 
//! 提供多种哈希算法实现，包括：
//! - 基础哈希算法（MD5、SHA系列、BLAKE3等）
//! - 一致性哈希（用于分布式系统）
//! - 安全哈希（密码学哈希、盐值哈希）
//! - 布隆过滤器哈希
//! - 向量哈希（AI数据处理）
//! - 模型哈希工具

use crate::error::{Error, Result};
use sha2::{Sha256, Sha512, Digest};
use sha1::Sha1;
use blake3::Hasher as Blake3Hasher;
use xxhash_rust::xxh3::{xxh3_64, xxh3_64_with_seed};
// highway 暂时禁用，因为版本兼容性问题
// use highway::{HighwayHasher, Key as HighwayKey};
use fnv::FnvHasher;
use serde::{Serialize, Deserialize};
use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use rand::{RngCore, SeedableRng, rngs::OsRng};
use hmac::{Hmac, Mac};
use pbkdf2::pbkdf2_hmac;

type HmacSha256 = Hmac<Sha256>;

/// 支持的哈希算法类型
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum HashAlgorithm {
    /// MD5（不推荐用于安全用途）
    MD5,
    /// SHA-1（不推荐用于安全用途）
    SHA1,
    /// SHA-256
    SHA256,
    /// SHA-512
    SHA512,
    /// BLAKE3（推荐）
    BLAKE3,
    /// xxHash（快速非加密哈希）
    XxHash,
    /// CityHash（快速非加密哈希）
    CityHash,
}

impl Default for HashAlgorithm {
    fn default() -> Self {
        HashAlgorithm::BLAKE3
    }
}

/// 哈希结果
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct HashResult {
    /// 算法类型
    pub algorithm: HashAlgorithm,
    /// 哈希值（十六进制字符串）
    pub hash: String,
    /// 原始字节
    pub bytes: Vec<u8>,
    /// 哈希时长（纳秒）
    pub duration_ns: u64,
}

impl HashResult {
    /// 创建新的哈希结果
    pub fn new(algorithm: HashAlgorithm, bytes: Vec<u8>, duration_ns: u64) -> Self {
        let hash = hex::encode(&bytes);
        Self {
            algorithm,
            hash,
            bytes,
            duration_ns,
        }
    }
    
    /// 获取哈希的前缀
    pub fn prefix(&self, len: usize) -> String {
        if len >= self.hash.len() {
            self.hash.clone()
        } else {
            self.hash[..len].to_string()
        }
    }
    
    /// 验证哈希值
    pub fn verify(&self, input: &[u8]) -> Result<bool> {
        let computed = GlobalHasher::hash(self.algorithm, input)?;
        Ok(computed == self.bytes)
    }
}

/// 全局哈希器
pub struct GlobalHasher {
    /// 默认算法
    default_algorithm: HashAlgorithm,
    /// 性能统计
    stats: HashMap<HashAlgorithm, (u64, u64)>, // (count, total_time)
}

impl Default for GlobalHasher {
    fn default() -> Self {
        Self::new(HashAlgorithm::BLAKE3)
    }
}

impl GlobalHasher {
    /// 创建新的哈希器
    pub fn new(default_algorithm: HashAlgorithm) -> Self {
        Self {
            default_algorithm,
            stats: HashMap::new(),
        }
    }
    
    /// 使用默认算法计算哈希
    pub fn hash_default(&mut self, input: &[u8]) -> Result<HashResult> {
        self.hash_with_algorithm(self.default_algorithm, input)
    }
    
    /// 使用指定算法计算哈希
    pub fn hash_with_algorithm(&mut self, algorithm: HashAlgorithm, input: &[u8]) -> Result<HashResult> {
        let start = std::time::Instant::now();
        let result = Self::hash(algorithm, input)?;
        let duration = start.elapsed().as_nanos() as u64;
        
        // 更新统计信息
        let (count, total_time) = self.stats.entry(algorithm).or_insert((0, 0));
        *count += 1;
        *total_time += duration;
        
        Ok(HashResult::new(algorithm, result, duration))
    }
    
    /// 静态哈希方法 - 生产级实现
    pub fn hash(algorithm: HashAlgorithm, input: &[u8]) -> Result<Vec<u8>> {
        match algorithm {
            HashAlgorithm::MD5 => {
                // MD5哈希计算：使用md5库进行哈希计算
                let digest = md5::compute(input);
                Ok(digest.0.to_vec())
            },
            HashAlgorithm::SHA1 => {
                // 生产级SHA1实现
                let mut hasher = Sha1::new();
                hasher.update(input);
                Ok(hasher.finalize().to_vec())
            },
            HashAlgorithm::SHA256 => {
                let mut hasher = Sha256::new();
                hasher.update(input);
                Ok(hasher.finalize().to_vec())
            },
            HashAlgorithm::SHA512 => {
                let mut hasher = Sha512::new();
                hasher.update(input);
                Ok(hasher.finalize().to_vec())
            },
            HashAlgorithm::BLAKE3 => {
                // 生产级BLAKE3实现
                let mut hasher = Blake3Hasher::new();
                hasher.update(input);
                Ok(hasher.finalize().as_bytes().to_vec())
            },
            HashAlgorithm::XxHash => {
                // 生产级xxHash实现
                let hash = xxh3_64(input);
                Ok(hash.to_le_bytes().to_vec())
            },
            HashAlgorithm::CityHash => {
                // 使用HighwayHash作为CityHash的替代（Google开发的现代化哈希）
                // 暂时使用xxhash作为替代，因为highway crate有版本兼容性问题
                let hash = xxhash_rust::xxh3::xxh3_64(input);
                Ok(hash.to_le_bytes().to_vec())
            },
        }
    }
    
    /// 批量哈希
    pub fn hash_batch(&mut self, inputs: &[&[u8]]) -> Result<Vec<HashResult>> {
        inputs.iter()
            .map(|input| self.hash_default(input))
            .collect()
    }
    
    /// 获取性能统计
    pub fn get_stats(&self) -> &HashMap<HashAlgorithm, (u64, u64)> {
        &self.stats
    }
    
    /// 重置统计
    pub fn reset_stats(&mut self) {
        self.stats.clear();
    }
}

/// 一致性哈希环
pub struct ConsistentHashRing<T> {
    /// 环上的节点
    ring: std::collections::BTreeMap<u64, T>,
    /// 虚拟节点数
    virtual_nodes: u32,
    /// 哈希算法
    algorithm: HashAlgorithm,
}

impl<T: Clone + Hash + Eq> ConsistentHashRing<T> {
    /// 创建新的一致性哈希环
    pub fn new(virtual_nodes: u32) -> Self {
        Self {
            ring: std::collections::BTreeMap::new(),
            virtual_nodes,
            algorithm: HashAlgorithm::BLAKE3,
        }
    }
    
    /// 添加节点
    pub fn add_node(&mut self, node: T) -> Result<()> {
        for i in 0..self.virtual_nodes {
            let mut hasher = std::collections::hash_map::DefaultHasher::new();
            node.hash(&mut hasher);
            i.hash(&mut hasher);
            let hash = hasher.finish();
            self.ring.insert(hash, node.clone());
        }
        Ok(())
    }
    
    /// 移除节点
    pub fn remove_node(&mut self, node: &T) -> Result<()> 
    where
        T: PartialEq
    {
        let mut keys_to_remove = Vec::new();
        for (key, value) in &self.ring {
            if value == node {
                keys_to_remove.push(*key);
            }
        }
        
        for key in keys_to_remove {
            self.ring.remove(&key);
        }
        
        Ok(())
    }
    
    /// 根据键获取对应的节点
    pub fn get_node(&self, key: &str) -> Result<Option<T>> {
        if self.ring.is_empty() {
            return Ok(None);
        }
        
        let hash_bytes = GlobalHasher::hash(self.algorithm, key.as_bytes())?;
        let hash = u64::from_le_bytes([
            hash_bytes[0], hash_bytes[1], hash_bytes[2], hash_bytes[3],
            hash_bytes[4], hash_bytes[5], hash_bytes[6], hash_bytes[7],
        ]);
        
        // 找到第一个大于等于hash的节点
        if let Some((_, node)) = self.ring.range(hash..).next() {
            Ok(Some(node.clone()))
        } else {
            // 如果没找到，返回环上的第一个节点
            Ok(self.ring.values().next().cloned())
        }
    }
    
    /// 获取多个节点（用于复制）
    pub fn get_nodes(&self, key: &str, count: usize) -> Result<Vec<T>> {
        if self.ring.is_empty() || count == 0 {
            return Ok(Vec::new());
        }
        
        let hash_bytes = GlobalHasher::hash(self.algorithm, key.as_bytes())?;
        let hash = u64::from_le_bytes([
            hash_bytes[0], hash_bytes[1], hash_bytes[2], hash_bytes[3],
            hash_bytes[4], hash_bytes[5], hash_bytes[6], hash_bytes[7],
        ]);
        
        let mut result = Vec::with_capacity(count);
        let mut seen = std::collections::HashSet::new();
        
        // 从指定位置开始遍历
        let mut iter = self.ring.range(hash..).chain(self.ring.iter());
        
        for (_, node) in iter {
            if !seen.contains(node) {
                seen.insert(node.clone());
                result.push(node.clone());
                if result.len() >= count {
                    break;
                }
            }
        }
        
        Ok(result)
    }
    
    /// 获取环的大小
    pub fn size(&self) -> usize {
        self.ring.len()
    }
    
    /// 获取实际节点数量（去重后）
    pub fn node_count(&self) -> usize {
        let mut unique_nodes = std::collections::HashSet::new();
        for node in self.ring.values() {
            unique_nodes.insert(node);
        }
        unique_nodes.len()
    }
}

/// 安全哈希器（用于密码等敏感数据）
pub struct SecureHasher {
    /// 随机数生成器（不需要存储状态）
    _marker: std::marker::PhantomData<()>,
}

impl Default for SecureHasher {
    fn default() -> Self {
        Self::new()
    }
}

impl SecureHasher {
    /// 创建新的安全哈希器
    pub fn new() -> Self {
        Self {
            _marker: std::marker::PhantomData,
        }
    }
    
    /// 哈希密码（使用PBKDF2）
    pub fn hash_password(&self, password: &str, iterations: Option<u32>) -> Result<PasswordHash> {
        let iterations = iterations.unwrap_or(10000);
        let salt = self.generate_salt(32)?;
        
        let mut output = [0u8; 32];
        pbkdf2_hmac::<Sha256>(
            password.as_bytes(),
            &salt,
            iterations,
            &mut output
        );
        
        Ok(PasswordHash {
            algorithm: "PBKDF2-SHA256".to_string(),
            iterations,
            salt,
            hash: output.to_vec(),
        })
    }
    
    /// 验证密码
    pub fn verify_password(&self, password: &str, stored_hash: &PasswordHash) -> Result<bool> {
        let mut output = [0u8; 32];
        pbkdf2_hmac::<Sha256>(
            password.as_bytes(),
            &stored_hash.salt,
            stored_hash.iterations,
            &mut output
        );
        
        Ok(output.to_vec() == stored_hash.hash)
    }
    
    /// 生成随机盐值
    pub fn generate_salt(&self, length: usize) -> Result<Vec<u8>> {
        let mut salt = vec![0u8; length];
        OsRng.fill_bytes(&mut salt);
        Ok(salt)
    }
    
    /// HMAC-SHA256
    pub fn hmac_sha256(&self, key: &[u8], data: &[u8]) -> Result<Vec<u8>> {
        let mut mac = HmacSha256::new_from_slice(key)
            .map_err(|e| Error::InvalidInput(format!("HMAC key error: {}", e)))?;
        mac.update(data);
        Ok(mac.finalize().into_bytes().to_vec())
    }
}

/// 密码哈希结果
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PasswordHash {
    /// 算法名称
    pub algorithm: String,
    /// 迭代次数
    pub iterations: u32,
    /// 盐值
    pub salt: Vec<u8>,
    /// 哈希值
    pub hash: Vec<u8>,
}

impl PasswordHash {
    /// 转换为字符串格式
    pub fn to_string(&self) -> String {
        format!(
            "{}:{}:{}:{}",
            self.algorithm,
            self.iterations,
            hex::encode(&self.salt),
            hex::encode(&self.hash)
        )
    }
    
    /// 从字符串解析
    pub fn from_string(s: &str) -> Result<Self> {
        let parts: Vec<&str> = s.split(':').collect();
        if parts.len() != 4 {
            return Err(Error::invalid_input("Invalid password hash format".to_string()));
        }
        
        let salt = hex::decode(parts[2])
            .map_err(|_| Error::invalid_input("Invalid salt hex encoding".to_string()))?;
        let hash = hex::decode(parts[3])
            .map_err(|_| Error::invalid_input("Invalid hash hex encoding".to_string()))?;
        let iterations = parts[1].parse::<u32>()
            .map_err(|_| Error::invalid_input("Invalid iterations".to_string()))?;
        
        Ok(Self {
            algorithm: parts[0].to_string(),
            iterations,
            salt,
            hash,
        })
    }
}

/// 布隆过滤器
pub struct BloomFilter {
    /// 位数组
    bits: Vec<bool>,
    /// 哈希函数数量
    hash_count: u32,
    /// 预期元素数量
    expected_elements: u64,
    /// 误判率
    false_positive_rate: f64,
}

impl BloomFilter {
    /// 创建新的布隆过滤器
    pub fn new(expected_elements: u64, false_positive_rate: f64) -> Self {
        let bit_count = Self::optimal_bit_count(expected_elements, false_positive_rate);
        let hash_count = Self::optimal_hash_count(bit_count, expected_elements);
        
        Self {
            bits: vec![false; bit_count as usize],
            hash_count,
            expected_elements,
            false_positive_rate,
        }
    }
    
    /// 计算最优位数组大小
    fn optimal_bit_count(expected_elements: u64, false_positive_rate: f64) -> u64 {
        let ln2 = std::f64::consts::LN_2;
        (-(expected_elements as f64 * false_positive_rate.ln()) / (ln2 * ln2)).ceil() as u64
    }
    
    /// 计算最优哈希函数数量
    fn optimal_hash_count(bit_count: u64, expected_elements: u64) -> u32 {
        ((bit_count as f64 / expected_elements as f64) * std::f64::consts::LN_2).ceil() as u32
    }
    
    /// 添加元素
    pub fn add(&mut self, item: &[u8]) -> Result<()> {
        for i in 0..self.hash_count {
            let hash = self.hash_with_seed(item, i)?;
            let index = (hash % self.bits.len() as u64) as usize;
            self.bits[index] = true;
        }
        Ok(())
    }
    
    /// 检查元素是否可能存在
    pub fn might_contain(&self, item: &[u8]) -> Result<bool> {
        for i in 0..self.hash_count {
            let hash = self.hash_with_seed(item, i)?;
            let index = (hash % self.bits.len() as u64) as usize;
            if !self.bits[index] {
                return Ok(false);
            }
        }
        Ok(true)
    }
    
    /// 使用种子计算哈希
    fn hash_with_seed(&self, item: &[u8], seed: u32) -> Result<u64> {
        // 使用xxHash with seed
        Ok(xxh3_64_with_seed(item, seed as u64))
    }
    
    /// 计算当前误判率
    pub fn current_false_positive_rate(&self) -> f64 {
        let set_bits = self.bits.iter().filter(|&&bit| bit).count() as f64;
        let total_bits = self.bits.len() as f64;
        let ratio = set_bits / total_bits;
        ratio.powf(self.hash_count as f64)
    }
    
    /// 清空过滤器
    pub fn clear(&mut self) {
        for bit in &mut self.bits {
            *bit = false;
        }
    }
    
    /// 获取统计信息
    pub fn stats(&self) -> BloomFilterStats {
        let set_bits = self.bits.iter().filter(|&&bit| bit).count();
        BloomFilterStats {
            total_bits: self.bits.len(),
            set_bits,
            hash_count: self.hash_count,
            expected_elements: self.expected_elements,
            target_false_positive_rate: self.false_positive_rate,
            current_false_positive_rate: self.current_false_positive_rate(),
        }
    }
}

/// 布隆过滤器统计信息
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BloomFilterStats {
    pub total_bits: usize,
    pub set_bits: usize,
    pub hash_count: u32,
    pub expected_elements: u64,
    pub target_false_positive_rate: f64,
    pub current_false_positive_rate: f64,
}

/// 向量哈希器（用于AI向量数据）
pub struct VectorHasher {
    /// 量化精度
    precision: f32,
    /// 哈希算法
    algorithm: HashAlgorithm,
}

impl VectorHasher {
    /// 创建新的向量哈希器
    pub fn new(precision: f32, algorithm: HashAlgorithm) -> Self {
        Self {
            precision,
            algorithm,
        }
    }
    
    /// 哈希向量
    pub fn hash_vector(&self, vector: &[f32]) -> Result<HashResult> {
        // 量化向量
        let quantized: Vec<i32> = vector.iter()
            .map(|&x| (x / self.precision).round() as i32)
            .collect();
        
        // 序列化量化后的向量
        let bytes = bincode::serialize(&quantized)
            .map_err(|e| Error::Serialization(format!("Vector serialization error: {}", e)))?;
        
        // 计算哈希
        let hash_bytes = GlobalHasher::hash(self.algorithm, &bytes)?;
        Ok(HashResult::new(self.algorithm, hash_bytes, 0))
    }
    
    /// 局部敏感哈希(LSH)
    pub fn lsh_hash(&self, vector: &[f32], num_hashes: usize) -> Result<Vec<HashResult>> {
        let mut results = Vec::new();
        
        for i in 0..num_hashes {
            // 生成随机投影向量
            let projection = self.generate_random_projection(vector.len(), i as u64)?;
            
            // 计算点积
            let dot_product: f32 = vector.iter()
                .zip(projection.iter())
                .map(|(a, b)| a * b)
                .sum();
            
            // 二值化
            let bit = if dot_product >= 0.0 { 1u8 } else { 0u8 };
            
            // 计算哈希
            let hash_bytes = GlobalHasher::hash(self.algorithm, &[bit])?;
            results.push(HashResult::new(self.algorithm, hash_bytes, 0));
        }
        
        Ok(results)
    }
    
    /// 生成随机投影向量
    fn generate_random_projection(&self, dimension: usize, seed: u64) -> Result<Vec<f32>> {
        use rand::prelude::*;
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
        let projection: Vec<f32> = (0..dimension)
            .map(|_| rng.gen::<f32>() * 2.0 - 1.0)
            .collect();
        Ok(projection)
    }
}

/// 生产级FNV哈希实现（使用专用库）
fn fnv_hash(data: &[u8]) -> u64 {
    let mut hasher = FnvHasher::default();
    hasher.write(data);
    hasher.finish()
}

/// 哈希工具函数
pub mod utils {
    use super::*;
    
    /// 快速字符串哈希
    pub fn quick_hash(s: &str) -> u64 {
        xxh3_64(s.as_bytes())
    }
    
    /// 文件哈希
    pub fn hash_file(path: &std::path::Path, algorithm: HashAlgorithm) -> Result<HashResult> {
        use std::io::Read;
        
        let mut file = std::fs::File::open(path)
            .map_err(|e| Error::Io(e))?;
        let mut buffer = Vec::new();
        file.read_to_end(&mut buffer)
            .map_err(|e| Error::Io(e))?;
        
        let start = std::time::Instant::now();
        let hash_bytes = GlobalHasher::hash(algorithm, &buffer)?;
        let duration = start.elapsed().as_nanos() as u64;
        
        Ok(HashResult::new(algorithm, hash_bytes, duration))
    }
    
    /// 增量哈希器
    pub struct IncrementalHasher {
        algorithm: HashAlgorithm,
        state: Box<dyn std::any::Any + Send>,
    }
    
    impl IncrementalHasher {
        pub fn new(algorithm: HashAlgorithm) -> Result<Self> {
            let state: Box<dyn std::any::Any + Send> = match algorithm {
                HashAlgorithm::MD5 => Box::new(md5::Context::new()),
                HashAlgorithm::SHA1 => Box::new(sha1::Sha1::new()),
                HashAlgorithm::SHA256 => Box::new(sha2::Sha256::new()),
                HashAlgorithm::SHA512 => Box::new(sha2::Sha512::new()),
                HashAlgorithm::BLAKE3 => Box::new(blake3::Hasher::new()),
                _ => return Err(Error::NotImplemented("Incremental hashing not supported for this algorithm".to_string())),
            };
            
            Ok(Self { algorithm, state })
        }
        
        pub fn update(&mut self, data: &[u8]) -> Result<()> {
            match self.algorithm {
                HashAlgorithm::MD5 => {
                    let hasher = self.state.downcast_mut::<md5::Context>()
                        .ok_or_else(|| Error::InvalidState("Invalid hasher state".to_string()))?;
                    hasher.consume(data);
                },
                HashAlgorithm::SHA1 => {
                    let hasher = self.state.downcast_mut::<sha1::Sha1>()
                        .ok_or_else(|| Error::InvalidState("Invalid hasher state".to_string()))?;
                    use sha1::Digest;
                    hasher.update(data);
                },
                HashAlgorithm::SHA256 => {
                    let hasher = self.state.downcast_mut::<sha2::Sha256>()
                        .ok_or_else(|| Error::InvalidState("Invalid hasher state".to_string()))?;
                    use sha2::Digest;
                    hasher.update(data);
                },
                HashAlgorithm::SHA512 => {
                    let hasher = self.state.downcast_mut::<sha2::Sha512>()
                        .ok_or_else(|| Error::InvalidState("Invalid hasher state".to_string()))?;
                    use sha2::Digest;
                    hasher.update(data);
                },
                HashAlgorithm::BLAKE3 => {
                    let hasher = self.state.downcast_mut::<blake3::Hasher>()
                        .ok_or_else(|| Error::invalid_state("Invalid hasher state".to_string()))?;
                    hasher.update(data);
                },
                _ => return Err(Error::not_implemented("Incremental hashing not supported for this algorithm".to_string())),
            }
            Ok(())
        }
        
        pub fn finalize(self) -> Result<HashResult> {
            let bytes = match self.algorithm {
                HashAlgorithm::MD5 => {
                    let hasher = *self.state.downcast::<md5::Context>()
                        .map_err(|_| Error::invalid_state("Invalid hasher state".to_string()))?;
                    hasher.compute().to_vec()
                },
                HashAlgorithm::SHA1 => {
                    let hasher = *self.state.downcast::<sha1::Sha1>()
                        .map_err(|_| Error::invalid_state("Invalid hasher state".to_string()))?;
                    use sha1::Digest;
                    hasher.finalize().to_vec()
                },
                HashAlgorithm::SHA256 => {
                    let hasher = *self.state.downcast::<sha2::Sha256>()
                        .map_err(|_| Error::invalid_state("Invalid hasher state".to_string()))?;
                    use sha2::Digest;
                    hasher.finalize().to_vec()
                },
                HashAlgorithm::SHA512 => {
                    let hasher = *self.state.downcast::<sha2::Sha512>()
                        .map_err(|_| Error::invalid_state("Invalid hasher state".to_string()))?;
                    use sha2::Digest;
                    hasher.finalize().to_vec()
                },
                HashAlgorithm::BLAKE3 => {
                    let hasher = *self.state.downcast::<blake3::Hasher>()
                        .map_err(|_| Error::invalid_state("Invalid hasher state".to_string()))?;
                    hasher.finalize().as_bytes().to_vec()
                },
                _ => return Err(Error::not_implemented("Incremental hashing not supported for this algorithm".to_string())),
            };
            
            Ok(HashResult::new(self.algorithm, bytes, 0))
        }
    }
}

/// 模型哈希工具
pub struct ModelHasher;

impl ModelHasher {
    /// 生成模型哈希
    pub fn generate_model_hash(
        model_id: &str, 
        version: Option<&str>, 
        metadata: Option<&HashMap<String, String>>
    ) -> String {
        let mut data = model_id.to_string();
        
        if let Some(v) = version {
            data.push_str(&format!(":v{}", v));
        }
        
        if let Some(meta) = metadata {
            let mut sorted_meta: Vec<_> = meta.iter().collect();
            sorted_meta.sort_by_key(|(k, _)| *k);
            
            for (key, value) in sorted_meta {
                data.push_str(&format!(":{}={}", key, value));
            }
        }
        
        Self::sha256_hash(data.as_bytes())
    }

    /// SHA256哈希
    pub fn sha256_hash(data: &[u8]) -> String {
        let hash_bytes = GlobalHasher::hash(HashAlgorithm::SHA256, data).unwrap();
        hex::encode(hash_bytes)
    }

    /// 快速哈希
    pub fn quick_hash(s: &str) -> u64 {
        utils::quick_hash(s)
    }

    /// 哈希任意值
    pub fn hash_value<T: Hash>(t: &T) -> u64 {
        use std::collections::hash_map::DefaultHasher;
        let mut hasher = DefaultHasher::new();
        t.hash(&mut hasher);
        hasher.finish()
    }

    /// 计算汉明距离
    pub fn hamming_distance(hash1: &[bool], hash2: &[bool]) -> Result<usize> {
        if hash1.len() != hash2.len() {
            return Err(Error::invalid_input("Hash lengths must match".to_string()));
        }
        
        Ok(hash1.iter()
            .zip(hash2.iter())
            .filter(|(a, b)| a != b)
            .count())
    }

    /// 检查哈希相似性
    pub fn is_similar_hash(hash1: &[bool], hash2: &[bool], threshold: f32) -> Result<bool> {
        let distance = Self::hamming_distance(hash1, hash2)? as f32;
        let similarity = 1.0 - (distance / hash1.len() as f32);
        Ok(similarity >= threshold)
    }

    /// MurmurHash3 32位实现
    pub fn murmur3_32(data: &[u8], seed: u32) -> u32 {
        const C1: u32 = 0xcc9e2d51;
        const C2: u32 = 0x1b873593;
        const R1: u32 = 15;
        const R2: u32 = 13;
        const M: u32 = 5;
        const N: u32 = 0xe6546b64;

        let mut hash = seed;
        let chunk_size = std::mem::size_of::<u32>();
        let chunks = data.len() / chunk_size;

        for i in 0..chunks {
            let start = i * chunk_size;
            let mut k = u32::from_le_bytes([
                data[start],
                data[start + 1],
                data[start + 2],
                data[start + 3],
            ]);

            k = k.wrapping_mul(C1);
            k = k.rotate_left(R1);
            k = k.wrapping_mul(C2);

            hash ^= k;
            hash = hash.rotate_left(R2);
            hash = hash.wrapping_mul(M).wrapping_add(N);
        }

        // 处理剩余字节
        let remainder = data.len() % chunk_size;
        if remainder > 0 {
            let start = chunks * chunk_size;
            let mut k = 0u32;
            for i in 0..remainder {
                k |= (data[start + i] as u32) << (8 * i);
            }
            k = k.wrapping_mul(C1);
            k = k.rotate_left(R1);
            k = k.wrapping_mul(C2);
            hash ^= k;
        }

        hash ^= data.len() as u32;
        hash ^= hash >> 16;
        hash = hash.wrapping_mul(0x85ebca6b);
        hash ^= hash >> 13;
        hash = hash.wrapping_mul(0xc2b2ae35);
        hash ^= hash >> 16;

        hash
    }

    /// 生成哈希种子
    pub fn generate_hash_seed() -> u64 {
        use rand::RngCore;
        let mut rng = OsRng;
        rng.next_u64()
    }
}

/// 哈希工厂
pub struct HashFactory;

impl HashFactory {
    /// SHA256哈希
    pub fn sha256(data: &[u8]) -> String {
        ModelHasher::sha256_hash(data)
    }

    /// 快速字符串哈希
    pub fn quick_string(s: &str) -> u64 {
        utils::quick_hash(s)
    }

    /// 模型哈希
    pub fn model(model_id: &str, version: Option<&str>) -> String {
        ModelHasher::generate_model_hash(model_id, version, None)
    }

    /// 带元数据的模型哈希
    pub fn model_with_metadata(
        model_id: &str, 
        version: Option<&str>, 
        metadata: &HashMap<String, String>
    ) -> String {
        ModelHasher::generate_model_hash(model_id, version, Some(metadata))
    }

    /// 一致性哈希
    pub fn consistent(key: &str, node_count: usize) -> usize {
        if node_count == 0 {
            return 0;
        }
        (utils::quick_hash(key) as usize) % node_count
    }

    /// LSH向量哈希
    pub fn lsh_vector(vector: &[f32], hash_functions: &[Vec<f32>]) -> Result<Vec<bool>> {
        let mut result = Vec::new();
        
        for hash_func in hash_functions {
            if hash_func.len() != vector.len() {
                return Err(Error::invalid_input("Hash function dimension mismatch".to_string()));
            }
            
            let dot_product: f32 = vector.iter()
                .zip(hash_func.iter())
                .map(|(a, b)| a * b)
                .sum();
            
            result.push(dot_product >= 0.0);
        }
        
        Ok(result)
    }

    /// 生成LSH函数
    pub fn generate_lsh_functions(dimension: usize, hash_count: usize) -> Vec<Vec<f32>> {
        use rand::prelude::*;
        let mut rng = rand::thread_rng();
        
        (0..hash_count)
            .map(|_| {
                (0..dimension)
                    .map(|_| rng.gen::<f32>() * 2.0 - 1.0)
                    .collect()
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_hashing() {
        let data = b"hello world";
        
        // 测试所有算法
        for algorithm in [HashAlgorithm::MD5, HashAlgorithm::SHA1, HashAlgorithm::SHA256, 
                         HashAlgorithm::SHA512, HashAlgorithm::BLAKE3, HashAlgorithm::XxHash, 
                         HashAlgorithm::CityHash] {
            let result = GlobalHasher::hash(algorithm, data).unwrap();
            assert!(!result.bytes.is_empty());
        }
    }

    #[test]
    fn test_consistent_hash_ring() {
        let mut ring = ConsistentHashRing::new(100);
        ring.add_node("node1".to_string()).unwrap();
        ring.add_node("node2".to_string()).unwrap();
        ring.add_node("node3".to_string()).unwrap();
        
        let node = ring.get_node("test_key").unwrap();
        assert!(node.is_some());
    }

    #[test]
    fn test_password_hashing() {
        let hasher = SecureHasher::new();
        let password = "test_password";
        let hash = hasher.hash_password(password, Some(1000)).unwrap();
        
        assert!(hasher.verify_password(password, &hash).unwrap());
        assert!(!hasher.verify_password("wrong_password", &hash).unwrap());
    }

    #[test]
    fn test_bloom_filter() {
        let mut filter = BloomFilter::new(1000, 0.01);
        
        filter.add(b"item1").unwrap();
        filter.add(b"item2").unwrap();
        
        assert!(filter.might_contain(b"item1").unwrap());
        assert!(filter.might_contain(b"item2").unwrap());
        assert!(!filter.might_contain(b"item3").unwrap());
    }

    #[test]
    fn test_vector_hashing() {
        let hasher = VectorHasher::new(0.1, HashAlgorithm::BLAKE3);
        let vector = vec![1.0, 2.0, 3.0, 4.0];
        
        let result = hasher.hash_vector(&vector).unwrap();
        assert!(!result.bytes.is_empty());
        
        let lsh_results = hasher.lsh_hash(&vector, 5).unwrap();
        assert_eq!(lsh_results.len(), 5);
    }
}
