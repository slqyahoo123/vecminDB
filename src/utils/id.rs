//! ID生成工具模块
//! 
//! 提供各种ID生成策略，包括UUID、雪花算法、自增ID等。

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Mutex;
use std::time::{SystemTime, UNIX_EPOCH};


/// ID生成错误
#[derive(Debug, thiserror::Error)]
pub enum IdError {
    #[error("时钟回拨错误")]
    ClockBackward,
    #[error("节点ID超出范围: {0}")]
    InvalidNodeId(u64),
    #[error("序列号溢出")]
    SequenceOverflow,
}

/// ID生成结果
pub type IdResult<T> = std::result::Result<T, IdError>;

/// UUID生成器
pub struct UuidGenerator;

impl UuidGenerator {
    /// 生成UUID v4
    pub fn generate_v4() -> String {
        
        
        let mut rng = SimpleRng::new();
        let mut bytes = [0u8; 16];
        
        for byte in &mut bytes {
            *byte = rng.next_u8();
        }
        
        // 设置版本位（版本4）
        bytes[6] = (bytes[6] & 0x0f) | 0x40;
        // 设置变体位
        bytes[8] = (bytes[8] & 0x3f) | 0x80;
        
        format!(
            "{:02x}{:02x}{:02x}{:02x}-{:02x}{:02x}-{:02x}{:02x}-{:02x}{:02x}-{:02x}{:02x}{:02x}{:02x}{:02x}{:02x}",
            bytes[0], bytes[1], bytes[2], bytes[3],
            bytes[4], bytes[5],
            bytes[6], bytes[7],
            bytes[8], bytes[9],
            bytes[10], bytes[11], bytes[12], bytes[13], bytes[14], bytes[15]
        )
    }

    /// 生成短UUID（去掉连字符）
    pub fn generate_short() -> String {
        Self::generate_v4().replace("-", "")
    }

    /// 生成紧凑UUID（只包含字母数字）
    pub fn generate_compact() -> String {
        let full_uuid = Self::generate_v4();
        full_uuid.chars().filter(|c| c.is_alphanumeric()).collect()
    }
}

/// 雪花算法ID生成器
pub struct SnowflakeGenerator {
    node_id: u64,
    sequence: AtomicU64,
    last_timestamp: Mutex<u64>,
}

impl SnowflakeGenerator {
    // 雪花算法常量
    const EPOCH: u64 = 1609459200000; // 2021-01-01 00:00:00 UTC
    const NODE_ID_BITS: u64 = 10;
    const SEQUENCE_BITS: u64 = 12;
    const MAX_NODE_ID: u64 = (1 << Self::NODE_ID_BITS) - 1;
    const MAX_SEQUENCE: u64 = (1 << Self::SEQUENCE_BITS) - 1;
    const NODE_ID_SHIFT: u64 = Self::SEQUENCE_BITS;
    const TIMESTAMP_SHIFT: u64 = Self::SEQUENCE_BITS + Self::NODE_ID_BITS;

    /// 创建新的雪花算法生成器
    pub fn new(node_id: u64) -> IdResult<Self> {
        if node_id > Self::MAX_NODE_ID {
            return Err(IdError::InvalidNodeId(node_id));
        }

        Ok(Self {
            node_id,
            sequence: AtomicU64::new(0),
            last_timestamp: Mutex::new(0),
        })
    }

    /// 生成下一个ID
    pub fn next_id(&self) -> IdResult<u64> {
        let mut last_timestamp = self.last_timestamp.lock().unwrap();
        let mut timestamp = self.current_timestamp();

        if timestamp < *last_timestamp {
            return Err(IdError::ClockBackward);
        }

        if timestamp == *last_timestamp {
            let sequence = self.sequence.fetch_add(1, Ordering::SeqCst);
            if sequence > Self::MAX_SEQUENCE {
                // 等待下一毫秒
                while timestamp <= *last_timestamp {
                    timestamp = self.current_timestamp();
                }
                self.sequence.store(0, Ordering::SeqCst);
            }
        } else {
            self.sequence.store(0, Ordering::SeqCst);
        }

        *last_timestamp = timestamp;
        let sequence = self.sequence.load(Ordering::SeqCst);

        let id = ((timestamp - Self::EPOCH) << Self::TIMESTAMP_SHIFT)
            | (self.node_id << Self::NODE_ID_SHIFT)
            | sequence;

        Ok(id)
    }

    /// 获取当前时间戳（毫秒）
    fn current_timestamp(&self) -> u64 {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64
    }

    /// 解析雪花ID
    pub fn parse_id(&self, id: u64) -> (u64, u64, u64) {
        let timestamp = (id >> Self::TIMESTAMP_SHIFT) + Self::EPOCH;
        let node_id = (id >> Self::NODE_ID_SHIFT) & Self::MAX_NODE_ID;
        let sequence = id & Self::MAX_SEQUENCE;
        
        (timestamp, node_id, sequence)
    }
}

/// 自增ID生成器
pub struct IncrementalIdGenerator {
    current: AtomicU64,
    step: u64,
}

impl IncrementalIdGenerator {
    /// 创建新的自增ID生成器
    pub fn new(start: u64, step: u64) -> Self {
        Self {
            current: AtomicU64::new(start),
            step,
        }
    }

    /// 生成下一个ID
    pub fn next_id(&self) -> u64 {
        self.current.fetch_add(self.step, Ordering::SeqCst)
    }

    /// 获取当前ID值
    pub fn current_id(&self) -> u64 {
        self.current.load(Ordering::SeqCst)
    }

    /// 重置ID生成器
    pub fn reset(&self, value: u64) {
        self.current.store(value, Ordering::SeqCst);
    }
}

/// 纳米ID生成器
pub struct NanoIdGenerator {
    alphabet: Vec<char>,
    size: usize,
}

impl NanoIdGenerator {
    /// 默认字母表
    const DEFAULT_ALPHABET: &'static str = "_-0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ";
    
    /// 创建默认的NanoID生成器
    pub fn new() -> Self {
        Self {
            alphabet: Self::DEFAULT_ALPHABET.chars().collect(),
            size: 21,
        }
    }

    /// 创建自定义NanoID生成器
    pub fn with_alphabet(alphabet: &str, size: usize) -> Self {
        Self {
            alphabet: alphabet.chars().collect(),
            size,
        }
    }

    /// 生成NanoID
    pub fn generate(&self) -> String {
        let mut rng = SimpleRng::new();
        let mut id = String::with_capacity(self.size);
        
        for _ in 0..self.size {
            let index = rng.next_usize() % self.alphabet.len();
            id.push(self.alphabet[index]);
        }
        
        id
    }
}

impl Default for NanoIdGenerator {
    fn default() -> Self {
        Self::new()
    }
}

/// 时间戳ID生成器
pub struct TimestampIdGenerator {
    format: TimestampFormat,
}

#[derive(Debug, Clone)]
pub enum TimestampFormat {
    /// Unix时间戳（秒）
    UnixSeconds,
    /// Unix时间戳（毫秒）
    UnixMillis,
    /// Unix时间戳（微秒）
    UnixMicros,
    /// Unix时间戳（纳秒）
    UnixNanos,
    /// 可读格式 YYYYMMDDHHMMSS
    Readable,
}

impl TimestampIdGenerator {
    /// 创建新的时间戳ID生成器
    pub fn new(format: TimestampFormat) -> Self {
        Self { format }
    }

    /// 生成时间戳ID
    pub fn generate(&self) -> String {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default();

        match self.format {
            TimestampFormat::UnixSeconds => now.as_secs().to_string(),
            TimestampFormat::UnixMillis => now.as_millis().to_string(),
            TimestampFormat::UnixMicros => now.as_micros().to_string(),
            TimestampFormat::UnixNanos => now.as_nanos().to_string(),
            TimestampFormat::Readable => {
                // 这里简化处理，实际应该使用chrono等库
                let secs = now.as_secs();
                format!("{}", secs) // 简化版本
            }
        }
    }
}

/// 组合ID生成器
pub struct CompositeIdGenerator {
    prefix: String,
    generator: Box<dyn IdGenerator + Send + Sync>,
    suffix: String,
}

/// ID生成器trait
pub trait IdGenerator {
    fn generate(&self) -> String;
}

impl IdGenerator for UuidGenerator {
    fn generate(&self) -> String {
        Self::generate_v4()
    }
}

impl IdGenerator for NanoIdGenerator {
    fn generate(&self) -> String {
        self.generate()
    }
}

impl IdGenerator for TimestampIdGenerator {
    fn generate(&self) -> String {
        self.generate()
    }
}

impl CompositeIdGenerator {
    /// 创建组合ID生成器
    pub fn new(
        prefix: impl Into<String>,
        generator: Box<dyn IdGenerator + Send + Sync>,
        suffix: impl Into<String>,
    ) -> Self {
        Self {
            prefix: prefix.into(),
            generator,
            suffix: suffix.into(),
        }
    }

    /// 生成组合ID
    pub fn generate(&self) -> String {
        format!("{}{}{}", self.prefix, self.generator.generate(), self.suffix)
    }
}

/// 简单随机数生成器（用于UUID生成）
struct SimpleRng {
    state: u64,
}

impl SimpleRng {
    fn new() -> Self {
        let seed = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos() as u64;
        
        Self { state: seed }
    }

    fn next_u64(&mut self) -> u64 {
        // 简单的线性同余生成器
        self.state = self.state.wrapping_mul(1103515245).wrapping_add(12345);
        self.state
    }

    fn next_u8(&mut self) -> u8 {
        (self.next_u64() & 0xff) as u8
    }

    fn next_usize(&mut self) -> usize {
        self.next_u64() as usize
    }
}

/// ID工厂
pub struct IdFactory;

impl IdFactory {
    /// 生成UUID
    pub fn uuid() -> String {
        UuidGenerator::generate_v4()
    }

    /// 生成短UUID
    pub fn short_uuid() -> String {
        UuidGenerator::generate_short()
    }

    /// 生成NanoID
    pub fn nano_id() -> String {
        NanoIdGenerator::new().generate()
    }

    /// 生成时间戳ID
    pub fn timestamp_id() -> String {
        TimestampIdGenerator::new(TimestampFormat::UnixMillis).generate()
    }

    /// 生成雪花ID
    pub fn snowflake_id(node_id: u64) -> IdResult<u64> {
        let generator = SnowflakeGenerator::new(node_id)?;
        generator.next_id()
    }

    /// 生成带前缀的ID
    pub fn prefixed_id(prefix: &str) -> String {
        format!("{}_{}", prefix, Self::short_uuid())
    }
} 