use std::fs::{File, OpenOptions};
use std::io::{Read, Write, Seek, SeekFrom};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, RwLock};
use std::time::{Duration, Instant, SystemTime};
use log::{debug, info, warn};
use serde::{Serialize, Deserialize};
use std::collections::HashMap;
use std::collections::VecDeque;

use crate::event::{Event, EventType, EventSystem};
use crate::Result;
use crate::Error;

// 导入EventSubscriber类型
use crate::event::memory::EventSubscriber;

/// 基于文件的事件存储系统
///
/// 将事件持久化存储到文件中，支持事件的写入、读取和查询。
/// 适合需要持久化记录事件的场景，如审计、历史数据分析等。
#[derive(Clone)]
pub struct FileEventStorage {
    /// 文件路径
    path: PathBuf,
    /// 文件句柄
    file: Arc<Mutex<File>>,
    /// 缓存的事件索引
    index: Arc<RwLock<EventIndex>>,
    /// 存储配置
    config: FileStorageConfig,
    /// 上次刷新时间
    last_flush: Arc<Mutex<Instant>>,
}

/// 文件存储配置
#[derive(Debug, Clone)]
pub struct FileStorageConfig {
    /// 自动刷新间隔
    pub flush_interval: Duration,
    /// 每个文件的最大大小（字节）
    pub max_file_size: u64,
    /// 是否启用压缩
    pub enable_compression: bool,
    /// 是否使用JSON格式存储
    pub use_json_format: bool,
    /// 是否启用索引
    pub enable_indexing: bool,
}

impl Default for FileStorageConfig {
    fn default() -> Self {
        Self {
            flush_interval: Duration::from_secs(5),
            max_file_size: 100 * 1024 * 1024, // 100MB
            enable_compression: false,
            use_json_format: true,
            enable_indexing: true,
        }
    }
}

/// 事件索引
#[derive(Debug, Default, Serialize, Deserialize)]
struct EventIndex {
    /// 按事件类型索引，存储事件在文件中的偏移量
    by_type: std::collections::HashMap<EventType, Vec<EventPosition>>,
    /// 按时间范围索引
    by_time: Vec<EventPosition>,
    /// 总事件数
    count: usize,
    /// 最后一个事件的位置
    last_position: Option<u64>,
}

/// 事件在文件中的位置信息
#[derive(Debug, Clone, Serialize, Deserialize)]
struct EventPosition {
    /// 文件偏移量
    offset: u64,
    /// 事件大小
    size: u32,
    /// 事件时间戳
    timestamp: SystemTime,
    /// 事件类型
    event_type: EventType,
    /// 事件ID
    event_id: String,
}

impl FileEventStorage {
    /// 创建新的文件事件存储
    pub fn new<P: AsRef<Path>>(path: P, config: Option<FileStorageConfig>) -> Result<Self> {
        let path = path.as_ref().to_path_buf();
        let config = config.unwrap_or_default();
        
        // 确保目录存在
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        
        // 打开或创建文件
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .open(&path)?;
            
        // 初始化索引
        let index = if config.enable_indexing {
            Arc::new(RwLock::new(Self::load_or_create_index(&path)?))
        } else {
            Arc::new(RwLock::new(EventIndex::default()))
        };
        
        Ok(Self {
            path,
            file: Arc::new(Mutex::new(file)),
            index,
            config,
            last_flush: Arc::new(Mutex::new(Instant::now())),
        })
    }
    
    /// 加载或创建事件索引
    fn load_or_create_index(path: &Path) -> Result<EventIndex> {
        let index_path = path.with_extension("idx");
        
        if index_path.exists() {
            // 尝试加载现有索引
            match std::fs::read_to_string(&index_path) {
                Ok(content) => {
                    match serde_json::from_str(&content) {
                        Ok(index) => return Ok(index),
                        Err(e) => {
                            warn!("无法解析事件索引文件，将创建新索引: {}", e);
                        }
                    }
                },
                Err(e) => {
                    warn!("无法读取事件索引文件，将创建新索引: {}", e);
                }
            }
        }
        
        // 创建新索引
        Ok(EventIndex::default())
    }
    
    /// 保存事件索引
    fn save_index(&self) -> Result<()> {
        if !self.config.enable_indexing {
            return Ok(());
        }
        
        let index_path = self.path.with_extension("idx");
        let index = self.index.read().map_err(|e| {
            Error::Internal(format!("无法获取索引读锁: {}", e))
        })?;
        
        let content = serde_json::to_string_pretty(&*index)?;
        std::fs::write(&index_path, content)?;
        
        debug!("事件索引已保存: {:?}", index_path);
        Ok(())
    }
    
    /// 写入事件
    pub fn write_event(&self, event: &Event) -> Result<()> {
        let event_data = if self.config.use_json_format {
            serde_json::to_vec(&event)?
        } else {
            bincode::serialize(&event)?
        };
        
        let event_data = if self.config.enable_compression {
            // 实现压缩逻辑
            // 这里简单起见，使用zstd压缩库
            let mut compressed = Vec::new();
            let mut encoder = zstd::Encoder::new(&mut compressed, 3)?;
            encoder.write_all(&event_data)?;
            encoder.finish()?;
            compressed
        } else {
            event_data
        };
        
        // 写入事件数据
        let mut file = self.file.lock().map_err(|e| {
            Error::Internal(format!("无法获取文件锁: {}", e))
        })?;
        
        // 检查文件大小是否超过限制
        let file_size = file.metadata()?.len();
        if file_size >= self.config.max_file_size {
            // 实现文件轮转逻辑
            self.rotate_file()?;
        }
        
        // 获取当前文件位置作为事件偏移量
        let offset = file.seek(SeekFrom::End(0))?;
        
        // 写入事件大小前缀（固定4字节）
        let event_size = event_data.len() as u32;
        file.write_all(&event_size.to_le_bytes())?;
        
        // 写入事件数据
        file.write_all(&event_data)?;
        
        // 更新索引
        if self.config.enable_indexing {
            let mut index = self.index.write().map_err(|e| {
                Error::Internal(format!("无法获取索引写锁: {}", e))
            })?;
            
            let event_type = event.event_type.clone();
            let position = EventPosition {
                offset,
                size: (4 + event_data.len()) as u32, // 包括大小前缀的总大小
                timestamp: SystemTime::UNIX_EPOCH + Duration::from_secs(event.timestamp),
                event_type: event_type.clone(),
                event_id: event.id.clone(),
            };
            
            // 更新类型索引
            index.by_type.entry(event_type)
                .or_insert_with(Vec::new)
                .push(position.clone());
                
            // 更新时间索引
            index.by_time.push(position);
            
            // 更新计数和最后位置
            index.count += 1;
            index.last_position = Some(offset);
        }
        
        // 检查是否需要刷新
        self.check_flush_needed()?;
        debug!("事件已写入，偏移量: {}，大小: {} 字节", offset, event_size);
        
        Ok(())
    }
    
    /// 检查是否需要刷新
    fn check_flush_needed(&self) -> Result<()> {
        let mut last_flush = self.last_flush.lock().map_err(|e| {
            Error::Internal(format!("无法获取刷新时间锁: {}", e))
        })?;
        
        let now = Instant::now();
        if now.duration_since(*last_flush) >= self.config.flush_interval {
            // 刷新文件
            let mut file = self.file.lock().map_err(|e| {
                Error::Internal(format!("无法获取文件锁: {}", e))
            })?;
            file.flush()?;
            
            // 保存索引
            if self.config.enable_indexing {
                self.save_index()?;
            }
            
            // 更新刷新时间
            *last_flush = now;
            debug!("事件文件已刷新");
        }
        
        Ok(())
    }
    
    /// 文件轮转
    fn rotate_file(&self) -> Result<()> {
        // 关闭当前文件
        let mut file = self.file.lock().map_err(|e| {
            Error::Internal(format!("无法获取文件锁: {}", e))
        })?;
        file.flush()?;
        drop(file);
        
        // 保存索引
        if self.config.enable_indexing {
            self.save_index()?;
        }
        
        // 生成新文件名
        let timestamp = chrono::Local::now().format("%Y%m%d_%H%M%S");
        let new_filename = format!("{}.{}.log", self.path.file_stem().unwrap().to_string_lossy(), timestamp);
        let new_path = self.path.with_file_name(new_filename);
        
        // 重命名当前文件
        std::fs::rename(&self.path, &new_path)?;
        info!("事件日志文件已轮转: {:?}", new_path);
        
        // 创建新文件
        let new_file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .open(&self.path)?;
            
        // 更新文件句柄
        let mut file_guard = self.file.lock().map_err(|e| {
            Error::Internal(format!("无法获取文件锁: {}", e))
        })?;
        *file_guard = new_file;
        
        // 重置索引
        if self.config.enable_indexing {
            let mut index = self.index.write().map_err(|e| {
                Error::Internal(format!("无法获取索引写锁: {}", e))
            })?;
            *index = EventIndex::default();
        }
        
        Ok(())
    }
    
    /// 读取指定位置的事件
    pub fn read_event_at(&self, offset: u64) -> Result<Event> {
        let mut file = self.file.lock().map_err(|e| {
            Error::Internal(format!("无法获取文件锁: {}", e))
        })?;
        
        // 设置文件位置
        file.seek(SeekFrom::Start(offset))?;
        
        // 读取事件大小前缀
        let mut size_bytes = [0u8; 4];
        file.read_exact(&mut size_bytes)?;
        let event_size = u32::from_le_bytes(size_bytes);
        
        // 读取事件数据
        let mut event_data = vec![0u8; event_size as usize];
        file.read_exact(&mut event_data)?;
        
        // 解析事件
        let event = if self.config.enable_compression {
            // 解压缩
            let mut decoder = zstd::Decoder::new(&event_data[..])?;
            let mut decompressed = Vec::new();
            std::io::Read::read_to_end(&mut decoder, &mut decompressed)?;
            
            if self.config.use_json_format {
                serde_json::from_slice(&decompressed)?
            } else {
                bincode::deserialize(&decompressed)?
            }
        } else {
            if self.config.use_json_format {
                serde_json::from_slice(&event_data)?
            } else {
                bincode::deserialize(&event_data)?
            }
        };
        
        Ok(event)
    }
    
    /// 查询事件
    pub fn query_events(&self, filter: EventQueryFilter) -> Result<Vec<Event>> {
        if !self.config.enable_indexing {
            return Err(Error::Unsupported("查询功能需要启用索引".to_string()));
        }
        
        let index = self.index.read().map_err(|e| {
            Error::Internal(format!("无法获取索引读锁: {}", e))
        })?;
        
        // 根据过滤条件选择事件位置
        let positions = match filter {
            EventQueryFilter::ByType(event_type) => {
                index.by_type.get(&event_type)
                    .map(|pos| pos.clone())
                    .unwrap_or_default()
            },
            EventQueryFilter::ByTimeRange(start, end) => {
                index.by_time.iter()
                    .filter(|pos| {
                        if let Some(start_time) = start {
                            if pos.timestamp < start_time {
                                return false;
                            }
                        }
                        if let Some(end_time) = end {
                            if pos.timestamp > end_time {
                                return false;
                            }
                        }
                        true
                    })
                    .cloned()
                    .collect()
            },
            EventQueryFilter::Latest(count) => {
                let len = index.by_time.len();
                if count >= len {
                    index.by_time.clone()
                } else {
                    index.by_time[len - count..].to_vec()
                }
            },
            EventQueryFilter::All => index.by_time.clone(),
        };
        
        // 读取事件
        let mut events = Vec::with_capacity(positions.len());
        for pos in positions {
            match self.read_event_at(pos.offset) {
                Ok(event) => events.push(event),
                Err(e) => {
                    warn!("读取事件失败，位置={}: {}", pos.offset, e);
                }
            }
        }
        
        Ok(events)
    }
    
    /// 获取事件总数
    pub fn count(&self) -> Result<usize> {
        if self.config.enable_indexing {
            let index = self.index.read().map_err(|e| {
                Error::Internal(format!("无法获取索引读锁: {}", e))
            })?;
            Ok(index.count)
        } else {
            Err(Error::Unsupported("计数功能需要启用索引".to_string()))
        }
    }
    
    /// 清空存储
    pub fn clear(&self) -> Result<()> {
        // 清空文件
        let mut file = self.file.lock().map_err(|e| {
            Error::Internal(format!("无法获取文件锁: {}", e))
        })?;
        file.set_len(0)?;
        file.flush()?;
        
        // 重置索引
        if self.config.enable_indexing {
            let mut index = self.index.write().map_err(|e| {
                Error::Internal(format!("无法获取索引写锁: {}", e))
            })?;
            *index = EventIndex::default();
            
            // 保存索引
            self.save_index()?;
        }
        
        info!("事件存储已清空");
        Ok(())
    }
}

/// 事件查询过滤器
#[derive(Debug, Clone)]
pub enum EventQueryFilter {
    /// 按事件类型查询
    ByType(EventType),
    /// 按时间范围查询
    ByTimeRange(Option<SystemTime>, Option<SystemTime>),
    /// 最新的N条事件
    Latest(usize),
    /// 所有事件
    All,
}

/// 基于文件的事件系统实现
/// 
/// 使用文件系统存储事件，支持持久化和大规模事件处理。
#[derive(Clone)]
pub struct FileEventSystem {
    /// 内存事件系统，用于实时事件处理
    memory_system: crate::event::memory::MemoryEventSystem,
    /// 文件存储系统，用于持久化事件
    storage: FileEventStorage,
    /// 事件文件路径
    file_path: PathBuf,
    /// 事件缓冲区
    buffer: Arc<Mutex<VecDeque<Event>>>,
    /// 事件订阅者
    subscribers: Arc<RwLock<HashMap<EventType, Vec<EventSubscriber>>>>,
    /// 全局订阅者
    global_subscribers: Arc<RwLock<Vec<EventSubscriber>>>,
    /// 运行状态
    is_running: Arc<Mutex<bool>>,
    /// 缓冲区大小
    buffer_size: usize,
    /// 文件写入间隔
    flush_interval: Duration,
    /// 最后刷新时间
    last_flush: Arc<Mutex<Instant>>,
}

impl FileEventSystem {
    /// 创建新的文件事件系统
    pub fn new<P: AsRef<Path>>(path: P, config: Option<FileStorageConfig>) -> Result<Self> {
        let storage = FileEventStorage::new(path.as_ref(), config)?;
        let memory_system = crate::event::memory::MemoryEventSystem::new(1000);
        
        Ok(Self {
            memory_system,
            storage,
            file_path: path.as_ref().to_path_buf(),
            buffer: Arc::new(Mutex::new(VecDeque::new())),
            subscribers: Arc::new(RwLock::new(HashMap::new())),
            global_subscribers: Arc::new(RwLock::new(Vec::new())),
            is_running: Arc::new(Mutex::new(false)),
            buffer_size: 1000,
            flush_interval: Duration::from_secs(5),
            last_flush: Arc::new(Mutex::new(Instant::now())),
        })
    }

    /// 就绪探针：注册最小回调并发布一次探针事件，行使 EventCallback 的导入
    pub fn readiness_probe(&self) -> Result<()> {
        struct ProbeCallback;
        impl crate::event::EventCallback for ProbeCallback {
            fn on_event(&self, _event: &crate::event::Event) -> crate::Result<()> { Ok(()) }
        }

        // 通过内存事件系统注册全局回调
        let id = self.memory_system.subscribe_all(Arc::new(ProbeCallback))?;
        // 发布一条探针事件（写入内存与文件）
        let evt = crate::event::Event::new(crate::event::EventType::SystemStarted, "file.readiness_probe");
        self.publish(evt)?;
        // 清理订阅
        let _ = self.memory_system.unsubscribe(&id);
        Ok(())
    }
    
    /// 启动事件处理循环
    pub fn start(&self) -> Result<()> {
        self.memory_system.start()
    }
    
    /// 停止事件处理
    pub fn stop(&self) -> Result<()> {
        self.memory_system.stop()
    }
    
    /// 查询历史事件
    pub fn query(&self, filter: EventQueryFilter) -> Result<Vec<Event>> {
        self.storage.query_events(filter)
    }
    
    /// 获取事件总数
    pub fn count(&self) -> Result<usize> {
        self.storage.count()
    }
    
    /// 清空事件存储
    pub fn clear(&self) -> Result<()> {
        self.storage.clear()
    }
}

impl EventSystem for FileEventSystem {
    fn publish(&self, event: Event) -> Result<()> {
        // 首先发布到内存系统，用于实时通知订阅者
        self.memory_system.publish(event.clone())?;
        
        // 然后持久化到文件
        self.storage.write_event(&event)?;
        
        Ok(())
    }
    
    fn subscribe(&self, event_type: EventType, callback: Arc<dyn crate::event::EventCallback>) -> Result<String> {
        // 委托给内存系统
        self.memory_system.subscribe(event_type, callback)
    }
    
    fn subscribe_all(&self, callback: Arc<dyn crate::event::EventCallback>) -> Result<String> {
        // 委托给内存系统
        self.memory_system.subscribe_all(callback)
    }
    
    fn unsubscribe(&self, subscription_id: &str) -> Result<()> {
        // 委托给内存系统
        self.memory_system.unsubscribe(subscription_id)
    }
    
    fn get_pending_events(&self) -> Result<Vec<Event>> {
        // 委托给内存系统
        self.memory_system.get_pending_events()
    }
    
    fn start(&self) -> Result<()> {
        // 调用已实现的start方法
        self.memory_system.start()
    }
    
    fn stop(&self) -> Result<()> {
        // 调用已实现的stop方法
        self.memory_system.stop()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;
    use std::time::UNIX_EPOCH;
    
    #[test]
    fn test_file_event_storage() {
        // 创建临时目录
        let dir = tempdir().unwrap();
        let file_path = dir.path().join("events.log");
        
        // 创建存储
        let config = FileStorageConfig {
            flush_interval: Duration::from_millis(100),
            max_file_size: 1024 * 1024,
            enable_compression: false,
            use_json_format: true,
            enable_indexing: true,
        };
        
        let storage = FileEventStorage::new(&file_path, Some(config)).unwrap();
        
        // 写入事件
        let event1 = Event {
            id: "event1".to_string(),
            event_type: EventType::SystemStarted,
            timestamp: UNIX_EPOCH + Duration::from_secs(1000),
            data: "事件1".to_string(),
            source: "test".to_string(),
            metadata: Default::default(),
        };
        
        storage.write_event(&event1).unwrap();
        
        // 读取事件
        let events = storage.query_events(EventQueryFilter::All).unwrap();
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].id, "event1");
        
        // 写入更多事件
        let event2 = Event {
            id: "event2".to_string(),
            event_type: EventType::SystemShutdown,
            timestamp: UNIX_EPOCH + Duration::from_secs(2000),
            data: "事件2".to_string(),
            source: "test".to_string(),
            metadata: Default::default(),
        };
        
        storage.write_event(&event2).unwrap();
        
        // 按类型查询
        let events = storage.query_events(EventQueryFilter::ByType(EventType::SystemStarted)).unwrap();
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].id, "event1");
        
        // 按时间范围查询
        let events = storage.query_events(EventQueryFilter::ByTimeRange(
            Some(UNIX_EPOCH + Duration::from_secs(1500)),
            Some(UNIX_EPOCH + Duration::from_secs(2500))
        )).unwrap();
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].id, "event2");
        
        // 查询最新事件
        let events = storage.query_events(EventQueryFilter::Latest(1)).unwrap();
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].id, "event2");
        
        // 检查计数
        assert_eq!(storage.count().unwrap(), 2);
    }
} 