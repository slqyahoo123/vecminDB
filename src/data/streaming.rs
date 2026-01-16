//! 流式数据处理模块
//! 提供大规模数据的流式处理能力，支持内存高效的数据处理

use std::path::Path;
use std::sync::{Arc, Mutex};
use std::collections::HashMap;
use std::io::{BufReader, BufRead};
use std::fs::File;
use std::time::{Duration, Instant};
 
 

use log::{debug, info, warn, error};
use serde::{Serialize, Deserialize};
use csv::ReaderBuilder;
use csv::Reader;
use tokio::sync::RwLock as AsyncRwLock;
use tokio::time::sleep as async_sleep;
 
use async_trait::async_trait;

use crate::error::{Result, Error};
use crate::data::value::{DataValue, UnifiedValue, ScalarValue, UnifiedToData};
use crate::data::types::{DataFormat};

/// 估算DataValue的字节大小
fn estimate_data_value_size(value: &DataValue) -> usize {
    match value {
        DataValue::Null => 0,
        DataValue::Boolean(_) => 1,
        DataValue::Integer(_) => 8,
        DataValue::Float(_) => 8,
        DataValue::Number(_) => 8,
        DataValue::String(s) => s.len(),
        DataValue::Text(s) => s.len(),
        DataValue::Array(arr) => arr.iter().map(estimate_data_value_size).sum::<usize>() + 8,
        DataValue::StringArray(arr) => arr.iter().map(|s| s.len()).sum::<usize>() + 8,
        DataValue::Object(obj) => {
            let keys_size: usize = obj.keys().map(|k| k.len()).sum();
            let values_size: usize = obj.values().map(estimate_data_value_size).sum();
            keys_size + values_size + 8
        },
        DataValue::Binary(bytes) => bytes.len(),
        DataValue::DateTime(s) => s.len(),
        DataValue::Tensor(tensor) => {
            // 估算张量大小：shape + data + metadata
            tensor.shape.len() * 8 + tensor.data.len() * 4 + tensor.metadata.len() * 16
        },
    }
}

/// 流式数据处理器配置
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StreamingConfig {
    /// 批次大小
    pub batch_size: usize,
    /// 缓冲区大小
    pub buffer_size: usize,
    /// 工作线程数
    pub worker_threads: usize,
    /// 最大内存使用量（字节）
    pub max_memory_mb: usize,
    /// 检查点间隔
    pub checkpoint_interval: Duration,
    /// 是否启用压缩
    pub enable_compression: bool,
    /// 错误容忍度
    pub error_threshold: f64,
}

impl Default for StreamingConfig {
    fn default() -> Self {
        Self {
            batch_size: 1000,
            buffer_size: 10000,
            worker_threads: num_cpus::get(),
            max_memory_mb: 1024, // 1GB
            checkpoint_interval: Duration::from_secs(60),
            enable_compression: true,
            error_threshold: 0.01, // 1% 错误率
        }
    }
}

/// 流式数据批次
#[derive(Debug, Clone)]
pub struct StreamingBatch {
    /// 批次ID
    pub batch_id: u64,
    /// 数据记录
    pub records: Vec<DataValue>,
    /// 元数据
    pub metadata: HashMap<String, String>,
    /// 时间戳
    pub timestamp: Instant,
    /// 大小（字节）
    pub size_bytes: usize,
}

impl StreamingBatch {
    /// 创建新的流式批次
    pub fn new(batch_id: u64, records: Vec<DataValue>) -> Self {
        let size_bytes = records.iter()
            .map(|r| estimate_data_value_size(r))
            .sum();
        
        Self {
            batch_id,
            records,
            metadata: HashMap::new(),
            timestamp: Instant::now(),
            size_bytes,
        }
    }

    /// 获取记录数量
    pub fn len(&self) -> usize {
        self.records.len()
    }

    /// 是否为空
    pub fn is_empty(&self) -> bool {
        self.records.is_empty()
    }

    /// 添加元数据
    pub fn add_metadata(&mut self, key: String, value: String) {
        self.metadata.insert(key, value);
    }
}

/// 流式数据处理状态
#[derive(Debug, Clone, Serialize)]
pub struct StreamingStats {
    /// 已处理的批次数
    pub batches_processed: u64,
    /// 已处理的记录数
    pub records_processed: u64,
    /// 错误数量
    pub error_count: u64,
    /// 平均处理时间（毫秒）
    pub avg_processing_time_ms: f64,
    /// 内存使用量（字节）
    pub memory_usage_bytes: usize,
    /// 处理速度（记录/秒）
    pub processing_rate: f64,
    /// 开始时间（不序列化，因为Instant无法序列化）
    #[serde(skip)]
    pub start_time: Instant,
    /// 最后更新时间（不序列化，因为Instant无法序列化）
    #[serde(skip)]
    pub last_update: Instant,
}

impl<'de> Deserialize<'de> for StreamingStats {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        use serde::de::{self, MapAccess, Visitor};
        use std::fmt;

        struct StreamingStatsVisitor;

        impl<'de> Visitor<'de> for StreamingStatsVisitor {
            type Value = StreamingStats;

            fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
                formatter.write_str("struct StreamingStats")
            }

            fn visit_map<V>(self, mut map: V) -> std::result::Result<StreamingStats, V::Error>
            where
                V: MapAccess<'de>,
            {
                let mut batches_processed = None;
                let mut records_processed = None;
                let mut error_count = None;
                let mut avg_processing_time_ms = None;
                let mut memory_usage_bytes = None;
                let mut processing_rate = None;

                while let Some(key) = map.next_key()? {
                    match key {
                        "batches_processed" => {
                            if batches_processed.is_some() {
                                return Err(de::Error::duplicate_field("batches_processed"));
                            }
                            batches_processed = Some(map.next_value()?);
                        }
                        "records_processed" => {
                            if records_processed.is_some() {
                                return Err(de::Error::duplicate_field("records_processed"));
                            }
                            records_processed = Some(map.next_value()?);
                        }
                        "error_count" => {
                            if error_count.is_some() {
                                return Err(de::Error::duplicate_field("error_count"));
                            }
                            error_count = Some(map.next_value()?);
                        }
                        "avg_processing_time_ms" => {
                            if avg_processing_time_ms.is_some() {
                                return Err(de::Error::duplicate_field("avg_processing_time_ms"));
                            }
                            avg_processing_time_ms = Some(map.next_value()?);
                        }
                        "memory_usage_bytes" => {
                            if memory_usage_bytes.is_some() {
                                return Err(de::Error::duplicate_field("memory_usage_bytes"));
                            }
                            memory_usage_bytes = Some(map.next_value()?);
                        }
                        "processing_rate" => {
                            if processing_rate.is_some() {
                                return Err(de::Error::duplicate_field("processing_rate"));
                            }
                            processing_rate = Some(map.next_value()?);
                        }
                        _ => {
                            let _ = map.next_value::<de::IgnoredAny>()?;
                        }
                    }
                }

                Ok(StreamingStats {
                    batches_processed: batches_processed.ok_or_else(|| de::Error::missing_field("batches_processed"))?,
                    records_processed: records_processed.ok_or_else(|| de::Error::missing_field("records_processed"))?,
                    error_count: error_count.ok_or_else(|| de::Error::missing_field("error_count"))?,
                    avg_processing_time_ms: avg_processing_time_ms.ok_or_else(|| de::Error::missing_field("avg_processing_time_ms"))?,
                    memory_usage_bytes: memory_usage_bytes.ok_or_else(|| de::Error::missing_field("memory_usage_bytes"))?,
                    processing_rate: processing_rate.ok_or_else(|| de::Error::missing_field("processing_rate"))?,
                    start_time: Instant::now(),
                    last_update: Instant::now(),
                })
            }
        }

        deserializer.deserialize_map(StreamingStatsVisitor)
    }
}

impl Default for StreamingStats {
    fn default() -> Self {
        let now = Instant::now();
        Self {
            batches_processed: 0,
            records_processed: 0,
            error_count: 0,
            avg_processing_time_ms: 0.0,
            memory_usage_bytes: 0,
            processing_rate: 0.0,
            start_time: now,
            last_update: now,
        }
    }
}

impl StreamingStats {
    /// 更新统计信息
    pub fn update_batch_processed(&mut self, batch: &StreamingBatch, processing_time: Duration) {
        self.batches_processed += 1;
        self.records_processed += batch.len() as u64;
        
        // 更新平均处理时间
        let processing_time_ms = processing_time.as_millis() as f64;
        self.avg_processing_time_ms = (self.avg_processing_time_ms * (self.batches_processed - 1) as f64 + processing_time_ms) / self.batches_processed as f64;
        
        // 更新处理速度
        let elapsed = self.start_time.elapsed().as_secs_f64();
        if elapsed > 0.0 {
            self.processing_rate = self.records_processed as f64 / elapsed;
        }
        
        self.last_update = Instant::now();
    }

    /// 记录错误
    pub fn record_error(&mut self) {
        self.error_count += 1;
    }

    /// 获取错误率
    pub fn error_rate(&self) -> f64 {
        if self.batches_processed == 0 {
            0.0
        } else {
            self.error_count as f64 / self.batches_processed as f64
        }
    }
}

/// 流式数据源特质
#[async_trait]
pub trait StreamingDataSource: Send + Sync {
    /// 读取下一个批次
    async fn next_batch(&mut self) -> Result<Option<StreamingBatch>>;
    
    /// 重置数据源
    async fn reset(&mut self) -> Result<()>;
    
    /// 获取数据源信息
    fn source_info(&self) -> HashMap<String, String>;
    
    /// 估计总记录数（如果可能）
    fn estimated_total_records(&self) -> Option<u64>;
}

/// 文件流式数据源
pub struct FileStreamingSource {
    /// 文件路径
    file_path: String,
    /// 文件格式
    format: DataFormat,
    /// 配置
    config: StreamingConfig,
    /// 当前文件读取器
    reader: Option<Arc<Mutex<Box<dyn BufRead + Send>>>>,
    /// CSV读取器（如果是CSV格式）
    csv_reader: Option<Reader<Box<BufReader<File>>>>,
    /// 当前批次ID
    current_batch_id: u64,
    /// 已读取的记录数
    records_read: u64,
    /// 文件大小
    file_size: u64,
}

impl FileStreamingSource {
    /// 创建新的文件流式数据源
    pub fn new<P: AsRef<Path>>(path: P, format: DataFormat, config: StreamingConfig) -> Result<Self> {
        let file_path = path.as_ref().to_string_lossy().to_string();
        let file_size = std::fs::metadata(&file_path)
            .map_err(|e| Error::io_error(format!("获取文件大小失败: {}", e)))?
            .len();
        
        Ok(Self {
            file_path,
            format,
            config,
            reader: None,
            csv_reader: None,
            current_batch_id: 0,
            records_read: 0,
            file_size,
        })
    }

    /// 初始化读取器
    fn initialize_reader(&mut self) -> Result<()> {
        let file = File::open(&self.file_path)
            .map_err(|e| Error::io_error(format!("打开文件失败: {}", e)))?;
        
        let buf_reader = Box::new(BufReader::new(file));
        
        match self.format {
            DataFormat::Csv => {
                let csv_reader = ReaderBuilder::new()
                    .has_headers(true)
                    .trim(csv::Trim::All)
                    .flexible(true)
                    .from_reader(buf_reader);
                
                self.csv_reader = Some(csv_reader);
            },
            _ => {
                self.reader = Some(Arc::new(Mutex::new(buf_reader)));
            }
        }
        
        Ok(())
    }

    /// 读取CSV批次
    fn read_csv_batch(&mut self) -> Result<Option<StreamingBatch>> {
        if self.csv_reader.is_none() {
            self.initialize_reader()?;
        }

        let csv_reader = self.csv_reader.as_mut().unwrap();
        let mut records = Vec::with_capacity(self.config.batch_size);
        
        for _ in 0..self.config.batch_size {
            match csv_reader.records().next() {
                Some(Ok(record)) => {
                    let mut data_map: HashMap<String, DataValue> = HashMap::new();
                    
                    // 如果是第一次读取，获取标题
                    let headers = if self.records_read == 0 {
                        csv_reader.headers().map_err(|e| Error::invalid_input(format!("读取CSV标题失败: {}", e)))?.clone()
                    } else {
                        // CSV 读取器每次调用 headers() 都会重新解析，性能开销较小
                        // 如需优化，可在首次读取时缓存标题到结构体中
                        csv_reader.headers().map_err(|e| Error::invalid_input(format!("读取CSV标题失败: {}", e)))?.clone()
                    };
                    
                    // 将CSV行转换为数据值
                    for (i, field) in record.iter().enumerate() {
                        if let Some(header) = headers.get(i) {
                            let unified_value = parse_csv_field(field);
                            let data_value = unified_value.unified_to_data()
                                .map_err(|e| Error::processing(format!("转换UnifiedValue失败: {}", e)))?;
                            data_map.insert(header.to_string(), data_value);
                        }
                    }
                    
                    records.push(DataValue::Object(data_map));
                    self.records_read += 1;
                },
                Some(Err(e)) => {
                    warn!("跳过无效的CSV行: {}", e);
                    continue;
                },
                None => break, // 文件结束
            }
        }
        
        if records.is_empty() {
            Ok(None)
        } else {
            self.current_batch_id += 1;
            Ok(Some(StreamingBatch::new(self.current_batch_id, records)))
        }
    }
}

/// 解析CSV字段
fn parse_csv_field(field: &str) -> UnifiedValue {
    if field.is_empty() {
        return UnifiedValue::Text(String::new());
    }
    
    // 尝试解析为数字
    if let Ok(i) = field.parse::<i64>() {
        return UnifiedValue::Integer(i);
    }
    
    if let Ok(f) = field.parse::<f64>() {
        return UnifiedValue::Float(f);
    }
    
    // 检查布尔值
    match field.to_lowercase().as_str() {
        "true" | "yes" | "1" => UnifiedValue::Scalar(ScalarValue::Bool(true)),
        "false" | "no" | "0" => UnifiedValue::Scalar(ScalarValue::Bool(false)),
        _ => UnifiedValue::Text(field.to_string()),
    }
}

#[async_trait]
impl StreamingDataSource for FileStreamingSource {
    async fn next_batch(&mut self) -> Result<Option<StreamingBatch>> {
        match self.format {
            DataFormat::Csv => self.read_csv_batch(),
            _ => Err(Error::not_implemented(format!("暂不支持流式处理格式: {:?}", self.format))),
        }
    }

    async fn reset(&mut self) -> Result<()> {
        self.reader = None;
        self.csv_reader = None;
        self.current_batch_id = 0;
        self.records_read = 0;
        Ok(())
    }

    fn source_info(&self) -> HashMap<String, String> {
        let mut info = HashMap::new();
        info.insert("type".to_string(), "file".to_string());
        info.insert("path".to_string(), self.file_path.clone());
        info.insert("format".to_string(), format!("{:?}", self.format));
        info.insert("file_size".to_string(), self.file_size.to_string());
        info.insert("records_read".to_string(), self.records_read.to_string());
        info
    }

    fn estimated_total_records(&self) -> Option<u64> {
        // 基于文件大小和已读记录数估算
        if self.records_read > 0 {
            let avg_record_size = self.file_size as f64 / self.records_read as f64;
            Some((self.file_size as f64 / avg_record_size) as u64)
        } else {
            None
        }
    }
}

/// 流式数据处理器
pub struct StreamingProcessor {
    /// 配置
    config: StreamingConfig,
    /// 统计信息
    stats: Arc<Mutex<StreamingStats>>,
    /// 数据源
    source: Box<dyn StreamingDataSource>,
    /// 处理函数
    processor_fn: Arc<dyn Fn(&mut StreamingBatch) -> Result<()> + Send + Sync>,
    /// 是否正在运行
    is_running: Arc<Mutex<bool>>,
}

impl StreamingProcessor {
    /// 创建新的流式处理器
    pub fn new(
        config: StreamingConfig,
        source: Box<dyn StreamingDataSource>,
        processor_fn: Arc<dyn Fn(&mut StreamingBatch) -> Result<()> + Send + Sync>,
    ) -> Self {
        Self {
            config,
            stats: Arc::new(Mutex::new(StreamingStats::default())),
            source,
            processor_fn,
            is_running: Arc::new(Mutex::new(false)),
        }
    }

    /// 开始流式处理
    pub async fn start_processing(&mut self) -> Result<()> {
        {
            let mut running = self.is_running.lock().unwrap();
            if *running {
                return Err(Error::invalid_state("流式处理器已在运行".to_string()));
            }
            *running = true;
        }

        info!("开始流式数据处理");
        
        while *self.is_running.lock().unwrap() {
            let start_time = Instant::now();
            
            // 读取下一批数据
            match self.source.next_batch().await {
                Ok(Some(mut batch)) => {
                    // 处理批次
                    match (self.processor_fn)(&mut batch) {
                        Ok(()) => {
                            let processing_time = start_time.elapsed();
                            
                            // 更新统计信息
                            {
                                let mut stats = self.stats.lock().unwrap();
                                stats.update_batch_processed(&batch, processing_time);
                                
                                // 检查错误率
                                if stats.error_rate() > self.config.error_threshold {
                                    error!("错误率超过阈值: {:.2}%", stats.error_rate() * 100.0);
                                    break;
                                }
                            }
                            
                            debug!("成功处理批次 {}, 记录数: {}, 耗时: {:?}", 
                                   batch.batch_id, batch.len(), processing_time);
                        },
                        Err(e) => {
                            error!("处理批次 {} 失败: {}", batch.batch_id, e);
                            self.stats.lock().unwrap().record_error();
                        }
                    }
                },
                Ok(None) => {
                    info!("数据源已耗尽，处理完成");
                    break;
                },
                Err(e) => {
                    error!("读取数据批次失败: {}", e);
                    self.stats.lock().unwrap().record_error();
                }
            }
            
            // 检查内存使用
            self.check_memory_usage().await?;
        }

        {
            let mut running = self.is_running.lock().unwrap();
            *running = false;
        }

        info!("流式数据处理完成");
        Ok(())
    }

    /// 停止处理
    pub fn stop_processing(&self) {
        let mut running = self.is_running.lock().unwrap();
        *running = false;
        info!("请求停止流式数据处理");
    }

    /// 获取统计信息
    pub fn get_stats(&self) -> StreamingStats {
        self.stats.lock().unwrap().clone()
    }

    /// 检查内存使用
    async fn check_memory_usage(&self) -> Result<()> {
        let current_memory = self.estimate_memory_usage();
        let max_memory = self.config.max_memory_mb * 1024 * 1024; // 转换为字节
        
        if current_memory > max_memory {
            warn!("内存使用超过限制: {} MB > {} MB", 
                  current_memory / 1024 / 1024, 
                  self.config.max_memory_mb);
            
            // 强制垃圾回收
            // 在实际实现中，这里可能需要更复杂的内存管理策略
            async_sleep(Duration::from_millis(100)).await;
        }
        
        Ok(())
    }

    /// 估算内存使用量
    fn estimate_memory_usage(&self) -> usize {
        // 内存使用估算：返回当前统计的内存使用量
        let stats = self.stats.lock().unwrap();
        stats.memory_usage_bytes
    }
}

/// 流式数据聚合器
pub struct StreamingAggregator {
    /// 配置
    config: StreamingConfig,
    /// 聚合状态
    aggregation_state: Arc<AsyncRwLock<HashMap<String, DataValue>>>,
    /// 窗口大小
    window_size: Duration,
    /// 滑动间隔
    slide_interval: Duration,
}

impl StreamingAggregator {
    /// 创建新的流式聚合器
    pub fn new(
        config: StreamingConfig,
        window_size: Duration,
        slide_interval: Duration,
    ) -> Self {
        Self {
            config,
            aggregation_state: Arc::new(AsyncRwLock::new(HashMap::new())),
            window_size,
            slide_interval,
        }
    }

    /// 处理新的数据批次
    pub async fn process_batch(&self, batch: &StreamingBatch) -> Result<()> {
        let mut state = self.aggregation_state.write().await;
        
        // 聚合逻辑：对数值类型进行计数和求和，字符串类型进行计数
        for record in &batch.records {
            if let DataValue::Object(map) = record {
                for (key, value) in map {
                    let count_key = format!("{}_count", key);
                    let sum_key = format!("{}_sum", key);
                    
                    // 增加计数
                    let current_count = state.get(&count_key)
                        .and_then(|v| if let DataValue::Number(n) = v { Some(*n) } else { None })
                        .unwrap_or(0.0);
                    state.insert(count_key, DataValue::Number(current_count + 1.0));
                    
                    // 累加数值
                    if let DataValue::Float(f) = value {
                        let current_sum = state.get(&sum_key)
                            .and_then(|v| if let DataValue::Number(n) = v { Some(*n) } else { None })
                            .unwrap_or(0.0);
                        state.insert(sum_key, DataValue::Number(current_sum + f));
                    } else if let DataValue::Integer(i) = value {
                        let current_sum = state.get(&sum_key)
                            .and_then(|v| if let DataValue::Number(n) = v { Some(*n) } else { None })
                            .unwrap_or(0.0);
                        state.insert(sum_key, DataValue::Number(current_sum + *i as f64));
                    }
                }
            }
        }
        
        Ok(())
    }

    /// 获取当前聚合结果
    pub async fn get_aggregation_result(&self) -> HashMap<String, DataValue> {
        self.aggregation_state.read().await.clone()
    }

    /// 重置聚合状态
    pub async fn reset(&self) {
        self.aggregation_state.write().await.clear();
    }
}

/// 创建文件流式数据源
pub fn create_file_streaming_source<P: AsRef<Path>>(
    path: P,
    format: DataFormat,
    config: Option<StreamingConfig>,
) -> Result<FileStreamingSource> {
    let config = config.unwrap_or_default();
    FileStreamingSource::new(path, format, config)
}

/// 创建默认的流式处理器
pub fn create_streaming_processor(
    source: Box<dyn StreamingDataSource>,
    config: Option<StreamingConfig>,
) -> StreamingProcessor {
    let config = config.unwrap_or_default();
    let processor_fn = Arc::new(|_batch: &mut StreamingBatch| {
        // 默认处理器 - 什么都不做
        Ok(())
    });
    
    StreamingProcessor::new(config, source, processor_fn)
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;
    use std::io::Write;

    #[tokio::test]
    async fn test_file_streaming_source() {
        let dir = tempdir().unwrap();
        let file_path = dir.path().join("test.csv");
        
        // 创建测试CSV文件
        let mut file = File::create(&file_path).unwrap();
        writeln!(file, "name,age,score").unwrap();
        writeln!(file, "Alice,25,95.5").unwrap();
        writeln!(file, "Bob,30,87.2").unwrap();
        writeln!(file, "Charlie,35,92.8").unwrap();
        
        let config = StreamingConfig {
            batch_size: 2,
            ..Default::default()
        };
        
        let mut source = FileStreamingSource::new(file_path, DataFormat::Csv, config).unwrap();
        
        // 读取第一批
        let batch1 = source.next_batch().await.unwrap();
        assert!(batch1.is_some());
        let batch1 = batch1.unwrap();
        assert_eq!(batch1.len(), 2);
        
        // 读取第二批
        let batch2 = source.next_batch().await.unwrap();
        assert!(batch2.is_some());
        let batch2 = batch2.unwrap();
        assert_eq!(batch2.len(), 1);
        
        // 应该没有更多数据
        let batch3 = source.next_batch().await.unwrap();
        assert!(batch3.is_none());
    }

    #[tokio::test]
    async fn test_streaming_aggregator() {
        let config = StreamingConfig::default();
        let aggregator = StreamingAggregator::new(
            config,
            Duration::from_secs(60),
            Duration::from_secs(10),
        );
        
        // 创建测试批次
        let mut data_map = HashMap::new();
        data_map.insert("value".to_string(), UnifiedValue::Float(10.5));
        let records = vec![DataValue::Object(data_map)];
        let batch = StreamingBatch::new(1, records);
        
        // 处理批次
        aggregator.process_batch(&batch).await.unwrap();
        
        // 检查结果
        let result = aggregator.get_aggregation_result().await;
        assert!(result.contains_key("value_count"));
        assert!(result.contains_key("value_sum"));
    }
} 