//! 数据迭代器模块
//! 
//! 提供各种数据迭代器的实现，支持批次处理和流式数据处理

use std::sync::{Arc, Mutex};
use std::collections::{VecDeque, HashMap};
use async_trait::async_trait;
use crate::error::{Error, Result};
use crate::data::{DataBatch, DataFormat, DataStatus};
use crate::data::loader::DataLoader;
use chrono::Utc;
use uuid::Uuid;

/// 数据批次迭代器
pub struct DataBatchIterator {
    /// 数据加载器
    loader: Arc<dyn DataLoader>,
    /// 数据源路径
    data_path: String,
    /// 批次大小
    batch_size: usize,
    /// 当前索引
    current_index: usize,
    /// 总数据大小
    total_size: usize,
    /// 是否启用shuffle
    shuffle: bool,
    /// 缓存的批次
    cached_batches: VecDeque<DataBatch>,
    /// 预加载批次数量
    prefetch_count: usize,
    /// 是否已结束
    is_finished: bool,
    /// 数据格式
    format: DataFormat,
    /// 随机种子
    random_seed: Option<u64>,
}

impl DataBatchIterator {
    /// 创建新的数据批次迭代器
    pub fn new(
        loader: Arc<dyn DataLoader>,
        data_path: String,
        batch_size: usize,
    ) -> Self {
        Self {
            loader,
            data_path,
            batch_size,
            current_index: 0,
            total_size: 0,
            shuffle: false,
            cached_batches: VecDeque::new(),
            prefetch_count: 2,
            is_finished: false,
            format: DataFormat::CSV,
            random_seed: None,
        }
    }

    /// 设置是否启用shuffle
    pub fn with_shuffle(mut self, shuffle: bool) -> Self {
        self.shuffle = shuffle;
        self
    }

    /// 设置预加载批次数量
    pub fn with_prefetch_count(mut self, count: usize) -> Self {
        self.prefetch_count = count.max(1);
        self
    }

    /// 设置数据格式
    pub fn with_format(mut self, format: DataFormat) -> Self {
        self.format = format;
        self
    }

    /// 设置随机种子
    pub fn with_random_seed(mut self, seed: u64) -> Self {
        self.random_seed = Some(seed);
        self
    }

    /// 初始化迭代器
    pub async fn initialize(&mut self) -> Result<()> {
        // 获取数据总大小
        self.total_size = self.loader.get_total_size(&self.data_path).await
            .unwrap_or(0);
        
        if self.total_size == 0 {
            return Err(Error::invalid_data("数据源为空或无法访问".to_string()));
        }

        // 预加载初始批次
        self.preload_batches().await?;
        
        Ok(())
    }

    /// 获取下一个批次
    pub async fn next_batch(&mut self) -> Result<Option<DataBatch>> {
        if self.is_finished && self.cached_batches.is_empty() {
            return Ok(None);
        }

        // 如果缓存中有批次，直接返回
        if let Some(batch) = self.cached_batches.pop_front() {
            // 异步预加载更多批次
            if self.cached_batches.len() < self.prefetch_count && !self.is_finished {
                tokio::spawn({
                    let mut iterator_clone = self.clone();
                    async move {
                        let _ = iterator_clone.preload_batches().await;
                    }
                });
            }
            return Ok(Some(batch));
        }

        // 如果缓存为空但未结束，尝试加载新批次
        if !self.is_finished {
            self.preload_batches().await?;
            if let Some(batch) = self.cached_batches.pop_front() {
                return Ok(Some(batch));
            }
        }

        Ok(None)
    }

    /// 预加载批次
    async fn preload_batches(&mut self) -> Result<()> {
        while self.cached_batches.len() < self.prefetch_count && !self.is_finished {
            let batch = self.load_next_batch().await?;
            if let Some(batch) = batch {
                self.cached_batches.push_back(batch);
            } else {
                self.is_finished = true;
                break;
            }
        }
        Ok(())
    }

    /// 加载下一个批次
    async fn load_next_batch(&mut self) -> Result<Option<DataBatch>> {
        if self.current_index >= self.total_size {
            return Ok(None);
        }

        let end_index = (self.current_index + self.batch_size).min(self.total_size);
        let actual_batch_size = end_index - self.current_index;

        // 从数据加载器获取数据
        let source = crate::data::loader::DataSource::File(self.data_path.clone());
        // 将data::DataFormat转换为loader::types::DataFormat
        let loader_format = match &self.format {
            crate::data::DataFormat::CSV => crate::data::loader::types::DataFormat::csv(),
            crate::data::DataFormat::JSON => crate::data::loader::types::DataFormat::json(),
            crate::data::DataFormat::Parquet => crate::data::loader::types::DataFormat::parquet(),
            crate::data::DataFormat::Avro => crate::data::loader::types::DataFormat::avro(),
            crate::data::DataFormat::CustomText(fmt) => crate::data::loader::types::DataFormat::CustomText(fmt.clone()),
            _ => crate::data::loader::types::DataFormat::json(), // 默认为JSON
        };
        
        let batch_data = self.loader
            .load_batch(&source, &loader_format, actual_batch_size, self.current_index)
            .await?;

        // 更新批次的索引信息
        let mut batch = batch_data;
        batch.index = self.current_index / self.batch_size;
        if batch.id.is_none() {
            batch.id = Some(Uuid::new_v4().to_string());
        }

        self.current_index = end_index;
        Ok(Some(batch))
    }

    /// 重置迭代器
    pub fn reset(&mut self) {
        self.current_index = 0;
        self.cached_batches.clear();
        self.is_finished = false;
    }

    /// 获取总批次数量
    pub fn total_batches(&self) -> usize {
        if self.batch_size == 0 {
            return 0;
        }
        (self.total_size + self.batch_size - 1) / self.batch_size
    }

    /// 从DataBatch创建迭代器
    pub fn from_batch(batch: crate::data::exports::DataBatch) -> Result<Self> {
        // 创建一个简单的内存数据加载器
        let loader = Arc::new(InMemoryDataLoader::new(vec![batch]));
        
        Ok(Self {
            loader,
            data_path: "memory".to_string(),
            batch_size: 1, // 只有一个批次
            current_index: 0,
            total_size: 1,
            shuffle: false,
            cached_batches: VecDeque::new(),
            prefetch_count: 1,
            is_finished: false,
            format: DataFormat::JSON,
            random_seed: None,
        })
    }

    /// 获取当前批次索引
    pub fn current_batch_index(&self) -> usize {
        if self.batch_size == 0 {
            return 0;
        }
        self.current_index / self.batch_size
    }

    /// 获取剩余批次数量
    pub fn remaining_batches(&self) -> usize {
        let total = self.total_batches();
        let current = self.current_batch_index();
        total.saturating_sub(current)
    }

    /// 获取进度百分比
    pub fn progress(&self) -> f32 {
        if self.total_size == 0 {
            return 100.0;
        }
        (self.current_index as f32 / self.total_size as f32) * 100.0
    }

    /// 是否已完成
    pub fn is_finished(&self) -> bool {
        self.is_finished && self.cached_batches.is_empty()
    }

    /// 获取批次大小
    pub fn batch_size(&self) -> usize {
        self.batch_size
    }

    /// 获取总数据大小
    pub fn total_size(&self) -> usize {
        self.total_size
    }

    /// 跳转到指定批次
    pub async fn seek_to_batch(&mut self, batch_index: usize) -> Result<()> {
        let target_index = batch_index * self.batch_size;
        if target_index >= self.total_size {
            return Err(Error::invalid_argument("批次索引超出范围"));
        }

        self.current_index = target_index;
        self.cached_batches.clear();
        self.is_finished = false;
        
        // 重新预加载批次
        self.preload_batches().await?;
        Ok(())
    }

    /// 获取迭代器统计信息
    pub fn stats(&self) -> DataIteratorStats {
        DataIteratorStats {
            total_size: self.total_size,
            current_index: self.current_index,
            batch_size: self.batch_size,
            total_batches: self.total_batches(),
            current_batch_index: self.current_batch_index(),
            remaining_batches: self.remaining_batches(),
            progress_percent: self.progress(),
            is_finished: self.is_finished(),
            cached_batches_count: self.cached_batches.len(),
            prefetch_count: self.prefetch_count,
        }
    }
}

/// 为DataBatchIterator实现Clone trait
impl Clone for DataBatchIterator {
    fn clone(&self) -> Self {
        Self {
            loader: self.loader.clone(),
            data_path: self.data_path.clone(),
            batch_size: self.batch_size,
            current_index: self.current_index,
            total_size: self.total_size,
            shuffle: self.shuffle,
            cached_batches: VecDeque::new(), // 不复制缓存的批次
            prefetch_count: self.prefetch_count,
            is_finished: self.is_finished,
            format: self.format.clone(),
            random_seed: self.random_seed,
        }
    }
}

/// 数据迭代器统计信息
#[derive(Debug, Clone)]
pub struct DataIteratorStats {
    /// 总数据大小
    pub total_size: usize,
    /// 当前索引
    pub current_index: usize,
    /// 批次大小
    pub batch_size: usize,
    /// 总批次数量
    pub total_batches: usize,
    /// 当前批次索引
    pub current_batch_index: usize,
    /// 剩余批次数量
    pub remaining_batches: usize,
    /// 进度百分比
    pub progress_percent: f32,
    /// 是否已完成
    pub is_finished: bool,
    /// 缓存的批次数量
    pub cached_batches_count: usize,
    /// 预加载批次数量
    pub prefetch_count: usize,
}

/// 流式数据迭代器
pub struct StreamingDataIterator {
    /// 数据源
    source: Arc<Mutex<dyn StreamingDataSource>>,
    /// 批次大小
    batch_size: usize,
    /// 当前批次
    current_batch: Option<DataBatch>,
    /// 是否活跃
    is_active: bool,
    /// 累积数据
    buffer: Vec<u8>,
    /// 元数据
    metadata: std::collections::HashMap<String, String>,
}

/// 流式数据源trait
#[async_trait]
pub trait StreamingDataSource: Send + Sync {
    /// 读取数据块
    async fn read_chunk(&mut self) -> Result<Option<Vec<u8>>>;
    
    /// 检查是否还有数据
    async fn has_more_data(&self) -> bool;
    
    /// 重置数据源
    async fn reset(&mut self) -> Result<()>;
    
    /// 获取数据源信息
    fn get_info(&self) -> StreamingSourceInfo;
}

/// 流式数据源信息
#[derive(Debug, Clone)]
pub struct StreamingSourceInfo {
    /// 数据源名称
    pub name: String,
    /// 数据源类型
    pub source_type: String,
    /// 预估大小（字节）
    pub estimated_size: Option<usize>,
    /// 数据格式
    pub format: DataFormat,
    /// 元数据
    pub metadata: std::collections::HashMap<String, String>,
}

impl StreamingDataIterator {
    /// 创建新的流式数据迭代器
    pub fn new(source: Arc<Mutex<dyn StreamingDataSource>>, batch_size: usize) -> Self {
        Self {
            source,
            batch_size,
            current_batch: None,
            is_active: true,
            buffer: Vec::new(),
            metadata: Default::default(),
        }
    }

    /// 获取下一个批次
    pub async fn next_streaming_batch(&mut self) -> Result<Option<DataBatch>> {
        if !self.is_active {
            return Ok(None);
        }

        // 读取数据直到达到批次大小
        while self.buffer.len() < self.batch_size {
            let chunk = {
                let mut source = self.source.lock().unwrap();
                source.read_chunk().await?
            };
            
            if let Some(chunk) = chunk {
                self.buffer.extend(chunk);
            } else {
                // 数据源已结束
                self.is_active = false;
                break;
            }
        }

        if self.buffer.is_empty() {
            return Ok(None);
        }

        // 创建批次
        let batch_data: Vec<u8> = if self.buffer.len() >= self.batch_size {
            self.buffer.drain(..self.batch_size).collect()
        } else {
            self.buffer.drain(..).collect()
        };

        let batch = DataBatch {
            id: Some(Uuid::new_v4().to_string()),
            dataset_id: "streaming_dataset".to_string(),
            index: 0, // 流式数据没有固定索引
            batch_index: 0,
            size: batch_data.len(),
            batch_size: batch_data.len(),
            status: DataStatus::Loaded,
            created_at: Utc::now(),
            data: Some(batch_data),
            labels: None,
            metadata: self.metadata.clone(),
            format: {
                let source = self.source.lock().unwrap();
                source.get_info().format.clone()
            },
            source: Some("streaming".to_string()),
            records: vec![], // 流式数据记录集
            schema: None, // 流式数据暂时没有固定schema
            field_names: Vec::new(),
            features: None,
            target: None,
            version: None,
            checksum: None,
            compression: None,
            encryption: None,
            tags: Vec::new(),
            child_batch_ids: Vec::new(),
            custom_data: std::collections::HashMap::new(),
            dependencies: Vec::new(),
            ..Default::default()
        };

        Ok(Some(batch))
    }

    /// 添加元数据
    pub fn add_metadata(&mut self, key: String, value: String) {
        self.metadata.insert(key, value);
    }

    /// 停止迭代
    pub fn stop(&mut self) {
        self.is_active = false;
    }

    /// 重置迭代器
    pub async fn reset(&mut self) -> Result<()> {
        {
            let mut source = self.source.lock().unwrap();
            source.reset().await?;
        }
        self.buffer.clear();
        self.current_batch = None;
        self.is_active = true;
        Ok(())
    }
}

/// 创建数据批次迭代器的便利函数
pub fn create_batch_iterator(
    loader: Arc<dyn DataLoader>,
    data_path: String,
    batch_size: usize,
) -> DataBatchIterator {
    DataBatchIterator::new(loader, data_path, batch_size)
}

/// 创建流式数据迭代器的便利函数
pub fn create_streaming_iterator(
    source: Arc<Mutex<dyn StreamingDataSource>>,
    batch_size: usize,
) -> StreamingDataIterator {
    StreamingDataIterator::new(source, batch_size)
}

/// 内存数据加载器，用于from_batch方法
pub struct InMemoryDataLoader {
    batches: Vec<crate::data::exports::DataBatch>,
}

impl InMemoryDataLoader {
    pub fn new(batches: Vec<crate::data::exports::DataBatch>) -> Self {
        Self { batches }
    }
}

#[async_trait]
impl DataLoader for InMemoryDataLoader {
    async fn load(&self, _source: &crate::data::loader::DataSource, _format: &crate::data::loader::types::DataFormat) -> Result<DataBatch> {
        if let Some(batch) = self.batches.first() {
            // 将exports::DataBatch转换为DataBatch
            // batch.data 是 Vec<Vec<f32>>，需要转换为 DataRow
            use crate::data::value::DataValue;
            let records: Vec<HashMap<String, DataValue>> = batch.data.iter().enumerate().map(|(i, row)| {
                let mut record = HashMap::new();
                record.insert("features".to_string(), DataValue::Array(
                    row.iter().map(|&v| DataValue::Float(v as f64)).collect()
                ));
                record.insert("index".to_string(), DataValue::Integer(i as i64));
                record
            }).collect();
            
            Ok(DataBatch {
                id: Some(batch.id.clone()),
                dataset_id: "memory".to_string(),
                index: batch.batch_index,
                batch_index: batch.batch_index,
                size: batch.data.len(),
                batch_size: batch.batch_size,
                status: DataStatus::Loaded,
                created_at: batch.created_at,
                data: None,
                labels: if batch.labels.is_empty() {
                    None
                } else {
                    Some(bincode::serialize(&batch.labels).unwrap_or_default())
                },
                metadata: batch.metadata.clone(),
                format: DataFormat::JSON,
                source: Some("memory".to_string()),
                records,
                schema: None,
                field_names: vec![],
                features: Some(batch.data.clone()),
                target: if batch.labels.is_empty() {
                    None
                } else {
                    Some(batch.labels.clone())
                },
                version: None,
                checksum: None,
                compression: None,
                encryption: None,
                tags: vec![],
                validation_status: None,
                validation_errors: vec![],
                processing_time: std::time::Duration::ZERO,
                quality_score: None,
                parent_batch_id: None,
                child_batch_ids: vec![],
                dependencies: vec![],
                priority: 0,
                retry_count: 0,
                max_retries: 3,
                timeout: None,
                error_message: None,
                result: None,
                custom_data: HashMap::new(),
            })
        } else {
            Err(Error::DataError("没有可用的批次数据".to_string()))
        }
    }

    async fn load_batch(
        &self,
        _source: &crate::data::loader::DataSource,
        _format: &crate::data::loader::types::DataFormat,
        _batch_size: usize,
        batch_index: usize,
    ) -> Result<DataBatch> {
        if batch_index < self.batches.len() {
            let batch = &self.batches[batch_index];
            // batch.data 是 Vec<Vec<f32>>，需要转换为 records
            use crate::data::value::DataValue;
            let records: Vec<HashMap<String, DataValue>> = batch.data.iter().enumerate().map(|(i, row)| {
                let mut record = HashMap::new();
                record.insert("features".to_string(), DataValue::Array(
                    row.iter().map(|&v| DataValue::Float(v as f64)).collect()
                ));
                record.insert("index".to_string(), DataValue::Integer(i as i64));
                record
            }).collect();
            
            Ok(DataBatch {
                id: Some(batch.id.clone()),
                dataset_id: "memory".to_string(),
                index: batch.batch_index,
                batch_index: batch.batch_index,
                size: batch.data.len(),
                batch_size: batch.batch_size,
                status: DataStatus::Loaded,
                created_at: batch.created_at,
                data: None,
                labels: if batch.labels.is_empty() {
                    None
                } else {
                    Some(bincode::serialize(&batch.labels).unwrap_or_default())
                },
                metadata: batch.metadata.clone(),
                format: DataFormat::JSON,
                source: Some("memory".to_string()),
                records,
                schema: None,
                field_names: vec![],
                features: Some(batch.data.clone()),
                target: if batch.labels.is_empty() {
                    None
                } else {
                    Some(batch.labels.clone())
                },
                version: None,
                checksum: None,
                compression: None,
                encryption: None,
                tags: vec![],
                validation_status: None,
                validation_errors: vec![],
                processing_time: std::time::Duration::ZERO,
                quality_score: None,
                parent_batch_id: None,
                child_batch_ids: vec![],
                dependencies: vec![],
                priority: 0,
                retry_count: 0,
                max_retries: 3,
                timeout: None,
                error_message: None,
                result: None,
                custom_data: HashMap::new(),
            })
        } else {
            Err(Error::DataError("批次索引超出范围".to_string()))
        }
    }

    async fn get_total_size(&self, _source: &str) -> Result<usize> {
        Ok(self.batches.len())
    }

    async fn validate_source(&self, _source: &crate::data::loader::DataSource) -> Result<bool> {
        Ok(true)
    }

    async fn get_schema(&self, _source: &crate::data::loader::DataSource, _format: &crate::data::loader::types::DataFormat) -> Result<crate::data::schema::schema::DataSchema> {
        // 创建一个默认的DataSchema
        Ok(crate::data::schema::schema::DataSchema {
            name: "in_memory_schema".to_string(),
            version: "1.0.0".to_string(),
            description: None,
            fields: vec![],
            primary_key: None,
            indexes: None,
            relationships: None,
            metadata: std::collections::HashMap::new(),
        })
    }
    
    /// 获取数据加载器名称
    fn name(&self) -> &'static str {
        "in_memory"
    }
    
    /// 检查是否支持指定格式
    fn supports_format(&self, _format: &crate::data::loader::types::DataFormat) -> bool {
        true // 内存加载器支持所有格式
    }
    
    /// 获取配置
    fn config(&self) -> &crate::data::loader::LoaderConfig {
        // 提供一个静态的默认配置
        static DEFAULT_CONFIG: std::sync::OnceLock<crate::data::loader::LoaderConfig> = std::sync::OnceLock::new();
        DEFAULT_CONFIG.get_or_init(|| crate::data::loader::LoaderConfig::default())
    }
    
    /// 设置配置
    fn set_config(&mut self, _config: crate::data::loader::LoaderConfig) {
        // 内存加载器不需要配置，这里是空实现
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    // 提供一个简易内存Mock以替代缺失的 loader::mock
    struct MockDataLoader;
    
    impl MockDataLoader {
        fn new() -> Self {
            Self
        }
    }
    
    #[async_trait::async_trait]
    impl super::DataLoader for MockDataLoader {
        async fn load_batch(&self, _source: &crate::data::loader::DataSource, _format: &crate::data::loader::DataFormat, batch_size: usize, offset: usize) -> crate::Result<super::DataBatch> {
            if offset > 0 { 
                return Ok(super::DataBatch { records: vec![], batch_size: 0 }); 
            }
            Ok(super::DataBatch { records: vec![], batch_size })
        }
        
        async fn load(&self, _source: &crate::data::loader::DataSource, _format: &crate::data::loader::DataFormat) -> crate::Result<super::DataBatch> {
            Ok(super::DataBatch { records: vec![], batch_size: 0 })
        }
        
        async fn get_schema(&self, _source: &crate::data::loader::DataSource, _format: &crate::data::loader::DataFormat) -> crate::Result<crate::data::loader::types::DataSchema> {
            Ok(crate::data::loader::types::DataSchema { fields: vec![] })
        }
        
        fn name(&self) -> &'static str {
            "MockDataLoader"
        }
        
        fn set_config(&mut self, _config: crate::data::loader::LoaderConfig) {}
    }

    #[tokio::test]
    async fn test_data_batch_iterator() {
        let loader = Arc::new(MockDataLoader::new());
        let mut iterator = DataBatchIterator::new(
            loader,
            "test_data.csv".to_string(),
            10,
        );

        iterator.initialize().await.unwrap();
        
        let batch = iterator.next_batch().await.unwrap();
        assert!(batch.is_some());
        
        let stats = iterator.stats();
        assert_eq!(stats.batch_size, 10);
    }

    #[tokio::test]
    async fn test_iterator_reset() {
        let loader = Arc::new(MockDataLoader::new());
        let mut iterator = DataBatchIterator::new(
            loader,
            "test_data.csv".to_string(),
            5,
        );

        iterator.initialize().await.unwrap();
        let _ = iterator.next_batch().await.unwrap();
        
        let current_before_reset = iterator.current_index;
        iterator.reset();
        assert_eq!(iterator.current_index, 0);
        assert!(current_before_reset > 0);
    }
} 