//! 批处理工具模块
//! 
//! 提供数据批处理、批量操作、分片处理等功能。

use std::collections::VecDeque;


/// 批处理错误类型
#[derive(Debug, thiserror::Error)]
pub enum BatchError {
    #[error("批处理大小必须大于 0")]
    InvalidBatchSize,
    #[error("批处理队列为空")]
    EmptyQueue,
    #[error("批处理操作失败: {0}")]
    ProcessingError(String),
}

/// 批处理结果
pub type BatchResult<T> = std::result::Result<T, BatchError>;

/// 批处理器
pub struct BatchProcessor<T> {
    batch_size: usize,
    queue: VecDeque<T>,
    max_queue_size: usize,
}

impl<T> BatchProcessor<T> {
    /// 创建新的批处理器
    pub fn new(batch_size: usize) -> BatchResult<Self> {
        if batch_size == 0 {
            return Err(BatchError::InvalidBatchSize);
        }

        Ok(Self {
            batch_size,
            queue: VecDeque::new(),
            max_queue_size: batch_size * 100,
        })
    }

    /// 添加元素到批处理队列
    pub fn add(&mut self, item: T) -> BatchResult<()> {
        self.queue.push_back(item);
        Ok(())
    }

    /// 获取下一个批次
    pub fn next_batch(&mut self) -> Option<Vec<T>> {
        if self.queue.is_empty() {
            return None;
        }

        let mut batch = Vec::with_capacity(self.batch_size);
        for _ in 0..self.batch_size {
            if let Some(item) = self.queue.pop_front() {
                batch.push(item);
            } else {
                break;
            }
        }

        if batch.is_empty() {
            None
        } else {
            Some(batch)
        }
    }

    /// 检查是否还有待处理的元素
    pub fn has_more(&self) -> bool {
        !self.queue.is_empty()
    }

    /// 获取队列中元素数量
    pub fn queue_size(&self) -> usize {
        self.queue.len()
    }
}

/// 批量操作工具
pub struct BatchOperations;

impl BatchOperations {
    /// 批量映射操作
    pub fn batch_map<T, R, F>(
        data: Vec<T>,
        batch_size: usize,
        mapper: F,
    ) -> BatchResult<Vec<R>>
    where
        F: Fn(T) -> R,
    {
        if batch_size == 0 {
            return Err(BatchError::InvalidBatchSize);
        }

        let results: Vec<R> = data.into_iter().map(mapper).collect();
        Ok(results)
    }

    /// 批量过滤操作
    pub fn batch_filter<T, F>(
        data: Vec<T>,
        batch_size: usize,
        predicate: F,
    ) -> BatchResult<Vec<T>>
    where
        F: Fn(&T) -> bool,
    {
        if batch_size == 0 {
            return Err(BatchError::InvalidBatchSize);
        }

        let results: Vec<T> = data.into_iter().filter(predicate).collect();
        Ok(results)
    }
}

/// 数据分片器
pub struct DataShard<T> {
    data: Vec<T>,
    shard_size: usize,
    current_index: usize,
}

impl<T> DataShard<T> {
    /// 创建新的数据分片器
    pub fn new(data: Vec<T>, shard_size: usize) -> BatchResult<Self> {
        if shard_size == 0 {
            return Err(BatchError::InvalidBatchSize);
        }

        Ok(Self {
            data,
            shard_size,
            current_index: 0,
        })
    }

    /// 获取下一个分片
    pub fn next_shard(&mut self) -> Option<&[T]> {
        if self.current_index >= self.data.len() {
            return None;
        }

        let end_index = std::cmp::min(
            self.current_index + self.shard_size,
            self.data.len()
        );
        
        let shard = &self.data[self.current_index..end_index];
        self.current_index = end_index;
        
        Some(shard)
    }

    /// 重置分片器
    pub fn reset(&mut self) {
        self.current_index = 0;
    }

    /// 检查是否还有更多分片
    pub fn has_more_shards(&self) -> bool {
        self.current_index < self.data.len()
    }
} 