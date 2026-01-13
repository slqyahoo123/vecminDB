use rayon::prelude::*;
use std::sync::{Arc, Mutex};
use std::collections::{HashSet, VecDeque};
use serde_json::Value;
use crate::Result;
use std::time::{Duration, Instant};
use rayon::ThreadPoolBuilder;
use crate::Error;
use std::sync::atomic::{AtomicU8, AtomicUsize};
use std::sync::atomic::Ordering;
use std::sync::mpsc;

/// 并行处理配置
#[derive(Debug, Clone)]
pub struct ParallelConfig {
    /// 是否启用并行处理
    pub enabled: bool,
    /// 线程数量，默认为可用CPU核心数
    pub num_threads: Option<usize>,
    /// 批处理大小，默认为自动计算
    pub batch_size: Option<usize>,
    /// 是否启用工作窃取算法
    pub work_stealing: bool,
    /// 是否启用自适应批处理大小
    pub adaptive_batch_size: bool,
    /// 最小批处理大小
    pub min_batch_size: usize,
    /// 最大批处理大小
    pub max_batch_size: usize,
    /// 性能监控间隔（毫秒）
    pub monitoring_interval_ms: u64,
}

impl Default for ParallelConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            num_threads: None,
            batch_size: None,
            work_stealing: true,
            adaptive_batch_size: true,
            min_batch_size: 10,
            max_batch_size: 1000,
            monitoring_interval_ms: 1000,
        }
    }
}

/// 工作窃取任务队列
pub struct WorkStealingQueue<T> {
    queues: Vec<Arc<Mutex<VecDeque<T>>>>,
}

impl<T: Send + 'static> WorkStealingQueue<T> {
    /// 创建新的工作窃取任务队列
    pub fn new(num_threads: usize) -> Self {
        let mut queues = Vec::with_capacity(num_threads);
        for _ in 0..num_threads {
            queues.push(Arc::new(Mutex::new(VecDeque::new())));
        }
        Self { queues }
    }
    
    /// 添加任务到队列
    pub fn push(&self, thread_id: usize, task: T) {
        if let Ok(mut queue) = self.queues[thread_id % self.queues.len()].lock() {
            queue.push_back(task);
        }
    }
    
    /// 从队列中获取任务
    pub fn pop(&self, thread_id: usize) -> Option<T> {
        // 先尝试从自己的队列中获取任务
        if let Ok(mut queue) = self.queues[thread_id % self.queues.len()].lock() {
            if let Some(task) = queue.pop_front() {
                return Some(task);
            }
        }
        
        // 如果自己的队列为空，尝试从其他队列中窃取任务
        for i in 0..self.queues.len() {
            if i == thread_id % self.queues.len() {
                continue;
            }
            
            if let Ok(mut queue) = self.queues[i].lock() {
                if let Some(task) = queue.pop_back() {
                    return Some(task);
                }
            }
        }
        
        None
    }
}

impl<T: Send + 'static> Clone for WorkStealingQueue<T> {
    fn clone(&self) -> Self {
        Self {
            queues: self.queues.clone(),
        }
    }
}

/// 自适应批处理大小计算器
pub struct AdaptiveBatchSizer {
    min_batch_size: usize,
    max_batch_size: usize,
    current_batch_size: usize,
    last_processing_time: Duration,
    target_processing_time: Duration,
    adjustment_factor: f64,
}

impl AdaptiveBatchSizer {
    /// 创建新的自适应批处理大小计算器
    pub fn new(min_batch_size: usize, max_batch_size: usize) -> Self {
        Self {
            min_batch_size,
            max_batch_size,
            current_batch_size: min_batch_size,
            last_processing_time: Duration::from_millis(0),
            target_processing_time: Duration::from_millis(100), // 目标处理时间为100毫秒
            adjustment_factor: 1.2, // 调整因子
        }
    }
    
    /// 更新批处理大小
    pub fn update(&mut self, processing_time: Duration, items_processed: usize) {
        self.last_processing_time = processing_time;
        
        // 计算每项处理时间
        let time_per_item = processing_time.as_secs_f64() / items_processed as f64;
        
        // 计算理想的批处理大小
        let ideal_batch_size = (self.target_processing_time.as_secs_f64() / time_per_item) as usize;
        
        // 根据理想批处理大小调整当前批处理大小
        if ideal_batch_size > self.current_batch_size {
            self.current_batch_size = std::cmp::min(
                (self.current_batch_size as f64 * self.adjustment_factor) as usize,
                self.max_batch_size
            );
        } else if ideal_batch_size < self.current_batch_size {
            self.current_batch_size = std::cmp::max(
                (self.current_batch_size as f64 / self.adjustment_factor) as usize,
                self.min_batch_size
            );
        }
    }
    
    /// 获取当前批处理大小
    pub fn get_batch_size(&self) -> usize {
        self.current_batch_size
    }
}

/// 并行数据处理结果
pub enum ParallelResult<T> {
    /// 处理成功
    Success(T),
    /// 处理中出现错误
    Error(String),
    /// 处理被取消
    Cancelled,
}

impl<T> ParallelResult<T> {
    /// 将 Result 转换为 ParallelResult
    pub fn from_result<E: std::fmt::Display>(result: Result<T, E>) -> Self {
        match result {
            Ok(data) => ParallelResult::Success(data),
            Err(e) => ParallelResult::Error(format!("{}", e)),
        }
    }
    
    /// 将 ParallelResult 转换为 Result
    pub fn into_result<E>(self) -> Result<T, E> 
    where 
        E: From<String>
    {
        match self {
            ParallelResult::Success(data) => Ok(data),
            ParallelResult::Error(msg) => Err(E::from(msg)),
            ParallelResult::Cancelled => Err(E::from("Processing was cancelled".to_string())),
        }
    }
    
    /// 检查是否成功
    pub fn is_success(&self) -> bool {
        matches!(self, ParallelResult::Success(_))
    }
    
    /// 检查是否出错
    pub fn is_error(&self) -> bool {
        matches!(self, ParallelResult::Error(_))
    }
    
    /// 检查是否已取消
    pub fn is_cancelled(&self) -> bool {
        matches!(self, ParallelResult::Cancelled)
    }
    
    /// 获取成功结果，如果失败则返回错误
    pub fn unwrap_or_error<E: From<String>>(self) -> Result<T, E> {
        match self {
            ParallelResult::Success(data) => Ok(data),
            ParallelResult::Error(msg) => Err(E::from(msg)),
            ParallelResult::Cancelled => Err(E::from("Processing was cancelled".to_string())),
        }
    }
}

/// 并行数据处理器 - 多线程安全的数据处理工具
pub struct ParallelProcessor<T, R> {
    /// 数据分片大小
    chunk_size: usize,
    /// 最大线程数
    max_threads: usize,
    /// 处理函数
    processor: Arc<dyn Fn(Vec<T>) -> Vec<ParallelResult<R>> + Send + Sync>,
    /// 处理状态
    state: Arc<AtomicU8>,
    /// 处理进度
    progress: Arc<AtomicUsize>,
    /// 总数据量
    total: Arc<AtomicUsize>,
}

impl<T, R> ParallelProcessor<T, R>
where
    T: Send + Clone + 'static,
    R: Send + 'static,
{
    /// 创建新的并行处理器
    pub fn new<F>(processor: F) -> Self
    where
        F: Fn(Vec<T>) -> Vec<ParallelResult<R>> + Send + Sync + 'static,
    {
        Self {
            chunk_size: 1000,
            max_threads: num_cpus::get(),
            processor: Arc::new(processor),
            state: Arc::new(AtomicU8::new(0)),
            progress: Arc::new(AtomicUsize::new(0)),
            total: Arc::new(AtomicUsize::new(0)),
        }
    }

    /// 设置数据分片大小
    pub fn with_chunk_size(mut self, chunk_size: usize) -> Self {
        self.chunk_size = chunk_size;
        self
    }

    /// 设置最大线程数
    pub fn with_max_threads(mut self, max_threads: usize) -> Self {
        self.max_threads = max_threads;
        self
    }

    /// 获取处理进度
    pub fn get_progress(&self) -> (usize, usize) {
        (
            self.progress.load(Ordering::Relaxed),
            self.total.load(Ordering::Relaxed),
        )
    }

    /// 取消处理
    pub fn cancel(&self) {
        self.state.store(2, Ordering::Relaxed);
    }

    /// 处理数据
    pub fn process(&self, data: Vec<T>) -> Result<Vec<ParallelResult<R>>, Error> {
        // 如果数据为空，返回空结果
        if data.is_empty() {
            return Ok(Vec::new());
        }
        
        self.total.store(data.len(), Ordering::Relaxed);
        self.progress.store(0, Ordering::Relaxed);
        self.state.store(1, Ordering::Relaxed); // 设置为运行状态

        // 分片数据
        let chunks: Vec<Vec<T>> = data
            .chunks(self.chunk_size)
            .map(|c| {
                let mut vec = Vec::with_capacity(c.len());
                vec.extend_from_slice(c);
                vec
            })
            .collect();

        let chunk_count = chunks.len();
        let processor = Arc::clone(&self.processor);
        let state = Arc::clone(&self.state);
        let progress = Arc::clone(&self.progress);

        // 创建线程池
        let pool = ThreadPoolBuilder::new()
            .num_threads(self.max_threads.min(chunk_count))
            .build()
            .map_err(|e| Error::execution(format!("Failed to create thread pool: {}", e)))?;

        let (tx, rx) = mpsc::channel();

        // 分发任务
        for (idx, chunk) in chunks.into_iter().enumerate() {
            let processor = Arc::clone(&processor);
            let state = Arc::clone(&state);
            let progress = Arc::clone(&progress);
            let tx = tx.clone();

            pool.spawn(move || {
                // 检查是否已取消
                if state.load(Ordering::Relaxed) == 2 {
                    let _ = tx.send((idx, Vec::new()));
                    return;
                }

                // 处理分片
                let result = processor(chunk);
                
                // 更新进度
                progress.fetch_add(result.len(), Ordering::Relaxed);
                
                // 发送结果
                let _ = tx.send((idx, result));
            });
        }

        drop(tx); // 丢弃发送端以便接收端可以正确结束

        // 收集结果
        let mut results: Vec<(usize, Vec<ParallelResult<R>>)> = rx.iter().collect();
        
        // 按原始顺序排序结果
        results.sort_by_key(|(idx, _)| *idx);
        
        // 展平结果
        let flattened: Vec<ParallelResult<R>> = results
            .into_iter()
            .flat_map(|(_, res)| res)
            .collect();

        // 如果已取消，将未完成的标记为已取消
        if self.state.load(Ordering::Relaxed) == 2 {
            let result_count = flattened.len();
            let total = self.total.load(Ordering::Relaxed);
            let mut final_results = flattened;
            
            // 填充已取消的结果
            for _ in result_count..total {
                final_results.push(ParallelResult::Cancelled);
            }
            
            return Ok(final_results);
        }

        Ok(flattened)
    }
    
    /// 处理数据并直接返回结果向量
    pub fn process_direct(&self, data: Vec<T>) -> Result<Vec<R>, Error> {
        let parallel_results = self.process(data)?;
        
        // 检查是否有任何错误
        for result in &parallel_results {
            if let ParallelResult::Error(msg) = result {
                return Err(Error::execution(format!("Processing error: {}", msg)));
            } else if let ParallelResult::Cancelled = result {
                return Err(Error::execution("Processing was cancelled"));
            }
        }
        
        // 转换结果
        let results = parallel_results
            .into_iter()
            .filter_map(|r| match r {
                ParallelResult::Success(data) => Some(data),
                _ => None,
            })
            .collect();
            
        Ok(results)
    }
}

/// 并行执行多个任务
pub fn execute_in_parallel<T, F, R>(tasks: Vec<T>, executor: F) -> Result<Vec<R>, Error>
where
    T: Send + Clone + 'static,
    R: Send + 'static,
    F: Fn(T) -> Result<R, Error> + Send + Sync + 'static,
{
    // 如果任务为空，返回空结果
    if tasks.is_empty() {
        return Ok(Vec::new());
    }
    
    let executor_arc = Arc::new(executor);
    
    // 创建处理函数，将每个任务转换为 ParallelResult
    let processor = move |chunk: Vec<T>| {
        chunk
            .into_iter()
            .map(|task| {
                let executor = executor_arc.clone();
                match executor(task) {
                    Ok(result) => ParallelResult::Success(result),
                    Err(e) => ParallelResult::Error(format!("{}", e)),
                }
            })
            .collect()
    };
    
    // 创建并行处理器并处理任务
    let parallel_processor = ParallelProcessor::new(processor)
        .with_max_threads(num_cpus::get());
        
    parallel_processor.process_direct(tasks)
}

/// 非泛型的并行处理器实现
pub struct SimpleParallelProcessor {
    config: ParallelConfig,
    thread_pool: rayon::ThreadPool,
    batch_sizer: Option<AdaptiveBatchSizer>,
}

impl SimpleParallelProcessor {
    /// 创建新的并行处理器
    pub fn new(config: ParallelConfig) -> Result<Self> {
        let num_threads = config.num_threads.unwrap_or_else(|| {
            std::thread::available_parallelism().map(|p| p.get()).unwrap_or(4)
        });
        
        let thread_pool = rayon::ThreadPoolBuilder::new()
            .num_threads(num_threads)
            .build()?;
        
        let batch_sizer = if config.adaptive_batch_size {
            Some(AdaptiveBatchSizer::new(config.min_batch_size, config.max_batch_size))
        } else {
            None
        };
        
        Ok(Self {
            config,
            thread_pool,
            batch_sizer,
        })
    }
    
    /// 并行处理数据
    pub fn process<T, F, R>(&mut self, data: &[T], processor: F) -> Result<Vec<R>>
    where
        T: Sync + Clone + Send + 'static,
        R: Send + Clone + 'static,
        F: Fn(&T) -> Result<R> + Sync + Clone + Send + 'static,
    {
        let batch_size = if let Some(ref mut sizer) = self.batch_sizer {
            sizer.get_batch_size()
        } else {
            self.config.batch_size.unwrap_or_else(|| {
                std::cmp::max(1, data.len() / self.thread_pool.current_num_threads())
            })
        };
        
        let start_time = Instant::now();
        
        let results = if self.config.work_stealing {
            SimpleParallelProcessor::process_with_work_stealing(data, processor, batch_size)?
        } else {
            self.process_with_rayon(data, processor, batch_size)?
        };
        
        let processing_time = start_time.elapsed();
        
        // 更新自适应批处理大小
        if let Some(ref mut sizer) = self.batch_sizer {
            sizer.update(processing_time, data.len());
        }
        
        Ok(results)
    }
    
    /// 使用Rayon并行处理数据
    fn process_with_rayon<T, F, R>(&self, data: &[T], processor: F, batch_size: usize) -> Result<Vec<R>>
    where
        T: Sync,
        R: Send,
        F: Fn(&T) -> Result<R> + Sync,
    {
        let results = self.thread_pool.install(|| {
            data.par_chunks(batch_size)
                .map(|chunk| {
                    chunk.iter()
                        .map(|item| processor(item))
                        .collect::<Result<Vec<R>>>()
                })
                .collect::<Result<Vec<Vec<R>>>>()
        })?;
        
        Ok(results.into_iter().flatten().collect())
    }
    
    /// 使用工作窃取算法并行处理数据
    pub fn process_with_work_stealing<T, F, R>(
        data: &[T],
        processor: F,
        num_threads: usize
    ) -> Result<Vec<R>>
    where
        T: Sync + Clone + Send + 'static,
        R: Send + Clone + 'static,
        F: Fn(&T) -> Result<R> + Sync + Clone + Send + 'static,
    {
        let queue = WorkStealingQueue::new(num_threads);
        let results = Arc::new(Mutex::new(Vec::with_capacity(data.len())));
        
        // 将数据分成批次并添加到任务队列
        for (i, chunk) in data.chunks(num_threads).enumerate() {
            let chunk_data = chunk.to_vec();
            queue.push(i % num_threads, chunk_data);
        }
        
        // 创建线程处理任务
        let handles = (0..num_threads).map(|thread_id| {
            let queue = queue.clone();
            let results = results.clone();
            let processor = processor.clone();
            
            std::thread::spawn(move || {
                while let Some(chunk) = queue.pop(thread_id) {
                    let chunk_results = chunk.iter()
                        .map(|item| processor(item))
                        .collect::<Result<Vec<R>>>();
                    
                    if let Ok(chunk_results) = chunk_results {
                        if let Ok(mut results_guard) = results.lock() {
                            results_guard.extend(chunk_results);
                        }
                    }
                }
            })
        }).collect::<Vec<_>>();
        
        // 等待所有线程完成
        for handle in handles {
            handle.join().map_err(|_| Error::execution("Thread join failed"))?;
        }
        
        let final_results = if let Ok(results_guard) = results.lock() {
            results_guard.iter().cloned().collect::<Vec<_>>()
        } else {
            return Err(Error::lock("Failed to acquire lock on results"));
        };
        
        Ok(final_results)
    }
    
    /// 并行处理JSON数据
    /// 
    /// 参数：
    /// * `json_data` - JSON数据数组
    /// * `processor` - 处理函数
    /// 
    /// 返回：
    /// 处理结果向量
    pub fn process_json<F, R>(&mut self, json_data: &[Value], processor: F) -> Result<Vec<R>>
    where
        F: Fn(&Value) -> Result<R> + Send + Sync + Clone + 'static,
        R: Send + Clone + 'static,
    {
        self.process(json_data, processor)
    }
    
    /// 获取当前批处理大小
    pub fn get_current_batch_size(&self) -> usize {
        if let Some(ref sizer) = self.batch_sizer {
            sizer.get_batch_size()
        } else {
            self.config.batch_size.unwrap_or(100)
        }
    }
}

/// 初始化并行处理环境
pub fn init_parallel_env(config: &ParallelConfig) -> Result<()> {
    if config.enabled && config.num_threads.is_some() && config.num_threads.unwrap() > 0 {
        rayon::ThreadPoolBuilder::new()
            .num_threads(config.num_threads.unwrap())
            .build_global()?;
    }
    Ok(())
}

/// 获取最佳批处理大小
pub fn get_optimal_batch_size(data_size: usize, config: &ParallelConfig) -> usize {
    if !config.enabled || data_size <= 1 {
        return data_size;
    }
    
    let num_threads = config.num_threads.unwrap_or_else(|| rayon::current_num_threads());
    
    if let Some(batch_size) = config.batch_size {
        return batch_size;
    }
    
    let mut optimal_size = data_size / num_threads;
    if optimal_size < config.min_batch_size {
        optimal_size = config.min_batch_size.min(data_size);
    }
    if optimal_size > config.max_batch_size {
        optimal_size = config.max_batch_size;
    }
    
    optimal_size
}

/// 并行处理数据并收集结果
pub fn parallel_process<T, F, R>(data: &[T], process_fn: F, config: &ParallelConfig) -> Vec<R>
where
    T: Sync,
    F: Fn(&T) -> R + Sync + Send + Clone,
    R: Send,
{
    if !config.enabled || data.len() <= 1 {
        return data.iter().map(|item| process_fn(item)).collect();
    }
    
    let batch_size = get_optimal_batch_size(data.len(), config);
    
    data.par_chunks(batch_size)
        .flat_map(|chunk| {
            let process_fn = process_fn.clone();
            chunk.iter().map(move |item| process_fn(item)).collect::<Vec<_>>()
        })
        .collect()
}

/// 并行处理数据并合并结果
pub fn parallel_reduce<T, F, R, M>(data: &[T], process_fn: F, merge_fn: M, config: &ParallelConfig) -> Result<R>
where
    T: Sync,
    F: Fn(&T) -> R + Sync + Send + Clone,
    R: Send + Clone,
    M: Fn(R, R) -> R + Sync + Send + Clone,
{
    if data.is_empty() {
        return Err(Error::data("并行处理失败: 数据为空".to_string()));
    }
    
    if !config.enabled || data.len() <= 1 {
        return data.iter().map(|item| process_fn(item)).reduce(|a, b| merge_fn(a, b))
            .ok_or_else(|| Error::data("并行处理失败: 数据处理过程中出现错误".to_string()));
    }
    
    let batch_size = get_optimal_batch_size(data.len(), config);
    let process_fn_clone = process_fn.clone();
    
    // 确保至少有一个元素可用于初始化
    if data.is_empty() {
        return Err(Error::data("并行处理失败: 数据为空".to_string()));
    }
    
    Ok(data.par_chunks(batch_size)
        .map(|chunk| {
            let process_fn = process_fn.clone();
            let merge_fn = merge_fn.clone();
            chunk.iter()
                .map(move |item| process_fn(item))
                .reduce(|a, b| merge_fn(a, b))
                .ok_or_else(|| Error::data("并行处理失败: 区块数据处理出错".to_string()))
                .expect("并行处理失败: 区块不应为空") // 安全，因为par_chunks保证每个chunk至少有一个元素
        })
        .reduce(|| process_fn_clone(&data[0]), |a, b| merge_fn(a, b)))
}

/// 并行收集JSON数据中的唯一值
pub fn parallel_collect_unique_values(data: &[Value], field: &str, config: &ParallelConfig) -> HashSet<String> {
    if !config.enabled || data.len() <= 1 {
        let mut result = HashSet::new();
        for item in data {
            if let Value::Object(obj) = item {
                if let Some(Value::String(s)) = obj.get(field) {
                    result.insert(s.clone());
                }
            }
        }
        return result;
    }
    
    let batch_size = get_optimal_batch_size(data.len(), config);
    let field = field.to_string();
    
    data.par_chunks(batch_size)
        .map(|chunk| {
            let mut local_set = HashSet::new();
            for item in chunk {
                if let Value::Object(obj) = item {
                    if let Some(Value::String(s)) = obj.get(&field) {
                        local_set.insert(s.clone());
                    }
                }
            }
            local_set
        })
        .reduce(HashSet::new, |mut a, b| {
            a.extend(b);
            a
        })
}

/// 并行计算数值统计信息
pub fn parallel_compute_numeric_stats(data: &[Value], field: &str, config: &ParallelConfig) -> (f64, f64, f64, f64, usize) {
    if !config.enabled || data.len() <= 1 {
        let mut sum = 0.0;
        let mut count = 0;
        let mut min = f64::MAX;
        let mut max = f64::MIN;
        
        for item in data {
            if let Value::Object(obj) = item {
                if let Some(value) = obj.get(field) {
                    if let Some(num) = extract_numeric_value(value) {
                        sum += num;
                        count += 1;
                        min = min.min(num);
                        max = max.max(num);
                    }
                }
            }
        }
        
        let mean = if count > 0 { sum / count as f64 } else { 0.0 };
        let mut sum_squared_diff = 0.0;
        
        for item in data {
            if let Value::Object(obj) = item {
                if let Some(value) = obj.get(field) {
                    if let Some(num) = extract_numeric_value(value) {
                        let diff = num - mean;
                        sum_squared_diff += diff * diff;
                    }
                }
            }
        }
        
        let std_dev = if count > 1 {
            (sum_squared_diff / (count - 1) as f64).sqrt()
        } else {
            0.0
        };
        
        return (min, max, mean, std_dev, count);
    }
    
    let batch_size = get_optimal_batch_size(data.len(), config);
    let field = field.to_string();
    
    // 第一遍：计算最小值、最大值、总和和计数
    let (min, max, sum, count) = data.par_chunks(batch_size)
        .map(|chunk| {
            let mut local_sum = 0.0;
            let mut local_count = 0;
            let mut local_min = f64::MAX;
            let mut local_max = f64::MIN;
            
            for item in chunk {
                if let Value::Object(obj) = item {
                    if let Some(value) = obj.get(&field) {
                        if let Some(num) = extract_numeric_value(value) {
                            local_sum += num;
                            local_count += 1;
                            local_min = local_min.min(num);
                            local_max = local_max.max(num);
                        }
                    }
                }
            }
            
            (local_min, local_max, local_sum, local_count)
        })
        .reduce(
            || (f64::MAX, f64::MIN, 0.0, 0),
            |(min1, max1, sum1, count1), (min2, max2, sum2, count2)| {
                (min1.min(min2), max1.max(max2), sum1 + sum2, count1 + count2)
            }
        );
    
    let mean = if count > 0 { sum / count as f64 } else { 0.0 };
    
    // 第二遍：计算方差和标准差
    let sum_squared_diff = data.par_chunks(batch_size)
        .map(|chunk| {
            let mut local_sum_squared_diff = 0.0;
            
            for item in chunk {
                if let Value::Object(obj) = item {
                    if let Some(value) = obj.get(&field) {
                        if let Some(num) = extract_numeric_value(value) {
                            let diff = num - mean;
                            local_sum_squared_diff += diff * diff;
                        }
                    }
                }
            }
            
            local_sum_squared_diff
        })
        .reduce(|| 0.0, |a, b| a + b);
    
    let std_dev = if count > 1 {
        (sum_squared_diff / (count - 1) as f64).sqrt()
    } else {
        0.0
    };
    
    (min, max, mean, std_dev, count)
}

/// 从JSON值中提取数值
fn extract_numeric_value(value: &Value) -> Option<f64> {
    match value {
        Value::Number(n) => n.as_f64(),
        Value::String(s) => s.parse::<f64>().ok(),
        Value::Bool(b) => Some(if *b { 1.0 } else { 0.0 }),
        _ => None,
    }
}

/// 并行处理数据批次
pub fn process_batch_parallel<F, T, R>(
    batch: &[T],
    worker_func: F,
    num_workers: usize,
) -> Result<Vec<R>, Error>
where
    F: Fn(&T) -> Result<R, Error> + Send + Sync,
    T: Send + Sync,
    R: Send,
{
    if batch.is_empty() {
        log::warn!("并行处理的批次为空");
        return Ok(Vec::new());
    }

    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(num_workers)
        .build()
        .map_err(|e| Error::execution(format!("创建线程池失败: {}", e)))?;

    let results: Result<Vec<_>, _> = pool.install(|| {
        batch
            .par_iter()
            .map(|item| worker_func(item))
            .collect()
    });

    results
}

/// 并行映射操作
pub fn parallel_map<T, R, F>(items: &[T], f: F, num_workers: Option<usize>) -> Result<Vec<R>, Error>
where
    T: Send + Sync,
    R: Send,
    F: Fn(&T) -> Result<R, Error> + Send + Sync,
{
    let workers = num_workers.unwrap_or_else(|| {
        // 默认使用逻辑核心数
        num_cpus::get()
    });

    if items.is_empty() {
        log::warn!("并行映射的数据为空");
        return Ok(Vec::new());
    }

    log::debug!("使用 {} 个工作线程进行并行映射", workers);

    process_batch_parallel(items, f, workers)
} 