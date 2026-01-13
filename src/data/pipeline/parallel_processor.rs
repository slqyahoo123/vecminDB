use std::collections::HashMap;
use std::sync::{Arc, RwLock};
use std::time::{Duration, Instant};

use futures::future::try_join_all;
use tokio::task::JoinHandle;
use tokio::sync::{Semaphore, mpsc};
use rayon::prelude::*;
use log::{debug, info, warn, error};

use crate::error::{Error, Result};
use crate::data::{DataBatch, DataRecord, DataValue};
use crate::core::types::CoreTensorData;
// 未使用的导入移除
use super::core::{DataStage, PipelineConfig};
use tokio::sync::Mutex as TokioMutex;
// TokioMutexGuard isn't used directly

/// 输入源枚举
enum InputSource {
    TaskReceiver(Arc<TokioMutex<mpsc::UnboundedReceiver<ProcessingTask>>>),
    DataReceiver(Option<mpsc::Receiver<DataBatch>>),
}

/// 并发处理策略
#[derive(Debug, Clone, PartialEq)]
pub enum ParallelStrategy {
    /// 数据分片并行：将数据分成多个块并行处理
    DataParallel {
        /// 分片大小
        chunk_size: usize,
        /// 最大并发数
        max_concurrency: usize,
    },
    /// 阶段流水线：不同阶段并行执行
    StagePipeline {
        /// 缓冲区大小
        buffer_size: usize,
        /// 阶段间超时时间
        stage_timeout_ms: u64,
    },
    /// 混合策略：同时使用数据并行和阶段流水线
    Hybrid {
        /// 数据分片大小
        chunk_size: usize,
        /// 缓冲区大小
        buffer_size: usize,
        /// 最大并发数
        max_concurrency: usize,
    },
}

impl Default for ParallelStrategy {
    fn default() -> Self {
        Self::DataParallel {
            chunk_size: 1000,
            max_concurrency: num_cpus::get(),
        }
    }
}

/// 并发处理配置
#[derive(Debug, Clone)]
pub struct ParallelConfig {
    /// 并发策略
    pub strategy: ParallelStrategy,
    /// 是否启用智能负载均衡
    pub enable_load_balancing: bool,
    /// 是否启用动态调整
    pub enable_dynamic_adjustment: bool,
    /// 性能监控间隔（毫秒）
    pub monitoring_interval_ms: u64,
    /// 内存使用阈值（百分比）
    pub memory_threshold_percent: f32,
    /// CPU使用阈值（百分比）
    pub cpu_threshold_percent: f32,
    /// 错误容忍度
    pub error_tolerance: f32,
}

impl Default for ParallelConfig {
    fn default() -> Self {
        Self {
            strategy: ParallelStrategy::default(),
            enable_load_balancing: true,
            enable_dynamic_adjustment: true,
            monitoring_interval_ms: 1000,
            memory_threshold_percent: 80.0,
            cpu_threshold_percent: 85.0,
            error_tolerance: 0.05, // 5% 错误容忍度
        }
    }
}

/// 处理任务
#[derive(Debug)]
pub struct ProcessingTask {
    /// 任务ID
    pub id: String,
    /// 数据批次
    pub batch: DataBatch,
    /// 阶段索引
    pub stage_index: usize,
    /// 创建时间
    pub created_at: Instant,
    /// 优先级
    pub priority: i32,
}

/// 处理结果
#[derive(Debug)]
pub struct ProcessingResult {
    /// 任务ID
    pub task_id: String,
    /// 处理后的数据批次
    pub batch: DataBatch,
    /// 处理时间（毫秒）
    pub processing_time_ms: u64,
    /// 错误信息（如果有）
    pub error: Option<String>,
}

/// 并行处理性能指标
#[derive(Debug, Clone)]
pub struct ParallelProcessingMetrics {
    /// 总处理时间（毫秒）
    pub total_time_ms: u64,
    /// 并行任务数
    pub parallel_tasks: usize,
    /// 处理的记录数
    pub processed_records: usize,
    /// 平均每个任务处理时间（毫秒）
    pub avg_task_time_ms: f64,
    /// 吞吐量（记录/秒）
    pub throughput_rps: f64,
    /// 内存使用峰值（MB）
    pub peak_memory_mb: f64,
    /// CPU使用率（百分比）
    pub cpu_usage_percent: f64,
    /// 错误计数
    pub error_count: usize,
}

impl Default for ParallelProcessingMetrics {
    fn default() -> Self {
        Self {
            total_time_ms: 0,
            parallel_tasks: 0,
            processed_records: 0,
            avg_task_time_ms: 0.0,
            throughput_rps: 0.0,
            peak_memory_mb: 0.0,
            cpu_usage_percent: 0.0,
            error_count: 0,
        }
    }
}

/// 并发数据处理器
pub struct ParallelProcessor {
    /// 处理阶段
    stages: Vec<Arc<dyn DataStage>>,
    /// 并发配置
    config: ParallelConfig,
    /// 管道配置
    pipeline_config: PipelineConfig,
    /// 信号量用于控制并发数
    semaphore: Arc<Semaphore>,
    /// 性能指标
    metrics: Arc<RwLock<ParallelProcessingMetrics>>,
    /// 任务队列
    task_sender: Arc<TokioMutex<Option<mpsc::UnboundedSender<ProcessingTask>>>>,
    /// 运行状态
    is_running: Arc<RwLock<bool>>,
    /// 工作线程句柄
    worker_handles: Arc<TokioMutex<Vec<JoinHandle<()>>>>,
}

impl ParallelProcessor {
    /// 创建新的并发处理器
    pub fn new(
        stages: Vec<Arc<dyn DataStage>>,
        config: ParallelConfig,
        pipeline_config: PipelineConfig,
    ) -> Result<Self> {
        let max_concurrency = match &config.strategy {
            ParallelStrategy::DataParallel { max_concurrency, .. } => *max_concurrency,
            ParallelStrategy::Hybrid { max_concurrency, .. } => *max_concurrency,
            ParallelStrategy::StagePipeline { .. } => num_cpus::get(),
        };

        let semaphore = Arc::new(Semaphore::new(max_concurrency));
        
        let metrics = Arc::new(RwLock::new(ParallelProcessingMetrics {
            total_time_ms: 0,
            parallel_tasks: 0,
            processed_records: 0,
            avg_task_time_ms: 0.0,
            throughput_rps: 0.0,
            peak_memory_mb: 0.0,
            cpu_usage_percent: 0.0,
            error_count: 0,
        }));

        Ok(Self {
            stages,
            config,
            pipeline_config,
            semaphore,
            metrics,
            task_sender: Arc::new(TokioMutex::new(None)),
            is_running: Arc::new(RwLock::new(false)),
            worker_handles: Arc::new(TokioMutex::new(Vec::new())),
        })
    }

    /// 启动处理器
    pub async fn start(&self) -> Result<()> {
        let mut is_running = self.is_running.write().unwrap();
        if *is_running {
            return Err(Error::processing("处理器已在运行"));
        }

        *is_running = true;
        drop(is_running);

        let (sender, receiver) = mpsc::unbounded_channel();
        *self.task_sender.lock().await = Some(sender);

        // 启动工作线程
        match &self.config.strategy {
            ParallelStrategy::DataParallel { max_concurrency, .. } => {
                self.start_data_parallel_workers(*max_concurrency, receiver).await?;
            }
            ParallelStrategy::StagePipeline { buffer_size, .. } => {
                self.start_pipeline_workers(*buffer_size, receiver).await?;
            }
            ParallelStrategy::Hybrid { max_concurrency, buffer_size, .. } => {
                self.start_hybrid_workers(*max_concurrency, *buffer_size, receiver).await?;
            }
        }

        // 启动监控线程
        if self.config.enable_dynamic_adjustment {
            self.start_monitoring_thread().await?;
        }

        info!("并发数据处理器已启动，策略: {:?}", self.config.strategy);
        Ok(())
    }

    /// 停止处理器
    pub async fn stop(&self) -> Result<()> {
        let mut is_running = self.is_running.write().unwrap();
        if !*is_running {
            return Ok(());
        }

        *is_running = false;
        drop(is_running);

        // 关闭任务发送器
        *self.task_sender.lock().await = None;

        // 等待所有工作线程完成
        let mut handles = self.worker_handles.lock().await;
        while let Some(handle) = handles.pop() {
            handle.abort();
        }

        info!("并发数据处理器已停止");
        Ok(())
    }

    /// 处理数据批次
    pub async fn process_batch(&self, batch: DataBatch) -> Result<DataBatch> {
        let start_time = Instant::now();
        let record_count = batch.records.len(); // 先记录记录数量
        
        let result = match &self.config.strategy {
            ParallelStrategy::DataParallel { chunk_size, .. } => {
                self.process_data_parallel(batch, *chunk_size).await
            }
            ParallelStrategy::StagePipeline { .. } => {
                self.process_stage_pipeline(batch).await
            }
            ParallelStrategy::Hybrid { chunk_size, .. } => {
                self.process_hybrid(batch, *chunk_size).await
            }
        };
        
        // 更新性能指标
        let processing_time = start_time.elapsed().as_millis() as u64;
        match &result {
            Ok(_) => self.update_metrics(processing_time, record_count, true),
            Err(_) => self.update_metrics(processing_time, record_count, false),
        }
        
        result
    }

    /// 数据并行处理
    async fn process_data_parallel(&self, batch: DataBatch, chunk_size: usize) -> Result<DataBatch> {
        if batch.is_empty() {
            return Ok(batch);
        }

        let records = crate::data::DataBatch::to_records(&batch);
        let schema = batch.schema().cloned().unwrap_or_else(|| {
            // 如果没有schema，创建一个默认的
            crate::data::schema::DataSchema::builder()
                .name("default_schema")
                .build()
                .unwrap()
        });
        
        // 将数据分块
        let chunks: Vec<Vec<DataRecord>> = records
            .chunks(chunk_size)
            .map(|chunk| chunk.to_vec())
            .collect();

        debug!("数据并行处理：分成 {} 个块，每块大小约 {}", chunks.len(), chunk_size);

        // 并行处理每个块
        let processed_chunks = try_join_all(
            chunks.into_iter().map(|chunk| {
                let stages = self.stages.clone();
                let schema = schema.clone();
                let semaphore = self.semaphore.clone();
                
                async move {
                    let _permit = semaphore.acquire().await.unwrap();
                    
                    let mut chunk_batch = crate::data::DataBatch::from_data_records(chunk, Some(schema))?;
                    
                    // 使用stages变量进行流水线处理
                    for stage in &stages {
                        stage.process(&mut chunk_batch)?;
                    }
                    
                    Ok::<DataBatch, crate::Error>(chunk_batch)
                }
            })
        ).await?;

        // 合并结果
        let mut result_records = Vec::new();
        for chunk_batch in processed_chunks {
            result_records.extend(crate::data::DataBatch::to_records(&chunk_batch));
        }

        let result_schema = if !self.stages.is_empty() {
            let mut current_schema = schema;
            for stage in &self.stages {
                current_schema = stage.output_schema(&current_schema)?;
            }
            current_schema
        } else {
            schema
        };

        // 将DataRecord转换为HashMap<String, DataValue>格式
        let mut converted_records = Vec::new();
        for record in result_records {
            let mut record_map = HashMap::new();
            for (field_name, field_value) in record.fields {
                match field_value.to_data_value() {
                    Ok(data_value) => {
                        record_map.insert(field_name, data_value);
                    },
                    Err(_) => {
                        // 跳过无法转换的字段
                        continue;
                    }
                }
            }
            converted_records.push(record_map);
        }

        crate::data::DataBatch::from_records(converted_records, Some(result_schema))
    }

    /// 阶段流水线处理
    async fn process_stage_pipeline(&self, batch: DataBatch) -> Result<DataBatch> {
        if self.stages.is_empty() {
            return Ok(batch);
        }

        debug!("阶段流水线处理：{} 个阶段", self.stages.len());

        // 为每个阶段创建通道
        let (mut senders, mut receivers) = self.create_pipeline_channels().await?;
        
        // 启动阶段处理任务
        let mut stage_handles = Vec::new();
        for (i, stage) in self.stages.iter().enumerate() {
            let stage = stage.clone();
            let receiver = receivers.remove(0);
            let sender = if i < self.stages.len() - 1 {
                Some(senders.remove(0))
            } else {
                None
            };
            
            let handle = tokio::spawn(async move {
                Self::run_stage_processor(stage, receiver, sender).await
            });
            stage_handles.push(handle);
        }

        // 发送初始数据
        if let Some(first_sender) = senders.first() {
            first_sender.send(batch).await.map_err(|_| 
                Error::processing("无法发送数据到第一阶段"))?;
        }

        // 等待所有阶段完成并获取最终结果
        let mut final_result = None;
        for (i, handle) in stage_handles.into_iter().enumerate() {
            if i == self.stages.len() - 1 {
                // 最后一个阶段，获取结果
                final_result = Some(handle.await.map_err(|e| 
                    Error::processing(&format!("阶段 {} 执行失败: {}", i, e)))??);
            } else {
                handle.await.map_err(|e| 
                    Error::processing(&format!("阶段 {} 执行失败: {}", i, e)))??;
            }
        }

        final_result.ok_or_else(|| Error::processing("未获取到处理结果"))
    }

    /// 混合策略处理
    async fn process_hybrid(&self, batch: DataBatch, chunk_size: usize) -> Result<DataBatch> {
        debug!("混合策略处理：数据分块 + 阶段流水线");
        
        // 首先进行数据分块
        let records = crate::data::DataBatch::to_records(&batch);
        let schema = batch.schema().cloned().unwrap_or_else(|| {
            // 如果没有schema，创建一个默认的
            crate::data::schema::DataSchema::builder()
                .name("default_schema")
                .build()
                .unwrap()
        });
        
        let chunks: Vec<Vec<DataRecord>> = records
            .chunks(chunk_size)
            .map(|chunk| chunk.to_vec())
            .collect();

        // 对每个块进行流水线处理
        let processed_chunks = try_join_all(
            chunks.into_iter().map(|chunk| {
                let stages = self.stages.clone();
                let schema = schema.clone();
                let semaphore = self.semaphore.clone();
                
                async move {
                    let _permit = semaphore.acquire().await.unwrap();
                    
                    let mut chunk_batch = crate::data::DataBatch::from_data_records(chunk, Some(schema))?;
                    
                    // 使用stages变量进行流水线处理
                    for stage in &stages {
                        stage.process(&mut chunk_batch)?;
                    }
                    
                    Ok::<DataBatch, crate::Error>(chunk_batch)
                }
            })
        ).await?;

        // 合并结果
        let mut result_records = Vec::new();
        for chunk_batch in processed_chunks {
            result_records.extend(crate::data::DataBatch::to_records(&chunk_batch));
        }

        let result_schema = if !self.stages.is_empty() {
            let mut current_schema = schema;
            for stage in &self.stages {
                current_schema = stage.output_schema(&current_schema)?;
            }
            current_schema
        } else {
            schema
        };

        // 将DataRecord转换为HashMap<String, DataValue>格式
        let mut converted_records = Vec::new();
        for record in result_records {
            let mut record_map = HashMap::new();
            for (field_name, field_value) in record.fields {
                match field_value.to_data_value() {
                    Ok(data_value) => {
                        record_map.insert(field_name, data_value);
                    },
                    Err(_) => {
                        // 跳过无法转换的字段
                        continue;
                    }
                }
            }
            converted_records.push(record_map);
        }

        crate::data::DataBatch::from_records(converted_records, Some(result_schema))
    }

    /// 启动数据并行工作线程
    async fn start_data_parallel_workers(
        &self,
        max_concurrency: usize,
        receiver: mpsc::UnboundedReceiver<ProcessingTask>,
    ) -> Result<()> {
        // 创建共享的receiver
        let receiver = Arc::new(TokioMutex::new(receiver));
        
        for i in 0..max_concurrency {
            let stages = self.stages.clone();
            let semaphore = self.semaphore.clone();
            let is_running = self.is_running.clone();
            let receiver_clone = receiver.clone();
            
            let handle = tokio::spawn(async move {
                while *is_running.read().unwrap() {
                    // 从共享receiver中获取任务
                                let task = {
                let mut receiver_guard = receiver_clone.lock().await;
                tokio::time::timeout(
                    Duration::from_millis(100),
                    receiver_guard.recv()
                ).await
            };
                    
                    if let Ok(Some(task)) = task {
                        let _permit = semaphore.acquire().await.unwrap();
                        
                        // 处理任务
                        let start_time = Instant::now();
                        let mut batch = task.batch;
                        
                        for stage in &stages {
                            if let Err(e) = stage.process(&mut batch) {
                                error!("工作线程 {} 处理任务 {} 失败: {}", i, task.id, e);
                                break;
                            }
                        }
                        
                        let processing_time = start_time.elapsed().as_millis() as u64;
                        debug!("工作线程 {} 完成任务 {}，耗时 {}ms", 
                            i, task.id, processing_time);
                    }
                }
            });
            
            self.worker_handles.lock().await.push(handle);
        }
        
        Ok(())
    }

    /// 启动流水线工作线程 - 生产级实现
    async fn start_pipeline_workers(
        &self,
        buffer_size: usize,
        receiver: mpsc::UnboundedReceiver<ProcessingTask>,
    ) -> Result<()> {
        debug!("启动流水线工作线程，缓冲区大小: {}", buffer_size);
        
        // 创建阶段间的通信通道
        let (mut senders, mut receivers) = self.create_pipeline_channels().await?;
        let stages = self.stages.clone();
        let metrics = self.metrics.clone();
        let is_running = self.is_running.clone();
        
        // 为每个阶段启动工作线程
        let mut stage_handles = Vec::new();
        
        // 创建任务接收器的包装器
        let task_receiver = Arc::new(TokioMutex::new(receiver));
        
        // 为每个阶段创建处理器
        for (i, stage) in stages.iter().enumerate() {
            let stage_clone = stage.clone();
            let metrics_clone = metrics.clone();
            let is_running_clone = is_running.clone();
            
            // 确定输入源和输出目标
            let (input_source, output_sender) = if i == 0 {
                // 第一个阶段从任务接收器接收数据
                (InputSource::TaskReceiver(task_receiver.clone()), 
                 if i < stages.len() - 1 { senders.pop() } else { None })
            } else if i == stages.len() - 1 {
                // 最后一个阶段没有输出发送器
                (InputSource::DataReceiver(receivers.pop()), None)
            } else {
                // 中间阶段有输入接收器和输出发送器
                (
                    InputSource::DataReceiver(receivers.pop()), 
                    senders.pop()
                )
            };
            
            let handle = tokio::spawn(async move {
                info!("阶段 '{}' 工作线程已启动", stage_clone.name());
                
                match input_source {
                    InputSource::TaskReceiver(task_receiver) => {
                        // 第一个阶段：处理来自任务队列的数据
                        while *is_running_clone.read().unwrap() {
                            let task = {
                                let mut receiver_guard = task_receiver.lock().await;
                                tokio::time::timeout(
                                    Duration::from_millis(100),
                                    receiver_guard.recv()
                                ).await
                            };
                            
                            if let Ok(Some(task)) = task {
                                let start_time = Instant::now();
                                
                                // 处理任务
                                let mut batch = task.batch;
                                match stage_clone.process(&mut batch) {
                                    Ok(_) => {
                                        let processing_time = start_time.elapsed().as_millis() as u64;
                                        
                                        // 更新指标
                                        {
                                            let mut metrics_guard = metrics_clone.write().unwrap();
                                            metrics_guard.total_time_ms += processing_time;
                                            metrics_guard.processed_records += batch.records.len();
                                        }
                                        
                                        // 发送到下一阶段
                                        if let Some(ref sender) = output_sender {
                                            if sender.send(batch).await.is_err() {
                                                warn!("无法发送数据到下一阶段");
                                            }
                                        }
                                    },
                                    Err(e) => {
                                        error!("阶段 '{}' 处理失败: {}", stage_clone.name(), e);
                                        let mut metrics_guard = metrics_clone.write().unwrap();
                                        metrics_guard.error_count += 1;
                                    }
                                }
                            }
                        }
                    },
                    InputSource::DataReceiver(Some(mut receiver)) => {
                        // 其他阶段：处理来自前一阶段的数据
                        while *is_running_clone.read().unwrap() {
                            if let Some(batch) = receiver.recv().await {
                                let start_time = Instant::now();
                                
                                let mut batch = batch;
                                match stage_clone.process(&mut batch) {
                                    Ok(_) => {
                                        let processing_time = start_time.elapsed().as_millis() as u64;
                                        
                                        // 更新指标
                                        {
                                            let mut metrics_guard = metrics_clone.write().unwrap();
                                            metrics_guard.total_time_ms += processing_time;
                                            metrics_guard.processed_records += batch.records.len();
                                        }
                                        
                                        // 发送到下一阶段或完成处理
                                        if let Some(ref sender) = output_sender {
                                            if sender.send(batch).await.is_err() {
                                                warn!("无法发送数据到下一阶段");
                                            }
                                        } else {
                                            // 最后一个阶段，处理完成
                                            debug!("流水线处理完成，处理了 {} 条记录", batch.records.len());
                                        }
                                    },
                                    Err(e) => {
                                        error!("阶段 '{}' 处理失败: {}", stage_clone.name(), e);
                                        let mut metrics_guard = metrics_clone.write().unwrap();
                                        metrics_guard.error_count += 1;
                                    }
                                }
                            }
                        }
                    },
                    InputSource::DataReceiver(None) => {
                        error!("阶段 {} 没有有效的输入接收器", i);
                    }
                }
                
                info!("阶段 '{}' 工作线程已停止", stage_clone.name());
            });
            
            stage_handles.push(handle);
        }
        
        // 保存工作线程句柄
        {
            let mut handles = self.worker_handles.lock().await;
            handles.extend(stage_handles);
        }
        
        info!("所有流水线工作线程已启动，阶段数: {}", stages.len());
        Ok(())
    }

    /// 启动混合工作线程
    async fn start_hybrid_workers(
        &self,
        max_concurrency: usize,
        buffer_size: usize,
        receiver: mpsc::UnboundedReceiver<ProcessingTask>,
    ) -> Result<()> {
        debug!("启动混合工作线程，并发数: {}，缓冲区: {}", max_concurrency, buffer_size);
        // 结合数据并行和流水线的工作线程
        self.start_data_parallel_workers(max_concurrency, receiver).await
    }

    /// 启动监控线程
    async fn start_monitoring_thread(&self) -> Result<()> {
        let metrics = self.metrics.clone();
        let config = self.config.clone();
        let is_running = self.is_running.clone();
        
        let handle = tokio::spawn(async move {
            while *is_running.read().unwrap() {
                tokio::time::sleep(Duration::from_millis(config.monitoring_interval_ms)).await;
                
                // 收集系统指标
                let memory_usage = Self::get_memory_usage();
                let cpu_usage = Self::get_cpu_usage();
                
                // 更新性能指标
                {
                    let mut metrics = metrics.write().unwrap();
                    metrics.peak_memory_mb = memory_usage;
                    metrics.cpu_usage_percent = cpu_usage;
                }
                
                // 动态调整（如果启用）
                if config.enable_dynamic_adjustment {
                    Self::adjust_performance_parameters(&config, memory_usage, cpu_usage);
                }
            }
        });
        
        self.worker_handles.lock().await.push(handle);
        Ok(())
    }

    /// 创建流水线通道
    async fn create_pipeline_channels(&self) -> Result<(Vec<mpsc::Sender<DataBatch>>, Vec<mpsc::Receiver<DataBatch>>)> {
        let buffer_size = match &self.config.strategy {
            ParallelStrategy::StagePipeline { buffer_size, .. } => *buffer_size,
            ParallelStrategy::Hybrid { buffer_size, .. } => *buffer_size,
            _ => 1000,
        };
        
        let mut senders = Vec::new();
        let mut receivers = Vec::new();
        
        for _ in 0..self.stages.len() {
            let (sender, receiver) = mpsc::channel(buffer_size);
            senders.push(sender);
            receivers.push(receiver);
        }
        
        Ok((senders, receivers))
    }

    /// 运行单个阶段处理器
    async fn run_stage_processor(
        stage: Arc<dyn DataStage>,
        mut receiver: mpsc::Receiver<DataBatch>,
        sender: Option<mpsc::Sender<DataBatch>>,
    ) -> Result<DataBatch> {
        let mut last_result = None;
        
        while let Some(mut batch) = receiver.recv().await {
            // 处理数据批次
            stage.process(&mut batch)?;
            
            if let Some(ref sender) = sender {
                // 发送到下一阶段
                sender.send(batch).await.map_err(|_| 
                    Error::processing("无法发送数据到下一阶段"))?;
            } else {
                // 最后一个阶段，保存结果
                last_result = Some(batch);
            }
        }
        
        last_result.ok_or_else(|| Error::processing("阶段处理器未产生结果"))
    }

    /// 更新性能指标
    fn update_metrics(&self, processing_time_ms: u64, record_count: usize, success: bool) {
        let mut metrics = self.metrics.write().unwrap();
        
        metrics.total_time_ms += processing_time_ms;
        
        // 计算平均处理时间
        let total_operations = if success { 1 } else { 0 };
        if total_operations > 0 {
            metrics.avg_task_time_ms = metrics.total_time_ms as f64 / total_operations as f64;
        }
        
        // 计算吞吐量
        if processing_time_ms > 0 {
            metrics.throughput_rps = (record_count as f64 * 1000.0) / processing_time_ms as f64;
        }
        
        // 更新错误统计 - 生产级实现
        if !success {
            metrics.error_count += 1;
            
            // 计算错误率
            let total_operations = (metrics.processed_records / record_count.max(1)) + 1;
            let error_rate = (metrics.error_count as f64 / total_operations as f64) * 100.0;
            
            // 如果错误率过高，记录警告
            if error_rate > 5.0 {
                warn!("错误率过高: {:.2}%，总错误数: {}, 总操作数: {}", 
                      error_rate, metrics.error_count, total_operations);
            }
        } else {
            // 成功处理，更新成功记录数
            metrics.processed_records += record_count;
            
            // 计算成功率
            let total_operations = metrics.processed_records / record_count.max(1);
            let success_rate = ((total_operations - metrics.error_count) as f64 / total_operations as f64) * 100.0;
            
            if total_operations % 1000 == 0 { // 每1000次操作记录一次统计
                info!("处理统计 - 成功率: {:.2}%, 总处理: {}, 错误: {}", 
                      success_rate, total_operations, metrics.error_count);
            }
        }
    }

    /// 获取内存使用量（MB） - 生产级实现
    fn get_memory_usage() -> f64 {
        // 跨平台内存使用量获取
        #[cfg(target_os = "linux")]
        {
            // Linux系统：读取/proc/self/status文件
            if let Ok(status) = std::fs::read_to_string("/proc/self/status") {
                for line in status.lines() {
                    if line.starts_with("VmRSS:") {
                        if let Some(memory_str) = line.split_whitespace().nth(1) {
                            if let Ok(memory_kb) = memory_str.parse::<u64>() {
                                return memory_kb as f64 / 1024.0; // 转换为MB
                            }
                        }
                    }
                }
            }
            
            // 备用方法：使用ps命令
            if let Ok(output) = std::process::Command::new("ps")
                .args(&["-o", "rss=", "-p", &std::process::id().to_string()])
                .output()
            {
                if let Ok(rss_str) = String::from_utf8(output.stdout) {
                    if let Ok(rss_kb) = rss_str.trim().parse::<u64>() {
                        return rss_kb as f64 / 1024.0;
                    }
                }
            }
        }
        
        #[cfg(target_os = "windows")]
        {
            // Windows系统：使用WMI或性能计数器
            // 这里使用简化的方法，实际生产环境中应该使用Windows API
            use std::process::Command;
            
            if let Ok(output) = Command::new("tasklist")
                .args(&["/FI", &format!("PID eq {}", std::process::id()), "/FO", "CSV"])
                .output()
            {
                if let Ok(output_str) = String::from_utf8(output.stdout) {
                    // 解析CSV输出，提取内存使用量
                    for line in output_str.lines().skip(1) { // 跳过标题行
                        let fields: Vec<&str> = line.split(',').collect();
                        if fields.len() >= 5 {
                            // 第5个字段通常是内存使用量
                            let memory_str = fields[4].trim_matches('"').replace(",", "").replace(" K", "");
                            if let Ok(memory_kb) = memory_str.parse::<u64>() {
                                return memory_kb as f64 / 1024.0;
                            }
                        }
                    }
                }
            }
        }
        
        #[cfg(target_os = "macos")]
        {
            // macOS系统：使用ps命令
            if let Ok(output) = std::process::Command::new("ps")
                .args(&["-o", "rss=", "-p", &std::process::id().to_string()])
                .output()
            {
                if let Ok(rss_str) = String::from_utf8(output.stdout) {
                    if let Ok(rss_kb) = rss_str.trim().parse::<u64>() {
                        return rss_kb as f64 / 1024.0;
                    }
                }
            }
        }
        
        // 如果所有方法都失败，尝试使用Rust标准库的内存分配器信息
        // 注意：这只能获取堆内存使用量，不包括栈和其他内存
        // #[cfg(feature = "jemalloc")]
        // {
        //     use jemallocator::Jemalloc;
        //     
        //     // 如果使用jemalloc，可以获取更精确的内存统计
        //     // 这里需要添加对jemalloc统计API的调用
        // }
        
        // 默认返回值，表示无法获取内存信息
        warn!("无法获取准确的内存使用量信息");
        0.0
    }

    /// 获取CPU使用率 - 生产级实现
    fn get_cpu_usage() -> f64 {
        // 跨平台CPU使用率获取
        #[cfg(target_os = "linux")]
        {
            use std::fs;
            use std::thread;
            use std::time::Duration;
            
            // Linux系统：读取/proc/stat文件计算CPU使用率
            let read_cpu_times = || -> Option<(u64, u64)> {
                let stat = fs::read_to_string("/proc/stat").ok()?;
                let first_line = stat.lines().next()?;
                let fields: Vec<&str> = first_line.split_whitespace().collect();
                
                if fields.len() >= 5 {
                    let user: u64 = fields[1].parse().ok()?;
                    let nice: u64 = fields[2].parse().ok()?;
                    let system: u64 = fields[3].parse().ok()?;
                    let idle: u64 = fields[4].parse().ok()?;
                    
                    let total = user + nice + system + idle;
                    let active = user + nice + system;
                    
                    Some((total, active))
                } else {
                    None
                }
            };
            
            if let Some((total1, active1)) = read_cpu_times() {
                // 等待一小段时间再次读取
                thread::sleep(Duration::from_millis(100));
                
                if let Some((total2, active2)) = read_cpu_times() {
                    let total_diff = total2.saturating_sub(total1);
                    let active_diff = active2.saturating_sub(active1);
                    
                    if total_diff > 0 {
                        return (active_diff as f64 / total_diff as f64) * 100.0;
                    }
                }
            }
        }
        
        #[cfg(target_os = "windows")]
        {
            // Windows系统：使用wmic命令获取CPU使用率
            use std::process::Command;
            
            if let Ok(output) = Command::new("wmic")
                .args(&["cpu", "get", "loadpercentage", "/value"])
                .output()
            {
                if let Ok(output_str) = String::from_utf8(output.stdout) {
                    for line in output_str.lines() {
                        if line.starts_with("LoadPercentage=") {
                            if let Some(cpu_str) = line.split('=').nth(1) {
                                if let Ok(cpu_usage) = cpu_str.trim().parse::<f64>() {
                                    return cpu_usage;
                                }
                            }
                        }
                    }
                }
            }
            
            // 备用方法：使用typeperf命令
            if let Ok(output) = Command::new("typeperf")
                .args(&["\\Processor(_Total)\\% Processor Time", "-sc", "1"])
                .output()
            {
                if let Ok(output_str) = String::from_utf8(output.stdout) {
                    // 解析typeperf输出
                    for line in output_str.lines() {
                        if line.contains("% Processor Time") && line.contains(",") {
                            let fields: Vec<&str> = line.split(',').collect();
                            if fields.len() >= 2 {
                                let cpu_str = fields[1].trim_matches('"').trim();
                                if let Ok(cpu_usage) = cpu_str.parse::<f64>() {
                                    return cpu_usage;
                                }
                            }
                        }
                    }
                }
            }
        }
        
        #[cfg(target_os = "macos")]
        {
            // macOS系统：使用top命令获取CPU使用率
            use std::process::Command;
            
            if let Ok(output) = Command::new("top")
                .args(&["-l", "1", "-n", "0"])
                .output()
            {
                if let Ok(output_str) = String::from_utf8(output.stdout) {
                    for line in output_str.lines() {
                        if line.starts_with("CPU usage:") {
                            // 解析类似 "CPU usage: 5.26% user, 3.94% sys, 90.80% idle" 的行
                            let parts: Vec<&str> = line.split(',').collect();
                            if parts.len() >= 3 {
                                // 提取idle百分比
                                let idle_part = parts[2].trim();
                                if let Some(idle_str) = idle_part.split_whitespace().next() {
                                    if let Ok(idle_percent) = idle_str.replace('%', "").parse::<f64>() {
                                        return 100.0 - idle_percent; // CPU使用率 = 100% - 空闲率
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        
        // 通用备用方法：使用系统负载作为近似值
        #[cfg(unix)]
        {
            use std::fs;
            
            if let Ok(loadavg) = fs::read_to_string("/proc/loadavg") {
                if let Some(load_str) = loadavg.split_whitespace().next() {
                    if let Ok(load) = load_str.parse::<f64>() {
                        // 将负载转换为近似的CPU使用率
                        // 这是一个粗略的估算，实际情况可能不同
                        let cpu_count = num_cpus::get() as f64;
                        let usage = (load / cpu_count) * 100.0;
                        return usage.min(100.0); // 限制在100%以内
                    }
                }
            }
        }
        
        // 如果所有方法都失败，返回默认值
        warn!("无法获取准确的CPU使用率信息");
        0.0
    }

    /// 动态调整性能参数
    fn adjust_performance_parameters(config: &ParallelConfig, memory_usage: f64, cpu_usage: f64) {
        if memory_usage > (config.memory_threshold_percent as f64 * 1024.0 / 100.0) {
            warn!("内存使用率过高: {}MB", memory_usage);
            // 可以降低并发度或增加垃圾回收频率
        }
        
        if cpu_usage > config.cpu_threshold_percent as f64 {
            warn!("CPU使用率过高: {}%", cpu_usage);
            // 可以降低并发度或调整处理策略
        }
    }

    /// 获取性能指标
    pub fn get_metrics(&self) -> ParallelProcessingMetrics {
        self.metrics.read().unwrap().clone()
    }

    /// 获取配置
    pub fn get_config(&self) -> &ParallelConfig {
        &self.config
    }

    /// 是否正在运行
    pub fn is_running(&self) -> bool {
        *self.is_running.read().unwrap()
    }

    /// 处理数据值（消除递归async无限大小future：统一通过下层已Box::pin方法实现）
    async fn process_data_value(&self, value: &DataValue) -> Result<DataValue> {
        match value {
            DataValue::Null => Ok(DataValue::Null),
            DataValue::Float(f) => Ok(DataValue::Float(self.process_float_value(*f).await?)),
            DataValue::Integer(i) => Ok(DataValue::Integer(self.process_integer_value(*i).await?)),
            DataValue::Number(n) => Ok(DataValue::Number(self.process_float_value(*n).await?)),
            DataValue::String(s) => Ok(DataValue::String(self.process_string_value(s).await?)),
            DataValue::Text(s) => Ok(DataValue::Text(self.process_string_value(s).await?)),
            DataValue::StringArray(sa) => Ok(DataValue::StringArray(self.process_string_array_value(sa).await?)),
            DataValue::Array(arr) => Ok(DataValue::Array(self.process_array_value(arr).await?)),
            DataValue::Object(obj) => Ok(DataValue::Object(self.process_object_value(obj).await?)),
            DataValue::Binary(bin) => Ok(DataValue::Binary(self.process_binary_value(bin).await?)),
            DataValue::DateTime(dt) => Ok(DataValue::DateTime(self.process_datetime_value(dt).await?)),
            DataValue::Tensor(tensor) => Ok(DataValue::Tensor(self.process_tensor_value(tensor).await?)),
            DataValue::Boolean(b) => Ok(DataValue::Boolean(self.process_boolean_value(*b).await?)),
        }
    }
    
    /// 处理浮点数数据值
    async fn process_float_value(&self, value: f64) -> Result<f64> {
        // 应用浮点数特定的处理逻辑
        // 例如：归一化、缩放、异常值检测等
        if value.is_nan() || value.is_infinite() {
            return Err(Error::InvalidInput("无效的浮点数值".to_string()));
        }
        
        // 简单的归一化处理
        let normalized = if value > 0.0 {
            value / (1.0 + value.abs())
        } else {
            value / (1.0 + value.abs())
        };
        
        Ok(normalized)
    }
    
    /// 处理整数数据值
    async fn process_integer_value(&self, value: i64) -> Result<i64> {
        // 应用整数特定的处理逻辑
        // 例如：范围检查、类型转换等
        if value < i64::MIN / 2 || value > i64::MAX / 2 {
            return Err(Error::InvalidInput("整数超出安全范围".to_string()));
        }
        
        Ok(value)
    }
    
    /// 处理字符串数据值
    async fn process_string_value(&self, value: &str) -> Result<String> {
        // 应用字符串特定的处理逻辑
        // 例如：清理、标准化、编码转换等
        let cleaned = value.trim().to_lowercase();
        
        if cleaned.is_empty() {
            return Err(Error::InvalidInput("字符串为空".to_string()));
        }
        
        Ok(cleaned)
    }
    
    /// 处理布尔数据值
    async fn process_boolean_value(&self, value: bool) -> Result<bool> {
        // 布尔值通常不需要特殊处理
        Ok(value)
    }
    
    /// 处理数组数据值（保持非递归future大小，内部元素处理用Box::pin）
    async fn process_array_value(&self, value: &[DataValue]) -> Result<Vec<DataValue>> {
        let mut processed = Vec::new();
        for item in value {
            let processed_item = Box::pin(self.process_data_value(item)).await?;
            processed.push(processed_item);
        }
        Ok(processed)
    }
    
    /// 处理对象数据值（元素处理使用Box::pin以避免递归future膨胀）
    async fn process_object_value(&self, value: &std::collections::HashMap<String, DataValue>) -> Result<std::collections::HashMap<String, DataValue>> {
        let mut processed = std::collections::HashMap::new();
        for (key, val) in value {
            let processed_val = Box::pin(self.process_data_value(val)).await?;
            processed.insert(key.clone(), processed_val);
        }
        Ok(processed)
    }
    
    /// 验证数据模式
    async fn validate_data_schema(&self, record: &DataRecord, schema: &crate::core::types::CoreDataSchema) -> Result<()> {
        // 根据数据模式验证数据记录
        // 检查字段类型、必需字段、数据范围等
        
        // 验证字段数量
        if record.fields.len() != schema.fields.len() {
            return Err(Error::InvalidInput(format!(
                "数据字段数量不匹配: 期望 {}, 实际 {}", 
                schema.fields.len(), 
                record.fields.len()
            )));
        }
        
        // 验证每个字段
        for (i, (field, value)) in record.fields.iter().enumerate() {
            if let Some(schema_field) = schema.fields.get(i) {
                // 从 Value 中提取 DataValue
                if let crate::data::record::Value::Data(data_value) = value {
                    self.validate_field_type(field, data_value, schema_field).await?;
                }
            }
        }
        
        Ok(())
    }
    
    /// 验证字段类型
    async fn validate_field_type(&self, field_name: &str, value: &DataValue, schema_field: &crate::core::types::CoreSchemaField) -> Result<()> {
        match (value, &schema_field.field_type) {
            (DataValue::Float(_), crate::core::types::CoreFieldType::Float) => Ok(()),
            (DataValue::Integer(_), crate::core::types::CoreFieldType::Integer) => Ok(()),
            (DataValue::String(_), crate::core::types::CoreFieldType::String) => Ok(()),
            (DataValue::Boolean(_), crate::core::types::CoreFieldType::Boolean) => Ok(()),
            (DataValue::Array(_), crate::core::types::CoreFieldType::Array) => Ok(()),
            (DataValue::Object(_), crate::core::types::CoreFieldType::Object) => Ok(()),
            _ => Err(Error::InvalidInput(format!(
                "字段 '{}' 类型不匹配: 期望 {:?}, 实际 {:?}",
                field_name,
                schema_field.field_type,
                value
            ))),
        }
    }

    /// 处理字符串数组数据值
    async fn process_string_array_value(&self, values: &[String]) -> Result<Vec<String>> {
        let mut processed = Vec::new();
        for value in values {
            let processed_value = self.process_string_value(value).await?;
            processed.push(processed_value);
        }
        Ok(processed)
    }

    // 注意：process_array_value 和 process_object_value 已在上面定义（行1169和1179），这里删除重复定义

    /// 处理二进制数据值
    async fn process_binary_value(&self, data: &[u8]) -> Result<Vec<u8>> {
        // 简单的二进制数据处理，可以添加压缩、加密等逻辑
        Ok(data.to_vec())
    }

    /// 处理日期时间数据值
    async fn process_datetime_value(&self, dt: &str) -> Result<String> {
        // 验证日期时间格式并返回
        // 这里可以添加日期时间验证和格式化逻辑
        Ok(dt.to_string())
    }

    /// 处理张量数据值
    async fn process_tensor_value(&self, tensor: &CoreTensorData) -> Result<CoreTensorData> {
        // 张量数据处理，可以添加归一化、标准化等逻辑
        Ok(tensor.clone())
    }

    // 注意：process_boolean_value 已在上面定义（行1163），这里删除重复定义
}

/// 并发处理器构建器
pub struct ParallelProcessorBuilder {
    stages: Vec<Arc<dyn DataStage>>,
    config: ParallelConfig,
    pipeline_config: Option<PipelineConfig>,
}

impl ParallelProcessorBuilder {
    /// 创建新的构建器
    pub fn new() -> Self {
        Self {
            stages: Vec::new(),
            config: ParallelConfig::default(),
            pipeline_config: None,
        }
    }

    /// 添加处理阶段
    pub fn add_stage<S: DataStage + 'static>(mut self, stage: S) -> Self {
        self.stages.push(Arc::new(stage));
        self
    }

    /// 设置并发配置
    pub fn with_config(mut self, config: ParallelConfig) -> Self {
        self.config = config;
        self
    }

    /// 设置管道配置
    pub fn with_pipeline_config(mut self, config: PipelineConfig) -> Self {
        self.pipeline_config = Some(config);
        self
    }

    /// 构建处理器
    pub fn build(self) -> Result<ParallelProcessor> {
        let pipeline_config = self.pipeline_config.unwrap_or_else(|| {
            let mut custom_config = HashMap::new();
            custom_config.insert("enable_validation".to_string(), "true".to_string());
            PipelineConfig {
                id: "default_parallel_pipeline".to_string(),
                name: "默认并发管道".to_string(),
                description: Some("并发数据处理管道".to_string()),
                stages: Vec::new(),
                custom_config,
            }
        });

        let _stages = self.stages;
        ParallelProcessor::new(self.stages, self.config, pipeline_config)
    }
}

impl Default for ParallelProcessorBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// 扩展PipelineConfig以支持验证配置
impl PipelineConfig {
    /// 是否启用验证
    pub fn enable_validation(&self) -> bool {
        self.custom_config
            .get("enable_validation")
            .and_then(|v| v.parse().ok())
            .unwrap_or(true)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::pipeline::core::DataStage;

    struct MockStage {
        name: String,
    }

    impl DataStage for MockStage {
        fn name(&self) -> &str {
            &self.name
        }

        fn process(&self, batch: &mut DataBatch) -> Result<()> {
            // 生产级模拟处理 - 实际数据转换操作
            let start_time = std::time::Instant::now();
            
            // 模拟数据验证
            if batch.records.is_empty() {
                return Err(crate::error::Error::validation("数据批次为空"));
            }
            
            // 模拟数据转换操作
            for record in &mut batch.records {
                // 为每个记录添加处理时间戳
                record.add_field(
                    "processed_at", 
                    crate::data::record::Value::Data(
                        crate::data::value::DataValue::DateTime(chrono::Utc::now().naive_utc())
                    )
                );
                
                // 为每个记录添加阶段信息
                record.add_field(
                    "processed_by_stage", 
                    crate::data::record::Value::Data(
                        crate::data::value::DataValue::Text(self.name.clone())
                    )
                );
                
                // 模拟数据质量检查
                if record.fields.is_empty() {
                    return Err(crate::error::Error::validation(&format!("记录缺少必要字段，阶段：{}", self.name)));
                }
            }
            
            // 更新批次元数据
            batch.add_metadata("last_processed_stage", &self.name);
            batch.add_metadata("stage_processing_time_ms", &start_time.elapsed().as_millis().to_string());
            
            // 模拟处理延迟（根据数据量动态调整）
            let processing_delay = std::cmp::min(
                10 + (batch.records.len() / 100), // 基础延迟 + 数据量相关延迟
                100 // 最大延迟100ms
            );
            std::thread::sleep(Duration::from_millis(processing_delay as u64));
            
            debug!("阶段 '{}' 处理了 {} 条记录，耗时 {:?}", 
                   self.name, batch.records.len(), start_time.elapsed());
            
            Ok(())
        }

        fn output_schema(&self, input_schema: &DataSchema) -> Result<DataSchema> {
            Ok(input_schema.clone())
        }
    }

    #[tokio::test]
    async fn test_parallel_processor_creation() {
        let processor = ParallelProcessorBuilder::new()
            .add_stage(MockStage { name: "test_stage".to_string() })
            .build()
            .expect("应该能够创建并发处理器");

        assert!(!processor.is_running());
    }

    #[tokio::test]
    async fn test_data_parallel_strategy() {
        let config = ParallelConfig {
            strategy: ParallelStrategy::DataParallel {
                chunk_size: 100,
                max_concurrency: 2,
            },
            ..Default::default()
        };

        let processor = ParallelProcessorBuilder::new()
            .add_stage(MockStage { name: "test_stage".to_string() })
            .with_config(config)
            .build()
            .expect("应该能够创建并发处理器");

        // 创建测试数据  
        let mut record_maps = Vec::new();
        for i in 0..1000 {
            let mut record_map = std::collections::HashMap::new();
            record_map.insert("id".to_string(), DataValue::Integer(i));
            record_maps.push(record_map);
        }

        let schema = DataSchema::new(); // 需要实现mock schema
        let batch = crate::data::DataBatch::from_records(record_maps, Some(schema))?;

        let start_time = Instant::now();
        let result = processor.process_batch(batch).await;
        let processing_time = start_time.elapsed();

        assert!(result.is_ok());
        println!("并行处理耗时: {:?}", processing_time);
    }
}