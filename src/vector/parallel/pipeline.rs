// 向量处理管道模块
//
// 提供用于构建高效向量处理流水线的功能

use std::sync::{Arc, Mutex, RwLock};
use crate::Result;
use crate::Error;
use super::{WorkerPool, ParallelConfig, TaskType};

/// 向量处理阶段
pub trait PipelineStage: Send + Sync {
    /// 执行处理
    fn process(&self, input: Vec<f32>) -> Result<Vec<f32>>;
    
    /// 获取处理阶段名称
    fn name(&self) -> &str;
    
    /// 获取处理阶段类型
    fn stage_type(&self) -> TaskType;
    
    /// 克隆处理阶段
    fn clone_box(&self) -> Box<dyn PipelineStage>;
}

/// 向量处理管道
pub struct VectorPipeline {
    /// 工作线程池
    workers: Arc<RwLock<WorkerPool>>,
    /// 配置
    config: ParallelConfig,
    /// 处理阶段
    stages: Vec<Box<dyn PipelineStage>>,
}

impl VectorPipeline {
    /// 创建新的向量处理管道
    pub fn new(workers: Arc<RwLock<WorkerPool>>, config: ParallelConfig) -> Self {
        Self {
            workers,
            config,
            stages: Vec::new(),
        }
    }
    
    /// 添加处理阶段
    pub fn add_stage<T: PipelineStage + 'static>(&mut self, stage: T) -> &mut Self {
        self.stages.push(Box::new(stage));
        self
    }
    
    /// 处理单个向量
    pub fn process(&self, input: Vec<f32>) -> Result<Vec<f32>> {
        let mut current = input;
        
        for stage in &self.stages {
            current = stage.process(current)?;
        }
        
        Ok(current)
    }
    
    /// 并行处理多个向量
    pub fn process_batch(&self, inputs: Vec<Vec<f32>>) -> Result<Vec<Vec<f32>>> {
        if inputs.is_empty() {
            return Ok(Vec::new());
        }
        
        if self.stages.is_empty() {
            return Ok(inputs);
        }
        
        let total_vectors = inputs.len();
        let batch_size = self.config.batch_size;
            
        // 创建结果容器
        let results = Arc::new(Mutex::new(vec![Vec::new(); total_vectors]));
        
        // 创建完成信号通道
        let total_batches = (total_vectors + batch_size - 1) / batch_size;
        let (tx, rx) = crossbeam_channel::bounded(total_batches);
        
        // 分批处理向量
        for batch_start in (0..total_vectors).step_by(batch_size) {
            let batch_end = (batch_start + batch_size).min(total_vectors);
            let batch_inputs = inputs[batch_start..batch_end].to_vec();
            
            // 为每个批次单独克隆一份处理阶段链路，避免跨线程共享可变状态
            let batch_stages: Vec<Box<dyn PipelineStage>> = self.stages
                .iter()
                .map(|stage| stage.clone_box())
                .collect();
            let results_clone = Arc::clone(&results);
            let tx_clone = tx.clone();
            
            let workers = self.workers.read().map_err(|_| 
                Error::vector("Failed to acquire read lock on worker pool".to_string()))?;
            
            workers.submit(move || {
                // 为这个批次中的每个向量执行整个处理管道
                for (batch_idx, mut vec) in batch_inputs.into_iter().enumerate() {
                    let idx = batch_start + batch_idx;
                    
                    // 依次执行每个处理阶段
                    for stage in &batch_stages {
                        match stage.process(vec) {
                            Ok(processed) => {
                                vec = processed;
                            },
                            Err(e) => {
                                log::error!("Error in pipeline stage {}: {:?}", stage.name(), e);
                                // 处理失败，但我们继续处理其他向量
                                break;
                            }
                        }
                    }
                    
                    // 存储结果
                    if let Ok(mut results_guard) = results_clone.lock() {
                        results_guard[idx] = vec;
                    }
                }
                
                // 发送完成信号
                let _ = tx_clone.send(());
            })?;
        }
        
        // 等待所有批次完成
        for _ in 0..total_batches {
            rx.recv().map_err(|_| Error::vector("Channel error while waiting for tasks to complete".to_string()))?;
        }
        
        // 获取结果
        let processed_vectors = results.lock().map_err(|_| 
            Error::vector("Failed to acquire lock on results".to_string()))?;
        
        Ok(processed_vectors.clone())
    }
    
    /// 获取处理阶段数量
    pub fn stages_count(&self) -> usize {
        self.stages.len()
    }
}

/// 向量归一化处理阶段
pub struct NormalizationStage;

impl PipelineStage for NormalizationStage {
    fn process(&self, input: Vec<f32>) -> Result<Vec<f32>> {
        let mut output = input.clone();
        crate::vector::utils::vector::normalize_vector(&mut output)?;
        Ok(output)
    }
    
    fn name(&self) -> &str {
        "Normalization"
    }
    
    fn stage_type(&self) -> TaskType {
        TaskType::Normalization
    }
    
    fn clone_box(&self) -> Box<dyn PipelineStage> {
        Box::new(NormalizationStage)
    }
}

/// 函数式处理阶段
pub struct FunctionStage {
    /// 处理函数
    func: Arc<dyn Fn(Vec<f32>) -> Result<Vec<f32>> + Send + Sync>,
    /// 阶段名称
    name: String,
    /// 阶段类型
    stage_type: TaskType,
}

impl FunctionStage {
    /// 创建新的函数式处理阶段
    pub fn new<F>(name: &str, func: F, stage_type: TaskType) -> Self 
    where 
        F: Fn(Vec<f32>) -> Result<Vec<f32>> + Send + Sync + 'static,
    {
        Self {
            func: Arc::new(func),
            name: name.to_string(),
            stage_type,
        }
    }
}

impl PipelineStage for FunctionStage {
    fn process(&self, input: Vec<f32>) -> Result<Vec<f32>> {
        (self.func)(input)
    }
    
    fn name(&self) -> &str {
        &self.name
    }
    
    fn stage_type(&self) -> TaskType {
        self.stage_type
    }
    
    fn clone_box(&self) -> Box<dyn PipelineStage> {
        Box::new(FunctionStage {
            func: self.func.clone(),
            name: self.name.clone(),
            stage_type: self.stage_type,
        })
    }
}

/// 维度约简处理阶段（PCA）
pub struct DimensionReductionStage {
    /// 目标维度
    target_dimension: usize,
    /// 投影矩阵
    projection: Vec<Vec<f32>>,
}

impl DimensionReductionStage {
    /// 创建新的维度约简处理阶段
    pub fn new(target_dimension: usize, projection: Vec<Vec<f32>>) -> Self {
        Self {
            target_dimension,
            projection,
        }
    }
}

impl PipelineStage for DimensionReductionStage {
    fn process(&self, input: Vec<f32>) -> Result<Vec<f32>> {
        if input.is_empty() {
            return Ok(Vec::new());
        }
        
        if input.len() != self.projection[0].len() {
            return Err(Error::vector(format!(
                "Input dimension mismatch: expected {}, got {}",
                self.projection[0].len(), input.len()
            )));
        }
        
        let mut output = vec![0.0; self.target_dimension];
        
        // 矩阵乘法：output = projection * input
        for i in 0..self.target_dimension {
            for (j, val) in input.iter().enumerate() {
                output[i] += self.projection[i][j] * val;
            }
        }
        
        Ok(output)
    }
    
    fn name(&self) -> &str {
        "DimensionReduction"
    }
    
    fn stage_type(&self) -> TaskType {
        TaskType::DimensionReduction
    }
    
    fn clone_box(&self) -> Box<dyn PipelineStage> {
        Box::new(DimensionReductionStage {
            target_dimension: self.target_dimension,
            projection: self.projection.clone(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_pipeline() {
        // 创建工作线程池
        let worker_pool = Arc::new(RwLock::new(WorkerPool::new(4, 100)));
        {
            let mut pool = worker_pool.write().unwrap();
            pool.start().unwrap();
        }
        
        let config = ParallelConfig {
            num_threads: 4,
            batch_size: 10,
            ..Default::default()
        };
        
        // 创建处理管道
        let mut pipeline = VectorPipeline::new(worker_pool.clone(), config);
        
        // 添加归一化阶段
        pipeline.add_stage(NormalizationStage);
        
        // 添加函数式处理阶段
        pipeline.add_stage(FunctionStage::new(
            "Scale", 
            |input| {
                let mut output = input.clone();
                for val in &mut output {
                    *val *= 2.0;
                }
                Ok(output)
            },
            TaskType::Custom
        ));
        
        // 测试单个向量处理
        let input = vec![3.0, 4.0];
        let output = pipeline.process(input).unwrap();
        
        // 归一化后为[0.6, 0.8]，然后缩放后为[1.2, 1.6]
        assert!((output[0] - 1.2).abs() < 1e-6);
        assert!((output[1] - 1.6).abs() < 1e-6);
        
        // 测试批量处理
        let inputs = vec![
            vec![3.0, 4.0],
            vec![5.0, 12.0],
            vec![1.0, 1.0],
        ];
        
        let outputs = pipeline.process_batch(inputs).unwrap();
        
        assert_eq!(outputs.len(), 3);
        
        // 检查第一个向量的结果
        assert!((outputs[0][0] - 1.2).abs() < 1e-6);
        assert!((outputs[0][1] - 1.6).abs() < 1e-6);
        
        // 停止工作线程池
        {
            let mut pool = worker_pool.write().unwrap();
            pool.stop().unwrap();
        }
    }
} 