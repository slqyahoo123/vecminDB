use std::sync::{Arc, RwLock, mpsc};
use std::time::{SystemTime, Duration};

use crate::error::Result;
// ModelManagerInterface已移除，使用compat::manager::traits::ModelManager
use crate::compat::manager::traits::ModelManager;
use crate::algorithm::executor::sandbox::AlgorithmExecutor;
use crate::algorithm::traits::Algorithm;
use crate::algorithm::types::{AlgorithmResult};
use crate::algorithm::base_types::{ExecutionParams, ExecutorControl};
use crate::algorithm::status::AlgorithmStatusTracker;
use crate::algorithm::resource::{ResourceMonitor, ResourceMonitoringConfig, AlgorithmResourceLimits};
use crate::algorithm::types::{TaskId, TaskStatus, AlgorithmMetrics};

/// 增强执行器配置
#[derive(Debug, Clone)]
pub struct EnhancedExecutorConfig {
    /// 最大执行时间（秒）
    pub max_runtime: Option<u64>,
    /// 最大内存使用（MB）
    pub max_memory: Option<usize>,
    /// 自动重试次数
    pub retry_count: usize,
    /// 重试间隔（秒）
    pub retry_interval: u64,
    /// 是否记录详细日志
    pub verbose_logging: bool,
    /// 进度更新间隔（秒）
    pub progress_update_interval: u64,
    /// 资源监控间隔（秒）
    pub resource_monitoring_interval: u64,
    /// 是否在完成时自动保存模型
    pub auto_save_model: bool,
    /// 模型保存路径
    pub model_save_path: Option<String>,
    /// 是否允许中断
    pub allow_interruption: bool,
    /// 资源监控配置
    pub resource_monitoring_config: Option<ResourceMonitoringConfig>,
    /// 资源限制
    pub resource_limits: Option<AlgorithmResourceLimits>,
    /// 是否启用GPU监控
    pub enable_gpu_monitoring: bool,
    /// 是否启用详细的磁盘I/O监控
    pub enable_disk_io_monitoring: bool,
    /// 是否启用网络I/O监控
    pub enable_network_monitoring: bool,
}

impl Default for EnhancedExecutorConfig {
    fn default() -> Self {
        Self {
            max_runtime: Some(3600), // 1小时
            max_memory: Some(4096),   // 4GB
            retry_count: 3,
            retry_interval: 5,
            verbose_logging: true,
            progress_update_interval: 10,
            resource_monitoring_interval: 5,
            auto_save_model: true,
            model_save_path: None,
            allow_interruption: true,
            resource_monitoring_config: Some(ResourceMonitoringConfig::default()),
            resource_limits: Some(AlgorithmResourceLimits::default()),
            enable_gpu_monitoring: true,
            enable_disk_io_monitoring: false,
            enable_network_monitoring: false,
        }
    }
}

/// 增强算法执行器
pub struct EnhancedAlgorithmExecutor {
    /// 基本执行器
    executor: Arc<dyn AlgorithmExecutor>,
    /// 模型管理器 - 使用compat接口
    model_manager: Arc<dyn ModelManager>,
    /// 状态跟踪器
    status_tracker: Arc<RwLock<AlgorithmStatusTracker>>,
    /// 资源监控器
    resource_monitor: Arc<RwLock<ResourceMonitor>>,
    /// 执行配置
    config: EnhancedExecutorConfig,
    /// 执行控制通道
    control_tx: Option<mpsc::Sender<ExecutorControl>>,
}

impl EnhancedAlgorithmExecutor {
    /// 创建新的增强算法执行器
    pub fn new(
        executor: Arc<dyn AlgorithmExecutor>,
        model_manager: Arc<dyn ModelManager>,
        config: Option<EnhancedExecutorConfig>,
    ) -> Self {
        let config = config.unwrap_or_default();
        
        // 创建状态跟踪器
        let status_tracker = Arc::new(RwLock::new(AlgorithmStatusTracker::new()));
        
        // 创建资源监控器
        let resource_config = config.resource_monitoring_config.clone()
            .unwrap_or_default();
        let resource_limits = config.resource_limits.clone()
            .unwrap_or_default();
        let resource_monitor = if config.enable_gpu_monitoring {
            Arc::new(RwLock::new(ResourceMonitor::with_gpu_monitoring(
                resource_config,
                resource_limits,
                0 // GPU索引
            )))
        } else {
            Arc::new(RwLock::new(ResourceMonitor::new(
                resource_config,
                resource_limits
            )))
        };
        
        Self {
            executor,
            model_manager,
            status_tracker,
            resource_monitor,
            config,
            control_tx: None,
        }
    }
    
    /// 执行算法
    pub async fn execute(&self, algorithm: Arc<dyn Algorithm + Send + Sync>, _params: ExecutionParams) -> Result<AlgorithmResult> {
        // 设置执行控制通道
        let (_tx, rx) = mpsc::channel();
        
        // 更新状态为初始化中
        {
            let mut tracker = self.status_tracker.write().unwrap();
            tracker.update_status(crate::algorithm::base_types::AlgorithmStatus::Initializing);
        }
        
        // 启动监控任务
        let monitoring_handle = self.start_monitoring_task();
        
        // 启动控制任务
        let control_handle = self.start_control_task(rx);
        
        // 执行算法
        let result = self.execute_with_monitoring(algorithm, _params).await;
        
        // 停止监控
        if let Some(handle) = monitoring_handle {
            // 发送停止信号
            drop(handle);
        }
        
        // 停止控制任务
        if let Some(handle) = control_handle {
            drop(handle);
        }
        
        // 更新最终状态
        {
            let mut tracker = self.status_tracker.write().unwrap();
            match &result {
                Ok(_) => tracker.update_status(crate::algorithm::base_types::AlgorithmStatus::Completed),
                Err(_) => tracker.update_status(crate::algorithm::base_types::AlgorithmStatus::Failed),
            }
        }
        
        result
    }
    
    /// 启动监控任务
    fn start_monitoring_task(&self) -> Option<std::thread::JoinHandle<()>> {
        let resource_monitor = Arc::clone(&self.resource_monitor);
        let status_tracker = Arc::clone(&self.status_tracker);
        let interval = Duration::from_secs(self.config.resource_monitoring_interval);
        
        Some(std::thread::spawn(move || {
            loop {
                // 更新资源使用情况
                let usage = {
                    let mut monitor = resource_monitor.write().unwrap();
                    monitor.update()
                };
                
                // 更新状态跟踪器
                {
                    let mut tracker = status_tracker.write().unwrap();
                    tracker.update_resource_usage(usage);
                }
                
                std::thread::sleep(interval);
            }
        }))
    }
    
    /// 启动控制任务
    fn start_control_task(&self, rx: mpsc::Receiver<ExecutorControl>) -> Option<std::thread::JoinHandle<()>> {
        Some(std::thread::spawn(move || {
            for control in rx {
                match control {
                    ExecutorControl::Pause => {
                        // 实现暂停逻辑
                        println!("执行器暂停");
                    }
                    ExecutorControl::Resume => {
                        // 实现恢复逻辑
                        println!("执行器恢复");
                    }
                    ExecutorControl::Cancel => {
                        // 实现取消逻辑
                        println!("执行器取消");
                        break;
                    }
                    ExecutorControl::UpdateResourceLimits(_limits) => {
                        // 实现资源限制更新逻辑
                        println!("更新资源限制");
                    }
                }
            }
        }))
    }
    
    /// 带监控的执行
    async fn execute_with_monitoring(&self, algorithm: Arc<dyn Algorithm + Send + Sync>, _params: ExecutionParams) -> Result<AlgorithmResult> {
        // 更新状态为运行中
        {
            let mut tracker = self.status_tracker.write().unwrap();
            tracker.update_status(crate::algorithm::base_types::AlgorithmStatus::Running);
        }
        
        // 执行算法（简化实现）
        let start_time = SystemTime::now();
        
        // 模拟算法执行过程
        for i in 0..=10 {
            let progress = i as f32 * 10.0;
            
            // 更新进度
            {
                let mut tracker = self.status_tracker.write().unwrap();
                tracker.update_progress(progress);
            }
            
            // 模拟执行时间
            tokio::time::sleep(Duration::from_millis(100)).await;
            
            // 检查是否超时
            if let Some(max_runtime) = self.config.max_runtime {
                if start_time.elapsed().unwrap_or(Duration::ZERO).as_secs() > max_runtime {
                    return Err(crate::error::Error::Timeout("算法执行超时".to_string()));
                }
            }
        }
        
        // 创建执行结果
        let output_json = serde_json::json!({
            "result": "success",
            "algorithm_id": algorithm.get_id(),
            "execution_time": start_time.elapsed().unwrap_or(Duration::ZERO).as_secs_f64()
        });

        let task_id = TaskId::new();
        let result = AlgorithmResult {
            task_id: task_id.clone(),
            algorithm_id: algorithm.get_id().to_string(),
            status: TaskStatus::Completed,
            output: Some(output_json),
            metrics: AlgorithmMetrics::new(algorithm.get_id().to_string(), task_id),
            error: None,
            execution_time_ms: start_time.elapsed().unwrap_or(Duration::ZERO).as_millis() as u64,
            memory_used_bytes: 0, // 需要从资源监控获取实际值
            cpu_usage_percent: 0.0, // 需要从资源监控获取实际值
            started_at: chrono::Utc::now(),
            completed_at: Some(chrono::Utc::now()),
            logs: Vec::new(),
            artifacts: Vec::new(),
        };
        
        Ok(result)
    }
    
    /// 获取执行状态
    pub fn get_status(&self) -> crate::algorithm::status::AlgorithmStatusReport {
        let tracker = self.status_tracker.read().unwrap();
        tracker.generate_report()
    }
    
    /// 获取资源报告
    pub fn get_resource_report(&self) -> crate::algorithm::resource::ResourceReport {
        let monitor = self.resource_monitor.read().unwrap();
        monitor.get_resource_report()
    }
    
    /// 发送控制命令
    pub fn send_control(&self, control: ExecutorControl) -> Result<()> {
        if let Some(ref tx) = self.control_tx {
            tx.send(control).map_err(|e| crate::error::Error::Communication(e.to_string()))?;
        }
        Ok(())
    }
    
    /// 暂停执行
    pub fn pause(&self) -> Result<()> {
        self.send_control(ExecutorControl::Pause)
    }
    
    /// 恢复执行
    pub fn resume(&self) -> Result<()> {
        self.send_control(ExecutorControl::Resume)
    }
    
    /// 取消执行
    pub fn cancel(&self) -> Result<()> {
        self.send_control(ExecutorControl::Cancel)
    }
    
    /// 更新资源限制
    pub fn update_resource_limits(&self, limits: AlgorithmResourceLimits) -> Result<()> {
        self.send_control(ExecutorControl::UpdateResourceLimits(limits))
    }
    
    /// 获取配置
    pub fn get_config(&self) -> &EnhancedExecutorConfig {
        &self.config
    }
    
    /// 更新配置
    pub fn update_config(&mut self, config: EnhancedExecutorConfig) {
        self.config = config;
    }
} 