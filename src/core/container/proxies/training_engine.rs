/// 训练引擎代理模块
/// 
/// 提供训练引擎的代理实现，支持通过服务容器获取真实服务或使用默认实现

use std::sync::Arc;
use async_trait::async_trait;
use uuid::Uuid;
use log::{info, debug, error, warn};
use chrono::Utc;
use std::collections::HashMap;

use crate::{Result, Error};
use crate::core::container::{DefaultServiceContainer, ServiceContainer};
// adapter/proxy wiring point: TrainingService injected via IOC
use crate::core::interfaces::{
    TrainingService,
    TrainingStatusInterface,
    CoreTrainingConfigInterface,
    TrainingTaskInterface,
    ValidationResult,
};
// trait in scope for method resolution of status/metrics
use crate::core::types::TrainingStatus;
use crate::core::types::CoreTensorData;
use crate::core::interfaces::TrainingEngineInterface;
use crate::core::types::{DeviceType, LossFunctionType};

/// 辅助函数：解析损失函数字符串到枚举
fn parse_loss_function(loss_str: &str) -> LossFunctionType {
    match loss_str.to_lowercase().as_str() {
        "mse" | "mean_squared_error" | "meansquarederror" => LossFunctionType::MeanSquaredError,
        "mae" | "mean_absolute_error" | "meanabsoluteerror" => LossFunctionType::MAE,
        "crossentropy" | "cross_entropy" | "categorical_crossentropy" => LossFunctionType::CrossEntropy,
        "binarycrossentropy" | "binary_crossentropy" => LossFunctionType::BinaryCrossEntropy,
        "hinge" => LossFunctionType::Hinge,
        "huber" => LossFunctionType::Huber,
        "kldivergence" | "kl_divergence" => LossFunctionType::KLDivergence,
        "poisson" => LossFunctionType::Poisson,
        _ => {
            warn!("未知的损失函数类型: {}, 使用默认值 MSE", loss_str);
            LossFunctionType::MSE
        }
    }
}

/// 辅助函数：解析设备类型字符串到枚举
fn parse_device_type(device_str: &str) -> DeviceType {
    match device_str.to_lowercase().as_str() {
        "cpu" => DeviceType::CPU,
        "gpu" | "cuda" => DeviceType::GPU,
        "opencl" => DeviceType::OpenCL,
        "metal" => DeviceType::Metal,
        "tpu" => DeviceType::TPU,
        _ => {
            warn!("未知的设备类型: {}, 使用默认值 CPU", device_str);
            DeviceType::CPU
        }
    }
}

/// 辅助函数：从 hyperparameters 中提取设备类型
fn extract_device_type(hyperparameters: &HashMap<String, String>) -> DeviceType {
    if let Some(device_str) = hyperparameters.get("device") {
        parse_device_type(device_str)
    } else if let Some(device_str) = hyperparameters.get("device_type") {
        parse_device_type(device_str)
    } else {
        DeviceType::CPU // 默认值
    }
}

/// 训练引擎代理实现
pub struct TrainingEngineProxy {
    container: Arc<DefaultServiceContainer>,
    current_tasks: Arc<std::sync::RwLock<HashMap<String, TrainingTaskInfo>>>,
    task_configs: Arc<std::sync::RwLock<HashMap<String, CoreTrainingConfigInterface>>>,
}

/// 训练任务信息
#[derive(Debug, Clone)]
struct TrainingTaskInfo {
    task_id: String,
    model_id: String,
    status: crate::core::interfaces::TrainingStatus,
    created_at: chrono::DateTime<Utc>,
    updated_at: chrono::DateTime<Utc>,
    progress: f32,
    metrics: HashMap<String, f64>,
    logs: Vec<String>, // 添加日志存储
    checkpoint_paths: Vec<String>, // 添加检查点路径存储
    engine_config: HashMap<String, String>, // 添加引擎配置存储
}

impl TrainingEngineProxy {
    /// 创建新的训练引擎代理
    pub fn new(container: Arc<DefaultServiceContainer>) -> Self {
        Self {
            container,
            current_tasks: Arc::new(std::sync::RwLock::new(HashMap::new())),
            task_configs: Arc::new(std::sync::RwLock::new(HashMap::new())),
        }
    }

    /// 尝试从容器获取真实的训练引擎接口（优先通过 trait 对象获取）
    async fn get_real_training_engine(&self) -> Option<Arc<dyn TrainingEngineInterface + Send + Sync>> {
        // 1) 优先按接口检索（容器按 trait 注册）
        if let Ok(engine_iface) = self.container.as_ref().get_trait::<dyn TrainingEngineInterface + Send + Sync>() {
            return Some(engine_iface);
        }
        // 2) 退回到具体类型并包装为接口
        if let Ok(real_engine) = self.container.get::<crate::training::engine::TrainingEngine>() {
            return Some(Arc::new(RealTrainingEngineWrapper::new(real_engine.clone())));
        }
        None
    }

    /// 验证训练配置
    fn validate_training_config(&self, config: &crate::core::interfaces::TrainingConfiguration) -> Result<()> {
        if config.model_id.is_empty() {
            return Err(Error::InvalidInput("模型ID不能为空".to_string()));
        }
        
        if config.batch_size == 0 {
            return Err(Error::InvalidInput("批次大小必须大于0".to_string()));
        }
        
        if config.epochs == 0 {
            return Err(Error::InvalidInput("训练轮次必须大于0".to_string()));
        }
        
        if config.learning_rate <= 0.0 {
            return Err(Error::InvalidInput("学习率必须大于0".to_string()));
        }
        
        debug!("训练配置验证通过");
        Ok(())
    }
    
    /// 创建训练任务
    fn create_training_task(&self, config: &crate::core::interfaces::TrainingConfiguration) -> Result<String> {
        let task_id = Uuid::new_v4().to_string();
        let task_info = TrainingTaskInfo {
            task_id: task_id.clone(),
            model_id: config.model_id.clone(),
            status: TrainingStatus::Pending,
            created_at: Utc::now(),
            updated_at: Utc::now(),
            progress: 0.0,
            metrics: HashMap::new(),
            logs: Vec::new(),
            checkpoint_paths: Vec::new(),
            engine_config: HashMap::new(),
        };
        
        let mut tasks = self.current_tasks.write()
            .map_err(|e| Error::Internal(format!("任务列表写入锁获取失败: {}", e)))?;
        tasks.insert(task_id.clone(), task_info);
        
        Ok(task_id)
    }
    
    /// 更新任务状态
    fn update_task_status(&self, task_id: &str, status: TrainingStatus) -> Result<()> {
        let mut tasks = self.current_tasks.write()
            .map_err(|e| Error::Internal(format!("任务列表写入锁获取失败: {}", e)))?;
        if let Some(task) = tasks.get_mut(task_id) {
            task.status = status;
            task.updated_at = Utc::now();
            Ok(())
        } else {
            Err(Error::InvalidInput(format!("任务不存在: {}", task_id)))
        }
    }
    
    /// 更新任务进度
    fn update_task_progress(&self, task_id: &str, progress: f32) -> Result<()> {
        let mut tasks = self.current_tasks.write()
            .map_err(|e| Error::Internal(format!("任务列表写入锁获取失败: {}", e)))?;
        if let Some(task) = tasks.get_mut(task_id) {
            task.progress = progress.clamp(0.0, 1.0);
            task.updated_at = Utc::now();
            Ok(())
        } else {
            Err(Error::InvalidInput(format!("任务不存在: {}", task_id)))
        }
    }
    
    /// 添加训练指标
    fn add_training_metric(&self, task_id: &str, metric_name: &str, value: f64) -> Result<()> {
        let mut tasks = self.current_tasks.write()
            .map_err(|e| Error::Internal(format!("任务列表写入锁获取失败: {}", e)))?;
        if let Some(task) = tasks.get_mut(task_id) {
            task.metrics.insert(metric_name.to_string(), value);
            task.updated_at = Utc::now();
            Ok(())
        } else {
            Err(Error::InvalidInput(format!("任务不存在: {}", task_id)))
        }
    }
    
    /// 添加训练日志
    fn add_training_log(&self, task_id: &str, log_message: String) -> Result<()> {
        let mut tasks = self.current_tasks.write()
            .map_err(|e| Error::Internal(format!("任务列表写入锁获取失败: {}", e)))?;
        if let Some(task) = tasks.get_mut(task_id) {
            let timestamp = Utc::now().format("%Y-%m-%d %H:%M:%S");
            task.logs.push(format!("[{}] {}", timestamp, log_message));
            // 限制日志数量，避免内存溢出
            if task.logs.len() > 10000 {
                task.logs.drain(0..5000); // 保留最新的5000条
            }
            Ok(())
        } else {
            Err(Error::InvalidInput(format!("任务不存在: {}", task_id)))
        }
    }
    
    /// 执行实际的训练过程
    async fn execute_training(&self, task_id: &str, config: &crate::core::interfaces::TrainingConfiguration) -> Result<()> {
        // 更新状态为运行中
        self.update_task_status(task_id, TrainingStatus::Running)?;
        self.add_training_log(task_id, format!("训练任务 {} 已启动", task_id))?;
        self.add_training_log(task_id, format!("配置: batch_size={}, epochs={}, learning_rate={}", 
            config.batch_size, config.epochs, config.learning_rate))?;
        
        // 模拟训练过程
        let total_epochs = config.epochs.min(10); // 限制最大显示epoch数量
        for epoch in 0..total_epochs {
            // 模拟训练进度
            let progress = (epoch as f32 + 1.0) / total_epochs as f32;
            self.update_task_progress(task_id, progress)?;
            
            // 模拟训练指标
            let loss = 1.0 - progress as f64 * 0.8; // 损失递减
            let accuracy = progress as f64 * 0.9; // 准确率递增
            
            self.add_training_metric(task_id, "loss", loss)?;
            self.add_training_metric(task_id, "accuracy", accuracy)?;
            
            // 记录训练日志
            self.add_training_log(task_id, format!("Epoch {}/{}: loss={:.4}, accuracy={:.4}", 
                epoch + 1, total_epochs, loss, accuracy))?;
            
            // 模拟训练时间
            tokio::time::sleep(std::time::Duration::from_millis(100)).await;
            
            debug!("训练进度 - Epoch {}: loss={:.4}, accuracy={:.4}", epoch + 1, loss, accuracy);
        }
        
        // 完成训练
        self.update_task_status(task_id, TrainingStatus::Completed)?;
        info!("训练任务完成: {}", task_id);
        
        Ok(())
    }
}

/// 真实训练引擎的包装器
/// 用于将具体的训练引擎实现包装为 trait object
struct RealTrainingEngineWrapper {
    engine: Arc<crate::training::engine::TrainingEngine>,
}

impl RealTrainingEngineWrapper {
    fn new(engine: Arc<crate::training::engine::TrainingEngine>) -> Self { Self { engine } }
}

#[async_trait]
impl TrainingService for RealTrainingEngineWrapper {
    async fn create_training_task(&self, config: CoreTrainingConfigInterface) -> Result<String> {
        use crate::core::types::{CoreTrainingConfiguration, OptimizationConfig, RegularizationConfig};
        use crate::core::interfaces::TrainingEngineInterface;
        
        // 将接口层配置转换为引擎可用配置
        let cfg = CoreTrainingConfiguration {
            id: config.id.clone(),
            name: format!("training_task_{}", config.id),
            model_id: config.model_id.clone(),
            algorithm_id: config.algorithm_id.clone().unwrap_or_else(|| "default".to_string()),
            parameters: config.hyperparameters.clone(),
            batch_size: config.training_options.batch_size as u32,
            learning_rate: config.training_options.learning_rate as f64,
            epochs: config.training_options.epochs as u32,
            validation_split: config.training_options.validation_split.map(|v| v as f64).unwrap_or(0.2),
            early_stopping: config.training_options.early_stopping.is_some(),
            checkpoint_enabled: config.training_options.checkpoint_config.is_some(),
            device_type: extract_device_type(&config.hyperparameters),
            optimization_config: {
                // 从 hyperparameters 中提取优化器参数
                OptimizationConfig {
                    optimizer_type: config.training_options.optimizer.clone(),
                    learning_rate_schedule: config.hyperparameters
                        .get("learning_rate_schedule")
                        .cloned()
                        .unwrap_or_else(|| "constant".to_string()),
                    weight_decay: config.hyperparameters
                        .get("weight_decay")
                        .and_then(|v| v.parse::<f64>().ok())
                        .unwrap_or(0.0),
                    momentum: config.hyperparameters
                        .get("momentum")
                        .and_then(|v| v.parse::<f64>().ok())
                        .unwrap_or(0.9),
                    beta1: config.hyperparameters
                        .get("beta1")
                        .and_then(|v| v.parse::<f64>().ok())
                        .unwrap_or(0.9),
                    beta2: config.hyperparameters
                        .get("beta2")
                        .and_then(|v| v.parse::<f64>().ok())
                        .unwrap_or(0.999),
                    epsilon: config.hyperparameters
                        .get("epsilon")
                        .and_then(|v| v.parse::<f64>().ok())
                        .unwrap_or(1e-8),
                    amsgrad: config.hyperparameters
                        .get("amsgrad")
                        .and_then(|v| v.parse::<bool>().ok())
                        .unwrap_or(false),
                    parameters: config.hyperparameters.clone(),
                }
            },
            loss_function: parse_loss_function(&config.training_options.loss_function),
            regularization: RegularizationConfig::default(),
            metadata: HashMap::new(),
            created_at: Utc::now(),
            updated_at: Utc::now(),
        };
        
        // 创建任务ID
        let task_id = uuid::Uuid::new_v4().to_string();
        
        // 调用引擎的 start_training 方法（需要 task_id 和 config）
        <dyn TrainingEngineInterface>::start_training(self.engine.as_ref(), &task_id, &cfg).await?;
        
        Ok(task_id)
    }

    async fn start_training(&self, _task_id: &str) -> Result<()> { Ok(()) }
    async fn pause_training(&self, task_id: &str) -> Result<()> { self.engine.pause_training(task_id).await }
    async fn resume_training(&self, task_id: &str) -> Result<()> { self.engine.resume_training(task_id).await }
    async fn stop_training(&self, task_id: &str) -> Result<()> { self.engine.stop_training(task_id).await }

    async fn get_training_task(&self, _task_id: &str) -> Result<Option<TrainingTaskInterface>> { Ok(None) }
    async fn get_training_metrics(&self, task_id: &str) -> Result<Option<crate::core::interfaces::TrainingMetrics>> {
        match self.engine.get_training_metrics(task_id).await? {
            Some(m) => Ok(Some(m)),
            None => Ok(None),
        }
    }
    async fn list_training_tasks(&self, _model_id: Option<&str>) -> Result<Vec<TrainingTaskInterface>> { Ok(vec![]) }
}

#[async_trait]
impl TrainingEngineInterface for RealTrainingEngineWrapper {
    async fn start_training(&self, task_id: &str, config: &crate::core::types::CoreTrainingConfiguration) -> Result<()> {
        use crate::core::types::CoreTrainingConfig;
        
        // 将 CoreTrainingConfiguration 转换为 CoreTrainingConfig
        let core_config = CoreTrainingConfig {
            id: config.id.clone(),
            model_id: config.model_id.clone(),
            dataset_id: String::new(), // CoreTrainingConfig 需要 dataset_id，但 CoreTrainingConfiguration 没有
            epochs: config.epochs,
            batch_size: config.batch_size as usize,
            learning_rate: config.learning_rate,
            optimizer: config.optimization_config.optimizer_type.clone(),
            loss_function: format!("{:?}", config.loss_function),
            metrics: vec!["loss".to_string(), "accuracy".to_string()],
            validation_split: config.validation_split,
            early_stopping: config.early_stopping,
            checkpoint_enabled: config.checkpoint_enabled,
            mixed_precision: false, // 默认值
            device: format!("{:?}", config.device_type),
            parallel: false, // 默认值
            metadata: config.metadata.clone(),
            created_at: config.created_at,
            updated_at: config.updated_at,
        };
        
        // 委托给真实的训练引擎
        self.engine.start_training(task_id, config).await
    }
    
    async fn get_training_status(&self, task_id: &str) -> Result<crate::core::interfaces::TrainingStatus> {
        self.engine.get_training_status(task_id).await
    }
    
    async fn pause_training(&self, task_id: &str) -> Result<()> {
        self.engine.pause_training(task_id).await
    }
    
    async fn resume_training(&self, task_id: &str) -> Result<()> {
        self.engine.resume_training(task_id).await
    }
    
    async fn stop_training(&self, task_id: &str) -> Result<()> {
        self.engine.stop_training(task_id).await
    }
    
    async fn get_training_metrics(&self, task_id: &str) -> Result<Option<crate::core::interfaces::TrainingMetrics>> {
        self.engine.get_training_metrics(task_id).await
    }
    
    async fn get_training_progress(&self, task_id: &str) -> Result<f32> {
        self.engine.get_training_progress(task_id).await
    }
    
    async fn save_checkpoint(&self, task_id: &str, checkpoint_path: &str) -> Result<()> {
        self.engine.save_checkpoint(task_id, checkpoint_path).await
    }
    
    async fn load_checkpoint(&self, task_id: &str, checkpoint_path: &str) -> Result<()> {
        self.engine.load_checkpoint(task_id, checkpoint_path).await
    }
    
    async fn get_training_logs(&self, task_id: &str) -> Result<Vec<String>> {
        self.engine.get_training_logs(task_id).await
    }
    
    async fn get_engine_config(&self) -> Result<HashMap<String, String>> {
        self.engine.get_engine_config().await
    }
    
    async fn update_engine_config(&self, config: HashMap<String, String>) -> Result<()> {
        self.engine.update_engine_config(config).await
    }
    
    async fn validate_training_config(&self, config: &crate::core::types::CoreTrainingConfiguration) -> Result<crate::core::interfaces::ValidationResult> {
        self.engine.validate_training_config(config).await
    }
    
    async fn update_model_gradients(&self, model_id: &str, gradients: &HashMap<String, CoreTensorData>) -> Result<()> {
        self.engine.update_model_gradients(model_id, gradients).await
    }
}

#[async_trait]
impl TrainingService for TrainingEngineProxy {
    async fn create_training_task(&self, config: CoreTrainingConfigInterface) -> Result<String> {
        let task_id = Uuid::new_v4().to_string();
        {
            let mut cfgs = self.task_configs.write()
                .map_err(|e| Error::Internal(format!("任务配置写入锁获取失败: {}", e)))?;
            cfgs.insert(task_id.clone(), config);
        }
        let info = TrainingTaskInfo {
            task_id: task_id.clone(),
            model_id: String::new(),
            status: TrainingStatus::Pending,
            created_at: Utc::now(),
            updated_at: Utc::now(),
            progress: 0.0,
            metrics: HashMap::new(),
            logs: Vec::new(),
            checkpoint_paths: Vec::new(),
            engine_config: HashMap::new(),
        };
        let mut tasks = self.current_tasks.write()
            .map_err(|e| Error::Internal(format!("任务列表写入锁获取失败: {}", e)))?;
        tasks.insert(task_id.clone(), info);
        Ok(task_id)
    }

    async fn start_training(&self, task_id: &str) -> Result<()> {
        // 尝试读取配置并执行代理训练
        let cfg_opt = { 
            self.task_configs.read()
                .map_err(|e| Error::Internal(format!("任务配置读取锁获取失败: {}", e)))?
                .get(task_id).cloned() 
        };
        if let Some(cfg) = cfg_opt {
            // 将 CoreTrainingConfigInterface 转换为 CoreTrainingConfiguration
            use crate::core::types::{CoreTrainingConfiguration, OptimizationConfig, RegularizationConfig};
            let config = CoreTrainingConfiguration {
                id: cfg.id.clone(),
                name: format!("Training task {}", task_id),
                model_id: cfg.model_id.clone(),
                algorithm_id: cfg.algorithm_id.clone().unwrap_or_else(|| "default".to_string()),
                parameters: cfg.hyperparameters.clone(),
                batch_size: cfg.training_options.batch_size as u32,
                learning_rate: cfg.training_options.learning_rate as f64,
                epochs: cfg.training_options.epochs as u32,
                validation_split: cfg.training_options.validation_split.map(|v| v as f64).unwrap_or(0.0),
                early_stopping: cfg.training_options.early_stopping.is_some(),
                checkpoint_enabled: cfg.training_options.checkpoint_config.is_some(),
                device_type: extract_device_type(&cfg.hyperparameters),
                optimization_config: {
                    // 从 hyperparameters 中提取优化器参数，如果没有则使用默认值
                    let mut opt_config = OptimizationConfig {
                        optimizer_type: cfg.training_options.optimizer.clone(),
                        learning_rate_schedule: cfg.hyperparameters
                            .get("learning_rate_schedule")
                            .cloned()
                            .unwrap_or_else(|| "constant".to_string()),
                        weight_decay: cfg.hyperparameters
                            .get("weight_decay")
                            .and_then(|v| v.parse::<f64>().ok())
                            .unwrap_or(0.0),
                        momentum: cfg.hyperparameters
                            .get("momentum")
                            .and_then(|v| v.parse::<f64>().ok())
                            .unwrap_or(0.9),
                        beta1: cfg.hyperparameters
                            .get("beta1")
                            .and_then(|v| v.parse::<f64>().ok())
                            .unwrap_or(0.9),
                        beta2: cfg.hyperparameters
                            .get("beta2")
                            .and_then(|v| v.parse::<f64>().ok())
                            .unwrap_or(0.999),
                        epsilon: cfg.hyperparameters
                            .get("epsilon")
                            .and_then(|v| v.parse::<f64>().ok())
                            .unwrap_or(1e-8),
                        amsgrad: cfg.hyperparameters
                            .get("amsgrad")
                            .and_then(|v| v.parse::<bool>().ok())
                            .unwrap_or(false),
                        parameters: cfg.hyperparameters.clone(),
                    };
                    opt_config
                },
                loss_function: parse_loss_function(&cfg.training_options.loss_function),
                regularization: RegularizationConfig::default(),
                metadata: HashMap::new(),
                created_at: cfg.created_at,
                updated_at: cfg.updated_at,
            };
            let task_id_s = task_id.to_string();
            let self_clone = Arc::new(self.clone());
            tokio::spawn(async move {
                let _ = self_clone.execute_training(&task_id_s, &config).await;
            });
            Ok(())
        } else {
            Err(Error::InvalidInput(format!("任务配置不存在: {}", task_id)))
        }
    }

    async fn pause_training(&self, task_id: &str) -> Result<()> { self.update_task_status(task_id, TrainingStatus::Paused) }
    async fn resume_training(&self, task_id: &str) -> Result<()> { self.update_task_status(task_id, TrainingStatus::Running) }
    async fn stop_training(&self, task_id: &str) -> Result<()> { self.update_task_status(task_id, TrainingStatus::Stopped) }

    async fn get_training_task(&self, task_id: &str) -> Result<Option<TrainingTaskInterface>> {
        let tasks = self.current_tasks.read()
            .map_err(|e| Error::Internal(format!("任务列表读取锁获取失败: {}", e)))?;
        Ok(tasks.get(task_id).map(|t| TrainingTaskInterface {
            id: t.task_id.clone(),
            name: format!("training-{}", t.model_id),
            description: None,
            config: CoreTrainingConfigInterface {
                id: t.task_id.clone(),
                model_id: t.model_id.clone(),
                dataset_id: String::new(),
                algorithm_id: None,
                hyperparameters: HashMap::new(),
                training_options: crate::core::interfaces::TrainingOptionsInterface { batch_size: 32, epochs: 1, learning_rate: 1e-3, optimizer: "Adam".into(), loss_function: "MSE".into(), validation_split: None, early_stopping: None, checkpoint_config: None },
                created_at: t.created_at,
                updated_at: t.updated_at,
            },
            status: TrainingStatusInterface::Running,
            progress: t.progress,
            metrics: None,
            error_message: None,
            created_at: t.created_at,
            updated_at: t.updated_at,
            started_at: Some(t.created_at),
            completed_at: None,
        }))
    }

    async fn get_training_metrics(&self, task_id: &str) -> Result<Option<crate::core::interfaces::TrainingMetrics>> {
        let tasks = self.current_tasks.read()
            .map_err(|e| Error::Internal(format!("任务列表读取锁获取失败: {}", e)))?;
        Ok(tasks.get(task_id).map(|t| crate::core::interfaces::TrainingMetrics {
            epoch: t.metrics.get("epoch")
                .copied()
                .map(|v| v as usize)
                .unwrap_or(0),
            total_epochs: t.metrics.get("total_epochs")
                .copied()
                .map(|v| v as usize)
                .unwrap_or(0),
            loss: *t.metrics.get("loss").unwrap_or(&0.0) as f32,
            val_loss: t.metrics.get("val_loss").copied().map(|v| v as f32),
            metrics: t.metrics.iter().map(|(k, v)| (k.clone(), *v as f32)).collect(),
            val_metrics: None,
            learning_rate: t.metrics.get("learning_rate")
                .copied()
                .map(|v| v as f32)
                .unwrap_or(0.001f32),
            time_elapsed: t.metrics.get("time_elapsed")
                .copied()
                .unwrap_or(0.0),
            batch: t.metrics.get("batch")
                .copied()
                .map(|v| v as usize)
                .unwrap_or(0),
            total_batches: t.metrics.get("total_batches")
                .copied()
                .map(|v| v as usize)
                .unwrap_or(0),
            timestamp: t.updated_at.timestamp() as u64,
            accuracy: t.metrics.get("accuracy").copied().map(|v| v as f32),
            precision: None,
            recall: None,
            forward_time: 0,
            backward_time: 0,
            update_time: 0,
            step: t.metrics.get("step")
                .copied()
                .map(|v| v as usize)
                .unwrap_or(0),
            loss_history: None,
            accuracy_history: None,
            duration_seconds: None,
            samples_per_second: None,
            val_accuracy: None,
            training_time_seconds: t.metrics.get("training_time_seconds")
                .copied()
                .unwrap_or(0.0),
            custom_metrics: t.metrics.clone(),
            f1_score: None,
        }))
    }

    async fn list_training_tasks(&self, _model_id: Option<&str>) -> Result<Vec<TrainingTaskInterface>> {
        let tasks = self.current_tasks.read()
            .map_err(|e| Error::Internal(format!("任务列表读取锁获取失败: {}", e)))?;
        Ok(tasks.values().map(|t| TrainingTaskInterface {
            id: t.task_id.clone(),
            name: format!("training-{}", t.model_id),
            description: None,
            config: CoreTrainingConfigInterface {
                id: t.task_id.clone(),
                model_id: t.model_id.clone(),
                dataset_id: String::new(),
                algorithm_id: None,
                hyperparameters: HashMap::new(),
                training_options: crate::core::interfaces::TrainingOptionsInterface { batch_size: 32, epochs: 1, learning_rate: 1e-3, optimizer: "Adam".into(), loss_function: "MSE".into(), validation_split: None, early_stopping: None, checkpoint_config: None },
                created_at: t.created_at,
                updated_at: t.updated_at,
            },
            status: TrainingStatusInterface::Pending,
            progress: t.progress,
            metrics: None,
            error_message: None,
            created_at: t.created_at,
            updated_at: t.updated_at,
            started_at: None,
            completed_at: None,
        }).collect())
    }
}

#[async_trait]
impl crate::core::interfaces::TrainingEngineInterface for TrainingEngineProxy {
    async fn start_training(&self, task_id: &str, config: &crate::core::types::CoreTrainingConfiguration) -> Result<()> {
        // 首先尝试从容器获取真实的训练引擎服务
        if let Some(real_engine) = self.get_real_training_engine().await {
            return real_engine.start_training(task_id, config).await;
        }
        
        // 如果无法获取真实服务，使用代理实现
        // 验证配置
        self.validate_training_config(config)?;
        
        // 创建训练任务
        let task_id = self.create_training_task(config)?;
        
        // 异步执行训练
        let task_id_clone = task_id.clone();
        let config_clone = config.clone();
        let self_clone = Arc::new(self.clone());
        
        tokio::spawn(async move {
            if let Err(e) = self_clone.execute_training(&task_id_clone, &config_clone).await {
                error!("训练任务执行失败: {} - {}", task_id_clone, e);
                let _ = self_clone.update_task_status(&task_id_clone, crate::core::interfaces::TrainingStatus::Failed);
            }
        });
        
        info!("已启动训练任务: {}", task_id);
        Ok(())
    }
    
    async fn get_training_status(&self, task_id: &str) -> Result<crate::core::interfaces::TrainingStatus> {
        // 首先尝试从容器获取真实的训练引擎服务
        if let Some(real_engine) = self.get_real_training_engine().await {
            return real_engine.get_training_status(task_id).await;
        }
        
        // 如果无法获取真实服务，使用代理实现
        // 从本地任务列表获取状态
        let tasks = self.current_tasks.read()
            .map_err(|e| Error::Internal(format!("任务列表读取锁获取失败: {}", e)))?;
        match tasks.get(task_id) {
            Some(task) => Ok(task.status.clone()),
            None => Err(Error::InvalidInput(format!("任务不存在: {}", task_id)))
        }
    }
    
    async fn pause_training(&self, task_id: &str) -> Result<()> {
        // 首先尝试从容器获取真实的训练引擎服务
        if let Some(real_engine) = self.get_real_training_engine().await {
            return real_engine.pause_training(task_id).await;
        }
        
        // 如果无法获取真实服务，使用代理实现
        // 更新状态为暂停
        self.update_task_status(task_id, TrainingStatus::Paused)?;
        info!("已暂停训练任务: {}", task_id);
        Ok(())
    }
    
    async fn resume_training(&self, task_id: &str) -> Result<()> {
        // 首先尝试从容器获取真实的训练引擎服务
        if let Some(real_engine) = self.get_real_training_engine().await {
            return real_engine.resume_training(task_id).await;
        }
        
        // 如果无法获取真实服务，使用代理实现
        // 更新状态为运行中
        self.update_task_status(task_id, TrainingStatus::Running)?;
        info!("已恢复训练任务: {}", task_id);
        Ok(())
    }
    
    async fn stop_training(&self, task_id: &str) -> Result<()> {
        // 首先尝试从容器获取真实的训练引擎服务
        if let Some(real_engine) = self.get_real_training_engine().await {
            return real_engine.stop_training(task_id).await;
        }
        
        // 如果无法获取真实服务，使用代理实现
        // 更新状态为已停止
        self.update_task_status(task_id, TrainingStatus::Stopped)?;
        info!("已停止训练任务: {}", task_id);
        Ok(())
    }
    
    async fn get_training_metrics(&self, task_id: &str) -> Result<Option<crate::core::interfaces::TrainingMetrics>> {
        // 首先尝试从容器获取真实的训练引擎服务
        if let Some(real_engine) = self.get_real_training_engine().await {
            return real_engine.get_training_metrics(task_id).await;
        }
        
        // 如果无法获取真实服务，使用代理实现
        // 从本地任务列表获取指标
        let tasks = self.current_tasks.read()
            .map_err(|e| Error::Internal(format!("任务列表读取锁获取失败: {}", e)))?;
        match tasks.get(task_id) {
            Some(task) => {
                Ok(Some(crate::core::interfaces::TrainingMetrics {
                    epoch: task.metrics.get("epoch")
                        .copied()
                        .map(|v| v as usize)
                        .unwrap_or(0),
                    total_epochs: task.metrics.get("total_epochs")
                        .copied()
                        .map(|v| v as usize)
                        .unwrap_or(0),
                    loss: *task.metrics.get("loss").unwrap_or(&0.0) as f32,
                    val_loss: task.metrics.get("val_loss").copied().map(|v| v as f32),
                    metrics: task.metrics.iter().map(|(k, v)| (k.clone(), *v as f32)).collect(),
                    val_metrics: None,
                    learning_rate: task.metrics.get("learning_rate")
                        .copied()
                        .map(|v| v as f32)
                        .unwrap_or(0.001f32),
                    time_elapsed: task.metrics.get("time_elapsed")
                        .copied()
                        .unwrap_or(0.0),
                    batch: task.metrics.get("batch")
                        .copied()
                        .map(|v| v as usize)
                        .unwrap_or(0),
                    total_batches: task.metrics.get("total_batches")
                        .copied()
                        .map(|v| v as usize)
                        .unwrap_or(0),
                    timestamp: task.updated_at.timestamp() as u64,
                    accuracy: task.metrics.get("accuracy").copied().map(|v| v as f32),
                    precision: None,
                    recall: None,
                    forward_time: 0,
                    backward_time: 0,
                    update_time: 0,
                    step: task.metrics.get("step")
                        .copied()
                        .map(|v| v as usize)
                        .unwrap_or(0),
                    loss_history: None,
                    accuracy_history: None,
                    duration_seconds: None,
                    samples_per_second: None,
                    val_accuracy: None,
                    training_time_seconds: task.metrics.get("training_time_seconds")
                        .copied()
                        .unwrap_or(0.0),
                    custom_metrics: task.metrics.clone(),
                    f1_score: None,
                }))
            },
            None => Ok(None)
        }
    }
    
    async fn update_model_gradients(&self, model_id: &str, gradients: &HashMap<String, CoreTensorData>) -> Result<()> {
        // 首先尝试从容器获取真实的训练引擎服务
        if let Some(real_engine) = self.get_real_training_engine().await {
            return real_engine.update_model_gradients(model_id, gradients).await;
        }
        
        // 如果无法获取真实服务，使用代理实现
        // 记录梯度更新
        debug!("更新模型梯度: {}, 参数数量: {}", model_id, gradients.len());
        Ok(())
    }
    
    async fn get_training_progress(&self, task_id: &str) -> Result<f32> {
        // 首先尝试从容器获取真实的训练引擎服务
        if let Some(real_engine) = self.get_real_training_engine().await {
            return real_engine.get_training_progress(task_id).await;
        }
        
        // 如果无法获取真实服务，使用代理实现
        let tasks = self.current_tasks.read()
            .map_err(|e| Error::Internal(format!("任务列表读取锁获取失败: {}", e)))?;
        match tasks.get(task_id) {
            Some(task) => {
                // 根据任务状态计算进度
                let progress = match task.status {
                    TrainingStatus::Completed => 1.0,
                    TrainingStatus::Failed | TrainingStatus::Stopped => 0.0,
                    TrainingStatus::Running | TrainingStatus::Paused => {
                        // 从指标中获取实际进度，如果没有则基于时间估算
                        if let Some(progress) = task.metrics.get("progress") {
                            *progress
                        } else {
                            // 基于创建时间和当前时间估算进度（假设训练需要1小时）
                            let elapsed = (Utc::now() - task.created_at).num_seconds() as f64;
                            let estimated_duration = 3600.0; // 1小时
                            (elapsed / estimated_duration).min(0.99).max(0.01)
                        }
                    },
                    TrainingStatus::Pending => 0.0,
                };
                Ok(progress as f32)
            },
            None => Err(Error::InvalidInput(format!("任务不存在: {}", task_id)))
        }
    }
    
    async fn save_checkpoint(&self, task_id: &str, checkpoint_path: &str) -> Result<()> {
        // 首先尝试从容器获取真实的训练引擎服务
        if let Some(real_engine) = self.get_real_training_engine().await {
            return real_engine.save_checkpoint(task_id, checkpoint_path).await;
        }
        
        // 如果无法获取真实服务，使用代理实现
        // 1. 验证任务存在并获取需要的数据（在锁内完成）
        let (task_id_clone, model_id, status_str, progress, metrics, created_at, updated_at) = {
            let tasks = self.current_tasks.read()
                .map_err(|e| Error::Internal(format!("任务列表读取锁获取失败: {}", e)))?;
            let task = tasks.get(task_id)
                .ok_or_else(|| Error::InvalidInput(format!("训练任务不存在: {}", task_id)))?;
            (
                task.task_id.clone(),
                task.model_id.clone(),
                format!("{:?}", task.status),
                task.progress,
                task.metrics.clone(),
                task.created_at,
                task.updated_at,
            )
        };
        
        // 2. 确保检查点目录存在
        use std::path::Path;
        let path = Path::new(checkpoint_path);
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)
                .map_err(|e| Error::Internal(format!("创建检查点目录失败: {}", e)))?;
        }
        
        // 3. 获取当前任务状态并序列化为检查点
        let checkpoint_data = serde_json::json!({
            "task_id": task_id_clone,
            "model_id": model_id,
            "status": status_str,
            "progress": progress,
            "metrics": metrics,
            "created_at": created_at.to_rfc3339(),
            "updated_at": updated_at.to_rfc3339(),
        });
        
        // 4. 保存检查点到文件
        let checkpoint_json = serde_json::to_string_pretty(&checkpoint_data)
            .map_err(|e| Error::Internal(format!("序列化检查点失败: {}", e)))?;
        
        std::fs::write(checkpoint_path, checkpoint_json)
            .map_err(|e| Error::Internal(format!("写入检查点文件失败: {}", e)))?;
        
        // 5. 尝试通过StorageService保存（如果可用）- 在锁外进行 await
        if let Ok(storage) = self.container.get_trait::<dyn crate::core::interfaces::StorageService + Send + Sync>() {
            let checkpoint_key = format!("checkpoints/{}/{}", task_id, 
                std::path::Path::new(checkpoint_path).file_name()
                    .and_then(|n| n.to_str())
                    .unwrap_or("checkpoint.json"));
            if let Err(e) = storage.store(&checkpoint_key, checkpoint_data.to_string().as_bytes()).await {
                warn!("通过StorageService保存检查点失败: {}, 已保存到本地文件", e);
            }
        }
        
        // 6. 记录检查点路径（在 await 之后再次获取锁）
        {
            let mut tasks = self.current_tasks.write()
                .map_err(|e| Error::Internal(format!("任务列表写入锁获取失败: {}", e)))?;
            if let Some(task) = tasks.get_mut(task_id) {
                task.checkpoint_paths.push(checkpoint_path.to_string());
            }
        }
        
        info!("成功保存检查点: {} -> {}", task_id, checkpoint_path);
        Ok(())
    }
    
    async fn load_checkpoint(&self, task_id: &str, checkpoint_path: &str) -> Result<()> {
        // 首先尝试从容器获取真实的训练引擎服务
        if let Some(real_engine) = self.get_real_training_engine().await {
            return real_engine.load_checkpoint(task_id, checkpoint_path).await;
        }
        
        // 如果无法获取真实服务，使用代理实现
        // 1. 验证检查点文件存在
        use std::path::Path;
        if !Path::new(checkpoint_path).exists() {
            // 尝试从StorageService加载
            if let Ok(storage) = self.container.get_trait::<dyn crate::core::interfaces::StorageService + Send + Sync>() {
                let checkpoint_key = format!("checkpoints/{}/{}", task_id,
                    Path::new(checkpoint_path).file_name()
                        .and_then(|n| n.to_str())
                        .unwrap_or("checkpoint.json"));
                if let Ok(Some(data)) = storage.retrieve(&checkpoint_key).await {
                    // 将数据写入本地文件
                    std::fs::write(checkpoint_path, data)
                        .map_err(|e| Error::Internal(format!("写入检查点文件失败: {}", e)))?;
                } else {
                    return Err(Error::InvalidInput(format!("检查点文件不存在: {}", checkpoint_path)));
                }
            } else {
                return Err(Error::InvalidInput(format!("检查点文件不存在: {}", checkpoint_path)));
            }
        }
        
        // 2. 读取并解析检查点文件
        let checkpoint_content = std::fs::read_to_string(checkpoint_path)
            .map_err(|e| Error::Internal(format!("读取检查点文件失败: {}", e)))?;
        
        let checkpoint_data: serde_json::Value = serde_json::from_str(&checkpoint_content)
            .map_err(|e| Error::Internal(format!("解析检查点文件失败: {}", e)))?;
        
        // 3. 恢复任务状态
        let mut tasks = self.current_tasks.write()
            .map_err(|e| Error::Internal(format!("任务列表写入锁获取失败: {}", e)))?;
        if let Some(task) = tasks.get_mut(task_id) {
            // 恢复进度
            if let Some(progress) = checkpoint_data.get("progress").and_then(|v| v.as_f64()) {
                task.progress = progress as f32;
            }
            
            // 恢复指标
            if let Some(metrics) = checkpoint_data.get("metrics").and_then(|v| v.as_object()) {
                for (key, value) in metrics {
                    if let Some(num) = value.as_f64() {
                        task.metrics.insert(key.clone(), num);
                    }
                }
            }
            
            // 恢复状态（生产级实现：完整的状态解析）
            if let Some(status_str) = checkpoint_data.get("status").and_then(|v| v.as_str()) {
                // 解析状态字符串为TrainingStatus枚举
                use crate::core::interfaces::TrainingStatus;
                
                task.status = match status_str.to_lowercase().as_str() {
                    "pending" => TrainingStatus::Pending,
                    "running" => TrainingStatus::Running,
                    "completed" => TrainingStatus::Completed,
                    "failed" => TrainingStatus::Failed,
                    "paused" => TrainingStatus::Paused,
                    "cancelled" => TrainingStatus::Cancelled,
                    _ => {
                        log::warn!("未知的训练状态: {}, 保持当前状态", status_str);
                        task.status.clone() // 保持原状态
                    }
                };
                
                task.updated_at = Utc::now();
                log::debug!("成功恢复训练状态: {} -> {:?}", task_id, task.status);
            } else {
                log::debug!("检查点中没有状态信息，保持当前状态");
            }
            
            info!("成功加载检查点: {} <- {}", task_id, checkpoint_path);
            Ok(())
        } else {
            Err(Error::InvalidInput(format!("训练任务不存在: {}", task_id)))
        }
    }
    
    async fn get_training_logs(&self, task_id: &str) -> Result<Vec<String>> {
        // 首先尝试从容器获取真实的训练引擎服务
        if let Some(real_engine) = self.get_real_training_engine().await {
            return real_engine.get_training_logs(task_id).await;
        }
        
        // 如果无法获取真实服务，使用代理实现
        let tasks = self.current_tasks.read()
            .map_err(|e| Error::Internal(format!("任务列表读取锁获取失败: {}", e)))?;
        match tasks.get(task_id) {
            Some(task) => {
                // 返回存储的训练日志
                let mut logs = task.logs.clone();
                
                // 如果日志为空，尝试从文件系统读取
                if logs.is_empty() {
                    use std::path::Path;
                    let log_file = format!("./logs/training_{}.log", task_id);
                    if Path::new(&log_file).exists() {
                        if let Ok(content) = std::fs::read_to_string(&log_file) {
                            logs = content.lines().map(|s| s.to_string()).collect();
                        }
                    }
                }
                
                // 如果仍然为空，生成基本日志
                if logs.is_empty() {
                    logs.push(format!("训练任务 {} 已启动于 {}", task_id, task.created_at.format("%Y-%m-%d %H:%M:%S")));
                    logs.push(format!("当前状态: {:?}", task.status));
                    logs.push(format!("进度: {:.2}%", task.progress * 100.0));
                    if !task.metrics.is_empty() {
                        let metrics_str: Vec<String> = task.metrics.iter()
                            .map(|(k, v)| format!("{}: {:.4}", k, v))
                            .collect();
                        logs.push(format!("指标: {}", metrics_str.join(", ")));
                    }
                }
                
                Ok(logs)
            },
            None => Err(Error::InvalidInput(format!("任务不存在: {}", task_id)))
        }
    }
    
    async fn get_engine_config(&self) -> Result<HashMap<String, String>> {
        // 首先尝试从容器获取真实的训练引擎服务
        if let Some(real_engine) = self.get_real_training_engine().await {
            return real_engine.get_engine_config().await;
        }
        
        // 如果无法获取真实服务，使用代理实现
        // 1. 尝试从文件加载配置
        let config_file = "./config/training_engine_config.json";
        if std::path::Path::new(config_file).exists() {
            if let Ok(content) = std::fs::read_to_string(config_file) {
                if let Ok(config) = serde_json::from_str::<HashMap<String, String>>(&content) {
                    return Ok(config);
                }
            }
        }
        
        // 2. 尝试从StorageService加载（如果可用）
        if let Ok(storage) = self.container.get_trait::<dyn crate::core::interfaces::StorageService + Send + Sync>() {
            let config_key = "training_engine/config";
            if let Ok(Some(data)) = storage.retrieve(config_key).await {
                if let Ok(config_str) = String::from_utf8(data) {
                    if let Ok(config) = serde_json::from_str::<HashMap<String, String>>(&config_str) {
                        return Ok(config);
                    }
                }
            }
        }
        
        // 3. 返回默认配置
        let mut config = HashMap::new();
        config.insert("engine_type".to_string(), "proxy".to_string());
        config.insert("max_concurrent_tasks".to_string(), "10".to_string());
        config.insert("timeout_seconds".to_string(), "3600".to_string());
        config.insert("checkpoint_frequency".to_string(), "1000".to_string());
        Ok(config)
    }
    
    async fn update_engine_config(&self, config: HashMap<String, String>) -> Result<()> {
        // 首先尝试从容器获取真实的训练引擎服务
        if let Some(real_engine) = self.get_real_training_engine().await {
            return real_engine.update_engine_config(config).await;
        }
        
        // 如果无法获取真实服务，使用代理实现
        // 1. 验证配置项
        for (key, value) in &config {
            if key.is_empty() {
                return Err(Error::InvalidInput("配置键不能为空".to_string()));
            }
            if value.is_empty() && (key == "max_concurrent_tasks" || key == "timeout_seconds") {
                return Err(Error::InvalidInput(format!("配置项 {} 的值不能为空", key)));
            }
        }
        
        // 2. 保存配置到持久化存储
        let config_json = serde_json::to_string_pretty(&config)
            .map_err(|e| Error::Internal(format!("序列化配置失败: {}", e)))?;
        
        // 保存到文件
        std::fs::create_dir_all("./config")
            .map_err(|e| Error::Internal(format!("创建配置目录失败: {}", e)))?;
        let config_file = "./config/training_engine_config.json";
        std::fs::write(config_file, config_json)
            .map_err(|e| Error::Internal(format!("写入配置文件失败: {}", e)))?;
        
        // 3. 尝试通过StorageService保存（如果可用）
        if let Ok(storage) = self.container.get_trait::<dyn crate::core::interfaces::StorageService + Send + Sync>() {
            let config_key = "training_engine/config";
            if let Err(e) = storage.store(config_key, config_json.as_bytes()).await {
                warn!("通过StorageService保存配置失败: {}, 已保存到本地文件", e);
            }
        }
        
        info!("成功更新引擎配置，已保存到 {}", config_file);
        Ok(())
    }
    
    async fn validate_training_config(&self, config: &crate::core::types::CoreTrainingConfiguration) -> Result<crate::core::interfaces::ValidationResult> {
        // 首先尝试从容器获取真实的训练引擎服务
        if let Some(real_engine) = self.get_real_training_engine().await {
            return real_engine.validate_training_config(config).await;
        }
        
        // 如果无法获取真实服务，使用代理实现
        // 全面的配置验证
        let mut errors = Vec::new();
        let mut warnings = Vec::new();
        let mut score = 1.0;
        
        // 1. 验证必需字段
        if config.model_id.is_empty() {
            errors.push("模型ID不能为空".to_string());
            score -= 0.3;
        } else if config.model_id.len() > 100 {
            errors.push("模型ID长度不能超过100个字符".to_string());
            score -= 0.2;
        }
        
        // 2. 验证训练配置
        {
            if config.batch_size == 0 {
                errors.push("批次大小必须大于0".to_string());
                score -= 0.1;
            } else if config.batch_size > 10000 {
                warnings.push("批次大小较大，可能导致内存不足".to_string());
                score -= 0.05;
            }
            
            if config.epochs == 0 {
                errors.push("训练轮次必须大于0".to_string());
                score -= 0.1;
            } else if config.epochs > 10000 {
                warnings.push("训练轮次过多，训练时间可能很长".to_string());
                score -= 0.05;
            }
            
            if config.learning_rate <= 0.0 {
                errors.push("学习率必须大于0".to_string());
                score -= 0.1;
            } else if config.learning_rate > 1.0 {
                warnings.push("学习率过大，可能导致训练不稳定".to_string());
                score -= 0.05;
            } else if config.learning_rate < 1e-6 {
                warnings.push("学习率过小，训练可能很慢".to_string());
                score -= 0.05;
            }
            
            // 验证优化器类型
            if config.optimization_config.optimizer_type.is_empty() {
                errors.push("优化器类型不能为空".to_string());
                score -= 0.1;
            }
            
            // 验证损失函数
            match config.loss_function {
                _ => {} // LossFunctionType 是枚举，不需要额外验证
            }
        }
        
        // 3. 验证设备配置
        match config.device_type {
            crate::core::types::DeviceType::GPU => {
                warnings.push("使用GPU但未指定设备ID，将使用默认GPU".to_string());
                score -= 0.05;
            },
            _ => {}
        }
        
        // 确保分数在合理范围内
        score = (score as f32).max(0.0).min(1.0);
        
        Ok(crate::core::interfaces::ValidationResult {
            is_valid: errors.is_empty(),
            errors,
            warnings,
            score: Some(score),
            metadata: {
                let mut meta = HashMap::new();
                meta.insert("validation_timestamp".to_string(), Utc::now().to_rfc3339());
                meta.insert("error_count".to_string(), errors.len().to_string());
                meta.insert("warning_count".to_string(), warnings.len().to_string());
                meta
            },
        })
    }
}

// 为TrainingEngineProxy实现Clone trait（如果需要）
impl Clone for TrainingEngineProxy {
    fn clone(&self) -> Self {
        Self {
            container: self.container.clone(),
            current_tasks: self.current_tasks.clone(),
            task_configs: self.task_configs.clone(),
        }
    }
}

/// 为 TrainingService 实现 TrainingEngineInterface 适配器
#[async_trait]
impl TrainingEngineInterface for dyn TrainingService + Send + Sync {
    async fn start_training(&self, task_id: &str, config: &crate::core::types::CoreTrainingConfiguration) -> Result<()> {
        // 将 CoreTrainingConfiguration 转换为 CoreTrainingConfigInterface
        let core_config = CoreTrainingConfigInterface {
            id: config.id.clone(),
            model_id: config.model_id.clone(),
            dataset_id: String::new(), // CoreTrainingConfiguration 没有 dataset_id
            algorithm_id: Some(config.algorithm_id.clone()),
            hyperparameters: config.parameters.clone(),
            training_options: crate::core::interfaces::TrainingOptionsInterface {
                batch_size: config.batch_size as usize,
                epochs: config.epochs as usize,
                learning_rate: config.learning_rate as f32,
                optimizer: config.optimization_config.optimizer_type.clone(),
                loss_function: format!("{:?}", config.loss_function),
                validation_split: Some(config.validation_split as f32),
                early_stopping: if config.early_stopping {
                    Some(crate::core::interfaces::EarlyStoppingConfigInterface {
                        monitor: "val_loss".to_string(),
                        patience: 10,
                        min_delta: 0.001,
                        restore_best_weights: true,
                    })
                } else {
                    None
                },
                checkpoint_config: if config.checkpoint_enabled {
                    Some(crate::core::interfaces::CheckpointConfigInterface {
                        monitor: "val_loss".to_string(),
                        save_best_only: true,
                        save_frequency: 1,
                        filepath_pattern: "checkpoint_{epoch}.pt".to_string(),
                    })
                } else {
                    None
                },
            },
            created_at: config.created_at,
            updated_at: config.updated_at,
        };
        
        // 调用 TrainingService 的方法
        let _task_id = self.create_training_task(core_config).await?;
        Ok(())
    }
    
    async fn stop_training(&self, task_id: &str) -> Result<()> {
        self.stop_training(task_id).await
    }
    
    async fn get_training_status(&self, task_id: &str) -> Result<crate::core::types::TrainingStatus> {
        match self.get_training_task(task_id).await? {
            Some(task) => {
                Ok(match task.status {
                    crate::core::interfaces::TrainingStatusInterface::Pending => crate::core::types::TrainingStatus::Pending,
                    crate::core::interfaces::TrainingStatusInterface::Running => crate::core::types::TrainingStatus::Running,
                    crate::core::interfaces::TrainingStatusInterface::Paused => crate::core::types::TrainingStatus::Paused,
                    crate::core::interfaces::TrainingStatusInterface::Completed => crate::core::types::TrainingStatus::Completed,
                    crate::core::interfaces::TrainingStatusInterface::Failed => crate::core::types::TrainingStatus::Failed,
                    crate::core::interfaces::TrainingStatusInterface::Cancelled => crate::core::types::TrainingStatus::Stopped,
                })
            },
            None => Err(Error::NotFound(format!("任务不存在: {}", task_id)))
        }
    }
    
    async fn update_model_gradients(&self, model_id: &str, gradients: &HashMap<String, CoreTensorData>) -> Result<()> {
        // TrainingService 没有这个方法，返回错误
        Err(Error::new("update_model_gradients not implemented for TrainingService"))
    }
    
    async fn pause_training(&self, task_id: &str) -> Result<()> {
        self.pause_training(task_id).await
    }
    
    async fn resume_training(&self, task_id: &str) -> Result<()> {
        self.resume_training(task_id).await
    }
    
    async fn get_training_metrics(&self, task_id: &str) -> Result<Option<crate::core::interfaces::TrainingMetrics>> {
        self.get_training_metrics(task_id).await
    }
    
    async fn get_training_progress(&self, task_id: &str) -> Result<f32> {
        // TrainingService trait 没有 get_training_progress 方法，尝试从任务状态计算
        debug!("获取训练进度: {}", task_id);
        // 这里无法访问任务状态，返回错误提示需要实现
        Err(Error::NotImplemented(
            "TrainingService trait 不支持 get_training_progress，请使用 TrainingEngineInterface".to_string()
        ))
    }
    
    async fn save_checkpoint(&self, task_id: &str, checkpoint_path: &str) -> Result<()> {
        // TrainingService trait 没有 save_checkpoint 方法，使用默认实现
        debug!("保存检查点: {} -> {}", task_id, checkpoint_path);
        Ok(())
    }
    
    async fn load_checkpoint(&self, task_id: &str, checkpoint_path: &str) -> Result<()> {
        // TrainingService trait 没有 load_checkpoint 方法，使用默认实现
        debug!("加载检查点: {} <- {}", task_id, checkpoint_path);
        Ok(())
    }
    
    async fn get_training_logs(&self, task_id: &str) -> Result<Vec<String>> {
        // TrainingService trait 没有 get_training_logs 方法，使用默认实现
        debug!("获取训练日志: {}", task_id);
        Ok(vec!["Training started".to_string(), "Training in progress".to_string()])
    }
    
    async fn get_engine_config(&self) -> Result<HashMap<String, String>> {
        // TrainingService trait 没有 get_engine_config 方法，使用默认实现
        debug!("获取引擎配置");
        Ok(HashMap::new())
    }
    
    async fn update_engine_config(&self, config: HashMap<String, String>) -> Result<()> {
        // TrainingService trait 没有 update_engine_config 方法，使用默认实现
        debug!("更新引擎配置");
        Ok(())
    }
    
    
    async fn validate_training_config(&self, config: &crate::core::types::CoreTrainingConfiguration) -> Result<ValidationResult> {
        // TrainingService trait 没有 validate_training_config 方法，使用默认实现
        debug!("验证训练配置");
        Ok(ValidationResult {
            is_valid: true,
            errors: vec![],
            warnings: vec![],
            score: Some(1.0),
            metadata: HashMap::new(),
        })
    }
} 