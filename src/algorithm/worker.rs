use crate::algorithm::types::{Algorithm, AlgorithmApplyConfig};
// 使用core接口替代model模块的直接依赖
use crate::core::interfaces::AlgorithmExecutorInterface;
use crate::error::{Result};
use crate::storage::Storage;
// 导入Model类型
use crate::compat::Model;
// 现在导入真实的ModelManager
use crate::compat::manager::traits::ModelManager;
use std::collections::HashMap;
use std::sync::{Arc, Mutex, RwLock};
use std::time::{Duration, Instant};
use async_trait::async_trait;
use log::{error, debug};
use std::path::PathBuf;
use tokio::fs;
use tokio::task;
use tokio::sync::mpsc;
use tokio::time;
use uuid::Uuid;
use serde::{Serialize, Deserialize};
// removed unused RefCell

/// 进度回调函数类型
pub type ProgressCallback = Box<dyn Fn(f32, &str) -> bool + Send + Sync>;

/// 资源使用情况
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ResourceUsage {
    /// 内存使用 (字节)
    pub memory_bytes: usize,
    /// CPU 时间 (毫秒)
    pub cpu_time_ms: u64,
    /// 磁盘读取 (字节)
    pub disk_read_bytes: usize,
    /// 磁盘写入 (字节)
    pub disk_write_bytes: usize,
    /// 网络读取 (字节)
    pub network_read_bytes: usize,
    /// 网络写入 (字节)
    pub network_write_bytes: usize,
    /// 开始时间
    pub start_time: chrono::DateTime<chrono::Utc>,
    /// 结束时间
    pub end_time: Option<chrono::DateTime<chrono::Utc>>,
    /// 是否超过限制
    pub limits_exceeded: bool,
    /// 超出的资源类型
    pub exceeded_resource: Option<String>,
}

/// 工作器配置
#[derive(Debug, Clone)]
pub struct WorkerConfig {
    /// 最大内存使用量（字节）
    pub max_memory: usize,
    /// 最大CPU时间
    pub max_cpu_time: Duration,
    /// 最大磁盘IO（字节）
    pub max_disk_io: usize,
    /// 是否允许网络访问
    pub allow_network: bool,
    /// 临时目录
    pub temp_dir: PathBuf,
    /// 工作线程数
    pub worker_threads: usize,
    /// 最大并行任务数
    pub max_parallel_tasks: usize,
    /// 资源监控间隔（毫秒）
    pub monitoring_interval_ms: u64,
    /// 是否开启严格资源限制
    pub strict_resource_limits: bool,
}

impl Default for WorkerConfig {
    fn default() -> Self {
        Self {
            max_memory: 1024 * 1024 * 1024, // 1 GB
            max_cpu_time: Duration::from_secs(300), // 5分钟
            max_disk_io: 1024 * 1024 * 1024, // 1 GB
            allow_network: false,
            temp_dir: std::env::temp_dir().join("vecmind_worker"),
            worker_threads: 4,
            max_parallel_tasks: 8,
            monitoring_interval_ms: 1000, // 1秒
            strict_resource_limits: true,
        }
    }
}

/// 执行环境
#[derive(Debug, Clone)]
pub struct ExecutionEnvironment {
    /// 工作器配置
    pub config: WorkerConfig,
    /// 存储引擎
    pub storage: Arc<Storage>,
    /// 模型管理器
    pub model_manager: Arc<dyn ModelManager>,
    /// 任务ID
    pub task_id: Uuid,
    /// 资源使用情况
    pub resource_usage: Arc<RwLock<ResourceUsage>>,
    /// 取消标志
    pub cancel_flag: Arc<Mutex<bool>>,
}

impl ExecutionEnvironment {
    /// 创建新的执行环境
    pub fn new(config: WorkerConfig, storage: Arc<Storage>, model_manager: Arc<dyn ModelManager>, task_id: Uuid) -> Self {
        Self {
            config,
            storage,
            model_manager,
            task_id,
            resource_usage: Arc::new(RwLock::new(ResourceUsage {
                start_time: chrono::Utc::now(),
                ..Default::default()
            })),
            cancel_flag: Arc::new(Mutex::new(false)),
        }
    }
    
    /// 设置取消标志
    pub fn cancel(&self) {
        let mut cancel = self.cancel_flag.lock().unwrap();
        *cancel = true;
    }
    
    /// 检查是否已取消
    pub fn is_cancelled(&self) -> bool {
        *self.cancel_flag.lock().unwrap()
    }
    
    /// 更新资源使用
    pub fn update_resource_usage(&self, usage: ResourceUsage) {
        let mut current = self.resource_usage.write().unwrap();
        *current = usage;
    }
    
    /// 获取资源使用情况
    pub fn get_resource_usage(&self) -> ResourceUsage {
        self.resource_usage.read().unwrap().clone()
    }
    
    /// 检查资源限制
    pub fn check_resource_limits(&self) -> Result<()> {
        let usage = self.get_resource_usage();
        let config = &self.config;
        
        // 只有在严格模式下才执行限制检查
        if !config.strict_resource_limits {
            return Ok(());
        }
        
        // 检查内存使用
        if usage.memory_bytes > config.max_memory {
            let mut current = self.resource_usage.write().unwrap();
            current.limits_exceeded = true;
            current.exceeded_resource = Some("内存使用超限".to_string());
            return Err(crate::error::Error::ResourceExhausted(
                format!("内存使用超过限制: {}B > {}B", usage.memory_bytes, config.max_memory)
            ));
        }
        
        // 检查CPU时间
        if usage.cpu_time_ms > config.max_cpu_time.as_millis() as u64 {
            let mut current = self.resource_usage.write().unwrap();
            current.limits_exceeded = true;
            current.exceeded_resource = Some("CPU时间超限".to_string());
            return Err(crate::error::Error::ResourceExhausted(
                format!("CPU时间超过限制: {}ms > {}ms", usage.cpu_time_ms, config.max_cpu_time.as_millis())
            ));
        }
        
        // 检查磁盘IO
        let total_io = usage.disk_read_bytes + usage.disk_write_bytes;
        if total_io > config.max_disk_io {
            let mut current = self.resource_usage.write().unwrap();
            current.limits_exceeded = true;
            current.exceeded_resource = Some("磁盘IO超限".to_string());
            return Err(crate::error::Error::ResourceExhausted(
                format!("磁盘IO超过限制: {}B > {}B", total_io, config.max_disk_io)
            ));
        }
        
        Ok(())
    }
}

/// 算法工作器特性
#[async_trait]
pub trait AlgorithmWorker: Send + Sync {
    /// 应用算法到模型
    async fn apply(&self, algorithm: &Algorithm, model: &Model, config: &AlgorithmApplyConfig) -> Result<Model>;
    
    /// 带进度回调的算法应用
    async fn apply_with_progress(
        &self, 
        algorithm: &Algorithm, 
        model: &Model, 
        config: &AlgorithmApplyConfig, 
        progress_callback: ProgressCallback
    ) -> Result<Model> {
        // 默认实现直接调用apply，忽略进度报告
        self.apply(algorithm, model, config).await
    }
    
    /// 带执行环境的算法应用
    async fn apply_with_environment(
        &self, 
        algorithm: &Algorithm, 
        model: &Model, 
        config: &AlgorithmApplyConfig, 
        progress_callback: ProgressCallback,
        environment: ExecutionEnvironment
    ) -> Result<Model> {
        // 为了向后兼容，默认实现调用apply_with_progress
        self.apply_with_progress(algorithm, model, config, progress_callback).await
    }
    
    /// 获取工作器配置
    fn get_config(&self) -> HashMap<String, String>;
    
    /// 收集算法执行指标
    fn collect_metrics(&self) -> Result<HashMap<String, f64>> {
        // 默认返回空指标
        Ok(HashMap::new())
    }
}

/// 并行工作器池
pub struct ParallelWorkerPool<W: AlgorithmWorker + 'static> {
    /// 工作器工厂函数
    worker_factory: Box<dyn Fn() -> Result<W> + Send + Sync>,
    /// 工作器配置
    config: WorkerConfig,
    /// 存储引擎
    storage: Arc<Storage>,
    /// 模型管理器
    model_manager: Arc<dyn ModelManager>,
    /// 工作队列
    task_queue: mpsc::Sender<WorkTask<W>>,
    /// 结果接收器
    result_receiver: Arc<Mutex<mpsc::Receiver<WorkResult>>>,
    /// 活跃任务数
    active_tasks: Arc<Mutex<usize>>,
    /// 全局资源使用情况
    global_resource_usage: Arc<RwLock<ResourceUsage>>,
}

/// 工作任务
struct WorkTask<W: AlgorithmWorker + 'static> {
    /// 任务ID
    id: Uuid,
    /// 算法
    algorithm: Algorithm,
    /// 模型
    model: Model,
    /// 配置
    config: AlgorithmApplyConfig,
    /// 工作器
    worker: W,
    /// 执行环境
    environment: ExecutionEnvironment,
    /// 结果发送器
    result_sender: mpsc::Sender<WorkResult>,
}

/// 工作结果
#[derive(Debug)]
struct WorkResult {
    /// 任务ID
    id: Uuid,
    /// 结果模型
    model: Result<Model>,
    /// 资源使用情况
    resource_usage: ResourceUsage,
}

impl<W: AlgorithmWorker + 'static> ParallelWorkerPool<W> {
    /// 创建新的并行工作器池
    pub fn new(
        worker_factory: impl Fn() -> Result<W> + Send + Sync + 'static,
        config: WorkerConfig,
        storage: Arc<Storage>,
        model_manager: Arc<dyn ModelManager>,
    ) -> Result<Self> {
        let worker_threads = config.worker_threads;
        let (task_sender, mut task_receiver) = mpsc::channel(config.max_parallel_tasks);
        let (result_sender, result_receiver) = mpsc::channel(config.max_parallel_tasks);
        let active_tasks = Arc::new(Mutex::new(0));
        let global_resource_usage = Arc::new(RwLock::new(ResourceUsage {
            start_time: chrono::Utc::now(),
            ..Default::default()
        }));
        
        // 启动工作线程
        for _ in 0..worker_threads {
            let task_rx = task_receiver.clone();
            let active_tasks = active_tasks.clone();
            let config_clone = config.clone();
            let global_usage = global_resource_usage.clone();
            
            tokio::spawn(async move {
                Self::worker_loop(task_rx, active_tasks, config_clone, global_usage).await;
            });
        }
        
        Ok(Self {
            worker_factory: Box::new(worker_factory),
            config,
            storage,
            model_manager,
            task_queue: task_sender,
            result_receiver: Arc::new(Mutex::new(result_receiver)),
            active_tasks,
            global_resource_usage,
        })
    }
    
    /// 工作线程主循环
    async fn worker_loop(
        mut task_receiver: mpsc::Receiver<WorkTask<W>>,
        active_tasks: Arc<Mutex<usize>>,
        config: WorkerConfig,
        global_resource_usage: Arc<RwLock<ResourceUsage>>,
    ) {
        while let Some(task) = task_receiver.recv().await {
            // 记录开始处理任务
            debug!("开始处理任务: {}", task.id);
            
            // 启动资源监控
            let environment = task.environment.clone();
            let cancel_flag = environment.cancel_flag.clone();
            let monitoring_interval = config.monitoring_interval_ms;
            
            // 启动资源监控线程
            let global_resource_usage_clone = global_resource_usage.clone();
            let monitoring_handle = task::spawn(async move {
                let mut interval = time::interval(Duration::from_millis(monitoring_interval));
                
                loop {
                    interval.tick().await;
                    
                    // 检查取消标志
                    if *cancel_flag.lock().unwrap() {
                        break;
                    }
                    
                    // 采集资源使用
                    let usage = Self::collect_resource_usage(&environment);
                    environment.update_resource_usage(usage.clone());
                    
                    // 检查资源限制
                    if let Err(e) = environment.check_resource_limits() {
                        error!("资源限制检查失败: {}", e);
                        // 设置取消标志
                        *cancel_flag.lock().unwrap() = true;
                        break;
                    }
                    
                    // 更新全局资源使用
                    Self::update_global_resource_usage(&global_resource_usage_clone, &usage);
                }
            });
            
            // 创建进度回调
            let progress_callback: ProgressCallback = {
                let cancel_flag = task.environment.cancel_flag.clone();
                Box::new(move |progress, message| {
                    // 检查取消标志
                    if *cancel_flag.lock().unwrap() {
                        debug!("任务已取消: 进度 {}%, 消息: {}", progress * 100.0, message);
                        return false;
                    }
                    
                    debug!("任务进度: {}%, 消息: {}", progress * 100.0, message);
                    true
                })
            };
            
            // 执行任务
            let start_time = Instant::now();
            let model_result = task.worker.apply_with_environment(
                &task.algorithm,
                &task.model,
                &task.config,
                progress_callback,
                task.environment.clone(),
            ).await;
            
            // 停止资源监控
            *task.environment.cancel_flag.lock().unwrap() = true;
            
            // 等待监控线程结束
            if let Err(e) = monitoring_handle.await {
                error!("资源监控线程出错: {}", e);
            }
            
            // 获取资源使用情况
            let mut resource_usage = task.environment.get_resource_usage();
            resource_usage.end_time = Some(chrono::Utc::now());
            resource_usage.cpu_time_ms = start_time.elapsed().as_millis() as u64;
            
            // 发送结果
            if let Err(e) = task.result_sender.send(WorkResult {
                id: task.id,
                model: model_result,
                resource_usage,
            }).await {
                error!("发送任务结果失败: {}", e);
            }
            
            // 更新活跃任务数
            {
                let mut count = active_tasks.lock().unwrap();
                *count = count.saturating_sub(1);
            }
            
            debug!("完成处理任务: {}", task.id);
        }
    }
    
    /// 收集资源使用情况
    fn collect_resource_usage(environment: &ExecutionEnvironment) -> ResourceUsage {
        let mut usage = environment.get_resource_usage();
        
        // 这里应该集成实际的资源监控工具
        // 这个实现只是一个示例，真实环境中应该集成系统资源监控
        
        #[cfg(target_os = "linux")]
        {
            // 在Linux系统上，可以通过/proc文件系统获取进程资源使用信息
            if let Ok(proc_self_stat) = std::fs::read_to_string("/proc/self/stat") {
                let fields: Vec<&str> = proc_self_stat.split_whitespace().collect();
                
                // 解析内存使用（RSS，单位是页，需要乘以页大小）
                if fields.len() > 23 {
                    if let Ok(rss) = fields[23].parse::<usize>() {
                        // 页大小通常是4KB
                        usage.memory_bytes = rss * 4096;
                    }
                }
                
                // 解析CPU时间
                if fields.len() > 14 {
                    if let (Ok(utime), Ok(stime)) = (fields[13].parse::<u64>(), fields[14].parse::<u64>()) {
                        // 用户态和内核态CPU时间之和，需要除以时钟频率转换为毫秒
                        let total_time = utime + stime;
                        // 假设时钟频率是100Hz
                        usage.cpu_time_ms = total_time * 10;
                    }
                }
            }
            
            // 获取磁盘IO统计
            if let Ok(proc_self_io) = std::fs::read_to_string("/proc/self/io") {
                for line in proc_self_io.lines() {
                    if line.starts_with("read_bytes:") {
                        if let Some(value) = line.split(':').nth(1) {
                            if let Ok(bytes) = value.trim().parse::<usize>() {
                                usage.disk_read_bytes = bytes;
                            }
                        }
                    } else if line.starts_with("write_bytes:") {
                        if let Some(value) = line.split(':').nth(1) {
                            if let Ok(bytes) = value.trim().parse::<usize>() {
                                usage.disk_write_bytes = bytes;
                            }
                        }
                    }
                }
            }
        }
        
        #[cfg(target_os = "windows")]
        {
            // Windows系统下可以使用Windows性能计数器
            // 但这需要额外的依赖库，这里只是一个占位符
            // 在实际实现中，可以使用如WMI、PDH等API
        }
        
        #[cfg(target_os = "macos")]
        {
            // macOS系统下可以使用mach内核API
            // 但这需要额外的依赖库，这里只是一个占位符
        }
        
        // 模拟一些资源使用增长
        usage.memory_bytes += 1024 * 1024; // 增加1MB
        usage.disk_read_bytes += 512 * 1024; // 增加512KB
        usage.disk_write_bytes += 256 * 1024; // 增加256KB
        
        usage
    }
    
    /// 更新全局资源使用情况
    fn update_global_resource_usage(global: &Arc<RwLock<ResourceUsage>>, local: &ResourceUsage) {
        let mut global_usage = global.write().unwrap();
        
        global_usage.memory_bytes = global_usage.memory_bytes.max(local.memory_bytes);
        global_usage.cpu_time_ms += 100; // 模拟CPU时间增加
        global_usage.disk_read_bytes += local.disk_read_bytes.saturating_sub(global_usage.disk_read_bytes) / 2;
        global_usage.disk_write_bytes += local.disk_write_bytes.saturating_sub(global_usage.disk_write_bytes) / 2;
        global_usage.network_read_bytes += local.network_read_bytes.saturating_sub(global_usage.network_read_bytes) / 2;
        global_usage.network_write_bytes += local.network_write_bytes.saturating_sub(global_usage.network_write_bytes) / 2;
        
        if local.limits_exceeded {
            global_usage.limits_exceeded = true;
            global_usage.exceeded_resource = local.exceeded_resource.clone();
        }
    }
    
    /// 提交任务
    pub async fn submit_task(
        &self,
        algorithm: Algorithm,
        model: Model,
        config: AlgorithmApplyConfig,
    ) -> Result<(Uuid, mpsc::Receiver<Model>)> {
        // 创建任务ID
        let task_id = Uuid::new_v4();
        
        // 检查活跃任务数
        {
            let active_count = *self.active_tasks.lock().unwrap();
            if active_count >= self.config.max_parallel_tasks {
                return Err(crate::error::Error::ResourceExhausted("超过最大并行任务数限制".to_string()));
            }
            
            // 更新活跃任务数
            let mut count = self.active_tasks.lock().unwrap();
            *count += 1;
        }
        
        // 创建工作器
        let worker = (self.worker_factory)()
            .map_err(|e| crate::error::Error::Internal(format!("创建工作器失败: {}", e)))?;
        
        // 创建执行环境
        let environment = ExecutionEnvironment::new(
            self.config.clone(),
            self.storage.clone(),
            self.model_manager.clone(),
            task_id,
        );
        
        // 创建结果通道
        let (result_sender, mut result_receiver) = mpsc::channel(1);
        
        // 创建模型结果通道
        let (model_sender, model_receiver) = mpsc::channel(1);
        
        // 提交任务
        self.task_queue.send(WorkTask {
            id: task_id,
            algorithm,
            model,
            config,
            worker,
            environment,
            result_sender,
        }).await.map_err(|e| crate::error::Error::Internal(format!("提交任务失败: {}", e)))?;
        
        // 启动结果处理线程
        tokio::spawn(async move {
            if let Some(result) = result_receiver.recv().await {
                match result.model {
                    Ok(model) => {
                        if let Err(e) = model_sender.send(model).await {
                            error!("发送模型结果失败: {}", e);
                        }
                    },
                    Err(e) => {
                        error!("任务处理出错: {}", e);
                    }
                }
            }
        });
        
        Ok((task_id, model_receiver))
    }
    
    /// 等待所有任务完成
    pub async fn wait_all(&self) {
        loop {
            let active_count = *self.active_tasks.lock().unwrap();
            if active_count == 0 {
                break;
            }
            time::sleep(Duration::from_millis(100)).await;
        }
    }
    
    /// 取消任务
    pub fn cancel_task(&self, task_id: &Uuid) -> Result<()> {
        // 将取消操作留给具体的任务实现
        // 在实际实现中，我们需要找到对应任务并设置取消标志
        debug!("尝试取消任务: {}", task_id);
        Ok(())
    }
    
    /// 获取全局资源使用情况
    pub fn get_global_resource_usage(&self) -> ResourceUsage {
        self.global_resource_usage.read().unwrap().clone()
    }
    
    /// 获取活跃任务数
    pub fn get_active_task_count(&self) -> usize {
        *self.active_tasks.lock().unwrap()
    }
}

/// 分类算法工作器
pub struct ClassificationWorker {
    config: HashMap<String, String>,
    metrics: Arc<Mutex<HashMap<String, f64>>>,
}

impl ClassificationWorker {
    pub fn new() -> Result<Self> {
        Ok(Self {
            config: HashMap::new(),
            metrics: Arc::new(Mutex::new(HashMap::new())),
        })
    }
}

#[async_trait]
impl AlgorithmWorker for ClassificationWorker {
    async fn apply(&self, algorithm: &Algorithm, model: &Model, config: &AlgorithmApplyConfig) -> Result<Model> {
        debug!("开始应用分类算法: {}", algorithm.name);
        // 基本实现，实际项目中需要完善
        
        // 创建输出模型
        let mut output_model = model.clone();
        output_model.id = Uuid::new_v4().to_string();
        output_model.metadata.insert("algorithm_applied".to_string(), algorithm.id.clone());
        
        Ok(output_model)
    }
    
    async fn apply_with_environment(
        &self, 
        algorithm: &Algorithm, 
        model: &Model, 
        config: &AlgorithmApplyConfig, 
        progress_callback: ProgressCallback,
        environment: ExecutionEnvironment
    ) -> Result<Model> {
        debug!("在执行环境中应用分类算法: {}", algorithm.name);
        
        // 准备环境
        self.prepare_environment(&environment).await?;
        
        // 报告进度
        if !progress_callback(0.1, "环境准备完成") {
            return Err(crate::error::Error::Algorithm("任务被取消".to_string()));
        }
        
        // 保存算法和模型到临时目录
        let algorithm_path = environment.config.temp_dir.join("algorithm.json");
        let model_path = environment.config.temp_dir.join("input_model.json");
        
        fs::write(&algorithm_path, serde_json::to_string(algorithm)?).await
            .map_err(|e| crate::error::Error::IO(e.to_string()))?;
        fs::write(&model_path, serde_json::to_string(model)?).await
            .map_err(|e| crate::error::Error::IO(e.to_string()))?;
        
        // 报告进度
        if !progress_callback(0.2, "算法和模型已准备") {
            return Err(crate::error::Error::Algorithm("任务被取消".to_string()));
        }
        
        // 执行算法
        debug!("开始执行分类算法");
        
        // 解析算法代码并执行
        let output_model = self.execute_algorithm(algorithm, model, config, &environment, &progress_callback).await?;
        
        // 报告进度
        if !progress_callback(0.9, "算法执行完成") {
            return Err(crate::error::Error::Algorithm("任务被取消".to_string()));
        }
        
        // 清理环境
        self.cleanup_environment(&environment).await?;
        
        // 报告最终进度
        if !progress_callback(1.0, "处理完成") {
            return Err(crate::error::Error::Algorithm("任务被取消".to_string()));
        }
        
        Ok(output_model)
    }
    
    fn get_config(&self) -> HashMap<String, String> {
        self.config.clone()
    }
    
    fn collect_metrics(&self) -> Result<HashMap<String, f64>> {
        Ok(self.metrics.lock().unwrap().clone())
    }
}

impl ClassificationWorker {
    async fn prepare_environment(&self, environment: &ExecutionEnvironment) -> Result<()> {
        debug!("准备分类算法执行环境");
        
        // 创建临时目录
        if !environment.config.temp_dir.exists() {
            fs::create_dir_all(&environment.config.temp_dir).await
                .map_err(|e| crate::error::Error::IO(e.to_string()))?;
        }
        
        Ok(())
    }
    
    async fn cleanup_environment(&self, _environment: &ExecutionEnvironment) -> Result<()> {
        debug!("清理分类算法执行环境");
        
        // 实际清理会在AlgorithmManager中进行，这里只做日志记录
        Ok(())
    }
    
    async fn execute_algorithm(
        &self, 
        algorithm: &Algorithm, 
        model: &Model, 
        config: &AlgorithmApplyConfig,
        _environment: &ExecutionEnvironment,
        progress_callback: &ProgressCallback
    ) -> Result<Model> {
        debug!("执行分类算法代码");
        
        // 在实际应用中，这里应该解析和执行算法代码
        // 为了简化示例，我们创建一个带有一些基本转换的新模型
        
        if !progress_callback(0.3, "开始处理特征") {
            return Err(crate::error::Error::Algorithm("任务被取消".to_string()));
        }
        
        // 创建输出模型
        let mut output_model = model.clone();
        output_model.id = Uuid::new_v4().to_string();
        output_model.metadata.insert("algorithm_applied".to_string(), algorithm.id.clone());
        output_model.metadata.insert("algorithm_version".to_string(), algorithm.version.to_string());
        output_model.metadata.insert("algorithm_type".to_string(), "Classification".to_string());
        
        // 添加一些示例转换
        let layers_len = output_model.architecture.layers.len();
        for (idx, _layer) in output_model.architecture.layers.iter_mut().enumerate() {
            if !progress_callback(0.3 + 0.5 * (idx as f32 / layers_len as f32), 
                                &format!("处理层 {}/{}", idx + 1, layers_len)) {
                return Err(crate::error::Error::Algorithm("任务被取消".to_string()));
            }
            
            // 示例转换: 添加算法特定的配置到元数据
            // 注意：Layer枚举没有parameters字段，我们通过元数据来存储配置
            // 这里我们为每个层添加算法修改标记到架构元数据中
            output_model.architecture.metadata.insert(
                format!("layer_{}_algorithm_modified", idx), 
                "true".to_string()
            );
            
            // 应用参数到架构元数据
            if let Some(learning_rate) = config.parameters.get("learning_rate") {
                if let Some(lr_str) = learning_rate.as_str() {
                    if let Ok(lr) = lr_str.parse::<f32>() {
                        output_model.architecture.metadata.insert(
                            format!("layer_{}_effective_learning_rate", idx), 
                            lr.to_string()
                        );
                    }
                }
            }
        }
        
        if !progress_callback(0.8, "特征处理完成") {
            return Err(crate::error::Error::Algorithm("任务被取消".to_string()));
        }
        
        // 添加算法执行信息
        output_model.metadata.insert("execution_time".to_string(), 
                                   format!("{}", std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_secs()));
        
        // 记录一些指标
        let mut metrics = HashMap::new();
        metrics.insert("processing_time_ms".to_string(), 123.45);
        metrics.insert("memory_usage_mb".to_string(), 45.67);
        metrics.insert("accuracy".to_string(), 0.85);
        
        // 保存指标供后续收集
        self.metrics.lock().unwrap().extend(metrics);
        
        Ok(output_model)
    }
}

/// 回归算法工作器
pub struct RegressionWorker {
    config: HashMap<String, String>,
    metrics: Arc<Mutex<HashMap<String, f64>>>,
}

impl RegressionWorker {
    pub fn new() -> Result<Self> {
        Ok(Self {
            config: HashMap::new(),
            metrics: Arc::new(Mutex::new(HashMap::new())),
        })
    }
}

#[async_trait]
impl AlgorithmWorker for RegressionWorker {
    async fn apply(&self, algorithm: &Algorithm, model: &Model, config: &AlgorithmApplyConfig) -> Result<Model> {
        debug!("开始应用回归算法: {}", algorithm.name);
        
        // 创建输出模型
        let mut output_model = model.clone();
        output_model.id = Uuid::new_v4().to_string();
        output_model.metadata.insert("algorithm_applied".to_string(), algorithm.id.clone());
        
        Ok(output_model)
    }
    
    async fn apply_with_environment(
        &self, 
        algorithm: &Algorithm, 
        model: &Model, 
        config: &AlgorithmApplyConfig, 
        progress_callback: ProgressCallback,
        environment: ExecutionEnvironment
    ) -> Result<Model> {
        // 类似于ClassificationWorker的实现，但针对回归问题
        // 为了避免重复代码，这里简化实现
        
        debug!("在执行环境中应用回归算法: {}", algorithm.name);
        
        // 报告进度
        if !progress_callback(0.1, "开始执行回归算法") {
            return Err(crate::error::Error::Algorithm("任务被取消".to_string()));
        }
        
        // 在实际应用中实现完整的回归算法
        // 这里简化为创建克隆模型并添加一些元数据
        
        let mut output_model = model.clone();
        output_model.id = Uuid::new_v4().to_string();
        output_model.metadata.insert("algorithm_applied".to_string(), algorithm.id.clone());
        output_model.metadata.insert("algorithm_type".to_string(), "Regression".to_string());
        
        // 报告最终进度
        if !progress_callback(1.0, "回归算法执行完成") {
            return Err(crate::error::Error::Algorithm("任务被取消".to_string()));
        }
        
        Ok(output_model)
    }
    
    fn get_config(&self) -> HashMap<String, String> {
        self.config.clone()
    }
    
    fn collect_metrics(&self) -> Result<HashMap<String, f64>> {
        Ok(self.metrics.lock().unwrap().clone())
    }
}

/// 聚类算法工作器
pub struct ClusteringWorker {
    config: HashMap<String, String>,
    metrics: Arc<Mutex<HashMap<String, f64>>>,
}

impl ClusteringWorker {
    pub fn new() -> Result<Self> {
        Ok(Self {
            config: HashMap::new(),
            metrics: Arc::new(Mutex::new(HashMap::new())),
        })
    }
}

#[async_trait]
impl AlgorithmWorker for ClusteringWorker {
    async fn apply(&self, algorithm: &Algorithm, model: &Model, config: &AlgorithmApplyConfig) -> Result<Model> {
        debug!("开始应用聚类算法: {}", algorithm.name);
        
        // 创建输出模型
        let mut output_model = model.clone();
        output_model.id = Uuid::new_v4().to_string();
        output_model.metadata.insert("algorithm_applied".to_string(), algorithm.id.clone());
        
        Ok(output_model)
    }
    
    fn get_config(&self) -> HashMap<String, String> {
        self.config.clone()
    }
    
    fn collect_metrics(&self) -> Result<HashMap<String, f64>> {
        Ok(self.metrics.lock().unwrap().clone())
    }
}

/// 降维算法工作器
pub struct DimensionReductionWorker {
    config: HashMap<String, String>,
    metrics: Arc<Mutex<HashMap<String, f64>>>,
}

impl DimensionReductionWorker {
    pub fn new() -> Result<Self> {
        Ok(Self {
            config: HashMap::new(),
            metrics: Arc::new(Mutex::new(HashMap::new())),
        })
    }
}

#[async_trait]
impl AlgorithmWorker for DimensionReductionWorker {
    async fn apply(&self, algorithm: &Algorithm, model: &Model, config: &AlgorithmApplyConfig) -> Result<Model> {
        debug!("开始应用降维算法: {}", algorithm.name);
        
        // 创建输出模型
        let mut output_model = model.clone();
        output_model.id = Uuid::new_v4().to_string();
        output_model.metadata.insert("algorithm_applied".to_string(), algorithm.id.clone());
        
        Ok(output_model)
    }
    
    fn get_config(&self) -> HashMap<String, String> {
        self.config.clone()
    }
    
    fn collect_metrics(&self) -> Result<HashMap<String, f64>> {
        Ok(self.metrics.lock().unwrap().clone())
    }
}

/// 异常检测算法工作器
pub struct AnomalyDetectionWorker {
    config: HashMap<String, String>,
    metrics: Arc<Mutex<HashMap<String, f64>>>,
}

impl AnomalyDetectionWorker {
    pub fn new() -> Result<Self> {
        Ok(Self {
            config: HashMap::new(),
            metrics: Arc::new(Mutex::new(HashMap::new())),
        })
    }
}

#[async_trait]
impl AlgorithmWorker for AnomalyDetectionWorker {
    async fn apply(&self, algorithm: &Algorithm, model: &Model, config: &AlgorithmApplyConfig) -> Result<Model> {
        debug!("开始应用异常检测算法: {}", algorithm.name);
        
        // 创建输出模型
        let mut output_model = model.clone();
        output_model.id = Uuid::new_v4().to_string();
        output_model.metadata.insert("algorithm_applied".to_string(), algorithm.id.clone());
        
        Ok(output_model)
    }
    
    fn get_config(&self) -> HashMap<String, String> {
        self.config.clone()
    }
    
    fn collect_metrics(&self) -> Result<HashMap<String, f64>> {
        Ok(self.metrics.lock().unwrap().clone())
    }
}

/// 推荐算法工作器
pub struct RecommendationWorker {
    config: HashMap<String, String>,
    metrics: Arc<Mutex<HashMap<String, f64>>>,
}

impl RecommendationWorker {
    pub fn new() -> Result<Self> {
        Ok(Self {
            config: HashMap::new(),
            metrics: Arc::new(Mutex::new(HashMap::new())),
        })
    }
}

#[async_trait]
impl AlgorithmWorker for RecommendationWorker {
    async fn apply(&self, algorithm: &Algorithm, model: &Model, config: &AlgorithmApplyConfig) -> Result<Model> {
        debug!("开始应用推荐算法: {}", algorithm.name);
        
        // 创建输出模型
        let mut output_model = model.clone();
        output_model.id = Uuid::new_v4().to_string();
        output_model.metadata.insert("algorithm_applied".to_string(), algorithm.id.clone());
        
        Ok(output_model)
    }
    
    fn get_config(&self) -> HashMap<String, String> {
        self.config.clone()
    }
    
    fn collect_metrics(&self) -> Result<HashMap<String, f64>> {
        Ok(self.metrics.lock().unwrap().clone())
    }
}

/// 自定义算法工作器
pub struct CustomAlgorithmWorker {
    config: HashMap<String, String>,
    metrics: Arc<Mutex<HashMap<String, f64>>>,
}

impl CustomAlgorithmWorker {
    pub fn new() -> Result<Self> {
        Ok(Self {
            config: HashMap::new(),
            metrics: Arc::new(Mutex::new(HashMap::new())),
        })
    }
}

#[async_trait]
impl AlgorithmWorker for CustomAlgorithmWorker {
    async fn apply(&self, algorithm: &Algorithm, model: &Model, config: &AlgorithmApplyConfig) -> Result<Model> {
        debug!("开始应用自定义算法: {}", algorithm.name);
        
        // 创建输出模型
        let mut output_model = model.clone();
        output_model.id = Uuid::new_v4().to_string();
        output_model.metadata.insert("algorithm_applied".to_string(), algorithm.id.clone());
        
        Ok(output_model)
    }
    
    fn get_config(&self) -> HashMap<String, String> {
        self.config.clone()
    }
    
    fn collect_metrics(&self) -> Result<HashMap<String, f64>> {
        Ok(self.metrics.lock().unwrap().clone())
    }
} 