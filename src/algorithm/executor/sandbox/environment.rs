use std::sync::{Arc, Mutex};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::{Duration, Instant};
use std::io::Write;
use tokio::sync::mpsc;
#[cfg(feature = "wasmtime")]
use wasmtime::{Engine as WasmEngine, Store as WasmStore};
#[cfg(feature = "tempfile")]
use tempfile::TempDir;
use log::{debug, error, warn};

use crate::{Error, Result};
use crate::algorithm::executor::config::SandboxConfig;
use crate::algorithm::types::ResourceUsage;
use super::types::SecurityContext;

/// 监控命令
#[derive(Debug)]
pub enum MonitorCommand {
    /// 停止监控
    Stop,
    /// 获取资源使用情况
    GetResourceUsage(mpsc::Sender<ResourceUsage>),
    /// 检查资源限制
    CheckResourceLimits,
}

/// 执行环境，管理沙箱执行过程中的资源和状态
#[derive(Debug)]
pub struct ExecutionEnvironment {
    /// 沙箱配置
    pub sandbox_config: SandboxConfig,
    /// 资源使用情况
    pub resource_usage: Arc<Mutex<ResourceUsage>>,
    /// 安全上下文
    pub security_context: SecurityContext,
    /// 临时目录
    pub temp_dir: Option<TempDir>,
    /// 监控通道发送端 - 使用内部可变性
    monitor_tx: Arc<Mutex<Option<mpsc::Sender<MonitorCommand>>>>,
    /// 是否已取消
    cancelled: Arc<AtomicBool>,
    /// CPU时间(毫秒)
    pub cpu_time_ms: Arc<AtomicU64>,
    /// 内存使用(字节)
    pub memory_bytes: Arc<AtomicU64>,
    /// 磁盘读取字节数
    pub disk_read_bytes: Arc<AtomicU64>,
    /// 磁盘写入字节数
    pub disk_write_bytes: Arc<AtomicU64>,
    /// 网络传输字节数
    pub network_bytes: Arc<AtomicU64>,
    /// 标准输出缓冲
    stdout_buffer: Mutex<Vec<u8>>,
    /// 标准错误缓冲
    stderr_buffer: Mutex<Vec<u8>>,
    /// WASM引擎 - 使用内部可变性
    pub wasm_engine: Arc<Mutex<Option<Arc<WasmEngine>>>>,
    /// WASM存储 - 使用内部可变性
    pub wasm_store: Arc<Mutex<Option<WasmStore<wasmtime_wasi::WasiCtx>>>>,
    /// 任务ID
    pub task_id: String,
    /// 开始时间
    pub start_time: Instant,
}

impl ExecutionEnvironment {
    /// 创建新的执行环境
    pub fn new(sandbox_config: SandboxConfig, task_id: &str) -> Self {
        let security_context = SecurityContext {
            allow_network: sandbox_config.allow_network.unwrap_or(false),
            allow_filesystem: sandbox_config.allow_filesystem.unwrap_or(false),
            allowed_env_vars: sandbox_config.allowed_env_vars.clone(),
            allowed_paths: sandbox_config.allowed_paths.clone(),
            allow_stdin: sandbox_config.allow_stdin.unwrap_or(false),
            allow_stdout: sandbox_config.allow_stdout.unwrap_or(true),
            allow_stderr: sandbox_config.allow_stderr.unwrap_or(true),
            memory_limit_bytes: 128 * 1024 * 1024, // 默认128MB
            cpu_time_limit_ms: 10000, // 默认10秒
            disk_io_limit_bytes: Some(10 * 1024 * 1024), // 默认10MB
            enable_wasm_memory_limit: true,
            max_wasm_memory_pages: Some(100), // 约6.4MB
            max_parallelism: Some(4),
            security_level: sandbox_config.security_level.unwrap_or_default(),
            allow_syscalls: false,
            allowed_syscalls: vec![],
        };
        
        Self {
            sandbox_config,
            resource_usage: Arc::new(Mutex::new(ResourceUsage::default())),
            security_context,
            temp_dir: TempDir::new().ok(),
            monitor_tx: Arc::new(Mutex::new(None)),
            cancelled: Arc::new(AtomicBool::new(false)),
            cpu_time_ms: Arc::new(AtomicU64::new(0)),
            memory_bytes: Arc::new(AtomicU64::new(0)),
            disk_read_bytes: Arc::new(AtomicU64::new(0)),
            disk_write_bytes: Arc::new(AtomicU64::new(0)),
            network_bytes: Arc::new(AtomicU64::new(0)),
            stdout_buffer: Mutex::new(Vec::new()),
            stderr_buffer: Mutex::new(Vec::new()),
            wasm_engine: Arc::new(Mutex::new(None)),
            wasm_store: Arc::new(Mutex::new(None)),
            task_id: task_id.to_string(),
            start_time: Instant::now(),
        }
    }
    
    /// 创建WASM执行环境
    pub fn create_wasm_environment(&self) -> Result<()> {
        debug!("创建WASM执行环境: {}", self.task_id);
        
        // 创建WASM引擎
        #[cfg(feature = "wasmtime")]
        {
            let engine = Arc::new(WasmEngine::new(&wasmtime::Config::new()).map_err(|e| {
                Error::internal(format!("创建WASM引擎失败: {}", e))
            })?);
            
            // 创建WASI上下文
            let wasi_ctx = wasmtime_wasi::WasiCtxBuilder::new()
                .inherit_stdio()
                .inherit_env()
                .build();
        
            // 创建存储
            let store = WasmStore::new(&*engine, wasi_ctx);
            
            // 设置引擎和存储
            {
                let mut engine_guard = self.wasm_engine.lock().unwrap();
                *engine_guard = Some(engine);
            }
            
            {
                let mut store_guard = self.wasm_store.lock().unwrap();
                *store_guard = Some(store);
            }
        }
        #[cfg(not(feature = "wasmtime"))]
        {
            return Err(Error::internal("WASM support requires 'wasmtime' feature".to_string()));
        }
        
        Ok(())
    }
    
    /// 启动资源监控
    pub async fn start_monitoring(&self) -> Result<()> {
        debug!("启动资源监控: {}", self.task_id);
        
        let (tx, mut rx) = mpsc::channel(32);
        
        // 设置监控通道
        {
            let mut monitor_tx_guard = self.monitor_tx.lock().unwrap();
            *monitor_tx_guard = Some(tx);
        }
        
        // 设置监控任务 - 提前克隆所有需要的数据
        let task_id = self.task_id.clone();
        let cancelled = self.cancelled.clone();
        let cpu_time = self.cpu_time_ms.clone();
        let memory_bytes = self.memory_bytes.clone();
        let disk_read_bytes = self.disk_read_bytes.clone();
        let disk_write_bytes = self.disk_write_bytes.clone();
        let network_bytes = self.network_bytes.clone();
        let cpu_limit = self.security_context.cpu_time_limit_ms;
        let memory_limit = self.security_context.memory_limit_bytes;
        
        tokio::spawn(async move {
            let mut interval = tokio::time::interval(Duration::from_millis(100));
            let start_time = Instant::now();
            
            loop {
                tokio::select! {
                    _ = interval.tick() => {
                        // 更新资源使用情况
                        cpu_time.store(start_time.elapsed().as_millis() as u64, Ordering::SeqCst);
                        
                        // 获取内存使用
                        let current_memory = memory_bytes.load(Ordering::SeqCst);
                        
                        // 检查资源限制
                        if cpu_time.load(Ordering::SeqCst) > cpu_limit {
                            warn!("CPU时间超出限制: {}/{} ms", cpu_time.load(Ordering::SeqCst), cpu_limit);
                            cancelled.store(true, Ordering::SeqCst);
                        }
                        
                        if current_memory > memory_limit as u64 {
                            warn!("内存使用超出限制: {}/{} bytes", current_memory, memory_limit);
                            cancelled.store(true, Ordering::SeqCst);
                        }
                        
                        // 检查是否已取消
                        if cancelled.load(Ordering::SeqCst) {
                            debug!("任务已取消，停止监控: {}", task_id);
                            break;
                        }
                    }
                    cmd = rx.recv() => {
                        match cmd {
                            Some(MonitorCommand::Stop) => {
                                debug!("收到停止监控命令: {}", task_id);
                                break;
                            }
                            Some(MonitorCommand::GetResourceUsage(resp_tx)) => {
                                // 构建资源使用情况
                                let current_memory = memory_bytes.load(Ordering::SeqCst);
                                let current_cpu_time = cpu_time.load(Ordering::SeqCst);
                                
                                let resource_usage = ResourceUsage {
                                    cpu_usage: 100.0, // CPU使用百分比
                                    memory_usage: current_memory,
                                    peak_memory_usage: current_memory, // 峰值内存使用
                                    disk_read: disk_read_bytes.load(Ordering::SeqCst),
                                    disk_write: disk_write_bytes.load(Ordering::SeqCst),
                                    network_received: network_bytes.load(Ordering::SeqCst) / 2, // 粗略估算接收
                                    network_sent: network_bytes.load(Ordering::SeqCst) / 2, // 粗略估算发送
                                    execution_time_ms: current_cpu_time,
                                    gpu_usage: None, // GPU使用情况（暂不支持）
                                    gpu_memory_usage: None, // GPU内存使用（暂不支持）
                                };
                                
                                if let Err(e) = resp_tx.send(resource_usage).await {
                                    error!("发送资源使用情况失败: {}", e);
                                }
                            }
                            Some(MonitorCommand::CheckResourceLimits) => {
                                // 检查并报告资源限制
                                let current_cpu = cpu_time.load(Ordering::SeqCst);
                                let current_memory = memory_bytes.load(Ordering::SeqCst) as usize;
                                
                                let cpu_percent = (current_cpu as f64 / cpu_limit as f64) * 100.0;
                                let memory_percent = (current_memory as f64 / memory_limit as f64) * 100.0;
                                
                                if cpu_percent > 80.0 {
                                    warn!("CPU使用接近限制: {:.1}%", cpu_percent);
                                }
                                
                                if memory_percent > 80.0 {
                                    warn!("内存使用接近限制: {:.1}%", memory_percent);
                                }
                            }
                            None => {
                                // 通道已关闭
                                debug!("监控通道已关闭: {}", task_id);
                                break;
                            }
                        }
                    }
                }
            }
            
            debug!("资源监控已停止: {}", task_id);
        });
        
        Ok(())
    }
    
    /// 停止资源监控
    pub async fn stop_monitoring(&self) -> Result<()> {
        debug!("停止资源监控: {}", self.task_id);
        
        let tx = {
            let monitor_tx_guard = self.monitor_tx.lock().unwrap();
            monitor_tx_guard.clone()
        };
        
        if let Some(tx) = tx {
            // 发送停止命令
            if let Err(e) = tx.send(MonitorCommand::Stop).await {
                warn!("发送停止监控命令失败: {}", e);
            }
        }
        
        Ok(())
    }
    
    /// 获取资源使用情况
    pub async fn get_resource_usage(&self) -> Result<ResourceUsage> {
        let tx = {
            let monitor_tx_guard = self.monitor_tx.lock().unwrap();
            monitor_tx_guard.clone()
        };
        
        if let Some(tx) = tx {
            // 创建响应通道
            let (resp_tx, mut resp_rx) = mpsc::channel(1);
            
            // 发送获取资源命令
            if let Err(e) = tx.send(MonitorCommand::GetResourceUsage(resp_tx)).await {
                return Err(Error::internal(format!("发送获取资源命令失败: {}", e)));
            }
            
            // 等待响应
            match tokio::time::timeout(Duration::from_millis(500), resp_rx.recv()).await {
                Ok(Some(usage)) => {
                    return Ok(usage);
                }
                Ok(None) => {
                    return Err(Error::internal("资源监控通道已关闭"));
                }
                Err(_) => {
                    return Err(Error::deadline_exceeded("获取资源使用情况超时"));
                }
            }
        }
        
        // 如果没有监控，返回默认值
        Ok(self.resource_usage.lock().unwrap().clone())
    }
    
    /// 清理资源
    pub fn cleanup(&self) -> Result<()> {
        debug!("清理执行环境资源: {}", self.task_id);
        
        // 释放WASM资源
        {
            let mut store_guard = self.wasm_store.lock().unwrap();
            *store_guard = None;
        }
        
        {
            let mut engine_guard = self.wasm_engine.lock().unwrap();
            *engine_guard = None;
        }
        
        // 清空缓冲区
        if let Ok(mut stdout) = self.stdout_buffer.lock() {
            stdout.clear();
        }
        
        if let Ok(mut stderr) = self.stderr_buffer.lock() {
            stderr.clear();
        }
        
        // 删除临时目录
        self.temp_dir = None;
        
        Ok(())
    }
    
    /// 取消执行
    pub fn cancel(&self) {
        debug!("取消执行: {}", self.task_id);
        self.cancelled.store(true, Ordering::SeqCst);
    }
    
    /// 检查是否已取消
    pub fn is_cancelled(&self) -> bool {
        self.cancelled.load(Ordering::SeqCst)
    }
    
    /// 获取标准输出
    pub fn get_stdout(&self) -> String {
        if let Ok(stdout) = self.stdout_buffer.lock() {
            String::from_utf8_lossy(&stdout).to_string()
        } else {
            String::new()
        }
    }
    
    /// 获取标准错误
    pub fn get_stderr(&self) -> String {
        if let Ok(stderr) = self.stderr_buffer.lock() {
            String::from_utf8_lossy(&stderr).to_string()
        } else {
            String::new()
        }
    }
    
    /// 写入标准输出
    pub fn write_stdout(&self, data: &[u8]) -> Result<()> {
        if let Ok(mut stdout) = self.stdout_buffer.lock() {
            stdout.write_all(data)?;
            Ok(())
        } else {
            Err(Error::lock("无法获取stdout锁"))
        }
    }
    
    /// 写入标准错误
    pub fn write_stderr(&self, data: &[u8]) -> Result<()> {
        if let Ok(mut stderr) = self.stderr_buffer.lock() {
            stderr.write_all(data)?;
            Ok(())
        } else {
            Err(Error::lock("无法获取stderr锁"))
        }
    }

    /// 取得WASM引擎的引用
    pub fn get_wasm_engine(&self) -> Option<Arc<WasmEngine>> {
        let engine_guard = self.wasm_engine.lock().unwrap();
        engine_guard.clone()
    }

    /// 取得WASM存储的可变引用
    pub fn take_wasm_store(&self) -> Option<WasmStore<wasmtime_wasi::WasiCtx>> {
        let mut store_guard = self.wasm_store.lock().unwrap();
        store_guard.take()
    }
} 