use serde::{Serialize, Deserialize};
use crate::algorithm::types::{SandboxSecurityLevel, ResourceLimits, ResourceUsage, SandboxStatus};
use std::time::Duration;
use std::path::PathBuf;
use std::collections::HashMap;
use uuid::Uuid;
use std::time::SystemTime;
use async_trait;
use std::time::Instant;
use log::{info, debug};
use crate::data::DataBatch;
use crate::error::{Error, Result};
use crate::algorithm::executor::resources::{ResourceMonitor};

/// 安全上下文，定义沙箱执行环境的安全策略
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SecurityContext {
    /// 是否允许网络访问
    pub allow_network: bool,
    /// 是否允许文件系统访问
    pub allow_filesystem: bool,
    /// 允许的环境变量
    pub allowed_env_vars: Vec<String>,
    /// 允许的文件系统路径
    pub allowed_paths: Vec<String>,
    /// 是否允许标准输入
    pub allow_stdin: bool,
    /// 是否允许标准输出
    pub allow_stdout: bool,
    /// 是否允许标准错误
    pub allow_stderr: bool,
    /// 最大内存使用限制(字节)
    pub memory_limit_bytes: usize,
    /// CPU时间限制(毫秒)
    pub cpu_time_limit_ms: u64,
    /// 磁盘I/O限制(字节)
    pub disk_io_limit_bytes: Option<usize>,
    /// 是否启用WASM内存限制
    pub enable_wasm_memory_limit: bool,
    /// 最大WASM内存页数
    pub max_wasm_memory_pages: Option<u32>,
    /// 最大并行度
    pub max_parallelism: Option<u32>,
    /// 安全级别
    pub security_level: SandboxSecurityLevel,
    /// 是否允许系统调用
    pub allow_syscalls: bool,
    /// 允许的系统调用列表
    pub allowed_syscalls: Vec<String>,
    /// Docker沙箱专用配置
    pub docker_config: Option<DockerSandboxConfig>,
}

impl SecurityContext {
    /// 创建新的安全上下文
    pub fn new(security_level: SandboxSecurityLevel) -> Self {
        Self {
            allow_network: false,
            allow_filesystem: false,
            allowed_env_vars: vec![],
            allowed_paths: vec![],
            allow_stdin: false,
            allow_stdout: true,
            allow_stderr: true,
            memory_limit_bytes: 512 * 1024 * 1024, // 512MB
            cpu_time_limit_ms: 30000, // 30秒
            disk_io_limit_bytes: Some(100 * 1024 * 1024), // 100MB
            enable_wasm_memory_limit: true,
            max_wasm_memory_pages: Some(1000), // 约64MB内存
            max_parallelism: Some(2),
            security_level,
            allow_syscalls: false,
            allowed_syscalls: vec![],
            docker_config: Some(DockerSandboxConfig::default()),
        }
    }
}

impl Default for SecurityContext {
    fn default() -> Self {
        Self::new(SandboxSecurityLevel::Standard)
    }
}

/// Docker沙箱配置
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DockerSandboxConfig {
    /// 基础镜像
    pub base_image: String,
    /// 镜像标签
    pub image_tag: String,
    /// 是否使用只读文件系统
    pub read_only_filesystem: bool,
    /// 移除容器特权
    pub drop_capabilities: bool,
    /// 是否启用安全内核选项
    pub enable_security_opt: bool,
    /// 自定义容器挂载点
    pub mounts: Vec<String>,
    /// 自定义容器参数
    pub extra_args: Vec<String>,
}

impl Default for DockerSandboxConfig {
    fn default() -> Self {
        Self {
            base_image: "alpine".to_string(),
            image_tag: "latest".to_string(),
            read_only_filesystem: true,
            drop_capabilities: true,
            enable_security_opt: true,
            mounts: vec![],
            extra_args: vec![],
        }
    }
}

/// 超时策略
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimeoutPolicy {
    /// 初始超时时间(毫秒)
    pub initial_timeout_ms: u64,
    /// 最大超时时间(毫秒)
    pub max_timeout_ms: u64,
    /// 是否启用超时自动调整
    pub enable_auto_adjust: bool,
    /// 调整因子
    pub adjustment_factor: f64,
}

impl Default for TimeoutPolicy {
    fn default() -> Self {
        Self {
            initial_timeout_ms: 10000, // 10秒
            max_timeout_ms: 60000,     // 1分钟
            enable_auto_adjust: true,
            adjustment_factor: 1.5,
        }
    }
}

/// 资源使用阈值，定义触发警告或错误的资源使用百分比
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct ResourceThresholds {
    /// CPU使用警告阈值(百分比)
    pub cpu_warning_percent: f64,
    /// CPU使用错误阈值(百分比)
    pub cpu_error_percent: f64,
    /// 内存使用警告阈值(百分比)
    pub memory_warning_percent: f64,
    /// 内存使用错误阈值(百分比)
    pub memory_error_percent: f64,
    /// 磁盘I/O警告阈值(百分比)
    pub disk_warning_percent: f64,
    /// 磁盘I/O错误阈值(百分比)
    pub disk_error_percent: f64,
}

impl Default for ResourceThresholds {
    fn default() -> Self {
        Self {
            cpu_warning_percent: 80.0,
            cpu_error_percent: 95.0,
            memory_warning_percent: 80.0,
            memory_error_percent: 95.0,
            disk_warning_percent: 80.0,
            disk_error_percent: 95.0,
        }
    }
}

/// 沙箱配置
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SandboxConfig {
    pub memory_limit: Option<u64>,
    pub cpu_limit: Option<f64>,
    pub timeout: Option<Duration>,
    pub network_access: bool,
    pub file_system_access: bool,
    pub allowed_syscalls: Vec<String>,
    pub environment_variables: HashMap<String, String>,
    pub working_directory: Option<PathBuf>,
    pub security_level: SandboxSecurityLevel,
    pub isolation_level: IsolationLevel,
}

impl Default for SandboxConfig {
    fn default() -> Self {
        Self {
            memory_limit: Some(512 * 1024 * 1024), // 512MB
            cpu_limit: Some(1.0), // 1 CPU core
            timeout: Some(Duration::from_secs(300)), // 5 minutes
            network_access: false,
            file_system_access: false,
            allowed_syscalls: vec![
                "read".to_string(),
                "write".to_string(),
                "open".to_string(),
                "close".to_string(),
                "mmap".to_string(),
                "munmap".to_string(),
                "brk".to_string(),
                "exit".to_string(),
            ],
            environment_variables: HashMap::new(),
            working_directory: None,
            security_level: SandboxSecurityLevel::Medium,
            isolation_level: IsolationLevel::High,
        }
    }
}

/// 隔离级别
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum IsolationLevel {
    Low,
    Medium,
    High,
    Maximum,
}

/// 沙箱接口
pub trait Sandbox: Send + Sync {
    /// 创建新的沙箱实例
    fn new(config: SandboxConfig) -> crate::error::Result<Self> where Self: Sized;
    
    /// 启动沙箱
    async fn start(&mut self) -> crate::error::Result<()>;
    
    /// 停止沙箱
    async fn stop(&mut self) -> crate::error::Result<()>;
    
    /// 暂停沙箱
    async fn pause(&mut self) -> crate::error::Result<()>;
    
    /// 恢复沙箱
    async fn resume(&mut self) -> crate::error::Result<()>;
    
    /// 在沙箱中执行代码
    async fn execute(&mut self, code: &str, input_data: &DataBatch) -> crate::error::Result<DataBatch>;
    
    /// 获取沙箱状态
    fn get_status(&self) -> SandboxStatus;
    
    /// 获取资源使用情况
    fn get_resource_usage(&self) -> ResourceUsage;
    
    /// 设置资源限制
    fn set_resource_limits(&mut self, limits: ResourceLimits) -> crate::error::Result<()>;
    
    /// 获取安全上下文
    fn get_security_context(&self) -> &SecurityContext;
    
    /// 清理沙箱资源
    async fn cleanup(&mut self) -> crate::error::Result<()>;
    
    /// 获取沙箱ID
    fn get_id(&self) -> &str;
}

/// 默认沙箱实现
#[derive(Debug)]
pub struct DefaultSandbox {
    id: String,
    config: SandboxConfig,
    status: SandboxStatus,
    resource_monitor: ResourceMonitor,
    security_context: SecurityContext,
    start_time: Option<SystemTime>,
    last_activity: Option<Instant>,
    resource_usage: ResourceUsage,
}

impl DefaultSandbox {
    /// 创建新的默认沙箱实例
    pub fn new(config: SandboxConfig) -> crate::error::Result<Self> {
        let id = Uuid::new_v4().to_string();
        let security_context = SecurityContext::new(config.security_level.clone());
        
        Ok(Self {
            id,
            config,
            status: SandboxStatus::Created,
            resource_monitor: ResourceMonitor::new(),
            security_context,
            start_time: None,
            last_activity: None,
            resource_usage: ResourceUsage::default(),
        })
    }

    /// 启动沙箱（内部方法，不是trait要求的）
    pub async fn internal_start(&mut self) -> crate::error::Result<()> {
        if self.status != SandboxStatus::Created && self.status != SandboxStatus::Ready {
            return Err(Error::InvalidState(
                format!("Cannot start sandbox in state: {:?}", self.status)
            ));
        }

        self.status = SandboxStatus::Running;
        self.start_time = Some(SystemTime::now());
        self.last_activity = Some(Instant::now());
        
        info!("Started sandbox: {}", self.id);
        Ok(())
    }

    /// 停止沙箱（内部方法，不是trait要求的）
    pub async fn internal_stop(&mut self) -> crate::error::Result<()> {
        if self.status == SandboxStatus::Completed {
            return Ok(());
        }

        self.status = SandboxStatus::Completed;
        info!("Stopped sandbox: {}", self.id);
        Ok(())
    }

    /// 暂停沙箱（内部方法，不是trait要求的）
    pub async fn internal_pause(&mut self) -> crate::error::Result<()> {
        if self.status != SandboxStatus::Running {
            return Err(Error::InvalidState(
                format!("Cannot pause sandbox in state: {:?}", self.status)
            ));
        }

        self.status = SandboxStatus::Paused;
        info!("Paused sandbox: {}", self.id);
        Ok(())
    }

    /// 恢复沙箱（内部方法，不是trait要求的）
    pub async fn internal_resume(&mut self) -> crate::error::Result<()> {
        if self.status != SandboxStatus::Paused {
            return Err(Error::InvalidState(
                format!("Cannot resume sandbox in state: {:?}", self.status)
            ));
        }

        self.status = SandboxStatus::Running;
        info!("Resumed sandbox: {}", self.id);
        Ok(())
    }

    /// 验证代码安全性
    fn validate_code(&self, code: &[u8]) -> Result<()> {
        let code_str = std::str::from_utf8(code)
            .map_err(|e| Error::InvalidInput(format!("Invalid UTF-8 code: {}", e)))?;

        // 检查危险操作
        let dangerous_patterns = [
            "import os",
            "import subprocess", 
            "import sys",
            "__import__",
            "eval(",
            "exec(",
            "open(",
            "file(",
            "input(",
            "raw_input(",
        ];

        for pattern in &dangerous_patterns {
            if code_str.contains(pattern) {
                return Err(Error::SecurityViolation(
                    format!("Dangerous operation detected: {}", pattern)
                ));
            }
        }

        // 检查代码长度
        if code.len() > 100_000 {
            return Err(Error::SecurityViolation(
                "Code too long".to_string()
            ));
        }

        Ok(())
    }

    /// 执行代码的内部实现
    async fn execute_code_internal(&self, code: &[u8], input: &[u8], timeout: Duration) -> Result<super::result::SandboxResult> {
        // 验证代码
        self.validate_code(code)?;

        let code_str = std::str::from_utf8(code)
            .map_err(|e| Error::InvalidInput(format!("Invalid UTF-8 code: {}", e)))?;
        let input_str = std::str::from_utf8(input)
            .map_err(|e| Error::InvalidInput(format!("Invalid UTF-8 input: {}", e)))?;

        // 模拟代码执行（实际实现中应该使用真正的沙箱环境）
        info!("Executing code in sandbox {}: {}", self.id, code_str);
        debug!("Code: {}", code_str);

        // 模拟处理时间，但不超过timeout
        let sleep_duration = std::cmp::min(Duration::from_millis(100), timeout);
        tokio::time::sleep(sleep_duration).await;

        // 创建执行结果
        let result = super::result::SandboxResult::new(
            true,                                    // success
            0,                                      // exit_code
            input_str.to_string(),                  // stdout
            String::new(),                          // stderr
            sleep_duration.as_millis() as u64,      // execution_time_ms
            crate::algorithm::types::ResourceUsage::default(), // resource_usage
        );

        Ok(result)
    }

    /// 设置资源限制
    pub fn set_resource_limits(&mut self, limits: ResourceLimits) -> Result<()> {
        self.config.memory_limit = limits.max_memory;
        self.config.cpu_limit = Some(limits.max_cpu_percent as f64 / 100.0);
        Ok(())
    }

    /// 获取安全上下文
    pub fn get_security_context(&self) -> &SecurityContext {
        &self.security_context
    }

    /// 清理沙箱资源（内部方法，不是trait要求的）
    pub async fn internal_cleanup(&mut self) -> crate::error::Result<()> {
        self.status = SandboxStatus::Disposed;
        info!("Cleaned up sandbox: {}", self.id);
        Ok(())
    }
}

// 实现interface.rs中定义的Sandbox trait
#[async_trait::async_trait]
impl crate::algorithm::executor::sandbox::interface::Sandbox for DefaultSandbox {
    fn id(&self) -> &str {
        &self.id
    }
    
    async fn prepare(&self) -> Result<()> {
        info!("Preparing sandbox: {}", self.id);
        Ok(())
    }
    
    async fn execute(&self, code: &[u8], input: &[u8], timeout: Duration) -> Result<super::result::SandboxResult> {
        if self.status != SandboxStatus::Running && self.status != SandboxStatus::Ready {
            return Err(Error::InvalidState(
                format!("Cannot execute in sandbox state: {:?}", self.status)
            ));
        }

        self.execute_code_internal(code, input, timeout).await
    }
    
    async fn cleanup(&self) -> Result<()> {
        info!("Cleaning up sandbox: {}", self.id);
        Ok(())
    }
    
    fn supports_file_type(&self, file_type: &str) -> bool {
        matches!(file_type, "py" | "js" | "txt" | "json")
    }
    
    async fn load_file(&self, _src_path: &std::path::Path, _sandbox_path: &str) -> Result<()> {
        // 默认实现 - 实际项目中需要具体实现
        info!("Loading file into sandbox: {}", self.id);
        Ok(())
    }
    
    async fn save_file(&self, _sandbox_path: &str, _dest_path: &std::path::Path) -> Result<()> {
        // 默认实现 - 实际项目中需要具体实现
        info!("Saving file from sandbox: {}", self.id);
        Ok(())
    }
    
    async fn cancel(&self) -> Result<()> {
        info!("Cancelling execution in sandbox: {}", self.id);
        Ok(())
    }
    
    async fn get_resource_usage(&self) -> Result<crate::algorithm::types::ResourceUsage> {
        Ok(self.resource_usage.clone())
    }
    
    async fn validate_code(&self, code: &[u8]) -> Result<Vec<String>> {
        match self.validate_code(code) {
            Ok(()) => Ok(vec![]),
            Err(e) => Ok(vec![e.to_string()])
        }
    }
    
    async fn set_env_var(&self, name: &str, value: &str) -> Result<()> {
        info!("Setting environment variable in sandbox {}: {}={}", self.id, name, value);
        Ok(())
    }
    
    async fn get_status(&self) -> Result<SandboxStatus> {
        Ok(self.status)
    }

    /// 获取所有算法定义
    async fn get_all_algorithm_definitions(&self) -> Result<HashMap<String, crate::core::interfaces::AlgorithmDefinition>> {
        // 生产级实现：从存储中获取所有算法定义
        // 这里应该使用实际的存储系统
        Ok(HashMap::new())
    }

    /// 执行算法定义
    async fn execute_algorithm(&self, algorithm: &crate::core::interfaces::AlgorithmDefinition, data: &crate::data::DataBatch) -> Result<crate::algorithm::types::ExecutionResult> {
        debug!("在沙箱中执行算法: {}", algorithm.name);
        
        // 验证算法代码
        let code_bytes = algorithm.code.as_bytes();
        self.validate_code(code_bytes)?;
        
        // 准备输入数据
        let input_data = self.prepare_input_data(data).await?;
        
        // 在沙箱中执行代码
        let timeout = self.config.timeout.unwrap_or(std::time::Duration::from_secs(300));
        let sandbox_result = self.execute_code_internal(code_bytes, &input_data, timeout).await?;
        
        // 处理执行结果
        let execution_result = self.process_sandbox_result(&algorithm.id, sandbox_result).await?;
        
        Ok(execution_result)
    }

    /// 准备输入数据
    async fn prepare_input_data(&self, data: &crate::data::DataBatch) -> Result<Vec<u8>> {
        // 将DataBatch序列化为字节数组
        let serialized = serde_json::to_vec(&data)
            .map_err(|e| Error::serialization(format!("Failed to serialize input data: {}", e)))?;
        
        Ok(serialized)
    }

    /// 处理沙箱执行结果
    async fn process_sandbox_result(&self, algorithm_id: &str, sandbox_result: super::result::SandboxResult) -> Result<crate::algorithm::types::ExecutionResult> {
        let execution_id = uuid::Uuid::new_v4().to_string();
        
        // 解析输出数据
        let outputs = if let Some(output_data) = sandbox_result.output {
            match serde_json::from_slice::<crate::data::DataBatch>(&output_data) {
                Ok(data_batch) => data_batch,
                Err(_) => {
                    // 如果解析失败，创建空的DataBatch
                    crate::data::DataBatch {
                        id: uuid::Uuid::new_v4().to_string(),
                        features: vec![],
                        labels: None,
                        metadata: HashMap::new(),
                        batch_size: 0,
                        created_at: chrono::Utc::now(),
                    }
                }
            }
        } else {
            // 如果没有输出，创建空的DataBatch
            crate::data::DataBatch {
                id: uuid::Uuid::new_v4().to_string(),
                features: vec![],
                labels: None,
                metadata: HashMap::new(),
                batch_size: 0,
                created_at: chrono::Utc::now(),
            }
        };

        // 确定执行状态
        let status = if sandbox_result.success {
            crate::algorithm::types::ExecutionStatus::Completed
        } else {
            crate::algorithm::types::ExecutionStatus::Failed
        };

        // 创建错误信息
        let error_message = if !sandbox_result.success {
            Some(sandbox_result.error.unwrap_or_else(|| "Unknown error".to_string()))
        } else {
            None
        };

        Ok(crate::algorithm::types::ExecutionResult {
            execution_id,
            algorithm_id: algorithm_id.to_string(),
            outputs,
            status,
            resource_usage: sandbox_result.resource_usage,
            execution_metrics: crate::algorithm::executor::metrics::ExecutionMetrics::default(),
            error_message,
        })
    }
} 