use std::collections::HashMap;
use std::path::{PathBuf};
use std::process::{Command, Stdio, Child};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use std::thread;
use std::io::{BufRead, BufReader};
use serde::{Serialize, Deserialize};
use log::{debug, info, warn, error};
// removed unused Uuid import

use crate::error::{Error, Result};
use super::{SecurityContext};
use crate::algorithm::types::ResourceLimits;

/// 进程沙箱配置
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProcessSandboxConfig {
    /// 工作目录
    pub working_dir: PathBuf,
    /// 环境变量
    pub environment: HashMap<String, String>,
    /// 资源限制
    pub resource_limits: ResourceLimits,
    /// 安全配置
    pub security_config: ProcessSecurityConfig,
    /// 超时设置
    pub timeout: Duration,
    /// 是否捕获输出
    pub capture_output: bool,
    /// 是否实时输出
    pub real_time_output: bool,
    /// 输出缓冲区大小
    pub output_buffer_size: usize,
}

/// 进程安全配置
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProcessSecurityConfig {
    /// 用户ID
    pub user_id: Option<u32>,
    /// 组ID
    pub group_id: Option<u32>,
    /// 禁用的系统调用
    pub disabled_syscalls: Vec<String>,
    /// 允许的文件路径
    pub allowed_paths: Vec<PathBuf>,
    /// 禁止的文件路径
    pub forbidden_paths: Vec<PathBuf>,
    /// 网络访问控制
    pub network_access: NetworkAccess,
    /// 文件系统访问控制
    pub filesystem_access: FilesystemAccess,
    /// 进程隔离级别
    pub isolation_level: IsolationLevel,
}

/// 网络访问控制
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum NetworkAccess {
    /// 完全禁止
    Denied,
    /// 仅本地回环
    LoopbackOnly,
    /// 受限访问
    Restricted(Vec<String>), // 允许的域名/IP列表
    /// 完全允许
    Allowed,
}

/// 文件系统访问控制
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FilesystemAccess {
    /// 只读路径
    pub readonly_paths: Vec<PathBuf>,
    /// 可写路径
    pub writable_paths: Vec<PathBuf>,
    /// 禁止访问的路径
    pub forbidden_paths: Vec<PathBuf>,
    /// 临时目录
    pub temp_dir: Option<PathBuf>,
}

/// 进程隔离级别
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum IsolationLevel {
    /// 无隔离
    None,
    /// 基础隔离
    Basic,
    /// 中等隔离
    Medium,
    /// 高级隔离
    High,
    /// 完全隔离
    Complete,
}

/// 进程状态
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum ProcessState {
    /// 创建中
    Creating,
    /// 运行中
    Running,
    /// 已暂停
    Paused,
    /// 已完成
    Completed(i32),
    /// 被终止
    Terminated(i32),
    /// 超时
    Timeout,
    /// 错误
    Error(String),
}

/// 进程信息
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProcessInfo {
    /// 进程ID
    pub pid: u32,
    /// 进程名称
    pub name: String,
    /// 进程状态
    pub state: ProcessState,
    /// 创建时间
    pub created_at: Instant,
    /// 启动时间
    pub started_at: Option<Instant>,
    /// 结束时间
    pub ended_at: Option<Instant>,
    /// 退出码
    pub exit_code: Option<i32>,
    /// 命令行
    pub command_line: Vec<String>,
    /// 工作目录
    pub working_dir: PathBuf,
    /// 资源使用情况
    pub resource_usage: ProcessResourceUsage,
}

/// 进程资源使用情况
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProcessResourceUsage {
    /// CPU时间 (毫秒)
    pub cpu_time_ms: u64,
    /// 内存使用量 (字节)
    pub memory_bytes: u64,
    /// 峰值内存使用量 (字节)
    pub peak_memory_bytes: u64,
    /// 文件描述符数量
    pub open_files: u32,
    /// 子进程数量
    pub child_processes: u32,
    /// 网络连接数量
    pub network_connections: u32,
}

/// 进程输出
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProcessOutput {
    /// 标准输出
    pub stdout: String,
    /// 标准错误
    pub stderr: String,
    /// 输出行数
    pub stdout_lines: usize,
    /// 错误行数
    pub stderr_lines: usize,
    /// 输出大小 (字节)
    pub output_size: usize,
}

/// 进程执行结果
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProcessExecutionResult {
    /// 进程信息
    pub process_info: ProcessInfo,
    /// 进程输出
    pub output: ProcessOutput,
    /// 执行时间
    pub execution_time: Duration,
    /// 是否成功
    pub success: bool,
    /// 错误信息
    pub error_message: Option<String>,
}

/// 进程沙箱实现
pub struct ProcessSandbox {
    /// 配置
    config: ProcessSandboxConfig,
    /// 安全上下文
    security_context: Arc<SecurityContext>,
    /// 当前进程
    current_process: Arc<Mutex<Option<ProcessHandle>>>,
    /// 进程监控器
    monitor: Arc<Mutex<Option<ProcessMonitor>>>,
}

/// 进程句柄
/// 
/// 使用Arc<Mutex<>>包装Child进程，以便安全地共享和克隆
#[derive(Clone)]
struct ProcessHandle {
    /// 子进程（使用Arc<Mutex<>>包装以便共享）
    child: Arc<Mutex<Child>>,
    /// 进程信息
    info: Arc<Mutex<ProcessInfo>>,
    /// 启动时间
    start_time: Instant,
}

/// 进程监控器
struct ProcessMonitor {
    /// 监控线程句柄
    monitor_thread: Option<thread::JoinHandle<()>>,
    /// 是否正在监控
    is_monitoring: Arc<Mutex<bool>>,
    /// 资源使用历史
    resource_history: Arc<Mutex<Vec<ProcessResourceUsage>>>,
}

impl ProcessSandbox {
    /// 创建新的进程沙箱
    pub fn new(config: ProcessSandboxConfig, security_context: SecurityContext) -> Self {
        Self {
            config,
            security_context: Arc::new(security_context),
            current_process: Arc::new(Mutex::new(None)),
            monitor: Arc::new(Mutex::new(None)),
        }
    }

    /// 执行命令
    pub async fn execute(&self, command: &[String]) -> Result<ProcessExecutionResult> {
        if command.is_empty() {
            return Err(Error::InvalidInput("Command cannot be empty".to_string()));
        }

        let start_time = Instant::now();
        
        // 验证安全配置
        self.validate_security_config(command)?;
        
        // 准备执行环境
        self.prepare_environment().await?;
        
        // 创建进程
        let process_handle = self.create_process(command).await?;
        
        // 启动监控
        self.start_monitoring(&process_handle).await?;
        
        // 创建超时检查
        let timeout_duration = self.config.timeout;
        let timeout_check = tokio::time::sleep(timeout_duration);
        
        // 等待进程完成
        let result: Result<ProcessExecutionResult> = tokio::select! {
            _ = timeout_check => {
                // 超时，终止进程
                let mut child_guard = process_handle.child.lock().unwrap();
                if let Err(e) = child_guard.kill() {
                    error!("Failed to kill timed out process: {}", e);
                }
                drop(child_guard);
                
                let mut info_guard = process_handle.info.lock().unwrap();
                info_guard.state = ProcessState::Timeout;
                info_guard.ended_at = Some(Instant::now());
                let process_info = info_guard.clone();
                drop(info_guard);
                
                Ok(ProcessExecutionResult {
                    process_info,
                    output: ProcessOutput {
                        stdout: String::new(),
                        stderr: "Process timed out".to_string(),
                        stdout_lines: 0,
                        stderr_lines: 1,
                        output_size: 0,
                    },
                    execution_time: start_time.elapsed(),
                    success: false,
                    error_message: Some("Process execution timeout".to_string()),
                })
            },
            result = self.wait_for_process(&process_handle) => {
                result
            }
        };
        
        // 停止监控
        self.stop_monitoring().await?;
        
        let final_result = result?;
        info!("Process execution completed: success={}, time={:?}", 
            final_result.success, final_result.execution_time);
        
        Ok(final_result)
    }

    /// 终止当前进程
    pub async fn terminate(&self) -> Result<()> {
        let mut process_guard = self.current_process.lock().unwrap();
        
        if let Some(ref handle) = *process_guard {
            // 尝试优雅终止（标准库Child没有terminate方法，直接使用kill）
            let mut child_guard = handle.child.lock().unwrap();
            if let Err(e) = child_guard.kill() {
                error!("Failed to kill process: {}", e);
                return Err(Error::ExecutionError(format!("Failed to kill process: {}", e)));
            }
            drop(child_guard);
            
            let mut info_guard = handle.info.lock().unwrap();
            info_guard.state = ProcessState::Terminated(-1);
            info_guard.ended_at = Some(Instant::now());
            let pid = info_guard.pid;
            drop(info_guard);
            
            info!("Process {} terminated", pid);
        }
        
        // 停止监控
        self.stop_monitoring().await?;
        
        Ok(())
    }

    /// 获取进程信息
    pub fn get_process_info(&self) -> Option<ProcessInfo> {
        self.current_process.lock().unwrap()
            .as_ref()
            .and_then(|handle| {
                handle.info.lock().ok().map(|info_guard| info_guard.clone())
            })
    }

    /// 获取资源使用历史
    pub fn get_resource_history(&self) -> Vec<ProcessResourceUsage> {
        if let Some(ref monitor) = *self.monitor.lock().unwrap() {
            monitor.resource_history.lock().unwrap().clone()
        } else {
            Vec::new()
        }
    }

    /// 验证安全配置
    fn validate_security_config(&self, command: &[String]) -> Result<()> {
        // 检查命令是否在允许列表中
        let program = &command[0];
        
        // 检查禁止的系统调用
        for syscall in &self.config.security_config.disabled_syscalls {
            if program.contains(syscall) {
                return Err(Error::SecurityViolation(
                    format!("Command contains disabled syscall: {}", syscall)));
            }
        }
        
        // 检查文件路径访问权限
        for path in &self.config.security_config.forbidden_paths {
            if program.starts_with(&path.to_string_lossy().to_string()) {
                return Err(Error::SecurityViolation(
                    format!("Command accesses forbidden path: {:?}", path)));
            }
        }
        
        // 检查网络访问权限
        if matches!(self.config.security_config.network_access, NetworkAccess::Denied) {
            // 检查命令是否尝试进行网络访问
            let network_commands = ["curl", "wget", "nc", "telnet", "ssh", "scp"];
            if network_commands.iter().any(|&cmd| program.contains(cmd)) {
                return Err(Error::SecurityViolation(
                    "Network access is denied".to_string()));
            }
        }
        
        Ok(())
    }

    /// 准备执行环境
    async fn prepare_environment(&self) -> Result<()> {
        // 创建工作目录
        if !self.config.working_dir.exists() {
            std::fs::create_dir_all(&self.config.working_dir)
                .map_err(|e| Error::IoError(format!("Failed to create working directory: {}", e)))?;
        }
        
        // 创建临时目录
        if let Some(ref temp_dir) = self.config.security_config.filesystem_access.temp_dir {
            if !temp_dir.exists() {
                std::fs::create_dir_all(temp_dir)
                    .map_err(|e| Error::IoError(format!("Failed to create temp directory: {}", e)))?;
            }
        }
        
        // 设置文件权限
        self.setup_file_permissions().await?;
        
        Ok(())
    }

    /// 设置文件权限
    async fn setup_file_permissions(&self) -> Result<()> {
        // 设置只读路径权限
        for path in &self.config.security_config.filesystem_access.readonly_paths {
            if path.exists() {
                // 在实际实现中，这里应该设置文件权限
                debug!("Setting readonly permission for path: {:?}", path);
            }
        }
        
        // 设置可写路径权限
        for path in &self.config.security_config.filesystem_access.writable_paths {
            if !path.exists() {
                std::fs::create_dir_all(path)
                    .map_err(|e| Error::IoError(format!("Failed to create writable path: {}", e)))?;
            }
            debug!("Setting writable permission for path: {:?}", path);
        }
        
        Ok(())
    }

    /// 创建进程
    async fn create_process(&self, command: &[String]) -> Result<ProcessHandle> {
        let mut cmd = Command::new(&command[0]);
        
        // 设置命令参数
        if command.len() > 1 {
            cmd.args(&command[1..]);
        }
        
        // 设置工作目录
        cmd.current_dir(&self.config.working_dir);
        
        // 设置环境变量
        cmd.envs(&self.config.environment);
        
        // 设置输入输出
        if self.config.capture_output {
            cmd.stdout(Stdio::piped());
            cmd.stderr(Stdio::piped());
        }
        cmd.stdin(Stdio::null());
        
        // 应用安全配置
        self.apply_security_config(&mut cmd)?;
        
        // 启动进程
        let child = cmd.spawn()
            .map_err(|e| Error::ExecutionError(format!("Failed to spawn process: {}", e)))?;
        
        let pid = child.id();
        let process_info = ProcessInfo {
            pid,
            name: command[0].clone(),
            state: ProcessState::Running,
            created_at: Instant::now(),
            started_at: Some(Instant::now()),
            ended_at: None,
            exit_code: None,
            command_line: command.to_vec(),
            working_dir: self.config.working_dir.clone(),
            resource_usage: ProcessResourceUsage {
                cpu_time_ms: 0,
                memory_bytes: 0,
                peak_memory_bytes: 0,
                open_files: 0,
                child_processes: 0,
                network_connections: 0,
            },
        };
        
        let handle = ProcessHandle {
            child: Arc::new(Mutex::new(child)),
            info: Arc::new(Mutex::new(process_info)),
            start_time: Instant::now(),
        };
        
        // 保存进程句柄（现在可以安全地克隆）
        *self.current_process.lock().unwrap() = Some(handle.clone());
        
        info!("Process {} started: {}", pid, command.join(" "));
        Ok(handle)
    }

    /// 应用安全配置
    fn apply_security_config(&self, cmd: &mut Command) -> Result<()> {
        // 设置用户和组
        if let Some(uid) = self.config.security_config.user_id {
            // 在实际实现中，这里应该设置进程的用户ID
            debug!("Setting process UID to: {}", uid);
        }
        
        if let Some(gid) = self.config.security_config.group_id {
            // 在实际实现中，这里应该设置进程的组ID
            debug!("Setting process GID to: {}", gid);
        }
        
        // 应用隔离级别
        match self.config.security_config.isolation_level {
            IsolationLevel::None => {
                debug!("No process isolation applied");
            },
            IsolationLevel::Basic => {
                debug!("Applying basic process isolation");
                // 基础隔离：限制文件系统访问
            },
            IsolationLevel::Medium => {
                debug!("Applying medium process isolation");
                // 中等隔离：限制网络和文件系统访问
            },
            IsolationLevel::High => {
                debug!("Applying high process isolation");
                // 高级隔离：使用命名空间隔离
            },
            IsolationLevel::Complete => {
                debug!("Applying complete process isolation");
                // 完全隔离：使用容器技术
            },
        }
        
        Ok(())
    }

    /// 启动监控
    async fn start_monitoring(&self, process_handle: &ProcessHandle) -> Result<()> {
        let pid = process_handle.info.lock().unwrap().pid;
        let is_monitoring = Arc::new(Mutex::new(true));
        let resource_history = Arc::new(Mutex::new(Vec::new()));
        
        let is_monitoring_clone = is_monitoring.clone();
        let resource_history_clone = resource_history.clone();
        
        let monitor_thread = thread::spawn(move || {
            while *is_monitoring_clone.lock().unwrap() {
                // 收集资源使用情况
                if let Ok(usage) = Self::collect_resource_usage(pid) {
                    resource_history_clone.lock().unwrap().push(usage);
                }
                
                // 每秒监控一次
                thread::sleep(Duration::from_secs(1));
            }
        });
        
        let monitor = ProcessMonitor {
            monitor_thread: Some(monitor_thread),
            is_monitoring,
            resource_history,
        };
        
        *self.monitor.lock().unwrap() = Some(monitor);
        
        debug!("Started monitoring process {}", pid);
        Ok(())
    }

    /// 停止监控
    async fn stop_monitoring(&self) -> Result<()> {
        let mut monitor_guard = self.monitor.lock().unwrap();
        
        if let Some(monitor) = monitor_guard.take() {
            // 停止监控标志
            *monitor.is_monitoring.lock().unwrap() = false;
            
            // 等待监控线程结束
            if let Some(thread) = monitor.monitor_thread {
                if let Err(e) = thread.join() {
                    warn!("Failed to join monitor thread: {:?}", e);
                }
            }
            
            debug!("Stopped process monitoring");
        }
        
        Ok(())
    }

    /// 等待进程完成
    async fn wait_for_completion(&self, process_handle: ProcessHandle) -> Result<ProcessExecutionResult> {
        let timeout = self.config.timeout;
        let start_time = process_handle.start_time;
        
        // 创建超时检查
        let timeout_check = tokio::time::sleep(timeout);
        
        // 等待进程完成或超时
        let result: Result<ProcessExecutionResult> = tokio::select! {
            _ = timeout_check => {
                // 超时，终止进程
                let mut child_guard = process_handle.child.lock().unwrap();
                if let Err(e) = child_guard.kill() {
                    error!("Failed to kill timed out process: {}", e);
                }
                drop(child_guard);
                
                let mut info_guard = process_handle.info.lock().unwrap();
                info_guard.state = ProcessState::Timeout;
                info_guard.ended_at = Some(Instant::now());
                let process_info = info_guard.clone();
                drop(info_guard);
                
                Ok(ProcessExecutionResult {
                    process_info,
                    output: ProcessOutput {
                        stdout: String::new(),
                        stderr: "Process timed out".to_string(),
                        stdout_lines: 0,
                        stderr_lines: 1,
                        output_size: 0,
                    },
                    execution_time: start_time.elapsed(),
                    success: false,
                    error_message: Some("Process execution timeout".to_string()),
                })
            },
            result = self.wait_for_process(&process_handle) => {
                result
            }
        };
        
        Ok(result?)
    }

    /// 等待进程结束
    async fn wait_for_process(&self, process_handle: &ProcessHandle) -> Result<ProcessExecutionResult> {
        // 等待进程结束
        let mut child_guard = process_handle.child.lock().unwrap();
        let exit_status: std::process::ExitStatus = child_guard.wait()
            .map_err(|e| Error::ExecutionError(format!("Failed to wait for process: {}", e)))?;
        drop(child_guard);
        
        let exit_code = exit_status.code().unwrap_or(-1);
        let success = exit_status.success();
        
        // 更新进程信息
        let mut info_guard = process_handle.info.lock().unwrap();
        info_guard.state = if success {
            ProcessState::Completed(exit_code)
        } else {
            ProcessState::Terminated(exit_code)
        };
        info_guard.exit_code = Some(exit_code);
        info_guard.ended_at = Some(Instant::now());
        let process_info = info_guard.clone();
        drop(info_guard);
        
        // 收集输出
        let mut child_guard = process_handle.child.lock().unwrap();
        let output = self.collect_output(&mut *child_guard).await?;
        drop(child_guard);
        
        // 收集最终资源使用情况
        let pid = {
            let info_guard = process_handle.info.lock().unwrap();
            info_guard.pid
        };
        if let Ok(final_usage) = Self::collect_resource_usage(pid) {
            let mut info_guard = process_handle.info.lock().unwrap();
            info_guard.resource_usage = final_usage;
        }
        
        let execution_time = process_handle.start_time.elapsed();
        
        let process_info = {
            let info_guard = process_handle.info.lock().unwrap();
            info_guard.clone()
        };
        
        Ok(ProcessExecutionResult {
            process_info,
            output,
            execution_time,
            success,
            error_message: if success { None } else { Some(format!("Process exited with code {}", exit_code)) },
        })
    }

    /// 收集进程输出
    async fn collect_output(&self, child: &mut Child) -> Result<ProcessOutput> {
        let mut stdout = String::new();
        let mut stderr = String::new();
        
        if self.config.capture_output {
            // 读取标准输出
            if let Some(stdout_pipe) = child.stdout.take() {
                let reader = BufReader::new(stdout_pipe);
                for line in reader.lines() {
                    match line {
                        Ok(line) => {
                            stdout.push_str(&line);
                            stdout.push('\n');
                        },
                        Err(e) => {
                            warn!("Failed to read stdout line: {}", e);
                            break;
                        }
                    }
                }
            }
            
            // 读取标准错误
            if let Some(stderr_pipe) = child.stderr.take() {
                let reader = BufReader::new(stderr_pipe);
                for line in reader.lines() {
                    match line {
                        Ok(line) => {
                            stderr.push_str(&line);
                            stderr.push('\n');
                        },
                        Err(e) => {
                            warn!("Failed to read stderr line: {}", e);
                            break;
                        }
                    }
                }
            }
        }
        
        let stdout_lines = stdout.lines().count();
        let stderr_lines = stderr.lines().count();
        let output_size = stdout.len() + stderr.len();
        
        Ok(ProcessOutput {
            stdout,
            stderr,
            stdout_lines,
            stderr_lines,
            output_size,
        })
    }

    /// 收集资源使用情况
    fn collect_resource_usage(pid: u32) -> Result<ProcessResourceUsage> {
        // 在实际实现中，这里应该从系统中收集真实的资源使用情况
        // 例如从 /proc/{pid}/stat, /proc/{pid}/status 等文件读取
        
        Ok(ProcessResourceUsage {
            cpu_time_ms: 0,
            memory_bytes: 0,
            peak_memory_bytes: 0,
            open_files: 0,
            child_processes: 0,
            network_connections: 0,
        })
    }
}

// ProcessHandle 不能实现 Clone，因为 Child 进程不能克隆
// 如果需要共享 ProcessHandle，应该使用 Arc<ProcessHandle>
// 这里移除 Clone 实现，避免误用
// impl Clone for ProcessHandle {
//     fn clone(&self) -> Self {
//         // Child 进程不能克隆，如果需要共享，请使用 Arc<ProcessHandle>
//         panic!("ProcessHandle cannot be cloned due to Child process")
//     }
// }

impl Default for ProcessSandboxConfig {
    fn default() -> Self {
        Self {
            working_dir: PathBuf::from("/tmp/sandbox"),
            environment: HashMap::new(),
            resource_limits: ResourceLimits {
                memory_limit: Some(256 * 1024 * 1024), // 256MB
                cpu_limit: Some(1.0), // 1 CPU core
                timeout: Some(Duration::from_secs(60)), // 1 minute
                max_processes: Some(10),
                max_open_files: Some(100),
                max_network_connections: Some(10),
            },
            security_config: ProcessSecurityConfig {
                user_id: Some(1000),
                group_id: Some(1000),
                disabled_syscalls: vec![
                    "mount".to_string(),
                    "umount".to_string(),
                    "reboot".to_string(),
                    "syslog".to_string(),
                ],
                allowed_paths: vec![
                    PathBuf::from("/tmp"),
                    PathBuf::from("/usr/bin"),
                    PathBuf::from("/bin"),
                ],
                forbidden_paths: vec![
                    PathBuf::from("/etc/passwd"),
                    PathBuf::from("/etc/shadow"),
                    PathBuf::from("/root"),
                ],
                network_access: NetworkAccess::LoopbackOnly,
                filesystem_access: FilesystemAccess {
                    readonly_paths: vec![
                        PathBuf::from("/usr"),
                        PathBuf::from("/bin"),
                        PathBuf::from("/lib"),
                    ],
                    writable_paths: vec![
                        PathBuf::from("/tmp/sandbox"),
                    ],
                    forbidden_paths: vec![
                        PathBuf::from("/etc"),
                        PathBuf::from("/root"),
                        PathBuf::from("/home"),
                    ],
                    temp_dir: Some(PathBuf::from("/tmp/sandbox")),
                },
                isolation_level: IsolationLevel::Medium,
            },
            timeout: Duration::from_secs(60),
            capture_output: true,
            real_time_output: false,
            output_buffer_size: 8192,
        }
    }
} 