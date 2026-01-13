use std::collections::HashMap;
use std::path::{PathBuf};
use std::process::{Command, Stdio};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use serde::{Serialize, Deserialize};
use log::{debug, info, warn};
use uuid::Uuid;

use crate::error::{Error, Result};
use super::{SecurityContext};
use crate::algorithm::types::ResourceLimits;

/// 容器运行时类型
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ContainerRuntime {
    /// Docker容器
    Docker,
    /// Podman容器
    Podman,
    /// 自定义容器运行时
    Custom(String),
}

/// 容器配置
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContainerConfig {
    /// 容器运行时
    pub runtime: ContainerRuntime,
    /// 基础镜像
    pub base_image: String,
    /// 容器名称
    pub container_name: Option<String>,
    /// 工作目录
    pub working_dir: PathBuf,
    /// 环境变量
    pub environment: HashMap<String, String>,
    /// 挂载点
    pub mounts: Vec<MountPoint>,
    /// 网络配置
    pub network_config: NetworkConfig,
    /// 资源限制
    pub resource_limits: ResourceLimits,
    /// 安全配置
    pub security_config: ContainerSecurityConfig,
    /// 超时设置
    pub timeout: Duration,
    /// 是否自动清理
    pub auto_cleanup: bool,
}

/// 挂载点配置
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MountPoint {
    /// 主机路径
    pub host_path: PathBuf,
    /// 容器内路径
    pub container_path: PathBuf,
    /// 挂载类型
    pub mount_type: MountType,
    /// 是否只读
    pub readonly: bool,
}

/// 挂载类型
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MountType {
    /// 绑定挂载
    Bind,
    /// 卷挂载
    Volume,
    /// 临时文件系统
    Tmpfs,
}

/// 网络配置
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkConfig {
    /// 网络模式
    pub mode: NetworkMode,
    /// 端口映射
    pub port_mappings: Vec<PortMapping>,
    /// DNS服务器
    pub dns_servers: Vec<String>,
    /// 主机名
    pub hostname: Option<String>,
}

/// 网络模式
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum NetworkMode {
    /// 无网络
    None,
    /// 主机网络
    Host,
    /// 桥接网络
    Bridge,
    /// 自定义网络
    Custom(String),
}

/// 端口映射
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PortMapping {
    /// 主机端口
    pub host_port: u16,
    /// 容器端口
    pub container_port: u16,
    /// 协议
    pub protocol: Protocol,
}

/// 网络协议
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Protocol {
    TCP,
    UDP,
}

/// 容器安全配置
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContainerSecurityConfig {
    /// 用户ID
    pub user_id: Option<u32>,
    /// 组ID
    pub group_id: Option<u32>,
    /// 特权模式
    pub privileged: bool,
    /// 只读根文件系统
    pub read_only_rootfs: bool,
    /// 禁用的系统调用
    pub disabled_syscalls: Vec<String>,
    /// 允许的能力
    pub allowed_capabilities: Vec<String>,
    /// 禁用的能力
    pub dropped_capabilities: Vec<String>,
    /// SELinux标签
    pub selinux_label: Option<String>,
    /// AppArmor配置文件
    pub apparmor_profile: Option<String>,
}

/// 容器状态
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum ContainerState {
    /// 创建中
    Creating,
    /// 运行中
    Running,
    /// 已暂停
    Paused,
    /// 已停止
    Stopped,
    /// 已退出
    Exited(i32),
    /// 错误状态
    Error(String),
}

/// 容器信息
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContainerInfo {
    /// 容器ID
    pub id: String,
    /// 容器名称
    pub name: String,
    /// 容器状态
    pub state: ContainerState,
    /// 创建时间
    pub created_at: Instant,
    /// 启动时间
    pub started_at: Option<Instant>,
    /// 退出时间
    pub exited_at: Option<Instant>,
    /// 退出码
    pub exit_code: Option<i32>,
    /// 资源使用情况
    pub resource_usage: ResourceUsage,
    /// 网络信息
    pub network_info: NetworkInfo,
}

/// 资源使用情况
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceUsage {
    /// CPU使用率 (%)
    pub cpu_percent: f64,
    /// 内存使用量 (字节)
    pub memory_bytes: u64,
    /// 网络接收字节数
    pub network_rx_bytes: u64,
    /// 网络发送字节数
    pub network_tx_bytes: u64,
    /// 磁盘读取字节数
    pub disk_read_bytes: u64,
    /// 磁盘写入字节数
    pub disk_write_bytes: u64,
}

/// 网络信息
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkInfo {
    /// IP地址
    pub ip_address: Option<String>,
    /// 网关地址
    pub gateway: Option<String>,
    /// 端口映射
    pub port_mappings: Vec<PortMapping>,
}

/// 容器沙箱实现
pub struct ContainerSandbox {
    /// 配置
    config: ContainerConfig,
    /// 容器信息
    container_info: Arc<Mutex<Option<ContainerInfo>>>,
    /// 安全上下文
    security_context: Arc<SecurityContext>,
    /// 是否正在运行
    is_running: Arc<Mutex<bool>>,
}

impl ContainerSandbox {
    /// 创建新的容器沙箱
    pub fn new(config: ContainerConfig, security_context: SecurityContext) -> Self {
        Self {
            config,
            container_info: Arc::new(Mutex::new(None)),
            security_context: Arc::new(security_context),
            is_running: Arc::new(Mutex::new(false)),
        }
    }

    /// 启动容器
    pub async fn start(&self) -> Result<String> {
        let container_id = self.generate_container_id();
        
        // 验证安全配置
        self.validate_security_config()?;
        
        // 准备容器环境
        self.prepare_environment().await?;
        
        // 创建容器
        self.create_container(&container_id).await?;
        
        // 启动容器
        self.start_container(&container_id).await?;
        
        // 更新状态
        self.update_container_info(&container_id, ContainerState::Running).await?;
        
        // 设置运行状态
        *self.is_running.lock().unwrap() = true;
        
        info!("Container {} started successfully", container_id);
        Ok(container_id)
    }

    /// 停止容器
    pub async fn stop(&self, container_id: &str) -> Result<()> {
        if !*self.is_running.lock().unwrap() {
            return Ok(());
        }

        // 优雅停止容器
        self.stop_container(container_id, false).await?;
        
        // 更新状态
        self.update_container_info(container_id, ContainerState::Stopped).await?;
        
        // 清理资源
        if self.config.auto_cleanup {
            self.cleanup_container(container_id).await?;
        }
        
        // 设置运行状态
        *self.is_running.lock().unwrap() = false;
        
        info!("Container {} stopped successfully", container_id);
        Ok(())
    }

    /// 强制停止容器
    pub async fn kill(&self, container_id: &str) -> Result<()> {
        if !*self.is_running.lock().unwrap() {
            return Ok(());
        }

        // 强制停止容器
        self.stop_container(container_id, true).await?;
        
        // 更新状态
        self.update_container_info(container_id, ContainerState::Exited(-1)).await?;
        
        // 清理资源
        if self.config.auto_cleanup {
            self.cleanup_container(container_id).await?;
        }
        
        // 设置运行状态
        *self.is_running.lock().unwrap() = false;
        
        warn!("Container {} killed forcefully", container_id);
        Ok(())
    }

    /// 执行命令
    pub async fn execute(&self, container_id: &str, command: &[String]) -> Result<ExecutionResult> {
        if !*self.is_running.lock().unwrap() {
            return Err(Error::InvalidState("Container is not running".to_string()));
        }

        let start_time = Instant::now();
        
        // 构建执行命令
        let mut cmd = self.build_exec_command(container_id, command)?;
        
        // 执行命令
        let output = cmd.output()
            .map_err(|e| Error::ExecutionError(format!("Failed to execute command: {}", e)))?;
        
        let execution_time = start_time.elapsed();
        
        // 构建执行结果
        let result = ExecutionResult {
            exit_code: output.status.code().unwrap_or(-1),
            stdout: String::from_utf8_lossy(&output.stdout).to_string(),
            stderr: String::from_utf8_lossy(&output.stderr).to_string(),
            execution_time,
            resource_usage: self.get_resource_usage(container_id).await?,
        };
        
        debug!("Command executed in container {}: exit_code={}", container_id, result.exit_code);
        Ok(result)
    }

    /// 获取容器信息
    pub async fn get_info(&self, container_id: &str) -> Result<ContainerInfo> {
        if let Some(info) = self.container_info.lock().unwrap().as_ref() {
            if info.id == container_id {
                return Ok(info.clone());
            }
        }
        
        // 从容器运行时获取信息
        self.fetch_container_info(container_id).await
    }

    /// 获取容器日志
    pub async fn get_logs(&self, container_id: &str, lines: Option<usize>) -> Result<String> {
        let mut cmd = self.build_logs_command(container_id, lines)?;
        
        let output = cmd.output()
            .map_err(|e| Error::ExecutionError(format!("Failed to get logs: {}", e)))?;
        
        if output.status.success() {
            Ok(String::from_utf8_lossy(&output.stdout).to_string())
        } else {
            Err(Error::ExecutionError(format!("Failed to get logs: {}", 
                String::from_utf8_lossy(&output.stderr))))
        }
    }

    /// 生成容器ID
    fn generate_container_id(&self) -> String {
        format!("sandbox-{}", Uuid::new_v4().to_string()[..8].to_lowercase())
    }

    /// 验证安全配置
    fn validate_security_config(&self) -> Result<()> {
        // 检查特权模式
        if self.config.security_config.privileged {
            warn!("Container running in privileged mode - security risk!");
        }
        
        // 检查网络配置
        if matches!(self.config.network_config.mode, NetworkMode::Host) {
            warn!("Container using host network - potential security risk!");
        }
        
        // 检查挂载点
        for mount in &self.config.mounts {
            if mount.host_path.starts_with("/") && !mount.readonly {
                warn!("Container has write access to host path: {:?}", mount.host_path);
            }
        }
        
        Ok(())
    }

    /// 准备容器环境
    async fn prepare_environment(&self) -> Result<()> {
        // 创建工作目录
        if !self.config.working_dir.exists() {
            std::fs::create_dir_all(&self.config.working_dir)
                .map_err(|e| Error::IoError(format!("Failed to create working directory: {}", e)))?;
        }
        
        // 准备挂载点
        for mount in &self.config.mounts {
            if !mount.host_path.exists() {
                std::fs::create_dir_all(&mount.host_path)
                    .map_err(|e| Error::IoError(format!("Failed to create mount point: {}", e)))?;
            }
        }
        
        Ok(())
    }

    /// 创建容器
    async fn create_container(&self, container_id: &str) -> Result<()> {
        let mut cmd = self.build_create_command(container_id)?;
        
        let output = cmd.output()
            .map_err(|e| Error::ExecutionError(format!("Failed to create container: {}", e)))?;
        
        if !output.status.success() {
            return Err(Error::ExecutionError(format!("Failed to create container: {}", 
                String::from_utf8_lossy(&output.stderr))));
        }
        
        debug!("Container {} created successfully", container_id);
        Ok(())
    }

    /// 启动容器
    async fn start_container(&self, container_id: &str) -> Result<()> {
        let mut cmd = self.build_start_command(container_id)?;
        
        let output = cmd.output()
            .map_err(|e| Error::ExecutionError(format!("Failed to start container: {}", e)))?;
        
        if !output.status.success() {
            return Err(Error::ExecutionError(format!("Failed to start container: {}", 
                String::from_utf8_lossy(&output.stderr))));
        }
        
        debug!("Container {} started successfully", container_id);
        Ok(())
    }

    /// 停止容器
    async fn stop_container(&self, container_id: &str, force: bool) -> Result<()> {
        let mut cmd = self.build_stop_command(container_id, force)?;
        
        let output = cmd.output()
            .map_err(|e| Error::ExecutionError(format!("Failed to stop container: {}", e)))?;
        
        if !output.status.success() {
            return Err(Error::ExecutionError(format!("Failed to stop container: {}", 
                String::from_utf8_lossy(&output.stderr))));
        }
        
        debug!("Container {} stopped successfully", container_id);
        Ok(())
    }

    /// 清理容器
    async fn cleanup_container(&self, container_id: &str) -> Result<()> {
        let mut cmd = self.build_remove_command(container_id)?;
        
        let output = cmd.output()
            .map_err(|e| Error::ExecutionError(format!("Failed to remove container: {}", e)))?;
        
        if !output.status.success() {
            warn!("Failed to remove container {}: {}", container_id, 
                String::from_utf8_lossy(&output.stderr));
        } else {
            debug!("Container {} removed successfully", container_id);
        }
        
        Ok(())
    }

    /// 构建创建命令
    fn build_create_command(&self, container_id: &str) -> Result<Command> {
        let mut cmd = Command::new(self.get_runtime_command());
        
        match self.config.runtime {
            ContainerRuntime::Docker => {
                cmd.arg("create");
                cmd.arg("--name").arg(container_id);
                
                // 添加资源限制
                if let Some(memory) = self.config.resource_limits.memory_limit {
                    cmd.arg("--memory").arg(format!("{}b", memory));
                }
                if let Some(cpu) = self.config.resource_limits.cpu_limit {
                    cmd.arg("--cpus").arg(cpu.to_string());
                }
                
                // 添加网络配置
                match &self.config.network_config.mode {
                    NetworkMode::None => { cmd.arg("--network").arg("none"); },
                    NetworkMode::Host => { cmd.arg("--network").arg("host"); },
                    NetworkMode::Bridge => { cmd.arg("--network").arg("bridge"); },
                    NetworkMode::Custom(network) => { cmd.arg("--network").arg(network); },
                }
                
                // 添加端口映射
                for port in &self.config.network_config.port_mappings {
                    cmd.arg("-p").arg(format!("{}:{}/{:?}", 
                        port.host_port, port.container_port, port.protocol).to_lowercase());
                }
                
                // 添加挂载点
                for mount in &self.config.mounts {
                    let mount_str = format!("{}:{}:{}", 
                        mount.host_path.display(),
                        mount.container_path.display(),
                        if mount.readonly { "ro" } else { "rw" });
                    cmd.arg("-v").arg(mount_str);
                }
                
                // 添加环境变量
                for (key, value) in &self.config.environment {
                    cmd.arg("-e").arg(format!("{}={}", key, value));
                }
                
                // 添加安全配置
                if self.config.security_config.privileged {
                    cmd.arg("--privileged");
                }
                if self.config.security_config.read_only_rootfs {
                    cmd.arg("--read-only");
                }
                if let Some(user_id) = self.config.security_config.user_id {
                    cmd.arg("--user").arg(user_id.to_string());
                }
                
                // 添加工作目录
                cmd.arg("-w").arg(&self.config.working_dir);
                
                // 添加基础镜像
                cmd.arg(&self.config.base_image);
            },
            ContainerRuntime::Podman => {
                // Podman命令构建逻辑
                cmd.arg("create");
                cmd.arg("--name").arg(container_id);
                // ... 类似Docker的配置
                cmd.arg(&self.config.base_image);
            },
            ContainerRuntime::Custom(ref runtime) => {
                return Err(Error::NotImplemented(format!("Custom runtime {} not implemented", runtime)));
            },
        }
        
        Ok(cmd)
    }

    /// 构建启动命令
    fn build_start_command(&self, container_id: &str) -> Result<Command> {
        let mut cmd = Command::new(self.get_runtime_command());
        cmd.arg("start").arg(container_id);
        Ok(cmd)
    }

    /// 构建停止命令
    fn build_stop_command(&self, container_id: &str, force: bool) -> Result<Command> {
        let mut cmd = Command::new(self.get_runtime_command());
        
        if force {
            cmd.arg("kill").arg(container_id);
        } else {
            cmd.arg("stop");
            if let Some(timeout) = self.config.timeout.as_secs().checked_sub(0) {
                cmd.arg("-t").arg(timeout.to_string());
            }
            cmd.arg(container_id);
        }
        
        Ok(cmd)
    }

    /// 构建执行命令
    fn build_exec_command(&self, container_id: &str, command: &[String]) -> Result<Command> {
        let mut cmd = Command::new(self.get_runtime_command());
        cmd.arg("exec").arg(container_id);
        
        for arg in command {
            cmd.arg(arg);
        }
        
        cmd.stdout(Stdio::piped());
        cmd.stderr(Stdio::piped());
        
        Ok(cmd)
    }

    /// 构建日志命令
    fn build_logs_command(&self, container_id: &str, lines: Option<usize>) -> Result<Command> {
        let mut cmd = Command::new(self.get_runtime_command());
        cmd.arg("logs");
        
        if let Some(n) = lines {
            cmd.arg("--tail").arg(n.to_string());
        }
        
        cmd.arg(container_id);
        cmd.stdout(Stdio::piped());
        cmd.stderr(Stdio::piped());
        
        Ok(cmd)
    }

    /// 构建删除命令
    fn build_remove_command(&self, container_id: &str) -> Result<Command> {
        let mut cmd = Command::new(self.get_runtime_command());
        cmd.arg("rm").arg("-f").arg(container_id);
        Ok(cmd)
    }

    /// 获取运行时命令
    fn get_runtime_command(&self) -> &str {
        match self.config.runtime {
            ContainerRuntime::Docker => "docker",
            ContainerRuntime::Podman => "podman",
            ContainerRuntime::Custom(ref cmd) => cmd,
        }
    }

    /// 更新容器信息
    async fn update_container_info(&self, container_id: &str, state: ContainerState) -> Result<()> {
        let mut info_guard = self.container_info.lock().unwrap();
        
        if let Some(ref mut info) = *info_guard {
            info.state = state.clone();
            match &state {
                ContainerState::Running => {
                    info.started_at = Some(Instant::now());
                },
                ContainerState::Stopped | ContainerState::Exited(_) => {
                    info.exited_at = Some(Instant::now());
                },
                _ => {},
            }
        } else {
            *info_guard = Some(ContainerInfo {
                id: container_id.to_string(),
                name: self.config.container_name.clone().unwrap_or_else(|| container_id.to_string()),
                state,
                created_at: Instant::now(),
                started_at: None,
                exited_at: None,
                exit_code: None,
                resource_usage: ResourceUsage {
                    cpu_percent: 0.0,
                    memory_bytes: 0,
                    network_rx_bytes: 0,
                    network_tx_bytes: 0,
                    disk_read_bytes: 0,
                    disk_write_bytes: 0,
                },
                network_info: NetworkInfo {
                    ip_address: None,
                    gateway: None,
                    port_mappings: self.config.network_config.port_mappings.clone(),
                },
            });
        }
        
        Ok(())
    }

    /// 获取资源使用情况
    async fn get_resource_usage(&self, container_id: &str) -> Result<ResourceUsage> {
        let mut cmd = Command::new(self.get_runtime_command());
        cmd.arg("stats").arg("--no-stream").arg("--format").arg("json").arg(container_id);
        
        let output = cmd.output()
            .map_err(|e| Error::ExecutionError(format!("Failed to get resource usage: {}", e)))?;
        
        if output.status.success() {
            // 解析JSON输出获取资源使用情况
            // 这里简化处理，实际应该解析JSON
            Ok(ResourceUsage {
                cpu_percent: 0.0,
                memory_bytes: 0,
                network_rx_bytes: 0,
                network_tx_bytes: 0,
                disk_read_bytes: 0,
                disk_write_bytes: 0,
            })
        } else {
            Err(Error::ExecutionError("Failed to get resource usage".to_string()))
        }
    }

    /// 从容器运行时获取信息
    async fn fetch_container_info(&self, container_id: &str) -> Result<ContainerInfo> {
        let mut cmd = Command::new(self.get_runtime_command());
        cmd.arg("inspect").arg("--format").arg("json").arg(container_id);
        
        let output = cmd.output()
            .map_err(|e| Error::ExecutionError(format!("Failed to inspect container: {}", e)))?;
        
        if output.status.success() {
            // 解析JSON输出获取容器信息
            // 这里简化处理，实际应该解析JSON
            Ok(ContainerInfo {
                id: container_id.to_string(),
                name: container_id.to_string(),
                state: ContainerState::Running,
                created_at: Instant::now(),
                started_at: Some(Instant::now()),
                exited_at: None,
                exit_code: None,
                resource_usage: ResourceUsage {
                    cpu_percent: 0.0,
                    memory_bytes: 0,
                    network_rx_bytes: 0,
                    network_tx_bytes: 0,
                    disk_read_bytes: 0,
                    disk_write_bytes: 0,
                },
                network_info: NetworkInfo {
                    ip_address: None,
                    gateway: None,
                    port_mappings: Vec::new(),
                },
            })
        } else {
            Err(Error::ExecutionError(format!("Container {} not found", container_id)))
        }
    }
}

/// 执行结果
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionResult {
    /// 退出码
    pub exit_code: i32,
    /// 标准输出
    pub stdout: String,
    /// 标准错误
    pub stderr: String,
    /// 执行时间
    pub execution_time: Duration,
    /// 资源使用情况
    pub resource_usage: ResourceUsage,
}

impl Default for ContainerConfig {
    fn default() -> Self {
        Self {
            runtime: ContainerRuntime::Docker,
            base_image: "ubuntu:20.04".to_string(),
            container_name: None,
            working_dir: PathBuf::from("/workspace"),
            environment: HashMap::new(),
            mounts: Vec::new(),
            network_config: NetworkConfig {
                mode: NetworkMode::Bridge,
                port_mappings: Vec::new(),
                dns_servers: Vec::new(),
                hostname: None,
            },
            resource_limits: ResourceLimits {
                memory_limit: Some(512 * 1024 * 1024), // 512MB
                cpu_limit: Some(1.0), // 1 CPU core
                timeout: Some(Duration::from_secs(300)), // 5 minutes
                max_processes: Some(100),
                max_open_files: Some(1024),
                max_network_connections: Some(100),
            },
            security_config: ContainerSecurityConfig {
                user_id: Some(1000),
                group_id: Some(1000),
                privileged: false,
                read_only_rootfs: true,
                disabled_syscalls: vec![
                    "mount".to_string(),
                    "umount".to_string(),
                    "reboot".to_string(),
                    "syslog".to_string(),
                ],
                allowed_capabilities: Vec::new(),
                dropped_capabilities: vec![
                    "SYS_ADMIN".to_string(),
                    "SYS_MODULE".to_string(),
                    "SYS_RAWIO".to_string(),
                ],
                selinux_label: None,
                apparmor_profile: None,
            },
            timeout: Duration::from_secs(300),
            auto_cleanup: true,
        }
    }
} 