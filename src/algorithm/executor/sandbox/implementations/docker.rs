use std::path::{Path, PathBuf};
use std::time::Duration;
use std::sync::{Arc, RwLock, Mutex};
use async_trait::async_trait;
#[cfg(feature = "tempfile")]
use tempfile::TempDir;
use uuid::Uuid;
use log::{debug, warn, error, info};
use tokio::process::Command;
use tokio::time::timeout;
use tokio::io::AsyncWriteExt;
use tokio::sync::oneshot;
use serde::Serialize;
use std::sync::atomic::{AtomicBool, Ordering};

use crate::{Error, Result};
use crate::algorithm::executor::config::ExecutorConfig;
#[cfg(feature = "tempfile")]
use crate::algorithm::types::NetworkPolicy;
use crate::algorithm::types::{ResourceUsage, SandboxSecurityLevel};
use crate::algorithm::executor::sandbox::interface::Sandbox;
use crate::algorithm::executor::sandbox::result::SandboxResult;
use crate::algorithm::types::SandboxStatus;
use crate::algorithm::executor::sandbox::environment::ExecutionEnvironment;
#[cfg(feature = "tempfile")]
use crate::algorithm::executor::sandbox::error::SandboxError;

/// Docker容器映像配置
#[derive(Debug, Clone, Serialize)]
pub struct DockerImageConfig {
    /// 基础镜像
    pub base_image: String,
    /// 标签
    pub tag: String,
    /// 额外的构建参数
    pub build_args: Vec<String>,
    /// 要安装的包
    pub packages: Vec<String>,
}

impl Default for DockerImageConfig {
    fn default() -> Self {
        Self {
            base_image: "alpine".to_string(),
            tag: "latest".to_string(),
            build_args: vec![],
            packages: vec!["python3".to_string(), "nodejs".to_string()],
        }
    }
}

/// Docker沙箱实现
/// 提供一个基于Docker容器的安全执行环境
#[derive(Debug)]
pub struct DockerSandbox {
    /// 沙箱ID
    id: String,
    /// 容器ID
    container_id: Option<String>,
    /// 执行器配置
    config: ExecutorConfig,
    /// 临时目录
    #[cfg(feature = "tempfile")]
    temp_dir: Option<TempDir>,
    /// 状态
    status: RwLock<SandboxStatus>,
    /// 取消通道
    cancel_tx: Option<Arc<AtomicBool>>,
    /// 资源监控器
    resource_monitor: Arc<Mutex<ResourceMonitor>>,
    /// 执行环境
    environment: Option<Arc<tokio::sync::RwLock<ExecutionEnvironment>>>,
    /// Docker镜像配置
    image_config: DockerImageConfig,
}

/// 资源监控器
#[derive(Debug)]
struct ResourceMonitor {
    /// 容器ID
    container_id: Option<String>,
    /// 开始时间
    start_time: std::time::Instant,
    /// 最大内存使用(bytes)
    peak_memory: u64,
    /// 最大CPU使用(%)
    peak_cpu: f64,
    /// 当前状态
    is_monitoring: bool,
    /// 停止通道
    stop_tx: Option<oneshot::Sender<()>>,
}

impl DockerSandbox {
    /// 创建新的Docker沙箱
    pub async fn new(config: &ExecutorConfig) -> Result<Self> {
        let id = format!("docker-sandbox-{}", Uuid::new_v4());
        #[cfg(feature = "tempfile")]
        let temp_dir = TempDir::new().ok();
        
        // 验证Docker是否可用
        Self::verify_docker_available().await?;
        
        let resource_monitor = Arc::new(Mutex::new(ResourceMonitor {
            container_id: None,
            start_time: std::time::Instant::now(),
            peak_memory: 0,
            peak_cpu: 0.0,
            is_monitoring: false,
            stop_tx: None,
        }));
        
        // 配置Docker镜像
        let image_config = match &config.sandbox_config.security_level {
            SandboxSecurityLevel::Low => DockerImageConfig {
                base_image: "python".to_string(),
                tag: "3.9-slim".to_string(),
                packages: vec![
                    "python3-numpy".to_string(),
                    "python3-pandas".to_string(),
                    "python3-scipy".to_string(),
                    "nodejs".to_string(),
                ],
                ..Default::default()
            },
            SandboxSecurityLevel::Medium => DockerImageConfig {
                base_image: "python".to_string(),
                tag: "3.9-alpine".to_string(),
                packages: vec![
                    "py3-numpy".to_string(),
                    "py3-pandas".to_string(),
                    "nodejs".to_string(),
                ],
                ..Default::default()
            },
            _ => DockerImageConfig::default(),
        };
        
        Ok(Self {
            id,
            container_id: None,
            config: config.clone(),
            #[cfg(feature = "tempfile")]
            temp_dir,
            status: RwLock::new(SandboxStatus::Uninitialized),
            cancel_tx: None,
            resource_monitor,
            environment: None,
            image_config,
        })
    }
    
    /// 验证Docker是否可用
    async fn verify_docker_available() -> Result<()> {
        debug!("验证Docker是否可用");
        
        match Command::new("docker").arg("--version").output().await {
            Ok(output) => {
                if output.status.success() {
                    let version = String::from_utf8_lossy(&output.stdout);
                    debug!("Docker版本: {}", version.trim());
                    Ok(())
                } else {
                    let error = String::from_utf8_lossy(&output.stderr);
                    error!("Docker命令返回非零状态码: {}", error);
                    Err(Error::invalid_state(format!("Docker不可用: {}", error)))
                }
            },
            Err(e) => {
                error!("Docker命令执行失败: {}", e);
                Err(Error::invalid_state(format!("Docker命令执行失败: {}", e)))
            }
        }
    }
    
    /// 创建容器
    async fn create_container(&mut self, _code: &[u8]) -> Result<String> {
        debug!("【DockerSandbox】创建容器: {}", self.id);
        
        // 准备临时目录
        #[cfg(not(feature = "tempfile"))]
        return Err(Error::invalid_state("tempfile feature未启用，无法创建临时目录"));
        
        #[cfg(feature = "tempfile")]
        {
            let temp_dir = self.temp_dir.as_ref().ok_or_else(|| {
                Error::invalid_state("临时目录未初始化")
            })?;
            
            // 创建Dockerfile
            let dockerfile_path = temp_dir.path().join("Dockerfile");
            let dockerfile_content = self.generate_dockerfile()?;
            tokio::fs::write(&dockerfile_path, dockerfile_content).await?;
            
            // 创建执行脚本
            let script_path = temp_dir.path().join("script.py");
            tokio::fs::write(&script_path, code).await?;
            
            // 创建工作目录
            let work_dir = temp_dir.path().join("workdir");
            tokio::fs::create_dir_all(&work_dir).await?;
            
            // 创建输入、输出目录
            let input_dir = temp_dir.path().join("input");
            let output_dir = temp_dir.path().join("output");
            tokio::fs::create_dir_all(&input_dir).await?;
            tokio::fs::create_dir_all(&output_dir).await?;
            
            // 构建镜像
            let image_name = format!("vecmind-sandbox-{}", self.id);
            let build_output = Command::new("docker")
                .arg("build")
                .arg("-t")
                .arg(&image_name)
                .arg("-f")
                .arg(&dockerfile_path)
                .arg(temp_dir.path())
                .output()
                .await?;
            
            if !build_output.status.success() {
                let error = String::from_utf8_lossy(&build_output.stderr);
                error!("Docker镜像构建失败: {}", error);
                return Err(Error::internal(format!("Docker镜像构建失败: {}", error)));
            }
            
            // 创建容器
            let memory_limit = format!("{}m", self.config.resource_limits.max_memory_usage / (1024 * 1024));
            let cpu_limit = format!("{}", self.config.resource_limits.max_execution_time_ms / 1000);
            
            let mut cmd = Command::new("docker");
            cmd.arg("create")
                .arg("--name")
                .arg(&self.id)
                .arg("--memory")
                .arg(&memory_limit)
                .arg("--memory-swap")
                .arg(&memory_limit) // 禁用交换
                .arg("--cpus")
                .arg(&cpu_limit)
                .arg("--pids-limit")
                .arg("256") // 临时设置一个合理的默认值
                .arg("--network")
                .arg(if matches!(self.config.sandbox_config.network_policy, NetworkPolicy::Denied) { "none" } else { "bridge" })
                .arg("-v")
                .arg(format!("{}:/app/input:ro", input_dir.to_str().unwrap()))
                .arg("-v")
                .arg(format!("{}:/app/output:rw", output_dir.to_str().unwrap()))
            .arg("-w")
            .arg("/app")
                .arg("--cap-drop=ALL"); // 移除所有能力
            
            // 根据安全级别设置额外限制
            match self.config.sandbox_config.security_level {
                SandboxSecurityLevel::Strict | SandboxSecurityLevel::High => {
                    cmd.arg("--security-opt=no-new-privileges")
                       .arg("--read-only")
                       .arg("--tmpfs")
                       .arg("/tmp:rw,noexec,nosuid,size=64m");
                },
                _ => {}
            }
            
            // 完成容器创建命令
            cmd.arg(&image_name)
               .arg("python3")
               .arg("/app/script.py");
            
            let output = cmd.output().await?;
            
            if !output.status.success() {
                let error = String::from_utf8_lossy(&output.stderr);
                error!("Docker容器创建失败: {}", error);
                return Err(Error::internal(format!("Docker容器创建失败: {}", error)));
            }
            
            let container_id = String::from_utf8_lossy(&output.stdout).trim().to_string();
            debug!("容器创建成功, ID: {}", container_id);
            
            self.container_id = Some(container_id.clone());
            
            // 初始化资源监控器
            {
                let mut monitor = self.resource_monitor.lock().unwrap();
                monitor.container_id = Some(container_id.clone());
            }
            
            Ok(container_id)
        }
    }
    
    /// 生成Dockerfile
    fn generate_dockerfile(&self) -> Result<String> {
        let mut dockerfile = String::new();
        
        // 基础镜像
        dockerfile.push_str(&format!("FROM {}:{}\n\n", 
                             self.image_config.base_image, 
                             self.image_config.tag));
        
        // 标签
        dockerfile.push_str("LABEL maintainer=\"VecMind\" \\\n");
        dockerfile.push_str("      description=\"VecMind安全执行环境\" \\\n");
        dockerfile.push_str(&format!("      sandbox_id=\"{}\"\n\n", self.id));
        
        // 安装包
        if !self.image_config.packages.is_empty() {
            if self.image_config.base_image.contains("alpine") {
                dockerfile.push_str("RUN apk update && apk add --no-cache \\\n");
                for (i, package) in self.image_config.packages.iter().enumerate() {
                    dockerfile.push_str(&format!("    {} {}\n", 
                        package,
                        if i < self.image_config.packages.len() - 1 { "\\" } else { "" }
                    ));
                }
            } else if self.image_config.base_image.contains("debian") || 
                      self.image_config.base_image.contains("ubuntu") {
                dockerfile.push_str("RUN apt-get update && apt-get install -y --no-install-recommends \\\n");
                for (i, package) in self.image_config.packages.iter().enumerate() {
                    dockerfile.push_str(&format!("    {} {}\n", 
                        package,
                        if i < self.image_config.packages.len() - 1 { "\\" } else { "" }
                    ));
                }
                dockerfile.push_str("    && apt-get clean && rm -rf /var/lib/apt/lists/*\n\n");
            }
        }
        
        // 工作目录
        dockerfile.push_str("WORKDIR /app\n\n");
        
        // 创建必要目录
        dockerfile.push_str("RUN mkdir -p /app/input /app/output\n\n");
        
        // 复制脚本
        dockerfile.push_str("COPY script.py /app/script.py\n\n");
        
        // 权限设置
        dockerfile.push_str("RUN chmod 755 /app/script.py\n\n");
        
        // 用户设置 (非root)
        dockerfile.push_str("RUN adduser -D -u 1000 vecmind\n");
        dockerfile.push_str("USER vecmind\n\n");
        
        // 执行命令
        dockerfile.push_str("CMD [\"python3\", \"/app/script.py\"]\n");
        
        Ok(dockerfile)
    }
    
    /// 启动资源监控
    async fn start_resource_monitoring(&self) -> Result<()> {
        let container_id = match &self.container_id {
            Some(id) => id.clone(),
            None => return Err(Error::invalid_state("容器未创建")),
        };
        
        let monitor_arc = self.resource_monitor.clone();
        
        // 创建取消通道
        let (stop_tx, mut stop_rx) = oneshot::channel::<()>();
        
        {
            let mut monitor = monitor_arc.lock().unwrap();
            monitor.is_monitoring = true;
            monitor.stop_tx = Some(stop_tx);
        }
        
        // 启动监控任务
        tokio::spawn(async move {
            let monitor_interval = Duration::from_millis(500);
            let mut interval = tokio::time::interval(monitor_interval);
            
            loop {
                tokio::select! {
                    _ = interval.tick() => {
                        let stats_cmd = Command::new("docker")
                            .arg("stats")
                            .arg("--no-stream")
                            .arg("--format")
                            .arg("{{.MemUsage}}|{{.CPUPerc}}")
                            .arg(&container_id)
                            .output()
                            .await;
                            
                        if let Ok(output) = stats_cmd {
                            if output.status.success() {
                                let stats_str = String::from_utf8_lossy(&output.stdout);
                                let parts: Vec<&str> = stats_str.trim().split('|').collect();
                                
                                if parts.len() >= 2 {
                                    // 解析内存使用
                                    if let Some(mem_part) = parts.get(0) {
                                        if let Some(mem_val) = Self::parse_memory_usage(mem_part) {
                                            let mut monitor = monitor_arc.lock().unwrap();
                                            if mem_val > monitor.peak_memory {
                                                monitor.peak_memory = mem_val;
                                            }
                                        }
                                    }
                                    
                                    // 解析CPU使用
                                    if let Some(cpu_part) = parts.get(1) {
                                        if let Some(cpu_val) = Self::parse_cpu_usage(cpu_part) {
                                            let mut monitor = monitor_arc.lock().unwrap();
                                            if cpu_val > monitor.peak_cpu {
                                                monitor.peak_cpu = cpu_val;
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                    _ = &mut stop_rx => {
                        debug!("资源监控停止: {}", container_id);
                        break;
                    }
                }
            }
            
            // 监控结束
            let mut monitor = monitor_arc.lock().unwrap();
            monitor.is_monitoring = false;
        });
        
        Ok(())
    }
    
    /// 解析内存使用量
    fn parse_memory_usage(mem_str: &str) -> Option<u64> {
        // 格式例如: "100MiB / 2GiB"
        let parts: Vec<&str> = mem_str.split('/').collect();
        if parts.is_empty() {
            return None;
        }
        
        let usage_str = parts[0].trim();
        
        // 提取数字部分
        let mut num_str = String::new();
        for c in usage_str.chars() {
            if c.is_digit(10) || c == '.' {
                num_str.push(c);
            } else {
                break;
            }
        }
        
        let num = match num_str.parse::<f64>() {
            Ok(n) => n,
            Err(_) => return None,
        };
        
        // 确定单位
        let unit = usage_str.trim_start_matches(|c: char| c.is_digit(10) || c == '.').trim();
        
        // 转换为字节
        let bytes = match unit {
            "B" => num as u64,
            "KB" | "KiB" => num as u64 * 1024,
            "MB" | "MiB" => num as u64 * 1024 * 1024,
            "GB" | "GiB" => num as u64 * 1024 * 1024 * 1024,
            _ => return None,
        };
        
        Some(bytes)
    }
    
    /// 解析CPU使用百分比
    fn parse_cpu_usage(cpu_str: &str) -> Option<f64> {
        // 格式例如: "0.50%"
        let cpu_str = cpu_str.trim().trim_end_matches('%');
        cpu_str.parse::<f64>().ok()
    }
    
    /// 停止资源监控
    fn stop_resource_monitoring(&self) -> Result<()> {
        let mut monitor = self.resource_monitor.lock().unwrap();
        
        if let Some(tx) = monitor.stop_tx.take() {
            let _ = tx.send(());
        }
        
        monitor.is_monitoring = false;
        
        Ok(())
    }
    
    /// 设置沙箱状态
    fn set_status(&self, status: SandboxStatus) -> Result<()> {
        let mut current_status = self.status.write().map_err(|_| {
            Error::lock("无法获取沙箱状态锁")
        })?;
        *current_status = status;
        Ok(())
    }
    
    /// 清理容器
    async fn cleanup_container(&mut self) -> Result<()> {
        // 停止资源监控
        self.stop_resource_monitoring()?;
        
        if let Some(container_id) = &self.container_id {
            debug!("清理容器: {}", container_id);
            
            // 停止容器
            let stop_result = Command::new("docker")
                .arg("stop")
                .arg(container_id)
                .output()
                .await;
                
            if let Err(e) = stop_result {
                warn!("停止容器失败: {}, 错误: {}", container_id, e);
            }
            
            // 删除容器
            let rm_result = Command::new("docker")
                .arg("rm")
                .arg("-f")
                .arg(container_id)
                .output()
                .await;
                
            if let Err(e) = rm_result {
                warn!("删除容器失败: {}, 错误: {}", container_id, e);
            }
            
            self.container_id = None;
        }
        
        // 清理镜像
        let image_name = format!("vecmind-sandbox-{}", self.id);
        let rmi_result = Command::new("docker")
            .arg("rmi")
            .arg("-f")
            .arg(&image_name)
            .output()
            .await;
            
        if let Err(e) = rmi_result {
            warn!("删除镜像失败: {}, 错误: {}", image_name, e);
        }
        
        Ok(())
    }
    
    /// 创建工作目录并返回路径
    async fn create_work_directory(&self) -> Result<PathBuf> {
        debug!("【DockerSandbox】创建工作目录");
        #[cfg(not(feature = "tempfile"))]
        return Err(Error::invalid_state("tempfile feature未启用，无法创建工作目录"));
        
        #[cfg(feature = "tempfile")]
        {
            let temp_dir = self.temp_dir.as_ref().ok_or_else(|| {
                Error::invalid_state("临时目录未初始化")
            })?;
            
            let work_dir = temp_dir.path().join("workdir");
            tokio::fs::create_dir_all(&work_dir).await?;
            
            info!("【DockerSandbox】工作目录已创建: {:?}", work_dir);
            Ok(work_dir)
        }
    }
    
    /// 执行命令并设置超时
    async fn execute_command_with_timeout(&self, 
                                         cmd: &mut Command, 
                                         timeout_duration: Duration) -> Result<std::process::Output> {
        debug!("【DockerSandbox】执行命令，超时设置: {:?}", timeout_duration);
        
        match timeout(timeout_duration, cmd.output()).await {
            Ok(result) => {
                let output = result?;
                if output.status.success() {
                    info!("命令执行成功，耗时小于超时时间");
                    Ok(output)
                } else {
                    let error = String::from_utf8_lossy(&output.stderr);
                    error!("命令执行失败: {}", error);
                    Err(Error::internal(format!("命令执行失败: {}", error)))
                }
            },
            Err(_) => {
                error!("命令执行超时: {:?}", timeout_duration);
                Err(Error::timeout(format!(
                    "命令执行超时，超过了设定的 {:?}", timeout_duration
                )))
            }
        }
    }
    
    /// 复制文件到指定沙箱路径
    async fn copy_file_to_sandbox(&self, src_path: &Path, dest_path: &str) -> Result<()> {
        let container_id = self.container_id.as_ref().ok_or_else(|| {
            Error::invalid_state("容器尚未创建")
        })?;
        
        // 确保源文件存在
        if !src_path.exists() {
            return Err(Error::not_found(format!(
                "源文件不存在: {:?}", src_path
            )));
        }
        
        // 构建目标路径
        let dest_pathbuf = PathBuf::from(dest_path);
        
        // 创建目标目录
        let parent_dir = dest_pathbuf.parent().ok_or_else(|| {
            Error::invalid_argument("无法获取目标路径的父目录")
        })?;
        
        let mkdir_cmd = Command::new("docker")
            .arg("exec")
            .arg(container_id)
            .arg("mkdir")
            .arg("-p")
            .arg(parent_dir.to_str().unwrap())
            .output()
            .await?;
            
        if !mkdir_cmd.status.success() {
            let error = String::from_utf8_lossy(&mkdir_cmd.stderr);
            warn!("创建目录失败，可能已存在: {}", error);
        }
        
        // 使用docker cp复制文件
        let copy_cmd = Command::new("docker")
            .arg("cp")
            .arg(src_path)
            .arg(format!("{}:{}", container_id, dest_path))
            .output()
            .await?;
            
        if !copy_cmd.status.success() {
            let error = String::from_utf8_lossy(&copy_cmd.stderr);
            error!("复制文件到容器失败: {}", error);
            return Err(Error::internal(format!(
                "复制文件到容器失败: {}", error
            )));
        }
        
        info!("文件已成功复制到沙箱: {:?} -> {}", src_path, dest_path);
        Ok(())
    }
}

#[async_trait]
impl Sandbox for DockerSandbox {
    fn id(&self) -> &str {
        &self.id
    }
    
    async fn prepare(&self) -> Result<()> {
        debug!("【DockerSandbox】准备环境: {}", self.id);
        self.set_status(SandboxStatus::Ready)?;
        Ok(())
    }
    
    async fn execute(&self, code: &[u8], input: &[u8], timeout_duration: Duration) -> Result<SandboxResult> {
        debug!("【DockerSandbox】执行代码: {}, 输入大小: {}, 超时: {:?}", 
              self.id, input.len(), timeout_duration);
        
        // 将状态设置为运行中
        self.set_status(SandboxStatus::Running)?;
        let start_time = std::time::Instant::now();
        
        // 创建容器
        let container_id = match &self.container_id {
            Some(id) => id.clone(),
            None => {
                // 需要转换self为可变引用，这里使用克隆然后修改的方式
                let mut this = self.clone();
                match this.create_container(code).await {
                    Ok(id) => {
                        // 将容器ID设置回原始结构
                        {
                            let mut status = self.container_id.write().unwrap();
                            *status = Some(id.clone());
                        }
                        id
                    },
                    Err(e) => {
                        self.set_status(SandboxStatus::Failed)?;
                        return Err(e);
                    }
                }
            }
        };
        
        // 准备临时目录
        #[cfg(not(feature = "tempfile"))]
        {
            self.set_status(SandboxStatus::Failed)?;
            return Err(Error::invalid_state("tempfile feature未启用，无法创建临时目录"));
        }
        
        #[cfg(feature = "tempfile")]
        {
            let temp_dir = match &self.temp_dir {
                Some(dir) => dir,
                None => {
                    self.set_status(SandboxStatus::Failed)?;
                    return Err(Error::invalid_state("临时目录未初始化"));
                }
            };
            
            // 写入输入数据
            let input_path = temp_dir.path().join("input/input.dat");
            tokio::fs::write(&input_path, input).await?;
        }
        
        // 启动资源监控
        self.start_resource_monitoring().await?;
        
        // 启动容器
        debug!("启动容器: {}", container_id);
        let start_output = Command::new("docker")
            .arg("start")
            .arg("-a")
            .arg(&container_id)
            .output()
            .await?;
            
        // 停止资源监控
        self.stop_resource_monitoring()?;
        
        let execution_time = start_time.elapsed();
        
        // 检查执行结果
        let success = start_output.status.success();
        let stdout = String::from_utf8_lossy(&start_output.stdout).to_string();
        let stderr = String::from_utf8_lossy(&start_output.stderr).to_string();
        
        // 读取输出文件
        #[cfg(not(feature = "tempfile"))]
        {
            self.set_status(SandboxStatus::Failed)?;
            return Err(Error::invalid_state("tempfile feature未启用，无法读取输出文件"));
        }
        
        #[cfg(feature = "tempfile")]
        let output_data = {
            let temp_dir = self.temp_dir.as_ref().ok_or_else(|| {
                Error::invalid_state("临时目录未初始化")
            })?;
            let output_path = temp_dir.path().join("output/output.dat");
            
            if output_path.exists() {
                match tokio::fs::read(&output_path).await {
                    Ok(data) => data,
                    Err(e) => {
                        warn!("读取输出文件失败: {}", e);
                        Vec::new()
                    }
                }
            } else {
                Vec::new()
            }
        };
        
        // 获取资源使用情况
        let resource_usage = {
            let monitor = self.resource_monitor.lock().unwrap();
            ResourceUsage {
                memory_bytes: monitor.peak_memory,
                cpu_time_ms: (execution_time.as_millis() as f64 * monitor.peak_cpu / 100.0) as u64,
                execution_time_ms: execution_time.as_millis() as u64,
                disk_io_bytes: 0, // 暂不支持
                network_io_bytes: 0, // 暂不支持
            }
        };
        
        // 设置状态为已完成
        if success {
            self.set_status(SandboxStatus::Completed)?;
        } else {
            self.set_status(SandboxStatus::Failed)?;
        }
        
        // 创建执行结果
        #[cfg(feature = "tempfile")]
        {
            let result = SandboxResult {
                success,
                output: output_data,
                stdout,
                stderr,
                resource_usage,
                error: if !success {
                    Some(SandboxError::ExecutionFailed(
                        format!("容器执行失败: {}", stderr)
                    ))
                } else {
                    None
                },
            };
            
            Ok(result)
        }
    }
    
    async fn cleanup(&self) -> Result<()> {
        debug!("【DockerSandbox】清理资源: {}", self.id);
        
        // 这里需要可变引用，使用克隆-修改的方式
        let mut this = self.clone();
        this.cleanup_container().await?;
        
        self.set_status(SandboxStatus::Disposed)?;
        Ok(())
    }
    
    fn supports_file_type(&self, file_type: &str) -> bool {
        // Docker沙箱支持多种文件类型
        match file_type.to_lowercase().as_str() {
            "py" | "python" => true,
            "js" | "javascript" => true,
            "sh" | "bash" => true,
            "txt" | "csv" | "json" => true,
            _ => false,
        }
    }
    
    async fn load_file(&self, src_path: &Path, sandbox_path: &str) -> Result<()> {
        #[cfg(not(feature = "tempfile"))]
        return Err(Error::invalid_state("tempfile feature未启用，无法加载文件"));
        
        #[cfg(feature = "tempfile")]
        if let Some(temp_dir) = &self.temp_dir {
            let dest_path = temp_dir.path().join("input").join(sandbox_path);
            
            // 确保目标目录存在
            if let Some(parent) = dest_path.parent() {
                tokio::fs::create_dir_all(parent).await?;
            }
            
            // 复制文件
            tokio::fs::copy(src_path, dest_path).await?;
            Ok(())
        } else {
            Err(Error::invalid_state("临时目录未初始化"))
        }
    }
    
    async fn save_file(&self, sandbox_path: &str, dest_path: &Path) -> Result<()> {
        #[cfg(not(feature = "tempfile"))]
        return Err(Error::invalid_state("tempfile feature未启用，无法保存文件"));
        
        #[cfg(feature = "tempfile")]
        if let Some(temp_dir) = &self.temp_dir {
            let src_path = temp_dir.path().join("output").join(sandbox_path);
            
            if src_path.exists() {
                // 确保目标目录存在
                if let Some(parent) = dest_path.parent() {
                    tokio::fs::create_dir_all(parent).await?;
                }
                
                // 复制文件
                tokio::fs::copy(src_path, dest_path).await?;
                Ok(())
            } else {
                Err(Error::not_found(format!("沙箱中文件不存在: {}", sandbox_path)))
            }
        } else {
            Err(Error::invalid_state("临时目录未初始化"))
        }
    }
    
    async fn cancel(&self) -> Result<()> {
        debug!("【DockerSandbox】取消执行: {}", self.id);
        
        if let Some(container_id) = &self.container_id {
            // 停止容器
            let stop_result = Command::new("docker")
                .arg("stop")
                .arg(container_id)
                .output()
                .await;
                
            if let Err(e) = stop_result {
                warn!("停止容器失败: {}, 错误: {}", container_id, e);
            }
        }
        
        if let Some(tx) = &self.cancel_tx {
            // 设置取消标志为true
            tx.store(true, Ordering::Relaxed);
        }
        
        self.set_status(SandboxStatus::Cancelled)?;
        Ok(())
    }
    
    async fn get_resource_usage(&self) -> Result<ResourceUsage> {
        let monitor = self.resource_monitor.lock().unwrap();
        
        Ok(ResourceUsage {
            memory_bytes: monitor.peak_memory,
            // Docker 容器中 CPU 时间统计需要 cgroup v2 支持，当前返回 0
            // 实际部署时应通过 cgroup 文件系统读取 CPU 使用时间
            cpu_time_ms: 0,
            execution_time_ms: monitor.start_time.elapsed().as_millis() as u64,
            // 磁盘 I/O 统计需要挂载 cgroup 文件系统，当前暂不支持
            disk_io_bytes: 0,
            // 网络 I/O 统计需要网络命名空间监控，当前暂不支持
            network_io_bytes: 0,
        })
    }
    
    async fn validate_code(&self, code: &[u8]) -> Result<Vec<String>> {
        let mut warnings = Vec::new();
        
        // 检查代码大小
        if code.len() > self.config.resource_limits.max_code_size {
            warnings.push(format!(
                "代码大小超过限制: {} > {} 字节",
                code.len(),
                self.config.resource_limits.max_code_size
            ));
        }
        
        // 检查敏感操作
        let code_str = String::from_utf8_lossy(code);
        
        // 检查系统调用
        if code_str.contains("os.system") || code_str.contains("subprocess") {
            warnings.push("检测到系统命令执行".to_string());
        }
        
        // 检查文件操作
        if code_str.contains("open(") || code_str.contains("file(") {
            warnings.push("检测到文件操作".to_string());
        }
        
        // 检查网络操作
        if code_str.contains("socket.") || code_str.contains("urllib") || code_str.contains("requests.") {
            warnings.push("检测到网络操作".to_string());
        }
        
        Ok(warnings)
    }
    
    async fn set_env_var(&self, name: &str, value: &str) -> Result<()> {
        // Docker沙箱中设置环境变量需要在创建容器时完成
        // 这里暂不实现动态设置
        warn!("Docker沙箱不支持动态设置环境变量: {}={}", name, value);
        Ok(())
    }
    
    async fn get_status(&self) -> Result<SandboxStatus> {
        let status = self.status.read().map_err(|_| {
            Error::lock("无法获取沙箱状态锁")
        })?;
        
        Ok(*status)
    }

    async fn get_all_algorithm_definitions(&self) -> Result<std::collections::HashMap<String, crate::core::interfaces::AlgorithmDefinition>> {
        Ok(std::collections::HashMap::new())
    }

    async fn execute_algorithm(&self, _algorithm: &crate::core::interfaces::AlgorithmDefinition, _data: &crate::data::DataBatch) -> Result<crate::algorithm::types::ExecutionResult> {
        Ok(crate::algorithm::types::ExecutionResult::success(vec![]))
    }

    async fn prepare_input_data(&self, _data: &crate::data::DataBatch) -> Result<Vec<u8>> {
        Ok(Vec::new())
    }

    async fn process_sandbox_result(&self, _algorithm_id: &str, _sandbox_result: crate::algorithm::executor::sandbox::result::SandboxResult) -> Result<crate::algorithm::types::ExecutionResult> {
        Ok(crate::algorithm::types::ExecutionResult::success(vec![]))
    }
}

// 为了支持clone
impl Clone for DockerSandbox {
    fn clone(&self) -> Self {
        Self {
            id: self.id.clone(),
            container_id: self.container_id.clone(),
            config: self.config.clone(),
            #[cfg(feature = "tempfile")]
            temp_dir: None, // 不克隆临时目录
            status: RwLock::new(*self.status.read().unwrap()),
            cancel_tx: None, // 不克隆取消通道
            resource_monitor: self.resource_monitor.clone(),
            environment: self.environment.clone(),
            image_config: self.image_config.clone(),
        }
    }
} 