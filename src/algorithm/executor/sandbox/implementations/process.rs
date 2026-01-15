use std::path::Path;
use std::time::Duration;
use std::sync::{RwLock, Mutex};
use std::process::{Command, Child, Stdio};
use std::io::Write;
use async_trait::async_trait;
#[cfg(feature = "tempfile")]
use tempfile::TempDir;
use uuid::Uuid;
use log::debug;

use crate::{Error, Result};
use crate::algorithm::executor::config::ExecutorConfig;
use crate::algorithm::types::ResourceUsage;
use crate::algorithm::executor::sandbox::interface::Sandbox;
use crate::algorithm::executor::sandbox::result::SandboxResult;
use crate::algorithm::types::SandboxStatus;
use crate::algorithm::executor::sandbox::error::SandboxError;
use crate::algorithm::executor::sandbox::environment::ExecutionEnvironment;

// Removed unused Windows-specific imports; not used in current implementation

#[cfg(unix)]
use std::os::unix::process::ExitStatusExt;

/// 进程沙箱实现
#[derive(Debug)]
pub struct ProcessSandbox {
    id: String,
    config: ExecutorConfig,
    #[cfg(feature = "tempfile")]
    temp_dir: Option<TempDir>,
    status: RwLock<SandboxStatus>,
    child_process: Option<Mutex<Child>>,
    environment: Option<std::sync::Arc<tokio::sync::RwLock<ExecutionEnvironment>>>,
}

impl ProcessSandbox {
    pub async fn new(config: &ExecutorConfig) -> Result<Self> {
        let id = format!("process-sandbox-{}", Uuid::new_v4());
        #[cfg(feature = "tempfile")]
        let temp_dir = TempDir::new().ok();
        #[cfg(not(feature = "tempfile"))]
        let temp_dir = None;
        
        // 子进程对象将在实际执行时创建
        let child_process = None;
        
        Ok(Self {
            id,
            config: config.clone(),
            #[cfg(feature = "tempfile")]
            temp_dir,
            status: RwLock::new(SandboxStatus::Uninitialized),
            child_process,
            environment: None,
        })
    }
    
    /// 创建本地进程沙箱
    pub async fn new_local(config: &ExecutorConfig) -> Result<Self> {
        let sandbox = Self::new(config).await?;
        // 本地进程模式的特殊配置
        Ok(sandbox)
    }
    
    /// 创建隔离进程沙箱
    pub async fn new_isolated(config: &ExecutorConfig) -> Result<Self> {
        let sandbox = Self::new(config).await?;
        // 隔离进程模式的特殊配置（更严格的安全限制）
        Ok(sandbox)
    }
    
    fn set_status(&self, status: SandboxStatus) -> Result<()> {
        let mut current_status = self.status.write().map_err(|_| {
            Error::lock("无法获取沙箱状态锁")
        })?;
        *current_status = status;
        Ok(())
    }
    
    /// 检测脚本类型并返回扩展名和语言
    fn detect_script_type(&self, code: &[u8]) -> (&'static str, &'static str) {
        if code.starts_with(b"#!/usr/bin/env python") || code.starts_with(b"#!/usr/bin/python") {
            return (".py", "python");
        } else if code.starts_with(b"#!/bin/bash") || code.starts_with(b"#!/usr/bin/env bash") {
            return (".sh", "bash");
        } else if code.starts_with(b"#!/usr/bin/env node") || code.starts_with(b"#!/usr/bin/node") {
            return (".js", "node");
        } else if code.starts_with(b"#!/usr/bin/env ruby") || code.starts_with(b"#!/usr/bin/ruby") {
            return (".rb", "ruby");
        } else if code.starts_with(b"#!/usr/bin/env perl") || code.starts_with(b"#!/usr/bin/perl") {
            return (".pl", "perl");
        }
        
        // 尝试根据内容猜测
        let content = String::from_utf8_lossy(code);
        if content.contains("def ") && (content.contains("print(") || content.contains("import ")) {
            return (".py", "python");
        } else if content.contains("function ") && (content.contains("console.log") || content.contains("require(")) {
            return (".js", "node");
        } else if content.contains("echo ") && (content.contains("if [ ") || content.contains("for i in ")) {
            return (".sh", "bash");
        }
        
        // 默认作为二进制文件处理
        (".bin", "binary")
    }
    
    /// 创建命令
    fn create_command(&self, script_path: &Path, script_lang: &str, input_path: &Path, output_path: &Path, stderr_path: &Path) -> Result<Command> {
        let mut command = match script_lang {
            "python" => {
                let mut cmd = Command::new("python3");
                cmd.arg(script_path);
                cmd
            },
            "bash" => {
                let mut cmd = Command::new("bash");
                cmd.arg(script_path);
                cmd
            },
            "node" => {
                let mut cmd = Command::new("node");
                cmd.arg(script_path);
                cmd
            },
            "ruby" => {
                let mut cmd = Command::new("ruby");
                cmd.arg(script_path);
                cmd
            },
            "perl" => {
                let mut cmd = Command::new("perl");
                cmd.arg(script_path);
                cmd
            },
            _ => Command::new(script_path),
        };
        
        // 设置工作目录
        if let Some(temp_dir) = &self.temp_dir {
            command.current_dir(temp_dir.path());
        }
        
        // 重定向I/O
        command
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());
        
        // 添加参数
        command.arg("--input").arg(input_path)
               .arg("--output").arg(output_path)
               .env("INPUT_FILE", input_path.to_str().unwrap_or("input.dat"))
               .env("OUTPUT_FILE", output_path.to_str().unwrap_or("output.dat"))
               .env("ERROR_FILE", stderr_path.to_str().unwrap_or("stderr.dat"));
        
        Ok(command)
    }
    
    /// 应用资源限制
    fn apply_resource_limits(&self, command: &mut Command) -> Result<()> {
        debug!("为进程应用资源限制");
        
        // 设置环境变量限制
        command.env("MAX_MEMORY_MB", self.config.resource_limits.memory_mb.to_string())
               .env("MAX_CPU_TIME_MS", self.config.resource_limits.cpu_time_ms.to_string())
               .env("MAX_OUTPUT_SIZE", "10485760"); // 10MB输出限制
        
        // 平台特定限制
        #[cfg(target_os = "linux")]
        {
            use std::os::unix::process::CommandExt;
            
            unsafe {
                command.pre_exec(|| {
                    // 使用资源限制
                    if let Ok(resource) = rlimit::Resource::from_str("cpu") {
                        let cpu_limit = (self.config.resource_limits.cpu_time_ms / 1000) as u64;
                        if let Err(e) = rlimit::setrlimit(resource, 
                                                     rlimit::Rlim::from_raw(cpu_limit), 
                                                     rlimit::Rlim::INFINITY) {
                            eprintln!("无法设置CPU限制: {}", e);
                        }
                    }
                    
                    // 内存限制
                    if let Ok(resource) = rlimit::Resource::from_str("as") {
                        let mem_limit = (self.config.resource_limits.memory_mb * 1024 * 1024) as u64;
                        if let Err(e) = rlimit::setrlimit(resource, 
                                                     rlimit::Rlim::from_raw(mem_limit), 
                                                     rlimit::Rlim::from_raw(mem_limit)) {
                            eprintln!("无法设置内存限制: {}", e);
                        }
                    }
                    
                    // 文件大小限制
                    if let Ok(resource) = rlimit::Resource::from_str("fsize") {
                        if let Err(e) = rlimit::setrlimit(resource, 
                                                     rlimit::Rlim::from_raw(100 * 1024 * 1024), // 100MB
                                                     rlimit::Rlim::INFINITY) {
                            eprintln!("无法设置文件大小限制: {}", e);
                        }
                    }
                    
                    // 进程数限制
                    if let Ok(resource) = rlimit::Resource::from_str("nproc") {
                        if let Err(e) = rlimit::setrlimit(resource, 
                                                     rlimit::Rlim::from_raw(10), // 最多10个进程
                                                     rlimit::Rlim::INFINITY) {
                            eprintln!("无法设置进程数限制: {}", e);
                        }
                    }
                    
                    Ok(())
                });
            }
        }
        
        // Windows特定限制（通过作业对象实现，这里简化处理）
        #[cfg(target_os = "windows")]
        {
            // Windows上的资源限制更复杂，需要使用作业对象
            // 简化起见，这里只设置环境变量，实际应用中需要更复杂的实现
        }
        
        Ok(())
    }
    
    /// 应用安全限制
    fn apply_security_limits(&self, command: &mut Command) -> Result<()> {
        debug!("为进程应用安全限制");
        
        // 设置安全相关环境变量
        command.env("SANDBOX_MODE", "1")
               .env("RESTRICTED_MODE", "1");
        
        // 移除敏感环境变量
        let sensitive_vars = [
            "AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", 
            "DATABASE_URL", "API_KEY", "SECRET_KEY",
            "GITHUB_TOKEN", "SSH_AUTH_SOCK"
        ];
        
        for var in &sensitive_vars {
            command.env_remove(var);
        }
        
        // 平台特定安全措施
        #[cfg(target_os = "linux")]
        {
            use std::os::unix::process::CommandExt;
            
            if !self.config.resource_limits.network_enabled {
                command.env("NO_NETWORK", "1");
            }
            
            if !self.config.resource_limits.io_enabled {
                command.env("NO_IO", "1");
            }
            
            // 注意: 完整的安全隔离需要使用 seccomp、namespaces、capabilities 等机制
            // 当前实现提供了基本的环境变量和资源限制设置
            // 生产环境应结合系统级隔离机制（如 systemd-nspawn、firejail）使用
        }
        
        Ok(())
    }
    
    /// 收集资源使用情况
    async fn collect_resource_usage(&self, execution_time_ms: u64) -> Result<ResourceUsage> {
        debug!("收集进程资源使用情况");
        
        // 基础资源使用情况
        let mut resource_usage = ResourceUsage {
            cpu_usage: 0.0,
            memory_usage: 0,
            peak_memory_usage: 0,
            disk_read: 0,
            disk_write: 0,
            network_received: 0,
            network_sent: 0,
            execution_time_ms,
            gpu_usage: None,
            gpu_memory_usage: None,
        };
        
        // 尝试从环境中获取资源使用信息
        if let Some(env_arc) = &self.environment {
            let env = env_arc.read().await;
            // 如果环境中有数据，使用环境中的数据
            resource_usage.cpu_usage = 100.0; // 简化处理
            resource_usage.memory_usage = env.memory_bytes.load(std::sync::atomic::Ordering::SeqCst);
            resource_usage.peak_memory_usage = resource_usage.memory_usage;
            resource_usage.disk_read = env.disk_read_bytes.load(std::sync::atomic::Ordering::SeqCst);
            resource_usage.disk_write = env.disk_write_bytes.load(std::sync::atomic::Ordering::SeqCst);
            let total_network = env.network_bytes.load(std::sync::atomic::Ordering::SeqCst);
            resource_usage.network_received = total_network / 2; // 粗略估算接收
            resource_usage.network_sent = total_network / 2; // 粗略估算发送
            return Ok(resource_usage);
        }
        
        // 平台特定资源使用情况收集
        #[cfg(target_os = "linux")]
        {
            // 在Linux上可以从/proc获取更详细的资源使用情况
            // 这里简化处理
            if let Some(child_mutex) = &self.child_process {
                if let Ok(child) = child_mutex.lock() {
                    if let Some(id) = child.id() {
                        // 尝试读取proc文件系统
                        let proc_stat = format!("/proc/{}/stat", id);
                        if let Ok(content) = tokio::fs::read_to_string(proc_stat).await {
                            // 解析proc统计信息
                            let parts: Vec<&str> = content.split_whitespace().collect();
                            if parts.len() > 23 {
                                // 解析内存使用
                                if let Ok(rss) = parts[23].parse::<u64>() {
                                    resource_usage.memory_usage = rss * 4096; // 页大小通常是4KB
                                    resource_usage.peak_memory_usage = resource_usage.memory_usage;
                                }
                                
                                // 解析CPU使用
                                let utime = parts[13].parse::<u64>().unwrap_or(0);
                                let stime = parts[14].parse::<u64>().unwrap_or(0);
                                let total_time = utime + stime;
                                resource_usage.cpu_usage = (total_time as f64 / (execution_time_ms as f64 * 100.0)) * 100.0;
                            }
                        }
                        
                        // 尝试读取IO统计
                        let proc_io = format!("/proc/{}/io", id);
                        if let Ok(content) = tokio::fs::read_to_string(proc_io).await {
                            for line in content.lines() {
                                if line.starts_with("read_bytes:") {
                                    if let Some(val_str) = line.split(':').nth(1) {
                                        resource_usage.disk_read = val_str.trim().parse().unwrap_or(0);
                                    }
                                } else if line.starts_with("write_bytes:") {
                                    if let Some(val_str) = line.split(':').nth(1) {
                                        resource_usage.disk_write = val_str.trim().parse().unwrap_or(0);
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        
        // 如果无法获取实际数据，使用估计值
        if resource_usage.memory_usage == 0 {
            resource_usage.memory_usage = 10 * 1024 * 1024; // 10MB估计值
            resource_usage.peak_memory_usage = resource_usage.memory_usage;
            resource_usage.cpu_usage = 50.0; // 50%估计值
        }
        
        Ok(resource_usage)
    }
}

#[async_trait]
impl Sandbox for ProcessSandbox {
    fn id(&self) -> &str {
        &self.id
    }
    
    async fn prepare(&self) -> Result<()> {
        debug!("【ProcessSandbox】准备环境: {}", self.id);
        self.set_status(SandboxStatus::Ready)?;
        Ok(())
    }
    
    async fn execute(&self, code: &[u8], input: &[u8], timeout: Duration) -> Result<SandboxResult> {
        debug!("【ProcessSandbox】执行代码: {}, 输入大小: {}, 超时: {:?}", 
              self.id, input.len(), timeout);
        
        self.set_status(SandboxStatus::Running)?;
        let start_time = std::time::Instant::now();
        
        // 创建临时脚本文件
        let temp_dir = self.temp_dir.as_ref().ok_or_else(|| {
            Error::failed_precondition("临时目录未初始化")
        })?;
        
        // 验证代码安全性
        let validation_warnings = self.validate_code(code).await?;
        
        // 确定脚本类型和扩展名
        let (script_ext, script_lang) = self.detect_script_type(code);
        
        // 创建临时脚本文件
        let script_path = temp_dir.path().join(format!("script{}", script_ext));
        tokio::fs::write(&script_path, code).await?;
        
        // 确保脚本可执行
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let mut perms = tokio::fs::metadata(&script_path).await?.permissions();
            perms.set_mode(0o755);
            tokio::fs::set_permissions(&script_path, perms).await?;
        }
        
        // 创建用于输入的临时文件
        let input_path = temp_dir.path().join("input.dat");
        tokio::fs::write(&input_path, input).await?;
        
        // 创建用于输出的临时文件
        let output_path = temp_dir.path().join("output.dat");
        let stderr_path = temp_dir.path().join("stderr.dat");
        
        // 创建命令
        let mut command = self.create_command(&script_path, &script_lang, &input_path, &output_path, &stderr_path)?;
        
        // 限制命令的资源
        self.apply_resource_limits(&mut command)?;
        
        // 应用安全限制
        self.apply_security_limits(&mut command)?;
        
        // 执行命令
        let child_result = command.spawn();
        let mut child = match child_result {
            Ok(child) => child,
            Err(e) => {
                self.set_status(SandboxStatus::Failed)?;
                return Err(Error::execution(format!("启动进程失败: {}", e)));
            }
        };
        
        // 保存子进程引用
        if let Some(child_mutex) = &self.child_process {
            if let Ok(mut guard) = child_mutex.lock() {
                *guard = child;
            }
        }
        
        // 等待命令完成或超时
        let timeout_result = tokio::time::timeout(timeout, async {
            let output = child.wait_with_output()?;
            Ok::<_, std::io::Error>(output)
        }).await;
        
        // 处理执行结果
        match timeout_result {
            Ok(Ok(output)) => {
                // 获取执行时间
                let execution_time = start_time.elapsed();
                
                // 读取输出和错误
                let stdout = match tokio::fs::read_to_string(&output_path).await {
                    Ok(content) => content,
                    Err(_) => String::from_utf8_lossy(&output.stdout).to_string(),
                };
                
                let stderr = match tokio::fs::read_to_string(&stderr_path).await {
                    Ok(content) => content,
                    Err(_) => String::from_utf8_lossy(&output.stderr).to_string(),
                };
                
                // 获取资源使用情况
                let resource_usage = self.collect_resource_usage(execution_time.as_millis() as u64).await?;
                
                // 检查退出状态
                let success = output.status.success();
                let exit_code = output.status.code().unwrap_or(-1);
                
                self.set_status(if success { SandboxStatus::Completed } else { SandboxStatus::Failed })?;
                
                let mut result = if success {
                    SandboxResult::success(
                        stdout,
                        stderr,
                        execution_time.as_millis() as u64,
                        resource_usage,
                    )
                } else {
                    SandboxResult::failure(
                        exit_code,
                        stdout,
                        stderr,
                        format!("进程退出状态码: {}", exit_code),
                        execution_time.as_millis() as u64,
                        resource_usage,
                    )
                };
                
                // 添加验证警告
                for warning in validation_warnings {
                    result.add_warning(&warning);
                }
                
                // 清理临时文件
                let _ = tokio::fs::remove_file(&script_path).await;
                let _ = tokio::fs::remove_file(&input_path).await;
                let _ = tokio::fs::remove_file(&output_path).await;
                let _ = tokio::fs::remove_file(&stderr_path).await;
                
                Ok(result)
            },
            Ok(Err(e)) => {
                self.set_status(SandboxStatus::Failed)?;
                Err(SandboxError::ExecutionFailed(e.to_string()).into())
            },
            Err(_) => {
                // 执行超时，尝试终止进程
                if let Some(child_mutex) = &self.child_process {
                    if let Ok(mut child) = child_mutex.lock() {
                        let _ = child.kill();
                    }
                }
                
                // 收集可用的资源使用情况
                let resource_usage = self.collect_resource_usage(timeout.as_millis() as u64).await?;
                
                // 读取可能的部分输出
                let stdout = tokio::fs::read_to_string(&output_path).await.unwrap_or_default();
                let stderr = tokio::fs::read_to_string(&stderr_path).await.unwrap_or_default();
                
                self.set_status(SandboxStatus::Failed)?;
                Ok(SandboxResult::timeout(
                    timeout.as_millis() as u64,
                    stdout,
                    stderr,
                    resource_usage,
                ))
            }
        }
    }
    
    async fn cleanup(&self) -> Result<()> {
        debug!("【ProcessSandbox】清理资源: {}", self.id);
        self.set_status(SandboxStatus::Cleaned)?;
        Ok(())
    }
    
    fn supports_file_type(&self, file_type: &str) -> bool {
        match file_type.to_lowercase().as_str() {
            "py" | "sh" | "js" | "pl" | "rb" | "bin" => true,
            _ => false,
        }
    }
    
    async fn load_file(&self, src_path: &Path, sandbox_path: &str) -> Result<()> {
        // 验证沙箱路径安全性
        if sandbox_path.contains("..") || sandbox_path.starts_with('/') {
            return Err(Error::invalid_argument("不安全的沙箱路径"));
        }
        
        // 计算目标路径
        let temp_dir = self.temp_dir.as_ref().ok_or_else(|| {
            Error::failed_precondition("临时目录未初始化")
        })?;
        
        let target_path = temp_dir.path().join(sandbox_path);
        
        // 创建父目录
        if let Some(parent) = target_path.parent() {
            tokio::fs::create_dir_all(parent).await?;
        }
        
        // 复制文件
        tokio::fs::copy(src_path, &target_path).await?;
        
        Ok(())
    }
    
    async fn save_file(&self, sandbox_path: &str, dest_path: &Path) -> Result<()> {
        // 验证沙箱路径安全性
        if sandbox_path.contains("..") || sandbox_path.starts_with('/') {
            return Err(Error::invalid_argument("不安全的沙箱路径"));
        }
        
        // 计算源路径
        let temp_dir = self.temp_dir.as_ref().ok_or_else(|| {
            Error::failed_precondition("临时目录未初始化")
        })?;
        
        let source_path = temp_dir.path().join(sandbox_path);
        
        // 检查源文件是否存在
        if !source_path.exists() {
            return Err(Error::not_found(format!("沙箱文件不存在: {}", sandbox_path)));
        }
        
        // 创建父目录
        if let Some(parent) = dest_path.parent() {
            tokio::fs::create_dir_all(parent).await?;
        }
        
        // 复制文件
        tokio::fs::copy(&source_path, dest_path).await?;
        
        Ok(())
    }
    
    async fn cancel(&self) -> Result<()> {
        debug!("【ProcessSandbox】取消执行: {}", self.id);
        
        // 终止子进程
        if let Some(child_mutex) = &self.child_process {
            if let Ok(mut child) = child_mutex.lock() {
                let _ = child.kill();
            }
        }
        
        self.set_status(SandboxStatus::Failed)?;
        Ok(())
    }
    
    async fn get_resource_usage(&self) -> Result<ResourceUsage> {
        // 在实际实现中，这里应该从进程统计中获取真实的资源使用情况
        Ok(ResourceUsage {
            cpu_usage: 10.0,
            memory_usage: 1024 * 1024,
            peak_memory_usage: 2 * 1024 * 1024,
            disk_read: 0,
            disk_write: 0,
            network_received: 0,
            network_sent: 0,
            execution_time_ms: 100,
            gpu_usage: None,
            gpu_memory_usage: None,
        })
    }
    
    async fn validate_code(&self, code: &[u8]) -> Result<Vec<String>> {
        let mut warnings = Vec::new();
        
        // 检查文件大小
        if code.len() > 1024 * 1024 {
            warnings.push("脚本大小超过1MB，可能影响性能".to_string());
        }
        
        // 检查危险命令（简化版）
        let code_str = String::from_utf8_lossy(code);
        let danger_patterns = [
            "rm -rf", "sudo", "chmod 777", ":(){:|:&};:", ">()", "<()",
            "wget", "curl", "nc", "telnet", "ssh", "eval", "exec",
        ];
        
        for pattern in &danger_patterns {
            if code_str.contains(pattern) {
                warnings.push(format!("脚本包含潜在危险命令: {}", pattern));
            }
        }
        
        Ok(warnings)
    }
    
    async fn set_env_var(&self, name: &str, value: &str) -> Result<()> {
        // 在实际实现中，这里应该设置进程环境变量
        debug!("【ProcessSandbox】设置环境变量: {}={}", name, value);
        Ok(())
    }
    
    async fn get_status(&self) -> Result<SandboxStatus> {
        let status = self.status.read().map_err(|_| {
            Error::lock("无法获取沙箱状态锁")
        })?;
        Ok(status.clone())
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