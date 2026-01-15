use std::sync::{Arc, Mutex};
use std::collections::HashMap;
use std::time::Duration;
use std::path::PathBuf;
use std::pin::Pin;
use std::future::Future;
use async_trait::async_trait;
use log::{debug, warn, error};

use crate::{Error, Result};
use crate::algorithm::types::Algorithm;
use crate::algorithm::executor::{ExecutorConfig, ExecutionStatus, ExecutionResult};
use crate::algorithm::executor::config::{SandboxConfig, SandboxType};
use crate::algorithm::executor::sandbox::interface::Sandbox;
use crate::algorithm::executor::sandbox::implementations::create_sandbox;
use crate::algorithm::executor::sandbox::environment::ExecutionEnvironment;
use crate::algorithm::executor::sandbox::result::SandboxResult;

type BoxFuture<'a, T> = Pin<Box<dyn Future<Output = T> + Send + 'a>>;

/// 沙箱统计信息
#[derive(Debug, Default, Clone)]
struct SandboxStats {
    /// 总任务数
    total_tasks: usize,
    /// 成功任务数
    successful_tasks: usize,
    /// 失败任务数
    failed_tasks: usize,
    /// 取消任务数
    cancelled_tasks: usize,
    /// 超时任务数
    timeout_tasks: usize,
    /// 总执行时间(毫秒)
    total_execution_time_ms: u64,
    /// 总内存使用(字节)
    total_memory_bytes: usize,
    /// 总CPU使用(毫秒)
    total_cpu_time_ms: u64,
    /// 最大任务并发数
    max_concurrent_tasks: usize,
}

/// 沙箱执行器
/// 实现了AlgorithmExecutor特性，用于在安全的沙箱环境中执行算法
#[derive(Debug)]
pub struct SandboxExecutor {
    /// 沙箱配置
    config: SandboxConfig,
    /// 运行中的任务
    running_tasks: Mutex<HashMap<String, Arc<tokio::sync::Mutex<ExecutionEnvironment>>>>,
    /// 已完成的任务
    completed_tasks: Mutex<HashMap<String, SandboxResult>>,
    /// 任务统计信息
    stats: Mutex<SandboxStats>,
    /// 缓存目录
    cache_dir: Option<PathBuf>,
    /// 沙箱工厂
    sandbox_factory: Box<dyn Fn(SandboxType, &ExecutorConfig) -> BoxFuture<'static, Result<Box<dyn Sandbox>>> + Send + Sync>,
}

impl SandboxExecutor {
    /// 创建新的沙箱执行器
    pub fn new(config: SandboxConfig) -> Self {
        // 设置缓存目录
        let cache_dir = std::env::temp_dir().join("sandbox_cache");
        if !cache_dir.exists() {
            let _ = std::fs::create_dir_all(&cache_dir);
        }
        
        // 创建沙箱工厂
        let sandbox_factory: Box<dyn Fn(_, _) -> BoxFuture<'static, _> + Send + Sync> = Box::new(
            |sandbox_type, config| {
                Box::pin(async move {
                    create_sandbox(sandbox_type, config).await
                })
            }
        );
        
        Self {
            config,
            running_tasks: Mutex::new(HashMap::new()),
            completed_tasks: Mutex::new(HashMap::new()),
            stats: Mutex::new(SandboxStats::default()),
            cache_dir: Some(cache_dir),
            sandbox_factory,
        }
    }
    
    /// 获取执行环境
    fn get_environment(&self, task_id: &str) -> Result<Arc<tokio::sync::Mutex<ExecutionEnvironment>>> {
        let tasks = self.running_tasks.lock().map_err(|_| {
            Error::lock("无法获取任务锁")
        })?;
        
        tasks.get(task_id)
            .cloned()
            .ok_or_else(|| Error::not_found(format!("任务不存在: {}", task_id)))
    }
    
    /// 创建新的执行环境
    fn create_environment(&self, task_id: &str) -> Result<Arc<tokio::sync::Mutex<ExecutionEnvironment>>> {
        let env = ExecutionEnvironment::new(self.config.clone(), task_id);
        let env_arc = Arc::new(tokio::sync::Mutex::new(env));
        
        let mut tasks = self.running_tasks.lock().map_err(|_| {
            Error::lock("无法获取任务锁")
        })?;
        tasks.insert(task_id.to_string(), env_arc.clone());
        
        // 更新统计信息
        let mut stats = self.stats.lock().map_err(|_| {
            Error::lock("无法获取统计锁")
        })?;
        stats.total_tasks += 1;
        stats.max_concurrent_tasks = stats.max_concurrent_tasks.max(tasks.len());
        
        Ok(env_arc)
    }
    
    /// 移除执行环境
    fn remove_environment(&self, task_id: &str) -> Result<()> {
        let mut tasks = self.running_tasks.lock().map_err(|_| {
            Error::lock("无法获取任务锁")
        })?;
        
        if tasks.remove(task_id).is_none() {
            return Err(Error::not_found(format!("任务不存在: {}", task_id)));
        }
        
        Ok(())
    }
    
    /// 保存执行结果
    fn save_result(&self, task_id: &str, result: SandboxResult) -> Result<()> {
        let mut completed_tasks = self.completed_tasks.lock().map_err(|_| {
            Error::lock("无法获取已完成任务锁")
        })?;
        
        completed_tasks.insert(task_id.to_string(), result.clone());
        
        // 更新统计信息
        let mut stats = self.stats.lock().map_err(|_| {
            Error::lock("无法获取统计锁")
        })?;
        
        if result.success {
            stats.successful_tasks += 1;
        } else if result.status == ExecutionStatus::Cancelled {
            stats.cancelled_tasks += 1;
        } else if result.error.as_ref().map(|e| e.contains("超时")).unwrap_or(false) {
            stats.timeout_tasks += 1;
        } else {
            stats.failed_tasks += 1;
        }
        
        stats.total_execution_time_ms += result.execution_time_ms;
        stats.total_memory_bytes += result.resource_usage.memory_usage_bytes;
        stats.total_cpu_time_ms += result.resource_usage.execution_time_ms;
        
        Ok(())
    }
    
    /// 获取执行结果
    fn get_result(&self, task_id: &str) -> Result<Option<SandboxResult>> {
        let completed_tasks = self.completed_tasks.lock().map_err(|_| {
            Error::lock("无法获取已完成任务锁")
        })?;
        
        Ok(completed_tasks.get(task_id).cloned())
    }
    
    /// 准备执行资源
    async fn prepare_execution(
        &self,
        algorithm: &Algorithm,
        parameters: &HashMap<String, serde_json::Value>,
        config: &ExecutorConfig,
    ) -> Result<Vec<u8>> {
        debug!("准备执行资源: 算法={}, 任务ID={}", algorithm.id, 
               config.task_id.as_ref().unwrap_or(&"unknown".to_string()));
        
        // 序列化算法和参数
        let mut payload = serde_json::to_vec(&algorithm)
            .map_err(|e| Error::serialization(format!("无法序列化算法: {}", e)))?;
        
        // 添加分隔符
        payload.extend_from_slice(b"\n---\n");
        
        // 添加参数
        let params_bytes = serde_json::to_vec(&parameters)
            .map_err(|e| Error::serialization(format!("无法序列化参数: {}", e)))?;
        payload.extend_from_slice(&params_bytes);
        
        // 计算资源需求（简化版）
        let algo_size = algorithm.code.len();
        let params_size = params_bytes.len();
        
        if algo_size + params_size > 100 * 1024 * 1024 {
            warn!("算法和参数大小超过100MB，可能影响性能: {}", 
                  config.task_id.as_ref().unwrap_or(&"unknown".to_string()));
        }
        
        // 创建执行环境中的临时目录结构
        let task_id = config.task_id.as_ref().unwrap_or(&"default".to_string()).clone();
        if let Some(env) = self.get_environment(&task_id).ok() {
            let env_guard = env.lock().await;
            #[cfg(feature = "tempfile")]
            if let Some(temp_dir) = &env_guard.temp_dir {
                // 创建输入和输出目录
                let input_dir = temp_dir.path().join("input");
                let output_dir = temp_dir.path().join("output");
                
                let _ = tokio::fs::create_dir_all(&input_dir).await;
                let _ = tokio::fs::create_dir_all(&output_dir).await;
                
                // 写入算法和参数文件
                let algo_file = input_dir.join("algorithm.json");
                let params_file = input_dir.join("parameters.json");
                
                if let Err(e) = tokio::fs::write(&algo_file, serde_json::to_string_pretty(algorithm).unwrap()).await {
                    warn!("写入算法文件失败: {}, 错误: {}", algo_file.display(), e);
                }
                
                if let Err(e) = tokio::fs::write(&params_file, serde_json::to_string_pretty(parameters).unwrap()).await {
                    warn!("写入参数文件失败: {}, 错误: {}", params_file.display(), e);
                }
            }
        }
        
        Ok(payload)
    }
    
    /// 处理执行结果
    fn process_result(
        &self,
        sandbox_result: SandboxResult,
        execution_time: Duration,
        task_id: &str,
    ) -> Result<ExecutionResult> {
        debug!(
            "处理执行结果: 任务={}, 成功={}, 执行时间={}ms",
            task_id, sandbox_result.success, sandbox_result.execution_time_ms
        );
        
        // 保存执行结果
        self.save_result(task_id, sandbox_result.clone())?;
        
        if !sandbox_result.success {
            return Err(Error::execution(
                sandbox_result.error.unwrap_or_else(|| "算法执行失败".to_string())
            ));
        }
        
        // 解析输出结果
        let output = match serde_json::from_str::<serde_json::Value>(&sandbox_result.stdout) {
            Ok(value) => value,
            Err(e) => {
                return Err(Error::deserialization(format!(
                    "无法解析算法输出: {}，错误: {}", 
                    sandbox_result.stdout, e
                )));
            }
        };
        
        // 记录详细日志
        debug!(
            "算法执行成功: 任务={}, 执行时间={}ms, 内存使用={}MB",
            task_id,
            sandbox_result.execution_time_ms,
            sandbox_result.resource_usage.memory_usage_bytes / (1024 * 1024)
        );
        
        let result = ExecutionResult {
            output,
            resource_usage: sandbox_result.resource_usage.clone(),
            execution_time_ms: execution_time.as_millis() as u64,
            status: ExecutionStatus::Completed,
            logs: Some(sandbox_result.stderr.clone()),
        };
        
        Ok(result)
    }
    
    /// 选择合适的沙箱类型
    fn select_sandbox_type(&self, algorithm: &Algorithm) -> SandboxType {
        // 根据算法代码类型选择合适的沙箱
        let code = &algorithm.code;
        
        if code.starts_with("(module") || code.contains("\x00\x61\x73\x6D") {
            // WebAssembly模块
            SandboxType::Wasm
        } else if code.starts_with("#!/usr/bin/env python") || 
                  code.starts_with("#!/usr/bin/python") ||
                  code.contains("import ") || 
                  code.contains("def ") {
            // Python脚本
            SandboxType::Process
        } else if code.starts_with("#!/bin/bash") || 
                  code.starts_with("#!/usr/bin/env bash") {
            // Bash脚本
            SandboxType::Process
        } else {
            // 默认使用WASM
            SandboxType::Wasm
        }
    }
    
    /// 清理过期结果
    pub fn cleanup_expired_results(&self, ttl: Duration) -> Result<usize> {
        let now = std::time::SystemTime::now();
        let mut completed_tasks = self.completed_tasks.lock().map_err(|_| {
            Error::lock("无法获取已完成任务锁")
        })?;
        
        // 找出过期结果
        let expired_tasks: Vec<String> = completed_tasks.iter()
            .filter(|(_, result)| {
                // 将chrono::DateTime转换为SystemTime
                let end_time = std::time::UNIX_EPOCH + std::time::Duration::from_secs(result.end_time.timestamp() as u64);
                if let Ok(age) = now.duration_since(end_time) {
                    age > ttl
                } else {
                    false
                }
            })
            .map(|(id, _)| id.clone())
            .collect();
        
        // 删除过期结果
        for task_id in &expired_tasks {
            completed_tasks.remove(task_id);
        }
        
        let count = expired_tasks.len();
        if count > 0 {
            debug!("已清理 {} 个过期任务结果", count);
        }
        
        Ok(count)
    }
    
    /// 获取统计信息
    pub fn get_stats(&self) -> Result<SandboxStats> {
        let stats = self.stats.lock().map_err(|_| {
            Error::lock("无法获取统计锁")
        })?;
        
        Ok(stats.clone())
    }
}

// 为SandboxExecutor定义AlgorithmExecutor特征
#[async_trait]
pub trait AlgorithmExecutor: Send + Sync {
    async fn execute(
        &self,
        algorithm: &Algorithm,
        parameters: &HashMap<String, serde_json::Value>,
        model_id: Option<&str>,
        config: &ExecutorConfig,
    ) -> Result<ExecutionResult>;
    
    async fn cancel(&self, task_id: &str) -> Result<()>;
    
    async fn get_status(&self, task_id: &str) -> Result<ExecutionStatus>;
}

// 使用正确的特征名称实现
#[async_trait]
impl AlgorithmExecutor for SandboxExecutor {
    async fn execute(
        &self,
        algorithm: &Algorithm,
        parameters: &HashMap<String, serde_json::Value>,
        model_id: Option<&str>,
        config: &ExecutorConfig,
    ) -> Result<ExecutionResult> {
        debug!(
            "执行算法: 算法={}, 任务ID={}, 模型ID={:?}",
            algorithm.id, config.task_id, model_id
        );
        
        let start_time = std::time::Instant::now();
        
        // 1. 创建执行环境
        let env_arc = self.create_environment(&config.task_id)?;
        
        // 2. 准备执行数据
        let payload = self.prepare_execution(algorithm, parameters, config).await?;
        
        // 3. 选择合适的沙箱类型
        let sandbox_type = self.select_sandbox_type(algorithm);
        
        // 4. 创建沙箱
        let sandbox_result = (self.sandbox_factory)(sandbox_type, config).await;
        let sandbox = match sandbox_result {
            Ok(sandbox) => sandbox,
            Err(e) => {
                self.remove_environment(&config.task_id)?;
                return Err(Error::internal(format!("创建沙箱失败: {}", e)));
            }
        };
        
        // 5. 准备沙箱环境
        if let Err(e) = sandbox.prepare().await {
            self.remove_environment(&config.task_id)?;
            return Err(Error::internal(format!("准备沙箱环境失败: {}", e)));
        }
        
        // 6. 向环境中添加模型ID信息
        let mut input_data = payload;
        if let Some(id) = model_id {
            // 添加模型ID
            input_data.extend_from_slice(b"\n---MODEL_ID---\n");
            input_data.extend_from_slice(id.as_bytes());
        }
        
        // 7. 设置执行超时
        let timeout = Duration::from_millis(config.timeout_ms);
        
        // 8. 执行算法代码
        let code = algorithm.code.as_bytes().to_vec();
        
        let execution_result = match sandbox.execute(&code, &input_data, timeout).await {
            Ok(result) => result,
            Err(e) => {
                error!("沙箱执行失败: {}", e);
                
                // 清理资源
                if let Err(cleanup_err) = sandbox.cleanup().await {
                    warn!("清理沙箱资源失败: {}", cleanup_err);
                }
                
                self.remove_environment(&config.task_id)?;
                return Err(e);
            }
        };
        
        // 9. 清理沙箱资源
        if let Err(e) = sandbox.cleanup().await {
            warn!("清理沙箱资源失败: {}", e);
        }
        
        // 10. 移除执行环境
        self.remove_environment(&config.task_id)?;
        
        // 11. 处理结果
        let execution_time = start_time.elapsed();
        self.process_result(execution_result, execution_time, &config.task_id)
    }
    
    async fn cancel(&self, task_id: &str) -> Result<()> {
        debug!("取消算法执行: 任务ID={}", task_id);
        
        // 获取执行环境
        let env_arc = match self.get_environment(task_id) {
            Ok(env) => env,
            Err(e) => {
                // 检查是否已完成
                if let Some(result) = self.get_result(task_id)? {
                    if result.status == ExecutionStatus::Completed {
                        return Err(Error::failed_precondition("任务已完成，无法取消"));
                    }
                }
                return Err(e);
            }
        };
        
        // 设置取消标志
        let env = env_arc.lock().await;
        env.cancel();
        
        // 创建取消结果
        let result = SandboxResult::cancelled(
            env.get_stdout(),
            env.get_stderr(),
            env.cpu_time_ms.load(std::sync::atomic::Ordering::SeqCst),
            env.resource_usage.clone(),
        );
        
        // 保存取消结果
        self.save_result(task_id, result)?;
        
        // 移除执行环境
        self.remove_environment(task_id)?;
        
        Ok(())
    }
    
    async fn get_status(&self, task_id: &str) -> Result<ExecutionStatus> {
        // 首先检查是否已完成
        if let Some(result) = self.get_result(task_id)? {
            return Ok(result.status);
        }
        
        // 其次检查是否正在运行
        if let Ok(env) = self.get_environment(task_id) {
            let env_guard = env.lock().await;
            
            if env_guard.is_cancelled() {
                return Ok(ExecutionStatus::Cancelled);
            }
            
            // 正在运行
            return Ok(ExecutionStatus::Running);
        }
        
        // 任务不存在
        Err(Error::not_found(format!("任务不存在: {}", task_id)))
    }
} 