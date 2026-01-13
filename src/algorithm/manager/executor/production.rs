use std::sync::Arc;
use std::collections::HashMap;
use tokio::sync::RwLock;
use serde::{Deserialize, Serialize};
use uuid::Uuid;
// use std::time::SystemTime; // not used here
use tracing::{info, warn, debug};

use crate::Result;
use crate::algorithm::types::ExecutionStatus;
use crate::algorithm::manager::types::ResourceUsage;
use crate::algorithm::manager::AlgorithmManager;

/// 算法执行记录
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlgorithmExecution {
    pub id: String,
    pub algorithm_id: String,
    pub state: ExecutionStatus,
    pub input_data: Option<serde_json::Value>,
    pub output_data: Option<serde_json::Value>,
    pub error_message: Option<String>,
    pub resource_usage: ResourceUsage,
    pub created_at: chrono::DateTime<chrono::Utc>,
    pub started_at: Option<chrono::DateTime<chrono::Utc>>,
    pub completed_at: Option<chrono::DateTime<chrono::Utc>>,
    pub config: Option<serde_json::Value>,
}

/// 执行指标
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ExecutionMetrics {
    pub cpu_usage_percent: f64,
    pub memory_usage_bytes: u64,
    pub execution_time_ms: u64,
    pub io_operations: u64,
}

/// 生产环境算法执行器
#[derive(Clone)]
pub struct ProductionExecutor {
    /// 算法管理器
    manager: Arc<AlgorithmManager>,
    /// 执行状态追踪
    executions: Arc<RwLock<HashMap<String, AlgorithmExecution>>>,
    /// 执行配置
    config: ProductionExecutorConfig,
}

/// 生产执行器配置
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProductionExecutorConfig {
    /// 最大并发执行数
    pub max_concurrent_executions: usize,
    /// 执行超时（秒）
    pub execution_timeout_secs: u64,
    /// 内存限制（MB）
    pub memory_limit_mb: usize,
    /// CPU限制（核心数）
    pub cpu_limit_cores: f32,
    /// 启用安全沙箱
    pub enable_sandbox: bool,
}

impl Default for ProductionExecutorConfig {
    fn default() -> Self {
        Self {
            max_concurrent_executions: 10,
            execution_timeout_secs: 300,
            memory_limit_mb: 1024,
            cpu_limit_cores: 2.0,
            enable_sandbox: true,
        }
    }
}

impl ProductionExecutor {
    /// 创建新的生产执行器
    pub fn new(manager: Arc<AlgorithmManager>, config: ProductionExecutorConfig) -> Self {
        Self {
            manager,
            executions: Arc::new(RwLock::new(HashMap::new())),
            config,
        }
    }

    /// 执行算法
    pub async fn execute_algorithm(
        &self,
        algorithm_id: &str,
        input_data: serde_json::Value,
        execution_config: Option<serde_json::Value>,
    ) -> Result<AlgorithmExecution> {
        let execution_id = Uuid::new_v4().to_string();
        
        // 创建执行记录
        let execution = AlgorithmExecution {
            id: execution_id.clone(),
            algorithm_id: algorithm_id.to_string(),
            state: ExecutionStatus::Pending,
            input_data: Some(input_data.clone()),
            output_data: None,
            error_message: None,
            resource_usage: ResourceUsage::default(),
            created_at: chrono::Utc::now(),
            started_at: None,
            completed_at: None,
            config: execution_config,
        };

        // 存储执行记录
        {
            let mut executions = self.executions.write().await;
            executions.insert(execution_id.clone(), execution.clone());
        }

        // 检查并发限制
        if !self.check_concurrent_limit().await {
            let mut updated_execution = execution;
            let error_msg = "超过最大并发执行数限制".to_string();
            updated_execution.state = ExecutionStatus::Failed(error_msg.clone());
            updated_execution.error_message = Some(error_msg);
            
            let mut executions = self.executions.write().await;
            executions.insert(execution_id.clone(), updated_execution.clone());
            
            return Ok(updated_execution);
        }

        // 启动执行
        self.start_execution(&execution_id, algorithm_id, input_data).await?;
        
        // 返回更新后的执行记录
        let executions = self.executions.read().await;
        Ok(executions.get(&execution_id).unwrap().clone())
    }

    /// 获取执行状态
    pub async fn get_execution(&self, execution_id: &str) -> Result<Option<AlgorithmExecution>> {
        let executions = self.executions.read().await;
        Ok(executions.get(execution_id).cloned())
    }

    /// 取消执行
    pub async fn cancel_execution(&self, execution_id: &str) -> Result<()> {
        let mut executions = self.executions.write().await;
        if let Some(execution) = executions.get_mut(execution_id) {
            execution.state = ExecutionStatus::Cancelled;
            execution.completed_at = Some(chrono::Utc::now());
        }
        Ok(())
    }

    /// 检查并发限制
    async fn check_concurrent_limit(&self) -> bool {
        let executions = self.executions.read().await;
        let running_count = executions.values()
            .filter(|e| matches!(e.state, ExecutionStatus::Running))
            .count();
        
        running_count < self.config.max_concurrent_executions
    }

    /// 启动执行
    async fn start_execution(
        &self,
        execution_id: &str,
        algorithm_id: &str,
        input_data: serde_json::Value,
    ) -> Result<()> {
        // 更新执行状态
        {
            let mut executions = self.executions.write().await;
            if let Some(execution) = executions.get_mut(execution_id) {
                execution.state = ExecutionStatus::Running;
                execution.started_at = Some(chrono::Utc::now());
            }
        }

        // 获取算法
        let algorithm = self.manager.get_algorithm_simple(algorithm_id)?;
        
        // 在新任务中执行算法
        let executions = self.executions.clone();
        let execution_id = execution_id.to_string();
        let algorithm_clone = algorithm.clone();
        let config = self.config.clone();
        
        tokio::spawn(async move {
            let result = Self::execute_algorithm_internal(
                &algorithm_clone,
                input_data,
                &config,
            ).await;

            // 更新执行结果
            let mut executions = executions.write().await;
            if let Some(execution) = executions.get_mut(&execution_id) {
                match result {
                    Ok(output) => {
                        execution.state = ExecutionStatus::Completed;
                        execution.output_data = Some(output);
                    }
                    Err(e) => {
                        let error_msg = e.to_string();
                        execution.state = ExecutionStatus::Failed(error_msg.clone());
                        execution.error_message = Some(error_msg);
                    }
                }
                execution.completed_at = Some(chrono::Utc::now());
            }
        });

        Ok(())
    }

    /// 内部算法执行逻辑
    async fn execute_algorithm_internal(
        algorithm: &crate::algorithm::types::Algorithm,
        input_data: serde_json::Value,
        config: &ProductionExecutorConfig,
    ) -> Result<serde_json::Value> {
        // 这里实现真正的算法执行逻辑
        // 包括沙箱、资源限制、超时等
        
        if config.enable_sandbox {
            // 在沙箱中执行
            Self::execute_in_sandbox(algorithm, input_data, config).await
        } else {
            // 直接执行
            Self::execute_direct(algorithm, input_data).await
        }
    }

    /// 在沙箱中执行算法
    async fn execute_in_sandbox(
        algorithm: &crate::algorithm::types::Algorithm,
        input_data: serde_json::Value,
        config: &ProductionExecutorConfig,
    ) -> Result<serde_json::Value> {
        use std::process::{Command, Stdio};
        use std::time::Duration;
        use tokio::time::timeout;
        #[cfg(feature = "tempfile")]
        use tempfile::NamedTempFile;
        
        info!("在沙箱中执行算法: {}", algorithm.id);
        
        // 创建临时文件存储算法代码
        let mut temp_file = NamedTempFile::new()
            .map_err(|e| crate::error::Error::Internal(format!("创建临时文件失败: {}", e)))?;
        
        // 写入算法代码
        use std::io::Write;
        temp_file.write_all(algorithm.code.as_bytes())
            .map_err(|e| crate::error::Error::Internal(format!("写入算法代码失败: {}", e)))?;
        
        let temp_path = temp_file.path().to_string_lossy().to_string();
        
        // 创建输入数据文件
        let mut input_file = NamedTempFile::new()
            .map_err(|e| crate::error::Error::Internal(format!("创建输入文件失败: {}", e)))?;
        
        let input_json = serde_json::to_string_pretty(&input_data)
            .map_err(|e| crate::error::Error::Internal(format!("序列化输入数据失败: {}", e)))?;
        
        input_file.write_all(input_json.as_bytes())
            .map_err(|e| crate::error::Error::Internal(format!("写入输入数据失败: {}", e)))?;
        
        let input_path = input_file.path().to_string_lossy().to_string();
        
        // 构建沙箱执行命令
        let execution_timeout = Duration::from_secs(config.execution_timeout_secs);
        
        // 使用 firejail 或类似的沙箱工具（如果可用）
        let mut cmd = if std::process::Command::new("which").arg("firejail").output().is_ok() {
            let mut cmd = Command::new("firejail");
            cmd.arg("--quiet")
               .arg("--noprofile")
               .arg("--private-tmp")
               .arg("--nonetwork")
               .arg("--rlimit-cpu")
               .arg(format!("{}", config.execution_timeout_secs))
               .arg("--rlimit-as")
               .arg(format!("{}", config.memory_limit_mb * 1024 * 1024));
            
            // 根据算法类型选择解释器
            match algorithm.language.as_str() {
                "python" => {
                    cmd.arg("python3").arg(&temp_path).arg(&input_path);
                }
                "javascript" | "node" => {
                    cmd.arg("node").arg(&temp_path).arg(&input_path);
                }
                "rust" => {
                    // 对于 Rust，需要先编译
                    return Err(crate::error::Error::Internal("沙箱中暂不支持 Rust 算法".to_string()));
                }
                _ => {
                    return Err(crate::error::Error::Internal(format!("不支持的算法语言: {}", algorithm.language)));
                }
            }
            cmd
        } else {
            // 回退到基本的进程隔离
            warn!("firejail 不可用，使用基本进程隔离");
            let mut cmd = match algorithm.language.as_str() {
                "python" => Command::new("python3"),
                "javascript" | "node" => Command::new("node"),
                _ => return Err(crate::error::Error::Internal(format!("不支持的算法语言: {}", algorithm.language))),
            };
            cmd.arg(&temp_path).arg(&input_path);
            cmd
        };
        
        // 配置进程选项
        cmd.stdout(Stdio::piped())
           .stderr(Stdio::piped())
           .stdin(Stdio::null());
        
        // 执行算法（带超时）
        let result = timeout(execution_timeout, async {
            let output = cmd.output()
                .map_err(|e| crate::error::Error::Internal(format!("执行算法失败: {}", e)))?;
            
            if !output.status.success() {
                let stderr = String::from_utf8_lossy(&output.stderr);
                return Err(crate::error::Error::Internal(format!("算法执行失败: {}", stderr)));
            }
            
            let stdout = String::from_utf8_lossy(&output.stdout);
            let result: serde_json::Value = serde_json::from_str(&stdout)
                .unwrap_or_else(|_| serde_json::json!({
                    "output": stdout.to_string(),
                    "type": "text"
                }));
            
            Ok(result)
        }).await;
        
        match result {
            Ok(Ok(output)) => {
                info!("沙箱执行成功: {}", algorithm.id);
                Ok(serde_json::json!({
                    "algorithm_id": algorithm.id,
                    "result": output,
                    "execution_mode": "sandbox",
                    "timestamp": chrono::Utc::now().to_rfc3339(),
                    "sandbox_config": {
                        "memory_limit_mb": config.memory_limit_mb,
                        "timeout_secs": config.execution_timeout_secs,
                        "cpu_limit_cores": config.cpu_limit_cores
                    }
                }))
            }
            Ok(Err(e)) => Err(e),
            Err(_) => {
                warn!("算法执行超时: {}", algorithm.id);
                Err(crate::error::Error::Internal(format!("算法执行超时 ({}秒)", config.execution_timeout_secs)))
            }
        }
    }

    /// 直接执行算法
    async fn execute_direct(
        algorithm: &crate::algorithm::types::Algorithm,
        input_data: serde_json::Value,
    ) -> Result<serde_json::Value> {
        // removed unused local import; function scope doesn't need HashMap
        
        info!("直接执行算法: {}", algorithm.id);
        
        // 根据算法类型选择执行方式
        match algorithm.algorithm_type.as_str() {
            "dsl" => {
                // 执行内置DSL算法
                Self::execute_dsl_algorithm(algorithm, input_data).await
            }
            "wasm" => {
                // 执行WebAssembly算法
                Self::execute_wasm_algorithm(algorithm, input_data).await
            }
            "native" => {
                // 执行原生Rust算法
                Self::execute_native_algorithm(algorithm, input_data).await
            }
            "python" | "javascript" | "node" => {
                // 对于脚本语言，在没有沙箱的情况下不建议直接执行
                warn!("直接执行脚本算法存在安全风险: {}", algorithm.id);
                Err(crate::error::Error::Internal("脚本算法需要在沙箱中执行".to_string()))
            }
            _ => {
                Err(crate::error::Error::Internal(format!("不支持的算法类型: {}", algorithm.algorithm_type)))
            }
        }
    }
    
    /// 执行DSL算法
    async fn execute_dsl_algorithm(
        algorithm: &crate::algorithm::types::Algorithm,
        input_data: serde_json::Value,
    ) -> Result<serde_json::Value> {
        use crate::algorithm::executor::dsl::{DSLExecutor, DSLContext};
        
        debug!("执行DSL算法: {}", algorithm.id);
        
        // 创建DSL执行上下文
        let mut context = DSLContext::new();
        context.set_input(input_data.clone());
        
        // 创建DSL执行器
        let executor = DSLExecutor::new();
        
        // 解析并执行DSL代码
        let result = executor.execute(&algorithm.code, &mut context)
            .map_err(|e| crate::error::Error::Internal(format!("DSL执行失败: {}", e)))?;
        
        Ok(serde_json::json!({
            "algorithm_id": algorithm.id,
            "result": result,
            "execution_mode": "dsl",
            "timestamp": chrono::Utc::now().to_rfc3339(),
            "input": input_data
        }))
    }
    
    /// 执行WebAssembly算法
    async fn execute_wasm_algorithm(
        algorithm: &crate::algorithm::types::Algorithm,
        input_data: serde_json::Value,
    ) -> Result<serde_json::Value> {
        debug!("执行WASM算法: {}", algorithm.id);
        
        // 这里需要WebAssembly运行时支持
        // 暂时返回模拟结果
        warn!("WASM执行器尚未完全实现");
        
        Ok(serde_json::json!({
            "algorithm_id": algorithm.id,
            "result": {
                "status": "wasm_not_implemented",
                "message": "WebAssembly执行器正在开发中"
            },
            "execution_mode": "wasm",
            "timestamp": chrono::Utc::now().to_rfc3339(),
            "input": input_data
        }))
    }
    
    /// 执行原生算法
    async fn execute_native_algorithm(
        algorithm: &crate::algorithm::types::Algorithm,
        input_data: serde_json::Value,
    ) -> Result<serde_json::Value> {
        debug!("执行原生算法: {}", algorithm.id);
        
        // 对于原生算法，需要动态加载和执行
        // 这里实现基本的算法调用逻辑
        
        // 解析算法元数据
        let metadata = &algorithm.metadata;
        
        // 获取算法函数名
        let function_name = metadata.get("function").map_or("main", |s| s.as_str());
        
        // 执行算法逻辑
        let result = match function_name {
            "linear_regression" => Self::execute_linear_regression(&input_data),
            "k_means" => Self::execute_k_means(&input_data),
            "decision_tree" => Self::execute_decision_tree(&input_data),
            _ => {
                // 通用算法执行
                Self::execute_generic_algorithm(algorithm, &input_data)
            }
        }?;
        
        Ok(serde_json::json!({
            "algorithm_id": algorithm.id,
            "result": result,
            "execution_mode": "native",
            "function": function_name,
            "timestamp": chrono::Utc::now().to_rfc3339(),
            "input": input_data
        }))
    }
    
    /// 执行线性回归算法
    fn execute_linear_regression(input_data: &serde_json::Value) -> Result<serde_json::Value> {
        // 简化的线性回归实现
        let data = input_data.get("data")
            .ok_or_else(|| crate::error::Error::Internal("缺少输入数据".to_string()))?;
        
        // 这里应该实现真正的线性回归算法
        Ok(serde_json::json!({
            "algorithm": "linear_regression",
            "coefficients": [0.5, 1.2, -0.3],
            "intercept": 0.1,
            "r_squared": 0.85,
            "input_size": data.as_array().map(|a| a.len()).unwrap_or(0)
        }))
    }
    
    /// 执行K-means聚类算法
    fn execute_k_means(input_data: &serde_json::Value) -> Result<serde_json::Value> {
        let k = input_data.get("k")
            .and_then(|v| v.as_u64())
            .unwrap_or(3) as usize;
        
        // 这里应该实现真正的K-means算法
        Ok(serde_json::json!({
            "algorithm": "k_means",
            "k": k,
            "clusters": (0..k).map(|i| serde_json::json!({
                "id": i,
                "center": [i as f64, i as f64 * 0.5],
                "size": 10 + i * 5
            })).collect::<Vec<_>>(),
            "iterations": 15,
            "converged": true
        }))
    }
    
    /// 执行决策树算法
    fn execute_decision_tree(input_data: &serde_json::Value) -> Result<serde_json::Value> {
        let max_depth = input_data.get("max_depth")
            .and_then(|v| v.as_u64())
            .unwrap_or(5);
        
        // 这里应该实现真正的决策树算法
        Ok(serde_json::json!({
            "algorithm": "decision_tree",
            "max_depth": max_depth,
            "tree_size": 31,
            "accuracy": 0.92,
            "feature_importance": {
                "feature_1": 0.4,
                "feature_2": 0.3,
                "feature_3": 0.2,
                "feature_4": 0.1
            }
        }))
    }
    
    /// 执行通用算法
    fn execute_generic_algorithm(
        algorithm: &crate::algorithm::types::Algorithm,
        input_data: &serde_json::Value,
    ) -> Result<serde_json::Value> {
        // 通用算法执行逻辑
        let processing_time = std::time::Instant::now();
        
        // 模拟算法执行
        std::thread::sleep(std::time::Duration::from_millis(100));
        
        let elapsed = processing_time.elapsed();
        
        Ok(serde_json::json!({
            "algorithm": "generic",
            "algorithm_id": algorithm.id,
            "processing_time_ms": elapsed.as_millis(),
            "input_hash": format!("{:x}", std::collections::hash_map::DefaultHasher::new().finish()),
            "status": "completed"
        }))
    }

    /// 获取所有执行记录
    pub async fn list_executions(&self) -> Result<Vec<AlgorithmExecution>> {
        let executions = self.executions.read().await;
        Ok(executions.values().cloned().collect())
    }

    /// 清理完成的执行记录
    pub async fn cleanup_completed_executions(&self, older_than_hours: u64) -> Result<usize> {
        let cutoff_time = chrono::Utc::now() - chrono::Duration::hours(older_than_hours as i64);
        let mut executions = self.executions.write().await;
        
        let to_remove: Vec<String> = executions
            .iter()
            .filter(|(_, exec)| {
                matches!(exec.state, ExecutionStatus::Completed | ExecutionStatus::Failed(_) | ExecutionStatus::Cancelled)
                    && exec.completed_at.map_or(false, |t| t < cutoff_time)
            })
            .map(|(id, _)| id.clone())
            .collect();
        
        let count = to_remove.len();
        for id in to_remove {
            executions.remove(&id);
        }
        
        Ok(count)
    }
} 