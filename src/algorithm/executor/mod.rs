//! 算法执行器模块
//!
//! 提供算法代码的安全执行环境和资源限制功能

pub mod config;
pub mod resources;
pub mod sandbox;
pub mod result;
pub mod metrics;
pub mod dsl;

pub use config::{
    SandboxType, SandboxConfig,
    ConfigNetworkPolicy, ConfigFilesystemPolicy,
    ExecutorConfig
};
// 从types模块导入核心类型
pub use crate::algorithm::types::{ResourceLimits, SandboxSecurityLevel};
pub use resources::{ResourceMonitor};
pub use sandbox::{Sandbox, SandboxResult, SandboxError};
pub use result::{
    ExecutionError, ExecutionMetrics,
    ExecutionTask, ExecutionState, ExecutionMode,
    ExecutorNetworkPolicy, ExecutorFilesystemPolicy,
    ExecutorSandboxStatus, ExecutorSandboxType, ExecutorSandboxConfig
};
pub use dsl::{AlgorithmAst, AlgorithmNode, NodeType, Operation, Activation};

use crate::core::types::{CoreTensorData, ResourceRequirements};
use crate::core::unified_system::AlgorithmDefinition;
use crate::api::routes::algorithm::types::AlgorithmInfo;
use crate::core::interfaces::{
    ValidationResult as ValidationResult,
    AlgorithmExecutorInterface,
};
use crate::algorithm::types::{ExecutionResult, ExecutionStatus, ResourceUsage};
use crate::algorithm::executor::sandbox::DefaultSandbox;
use crate::data::DataBatch;
use crate::error::{Error, Result};
use log::{info, debug, error, warn};
use std::time::Instant;
use std::sync::{Arc, Mutex};
use std::collections::HashMap;
use crate::resource::manager::ResourceManager;
use crate::resource::manager::ResourceManagerConfig;
use crate::types::{ResourceType, ResourceRequest, ResourceAllocation};
use tokio;
use tokio::sync::{Mutex as AsyncMutex};
use uuid::Uuid;
use chrono::Utc;
#[cfg(feature = "wasmtime")]
use wasmtime::Module;
use crate::algorithm::types::SandboxStatus as AlgoSandboxStatus;

// 移除重复导入，已在config模块中导入

/// 算法执行器的统一接口trait
/// 
/// 提供算法执行的标准化接口，支持异步执行、资源管理、安全控制等功能
#[async_trait::async_trait]
pub trait ExecutorTrait: Send + Sync {
    /// 执行算法
    /// 
    /// # 参数
    /// - `algorithm`: 要执行的算法
    /// - `parameters`: 算法执行参数
    /// - `model_id`: 可选的模型ID
    /// - `config`: 执行器配置
    /// 
    /// # 返回值
    /// 返回执行结果
    async fn execute(
        &self,
        algorithm: &crate::algorithm::types::Algorithm,
        parameters: &std::collections::HashMap<String, serde_json::Value>,
        model_id: Option<&str>,
        config: &crate::algorithm::executor::config::ExecutorConfig,
    ) -> crate::error::Result<ExecutionResult>;

    /// 取消任务执行
    /// 
    /// # 参数
    /// - `task_id`: 要取消的任务ID
    /// 
    /// # 返回值
    /// 成功取消返回Ok(())，否则返回错误
    async fn cancel_execution(&self, task_id: &crate::task_scheduler::core::TaskId) -> crate::error::Result<()>;
    
    /// 取消任务（简短方法名）
    async fn cancel(&self, task_id: &str) -> crate::error::Result<()>;
    
    /// 获取任务状态（简短方法名）
    async fn get_status(&self, task_id: &str) -> crate::error::Result<ExecutionStatus>;

    /// 获取任务执行状态
    /// 
    /// # 参数
    /// - `task_id`: 要查询的任务ID
    /// 
    /// # 返回值
    /// 返回任务当前状态
    async fn get_execution_status(&self, task_id: &crate::task_scheduler::core::TaskId) -> crate::error::Result<ExecutionStatus>;

    /// 获取执行器当前负载
    /// 
    /// # 返回值
    /// 返回当前活跃任务数量
    async fn get_current_load(&self) -> u32;

    /// 检查执行器是否可以接受新任务
    /// 
    /// # 返回值
    /// true表示可以接受新任务，false表示已达到负载上限
    async fn can_accept_task(&self) -> bool;

    /// 获取执行器支持的算法类型
    /// 
    /// # 返回值
    /// 返回支持的算法类型列表
    fn supported_algorithm_types(&self) -> Vec<crate::algorithm::types::AlgorithmType>;

    /// 验证算法是否可以执行
    /// 
    /// # 参数
    /// - `algorithm`: 要验证的算法
    /// 
    /// # 返回值
    /// 验证通过返回Ok(())，否则返回错误
    async fn validate_algorithm(&self, algorithm: &crate::algorithm::types::Algorithm) -> crate::error::Result<()>;

    /// 准备执行环境
    /// 
    /// # 参数
    /// - `algorithm`: 要执行的算法
    /// - `config`: 执行器配置
    /// 
    /// # 返回值
    /// 环境准备成功返回Ok(())，否则返回错误
    async fn prepare_environment(&self, algorithm: &crate::algorithm::types::Algorithm, config: &crate::algorithm::executor::config::ExecutorConfig) -> crate::error::Result<()>;

    /// 清理执行环境
    /// 
    /// # 参数
    /// - `task_id`: 相关联的任务ID
    /// 
    /// # 返回值
    /// 清理成功返回Ok(())，否则返回错误
    async fn cleanup_environment(&self, task_id: &crate::task_scheduler::core::TaskId) -> crate::error::Result<()>;
}

/// 算法执行器
/// 
/// 提供算法代码的安全执行能力，包括：
/// - 不同语言的代码执行
/// - 资源限制和监控
/// - 安全沙箱隔离
/// - 执行指标收集
#[derive(Debug)]
pub struct AlgorithmExecutor {
    /// 执行器配置
    config: ExecutorConfig,
    /// WASM引擎
    #[cfg(feature = "wasmtime")]
    wasm_engine: wasmtime::Engine,
    /// 资源管理器
    resource_manager: Arc<ResourceManager>,
    /// 资源分配记录
    resource_allocations: AsyncMutex<HashMap<String, ResourceAllocation>>,
    /// 活跃的沙箱实例
    active_sandboxes: Arc<AsyncMutex<HashMap<String, Arc<DefaultSandbox>>>>,
    /// 算法定义存储（内存存储，生产环境应使用持久化存储）
    algorithm_definitions: Arc<AsyncMutex<HashMap<String, AlgorithmDefinition>>>,
    /// 执行任务记录（用于查询已完成任务的状态）
    execution_tasks: Arc<AsyncMutex<HashMap<String, ExecutionStatus>>>,
}

impl AlgorithmExecutor {
    /// 创建新的算法执行器实例
    pub fn new(config: ExecutorConfig) -> Self {
        debug!("创建算法执行器: {:?}", config);
        
        Self {
            config,
            #[cfg(feature = "wasmtime")]
            wasm_engine: wasmtime::Engine::default(),
            resource_manager: Arc::new(ResourceManager::new(ResourceManagerConfig::default())),
            resource_allocations: Mutex::new(HashMap::new()),
            active_sandboxes: Arc::new(AsyncMutex::new(HashMap::new())),
            algorithm_definitions: Arc::new(AsyncMutex::new(HashMap::new())),
            execution_tasks: Arc::new(AsyncMutex::new(HashMap::new())),
        }
    }
    
    /// 检查代码安全性
    fn check_code_security(&self, _code: &str, _language: &str) -> Result<()> {
        // 生产级实现应包含静态代码分析、依赖项检查等
        Ok(())
    }

    /// 远程执行代码的占位实现
    fn execute_remotely(
        &self,
        algorithm: &crate::algorithm::types::Algorithm,
        _inputs: &DataBatch,
    ) -> Result<ExecutionResult> {
        let execution_id = Uuid::new_v4().to_string();
        Ok(ExecutionResult::new(
            execution_id,
            algorithm.id.clone(),
            vec![], // output
            ExecutionStatus::Pending,
            ResourceUsage::default(),
            ExecutionMetrics::default(),
            None,
        ))
    }
    
    /// 清理所有完成或失败的沙箱实例
    pub async fn cleanup_sandboxes(&self) -> Result<usize> {
        debug!("清理过期的沙箱实例");
        let mut sandboxes = self.active_sandboxes.lock().await;
        let initial_len = sandboxes.len();
        
        let mut completed_ids = Vec::new();
        for (id, sandbox) in sandboxes.iter() {
            if let Ok(status) = sandbox.get_status().await {
                if matches!(status, AlgoSandboxStatus::Success | AlgoSandboxStatus::Failure(_)) {
                    completed_ids.push(id.clone());
                }
            }
        }
        
        for id in completed_ids {
            sandboxes.remove(&id);
        }
        
        let removed_count = initial_len - sandboxes.len();
        if removed_count > 0 {
            debug!("清理了 {} 个沙箱实例", removed_count);
        }
        Ok(removed_count)
    }

    /// 请求算法执行所需的资源
    async fn request_resources(&self, algo_id: &str, priority: crate::algorithm::types::TaskPriority) -> Result<()> {
        debug!("为算法 {} 请求资源，优先级: {:?}", algo_id, priority);
        let mut allocations = self.resource_allocations.lock().await;
        
        if allocations.contains_key(algo_id) {
            debug!("算法 {} 已有资源分配", algo_id);
            return Ok(());
        }

        // 转换类型到资源模块的TaskPriority
        let resource_priority = match priority {
            crate::algorithm::types::TaskPriority::Low => crate::types::TaskPriority::Low,
            crate::algorithm::types::TaskPriority::Normal => crate::types::TaskPriority::Normal,
            crate::algorithm::types::TaskPriority::High => crate::types::TaskPriority::High,
            crate::algorithm::types::TaskPriority::Critical => crate::types::TaskPriority::Critical,
        };

        let request = ResourceRequest::new(
            algo_id.to_string(),
            ResourceType::AlgorithmExecution,
            1, // 假设一次执行需要1个单位的资源
        ).with_priority(resource_priority);

        match self.resource_manager.request_resource(request).await {
            Ok(allocation) => {
                info!("为算法 {} 成功分配资源: {:?}", algo_id, allocation);
                allocations.insert(algo_id.to_string(), allocation);
                Ok(())
            }
            Err(e) => {
                error!("为算法 {} 分配资源失败: {}", algo_id, e);
                Err(e)
            }
        }
    }

    async fn release_resources(&self, algo_id: &str) -> Result<()> {
        debug!("为算法 {} 释放资源", algo_id);
        let mut allocations = self.resource_allocations.lock().await;
        
        if let Some(allocation) = allocations.remove(algo_id) {
            if let Err(e) = self.resource_manager.release_resource(&allocation.allocation_id).await {
                error!("释放资源失败: {}", e);
                // 即使释放失败，也继续执行，但记录错误
            }
        } else {
            warn!("尝试释放不存在的算法 {} 的资源", algo_id);
        }
        Ok(())
    }
}

impl AlgorithmExecutor {
    /// 就绪检查：验证执行器关键组件可用性（策略/沙箱/注册表接线）
    pub async fn readiness_probe(&self) -> Result<()> {
        // 1) 进行一次最小的算法校验（不执行）
        let dummy_code = "fn main() { }";
        let _ = self.validate_algorithm_code(dummy_code).await?;

        // 2) 进行一次最小资源估算，确保资源路径可用
        let _req = self.estimate_resource_requirements(dummy_code);

        // 3) 检查内部注册表是否可访问（如算法定义获取路径）
        let _ = self.list_algorithms().await?;
        Ok(())
    }
}

#[async_trait::async_trait]
impl ExecutorTrait for AlgorithmExecutor {
    async fn execute(
        &self,
        algorithm: &crate::algorithm::types::Algorithm,
        parameters: &std::collections::HashMap<String, serde_json::Value>,
        _model_id: Option<&str>,
        config: &crate::algorithm::executor::config::ExecutorConfig,
    ) -> crate::error::Result<ExecutionResult> {
        info!("开始执行算法: {}, 版本: {}", algorithm.id, algorithm.version);
        // 简化输入处理
        let input_data = parameters
            .get("input")
            .cloned()
            .unwrap_or(serde_json::Value::Null);

        // 远程执行（如果配置如此）
        if config.execution_mode == ExecutionMode::Asynchronous {
            info!("为算法 {} 启动异步执行", algorithm.id);
            // 这里是同步调用，因为它只是触发一个任务，而不是等待它完成
            let result = self.execute_remotely(algorithm, &DataBatch::default())?;
            return Ok(result);
        }

        // 本地沙箱执行
        let sandbox_id = Uuid::new_v4().to_string();
        info!("为算法 {} 创建沙箱，ID: {}", algorithm.id, sandbox_id);

        let wasm_binary = algorithm.code.as_bytes();
        #[cfg(feature = "wasmtime")]
        let wasm_module = Arc::new(Module::new(wasm_binary)
            .map_err(|e| Error::compilation(format!("无法编译WASM模块: {}", e)))?);
        #[cfg(not(feature = "wasmtime"))]
        return Err(Error::feature_not_enabled("wasmtime"));

        let input_bytes = bincode::serialize(&input_data)
            .map_err(|e| Error::serialization(format!("无法序列化输入数据: {}", e)))?;

        let sandbox_config = config.sandbox_config.clone().unwrap_or_default();
        let limits = ResourceLimits::from(config.resource_limits.clone());
        let timeout = config.timeout;

        #[cfg(feature = "wasmtime")]
        let sandbox_result = sandbox::utils::execute_in_sandbox(
            &wasm_module,
            &input_bytes,
            &sandbox_config,
            limits,
            timeout,
        )
        .await?;

        #[cfg(feature = "wasmtime")]
        {
            let status = match sandbox_result.status {
                AlgoSandboxStatus::Success => ExecutionStatus::Completed,
                AlgoSandboxStatus::Error => ExecutionStatus::Failed(sandbox_result.stderr.clone()),
                AlgoSandboxStatus::Timeout => ExecutionStatus::Timeout,
            };
            
            let execution_id = sandbox_id;
            let result = ExecutionResult::new(
                execution_id,
                algorithm.id.clone(),
                sandbox_result.stdout.clone().into_bytes(),
                status,
                sandbox_result.resource_usage.clone(),
                ExecutionMetrics::from(sandbox_result.clone()),
                Some(sandbox_result),
            );

            Ok(result)
        }
        #[cfg(not(feature = "wasmtime"))]
        {
            Err(Error::feature_not_enabled("wasmtime"))
        }
    }

    async fn cancel_execution(&self, task_id: &crate::task_scheduler::core::TaskId) -> crate::error::Result<()> {
        info!("请求取消任务: {}", task_id);
        let mut sandboxes = self.active_sandboxes.lock().await;
        if let Some(sandbox) = sandboxes.get_mut(task_id.to_string().as_str()) {
            sandbox.cancel().await
        } else {
            Err(Error::NotFound(format!("找不到任务ID为 {} 的活跃沙箱", task_id)))
        }
    }

    async fn get_execution_status(&self, task_id: &crate::task_scheduler::core::TaskId) -> crate::error::Result<ExecutionStatus> {
        let sandboxes = self.active_sandboxes.lock().await;
        if let Some(sandbox) = sandboxes.get(task_id.to_string().as_str()) {
            Ok(sandbox.get_status())
        } else {
            // 在生产环境中，这里应该查询持久化存储来获取已完成任务的状态
            Err(Error::NotFound(format!("找不到ID为 {} 的活跃任务", task_id)))
        }
    }

    async fn get_current_load(&self) -> u32 {
        self.active_sandboxes.lock().await.len() as u32
    }

    async fn can_accept_task(&self) -> bool {
        let current_load = self.get_current_load().await;
        current_load < self.config.max_concurrent_tasks
    }

    fn supported_algorithm_types(&self) -> Vec<crate::algorithm::types::AlgorithmType> {
        vec![
            crate::algorithm::types::AlgorithmType::Wasm,
            crate::algorithm::types::AlgorithmType::DataProcessing,
        ]
    }

    async fn validate_algorithm(&self, algorithm: &crate::algorithm::types::Algorithm) -> crate::error::Result<()> {
        info!("验证算法: {}", algorithm.id);
        self.check_code_security(&algorithm.code, &algorithm.language)?;
        
        let wasm_binary = algorithm.code.as_bytes();
        if let Err(e) = crate::algorithm::wasm::check_security(wasm_binary) {
             return Err(Error::validation(format!("WASM 安全检查失败: {}", e)));
        }
        #[cfg(feature = "wasmtime")]
        if let Err(e) = Module::new(wasm_binary) {
            return Err(Error::validation(format!("WASM 模块验证失败: {}", e)));
        }
        #[cfg(not(feature = "wasmtime"))]
        return Err(Error::feature_not_enabled("wasmtime"));

        Ok(())
    }

    async fn prepare_environment(&self, _algorithm: &crate::algorithm::types::Algorithm, _config: &crate::algorithm::executor::config::ExecutorConfig) -> crate::error::Result<()> {
        Ok(())
    }

    async fn cleanup_environment(&self, task_id: &crate::task_scheduler::core::TaskId) -> crate::error::Result<()> {
        self.release_resources(&task_id.to_string()).await?;
        let mut sandboxes = self.active_sandboxes.lock().await;
        if sandboxes.remove(task_id.to_string().as_str()).is_some() {
            info!("已清理任务 {} 的环境", task_id);
        }
        Ok(())
    }

    async fn cancel(&self, task_id: &str) -> crate::error::Result<()> {
        let task_id = task_id.parse().map_err(|_| Error::invalid_input("无效的任务ID格式".to_string()))?;
        self.cancel_execution(&task_id).await
    }

    async fn get_status(&self, task_id: &str) -> crate::error::Result<ExecutionStatus> {
        let task_id = task_id.parse().map_err(|_| Error::invalid_input("无效的任务ID格式".to_string()))?;
        self.get_execution_status(&task_id).await
    }
}

impl Default for AlgorithmExecutor {
    fn default() -> Self {
        Self::new(ExecutorConfig::default())
    }
}

#[async_trait::async_trait]
impl AlgorithmExecutorInterface for AlgorithmExecutor {
    async fn execute_algorithm(&self, algo_id: &str, inputs: &[CoreTensorData]) -> Result<Vec<CoreTensorData>> {
        debug!("执行算法: {}, 输入数量: {}", algo_id, inputs.len());
        
        // 获取算法定义
        let algorithm = self.get_algorithm_definition(algo_id).await?;
        
        // 转换输入数据格式
        let data_batch = self.convert_core_tensors_to_batch(inputs).await?;
        
        // 执行算法
        let execution_result = self.execute_algorithm_internal(&algorithm, &data_batch).await?;
        
        // 转换输出数据格式 - 从execution_result.output创建DataBatch
        // 这里需要从output字节数组重新构造DataBatch
        let output_batch = DataBatch { id: Some(Uuid::new_v4().to_string()), dataset_id: "output".to_string(), index: 0, batch_index: 0, size: 0, batch_size: 0, status: crate::data::types::DataStatus::Created, created_at: Utc::now(), data: None, labels: None, metadata: HashMap::new(), format: crate::data::types::DataFormat::Binary, source: None, records: Vec::new(), schema: None, field_names: Vec::new(), features: None, target: None, version: Some(1), checksum: None, compression: None, encryption: None, tags: Vec::new(), ..Default::default() };
        let outputs = self.convert_batch_to_core_tensors(&output_batch).await?;
        
        Ok(outputs)
    }

    async fn validate_algorithm(&self, algo_code: &str) -> Result<ValidationResult> {
        // 复用现有代码验证逻辑
        self.validate_algorithm_code(algo_code).await
    }

    async fn validate_algorithm_code(&self, algo_code: &str) -> Result<ValidationResult> {
        debug!("验证算法代码");
        
        let mut errors = Vec::new();
        let mut warnings = Vec::new();
        let mut details = HashMap::new();
        
        // 语法检查
        if let Err(e) = self.check_code_security(algo_code, "rust") {
            errors.push(format!("代码安全检查失败: {}", e));
        }
        
        // 依赖检查
        let dependencies = self.extract_dependencies(algo_code);
        if !dependencies.is_empty() {
            details.insert("dependencies".to_string(), dependencies.join(", "));
        }
        
        // 资源需求检查
        let resource_req = self.estimate_resource_requirements(algo_code);
        details.insert("estimated_memory_mb".to_string(), resource_req.max_memory_mb.to_string());
        details.insert("estimated_cpu_percent".to_string(), resource_req.preferred_cpu_cores.to_string());
        
        let is_valid = errors.is_empty();
        let mut vr = ValidationResult::success();
        vr.is_valid = is_valid;
        vr.errors = errors;
        vr.warnings = warnings;
        vr.details = details;
        vr.performance_hints = vec!["建议在沙箱环境中测试".to_string()];
        vr.dependencies = self.extract_dependencies(algo_code);
        Ok(vr)
    }

    async fn register_algorithm(&self, algo_def: &AlgorithmDefinition) -> Result<String> {
        debug!("注册算法: {}", algo_def.name);
        
        // 验证算法定义
        let validation = self.validate_algorithm(&algo_def.source_code).await?;
        if !validation.is_valid {
            return Err(Error::InvalidInput(format!("算法验证失败: {:?}", validation.errors)));
        }
        
        // 生成算法ID
        let algorithm_id = Uuid::new_v4().to_string();
        
        // 存储算法定义
        self.store_algorithm_definition(&algorithm_id, algo_def).await?;
        
        info!("算法注册成功: {} -> {}", algo_def.name, algorithm_id);
        Ok(algorithm_id)
    }

    async fn get_algorithm(&self, algo_id: &str) -> Result<Option<AlgorithmDefinition>> {
        debug!("获取算法定义: {}", algo_id);
        
        // 从存储中获取算法定义
        match self.get_algorithm_definition(algo_id).await {
            Ok(definition) => Ok(Some(definition)),
            Err(_) => Ok(None),
        }
    }

    async fn list_algorithms(&self) -> Result<Vec<AlgorithmInfo>> {
        debug!("列出所有算法");
        
        // 从存储中获取所有算法信息
        let algorithms = self.get_all_algorithm_definitions().await?;
        
        let mut algorithm_infos = Vec::new();
        for (id, definition) in algorithms {
            algorithm_infos.push(AlgorithmInfo {
                id,
                name: definition.name,
                category: format!("{:?}", definition.algorithm_type),
                version: "1.0.0".to_string(), // AlgorithmDefinition 没有 version 字段，使用默认值
                description: Some(definition.description),
            });
        }
        
        Ok(algorithm_infos)
    }

    async fn update_algorithm(&self, algo_id: &str, algo_def: &AlgorithmDefinition) -> Result<()> {
        debug!("更新算法: {}", algo_id);
        
        // 验证算法定义
        let validation = self.validate_algorithm(&algo_def.source_code).await?;
        if !validation.is_valid {
            return Err(Error::InvalidInput(format!("算法验证失败: {:?}", validation.errors)));
        }
        
        // 检查算法是否存在
        if self.get_algorithm(algo_id).await?.is_none() {
            return Err(Error::InvalidInput(format!("算法不存在: {}", algo_id)));
        }
        
        // 更新算法定义
        self.store_algorithm_definition(algo_id, algo_def).await?;
        
        info!("算法更新成功: {}", algo_id);
        Ok(())
    }
    
    async fn execute(&self, algorithm: &crate::interfaces::algorithm::AlgorithmDefinitionInterface, inputs: std::collections::HashMap<String, serde_json::Value>) -> Result<crate::interfaces::algorithm::AlgorithmResultInterface> {
        debug!("执行算法: {}", algorithm.name);
        let start_time = Instant::now();
        let execution_id = Uuid::new_v4().to_string();
        
        // 转换输入：将 serde_json::Value 转换为 CoreTensorData
        let mut core_inputs = Vec::new();
        for (key, value) in inputs {
            match value {
                serde_json::Value::Array(arr) => {
                    let data: Result<Vec<f32>, _> = arr.iter()
                        .map(|v| {
                            v.as_f64()
                                .ok_or_else(|| Error::InvalidInput(format!("无法将值转换为f64: {:?}", v)))
                                .map(|f| f as f32)
                        })
                        .collect();
                    core_inputs.push(CoreTensorData {
                        id: Uuid::new_v4(),
                        data: data?,
                        shape: vec![arr.len()],
                        dtype: crate::core::types::DataType::Float32,
                        metadata: std::collections::HashMap::new(),
                    });
                }
                serde_json::Value::Number(n) => {
                    if let Some(f) = n.as_f64() {
                        core_inputs.push(CoreTensorData {
                            id: Uuid::new_v4(),
                            data: vec![f as f32],
                            shape: vec![1],
                            dtype: crate::core::types::DataType::Float32,
                            metadata: std::collections::HashMap::new(),
                        });
                    }
                }
                _ => {
                    return Err(Error::InvalidInput(format!("不支持的输入类型: {:?}", value)));
                }
            }
        }
        
        // 执行算法
        let outputs = self.execute_algorithm(&algorithm.id, &core_inputs).await?;
        
        // 记录执行状态
        let execution_time_ms = start_time.elapsed().as_millis() as f64;
        let mut tasks = self.execution_tasks.lock().await;
        tasks.insert(execution_id.clone(), ExecutionStatus::Completed);
        
        Ok(crate::interfaces::algorithm::AlgorithmResultInterface {
            execution_id: execution_id.clone(),
            outputs: outputs,
            execution_time_ms,
            resource_usage: crate::algorithm::types::ResourceUsage::default(),
            metadata: {
                let mut meta = std::collections::HashMap::new();
                meta.insert("algorithm_id".to_string(), algorithm.id.clone());
                meta.insert("algorithm_name".to_string(), algorithm.name.clone());
                meta
            },
        })
    }
    
    async fn validate_inputs(&self, algorithm: &crate::interfaces::algorithm::AlgorithmDefinitionInterface, inputs: &std::collections::HashMap<String, serde_json::Value>) -> Result<crate::interfaces::algorithm::ValidationResultInterface> {
        debug!("验证输入: {}", algorithm.name);
        let mut errors = Vec::new();
        let mut warnings = Vec::new();
        
        // 验证输入数量
        if inputs.len() != algorithm.input_schema.len() {
            errors.push(format!(
                "输入数量不匹配: 期望 {}, 实际 {}",
                algorithm.input_schema.len(),
                inputs.len()
            ));
        }
        
        // 验证每个输入
        for (i, schema) in algorithm.input_schema.iter().enumerate() {
            let input_key = format!("input_{}", i);
            if let Some(value) = inputs.get(&input_key) {
                // 验证数据类型
                match (value, &schema.data_type) {
                    (serde_json::Value::Array(_), crate::core::types::DataType::Float32) |
                    (serde_json::Value::Array(_), crate::core::types::DataType::Float64) => {
                        // 数组类型匹配
                    }
                    (serde_json::Value::Number(_), crate::core::types::DataType::Float32) |
                    (serde_json::Value::Number(_), crate::core::types::DataType::Float64) => {
                        // 标量类型匹配
                    }
                    _ => {
                        warnings.push(format!(
                            "输入 {} 的数据类型可能不匹配: 期望 {:?}, 实际 {:?}",
                            input_key, schema.data_type, value
                        ));
                    }
                }
            } else if !schema.optional {
                errors.push(format!("缺少必需的输入: {}", input_key));
            }
        }
        
        Ok(crate::interfaces::algorithm::ValidationResultInterface {
            is_valid: errors.is_empty(),
            errors,
            warnings,
        })
    }
    
    async fn get_execution_status(&self, execution_id: &str) -> Result<crate::interfaces::algorithm::ExecutionStatusInterface> {
        debug!("获取执行状态: {}", execution_id);
        
        // 首先检查活跃的沙箱
        let sandboxes = self.active_sandboxes.lock().await;
        if let Some(sandbox) = sandboxes.get(execution_id) {
            let status = sandbox.get_status();
            drop(sandboxes);
            return Ok(match status {
                ExecutionStatus::Pending => crate::interfaces::algorithm::ExecutionStatusInterface::Pending,
                ExecutionStatus::Running => crate::interfaces::algorithm::ExecutionStatusInterface::Running,
                ExecutionStatus::Completed => crate::interfaces::algorithm::ExecutionStatusInterface::Completed,
                ExecutionStatus::Failed(_) => crate::interfaces::algorithm::ExecutionStatusInterface::Failed,
                ExecutionStatus::Cancelled => crate::interfaces::algorithm::ExecutionStatusInterface::Cancelled,
                ExecutionStatus::Timeout => crate::interfaces::algorithm::ExecutionStatusInterface::Timeout,
            });
        }
        drop(sandboxes);
        
        // 检查已完成的任务记录
        let tasks = self.execution_tasks.lock().await;
        if let Some(status) = tasks.get(execution_id) {
            return Ok(match status {
                ExecutionStatus::Pending => crate::interfaces::algorithm::ExecutionStatusInterface::Pending,
                ExecutionStatus::Running => crate::interfaces::algorithm::ExecutionStatusInterface::Running,
                ExecutionStatus::Completed => crate::interfaces::algorithm::ExecutionStatusInterface::Completed,
                ExecutionStatus::Failed(_) => crate::interfaces::algorithm::ExecutionStatusInterface::Failed,
                ExecutionStatus::Cancelled => crate::interfaces::algorithm::ExecutionStatusInterface::Cancelled,
                ExecutionStatus::Timeout => crate::interfaces::algorithm::ExecutionStatusInterface::Timeout,
            });
        }
        
        Err(Error::NotFound(format!("找不到执行ID为 {} 的任务", execution_id)))
    }
    
    async fn cancel_execution(&self, execution_id: &str) -> Result<()> {
        debug!("取消执行: {}", execution_id);
        
        // 尝试从活跃沙箱中取消
        let mut sandboxes = self.active_sandboxes.lock().await;
        if let Some(sandbox) = sandboxes.get_mut(execution_id) {
            sandbox.cancel().await?;
            // 更新任务状态
            let mut tasks = self.execution_tasks.lock().await;
            tasks.insert(execution_id.to_string(), ExecutionStatus::Cancelled);
            return Ok(());
        }
        drop(sandboxes);
        
        // 检查任务是否已完成或已取消
        let tasks = self.execution_tasks.lock().await;
        if let Some(status) = tasks.get(execution_id) {
            match status {
                ExecutionStatus::Completed | ExecutionStatus::Cancelled | ExecutionStatus::Failed(_) => {
                    return Err(Error::InvalidInput(format!("任务 {} 已经完成或已取消，无法再次取消", execution_id)));
                }
                _ => {}
            }
        }
        
        Err(Error::NotFound(format!("找不到执行ID为 {} 的活跃任务", execution_id)))
    }
}

impl AlgorithmExecutor {
    /// 获取算法定义
    async fn get_algorithm_definition(&self, algo_id: &str) -> Result<AlgorithmDefinition> {
        // 从内存存储中获取算法定义
        let definitions = self.algorithm_definitions.lock().await;
        if let Some(definition) = definitions.get(algo_id) {
            return Ok(definition.clone());
        }
        drop(definitions);
        
        // 如果内存中没有，返回错误（生产环境应查询持久化存储）
        Err(Error::NotFound(format!("找不到ID为 {} 的算法定义", algo_id)))
    }

    /// 转换CoreTensorData到DataBatch
    async fn convert_core_tensors_to_batch(&self, tensors: &[CoreTensorData]) -> Result<DataBatch> {
        let mut records = Vec::new();
        
        for tensor in tensors {
            let mut record = HashMap::new();
            for (i, &value) in tensor.data.iter().enumerate() {
                record.insert(format!("feature_{}", i), crate::data::value::DataValue::Float(value as f64));
            }
            records.push(record);
        }
        
        Ok(DataBatch {
            id: Some(Uuid::new_v4().to_string()),
            dataset_id: "converted".to_string(),
            index: 0,
            batch_index: 0,
            size: records.len(),
            batch_size: records.len(),
            status: crate::data::types::DataStatus::Created,
            created_at: Utc::now(),
            data: None,
            labels: None,
            metadata: HashMap::new(),
            format: crate::data::types::DataFormat::Binary,
            source: None,
            records,
            schema: None,
            field_names: Vec::new(),
            features: None,
            target: None,
            version: Some(1),
            checksum: None,
            compression: None,
            encryption: None,
            tags: Vec::new(),
            child_batch_ids: Vec::new(),
            custom_data: std::collections::HashMap::new(),
            dependencies: Vec::new(),
            ..Default::default()
        })
    }

    /// 转换DataBatch到CoreTensorData
    async fn convert_batch_to_core_tensors(&self, batch: &DataBatch) -> Result<Vec<CoreTensorData>> {
        let mut tensors = Vec::new();
        
        // 从DataBatch的records字段提取数据
        for record in &batch.records {
            // 将记录转换为张量数据
            let mut data = Vec::new();
            for (_, value) in record {
                match value {
                    crate::data::value::DataValue::Float(f) => data.push(*f as f32),
                    crate::data::value::DataValue::Integer(i) => data.push(*i as f32),
                    crate::data::value::DataValue::Boolean(b) => data.push(if *b { 1.0 } else { 0.0 }),
                    _ => data.push(0.0), // 其他类型默认为0
                }
            }
            
            if !data.is_empty() {
                tensors.push(CoreTensorData {
                    id: Uuid::new_v4().to_string(),
                    shape: vec![data.len()],
                    data,
                    dtype: "float32".to_string(),
                    device: "cpu".to_string(),
                    requires_grad: false,
                    metadata: HashMap::new(),
                    created_at: chrono::Utc::now(),
                    updated_at: chrono::Utc::now(),
                });
            }
        }
        
        Ok(tensors)
    }

    /// 将 CoreAlgorithmDefinition 转换为 AlgorithmDefinition (core::interfaces)
    fn convert_to_algorithm_definition(&self, algo_def: &AlgorithmDefinition) -> Result<crate::core::interfaces::AlgorithmDefinition> {
        use crate::core::interfaces::{TensorSchemaInterface, ResourceRequirementsInterface};
        
        // 转换 algorithm_type 为 String
        let algorithm_type = match algo_def.algorithm_type {
            crate::core::types::AlgorithmType::Classification => "classification".to_string(),
            crate::core::types::AlgorithmType::Regression => "regression".to_string(),
            crate::core::types::AlgorithmType::Clustering => "clustering".to_string(),
            crate::core::types::AlgorithmType::DataProcessing => "data_processing".to_string(),
            crate::core::types::AlgorithmType::FeatureExtraction => "feature_extraction".to_string(),
            crate::core::types::AlgorithmType::Custom => "custom".to_string(),
            _ => "data_processing".to_string(),
        };
        
        // 转换 input_schema 和 output_schema
        let input_schema: Vec<TensorSchemaInterface> = algo_def.input_schema.iter().map(|s| {
            TensorSchemaInterface {
                name: s.name.clone(),
                shape: s.shape.clone(),
                data_type: s.data_type.clone(),
                optional: s.optional,
                description: None,
            }
        }).collect();
        
        let output_schema: Vec<TensorSchemaInterface> = algo_def.output_schema.iter().map(|s| {
            TensorSchemaInterface {
                name: s.name.clone(),
                shape: s.shape.clone(),
                data_type: s.data_type.clone(),
                optional: s.optional,
                description: None,
            }
        }).collect();
        
        // 转换 resource_requirements，保留所有原始信息
        let resource_requirements = ResourceRequirementsInterface {
            max_memory_mb: algo_def.resource_requirements.max_memory_mb,
            max_cpu_percent: algo_def.resource_requirements.max_cpu_percent,
            max_execution_time_seconds: algo_def.resource_requirements.max_execution_time_seconds,
            requires_gpu: algo_def.resource_requirements.requires_gpu,
            max_gpu_memory_mb: algo_def.resource_requirements.max_gpu_memory_mb,
            network_access: algo_def.resource_requirements.network_access,
            file_system_access: algo_def.resource_requirements.file_system_access.clone(),
        };
        
        // 从 metadata 中提取 author，如果不存在则使用默认值
        let author = algo_def.metadata.get("author")
            .cloned()
            .unwrap_or_else(|| "system".to_string());
        
        // 从 metadata 中提取 security_level，如果不存在则使用默认值 Safe
        let security_level = algo_def.metadata.get("security_level")
            .and_then(|s| match s.as_str() {
                "unsafe" => Some(crate::core::interfaces::SecurityLevelInterface::Unsafe),
                "restricted" => Some(crate::core::interfaces::SecurityLevelInterface::Restricted),
                "safe" | _ => Some(crate::core::interfaces::SecurityLevelInterface::Safe),
            })
            .unwrap_or(crate::core::interfaces::SecurityLevelInterface::Safe);
        
        Ok(crate::core::interfaces::AlgorithmDefinition {
            id: algo_def.id.clone(),
            name: algo_def.name.clone(),
            description: algo_def.description.clone(),
            algorithm_type,
            version: algo_def.version.clone(),
            author,
            source_code: algo_def.source_code.clone(),
            language: algo_def.language.clone(),
            dependencies: algo_def.dependencies.clone(),
            input_schema,
            output_schema,
            resource_requirements,
            security_level,
            metadata: algo_def.metadata.clone(),
            created_at: algo_def.created_at,
            updated_at: algo_def.updated_at,
        })
    }

    /// 内部算法执行方法
    async fn execute_algorithm_internal(&self, algorithm: &AlgorithmDefinition, data: &DataBatch) -> Result<ExecutionResult> {
        let execution_id = Uuid::new_v4().to_string();
        let start_time = Instant::now();
        
        // 将 CoreAlgorithmDefinition 转换为 AlgorithmDefinition (core::interfaces)
        let algo_def_interface = self.convert_to_algorithm_definition(algorithm)?;
        
        // 创建沙箱环境
        let sandbox = self.create_sandbox_for_execution().await?;
        
        // 在沙箱中执行算法
        let result = sandbox.execute_algorithm(&algo_def_interface, data).await?;
        
        let _execution_time = start_time.elapsed().as_millis() as u64;
        
        Ok(ExecutionResult {
            execution_id,
            algorithm_id: algorithm.id.clone(),
            output: result.output,
            status: ExecutionStatus::Completed,
            resource_usage: ResourceUsage::default(),
            metrics: crate::algorithm::executor::metrics::ExecutionMetrics::default(),
            error: None,
            start_time: Utc::now(),
            end_time: Utc::now(),
            sandbox_result: None,
        })
    }

    /// 创建执行沙箱
    async fn create_sandbox_for_execution(&self) -> Result<Arc<DefaultSandbox>> {
        let sandbox_id = Uuid::new_v4().to_string();
        
        // 创建沙箱配置
        let sandbox_config = crate::algorithm::executor::sandbox::types::SandboxConfig {
            memory_limit: Some(1024 * 1024 * 100), // 100MB
            cpu_limit: Some(50.0), // 50% CPU
            timeout: Some(std::time::Duration::from_secs(300)), // 5分钟
            network_access: false,
            file_system_access: false,
            allowed_syscalls: vec![],
            environment_variables: HashMap::new(),
            working_directory: None,
            security_level: SandboxSecurityLevel::Medium,
            isolation_level: crate::algorithm::executor::sandbox::types::IsolationLevel::Medium,
        };
        
        let sandbox = Arc::new(DefaultSandbox::new(sandbox_config)?);
        
        // 将沙箱添加到活跃列表
        let mut active_sandboxes = self.active_sandboxes.lock().await;
        active_sandboxes.insert(sandbox_id, sandbox.clone());
        
        Ok(sandbox)
    }

    /// 提取代码依赖
    fn extract_dependencies(&self, code: &str) -> Vec<String> {
        // 简单的依赖提取逻辑
        let mut dependencies = Vec::new();
        
        if code.contains("use std::") {
            dependencies.push("std".to_string());
        }
        if code.contains("use tokio::") {
            dependencies.push("tokio".to_string());
        }
        if code.contains("use serde::") {
            dependencies.push("serde".to_string());
        }
        
        dependencies
    }

    /// 估算资源需求
    fn estimate_resource_requirements(&self, _code: &str) -> ResourceRequirements {
        ResourceRequirements {
            min_memory_mb: 128,
            max_memory_mb: 1024,
            min_cpu_cores: 1,
            preferred_cpu_cores: 2,
            gpu_required: false,
            estimated_execution_time_ms: 5000,
        }
    }

    /// 存储算法定义
    async fn store_algorithm_definition(&self, algo_id: &str, definition: &AlgorithmDefinition) -> Result<()> {
        // 存储到内存存储（生产环境应同时写入持久化存储）
        let mut definitions = self.algorithm_definitions.lock().await;
        definitions.insert(algo_id.to_string(), definition.clone());
        info!("算法定义已存储: {}", algo_id);
        Ok(())
    }

    /// 获取所有算法定义
    async fn get_all_algorithm_definitions(&self) -> Result<HashMap<String, AlgorithmDefinition>> {
        // 从内存存储中获取所有算法定义（生产环境应从持久化存储获取）
        let definitions = self.algorithm_definitions.lock().await;
        Ok(definitions.clone())
    }
}
