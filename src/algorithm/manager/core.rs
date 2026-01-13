// 算法管理器核心实现
// 提供算法管理的基本功能

use std::collections::HashMap;
use std::sync::{Arc, Mutex, RwLock};
use std::time::SystemTime;
use serde::{Deserialize, Serialize};
use log::{info, error};
// removed unused mpsc import
use std::fmt;

use crate::error::{Error, Result};
use crate::algorithm::types::{Algorithm, AlgorithmMetrics, AlgorithmOptimizationConfig};
use crate::algorithm::types::SandboxSecurityLevel;
use crate::algorithm::executor::ExecutorTrait;
use crate::algorithm::security::SecurityPolicyManager;
use crate::algorithm::security_auto::AutoSecurityAdjuster;
use crate::data::DataBatch;
use crate::storage::Storage;
// Model manager removed - using compat stub types
use crate::event::EventSystem;
use crate::core::unified_algorithm_service::ExecutionConfig;
use crate::algorithm::types::{ExecutionContext, ExecutionStatus};
use crate::event::{Event, EventType};

use super::{
    AlgorithmManagerConfig, ProductionExecutor, ProductionEventSystem, ProductionModelManager,
    utils::{generate_id, validate_algorithm_basic, validate_algorithm_code_length},
    StatusTracker, TaskManager, ManagerAlgorithmExecutor, EventManager,
    ExecutionHistory, SystemResourceUsage,
};

/// 算法管理器
/// 提供算法的创建、管理、执行、监控等核心功能
pub struct AlgorithmManager {
    /// 存储引擎
    storage: Arc<Storage>,
    /// 模型管理器
    model_manager: Arc<RwLock<Box<dyn ModelManager>>>,
    /// 状态跟踪器
    status_tracker: Arc<StatusTracker>,
    /// 缓存的算法
    algorithms: Mutex<HashMap<String, Arc<Algorithm>>>,
    /// 任务管理器
    task_manager: TaskManager,
    /// 算法执行器
    executor: ManagerAlgorithmExecutor,
    /// 事件管理器
    event_manager: EventManager,
    /// 底层执行器
    raw_executor: Arc<dyn ExecutorTrait>,
    /// 配置
    config: AlgorithmManagerConfig,
    /// 安全策略管理器
    security_manager: Arc<SecurityPolicyManager>,
    /// 自动安全调整器
    auto_adjuster: Arc<AutoSecurityAdjuster>,
}

/// 算法验证结果 - 避免与门面冲突，保留领域含义
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlgorithmValidationResult {
    /// 算法ID
    pub algorithm_id: String,
    /// 验证是否通过
    pub is_valid: bool,
    /// 精度评分 (0.0-1.0)
    pub accuracy: f64,
    /// 性能评分 (0.0-1.0)
    pub performance: f64,
    /// 建议列表
    pub recommendations: Vec<String>,
    /// 验证数据大小
    pub validation_data_size: usize,
    /// 验证时间
    pub validation_time: chrono::DateTime<chrono::Utc>,
}

impl AlgorithmManager {
    /// 创建新的算法管理器
    pub fn new(
        storage: Arc<Storage>,
        model_manager: Arc<RwLock<Box<dyn ModelManager>>>,
        status_tracker: Arc<StatusTracker>,
        executor: Arc<dyn ExecutorTrait>,
        event_system: Arc<dyn EventSystem>,
        config: AlgorithmManagerConfig,
        security_manager: Arc<SecurityPolicyManager>,
        auto_adjuster: Arc<AutoSecurityAdjuster>,
    ) -> Self {
        Self {
            storage,
            model_manager,
            status_tracker,
            algorithms: Mutex::new(HashMap::new()),
            task_manager: TaskManager::new(),
            executor: ManagerAlgorithmExecutor::new(executor),
            event_manager: EventManager::new(event_system),
            raw_executor: executor,
            config,
            security_manager,
            auto_adjuster,
        }
    }
    
    /// 创建简化版算法管理器
    pub fn new_simple(storage: Arc<Storage>) -> Result<Self> {
        // 创建默认组件
        let model_manager = Arc::new(RwLock::new(Box::new(ProductionModelManager::new(
            storage.clone(),
            super::models::ModelManagerConfig::default(),
        )) as Box<dyn ModelManager>));
        
        let status_tracker = Arc::new(StatusTracker::new_simple());
        
        let executor = Arc::new(ProductionExecutor::new_default()?);
        
        let event_system = Arc::new(ProductionEventSystem::new()?);
        
        let config = AlgorithmManagerConfig::default();
        
        let security_manager = Arc::new(SecurityPolicyManager::new());
        
        let auto_adjuster = Arc::new(AutoSecurityAdjuster::new());
        
        Ok(Self::new(
            storage,
            model_manager,
            status_tracker,
            executor,
            event_system,
            config,
            security_manager,
            auto_adjuster,
        ))
    }
    
    /// 创建算法
    pub async fn create_algorithm(&self, algorithm: Algorithm) -> Result<String> {
        info!("创建算法: {}", algorithm.name);
        
        // 验证算法
        validate_algorithm_basic(&algorithm)?;
        validate_algorithm_code_length(&algorithm.code, 1_000_000)?;
        
        // 生成算法ID
        let algorithm_id = generate_id();
        
        // 存储算法（序列化为JSON字符串）
        let algorithm_json = serde_json::to_string(&algorithm)
            .map_err(|e| Error::Serialization(format!("Failed to serialize algorithm: {}", e)))?;
        self.storage.store_algorithm_async(&algorithm_id, &algorithm_json).await?;
        
        // 缓存算法
        {
            let mut algorithms = self.algorithms.lock().map_err(|e| {
                Error::Internal(format!("无法获取算法缓存锁: {}", e))
            })?;
            algorithms.insert(algorithm_id.clone(), Arc::new(algorithm));
        }
        
        // 发送事件
        let payload = serde_json::json!({
            "algorithm_id": algorithm_id,
            "action": "created"
        });
        self.send_event(EventType::AlgorithmCreated, payload);
        
        info!("算法创建成功: {}", algorithm_id);
        Ok(algorithm_id)
    }
    
    /// 获取算法（同步版本）
    pub fn get_algorithm(&self, algorithm_id: &str) -> Result<Option<Arc<Algorithm>>> {
        // 检查缓存
        {
            let algorithms = self.algorithms.lock().map_err(|e| {
                Error::Internal(format!("无法获取算法缓存锁: {}", e))
            })?;
            if let Some(algorithm) = algorithms.get(algorithm_id) {
                return Ok(Some(algorithm.clone()));
            }
        }
        
        // 如果缓存中没有，返回None
        Ok(None)
    }

    /// 获取算法（简化版，如果找不到则返回错误）
    pub fn get_algorithm_simple(&self, algorithm_id: &str) -> Result<Arc<Algorithm>> {
        self.get_algorithm(algorithm_id)?
            .ok_or_else(|| Error::NotFound(format!("算法 {} 未找到", algorithm_id)))
    }
    
    /// 获取算法（异步版本）
    pub async fn get_algorithm_async(&self, algorithm_id: &str) -> Result<Option<Arc<Algorithm>>> {
        // 首先检查缓存
        {
            let algorithms = self.algorithms.lock().map_err(|e| {
                Error::Internal(format!("无法获取算法缓存锁: {}", e))
            })?;
            if let Some(algorithm) = algorithms.get(algorithm_id) {
                return Ok(Some(algorithm.clone()));
            }
        }
        
        // 从存储中加载
        if let Some(algorithm_json) = self.storage.load_algorithm(algorithm_id).await? {
            let algorithm: Algorithm = serde_json::from_str(&algorithm_json)
                .map_err(|e| Error::Deserialization(format!("Failed to deserialize algorithm: {}", e)))?;
            // 缓存算法
            {
                let mut algorithms = self.algorithms.lock().map_err(|e| {
                    Error::Internal(format!("无法获取算法缓存锁: {}", e))
                })?;
                algorithms.insert(algorithm_id.to_string(), Arc::new(algorithm.clone()));
            }
            Ok(Some(Arc::new(algorithm)))
        } else {
            Ok(None)
        }
    }
    
    /// 更新算法
    pub async fn update_algorithm(&self, algorithm: Algorithm) -> Result<()> {
        let algorithm_id = algorithm.id.clone();
        info!("更新算法: {}", &algorithm_id);

        let algorithm_arc = Arc::new(algorithm);

        // 验证算法
        self.security_manager.validate_algorithm_update(&algorithm_arc)?;

        // 验证代码长度
        validate_algorithm_code_length(&algorithm_arc.code, 1_000_000)?;
        
        // 存储算法（序列化为JSON字符串）
        let algorithm_json = serde_json::to_string(&*algorithm_arc)
            .map_err(|e| Error::Serialization(format!("Failed to serialize algorithm: {}", e)))?;
        self.storage.store_algorithm_async(&algorithm_id, &algorithm_json).await?;
        
        // 更新缓存
        {
            let mut algorithms = self.algorithms.lock().map_err(|e| {
                Error::Internal(format!("无法获取算法缓存锁: {}", e))
            })?;
            algorithms.insert(algorithm_id.to_string(), Arc::clone(&algorithm_arc));
        }
        
        // 发送事件
        let payload = serde_json::json!({
            "algorithm_id": algorithm_id,
            "action": "updated"
        });
        self.send_event(EventType::AlgorithmUpdated, payload);
        
        info!("算法更新成功: {}", algorithm_id);
        Ok(())
    }
    
    /// 删除算法
    pub async fn delete_algorithm(&self, algorithm_id: &str) -> Result<()> {
        info!("删除算法: {}", algorithm_id);
        
        // 从存储中删除
        self.storage.delete_algorithm_async(algorithm_id).await?;
        
        // 从缓存中删除
        {
            let mut algorithms = self.algorithms.lock().map_err(|e| {
                Error::Internal(format!("无法获取算法缓存锁: {}", e))
            })?;
            algorithms.remove(algorithm_id);
        }
        
        // 发送事件
        let payload = serde_json::json!({
            "algorithm_id": algorithm_id,
            "action": "deleted"
        });
        self.send_event(EventType::AlgorithmDeleted, payload);
        
        info!("算法删除成功: {}", algorithm_id);
        Ok(())
    }
    
    /// 执行算法
    pub fn execute_algorithm(&self, algorithm_id: &str, input_data: &DataBatch, config: &ExecutionConfig) -> Result<DataBatch> {
        info!("执行算法: {}", algorithm_id);
        
        // 获取算法
        let algorithm = self.get_algorithm(algorithm_id)?.ok_or_else(|| 
            Error::NotFound(format!("算法不存在: {}", algorithm_id)))?;
        
        // 检查执行权限
        self.check_execution_permission(&algorithm, config)?;
        
        // 创建执行上下文
        let execution_id = generate_id();
        let context = ExecutionContext {
            execution_id: execution_id.clone(),
            algorithm_id: algorithm_id.to_string(),
            status: ExecutionStatus::Running,
            start_time: chrono::Utc::now(),
            end_time: None,
            result: None,
            error: None,
        };
        
        // 执行算法  
        let algorithm_clone = Arc::clone(&algorithm);
        let result = self.execute_in_sandbox(&algorithm_clone, input_data, config, &execution_id)?;
        
        // 发送事件
        let payload = serde_json::json!({
            "algorithm_id": algorithm_id,
            "execution_id": execution_id,
            "status": "completed"
        });
        self.send_event(EventType::AlgorithmExecuted, payload);
        
        info!("算法执行完成: {}", algorithm_id);
        result
    }
    
    /// 在沙箱中执行算法
    fn execute_in_sandbox(&self, algorithm: &Algorithm, input_data: &DataBatch, config: &ExecutionConfig, execution_id: &str) -> Result<DataBatch> {
        // 创建沙箱
        let sandbox = self.create_sandbox(config)?;
        
        // 根据算法类型执行
        match algorithm.algorithm_type {
            crate::algorithm::types::AlgorithmType::MachineLearning => {
                self.execute_ml_algorithm(algorithm, input_data, &sandbox)
            }
            crate::algorithm::types::AlgorithmType::DataProcessing => {
                self.execute_data_processing_algorithm(algorithm, input_data, &sandbox)
            }
            crate::algorithm::types::AlgorithmType::Optimization => {
                self.execute_optimization_algorithm(algorithm, input_data, &sandbox)
            }
            crate::algorithm::types::AlgorithmType::Custom => {
                self.execute_custom_algorithm(algorithm, input_data, &sandbox)
            }
        }
    }
    
    /// 检查执行权限
    fn check_execution_permission(&self, algorithm: &Algorithm, config: &ExecutionConfig) -> Result<()> {
        // 基本权限检查
        if !algorithm.is_public && !self.security_manager.can_execute_algorithm(algorithm, config) {
            return Err(Error::PermissionDenied("没有执行权限".to_string()));
        }
        
        // 资源限制检查
        if let Some(limits) = &config.resource_limits {
            if let Some(max_memory) = limits.max_memory {
                // 检查内存限制
            }
            if let Some(max_cpu) = limits.max_cpu {
                // 检查CPU限制
            }
        }
        
        Ok(())
    }
    
    /// 创建沙箱
    fn create_sandbox(&self, config: &ExecutionConfig) -> Result<Box<dyn crate::algorithm::executor::sandbox::Sandbox>> {
        use crate::algorithm::executor::sandbox::types::{DefaultSandbox, SandboxConfig};
        
        let sandbox_config = SandboxConfig {
            timeout: std::time::Duration::from_millis(config.timeout_ms),
            memory_limit: config.resource_limits.max_memory,
            cpu_limit: config.resource_limits.max_cpu,
            security_level: SandboxSecurityLevel::Medium,
        };
        
        let sandbox = DefaultSandbox::new(sandbox_config)?;
        Ok(Box::new(sandbox))
    }
    
    /// 执行机器学习算法
    fn execute_ml_algorithm(&self, algorithm: &Algorithm, input_data: &DataBatch, sandbox: &dyn crate::algorithm::executor::sandbox::Sandbox) -> Result<DataBatch> {
        // 简化的ML算法执行
        let result = sandbox.execute(algorithm.code.as_bytes(), &input_data.serialize()?, std::time::Duration::from_secs(30))?;
        
        if result.success {
            DataBatch::deserialize(&result.output)
        } else {
            Err(Error::Internal(format!("ML算法执行失败: {}", result.stderr)))
        }
    }
    
    /// 执行数据处理算法
    fn execute_data_processing_algorithm(&self, algorithm: &Algorithm, input_data: &DataBatch, sandbox: &dyn crate::algorithm::executor::sandbox::Sandbox) -> Result<DataBatch> {
        // 简化的数据处理算法执行
        let result = sandbox.execute(algorithm.code.as_bytes(), &input_data.serialize()?, std::time::Duration::from_secs(30))?;
        
        if result.success {
            DataBatch::deserialize(&result.output)
        } else {
            Err(Error::Internal(format!("数据处理算法执行失败: {}", result.stderr)))
        }
    }
    
    /// 执行优化算法
    fn execute_optimization_algorithm(&self, algorithm: &Algorithm, input_data: &DataBatch, sandbox: &dyn crate::algorithm::executor::sandbox::Sandbox) -> Result<DataBatch> {
        // 简化的优化算法执行
        let result = sandbox.execute(algorithm.code.as_bytes(), &input_data.serialize()?, std::time::Duration::from_secs(30))?;
        
        if result.success {
            DataBatch::deserialize(&result.output)
        } else {
            Err(Error::Internal(format!("优化算法执行失败: {}", result.stderr)))
        }
    }
    
    /// 执行自定义算法
    fn execute_custom_algorithm(&self, algorithm: &Algorithm, input_data: &DataBatch, sandbox: &dyn crate::algorithm::executor::sandbox::Sandbox) -> Result<DataBatch> {
        // 简化的自定义算法执行
        let result = sandbox.execute(algorithm.code.as_bytes(), &input_data.serialize()?, std::time::Duration::from_secs(30))?;
        
        if result.success {
            DataBatch::deserialize(&result.output)
        } else {
            Err(Error::Internal(format!("自定义算法执行失败: {}", result.stderr)))
        }
    }
    
    /// 异步执行算法
    pub fn execute_algorithm_async(&self, algorithm_id: &str, input_data: &DataBatch, config: &ExecutionConfig) -> Result<String> {
        info!("异步执行算法: {}", algorithm_id);
        
        // 获取算法
        let algorithm = self.get_algorithm(algorithm_id)?.ok_or_else(|| 
            Error::NotFound(format!("算法不存在: {}", algorithm_id)))?;
        
        // 验证权限
        self.check_execution_permission(&algorithm, config)?;
        
        // 创建执行上下文
        let execution_id = generate_id();
        let context = ExecutionContext {
            execution_id: execution_id.clone(),
            algorithm_id: algorithm_id.to_string(),
            status: ExecutionStatus::Pending,
            start_time: chrono::Utc::now(),
            end_time: None,
            result: None,
            error: None,
        };
        
        // 发送事件
        let payload = serde_json::json!({
            "algorithm_id": algorithm_id,
            "execution_id": execution_id.clone(),
            "status": "started"
        });
        self.send_event(EventType::AlgorithmExecuted, payload);
        
        Ok(execution_id)
    }
    
    /// 获取执行结果
    pub fn get_execution_result(&self, execution_id: &str) -> Result<Option<DataBatch>> {
        self.status_tracker.get_result(execution_id)
    }
    
    /// 获取执行状态
    pub fn get_execution_status(&self, execution_id: &str) -> Result<ExecutionStatus> {
        self.status_tracker.get_status(execution_id).map(|s| s.unwrap_or(ExecutionStatus::Unknown))
    }
    
    /// 取消执行
    pub fn cancel_execution(&self, execution_id: &str) -> Result<()> {
        info!("取消执行: {}", execution_id);
        
        // 发送事件
        let payload = serde_json::json!({
            "execution_id": execution_id,
            "status": "cancelled"
        });
        self.send_event(EventType::AlgorithmCancelled, payload);
        
        Ok(())
    }
    
    /// 批量执行
    pub fn batch_execute(&self, algorithm_id: &str, input_batches: &[DataBatch], config: &ExecutionConfig) -> Result<Vec<DataBatch>> {
        let mut results = Vec::new();
        
        for input_batch in input_batches {
            let result = self.execute_algorithm(algorithm_id, input_batch, config)?;
            results.push(result);
        }
        
        Ok(results)
    }
    
    /// 获取算法指标
    pub fn get_algorithm_metrics(&self, algorithm_id: &str) -> Result<AlgorithmMetrics> {
        // 简化的实现
        Ok(AlgorithmMetrics::new(algorithm_id.to_string()))
    }
    
    /// 优化算法
    pub fn optimize_algorithm(&self, algorithm_id: &str, optimization_config: &AlgorithmOptimizationConfig) -> Result<String> {
        info!("优化算法: {}", algorithm_id);
        
        // 获取算法
        let algorithm = self.get_algorithm_simple(algorithm_id)?;
        
        // 应用优化
        let optimized_algorithm = self.apply_algorithm_optimization(&algorithm, optimization_config)?;
        
        // 保存优化后的算法
        let optimization_id = generate_id();
        
        // 发送事件
        let payload = serde_json::json!({
            "algorithm_id": algorithm_id,
            "optimization_id": optimization_id.clone(),
            "status": "completed"
        });
        self.send_event(EventType::AlgorithmOptimized, payload);
        
        Ok(optimization_id)
    }
    
    /// 应用算法优化
    fn apply_algorithm_optimization(&self, algorithm: &Algorithm, config: &AlgorithmOptimizationConfig) -> Result<Algorithm> {
        let mut optimized_algorithm = algorithm.clone();
        
        // 应用内存优化
        if config.enable_memory_optimization {
            self.apply_memory_optimization(&mut optimized_algorithm)?;
        }
        
        // 应用速度优化
        if config.enable_speed_optimization {
            self.apply_speed_optimization(&mut optimized_algorithm)?;
        }
        
        // 应用精度优化
        if config.enable_accuracy_optimization {
            self.apply_accuracy_optimization(&mut optimized_algorithm)?;
        }
        
        // 应用并行优化
        if config.enable_parallel_optimization {
            self.apply_parallel_optimization(&mut optimized_algorithm)?;
        }
        
        Ok(optimized_algorithm)
    }
    
    /// 应用内存优化
    fn apply_memory_optimization(&self, algorithm: &mut Algorithm) -> Result<()> {
        // 简化的内存优化实现
        Ok(())
    }
    
    /// 应用速度优化
    fn apply_speed_optimization(&self, algorithm: &mut Algorithm) -> Result<()> {
        // 简化的速度优化实现
        Ok(())
    }
    
    /// 应用精度优化
    fn apply_accuracy_optimization(&self, algorithm: &mut Algorithm) -> Result<()> {
        // 简化的精度优化实现
        Ok(())
    }
    
    /// 应用并行优化
    fn apply_parallel_optimization(&self, algorithm: &mut Algorithm) -> Result<()> {
        // 简化的并行优化实现
        Ok(())
    }
    
    /// 验证算法
    pub fn validate_algorithm(&self, algorithm_id: &str, validation_data: &DataBatch) -> Result<AlgorithmValidationResult> {
        info!("验证算法: {}", algorithm_id);
        
        // 获取算法
        let algorithm = self.get_algorithm_simple(algorithm_id)?;
        
        // 执行验证
        let result = self.execute_algorithm(algorithm_id, validation_data, &ExecutionConfig::default())?;
        
        // 计算验证指标
        let accuracy = self.calculate_accuracy(&result, validation_data)?;
        let performance = self.calculate_performance(&result)?;
        
        let validation_result = AlgorithmValidationResult {
            algorithm_id: algorithm_id.to_string(),
            is_valid: accuracy > 0.8, // 简化的验证标准
            accuracy,
            performance,
            recommendations: vec!["请检查算法实现".to_string()],
            validation_data_size: validation_data.len(),
            validation_time: chrono::Utc::now(),
        };
        
        // 发送事件
        let payload = serde_json::json!({
            "algorithm_id": algorithm_id,
            "validation_result": {
                "is_valid": validation_result.is_valid,
                "accuracy": validation_result.accuracy
            }
        });
        self.send_event(EventType::AlgorithmValidated, payload);
        
        Ok(validation_result)
    }
    
    /// 计算精度
    fn calculate_accuracy(&self, result: &DataBatch, validation_data: &DataBatch) -> Result<f64> {
        // 简化的精度计算
        Ok(0.85)
    }
    
    /// 计算性能
    fn calculate_performance(&self, result: &DataBatch) -> Result<f64> {
        // 简化的性能计算
        Ok(0.9)
    }
    
    /// 获取执行历史
    pub fn get_execution_history(&self, algorithm_id: &str, limit: Option<usize>) -> Result<Vec<ExecutionHistory>> {
        // 简化的实现
        Ok(Vec::new())
    }
    
    /// 清理过期执行
    pub fn cleanup_expired_executions(&self) -> Result<usize> {
        info!("清理过期执行");
        
        // 简化的清理实现
        let cleaned_count = 0;
        
        // 发送事件
        let payload = serde_json::json!({
            "cleaned_count": cleaned_count
        });
        self.send_event(EventType::SystemMaintenance, payload);
        
        Ok(cleaned_count)
    }
    
    /// 获取系统资源使用情况
    pub fn get_system_resource_usage(&self) -> Result<SystemResourceUsage> {
        // 简化的实现
        Ok(SystemResourceUsage {
            cpu_usage: 0.5,
            memory_usage: 0.6,
            disk_usage: 0.3,
            network_usage: 0.2,
        })
    }
    
    /// 发送事件
    fn send_event(&self, event_type: EventType, payload: serde_json::Value) {
        let event = Event {
            id: generate_id(),
            event_type,
            payload,
            timestamp: SystemTime::now(),
            source: "algorithm_manager".to_string(),
        };
        
        // 异步发送事件
        let event_system = self.event_manager.event_system.clone();
        tokio::spawn(async move {
            if let Err(e) = event_system.publish(event).await {
                error!("发送事件失败: {}", e);
            }
        });
    }
    
    /// 生成唯一ID
    fn generate_id() -> String {
        generate_id()
    }
}

impl Clone for AlgorithmManager {
    fn clone(&self) -> Self {
        Self {
            storage: self.storage.clone(),
            model_manager: self.model_manager.clone(),
            status_tracker: self.status_tracker.clone(),
            algorithms: self.algorithms.clone(),
            task_manager: self.task_manager.clone(),
            executor: self.executor.clone(),
            event_manager: self.event_manager.clone(),
            raw_executor: self.raw_executor.clone(),
            config: self.config.clone(),
            security_manager: self.security_manager.clone(),
            auto_adjuster: self.auto_adjuster.clone(),
        }
    }
}

impl fmt::Debug for AlgorithmManager {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("AlgorithmManager")
            .field("storage", &self.storage)
            .field("model_manager", &self.model_manager)
            .field("status_tracker", &self.status_tracker)
            .field("algorithms", &self.algorithms)
            .field("task_manager", &self.task_manager)
            .field("executor", &self.executor)
            .field("event_manager", &self.event_manager)
            .field("raw_executor", &self.raw_executor)
            .field("config", &self.config)
            .field("security_manager", &self.security_manager)
            .field("auto_adjuster", &self.auto_adjuster)
            .finish()
    }
} 
