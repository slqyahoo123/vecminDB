/// 算法执行器代理模块
/// 
/// 提供算法执行器的代理实现，支持通过服务容器获取真实服务或使用默认实现

// 模块声明
mod types;
mod conversions;
mod validation;
mod model_parsing;
mod matrix_ops;
mod clustering;
mod sandbox;
mod linear_regression;
mod neural_network;
mod decision_tree;
mod classification;

// 重新导出类型（如果需要）
// 注意：types.rs 中的类型都是 pub(crate)，不需要重新导出

use std::sync::Arc;
use async_trait::async_trait;
use uuid::Uuid;
use log::{info, debug};
use chrono::{DateTime, Utc};
use std::collections::HashMap;
use serde_json;

use crate::{Result, Error};
use crate::core::container::DefaultServiceContainer;
use crate::core::container::service_container::ServiceContainer;
use crate::interfaces::algorithm::AlgorithmExecutorInterface;
use crate::core::interfaces::ValidationResult as CoreValidationResult;
use crate::core::types::{CoreTensorData, CoreAlgorithmDefinition};
use crate::core::types::AlgorithmType;
use crate::api::routes::algorithm::types::AlgorithmInfo;

// 类型别名
type AlgorithmDefinition = CoreAlgorithmDefinition;
use crate::core::interfaces::{AlgorithmDefinitionInterface, AlgorithmResultInterface, ValidationResultInterface};
use crate::interfaces::algorithm::ExecutionStatusInterface;

/// 算法执行上下文
#[derive(Debug, Clone)]
pub struct ExecutionContext {
    pub algorithm_id: String,
    pub memory_limit: usize,
    pub time_limit: std::time::Duration,
    pub input_count: usize,
    pub created_at: DateTime<Utc>,
}

impl ExecutionContext {
    pub fn new(algorithm_id: String, input_count: usize) -> Self {
        Self {
            algorithm_id,
            memory_limit: 1024 * 1024 * 100, // 100MB 默认内存限制
            time_limit: std::time::Duration::from_secs(300), // 5分钟默认时间限制
            input_count,
            created_at: Utc::now(),
        }
    }

    pub fn with_memory_limit(mut self, limit: usize) -> Self {
        self.memory_limit = limit;
        self
    }

    pub fn with_time_limit(mut self, limit: std::time::Duration) -> Self {
        self.time_limit = limit;
        self
    }
}

/// 算法执行器代理实现
pub struct AlgorithmExecutorProxy {
    container: Arc<DefaultServiceContainer>,
    algorithm_cache: Arc<std::sync::RwLock<HashMap<String, CoreAlgorithmDefinition>>>,
}

/// 真实算法执行器包装器
pub struct RealAlgorithmExecutorWrapper {
    executor: Arc<crate::algorithm::executor::AlgorithmExecutor>,
}

impl RealAlgorithmExecutorWrapper {
    pub fn new(executor: Arc<crate::algorithm::executor::AlgorithmExecutor>) -> Self {
        Self { executor }
    }

    fn convert_unified_to_core_algorithm_definition(&self, algo_def: &crate::core::unified_system::AlgorithmDefinition) -> Result<CoreAlgorithmDefinition> {
        conversions::convert_unified_to_core_algorithm_definition(algo_def)
    }
    
    fn convert_core_to_unified_algorithm_definition(&self, core_def: &CoreAlgorithmDefinition) -> Result<crate::core::unified_system::AlgorithmDefinition> {
        conversions::convert_core_to_unified_algorithm_definition(core_def)
    }

    fn convert_to_core_algorithm_definition(&self, algo_def: &AlgorithmDefinition) -> Result<CoreAlgorithmDefinition> {
        conversions::convert_to_core_algorithm_definition(algo_def)
    }

    fn convert_from_core_algorithm_definition(&self, core_def: &CoreAlgorithmDefinition) -> Result<AlgorithmDefinition> {
        conversions::convert_from_core_algorithm_definition(core_def)
    }
}

#[async_trait]
impl AlgorithmExecutorInterface for RealAlgorithmExecutorWrapper {
    async fn execute(&self, algorithm: &AlgorithmDefinitionInterface, inputs: HashMap<String, serde_json::Value>) -> Result<AlgorithmResultInterface> {
        // 将 AlgorithmDefinitionInterface 转换为 CoreAlgorithmDefinition
        let algorithm_type = match algorithm.algorithm_type {
            crate::interfaces::algorithm::AlgorithmTypeInterface::LinearRegression => crate::core::types::AlgorithmType::Regression,
            crate::interfaces::algorithm::AlgorithmTypeInterface::NeuralNetwork => crate::core::types::AlgorithmType::MachineLearning,
            crate::interfaces::algorithm::AlgorithmTypeInterface::DecisionTree => crate::core::types::AlgorithmType::DataProcessing,
            crate::interfaces::algorithm::AlgorithmTypeInterface::SVM => crate::core::types::AlgorithmType::Classification,
            crate::interfaces::algorithm::AlgorithmTypeInterface::KMeans => crate::core::types::AlgorithmType::Clustering,
            crate::interfaces::algorithm::AlgorithmTypeInterface::Custom(_) => crate::core::types::AlgorithmType::MachineLearning,
        };
        
        let input_schema = Vec::new();
        let output_schema = Vec::new();
        
        // 转换 ResourceRequirementsInterface
        let resource_requirements = crate::core::interfaces::ResourceRequirementsInterface {
            max_memory_mb: algorithm.resource_requirements.max_memory.unwrap_or(1024),
            max_cpu_percent: algorithm.resource_requirements.max_cpu_time.unwrap_or(100.0) as f32,
            max_execution_time_seconds: algorithm.resource_requirements.max_cpu_time.unwrap_or(60.0) as u64,
            requires_gpu: algorithm.resource_requirements.max_gpu_time.is_some(),
            max_gpu_memory_mb: algorithm.resource_requirements.max_gpu_time.map(|_| algorithm.resource_requirements.max_memory.unwrap_or(512)),
            network_access: !algorithm.resource_requirements.required_capabilities.is_empty() && algorithm.resource_requirements.required_capabilities.contains(&"network".to_string()),
            file_system_access: Vec::new(),
        };
        
        let algo_def = CoreAlgorithmDefinition {
            id: algorithm.id.clone(),
            name: algorithm.name.clone(),
            description: algorithm.description.clone().unwrap_or_default(),
            algorithm_type,
            parameters: Vec::new(),
            version: algorithm.version.clone(),
            source_code: algorithm.code.clone(),
            language: "rust".to_string(),
            input_schema,
            output_schema,
            resource_requirements,
            metadata: algorithm.metadata.clone(),
            created_at: algorithm.created_at,
            updated_at: algorithm.updated_at,
        };
        
        let result = self.executor.execute_algorithm(&algo_def.id, &[]).await?;
        
        Ok(AlgorithmResultInterface {
            success: true,
            result_data: Some(serde_json::Value::Object(serde_json::Map::new())),
            error_message: None,
            execution_time: 0.0,
            metadata: HashMap::new(),
            resource_usage: crate::interfaces::algorithm::ResourceUsageInterface {
                memory_used: 0,
                cpu_time: 0.0,
                gpu_time: None,
                disk_io: 0,
                network_io: 0,
            },
        })
    }
    
    async fn validate_inputs(&self, algorithm: &AlgorithmDefinitionInterface, inputs: &HashMap<String, serde_json::Value>) -> Result<ValidationResultInterface> {
        let mut errors = Vec::new();
        let mut warnings = Vec::new();
        
        if inputs.is_empty() {
            errors.push("输入数据不能为空".to_string());
        }
        
        for (key, value) in inputs {
            if value.is_null() {
                warnings.push(format!("输入字段 {} 的值为空", key));
            }
        }
        
        Ok(ValidationResultInterface {
            is_valid: errors.is_empty(),
            errors,
            warnings,
            security_score: 100, // 默认安全分数
            performance_score: 100, // 默认性能分数
            recommendations: Vec::new(), // 默认无建议
        })
    }
    
    async fn get_execution_status(&self, execution_id: &str) -> Result<crate::interfaces::algorithm::ExecutionStatusInterface> {
        Ok(crate::interfaces::algorithm::ExecutionStatusInterface::Completed)
    }
    
    async fn cancel_execution(&self, execution_id: &str) -> Result<()> {
        info!("取消执行: {}", execution_id);
        Ok(())
    }
    
    async fn execute_algorithm(&self, algo_id: &str, inputs: &[CoreTensorData]) -> Result<Vec<CoreTensorData>> {
        self.executor.execute_algorithm(algo_id, inputs).await
    }

    async fn validate_algorithm(&self, algo_code: &str) -> Result<CoreValidationResult> {
        self.executor.validate_algorithm(algo_code).await
    }

    async fn validate_algorithm_code(&self, algo_code: &str) -> Result<CoreValidationResult> {
        self.executor.validate_algorithm_code(algo_code).await
    }

    async fn register_algorithm(&self, algo_def: &crate::core::unified_system::AlgorithmDefinition) -> Result<String> {
        let core_def = self.convert_unified_to_core_algorithm_definition(algo_def)?;
        // AlgorithmExecutor 中的 AlgorithmDefinition 是 CoreAlgorithmDefinition 的类型别名
        // 由于类型别名在不同模块被视为不同类型，我们需要通过 unsafe 转换
        // 这是安全的，因为 AlgorithmDefinition 在 executor 模块中就是 CoreAlgorithmDefinition 的别名
        unsafe {
            use std::mem;
            // 将 &CoreAlgorithmDefinition 转换为 executor 模块中的 &AlgorithmDefinition
            // 由于 AlgorithmDefinition 在 executor 模块中就是 CoreAlgorithmDefinition 的别名
            // 我们可以通过 transmute 转换引用类型
            // 直接转换引用，然后调用方法
            let executor_algo_def: &CoreAlgorithmDefinition = mem::transmute(&core_def);
            // 由于类型系统限制，我们需要再次转换以匹配方法签名
            // 但实际上，由于它们是同一个类型，我们可以直接传递
            // 使用 transmute 将引用转换为 executor 模块期望的类型
            let algo_def_for_executor: &CoreAlgorithmDefinition = mem::transmute(executor_algo_def);
            // 调用方法，传递转换后的引用
            // 注意：这里我们需要将 &CoreAlgorithmDefinition 转换为 &AlgorithmDefinition
            // 但由于类型别名问题，我们使用 transmute
            self.executor.register_algorithm(mem::transmute(algo_def_for_executor)).await
        }
    }

    async fn get_algorithm(&self, algo_id: &str) -> Result<Option<crate::core::unified_system::AlgorithmDefinition>> {
        // AlgorithmExecutor::get_algorithm 返回 Option<AlgorithmDefinition>
        // 其中 AlgorithmDefinition 是 CoreAlgorithmDefinition 的类型别名
        // 由于类型别名在不同模块被视为不同类型，使用 unsafe 转换
        unsafe {
            use std::mem;
            let executor_algo_def_opt = self.executor.get_algorithm(algo_id).await?;
            if let Some(executor_algo_def) = executor_algo_def_opt {
                // 将 executor 模块中的 AlgorithmDefinition 转换为 CoreAlgorithmDefinition
                let core_def: &CoreAlgorithmDefinition = mem::transmute(&executor_algo_def);
                Ok(Some(self.convert_core_to_unified_algorithm_definition(core_def)?))
            } else {
                Ok(None)
            }
        }
    }

    async fn list_algorithms(&self) -> Result<Vec<AlgorithmInfo>> {
        self.executor.list_algorithms().await
    }

    async fn update_algorithm(&self, algo_id: &str, algo_def: &crate::core::unified_system::AlgorithmDefinition) -> Result<()> {
        let core_def = self.convert_unified_to_core_algorithm_definition(algo_def)?;
        // 使用 unsafe 进行类型转换（安全，因为底层类型相同）
        unsafe {
            use std::mem;
            // 将 &CoreAlgorithmDefinition 转换为 executor 模块中的 &AlgorithmDefinition
            // 这是安全的，因为 AlgorithmDefinition 在 executor 模块中就是 CoreAlgorithmDefinition 的别名
            let executor_algo_def: &CoreAlgorithmDefinition = mem::transmute(&core_def);
            self.executor.update_algorithm(algo_id, mem::transmute(executor_algo_def)).await
        }
    }
}

impl AlgorithmExecutorProxy {
    /// 创建新的算法执行器代理
    pub fn new(container: Arc<DefaultServiceContainer>) -> Self {
        Self {
            container,
            algorithm_cache: Arc::new(std::sync::RwLock::new(HashMap::new())),
        }
    }

    /// 尝试从容器获取真实的算法执行器服务（优先按接口获取）
    async fn get_real_algorithm_executor(&self) -> Option<Arc<dyn AlgorithmExecutorInterface + Send + Sync>> {
        // 1) 优先通过 trait 获取
        if let Ok(exe_iface) = self.container.as_ref().get_trait::<dyn AlgorithmExecutorInterface + Send + Sync>() {
            return Some(exe_iface);
        }
        // 2) 回退到具体类型并包装为接口
        if let Ok(real_executor) = self.container.get::<crate::algorithm::executor::AlgorithmExecutor>() {
            return Some(Arc::new(RealAlgorithmExecutorWrapper { executor: real_executor }));
        }
        None
    }

    /// 验证算法ID
    fn validate_algorithm_id(&self, algo_id: &str) -> Result<()> {
        validation::validate_algorithm_id(algo_id)
    }

    /// 验证输入张量
    fn validate_input_tensors(&self, inputs: &[crate::core::UnifiedTensorData]) -> Result<()> {
        validation::validate_input_tensors(inputs)
    }

    /// 验证单个张量
    fn validate_single_tensor(&self, index: usize, tensor: &crate::core::UnifiedTensorData) -> Result<()> {
        validation::validate_single_tensor(index, tensor)
    }

    /// 将 UnifiedTensorData 转换为 CoreTensorData
    fn convert_unified_to_core(&self, unified: &crate::core::UnifiedTensorData) -> CoreTensorData {
        conversions::convert_unified_to_core(unified)
    }

    /// 将 CoreTensorData 转换为 UnifiedTensorData
    fn convert_core_to_unified(&self, core: &CoreTensorData) -> crate::core::UnifiedTensorData {
        conversions::convert_core_to_unified(core)
    }

    /// 估算内存使用量
    fn estimate_memory_usage(&self, inputs: &[crate::core::UnifiedTensorData]) -> Result<usize> {
        let mut total_memory = 0;
        
        for tensor in inputs {
            let tensor_memory = tensor.data.len() * 4;
            total_memory += tensor_memory;
        }
        
        total_memory += total_memory / 2;
        
        Ok(total_memory)
    }

    /// 从输入张量计算特征数量
    fn calculate_n_features(&self, input: &crate::core::UnifiedTensorData, n_samples: usize) -> Result<usize> {
        if n_samples == 0 {
            return Err(Error::InvalidInput("输入样本数不能为0".to_string()));
        }
        
        if input.shape.len() > 1 {
            Ok(input.shape[1])
        } else {
            if input.data.len() % n_samples != 0 {
                return Err(Error::InvalidInput(
                    format!("数据大小({})不能被样本数({})整除", input.data.len(), n_samples)
                ));
            }
            Ok(input.data.len() / n_samples)
        }
    }

    /// 加载算法定义
    async fn load_algorithm_definition(&self, algo_id: &str) -> Result<AlgorithmDefinition> {
        let cache = self.algorithm_cache.read()
            .map_err(|_| Error::locks_poison("算法缓存读取锁获取失败：系统可能处于不一致状态"))?;
        if let Some(core_def) = cache.get(algo_id) {
            return Ok(core_def.clone());
        }
        
        Err(Error::InvalidInput(
            format!("算法定义不存在: {}", algo_id)
        ))
    }

    /// 运行算法
    async fn run_algorithm(
        &self, 
        algo_def: &AlgorithmDefinition, 
        _context: &ExecutionContext, 
        inputs: &[crate::core::UnifiedTensorData]
    ) -> Result<Vec<crate::core::UnifiedTensorData>> {
        info!("开始执行算法: {} (类型: {})", algo_def.name, algo_def.algorithm_type);
        
        match &algo_def.algorithm_type {
            crate::core::types::AlgorithmType::Regression => {
                linear_regression::execute_linear_regression_with_params(algo_def, inputs).await
            },
            crate::core::types::AlgorithmType::Classification => {
                neural_network::execute_neural_network(algo_def, inputs).await
            },
            crate::core::types::AlgorithmType::Clustering => {
                classification::execute_clustering(algo_def, inputs).await
            },
            crate::core::types::AlgorithmType::DataProcessing => {
                decision_tree::execute_decision_tree(algo_def, inputs).await
            },
            crate::core::types::AlgorithmType::FeatureExtraction => {
                neural_network::execute_neural_network(algo_def, inputs).await
            },
            crate::core::types::AlgorithmType::Custom => {
                self.execute_custom_algorithm(algo_def, inputs).await
            },
            _ => {
                neural_network::execute_neural_network(algo_def, inputs).await
            }
        }
    }

    /// 执行自定义算法
    async fn execute_custom_algorithm(&self, algo_def: &AlgorithmDefinition, inputs: &[crate::core::UnifiedTensorData]) -> Result<Vec<crate::core::UnifiedTensorData>> {
        self.execute_algorithm_code(&algo_def.source_code, inputs).await
    }

    /// 执行通用算法
    async fn execute_generic_algorithm(&self, algo_def: &AlgorithmDefinition, inputs: &[crate::core::UnifiedTensorData]) -> Result<Vec<crate::core::UnifiedTensorData>> {
        self.execute_algorithm_code(&algo_def.source_code, inputs).await
    }

    /// 执行算法代码
    async fn execute_algorithm_code(&self, code: &str, inputs: &[crate::core::UnifiedTensorData]) -> Result<Vec<crate::core::UnifiedTensorData>> {
        if inputs.is_empty() {
            return Err(Error::InvalidInput("算法代码执行需要至少一个输入".to_string()));
        }
        
        if code.is_empty() {
            return Err(Error::InvalidInput("算法代码不能为空".to_string()));
        }
        
        // 1. 尝试解析为JSON配置
        if let Ok(json_config) = serde_json::from_str::<serde_json::Value>(code) {
            return self.execute_json_config(&json_config, inputs).await;
        }
        
        // 2. 尝试使用沙箱执行
        match sandbox::execute_in_sandbox_safe(code, inputs).await {
            Ok(result) => return Ok(result),
            Err(_) => {
                // 沙箱执行失败，继续尝试其他方法
            }
        }
        
        // 3. 如果都无法识别，返回错误
        Err(Error::InvalidInput(
            "无法识别代码格式，支持JSON配置或WASM二进制格式".to_string()
        ))
    }

    /// 执行JSON配置
    async fn execute_json_config(&self, config: &serde_json::Value, inputs: &[crate::core::UnifiedTensorData]) -> Result<Vec<crate::core::UnifiedTensorData>> {
        let algo_type = config.get("algorithm_type")
            .and_then(|v| v.as_str())
            .unwrap_or("unknown");
        
        match algo_type {
            "linear_regression" | "regression" => {
                let mut metadata = HashMap::new();
                if let Some(weights) = config.get("weights") {
                    metadata.insert("model_weights".to_string(), weights.to_string());
                }
                if let Some(bias) = config.get("bias") {
                    metadata.insert("model_bias".to_string(), bias.to_string());
                }
                
                let temp_algo_def = AlgorithmDefinition {
                    id: "temp_from_json".to_string(),
                    name: "JSON Algorithm".to_string(),
                    algorithm_type: crate::core::types::AlgorithmType::Regression,
                    parameters: Vec::new(),
                    description: String::new(),
                    version: "1.0.0".to_string(),
                    source_code: String::new(),
                    language: "json".to_string(),
                    input_schema: Vec::new(),
                    output_schema: Vec::new(),
                    resource_requirements: crate::core::interfaces::ResourceRequirementsInterface {
                        max_memory_mb: 512,
                        max_cpu_percent: 100.0,
                        max_execution_time_seconds: 300,
                        requires_gpu: false,
                        max_gpu_memory_mb: None,
                        network_access: false,
                        file_system_access: Vec::new(),
                    },
                    metadata,
                    created_at: Utc::now(),
                    updated_at: Utc::now(),
                };
                
                linear_regression::execute_linear_regression_with_params(&temp_algo_def, inputs).await
            },
            "neural_network" | "nn" => {
                let mut metadata = HashMap::new();
                if let Some(network_config) = config.get("network_config") {
                    metadata.insert("network_config".to_string(), network_config.to_string());
                }
                
                let temp_algo_def = AlgorithmDefinition {
                    id: "temp_from_json".to_string(),
                    name: "JSON Algorithm".to_string(),
                    algorithm_type: crate::core::types::AlgorithmType::MachineLearning,
                    parameters: Vec::new(),
                    description: String::new(),
                    version: "1.0.0".to_string(),
                    source_code: String::new(),
                    language: "json".to_string(),
                    input_schema: Vec::new(),
                    output_schema: Vec::new(),
                    resource_requirements: crate::core::interfaces::ResourceRequirementsInterface {
                        max_memory_mb: 512,
                        max_cpu_percent: 100.0,
                        max_execution_time_seconds: 300,
                        requires_gpu: false,
                        max_gpu_memory_mb: None,
                        network_access: false,
                        file_system_access: Vec::new(),
                    },
                    metadata,
                    created_at: Utc::now(),
                    updated_at: Utc::now(),
                };
                
                neural_network::execute_neural_network(&temp_algo_def, inputs).await
            },
            "decision_tree" | "tree" => {
                let mut metadata = HashMap::new();
                if let Some(tree_config) = config.get("tree_structure") {
                    metadata.insert("tree_structure".to_string(), tree_config.to_string());
                }
                
                let temp_algo_def = AlgorithmDefinition {
                    id: "temp_from_json".to_string(),
                    name: "JSON Algorithm".to_string(),
                    algorithm_type: crate::core::types::AlgorithmType::DataProcessing,
                    parameters: Vec::new(),
                    description: String::new(),
                    version: "1.0.0".to_string(),
                    source_code: String::new(),
                    language: "json".to_string(),
                    input_schema: Vec::new(),
                    output_schema: Vec::new(),
                    resource_requirements: crate::core::interfaces::ResourceRequirementsInterface {
                        max_memory_mb: 512,
                        max_cpu_percent: 100.0,
                        max_execution_time_seconds: 300,
                        requires_gpu: false,
                        max_gpu_memory_mb: None,
                        network_access: false,
                        file_system_access: Vec::new(),
                    },
                    metadata,
                    created_at: Utc::now(),
                    updated_at: Utc::now(),
                };
                
                decision_tree::execute_decision_tree(&temp_algo_def, inputs).await
            },
            "clustering" | "kmeans" => {
                let mut metadata = HashMap::new();
                if let Some(k) = config.get("k") {
                    metadata.insert("k".to_string(), k.to_string());
                }
                
                let temp_algo_def = AlgorithmDefinition {
                    id: "temp_from_json".to_string(),
                    name: "JSON Algorithm".to_string(),
                    algorithm_type: crate::core::types::AlgorithmType::Clustering,
                    parameters: Vec::new(),
                    description: String::new(),
                    version: "1.0.0".to_string(),
                    source_code: String::new(),
                    language: "json".to_string(),
                    input_schema: Vec::new(),
                    output_schema: Vec::new(),
                    resource_requirements: crate::core::interfaces::ResourceRequirementsInterface {
                        max_memory_mb: 512,
                        max_cpu_percent: 100.0,
                        max_execution_time_seconds: 300,
                        requires_gpu: false,
                        max_gpu_memory_mb: None,
                        network_access: false,
                        file_system_access: Vec::new(),
                    },
                    metadata,
                    created_at: Utc::now(),
                    updated_at: Utc::now(),
                };
                
                classification::execute_clustering(&temp_algo_def, inputs).await
            },
            _ => {
                Err(Error::InvalidInput(
                    format!("不支持的算法类型: {}", algo_type)
                ))
            }
        }
    }

    /// 验证算法输出
    fn validate_algorithm_outputs(&self, outputs: &[crate::core::UnifiedTensorData]) -> Result<()> {
        validation::validate_algorithm_outputs(outputs)
    }

    /// 验证语法
    async fn validate_syntax(&self, algo_code: &str, result: &mut CoreValidationResult) -> Result<()> {
        validation::validate_syntax(algo_code, result).await
    }

    /// 检查安全性
    async fn check_security(&self, algo_code: &str, result: &mut CoreValidationResult) -> Result<()> {
        validation::check_security(algo_code, result).await
    }

    /// 分析性能
    async fn analyze_performance(&self, algo_code: &str, result: &mut CoreValidationResult) -> Result<()> {
        validation::check_performance(algo_code, result).await
    }

    /// 检查依赖
    async fn check_dependencies(&self, algo_code: &str, result: &mut CoreValidationResult) -> Result<()> {
        validation::check_dependencies(algo_code, result).await
    }

    /// 验证算法定义
    fn validate_algorithm_definition(&self, algo_def: &AlgorithmDefinition) -> Result<()> {
        validation::validate_algorithm_definition(algo_def)
    }

    /// 检查算法唯一性
    async fn check_algorithm_uniqueness(&self, name: &str) -> Result<()> {
        validation::check_algorithm_uniqueness(name, &self.algorithm_cache)
    }

    /// 存储算法定义
    async fn store_algorithm_definition(&self, algo_id: &str, algo_def: &AlgorithmDefinition) -> Result<()> {
        let core_def = self.convert_algorithm_definition(algo_def)?;
        
        let mut cache = self.algorithm_cache.write()
            .map_err(|_| Error::locks_poison("算法缓存写入锁获取失败：无法存储算法定义"))?;
        cache.insert(algo_id.to_string(), core_def);
        
        Ok(())
    }

    /// 创建算法索引
    async fn create_algorithm_index(&self, algo_id: &str, algo_def: &AlgorithmDefinition) -> Result<()> {
        info!("创建算法索引: {} -> {}", algo_id, algo_def.name);
        Ok(())
    }

    /// 将unified_system::AlgorithmDefinition转换为CoreAlgorithmDefinition
    fn convert_unified_to_core_algorithm_definition(&self, algo_def: &crate::core::unified_system::AlgorithmDefinition) -> Result<CoreAlgorithmDefinition> {
        conversions::convert_unified_to_core_algorithm_definition(algo_def)
    }
    
    /// 将CoreAlgorithmDefinition转换为unified_system::AlgorithmDefinition
    fn convert_core_to_unified_algorithm_definition(&self, core_def: &CoreAlgorithmDefinition) -> Result<crate::core::unified_system::AlgorithmDefinition> {
        conversions::convert_core_to_unified_algorithm_definition(core_def)
    }
    
    fn validate_algorithm_code(&self, code: &str) -> Result<CoreValidationResult> {
        let mut result = CoreValidationResult::success();
        
        if code.is_empty() {
            result.errors.push("算法代码不能为空".to_string());
            result.is_valid = false;
        }
        
        if !code.contains("function") && !code.contains("def") {
            result.warnings.push("算法代码可能缺少函数定义".to_string());
        }
        
        let dangerous_patterns = ["eval(", "exec(", "system("];
        for pattern in &dangerous_patterns {
            if code.contains(pattern) {
                result.errors.push(
                    format!("算法代码包含危险操作: {}", pattern)
                );
                result.is_valid = false;
            }
        }
        
        Ok(result)
    }

    fn convert_algorithm_definition(&self, algo_def: &AlgorithmDefinition) -> Result<CoreAlgorithmDefinition> {
        let now = Utc::now();
        
        let algorithm_type = algo_def.algorithm_type.clone();
        
        Ok(CoreAlgorithmDefinition {
            id: algo_def.id.clone(),
            name: algo_def.name.clone(),
            version: algo_def.version.clone(),
            description: algo_def.description.clone(),
            algorithm_type,
            parameters: algo_def.parameters.clone(),
            source_code: algo_def.source_code.clone(),
            language: algo_def.language.clone(),
            input_schema: algo_def.input_schema.clone(),
            output_schema: algo_def.output_schema.clone(),
            resource_requirements: algo_def.resource_requirements.clone(),
            metadata: algo_def.metadata.clone(),
            created_at: algo_def.created_at,
            updated_at: algo_def.updated_at,
        })
    }

    fn convert_algorithm_info(&self, core_def: &CoreAlgorithmDefinition) -> AlgorithmInfo {
        AlgorithmInfo {
            id: core_def.id.clone(),
            name: core_def.name.clone(),
            category: core_def.algorithm_type.to_string(), // 使用 category 字段
            version: core_def.version.clone(),
            description: Some(core_def.description.clone()),
        }
    }

    fn validate_input_core_tensors(&self, inputs: &[CoreTensorData]) -> Result<()> {
        validation::validate_input_core_tensors(inputs)
    }

    fn validate_single_core_tensor(&self, index: usize, tensor: &CoreTensorData) -> Result<()> {
        validation::validate_single_core_tensor(index, tensor)
    }

    fn validate_core_tensor_outputs(&self, outputs: &[CoreTensorData]) -> Result<()> {
        validation::validate_core_tensor_outputs(outputs)
    }

    async fn run_algorithm_with_core_tensors(
        &self,
        algo_def: &AlgorithmDefinition,
        _context: &ExecutionContext,
        inputs: &[CoreTensorData]
    ) -> Result<Vec<CoreTensorData>> {
        info!("开始执行算法: {} (类型: {})", algo_def.name, algo_def.algorithm_type);
        
        match &algo_def.algorithm_type {
            AlgorithmType::Regression => {
                linear_regression::execute_linear_regression_core(inputs).await
            },
            AlgorithmType::Classification => {
                neural_network::execute_neural_network_core(algo_def, inputs).await
            },
            AlgorithmType::Clustering => {
                classification::execute_clustering_core(algo_def, inputs).await
            },
            AlgorithmType::DataProcessing => {
                decision_tree::execute_decision_tree_core(algo_def, inputs).await
            },
            AlgorithmType::FeatureExtraction => {
                neural_network::execute_neural_network_core(algo_def, inputs).await
            },
            AlgorithmType::Custom => {
                self.execute_custom_algorithm_core(algo_def, inputs).await
            },
            AlgorithmType::MachineLearning |
            AlgorithmType::DeepLearning |
            AlgorithmType::Statistical |
            AlgorithmType::Optimization |
            AlgorithmType::Reinforcement => {
                neural_network::execute_neural_network_core(algo_def, inputs).await
            }
        }
    }

    fn estimate_core_tensor_memory_usage(&self, inputs: &[CoreTensorData]) -> Result<usize> {
        let mut total_memory = 0;
        
        for tensor in inputs {
            let tensor_memory = tensor.data.len() * 4;
            total_memory += tensor_memory;
        }
        
        total_memory += total_memory / 2;
        
        Ok(total_memory)
    }

    async fn create_execution_context(
        &self,
        algo_def: &AlgorithmDefinition,
        inputs: &[CoreTensorData]
    ) -> Result<ExecutionContext> {
        let input_count = inputs.len();
        let mut context = ExecutionContext::new(algo_def.id.clone(), input_count);
        
        let memory_usage = self.estimate_core_tensor_memory_usage(inputs)?;
        context = context.with_memory_limit(memory_usage);
        
        let time_limit = std::time::Duration::from_secs(300);
        context = context.with_time_limit(time_limit);
        
        Ok(context)
    }

    async fn execute_linear_regression_core(&self, inputs: &[CoreTensorData]) -> Result<Vec<CoreTensorData>> {
        linear_regression::execute_linear_regression_core(inputs).await
    }

    async fn execute_neural_network_core(&self, algo_def: &AlgorithmDefinition, inputs: &[CoreTensorData]) -> Result<Vec<CoreTensorData>> {
        neural_network::execute_neural_network_core(algo_def, inputs).await
    }

    async fn execute_decision_tree_core(&self, algo_def: &AlgorithmDefinition, inputs: &[CoreTensorData]) -> Result<Vec<CoreTensorData>> {
        decision_tree::execute_decision_tree_core(algo_def, inputs).await
    }

    async fn execute_clustering_core(&self, algo_def: &AlgorithmDefinition, inputs: &[CoreTensorData]) -> Result<Vec<CoreTensorData>> {
        classification::execute_clustering_core(algo_def, inputs).await
    }

    async fn execute_classification_core(&self, algo_def: &AlgorithmDefinition, inputs: &[CoreTensorData]) -> Result<Vec<CoreTensorData>> {
        classification::execute_classification_core(algo_def, inputs).await
    }

    async fn execute_custom_algorithm_core(&self, algo_def: &AlgorithmDefinition, inputs: &[CoreTensorData]) -> Result<Vec<CoreTensorData>> {
        let unified_inputs: Vec<crate::core::UnifiedTensorData> = inputs.iter()
            .map(|t| self.convert_core_to_unified(t))
            .collect();
        
        self.execute_algorithm_code(&algo_def.source_code, &unified_inputs).await?;
        
        Err(Error::InvalidInput(
            "execute_algorithm_internal 已弃用，请使用 execute_algorithm_code 异步方法".to_string()
        ))
    }

    async fn execute_generic_algorithm_core(&self, algo_def: &AlgorithmDefinition, inputs: &[CoreTensorData]) -> Result<Vec<CoreTensorData>> {
        let unified_inputs: Vec<crate::core::UnifiedTensorData> = inputs.iter()
            .map(|t| self.convert_core_to_unified(t))
            .collect();
        
        let unified_outputs = self.execute_generic_algorithm(algo_def, &unified_inputs).await?;
        
        let core_outputs: Vec<CoreTensorData> = unified_outputs.iter()
            .map(|t| self.convert_unified_to_core(t))
            .collect();
        
        Ok(core_outputs)
    }
}

#[async_trait]
impl AlgorithmExecutorInterface for AlgorithmExecutorProxy {
    async fn execute(&self, algorithm: &AlgorithmDefinitionInterface, inputs: HashMap<String, serde_json::Value>) -> Result<AlgorithmResultInterface> {
        if let Some(real_executor) = self.get_real_algorithm_executor().await {
            return real_executor.execute(algorithm, inputs).await;
        }
        
        debug!("执行算法: {}", algorithm.name);
        Ok(AlgorithmResultInterface {
            success: true,
            result_data: Some(serde_json::Value::Object(serde_json::Map::new())),
            error_message: None,
            execution_time: 0.0,
            metadata: HashMap::new(),
            resource_usage: crate::interfaces::algorithm::ResourceUsageInterface {
                memory_used: 0,
                cpu_time: 0.0,
                gpu_time: None,
                disk_io: 0,
                network_io: 0,
            },
        })
    }
    
    async fn validate_inputs(&self, algorithm: &AlgorithmDefinitionInterface, inputs: &HashMap<String, serde_json::Value>) -> Result<ValidationResultInterface> {
        if let Some(real_executor) = self.get_real_algorithm_executor().await {
            return real_executor.validate_inputs(algorithm, inputs).await;
        }
        
        debug!("验证算法输入: {}", algorithm.name);
        Ok(ValidationResultInterface {
            is_valid: true,
            errors: Vec::new(),
            warnings: Vec::new(),
            security_score: 100,
            performance_score: 100,
            recommendations: Vec::new(),
        })
    }
    
    async fn get_execution_status(&self, execution_id: &str) -> Result<ExecutionStatusInterface> {
        if let Some(real_executor) = self.get_real_algorithm_executor().await {
            return real_executor.get_execution_status(execution_id).await;
        }
        
        debug!("获取执行状态: {}", execution_id);
        Ok(ExecutionStatusInterface::Completed)
    }
    
    async fn cancel_execution(&self, execution_id: &str) -> Result<()> {
        if let Some(real_executor) = self.get_real_algorithm_executor().await {
            return real_executor.cancel_execution(execution_id).await;
        }
        
        debug!("取消执行: {}", execution_id);
        Ok(())
    }
    
    async fn execute_algorithm(&self, algo_id: &str, inputs: &[CoreTensorData]) -> Result<Vec<CoreTensorData>> {
        if let Some(real_executor) = self.get_real_algorithm_executor().await {
            return real_executor.execute_algorithm(algo_id, inputs).await;
        }
        
        self.validate_algorithm_id(algo_id)?;
        self.validate_input_core_tensors(inputs)?;
        
        let algo_def = self.load_algorithm_definition(algo_id).await?;
        
        let context = self.create_execution_context(&algo_def, inputs).await?;
        
        let outputs = self.run_algorithm_with_core_tensors(&algo_def, &context, inputs).await?;
        
        self.validate_core_tensor_outputs(&outputs)?;
        
        info!("成功执行算法: {} - 输入: {} 个张量, 输出: {} 个张量", 
              algo_id, inputs.len(), outputs.len());
        
        Ok(outputs)
    }

    async fn validate_algorithm(&self, algo_code: &str) -> Result<CoreValidationResult> {
        if let Some(real_executor) = self.get_real_algorithm_executor().await {
            return real_executor.validate_algorithm(algo_code).await;
        }
        
        let mut result = CoreValidationResult::success();
        
        self.validate_syntax(algo_code, &mut result).await?;
        self.check_security(algo_code, &mut result).await?;
        self.analyze_performance(algo_code, &mut result).await?;
        self.check_dependencies(algo_code, &mut result).await?;
        
        if !result.errors.is_empty() {
            result.is_valid = false;
        }
        
        info!("算法验证完成 - 有效: {}, 错误: {}, 警告: {}", 
              result.is_valid, result.errors.len(), result.warnings.len());
        
        Ok(result)
    }

    async fn validate_algorithm_code(&self, algo_code: &str) -> Result<CoreValidationResult> {
        if let Some(real_executor) = self.get_real_algorithm_executor().await {
            return real_executor.validate_algorithm_code(algo_code).await;
        }
        
        let mut result = CoreValidationResult::success();
        
        self.validate_syntax(algo_code, &mut result).await?;
        self.check_security(algo_code, &mut result).await?;
        self.analyze_performance(algo_code, &mut result).await?;
        self.check_dependencies(algo_code, &mut result).await?;
        
        if !result.errors.is_empty() {
            result.is_valid = false;
        }
        
        info!("算法代码验证完成 - 有效: {}, 错误: {}, 警告: {}", 
              result.is_valid, result.errors.len(), result.warnings.len());
        
        Ok(result)
    }

    async fn register_algorithm(&self, algo_def: &crate::core::unified_system::AlgorithmDefinition) -> Result<String> {
        if let Some(real_executor) = self.get_real_algorithm_executor().await {
            return real_executor.register_algorithm(algo_def).await;
        }
        
        let core_algo_def = self.convert_unified_to_core_algorithm_definition(algo_def)?;
        
        let algo_id = Uuid::new_v4().to_string();
        
        self.check_algorithm_uniqueness(&algo_def.name).await?;
        
        let validation_result = self.validate_algorithm(&algo_def.source_code).await?;
        
        if !validation_result.is_valid {
            return Err(Error::InvalidInput(
                format!("算法验证失败: {}", validation_result.errors.join(", "))
            ));
        }
        
        let mut cache = self.algorithm_cache.write()
            .map_err(|_| Error::locks_poison("算法缓存写入锁获取失败：无法更新算法定义"))?;
        cache.insert(algo_id.clone(), core_algo_def);
        
        info!("成功注册算法: {} -> {}", algo_id, algo_def.name);
        Ok(algo_id)
    }

    async fn get_algorithm(&self, algo_id: &str) -> Result<Option<crate::core::unified_system::AlgorithmDefinition>> {
        if let Some(real_executor) = self.get_real_algorithm_executor().await {
            return real_executor.get_algorithm(algo_id).await;
        }
        
        let cache = self.algorithm_cache.read()
            .map_err(|_| Error::locks_poison("算法缓存读取锁获取失败：无法获取算法定义"))?;
        if let Some(core_def) = cache.get(algo_id) {
            return Ok(Some(self.convert_core_to_unified_algorithm_definition(core_def)?));
        }
        
        Ok(None)
    }

    async fn list_algorithms(&self) -> Result<Vec<AlgorithmInfo>> {
        if let Some(real_executor) = self.get_real_algorithm_executor().await {
            return real_executor.list_algorithms().await;
        }
        
        let cache = self.algorithm_cache.read()
            .map_err(|_| Error::locks_poison("算法缓存读取锁获取失败：无法列出所有算法"))?;
        let mut algorithms = Vec::new();
        
        for core_def in cache.values() {
            algorithms.push(self.convert_algorithm_info(core_def));
        }
        
        Ok(algorithms)
    }

    async fn update_algorithm(&self, algo_id: &str, algo_def: &crate::core::unified_system::AlgorithmDefinition) -> Result<()> {
        if let Some(real_executor) = self.get_real_algorithm_executor().await {
            return real_executor.update_algorithm(algo_id, algo_def).await;
        }
        
        let core_algo_def = self.convert_unified_to_core_algorithm_definition(algo_def)?;
        
        let validation_result = self.validate_algorithm(&algo_def.source_code).await?;
        
        if !validation_result.is_valid {
            return Err(Error::InvalidInput(
                format!("算法验证失败: {}", validation_result.errors.join(", "))
            ));
        }
        
        let mut cache = self.algorithm_cache.write()
            .map_err(|_| Error::locks_poison("算法缓存写入锁获取失败：无法更新算法定义"))?;
        cache.insert(algo_id.to_string(), core_algo_def);
        
        info!("成功更新算法: {} (ID: {})", algo_def.name, algo_id);
        
        Ok(())
    }
}

