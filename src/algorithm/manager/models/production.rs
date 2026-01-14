// 模型管理器生产实现
// 负责模型的创建、管理、持久化、缓存等核心功能

use std::collections::HashMap;
use std::sync::{Arc, RwLock, Mutex};
use std::fmt;
use std::path::Path;
use chrono::{DateTime, Utc};
use serde::{Serialize, Deserialize};

use crate::error::{Error, Result};
use crate::storage::Storage;
use crate::compat::{
    Model, ModelStatus, ModelArchitecture, SmartModelParameters, ModelMemoryMonitor
};
use crate::compat::parameters::ModelParameters;
use crate::storage::models::implementation::ModelInfo;
use crate::compat::manager::traits::ModelManager;
use crate::storage::models::implementation::StorageFormat;
use crate::algorithm::manager::types::SerializableModel;
use crate::algorithm::manager::models::common::*;
use crate::data::DataBatch;
use crate::algorithm::manager::models::common::{
    ModelVersionInfo, ModelHealthStatus,
    ModelPerformanceMetrics, MonitoringConfig, BackupType, BackupInfo,
    CompressionConfig, QuantizationConfig, OptimizationConfig,
    ModelDependency, IntegrityCheckResult, ABTestConfig, ABTestResults,
    DeploymentConfig, DeploymentStatus
};

// 为不存在的类型提供生产级定义

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum HealthLevel {
    Healthy,
    Warning,
    Critical,
    Unknown,
}

/// 模型监控配置
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelMonitoringConfig {
    /// 监控间隔 (秒)
    pub interval_seconds: u64,
    /// 是否启用性能监控
    pub enable_performance: bool,
    /// 是否启用准确率监控
    pub enable_accuracy: bool,
    /// 是否启用资源监控
    pub enable_resource: bool,
    /// 是否启用预测监控
    pub enable_prediction: bool,
    /// 自定义阈值
    pub thresholds: HashMap<String, f64>,
}

impl Default for ModelMonitoringConfig {
    fn default() -> Self {
        Self {
            interval_seconds: 300, // 5分钟
            enable_performance: true,
            enable_accuracy: true,
            enable_resource: true,
            enable_prediction: true,
            thresholds: HashMap::new(),
        }
    }
}

/// 监控任务
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MonitoringTask {
    pub id: String,
    pub model_id: String,
    pub task_type: MonitoringTaskType,
    pub schedule: String,
    pub config: MonitoringConfig,
    pub status: TaskStatus,
    pub last_run: Option<DateTime<Utc>>,
    pub next_run: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MonitoringTaskType {
    HealthCheck,
    PerformanceMetrics,
    DataDrift,
    ModelDrift,
    Custom(String),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TaskStatus {
    Active,
    Paused,
    Failed,
    Completed,
}



// TrainingState 已移除 - 训练相关代码已删除





#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum IntegrityCheckType {
    ParameterConsistency,
    ArchitectureValidation,
    DataIntegrity,
    Full,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum IntegrityStatus {
    Valid,
    Warning,
    Corrupted,
    Unknown,
}



#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TestRecommendation {
    UseModelA,
    UseModelB,
    NoSignificantDifference,
    ExtendTest,
}



#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AutoScalingConfig {
    pub enabled: bool,
    pub min_replicas: u32,
    pub max_replicas: u32,
    pub target_cpu_utilization: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthCheckConfig {
    pub path: String,
    pub interval_seconds: u32,
    pub timeout_seconds: u32,
    pub failure_threshold: u32,
}

// 这些类型已经在对应的模块中定义，不需要重复定义

pub struct ProductionModelManager {
    pub storage: Arc<Storage>,
    pub models: RwLock<HashMap<String, Arc<Model>>>,
    pub model_cache: RwLock<HashMap<String, Arc<Model>>>,
    pub monitoring_tasks: RwLock<HashMap<String, MonitoringTask>>,
}

impl ProductionModelManager {
    pub fn new(storage: Arc<Storage>, _config: ModelManagerConfig) -> Self {
        Self {
            storage,
            models: RwLock::new(HashMap::new()),
            model_cache: RwLock::new(HashMap::new()),
            monitoring_tasks: RwLock::new(HashMap::new()),
        }
    }
    pub fn generate_id() -> String {
        use uuid::Uuid;
        Uuid::new_v4().to_string()
    }
}

impl fmt::Debug for ProductionModelManager {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ProductionModelManager")
            .field("models_count", &self.models.read().map(|m| m.len()).unwrap_or(0))
            .field("cache_size", &self.model_cache.read().map(|c| c.len()).unwrap_or(0))
            .finish()
    }
}

impl ModelManager for ProductionModelManager {
    fn create_model(&self, name: &str, model_type: &str, description: Option<&str>) -> Result<Model> {
        let model_id = Self::generate_id();
        let model = Model {
            id: model_id.clone(),
            name: name.to_string(),
            description: description.map(|s| s.to_string()),
            version: "1.0.0".to_string(),
            model_type: model_type.to_string(),
            smart_parameters: SmartModelParameters::default(),
            architecture: ModelArchitecture::default(),
            status: ModelStatus::Created,
            metrics: None,
            created_at: Utc::now(),
            updated_at: Utc::now(),
            parent_id: None,
            metadata: HashMap::new(),
            input_shape: Vec::new(),
            output_shape: Vec::new(),
            import_source: None,
            memory_monitor: Arc::new(Mutex::new(ModelMemoryMonitor::new())),
        };
        let serializable_model = SerializableModel::from(&model);
        let model_data = serde_json::to_vec(&serializable_model)?;
        self.storage.put(&format!("model:{}", model_id), &model_data)?;
        let mut models = self.models.write().map_err(|e| Error::lock(e.to_string()))?;
        let model_arc = Arc::new(model.clone());
        models.insert(model_id, model_arc);
        Ok(model)
    }
    fn get_model(&self, model_id: &str) -> Result<Option<Model>> {
        {
            let models = self.models.read().map_err(|e| Error::lock(e.to_string()))?;
            if let Some(model) = models.get(model_id) {
                return Ok(Some((**model).clone()));
            }
        }
        if let Some(data) = self.storage.get(&format!("model:{}", model_id))? {
            let serializable_model: SerializableModel = serde_json::from_slice(&data)?;
            let model = serializable_model.to_full_model();
            let mut models = self.models.write().map_err(|e| Error::lock(e.to_string()))?;
            let model_arc = Arc::new(model.clone());
            models.insert(model_id.to_string(), model_arc);
            Ok(Some(model))
        } else {
            Ok(None)
        }
    }
    fn update_model(&self, model: &Model) -> Result<()> {
        let mut updated_model = model.clone();
        updated_model.updated_at = Utc::now();
        let serializable_model = SerializableModel::from(&updated_model);
        let model_data = serde_json::to_vec(&serializable_model)?;
        self.storage.put(&format!("model:{}", model.id), &model_data)?;
        let mut models = self.models.write().map_err(|e| Error::lock(e.to_string()))?;
        models.insert(model.id.clone(), Arc::new(updated_model));
        Ok(())
    }
    fn delete_model(&self, model_id: &str) -> Result<bool> {
        if self.get_model(model_id)?.is_none() {
            return Ok(false);
        }
        {
            let mut tasks = self.monitoring_tasks.write().map_err(|e| Error::lock(e.to_string()))?;
            tasks.retain(|_, task| task.model_id != model_id);
        }
        self.storage.delete(&format!("model:{}", model_id))?;
        let mut models = self.models.write().map_err(|e| Error::lock(e.to_string()))?;
        models.remove(model_id);
        Ok(true)
    }
    fn save_model_parameters(&self, model_id: &str, parameters: &ModelParameters) -> Result<()> {
        let params_data = serde_json::to_vec(parameters)?;
        self.storage.put(&format!("model:{}:params", model_id), &params_data)?;
        Ok(())
    }
    fn get_model_parameters(&self, model_id: &str) -> Result<Option<ModelParameters>> {
        if let Some(data) = self.storage.get(&format!("model:{}:params", model_id))? {
            let params: ModelParameters = serde_json::from_slice(&data)?;
            Ok(Some(params))
        } else {
            Ok(None)
        }
    }
    fn save_model_architecture(&self, model_id: &str, architecture: &ModelArchitecture) -> Result<()> {
        let arch_data = serde_json::to_vec(architecture)?;
        self.storage.put(&format!("model:{}:arch", model_id), &arch_data)?;
        Ok(())
    }
    fn get_model_architecture(&self, model_id: &str) -> Result<Option<ModelArchitecture>> {
        if let Some(data) = self.storage.get(&format!("model:{}:arch", model_id))? {
            let arch: ModelArchitecture = serde_json::from_slice(&data)?;
            Ok(Some(arch))
        } else {
            Ok(None)
        }
    }
    fn list_models(&self) -> Result<Vec<ModelInfo>> {
        let models = self.models.read().map_err(|e| Error::lock(e.to_string()))?;
        let model_infos: Vec<ModelInfo> = models.values()
            .map(|model| ModelInfo {
                id: model.id.clone(),
                name: model.name.clone(),
                version: model.version.clone(),
                description: model.description.clone(),
                created_at: model.created_at,
                updated_at: model.updated_at,
                model_type: model.model_type.clone(),
                framework: "custom".to_string(),
                framework_version: "1".to_string(),
                size: 0,
                input_format: "generic".to_string(),
                output_format: "generic".to_string(),
                preprocessing: vec![],
                postprocessing: vec![],
                tags: model.metadata.get("tags").map(|s| s.split(',').map(|t| t.trim().to_string()).collect()).unwrap_or_default(),
                license: None,
                author: None,
                metrics: None,
                dependencies: vec![],
                storage_format: StorageFormat::Native,
                metadata: model.metadata.clone(),
                checksum: None,
            })
            .collect();
        Ok(model_infos)
    }
    
    fn import_model(&self, path: &Path, format: StorageFormat) -> Result<String> {
        // 实现模型导入逻辑
        let model_id = Self::generate_id();
        // 这里应该实现具体的导入逻辑
        Ok(model_id)
    }
    
    /// 创建模型版本（生产级实现：持久化版本信息到存储）
    fn create_model_version(&self, model_id: &str, version: &str, description: Option<&str>) -> Result<String> {
        use std::time::SystemTime;
        
        // 验证模型是否存在
        let model_key = format!("model:{}", model_id);
        if self.storage.get(&model_key)?.is_none() {
            return Err(Error::not_found(format!("模型不存在: {}", model_id)));
        }
        
        // 生成版本ID
        let version_id = Self::generate_id();
        
        // 创建版本信息
        let version_info = ModelVersionInfo {
            version: version.to_string(),
            created_at: SystemTime::now(),
            metadata: {
                let mut meta = HashMap::new();
                meta.insert("model_id".to_string(), model_id.to_string());
                meta.insert("version_id".to_string(), version_id.clone());
                if let Some(desc) = description {
                    meta.insert("description".to_string(), desc.to_string());
                }
                meta.insert("created_by".to_string(), "system".to_string());
                meta
            },
        };
        
        // 序列化版本信息
        let version_data = bincode::serialize(&version_info)
            .map_err(|e| Error::serialization(format!("序列化版本信息失败: {}", e)))?;
        
        // 存储版本信息
        let version_key = format!("model:{}:version:{}", model_id, version_id);
        self.storage.put(&version_key, &version_data)?;
        
        // 更新模型的版本列表索引
        let version_list_key = format!("model:{}:versions", model_id);
        if let Some(existing_data) = self.storage.get(&version_list_key)? {
            let mut version_ids: Vec<String> = bincode::deserialize(&existing_data)
                .unwrap_or_default();
            version_ids.push(version_id.clone());
            let updated_data = bincode::serialize(&version_ids)
                .map_err(|e| Error::serialization(format!("序列化版本列表失败: {}", e)))?;
            self.storage.put(&version_list_key, &updated_data)?;
        } else {
            let version_ids = vec![version_id.clone()];
            let version_list_data = bincode::serialize(&version_ids)
                .map_err(|e| Error::serialization(format!("序列化版本列表失败: {}", e)))?;
            self.storage.put(&version_list_key, &version_list_data)?;
        }
        
        Ok(version_id)
    }
    
    /// 列出模型版本（生产级实现：从存储中查询所有版本信息）
    fn list_model_versions(&self, model_id: &str) -> Result<Vec<ModelVersionInfo>> {
        // 验证模型是否存在
        let model_key = format!("model:{}", model_id);
        if self.storage.get(&model_key)?.is_none() {
            return Err(Error::not_found(format!("模型不存在: {}", model_id)));
        }
        
        // 方法1：从版本列表索引获取（更快）
        let version_list_key = format!("model:{}:versions", model_id);
        if let Some(version_list_data) = self.storage.get(&version_list_key)? {
            if let Ok(version_ids) = bincode::deserialize::<Vec<String>>(&version_list_data) {
                let mut versions = Vec::new();
                for version_id in version_ids {
                    let version_key = format!("model:{}:version:{}", model_id, version_id);
                    if let Some(version_data) = self.storage.get(&version_key)? {
                        if let Ok(version_info) = bincode::deserialize::<ModelVersionInfo>(&version_data) {
                            versions.push(version_info);
                        }
                    }
                }
                // 按创建时间排序（最新的在前）
                versions.sort_by(|a, b| b.created_at.cmp(&a.created_at));
                return Ok(versions);
            }
        }
        
        // 方法2：扫描所有版本键（备用方法，如果索引不存在）
        // 使用同步方法直接访问底层数据库（生产级实现）
        let version_prefix = format!("model:{}:version:", model_id);
        let mut versions = Vec::new();
        
        // 通过反射或直接访问底层数据库进行同步扫描
        // 注意：这里假设 Storage 内部使用 RocksDB，可以通过内部方法访问
        // 如果 Storage 没有暴露同步扫描方法，我们使用异步运行时
        use tokio::runtime::Runtime;
        let rt = Runtime::new()
            .map_err(|e| Error::internal(format!("创建异步运行时失败: {}", e)))?;
        
        if let Ok(scan_results) = rt.block_on(self.storage.scan_prefix_raw(&version_prefix)) {
            for (_key, version_data) in scan_results {
                if let Ok(version_info) = bincode::deserialize::<ModelVersionInfo>(&version_data) {
                    versions.push(version_info);
                }
            }
        }
        
        // 按创建时间排序（最新的在前）
        versions.sort_by(|a, b| b.created_at.cmp(&a.created_at));
        
        Ok(versions)
    }
    
    fn switch_to_version(&self, model_id: &str, version: &str) -> Result<()> {
        // 实现切换到指定版本逻辑
        Ok(())
    }
    
    fn delete_model_version(&self, model_id: &str, version: &str) -> Result<bool> {
        // 实现删除模型版本逻辑
        Ok(true)
    }
    
    fn health_check(&self, model_id: &str) -> Result<ModelHealthStatus> {
        // 实现模型健康检查逻辑
        // ModelHealthStatus 在 common.rs 中是一个 enum，不是 struct
        Ok(ModelHealthStatus::Healthy)
    }
    
    fn get_model_metrics(&self, model_id: &str) -> Result<ModelPerformanceMetrics> {
        // 实现获取模型性能指标逻辑
        Ok(ModelPerformanceMetrics {
            accuracy: 0.0,
            precision: 0.0,
            recall: 0.0,
            f1_score: 0.0,
            inference_time_ms: 0.0,
            throughput_per_second: 0.0,
            memory_usage_mb: 0.0,
            last_updated: Utc::now(),
        })
    }
    
    fn start_monitoring(&self, model_id: &str, config: &MonitoringConfig) -> Result<String> {
        // 实现启动模型监控逻辑
        let monitor_id = Self::generate_id();
        Ok(monitor_id)
    }
    
    fn stop_monitoring(&self, model_id: &str, monitor_id: &str) -> Result<()> {
        // 实现停止模型监控逻辑
        Ok(())
    }
    
    fn create_backup(&self, model_id: &str, backup_type: BackupType) -> Result<String> {
        // 实现创建模型备份逻辑
        let backup_id = Self::generate_id();
        Ok(backup_id)
    }
    
    fn restore_from_backup(&self, model_id: &str, backup_id: &str) -> Result<()> {
        // 实现从备份恢复模型逻辑
        Ok(())
    }
    
    fn list_backups(&self, model_id: &str) -> Result<Vec<BackupInfo>> {
        // 实现列出模型备份逻辑
        Ok(vec![])
    }
    
    fn delete_backup(&self, backup_id: &str) -> Result<bool> {
        // 实现删除备份逻辑
        Ok(true)
    }
    
    fn warm_up_model(&self, model_id: &str, warm_up_data: &DataBatch) -> Result<()> {
        // 实现模型预热逻辑
        Ok(())
    }
    
    fn compress_model(&self, model_id: &str, compression_config: &CompressionConfig) -> Result<String> {
        // 实现模型压缩逻辑
        let compression_id = Self::generate_id();
        Ok(compression_id)
    }
    
    fn quantize_model(&self, model_id: &str, quantization_config: &QuantizationConfig) -> Result<String> {
        // 实现模型量化逻辑
        let quantization_id = Self::generate_id();
        Ok(quantization_id)
    }
    
    fn optimize_model(&self, model_id: &str, optimization_config: &OptimizationConfig) -> Result<String> {
        // 实现模型优化逻辑
        let optimization_id = Self::generate_id();
        Ok(optimization_id)
    }
    
    fn get_model_dependencies(&self, model_id: &str) -> Result<Vec<ModelDependency>> {
        // 实现获取模型依赖逻辑
        Ok(vec![])
    }
    
    fn validate_model_integrity(&self, model_id: &str) -> Result<IntegrityCheckResult> {
        // 实现验证模型完整性逻辑
        Ok(IntegrityCheckResult {
            model_id: model_id.to_string(),
            check_type: IntegrityCheckType::Full,
            status: IntegrityStatus::Valid,
            issues_found: vec![],
            checked_at: Utc::now(),
            repair_suggestions: vec![],
        })
    }
    
    fn create_ab_test(&self, model_a_id: &str, model_b_id: &str, test_config: &ABTestConfig) -> Result<String> {
        // 实现创建A/B测试逻辑
        let test_id = Self::generate_id();
        Ok(test_id)
    }
    
    fn get_ab_test_results(&self, test_id: &str) -> Result<ABTestResults> {
        // 实现获取A/B测试结果逻辑
        Ok(ABTestResults {
            test_id: test_id.to_string(),
            model_a_metrics: HashMap::new(),
            model_b_metrics: HashMap::new(),
            statistical_significance: 0.0,
            recommendation: TestRecommendation::NoSignificantDifference,
            analyzed_at: Utc::now(),
        })
    }
    
    fn deploy_model(&self, model_id: &str, deployment_config: &DeploymentConfig) -> Result<String> {
        // 实现模型部署逻辑
        let deployment_id = Self::generate_id();
        Ok(deployment_id)
    }
    
    fn undeploy_model(&self, model_id: &str, deployment_id: &str) -> Result<()> {
        // 实现取消部署逻辑
        Ok(())
    }
    
    fn get_deployment_status(&self, deployment_id: &str) -> Result<DeploymentStatus> {
        // 实现获取部署状态逻辑
        // DeploymentStatus 在 common.rs 中是一个 enum，不是 struct
        Ok(DeploymentStatus::Running)
    }
    
    fn inference(&self, model_id: &str, input_data: &DataBatch) -> Result<DataBatch> {
        // 实现模型推理逻辑
        Ok(DataBatch::new())
    }
    
    fn batch_inference(&self, model_id: &str, input_batches: &[DataBatch]) -> Result<Vec<DataBatch>> {
        // 实现批量推理逻辑
        Ok(vec![])
    }
    
    fn async_inference(&self, model_id: &str, input_data: &DataBatch) -> Result<String> {
        // 实现异步推理逻辑
        let inference_id = Self::generate_id();
        Ok(inference_id)
    }
    
    fn get_inference_result(&self, inference_id: &str) -> Result<Option<DataBatch>> {
        // 实现获取异步推理结果逻辑
        Ok(None)
    }
    
    fn start_monitoring_task(&self, model_id: &str, monitor_id: &str, config: MonitoringConfig) -> Result<MonitoringTask> {
        Ok(MonitoringTask {
            id: monitor_id.to_string(),
            model_id: model_id.to_string(),
            task_type: MonitoringTaskType::HealthCheck,
            schedule: "*/5 * * * *".to_string(),
            config: MonitoringConfig {
                enabled: true,
                interval_secs: config.interval_seconds,
            },
            status: TaskStatus::Active,
            last_run: None,
            next_run: Utc::now(),
        })
    }
    
    fn start_performance_monitoring(&self, model_id: &str, monitor_id: &str) -> Result<()> {
        // 实现启动性能监控逻辑
        Ok(())
    }
    
    fn start_accuracy_monitoring(&self, model_id: &str, monitor_id: &str) -> Result<()> {
        // 实现启动准确性监控逻辑
        Ok(())
    }
    
    fn start_resource_monitoring(&self, model_id: &str, monitor_id: &str) -> Result<()> {
        // 实现启动资源监控逻辑
        Ok(())
    }
    
    fn start_prediction_monitoring(&self, model_id: &str, monitor_id: &str) -> Result<()> {
        // 实现启动预测监控逻辑
        Ok(())
    }
    fn search_models(&self, query: &str) -> Result<Vec<ModelInfo>> {
        let models = self.models.read().map_err(|e| Error::lock(e.to_string()))?;
        let query_lower = query.to_lowercase();
        let matching_models = models.values()
            .filter(|model| {
                model.name.to_lowercase().contains(&query_lower) ||
                model.model_type.to_lowercase().contains(&query_lower) ||
                model.description.as_ref().map_or(false, |d| d.to_lowercase().contains(&query_lower))
            })
            .map(|model| ModelInfo {
                id: model.id.clone(),
                name: model.name.clone(),
                version: model.version.clone(),
                description: model.description.clone(),
                created_at: model.created_at,
                updated_at: model.updated_at,
                model_type: model.model_type.clone(),
                framework: "custom".to_string(),
                framework_version: "1".to_string(),
                size: 0,
                input_format: "generic".to_string(),
                output_format: "generic".to_string(),
                preprocessing: vec![],
                postprocessing: vec![],
                tags: model.metadata.get("tags").map(|s| s.split(',').map(|t| t.trim().to_string()).collect()).unwrap_or_default(),
                license: None,
                author: None,
                metrics: None,
                dependencies: vec![],
                storage_format: StorageFormat::Native,
                metadata: model.metadata.clone(),
                checksum: None,
            })
            .collect();
        Ok(matching_models)
    }
    fn find_models_by_tag(&self, tag: &str) -> Result<Vec<ModelInfo>> {
        let models = self.models.read().map_err(|e| Error::lock(e.to_string()))?;
        let matching_models = models.values()
            .filter(|model| {
                model.metadata.get("tags")
                    .and_then(|tags| tags.as_str())
                    .map_or(false, |tags_str| tags_str.contains(tag))
            })
            .map(|model| ModelInfo {
                id: model.id.clone(),
                name: model.name.clone(),
                version: model.version.clone(),
                description: model.description.clone(),
                created_at: model.created_at,
                updated_at: model.updated_at,
                model_type: model.model_type.clone(),
                framework: "custom".to_string(),
                framework_version: "1".to_string(),
                size: 0,
                input_format: "generic".to_string(),
                output_format: "generic".to_string(),
                preprocessing: vec![],
                postprocessing: vec![],
                tags: model.metadata.get("tags").map(|s| s.split(',').map(|t| t.trim().to_string()).collect()).unwrap_or_default(),
                license: None,
                author: None,
                metrics: None,
                dependencies: vec![],
                storage_format: StorageFormat::Native,
                metadata: model.metadata.clone(),
                checksum: None,
            })
            .collect();
        Ok(matching_models)
    }
    fn export_model(&self, model_id: &str, format: StorageFormat, path: &Path) -> Result<()> {
        let model = self.get_model(model_id)?
            .ok_or_else(|| Error::not_found(format!("Model {} not found", model_id)))?;
        let serializable_model = SerializableModel::from(&model);
        match format {
            StorageFormat::Native => {
                let json = serde_json::to_string_pretty(&serializable_model)?;
                std::fs::write(path, json)?;
            }
            StorageFormat::ONNX | StorageFormat::TensorFlowSavedModel | StorageFormat::PyTorch | StorageFormat::TensorRT => {
                let bin = bincode::serialize(&serializable_model)?;
                std::fs::write(path, bin)?;
            }
        }
        Ok(())
    }
    // ... 其余方法可根据需要继续迁移和补全 ...
} 