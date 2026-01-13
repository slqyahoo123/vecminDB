/// 核心接口实现模块
/// 
/// 提供interfaces.rs中定义的所有接口的生产级实现
/// 解决功能实现不完整性问题

pub mod storage_impl;
pub mod model_impl;
// pub mod training_impl; // 已移除：向量数据库系统不需要训练功能
pub mod data_impl;
pub mod algorithm_impl;
pub mod monitoring_impl;

// 重新导出主要的实现结构体
pub use storage_impl::{
    ProductionKeyValueStore,
    ProductionTransactionalStore,
    ProductionObjectStore,
    StorageMetrics,
};

pub use model_impl::{
    ProductionModelRepository,
    ProductionParameterManager,
    ProductionVersionManager,
    ModelStorageBackend,
    ParameterStorageBackend,
    VersionStorageBackend,
    ModelEventPublisher,
};

pub use training_impl::{
    ProductionTaskManager,
    ProductionOptimizer,
    ProductionLossFunction,
    ProductionMetricsCalculator,
    TrainingEventPublisher,
};

/// 生产级服务工厂
/// 
/// 提供创建各种服务实例的工厂方法
pub struct ProductionServiceFactory;

impl ProductionServiceFactory {
    /// 创建存储服务实例
    pub fn create_storage_services(storage_path: &str) -> crate::Result<StorageServices> {
        let kv_store = storage_impl::ProductionKeyValueStore::new(storage_path)?;
        let transactional_store = storage_impl::ProductionTransactionalStore::new(storage_path)?;
        let object_store = storage_impl::ProductionObjectStore::new(&format!("{}/objects", storage_path))?;

        Ok(StorageServices {
            kv_store: std::sync::Arc::new(kv_store),
            transactional_store: std::sync::Arc::new(transactional_store),
            object_store: std::sync::Arc::new(object_store),
        })
    }

    /// 创建模型服务实例
    pub fn create_model_services(storage_backend: std::sync::Arc<dyn ModelStorageBackend>) -> ModelServices {
        let repository = model_impl::ProductionModelRepository::new(storage_backend.clone());
        
        // 创建简单的内存参数存储后端
        let param_backend = std::sync::Arc::new(InMemoryParameterBackend::new());
        let parameter_manager = model_impl::ProductionParameterManager::new(param_backend);
        
        // 创建简单的内存版本存储后端
        let version_backend = std::sync::Arc::new(InMemoryVersionBackend::new());
        let version_manager = model_impl::ProductionVersionManager::new(version_backend);

        ModelServices {
            repository: std::sync::Arc::new(repository),
            parameter_manager: std::sync::Arc::new(parameter_manager),
            version_manager: std::sync::Arc::new(version_manager),
        }
    }

    /// 创建训练服务实例
    pub fn create_training_services() -> TrainingServices {
        let task_manager = training_impl::ProductionTaskManager::new();
        
        // 创建默认优化器配置
        let optimizer_config = training_impl::OptimizerConfig {
            momentum: Some(0.9),
            weight_decay: Some(1e-4),
            beta1: Some(0.9),
            beta2: Some(0.999),
            epsilon: Some(1e-8),
        };
        
        let optimizer = training_impl::ProductionOptimizer::new(
            training_impl::OptimizerType::Adam,
            0.001,
            optimizer_config,
        );

        // 创建默认损失函数配置
        let loss_config = training_impl::LossFunctionConfig {
            reduction: "mean".to_string(),
            ignore_index: None,
        };
        
        let loss_function = training_impl::ProductionLossFunction::new(
            training_impl::LossFunctionType::MSE,
            loss_config,
        );

        let metrics_calculator = training_impl::ProductionMetricsCalculator::new();

        TrainingServices {
            task_manager: std::sync::Arc::new(task_manager),
            optimizer: std::sync::Arc::new(optimizer),
            loss_function: std::sync::Arc::new(loss_function),
            metrics_calculator: std::sync::Arc::new(metrics_calculator),
        }
    }

    /// 创建完整的生产级服务套件
    pub fn create_full_services(config: &ProductionConfig) -> crate::Result<ProductionServices> {
        let storage_services = Self::create_storage_services(&config.storage_path)?;
        
        // 创建文件系统模型存储后端
        let model_storage_backend = std::sync::Arc::new(
            FileSystemModelBackend::new(&format!("{}/models", config.storage_path))?
        );
        
        let model_services = Self::create_model_services(model_storage_backend);
        // let training_services = Self::create_training_services(); // 已移除

        Ok(ProductionServices {
            storage: storage_services,
            model: model_services,
            // training: training_services, // 已移除
        })
    }
}

/// 存储服务集合
pub struct StorageServices {
    pub kv_store: std::sync::Arc<storage_impl::ProductionKeyValueStore>,
    pub transactional_store: std::sync::Arc<storage_impl::ProductionTransactionalStore>,
    pub object_store: std::sync::Arc<storage_impl::ProductionObjectStore>,
}

/// 模型服务集合
pub struct ModelServices {
    pub repository: std::sync::Arc<model_impl::ProductionModelRepository>,
    pub parameter_manager: std::sync::Arc<model_impl::ProductionParameterManager>,
    pub version_manager: std::sync::Arc<model_impl::ProductionVersionManager>,
}

// 训练服务集合已移除：向量数据库系统不需要训练功能
// pub struct TrainingServices { ... }

/// 完整的生产级服务
pub struct ProductionServices {
    pub storage: StorageServices,
    pub model: ModelServices,
    // pub training: TrainingServices, // 已移除
}

/// 生产级配置
#[derive(Debug, Clone)]
pub struct ProductionConfig {
    pub storage_path: String,
    pub max_concurrent_tasks: usize,
    pub enable_metrics: bool,
    pub enable_events: bool,
}

impl Default for ProductionConfig {
    fn default() -> Self {
        Self {
            storage_path: "./data".to_string(),
            max_concurrent_tasks: 10,
            enable_metrics: true,
            enable_events: true,
        }
    }
}

/// 内存参数存储后端实现
pub struct InMemoryParameterBackend {
    parameters: std::sync::RwLock<std::collections::HashMap<String, std::collections::HashMap<String, Vec<u8>>>>,
}

impl InMemoryParameterBackend {
    pub fn new() -> Self {
        Self {
            parameters: std::sync::RwLock::new(std::collections::HashMap::new()),
        }
    }
}

#[async_trait::async_trait]
impl model_impl::ParameterStorageBackend for InMemoryParameterBackend {
    async fn store_parameters(&self, model_id: &str, version: &str, data: &[u8]) -> crate::Result<()> {
        let mut params = self.parameters.write().unwrap();
        let model_params = params.entry(model_id.to_string()).or_insert_with(std::collections::HashMap::new);
        model_params.insert(version.to_string(), data.to_vec());
        Ok(())
    }

    async fn load_parameters(&self, model_id: &str, version: &str) -> crate::Result<Option<Vec<u8>>> {
        let params = self.parameters.read().unwrap();
        if let Some(model_params) = params.get(model_id) {
            Ok(model_params.get(version).cloned())
        } else {
            Ok(None)
        }
    }

    async fn list_parameter_versions(&self, model_id: &str) -> crate::Result<Vec<String>> {
        let params = self.parameters.read().unwrap();
        if let Some(model_params) = params.get(model_id) {
            Ok(model_params.keys().cloned().collect())
        } else {
            Ok(Vec::new())
        }
    }
}

/// 内存版本存储后端实现
pub struct InMemoryVersionBackend {
    versions: std::sync::RwLock<std::collections::HashMap<String, Vec<model_impl::ModelVersion>>>,
}

impl InMemoryVersionBackend {
    pub fn new() -> Self {
        Self {
            versions: std::sync::RwLock::new(std::collections::HashMap::new()),
        }
    }
}

#[async_trait::async_trait]
impl model_impl::VersionStorageBackend for InMemoryVersionBackend {
    async fn store_version(&self, model_id: &str, version: &model_impl::ModelVersion) -> crate::Result<()> {
        let mut versions = self.versions.write().unwrap();
        let model_versions = versions.entry(model_id.to_string()).or_insert_with(Vec::new);
        model_versions.push(version.clone());
        Ok(())
    }

    async fn load_versions(&self, model_id: &str) -> crate::Result<Vec<model_impl::ModelVersion>> {
        let versions = self.versions.read().unwrap();
        Ok(versions.get(model_id).cloned().unwrap_or_default())
    }
}

/// 文件系统模型存储后端实现
pub struct FileSystemModelBackend {
    base_path: String,
}

impl FileSystemModelBackend {
    pub fn new(base_path: &str) -> crate::Result<Self> {
        std::fs::create_dir_all(base_path)?;
        Ok(Self {
            base_path: base_path.to_string(),
        })
    }

    fn get_model_path(&self, model_id: &str) -> String {
        format!("{}/{}.json", self.base_path, model_id)
    }
}

#[async_trait::async_trait]
impl model_impl::ModelStorageBackend for FileSystemModelBackend {
    async fn store_model(&self, model_id: &str, model: &model_impl::StoredModel) -> crate::Result<()> {
        let path = self.get_model_path(model_id);
        let data = serde_json::to_string_pretty(model)
            .map_err(|e| crate::Error::InvalidInput(e.to_string()))?;
        tokio::fs::write(path, data).await?;
        Ok(())
    }

    async fn load_model(&self, model_id: &str) -> crate::Result<Option<model_impl::StoredModel>> {
        let path = self.get_model_path(model_id);
        match tokio::fs::read_to_string(path).await {
            Ok(data) => {
                let model = serde_json::from_str(&data)
                    .map_err(|e| crate::Error::InvalidInput(e.to_string()))?;
                Ok(Some(model))
            },
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(None),
            Err(e) => Err(crate::Error::Io(e)),
        }
    }

    async fn delete_model(&self, model_id: &str) -> crate::Result<()> {
        let path = self.get_model_path(model_id);
        if tokio::fs::metadata(&path).await.is_ok() {
            tokio::fs::remove_file(path).await?;
        }
        Ok(())
    }

    async fn model_exists(&self, model_id: &str) -> crate::Result<bool> {
        let path = self.get_model_path(model_id);
        Ok(tokio::fs::metadata(path).await.is_ok())
    }
} 