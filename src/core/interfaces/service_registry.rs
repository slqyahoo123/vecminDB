use super::*;
use async_trait::async_trait;
use crate::error::Result;
// use std::sync::Arc; // 预留

/// 服务注册表接口
#[async_trait]
pub trait ServiceRegistry: Send + Sync {
    async fn register_data_service(&self, service: std::sync::Arc<dyn data_interface::DataProcessingService>) -> Result<()>;
    async fn register_model_service(&self, service: std::sync::Arc<dyn model_interface::ModelService>) -> Result<()>;
    async fn register_training_service(&self, service: std::sync::Arc<dyn training_interface::TrainingService>) -> Result<()>;
    async fn register_algorithm_service(&self, service: std::sync::Arc<dyn algorithm_interface::AlgorithmService>) -> Result<()>;
    async fn register_storage_service(&self, service: std::sync::Arc<dyn storage_interface::StorageService>) -> Result<()>;

    async fn get_data_service(&self) -> Result<Option<std::sync::Arc<dyn data_interface::DataProcessingService>>>;
    async fn get_model_service(&self) -> Result<Option<std::sync::Arc<dyn model_interface::ModelService>>>;
    async fn get_training_service(&self) -> Result<Option<std::sync::Arc<dyn training_interface::TrainingService>>>;
    async fn get_algorithm_service(&self) -> Result<Option<std::sync::Arc<dyn algorithm_interface::AlgorithmService>>>;
    async fn get_storage_service(&self) -> Result<Option<std::sync::Arc<dyn storage_interface::StorageService>>>;
}

/// 默认服务注册表实现
pub struct DefaultServiceRegistry {
    pub(crate) data_service: std::sync::RwLock<Option<std::sync::Arc<dyn data_interface::DataProcessingService>>>,
    pub(crate) model_service: std::sync::RwLock<Option<std::sync::Arc<dyn model_interface::ModelService>>>,
    pub(crate) training_service: std::sync::RwLock<Option<std::sync::Arc<dyn training_interface::TrainingService>>>,
    pub(crate) algorithm_service: std::sync::RwLock<Option<std::sync::Arc<dyn algorithm_interface::AlgorithmService>>>,
    pub(crate) storage_service: std::sync::RwLock<Option<std::sync::Arc<dyn storage_interface::StorageService>>>,
}

impl DefaultServiceRegistry {
    pub fn new() -> Self {
        Self {
            data_service: std::sync::RwLock::new(None),
            model_service: std::sync::RwLock::new(None),
            training_service: std::sync::RwLock::new(None),
            algorithm_service: std::sync::RwLock::new(None),
            storage_service: std::sync::RwLock::new(None),
        }
    }
}

#[async_trait]
impl ServiceRegistry for DefaultServiceRegistry {
    async fn register_data_service(&self, service: std::sync::Arc<dyn data_interface::DataProcessingService>) -> Result<()> {
        let mut data_service = self.data_service.write().map_err(|_| crate::error::Error::lock("无法获取数据服务写锁".to_string()))?;
        *data_service = Some(service);
        Ok(())
    }
    async fn register_model_service(&self, service: std::sync::Arc<dyn model_interface::ModelService>) -> Result<()> {
        let mut model_service = self.model_service.write().map_err(|_| crate::error::Error::lock("无法获取模型服务写锁".to_string()))?;
        *model_service = Some(service);
        Ok(())
    }
    async fn register_training_service(&self, service: std::sync::Arc<dyn training_interface::TrainingService>) -> Result<()> {
        let mut training_service = self.training_service.write().map_err(|_| crate::error::Error::lock("无法获取训练服务写锁".to_string()))?;
        *training_service = Some(service);
        Ok(())
    }
    async fn register_algorithm_service(&self, service: std::sync::Arc<dyn algorithm_interface::AlgorithmService>) -> Result<()> {
        let mut algorithm_service = self.algorithm_service.write().map_err(|_| crate::error::Error::lock("无法获取算法服务写锁".to_string()))?;
        *algorithm_service = Some(service);
        Ok(())
    }
    async fn register_storage_service(&self, service: std::sync::Arc<dyn storage_interface::StorageService>) -> Result<()> {
        let mut storage_service = self.storage_service.write().map_err(|_| crate::error::Error::lock("无法获取存储服务写锁".to_string()))?;
        *storage_service = Some(service);
        Ok(())
    }

    async fn get_data_service(&self) -> Result<Option<std::sync::Arc<dyn data_interface::DataProcessingService>>> {
        let data_service = self.data_service.read().map_err(|_| crate::error::Error::lock("无法获取数据服务读锁".to_string()))?;
        Ok(data_service.clone())
    }
    async fn get_model_service(&self) -> Result<Option<std::sync::Arc<dyn model_interface::ModelService>>> {
        let model_service = self.model_service.read().map_err(|_| crate::error::Error::lock("无法获取模型服务读锁".to_string()))?;
        Ok(model_service.clone())
    }
    async fn get_training_service(&self) -> Result<Option<std::sync::Arc<dyn training_interface::TrainingService>>> {
        let training_service = self.training_service.read().map_err(|_| crate::error::Error::lock("无法获取训练服务读锁".to_string()))?;
        Ok(training_service.clone())
    }
    async fn get_algorithm_service(&self) -> Result<Option<std::sync::Arc<dyn algorithm_interface::AlgorithmService>>> {
        let algorithm_service = self.algorithm_service.read().map_err(|_| crate::error::Error::lock("无法获取算法服务读锁".to_string()))?;
        Ok(algorithm_service.clone())
    }
    async fn get_storage_service(&self) -> Result<Option<std::sync::Arc<dyn storage_interface::StorageService>>> {
        let storage_service = self.storage_service.read().map_err(|_| crate::error::Error::lock("无法获取存储服务读锁".to_string()))?;
        Ok(storage_service.clone())
    }
}


