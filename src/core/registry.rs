/// 服务注册表实现
/// 用于管理各模块间的依赖注入，解决循环依赖问题

use std::collections::HashMap;
use std::sync::{Arc, RwLock};
use async_trait::async_trait;
use crate::error::{Error, Result};
use crate::core::interfaces::{
    service_registry::ServiceRegistry,
    model_training::{ModelTrainingService, ModelStateService},
    algorithm_model::{AlgorithmExecutionService, ModelAlgorithmService},
    data_model::{DataProcessingService, ModelDataService},
};

/// 核心服务注册表实现
#[derive(Default)]
pub struct CoreServiceRegistry {
    /// 模型训练服务
    model_training_service: RwLock<Option<Arc<dyn ModelTrainingService>>>,
    /// 模型状态服务
    model_state_service: RwLock<Option<Arc<dyn ModelStateService>>>,
    /// 算法执行服务
    algorithm_execution_service: RwLock<Option<Arc<dyn AlgorithmExecutionService>>>,
    /// 模型算法服务
    model_algorithm_service: RwLock<Option<Arc<dyn ModelAlgorithmService>>>,
    /// 数据处理服务
    data_processing_service: RwLock<Option<Arc<dyn DataProcessingService>>>,
    /// 模型数据服务
    model_data_service: RwLock<Option<Arc<dyn ModelDataService>>>,
    /// 服务状态
    service_status: RwLock<HashMap<String, ServiceStatus>>,
}

#[derive(Debug, Clone)]
pub struct ServiceStatus {
    pub registered: bool,
    pub active: bool,
    pub last_health_check: chrono::DateTime<chrono::Utc>,
    pub error_count: usize,
}

impl CoreServiceRegistry {
    /// 创建新的服务注册表
    pub fn new() -> Self {
        Self::default()
    }

    /// 检查服务健康状态
    pub async fn check_service_health(&self, service_name: &str) -> Result<ServiceStatus> {
        let status_map = self.service_status.read()
            .map_err(|_| Error::lock("无法获取服务状态读锁"))?;
        
        status_map.get(service_name)
            .cloned()
            .ok_or_else(|| Error::not_found(format!("服务未注册: {}", service_name)))
    }

    /// 更新服务状态
    pub async fn update_service_status(&self, service_name: &str, status: ServiceStatus) -> Result<()> {
        let mut status_map = self.service_status.write()
            .map_err(|_| Error::lock("无法获取服务状态写锁"))?;
        
        status_map.insert(service_name.to_string(), status);
        Ok(())
    }

    /// 列出所有已注册的服务
    pub async fn list_services(&self) -> Result<Vec<String>> {
        let status_map = self.service_status.read()
            .map_err(|_| Error::lock("无法获取服务状态读锁"))?;
        
        Ok(status_map.keys().cloned().collect())
    }

    /// 验证所有必要服务是否已注册
    pub async fn validate_all_services(&self) -> Result<bool> {
        let required_services = vec![
            "model_training_service",
            "model_state_service", 
            "algorithm_execution_service",
            "model_algorithm_service",
            "data_processing_service",
            "model_data_service",
        ];

        for service_name in required_services {
            let status = self.check_service_health(service_name).await?;
            if !status.registered || !status.active {
                return Ok(false);
            }
        }

        Ok(true)
    }

    /// 私有方法：更新服务注册状态
    async fn mark_service_registered(&self, service_name: &str) -> Result<()> {
        let status = ServiceStatus {
            registered: true,
            active: true,
            last_health_check: chrono::Utc::now(),
            error_count: 0,
        };
        
        self.update_service_status(service_name, status).await
    }
}

#[async_trait]
impl ServiceRegistry for CoreServiceRegistry {
    /// 注册模型训练服务
    async fn register_model_training_service(&self, service: Arc<dyn ModelTrainingService>) -> Result<()> {
        let mut service_slot = self.model_training_service.write()
            .map_err(|_| Error::lock("无法获取模型训练服务写锁"))?;
        
        *service_slot = Some(service);
        self.mark_service_registered("model_training_service").await?;
        
        log::info!("模型训练服务已注册");
        Ok(())
    }

    /// 注册模型状态服务
    async fn register_model_state_service(&self, service: Arc<dyn ModelStateService>) -> Result<()> {
        let mut service_slot = self.model_state_service.write()
            .map_err(|_| Error::lock("无法获取模型状态服务写锁"))?;
        
        *service_slot = Some(service);
        self.mark_service_registered("model_state_service").await?;
        
        log::info!("模型状态服务已注册");
        Ok(())
    }

    /// 注册算法执行服务
    async fn register_algorithm_execution_service(&self, service: Arc<dyn AlgorithmExecutionService>) -> Result<()> {
        let mut service_slot = self.algorithm_execution_service.write()
            .map_err(|_| Error::lock("无法获取算法执行服务写锁"))?;
        
        *service_slot = Some(service);
        self.mark_service_registered("algorithm_execution_service").await?;
        
        log::info!("算法执行服务已注册");
        Ok(())
    }

    /// 注册模型算法服务
    async fn register_model_algorithm_service(&self, service: Arc<dyn ModelAlgorithmService>) -> Result<()> {
        let mut service_slot = self.model_algorithm_service.write()
            .map_err(|_| Error::lock("无法获取模型算法服务写锁"))?;
        
        *service_slot = Some(service);
        self.mark_service_registered("model_algorithm_service").await?;
        
        log::info!("模型算法服务已注册");
        Ok(())
    }

    /// 注册数据处理服务
    async fn register_data_processing_service(&self, service: Arc<dyn DataProcessingService>) -> Result<()> {
        let mut service_slot = self.data_processing_service.write()
            .map_err(|_| Error::lock("无法获取数据处理服务写锁"))?;
        
        *service_slot = Some(service);
        self.mark_service_registered("data_processing_service").await?;
        
        log::info!("数据处理服务已注册");
        Ok(())
    }

    /// 注册模型数据服务
    async fn register_model_data_service(&self, service: Arc<dyn ModelDataService>) -> Result<()> {
        let mut service_slot = self.model_data_service.write()
            .map_err(|_| Error::lock("无法获取模型数据服务写锁"))?;
        
        *service_slot = Some(service);
        self.mark_service_registered("model_data_service").await?;
        
        log::info!("模型数据服务已注册");
        Ok(())
    }

    /// 获取模型训练服务
    async fn get_model_training_service(&self) -> Result<Arc<dyn ModelTrainingService>> {
        let service_slot = self.model_training_service.read()
            .map_err(|_| Error::lock("无法获取模型训练服务读锁"))?;
        
        service_slot.as_ref()
            .cloned()
            .ok_or_else(|| Error::not_found("模型训练服务未注册"))
    }

    /// 获取模型状态服务
    async fn get_model_state_service(&self) -> Result<Arc<dyn ModelStateService>> {
        let service_slot = self.model_state_service.read()
            .map_err(|_| Error::lock("无法获取模型状态服务读锁"))?;
        
        service_slot.as_ref()
            .cloned()
            .ok_or_else(|| Error::not_found("模型状态服务未注册"))
    }

    /// 获取算法执行服务
    async fn get_algorithm_execution_service(&self) -> Result<Arc<dyn AlgorithmExecutionService>> {
        let service_slot = self.algorithm_execution_service.read()
            .map_err(|_| Error::lock("无法获取算法执行服务读锁"))?;
        
        service_slot.as_ref()
            .cloned()
            .ok_or_else(|| Error::not_found("算法执行服务未注册"))
    }

    /// 获取模型算法服务
    async fn get_model_algorithm_service(&self) -> Result<Arc<dyn ModelAlgorithmService>> {
        let service_slot = self.model_algorithm_service.read()
            .map_err(|_| Error::lock("无法获取模型算法服务读锁"))?;
        
        service_slot.as_ref()
            .cloned()
            .ok_or_else(|| Error::not_found("模型算法服务未注册"))
    }

    /// 获取数据处理服务
    async fn get_data_processing_service(&self) -> Result<Arc<dyn DataProcessingService>> {
        let service_slot = self.data_processing_service.read()
            .map_err(|_| Error::lock("无法获取数据处理服务读锁"))?;
        
        service_slot.as_ref()
            .cloned()
            .ok_or_else(|| Error::not_found("数据处理服务未注册"))
    }

    /// 获取模型数据服务
    async fn get_model_data_service(&self) -> Result<Arc<dyn ModelDataService>> {
        let service_slot = self.model_data_service.read()
            .map_err(|_| Error::lock("无法获取模型数据服务读锁"))?;
        
        service_slot.as_ref()
            .cloned()
            .ok_or_else(|| Error::not_found("模型数据服务未注册"))
    }
} 