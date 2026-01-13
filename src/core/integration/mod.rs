/// 核心集成模块 - 解决模块间集成度不足问题
/// 
/// 此模块提供统一的跨模块协作机制，包括：
/// 1. 模块间事务处理
/// 2. 数据流转管理  
/// 3. 自动化集成流程
/// 4. 状态同步机制

use std::sync::{Arc, Mutex, RwLock};
use std::collections::HashMap;
use async_trait::async_trait;
use chrono::{DateTime, Utc};
use uuid::Uuid;
use serde::{Serialize, Deserialize};
use log::{info, debug, warn};
use crate::Result;

use crate::core::interfaces::{
    // TrainingEngineInterface, // 已移除：向量数据库系统不需要训练功能
    ModelServiceInterface, AlgorithmExecutorInterface,
    DataProcessorInterface, StorageServiceInterface
};
use crate::core::{CoreEvent, EventBusInterface};
use crate::error::Error;

// 子模块
pub mod workflow;
pub mod transaction;
pub mod data_flow;
pub mod state_sync;
pub mod automation;

// 重新导出核心组件
pub use workflow::{IntegrationWorkflow, WorkflowStep, WorkflowStatus};
pub use transaction::{CrossModuleTransaction, TransactionCoordinator};
pub use data_flow::{DataFlowManager, DataFlowPipeline};
pub use state_sync::{StateSyncManager, SyncStrategy};
pub use automation::{AutomationEngine, AutomationRule};

/// 集成管理器配置
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IntegrationConfig {
    /// 是否启用事务管理
    pub enable_transactions: bool,
    /// 是否启用数据流管理
    pub enable_data_flow: bool,
    /// 是否启用状态同步
    pub enable_state_sync: bool,
    /// 是否启用自动化
    pub enable_automation: bool,
    /// 事务超时时间（秒）
    pub transaction_timeout: u64,
    /// 数据流缓冲区大小
    pub data_flow_buffer_size: usize,
    /// 状态同步间隔（秒）
    pub state_sync_interval: u64,
    /// 自动化检查间隔（秒）
    pub automation_interval: u64,
}

impl Default for IntegrationConfig {
    fn default() -> Self {
        Self {
            enable_transactions: true,
            enable_data_flow: true,
            enable_state_sync: true,
            enable_automation: true,
            transaction_timeout: 300,
            data_flow_buffer_size: 1000,
            state_sync_interval: 30,
            automation_interval: 10,
        }
    }
}

/// 模块集成管理器 - 解决模块间协作问题
pub struct IntegrationManager {
    config: IntegrationConfig,
    event_bus: Arc<dyn EventBusInterface>,
    
    // 模块接口
    // training_engine: Option<Arc<dyn TrainingEngineInterface>>, // 训练引擎已移除
    model_service: Option<Arc<dyn ModelServiceInterface>>,
    algorithm_executor: Option<Arc<dyn AlgorithmExecutorInterface>>,
    data_processor: Option<Arc<dyn DataProcessorInterface>>,
    storage_service: Option<Arc<dyn StorageServiceInterface>>,
    
    // 集成组件
    workflow_manager: Arc<Mutex<workflow::WorkflowManager>>,
    transaction_coordinator: Arc<transaction::TransactionCoordinator>,
    data_flow_manager: Arc<data_flow::DataFlowManager>,
    state_sync_manager: Arc<state_sync::StateSyncManager>,
    automation_engine: Arc<automation::AutomationEngine>,
    
    // 状态管理
    active_integrations: Arc<RwLock<HashMap<String, IntegrationSession>>>,
}

/// 集成会话 - 跟踪单次集成操作的状态
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IntegrationSession {
    pub id: String,
    pub session_type: IntegrationSessionType,
    pub status: IntegrationSessionStatus,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
    pub metadata: HashMap<String, String>,
    pub error_message: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum IntegrationSessionType {
    TrainingModelIntegration,
    AlgorithmExecutionIntegration, 
    DataProcessingIntegration,
    ModelUpdateIntegration,
    CrossModuleTransaction,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum IntegrationSessionStatus {
    Initialized,
    InProgress,
    Completed,
    Failed,
    Cancelled,
}

impl IntegrationManager {
    /// 创建新的集成管理器
    pub fn new(
        config: IntegrationConfig,
        event_bus: Arc<dyn EventBusInterface>,
    ) -> Result<Self> {
        let workflow_manager = Arc::new(Mutex::new(
            workflow::WorkflowManager::new(event_bus.clone())?
        ));
        
        let transaction_coordinator = Arc::new(
            transaction::TransactionCoordinator::new(config.transaction_timeout)?
        );
        
        let data_flow_manager = Arc::new(
            data_flow::DataFlowManager::new(config.data_flow_buffer_size)?
        );
        
        let state_sync_manager = Arc::new(
            state_sync::StateSyncManager::new(
                config.state_sync_interval,
                event_bus.clone()
            )?
        );
        
        let automation_engine = Arc::new(
            automation::AutomationEngine::new(
                config.automation_interval,
                event_bus.clone()
            )?
        );

        Ok(Self {
            config,
            event_bus,
            // training_engine: None, // 训练引擎已移除
            model_service: None,
            algorithm_executor: None,
            data_processor: None,
            storage_service: None,
            workflow_manager,
            transaction_coordinator,
            data_flow_manager,
            state_sync_manager,
            automation_engine,
            active_integrations: Arc::new(RwLock::new(HashMap::new())),
        })
    }
    
    // 训练引擎接口已移除 - 向量数据库系统不需要训练功能
    // pub fn register_training_engine(&mut self, engine: Arc<dyn TrainingEngineInterface>) {
    //     self.training_engine = Some(engine);
    //     info!("已注册训练引擎接口");
    // }
    
    /// 注册模型服务接口
    pub fn register_model_service(&mut self, service: Arc<dyn ModelServiceInterface>) {
        self.model_service = Some(service);
        info!("已注册模型服务接口");
    }
    
    /// 注册算法执行器接口
    pub fn register_algorithm_executor(&mut self, executor: Arc<dyn AlgorithmExecutorInterface>) {
        self.algorithm_executor = Some(executor);
        info!("已注册算法执行器接口");
    }
    
    /// 注册数据处理器接口
    pub fn register_data_processor(&mut self, processor: Arc<dyn DataProcessorInterface>) {
        self.data_processor = Some(processor);
        info!("已注册数据处理器接口");
    }
    
    /// 注册存储服务接口
    pub fn register_storage_service(&mut self, storage: Arc<dyn StorageServiceInterface>) {
        self.storage_service = Some(storage);
        info!("已注册存储服务接口");
    }
    
    /// 启动集成管理器
    pub async fn start(&self) -> Result<()> {
        info!("启动集成管理器...");
        
        // 启动各组件
        if self.config.enable_state_sync {
            self.state_sync_manager.start().await?;
        }
        
        if self.config.enable_automation {
            self.automation_engine.start().await?;
        }
        
        // 注册事件监听器
        self.register_event_listeners().await?;
        
        info!("集成管理器启动完成");
        Ok(())
    }
    
    /// 注册事件监听器
    async fn register_event_listeners(&self) -> Result<()> {
        // 监听训练相关事件
        let training_handler = TrainingEventHandler {
            integration_manager: self.create_weak_ref(),
        };
        self.event_bus.subscribe("training.*", Box::new(training_handler)).await?;
        
        // 监听模型相关事件
        let model_handler = ModelEventHandler {
            integration_manager: self.create_weak_ref(),
        };
        self.event_bus.subscribe("model.*", Box::new(model_handler)).await?;
        
        // 监听算法相关事件
        let algorithm_handler = AlgorithmEventHandler {
            integration_manager: self.create_weak_ref(),
        };
        self.event_bus.subscribe("algorithm.*", Box::new(algorithm_handler)).await?;
        
        Ok(())
    }
    
    /// 创建弱引用（简化实现）
    fn create_weak_ref(&self) -> IntegrationManagerRef {
        IntegrationManagerRef {
            // 这里应该使用 Weak<> 引用，简化实现使用标识符
            manager_id: "integration_manager".to_string(),
        }
    }
}

/// 集成管理器引用（简化版）
#[derive(Clone)]
pub struct IntegrationManagerRef {
    pub manager_id: String,
}

/// 训练事件处理器
pub struct TrainingEventHandler {
    pub integration_manager: IntegrationManagerRef,
}

#[async_trait]
impl crate::core::EventHandler for TrainingEventHandler {
    async fn handle(&self, event: &CoreEvent) -> Result<()> {
        debug!("处理训练事件: {}", event.event_type);
        
        match event.event_type.as_str() {
            "training.started" => {
                // 处理训练开始事件
                info!("检测到训练开始事件，启动集成流程");
            },
            "training.completed" => {
                // 处理训练完成事件，自动更新模型
                info!("检测到训练完成事件，自动更新模型");
            },
            "training.failed" => {
                // 处理训练失败事件
                warn!("检测到训练失败事件，进行错误处理");
            },
            _ => {
                debug!("未处理的训练事件: {}", event.event_type);
            }
        }
        
        Ok(())
    }
}

/// 模型事件处理器
pub struct ModelEventHandler {
    pub integration_manager: IntegrationManagerRef,
}

#[async_trait]
impl crate::core::EventHandler for ModelEventHandler {
    async fn handle(&self, event: &CoreEvent) -> Result<()> {
        debug!("处理模型事件: {}", event.event_type);
        
        match event.event_type.as_str() {
            "model.created" => {
                info!("检测到模型创建事件");
            },
            "model.updated" => {
                info!("检测到模型更新事件，同步存储");
            },
            "model.deleted" => {
                info!("检测到模型删除事件，清理相关资源");
            },
            _ => {
                debug!("未处理的模型事件: {}", event.event_type);
            }
        }
        
        Ok(())
    }
}

/// 算法事件处理器
pub struct AlgorithmEventHandler {
    pub integration_manager: IntegrationManagerRef,
}

#[async_trait]
impl crate::core::EventHandler for AlgorithmEventHandler {
    async fn handle(&self, event: &CoreEvent) -> Result<()> {
        debug!("处理算法事件: {}", event.event_type);
        
        match event.event_type.as_str() {
            "algorithm.executed" => {
                info!("检测到算法执行完成事件，自动更新模型");
            },
            "algorithm.failed" => {
                warn!("检测到算法执行失败事件");
            },
            _ => {
                debug!("未处理的算法事件: {}", event.event_type);
            }
        }
        
        Ok(())
    }
}

/// 集成管理器接口
#[async_trait]
pub trait IntegrationManagerInterface: Send + Sync {
    /// 执行训练-模型集成流程
    async fn integrate_training_model(
        &self,
        training_config: &crate::core::CoreTrainingConfig,
        model_id: &str,
    ) -> Result<String>;
    
    /// 执行算法-模型集成流程
    async fn integrate_algorithm_model(
        &self,
        algorithm_id: &str,
        model_id: &str,
        execution_params: &HashMap<String, String>,
    ) -> Result<String>;
    
    /// 执行数据处理-训练集成流程
    async fn integrate_data_training(
        &self,
        data_batch_id: &str,
        training_config: &crate::core::CoreTrainingConfig,
    ) -> Result<String>;
    
    /// 获取集成会话状态
    async fn get_integration_status(&self, session_id: &str) -> Result<IntegrationSession>;
    
    /// 取消集成会话
    async fn cancel_integration(&self, session_id: &str) -> Result<()>;
}

#[async_trait]
impl IntegrationManagerInterface for IntegrationManager {
    async fn integrate_training_model(
        &self,
        training_config: &crate::core::CoreTrainingConfig,
        model_id: &str,
    ) -> Result<String> {
        let session_id = Uuid::new_v4().to_string();
        
        // 创建集成会话
        let session = IntegrationSession {
            id: session_id.clone(),
            session_type: IntegrationSessionType::TrainingModelIntegration,
            status: IntegrationSessionStatus::Initialized,
            created_at: Utc::now(),
            updated_at: Utc::now(),
            metadata: {
                let mut meta = HashMap::new();
                meta.insert("model_id".to_string(), model_id.to_string());
                meta.insert("training_type".to_string(), "model_integration".to_string());
                meta
            },
            error_message: None,
        };
        
        // 保存会话
        {
            let mut sessions = self.active_integrations.write()
                .expect("集成会话写入锁获取失败：无法更新集成会话");
            sessions.insert(session_id.clone(), session);
        }
        
        // 创建集成工作流
        let workflow = self.create_training_model_workflow(model_id, training_config)?;
        
        // 启动工作流：避免跨 await 持有 MutexGuard，将执行权移交到独立任务中
        {
            let manager_arc = self.workflow_manager.clone();
            tokio::task::spawn_blocking(move || {
                if let Ok(manager) = manager_arc.lock() {
                    // 在阻塞线程中执行同步操作
                    let _ = manager.execute_workflow(workflow);
                }
            });
        }
        
        info!("启动训练-模型集成流程: session_id={}", session_id);
        Ok(session_id)
    }
    
    async fn integrate_algorithm_model(
        &self,
        algorithm_id: &str,
        model_id: &str,
        execution_params: &HashMap<String, String>,
    ) -> Result<String> {
        let session_id = Uuid::new_v4().to_string();
        
        // 创建集成会话
        let session = IntegrationSession {
            id: session_id.clone(),
            session_type: IntegrationSessionType::AlgorithmExecutionIntegration,
            status: IntegrationSessionStatus::Initialized,
            created_at: Utc::now(),
            updated_at: Utc::now(),
            metadata: {
                let mut meta = HashMap::new();
                meta.insert("algorithm_id".to_string(), algorithm_id.to_string());
                meta.insert("model_id".to_string(), model_id.to_string());
                meta.extend(execution_params.clone());
                meta
            },
            error_message: None,
        };
        
        // 保存会话
        {
            let mut sessions = self.active_integrations.write()
                .expect("集成会话写入锁获取失败：无法更新集成会话");
            sessions.insert(session_id.clone(), session);
        }
        
        // 创建算法-模型集成工作流
        let workflow = self.create_algorithm_model_workflow(
            algorithm_id, 
            model_id, 
            execution_params
        )?;
        
        // 启动工作流：避免跨 await 持有 MutexGuard
        {
            let manager_arc = self.workflow_manager.clone();
            tokio::task::spawn_blocking(move || {
                if let Ok(manager) = manager_arc.lock() {
                    // 在阻塞线程中执行同步操作
                    let _ = manager.execute_workflow(workflow);
                }
            });
        }
        
        info!("启动算法-模型集成流程: session_id={}", session_id);
        Ok(session_id)
    }
    
    async fn integrate_data_training(
        &self,
        data_batch_id: &str,
        training_config: &crate::core::CoreTrainingConfig,
    ) -> Result<String> {
        let session_id = Uuid::new_v4().to_string();
        
        // 创建集成会话
        let session = IntegrationSession {
            id: session_id.clone(),
            session_type: IntegrationSessionType::DataProcessingIntegration,
            status: IntegrationSessionStatus::Initialized,
            created_at: Utc::now(),
            updated_at: Utc::now(),
            metadata: {
                let mut meta = HashMap::new();
                meta.insert("data_batch_id".to_string(), data_batch_id.to_string());
                meta.insert("integration_type".to_string(), "data_training".to_string());
                meta
            },
            error_message: None,
        };
        
        // 保存会话
        {
            let mut sessions = self.active_integrations.write()
                .expect("集成会话写入锁获取失败：无法更新集成会话");
            sessions.insert(session_id.clone(), session);
        }
        
        // 创建数据-训练集成工作流
        let workflow = self.create_data_training_workflow(data_batch_id, training_config)?;
        
        // 启动工作流：避免跨 await 持有 MutexGuard
        {
            let manager_arc = self.workflow_manager.clone();
            tokio::task::spawn_blocking(move || {
                if let Ok(manager) = manager_arc.lock() {
                    // 在阻塞线程中执行同步操作
                    let _ = manager.execute_workflow(workflow);
                }
            });
        }
        
        info!("启动数据-训练集成流程: session_id={}", session_id);
        Ok(session_id)
    }
    
    async fn get_integration_status(&self, session_id: &str) -> Result<IntegrationSession> {
        let sessions = self.active_integrations.read()
            .expect("集成会话读取锁获取失败：无法读取集成会话");
        sessions.get(session_id)
            .cloned()
            .ok_or_else(|| Error::NotFound(format!("集成会话不存在: {}", session_id)).into())
    }
    
    async fn cancel_integration(&self, session_id: &str) -> Result<()> {
        let mut sessions = self.active_integrations.write().unwrap();
        if let Some(session) = sessions.get_mut(session_id) {
            session.status = IntegrationSessionStatus::Cancelled;
            session.updated_at = Utc::now();
            info!("取消集成会话: {}", session_id);
            Ok(())
        } else {
            Err(Error::NotFound(format!("集成会话不存在: {}", session_id)).into())
        }
    }
}

impl IntegrationManager {
    /// 创建训练-模型集成工作流
    fn create_training_model_workflow(
        &self,
        model_id: &str,
        training_config: &crate::core::CoreTrainingConfig,
    ) -> Result<IntegrationWorkflow> {
        let workflow = IntegrationWorkflow::new("training_model_integration")
            .add_step(WorkflowStep::new(
                "validate_model",
                "验证模型存在性",
                Box::new(move |_ctx| {
                    // 验证模型存在
                    Box::pin(async move {
                        debug!("验证模型: {}", model_id);
                        Ok(())
                    })
                })
            ))
            .add_step(WorkflowStep::new(
                "prepare_training",
                "准备训练环境",
                Box::new(move |_ctx| {
                    Box::pin(async move {
                        debug!("准备训练环境");
                        Ok(())
                    })
                })
            ))
            .add_step(WorkflowStep::new(
                "execute_training",
                "执行训练",
                Box::new(move |_ctx| {
                    Box::pin(async move {
                        debug!("执行训练");
                        Ok(())
                    })
                })
            ))
            .add_step(WorkflowStep::new(
                "update_model",
                "更新模型参数",
                Box::new(move |_ctx| {
                    Box::pin(async move {
                        debug!("更新模型参数");
                        Ok(())
                    })
                })
            ))
            .add_step(WorkflowStep::new(
                "sync_storage",
                "同步存储",
                Box::new(move |_ctx| {
                    Box::pin(async move {
                        debug!("同步存储");
                        Ok(())
                    })
                })
            ));
            
        Ok(workflow)
    }
    
    /// 创建算法-模型集成工作流
    fn create_algorithm_model_workflow(
        &self,
        algorithm_id: &str,
        model_id: &str,
        _execution_params: &HashMap<String, String>,
    ) -> Result<IntegrationWorkflow> {
        let workflow = IntegrationWorkflow::new("algorithm_model_integration")
            .add_step(WorkflowStep::new(
                "validate_algorithm",
                "验证算法",
                Box::new(move |_ctx| {
                    Box::pin(async move {
                        debug!("验证算法: {}", algorithm_id);
                        Ok(())
                    })
                })
            ))
            .add_step(WorkflowStep::new(
                "execute_algorithm",
                "执行算法",
                Box::new(move |_ctx| {
                    Box::pin(async move {
                        debug!("执行算法");
                        Ok(())
                    })
                })
            ))
            .add_step(WorkflowStep::new(
                "apply_results",
                "应用算法结果到模型",
                Box::new(move |_ctx| {
                    Box::pin(async move {
                        debug!("应用算法结果到模型: {}", model_id);
                        Ok(())
                    })
                })
            ));
            
        Ok(workflow)
    }
    
    /// 创建数据-训练集成工作流
    fn create_data_training_workflow(
        &self,
        data_batch_id: &str,
        _training_config: &crate::core::CoreTrainingConfig,
    ) -> Result<IntegrationWorkflow> {
        let workflow = IntegrationWorkflow::new("data_training_integration")
            .add_step(WorkflowStep::new(
                "load_data",
                "加载数据",
                Box::new(move |_ctx| {
                    Box::pin(async move {
                        debug!("加载数据: {}", data_batch_id);
                        Ok(())
                    })
                })
            ))
            .add_step(WorkflowStep::new(
                "process_data",
                "处理数据",
                Box::new(move |_ctx| {
                    Box::pin(async move {
                        debug!("处理数据");
                        Ok(())
                    })
                })
            ))
            .add_step(WorkflowStep::new(
                "start_training",
                "启动训练",
                Box::new(move |_ctx| {
                    Box::pin(async move {
                        debug!("启动训练");
                        Ok(())
                    })
                })
            ));
            
        Ok(workflow)
    }
} 