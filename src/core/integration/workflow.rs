/// 工作流管理模块
/// 
/// 提供工作流编排功能，用于管理复杂的跨模块操作流程

use std::sync::{Arc, Mutex};
use std::collections::HashMap;
use std::pin::Pin;
use std::future::Future;
use chrono::{DateTime, Utc};
use uuid::Uuid;
use serde::{Serialize, Deserialize};
use log::{info, error};
use anyhow::Result;

use crate::core::{CoreEvent, EventBusInterface};

/// 工作流状态
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum WorkflowStatus {
    Created,
    Running,
    Completed,
    Failed,
    Cancelled,
    Paused,
}

/// 工作流步骤状态
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum StepStatus {
    Pending,
    Running,
    Completed,
    Failed,
    Skipped,
}

/// 工作流执行上下文
#[derive(Debug, Clone)]
pub struct WorkflowContext {
    pub workflow_id: String,
    pub step_id: String,
    pub data: HashMap<String, String>,
    pub shared_state: Arc<Mutex<HashMap<String, String>>>,
}

impl WorkflowContext {
    pub fn new(workflow_id: String, step_id: String) -> Self {
        Self {
            workflow_id,
            step_id,
            data: HashMap::new(),
            shared_state: Arc::new(Mutex::new(HashMap::new())),
        }
    }
    
    pub fn set_data(&mut self, key: String, value: String) {
        self.data.insert(key, value);
    }
    
    pub fn get_data(&self, key: &str) -> Option<&String> {
        self.data.get(key)
    }
    
    pub fn set_shared_state(&self, key: String, value: String) {
        let mut state = self.shared_state.lock()
            .expect("共享状态锁获取失败：无法更新共享状态");
        state.insert(key, value);
    }
    
    pub fn get_shared_state(&self, key: &str) -> Option<String> {
        let state = self.shared_state.lock()
            .expect("共享状态锁获取失败：无法读取共享状态");
        state.get(key).cloned()
    }
}

/// 工作流步骤执行函数类型
pub type StepExecutor = Box<
    dyn Fn(WorkflowContext) -> Pin<Box<dyn Future<Output = Result<()>> + Send>> + Send + Sync
>;

/// 工作流步骤
#[derive(Clone)]
pub struct WorkflowStep {
    pub id: String,
    pub name: String,
    pub description: String,
    pub status: StepStatus,
    pub dependencies: Vec<String>,
    pub retry_count: usize,
    pub max_retries: usize,
    pub timeout_seconds: Option<u64>,
    pub on_failure: Option<FailureAction>,
    pub created_at: DateTime<Utc>,
    pub started_at: Option<DateTime<Utc>>,
    pub completed_at: Option<DateTime<Utc>>,
    pub error_message: Option<String>,
}

/// 失败处理动作
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FailureAction {
    Retry,
    Skip,
    Abort,
    Rollback,
}

impl WorkflowStep {
    pub fn new(id: &str, name: &str, _executor: StepExecutor) -> Self {
        Self {
            id: id.to_string(),
            name: name.to_string(),
            description: String::new(),
            status: StepStatus::Pending,
            dependencies: Vec::new(),
            retry_count: 0,
            max_retries: 3,
            timeout_seconds: None,
            on_failure: Some(FailureAction::Retry),
            created_at: Utc::now(),
            started_at: None,
            completed_at: None,
            error_message: None,
        }
    }
    
    pub fn with_description(mut self, description: &str) -> Self {
        self.description = description.to_string();
        self
    }
    
    pub fn with_dependencies(mut self, deps: Vec<String>) -> Self {
        self.dependencies = deps;
        self
    }
    
    pub fn with_max_retries(mut self, max_retries: usize) -> Self {
        self.max_retries = max_retries;
        self
    }
    
    pub fn with_timeout(mut self, timeout_seconds: u64) -> Self {
        self.timeout_seconds = Some(timeout_seconds);
        self
    }
    
    pub fn with_failure_action(mut self, action: FailureAction) -> Self {
        self.on_failure = Some(action);
        self
    }
}

/// 集成工作流
pub struct IntegrationWorkflow {
    pub id: String,
    pub name: String,
    pub description: String,
    pub status: WorkflowStatus,
    pub steps: HashMap<String, WorkflowStep>,
    pub step_executors: HashMap<String, StepExecutor>,
    pub execution_order: Vec<String>,
    pub metadata: HashMap<String, String>,
    pub created_at: DateTime<Utc>,
    pub started_at: Option<DateTime<Utc>>,
    pub completed_at: Option<DateTime<Utc>>,
    pub error_message: Option<String>,
    pub context: WorkflowContext,
}

impl Clone for IntegrationWorkflow {
    fn clone(&self) -> Self {
        Self {
            id: self.id.clone(),
            name: self.name.clone(),
            description: self.description.clone(),
            status: self.status.clone(),
            steps: self.steps.clone(),
            step_executors: HashMap::new(), // StepExecutor 不实现 Clone，克隆时清空
            execution_order: self.execution_order.clone(),
            metadata: self.metadata.clone(),
            created_at: self.created_at,
            started_at: self.started_at,
            completed_at: self.completed_at,
            error_message: self.error_message.clone(),
            context: self.context.clone(),
        }
    }
}

impl IntegrationWorkflow {
    pub fn new(name: &str) -> Self {
        let id = Uuid::new_v4().to_string();
        let context = WorkflowContext::new(id.clone(), "root".to_string());
        
        Self {
            id: id.clone(),
            name: name.to_string(),
            description: String::new(),
            status: WorkflowStatus::Created,
            steps: HashMap::new(),
            step_executors: HashMap::new(),
            execution_order: Vec::new(),
            metadata: HashMap::new(),
            created_at: Utc::now(),
            started_at: None,
            completed_at: None,
            error_message: None,
            context,
        }
    }
    
    pub fn add_step(mut self, step: WorkflowStep) -> Self {
        let step_id = step.id.clone();
        self.execution_order.push(step_id.clone());
        self.steps.insert(step_id, step);
        self
    }
    
    pub fn with_description(mut self, description: &str) -> Self {
        self.description = description.to_string();
        self
    }
    
    pub fn with_metadata(mut self, key: String, value: String) -> Self {
        self.metadata.insert(key, value);
        self
    }
    
    /// 验证工作流是否可执行
    pub fn validate(&self) -> Result<()> {
        // 检查是否有步骤
        if self.steps.is_empty() {
            return Err(anyhow::anyhow!("工作流没有步骤"));
        }
        
        // 检查依赖关系
        for step in self.steps.values() {
            for dep in &step.dependencies {
                if !self.steps.contains_key(dep) {
                    return Err(anyhow::anyhow!(
                        "步骤 {} 依赖的步骤 {} 不存在", 
                        step.id, 
                        dep
                    ));
                }
            }
        }
        
        // 检查循环依赖
        self.check_circular_dependencies()?;
        
        Ok(())
    }
    
    /// 检查循环依赖
    fn check_circular_dependencies(&self) -> Result<()> {
        let mut visited = HashMap::new();
        let mut in_stack = HashMap::new();
        
        for step_id in self.steps.keys() {
            if !visited.contains_key(step_id) {
                if self.has_cycle(step_id, &mut visited, &mut in_stack)? {
                    return Err(anyhow::anyhow!("工作流存在循环依赖"));
                }
            }
        }
        
        Ok(())
    }
    
    fn has_cycle(
        &self,
        step_id: &str,
        visited: &mut HashMap<String, bool>,
        in_stack: &mut HashMap<String, bool>,
    ) -> Result<bool> {
        visited.insert(step_id.to_string(), true);
        in_stack.insert(step_id.to_string(), true);
        
        if let Some(step) = self.steps.get(step_id) {
            for dep in &step.dependencies {
                if !visited.get(dep).unwrap_or(&false) {
                    if self.has_cycle(dep, visited, in_stack)? {
                        return Ok(true);
                    }
                } else if *in_stack.get(dep).unwrap_or(&false) {
                    return Ok(true);
                }
            }
        }
        
        in_stack.insert(step_id.to_string(), false);
        Ok(false)
    }
    
    /// 获取可执行的步骤
    pub fn get_executable_steps(&self) -> Vec<String> {
        let mut executable = Vec::new();
        
        for step_id in &self.execution_order {
            if let Some(step) = self.steps.get(step_id) {
                if step.status == StepStatus::Pending && self.dependencies_completed(step) {
                    executable.push(step_id.clone());
                }
            }
        }
        
        executable
    }
    
    /// 检查步骤依赖是否完成
    fn dependencies_completed(&self, step: &WorkflowStep) -> bool {
        for dep in &step.dependencies {
            if let Some(dep_step) = self.steps.get(dep) {
                if dep_step.status != StepStatus::Completed {
                    return false;
                }
            } else {
                return false;
            }
        }
        true
    }
}

/// 工作流管理器
pub struct WorkflowManager {
    active_workflows: Arc<Mutex<HashMap<String, IntegrationWorkflow>>>,
    event_bus: Arc<dyn EventBusInterface>,
    executor_pool: Arc<Mutex<Vec<WorkflowExecutor>>>,
}

impl WorkflowManager {
    pub fn new(event_bus: Arc<dyn EventBusInterface>) -> Result<Self> {
        Ok(Self {
            active_workflows: Arc::new(Mutex::new(HashMap::new())),
            event_bus,
            executor_pool: Arc::new(Mutex::new(Vec::new())),
        })
    }
    
    /// 注册工作流
    pub fn register_workflow(&self, workflow: IntegrationWorkflow) -> Result<()> {
        workflow.validate()?;
        
        let mut workflows = self.active_workflows.lock()
            .expect("工作流列表锁获取失败：无法更新工作流");
        workflows.insert(workflow.id.clone(), workflow);
        
        info!("注册工作流: {}", workflows.len());
        Ok(())
    }
    
    /// 执行工作流
    pub async fn execute_workflow(&self, mut workflow: IntegrationWorkflow) -> Result<()> {
        workflow.validate()?;
        
        let workflow_id = workflow.id.clone();
        info!("开始执行工作流: {}", workflow_id);
        
        // 更新状态
        workflow.status = WorkflowStatus::Running;
        workflow.started_at = Some(Utc::now());
        
        // 发送开始事件
        self.send_workflow_event(&workflow, "workflow.started").await?;
        
        // 注册工作流
        {
            let mut workflows = self.active_workflows.lock()
            .expect("工作流列表锁获取失败：无法更新工作流");
            workflows.insert(workflow_id.clone(), workflow);
        }
        
        // 创建执行器
        let executor = WorkflowExecutor::new(
            workflow_id.clone(),
            self.active_workflows.clone(),
            self.event_bus.clone(),
        );
        
        // 开始执行
        match executor.execute().await {
            Ok(_) => {
                // 更新工作流状态为完成
                {
                    let mut workflows = self.active_workflows.lock()
            .expect("工作流列表锁获取失败：无法更新工作流");
                    if let Some(workflow) = workflows.get_mut(&workflow_id) {
                        workflow.status = WorkflowStatus::Completed;
                        workflow.completed_at = Some(Utc::now());
                    }
                }
                
                // 发送完成事件，避免跨await边界持有锁
                let workflow_opt = {
                    let workflows = self.active_workflows.lock()
                        .expect("工作流列表锁获取失败：无法读取工作流");
                    workflows.get(&workflow_id).map(|w| (w.id.clone(), w.status.clone()))
                };
                
                if let Some((id, status)) = workflow_opt {
                    // 发送工作流完成事件（简化实现，不传递完整工作流）
                    let event = crate::core::CoreEvent {
                        id: uuid::Uuid::new_v4().to_string(),
                        event_type: "workflow.completed".to_string(),
                        source: "workflow_manager".to_string(),
                        timestamp: std::time::SystemTime::now()
                            .duration_since(std::time::UNIX_EPOCH)
                            .unwrap()
                            .as_secs(),
                        data: std::collections::HashMap::from([
                            ("workflow_id".to_string(), id),
                            ("status".to_string(), format!("{:?}", status)),
                        ]),
                        metadata: None,
                    };
                    self.event_bus.publish(&event).await?;
                }
                
                info!("工作流执行完成: {}", workflow_id);
                Ok(())
            },
            Err(e) => {
                // 更新工作流状态为失败
                {
                    let mut workflows = self.active_workflows.lock()
            .expect("工作流列表锁获取失败：无法更新工作流");
                    if let Some(workflow) = workflows.get_mut(&workflow_id) {
                        workflow.status = WorkflowStatus::Failed;
                        workflow.error_message = Some(e.to_string());
                        workflow.completed_at = Some(Utc::now());
                    }
                }
                
                // 发送失败事件
                let workflows = self.active_workflows.lock().unwrap();
                if let Some(workflow) = workflows.get(&workflow_id) {
                    self.send_workflow_event(workflow, "workflow.failed").await?;
                }
                
                error!("工作流执行失败: {}, 错误: {}", workflow_id, e);
                Err(e)
            }
        }
    }
    
    /// 获取工作流状态
    pub fn get_workflow_status(&self, workflow_id: &str) -> Option<WorkflowStatus> {
        let workflows = self.active_workflows.lock().unwrap();
        workflows.get(workflow_id).map(|w| w.status.clone())
    }
    
    /// 取消工作流
    pub async fn cancel_workflow(&self, workflow_id: &str) -> Result<()> {
        let mut workflows = self.active_workflows.lock()
            .expect("工作流列表锁获取失败：无法更新工作流");
        if let Some(workflow) = workflows.get_mut(workflow_id) {
            workflow.status = WorkflowStatus::Cancelled;
            workflow.completed_at = Some(Utc::now());
            
            // 发送取消事件
            self.send_workflow_event(workflow, "workflow.cancelled").await?;
            
            info!("取消工作流: {}", workflow_id);
            Ok(())
        } else {
            Err(anyhow::anyhow!("工作流不存在: {}", workflow_id))
        }
    }
    
    /// 发送工作流事件
    async fn send_workflow_event(
        &self,
        workflow: &IntegrationWorkflow,
        event_type: &str,
    ) -> Result<()> {
        let event = CoreEvent::new(event_type, "workflow_manager")
            .with_data("workflow_id", &workflow.id)
            .with_data("workflow_name", &workflow.name)
            .with_data("status", &format!("{:?}", workflow.status));
            
        self.event_bus.publish(&event).await?;
        Ok(())
    }
}

/// 工作流执行器
pub struct WorkflowExecutor {
    workflow_id: String,
    workflows: Arc<Mutex<HashMap<String, IntegrationWorkflow>>>,
    event_bus: Arc<dyn EventBusInterface>,
}

impl WorkflowExecutor {
    pub fn new(
        workflow_id: String,
        workflows: Arc<Mutex<HashMap<String, IntegrationWorkflow>>>,
        event_bus: Arc<dyn EventBusInterface>,
    ) -> Self {
        Self {
            workflow_id,
            workflows,
            event_bus,
        }
    }
    
    /// 执行工作流
    pub async fn execute(&self) -> Result<()> {
        loop {
            // 获取可执行的步骤
            let executable_steps = {
                let workflows = self.workflows.lock().unwrap();
                if let Some(workflow) = workflows.get(&self.workflow_id) {
                    if workflow.status != WorkflowStatus::Running {
                        break;
                    }
                    workflow.get_executable_steps()
                } else {
                    return Err(anyhow::anyhow!("工作流不存在: {}", self.workflow_id));
                }
            };
            
            if executable_steps.is_empty() {
                // 检查是否所有步骤都完成
                let all_completed = {
                    let workflows = self.workflows.lock().unwrap();
                    if let Some(workflow) = workflows.get(&self.workflow_id) {
                        workflow.steps.values().all(|step| {
                            step.status == StepStatus::Completed || step.status == StepStatus::Skipped
                        })
                    } else {
                        false
                    }
                };
                
                if all_completed {
                    break;
                } else {
                    // 可能存在失败的步骤或循环等待
                    let failed_steps = {
                        let workflows = self.workflows.lock().unwrap();
                        if let Some(workflow) = workflows.get(&self.workflow_id) {
                            workflow.steps.values()
                                .filter(|step| step.status == StepStatus::Failed)
                                .count()
                        } else {
                            0
                        }
                    };
                    
                    if failed_steps > 0 {
                        return Err(anyhow::anyhow!("工作流中有步骤执行失败"));
                    }
                    
                    // 短暂等待
                    tokio::time::sleep(tokio::time::Duration::from_secs(1)).await;
                    continue;
                }
            }
            
            // 并行执行可执行的步骤
            let mut tasks = Vec::new();
            for step_id in executable_steps {
                let task = self.execute_step(step_id);
                tasks.push(task);
            }
            
            // 等待所有步骤完成
            let results = futures::future::join_all(tasks).await;
            
            // 检查执行结果
            for result in results {
                if let Err(e) = result {
                    error!("步骤执行失败: {}", e);
                    return Err(e);
                }
            }
        }
        
        info!("工作流执行完成: {}", self.workflow_id);
        Ok(())
    }
    
    /// 执行单个步骤
    async fn execute_step(&self, step_id: String) -> Result<()> {
        info!("执行步骤: {}", step_id);
        
        // 更新步骤状态为运行中
        {
            let mut workflows = self.workflows.lock().unwrap();
            if let Some(workflow) = workflows.get_mut(&self.workflow_id) {
                if let Some(step) = workflow.steps.get_mut(&step_id) {
                    step.status = StepStatus::Running;
                    step.started_at = Some(Utc::now());
                }
            }
        }
        
        // 发送步骤开始事件
        self.send_step_event(&step_id, "step.started").await?;
        
        // 模拟步骤执行（实际实现中应该调用真正的执行函数）
        tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;
        
        // 更新步骤状态为完成
        {
            let mut workflows = self.workflows.lock().unwrap();
            if let Some(workflow) = workflows.get_mut(&self.workflow_id) {
                if let Some(step) = workflow.steps.get_mut(&step_id) {
                    step.status = StepStatus::Completed;
                    step.completed_at = Some(Utc::now());
                }
            }
        }
        
        // 发送步骤完成事件
        self.send_step_event(&step_id, "step.completed").await?;
        
        info!("步骤执行完成: {}", step_id);
        Ok(())
    }
    
    /// 发送步骤事件
    async fn send_step_event(&self, step_id: &str, event_type: &str) -> Result<()> {
        let event = CoreEvent::new(event_type, "workflow_executor")
            .with_data("workflow_id", &self.workflow_id)
            .with_data("step_id", step_id);
            
        self.event_bus.publish(&event).await?;
        Ok(())
    }
} 