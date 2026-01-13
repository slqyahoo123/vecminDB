/// 自动化引擎模块
/// 
/// 提供事件驱动的自动化规则执行，实现跨模块集成流程的自动化

use std::sync::{Arc, RwLock};
use std::collections::HashMap;
use std::time::{Duration, Instant};
use async_trait::async_trait;
use chrono::{DateTime, Utc};
use uuid::Uuid;
use serde::{Serialize, Deserialize};
use log::{info, error, debug, warn};
use crate::Result;
use tokio::time::interval;

use crate::core::{CoreEvent, EventBusInterface};

/// 自动化规则类型
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum AutomationRuleType {
    TrainingToModel,        // 训练完成后自动更新模型
    AlgorithmToModel,       // 算法执行后自动更新模型
    DataToTraining,         // 数据更新后自动触发训练
    ModelToStorage,         // 模型更新后自动同步存储
    ErrorHandling,          // 错误处理自动化
    PerformanceOptimization, // 性能优化自动化
    ResourceManagement,     // 资源管理自动化
}

/// 自动化规则状态
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum AutomationRuleStatus {
    Active,
    Inactive,
    Suspended,
    Error,
}

/// 自动化触发器
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AutomationTrigger {
    Event {
        event_pattern: String,
        conditions: HashMap<String, String>,
    },
    Schedule {
        cron_expression: String,
        timezone: String,
    },
    Threshold {
        metric_name: String,
        operator: ComparisonOperator,
        threshold_value: f64,
    },
    Manual {
        trigger_id: String,
    },
}

/// 比较操作符
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ComparisonOperator {
    GreaterThan,
    LessThan,
    Equal,
    GreaterThanOrEqual,
    LessThanOrEqual,
    NotEqual,
}

/// 自动化动作
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AutomationAction {
    StartTraining {
        model_id: String,
        config: HashMap<String, String>,
    },
    UpdateModel {
        model_id: String,
        parameters: HashMap<String, String>,
    },
    ExecuteAlgorithm {
        algorithm_id: String,
        target_model: String,
        params: HashMap<String, String>,
    },
    SyncStorage {
        source: String,
        target: String,
        sync_type: String,
    },
    SendNotification {
        recipients: Vec<String>,
        message: String,
        priority: NotificationPriority,
    },
    CreateWorkflow {
        workflow_name: String,
        steps: Vec<HashMap<String, String>>,
    },
    ScaleResources {
        resource_type: String,
        scale_factor: f64,
    },
}

/// 通知优先级
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum NotificationPriority {
    Low,
    Medium,
    High,
    Critical,
}

/// 自动化规则
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AutomationRule {
    pub id: String,
    pub name: String,
    pub description: String,
    pub rule_type: AutomationRuleType,
    pub status: AutomationRuleStatus,
    pub trigger: AutomationTrigger,
    pub actions: Vec<AutomationAction>,
    pub conditions: HashMap<String, String>,
    pub retry_policy: RetryPolicy,
    pub timeout_seconds: u64,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
    pub last_executed_at: Option<DateTime<Utc>>,
    pub execution_count: u64,
    pub success_count: u64,
    pub failure_count: u64,
    pub metadata: HashMap<String, String>,
}

/// 重试策略
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RetryPolicy {
    pub max_retries: usize,
    pub retry_delay_seconds: u64,
    pub exponential_backoff: bool,
    pub max_delay_seconds: u64,
}

impl Default for RetryPolicy {
    fn default() -> Self {
        Self {
            max_retries: 3,
            retry_delay_seconds: 10,
            exponential_backoff: true,
            max_delay_seconds: 300,
        }
    }
}

impl AutomationRule {
    pub fn new(
        name: String,
        rule_type: AutomationRuleType,
        trigger: AutomationTrigger,
        actions: Vec<AutomationAction>,
    ) -> Self {
        Self {
            id: Uuid::new_v4().to_string(),
            name,
            description: String::new(),
            rule_type,
            status: AutomationRuleStatus::Active,
            trigger,
            actions,
            conditions: HashMap::new(),
            retry_policy: RetryPolicy::default(),
            timeout_seconds: 300,
            created_at: Utc::now(),
            updated_at: Utc::now(),
            last_executed_at: None,
            execution_count: 0,
            success_count: 0,
            failure_count: 0,
            metadata: HashMap::new(),
        }
    }
    
    pub fn with_description(mut self, description: String) -> Self {
        self.description = description;
        self
    }
    
    pub fn with_conditions(mut self, conditions: HashMap<String, String>) -> Self {
        self.conditions = conditions;
        self
    }
    
    pub fn with_retry_policy(mut self, retry_policy: RetryPolicy) -> Self {
        self.retry_policy = retry_policy;
        self
    }
    
    pub fn with_timeout(mut self, timeout_seconds: u64) -> Self {
        self.timeout_seconds = timeout_seconds;
        self
    }
    
    /// 检查规则是否匹配给定事件
    pub fn matches_event(&self, event: &CoreEvent) -> bool {
        match &self.trigger {
            AutomationTrigger::Event { event_pattern, conditions } => {
                // 检查事件类型是否匹配
                if !self.event_pattern_matches(event_pattern, &event.event_type) {
                    return false;
                }
                
                // 检查条件是否匹配
                for (key, expected_value) in conditions {
                    if let Some(actual_value) = event.data.get(key) {
                        if actual_value != expected_value {
                            return false;
                        }
                    } else {
                        return false;
                    }
                }
                
                true
            },
            _ => false, // 其他触发器类型不通过事件匹配
        }
    }
    
    /// 检查事件模式是否匹配
    fn event_pattern_matches(&self, pattern: &str, event_type: &str) -> bool {
        if pattern.contains('*') {
            // 简单的通配符匹配
            let pattern_parts: Vec<&str> = pattern.split('*').collect();
            if pattern_parts.len() == 2 {
                let prefix = pattern_parts[0];
                let suffix = pattern_parts[1];
                event_type.starts_with(prefix) && event_type.ends_with(suffix)
            } else {
                false
            }
        } else {
            pattern == event_type
        }
    }
    
    /// 更新执行统计
    pub fn update_execution_stats(&mut self, success: bool) {
        self.execution_count += 1;
        if success {
            self.success_count += 1;
        } else {
            self.failure_count += 1;
        }
        self.last_executed_at = Some(Utc::now());
        self.updated_at = Utc::now();
    }
}

/// 自动化执行上下文
#[derive(Debug, Clone)]
pub struct AutomationContext {
    pub rule_id: String,
    pub execution_id: String,
    pub trigger_event: Option<CoreEvent>,
    pub variables: HashMap<String, String>,
    pub start_time: DateTime<Utc>,
}

impl AutomationContext {
    pub fn new(rule_id: String, trigger_event: Option<CoreEvent>) -> Self {
        Self {
            rule_id,
            execution_id: Uuid::new_v4().to_string(),
            trigger_event,
            variables: HashMap::new(),
            start_time: Utc::now(),
        }
    }
    
    pub fn set_variable(&mut self, key: String, value: String) {
        self.variables.insert(key, value);
    }
    
    pub fn get_variable(&self, key: &str) -> Option<&String> {
        self.variables.get(key)
    }
}

/// 自动化动作执行器接口
#[async_trait]
pub trait AutomationActionExecutor: Send + Sync {
    /// 执行器类型
    fn executor_type(&self) -> &str;
    
    /// 执行动作
    async fn execute(&self, action: &AutomationAction, context: &AutomationContext) -> Result<()>;
    
    /// 验证动作是否可执行
    async fn validate(&self, action: &AutomationAction) -> Result<bool>;
}

/// 自动化引擎
pub struct AutomationEngine {
    rules: Arc<RwLock<HashMap<String, AutomationRule>>>,
    executors: Arc<RwLock<HashMap<String, Arc<dyn AutomationActionExecutor>>>>,
    event_bus: Arc<dyn EventBusInterface>,
    check_interval: Duration,
    is_running: Arc<RwLock<bool>>,
    execution_history: Arc<RwLock<Vec<AutomationExecution>>>,
    metrics: Arc<RwLock<AutomationMetrics>>,
}

/// 自动化执行记录
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AutomationExecution {
    pub execution_id: String,
    pub rule_id: String,
    pub rule_name: String,
    pub trigger_event: Option<String>,
    pub actions_executed: Vec<String>,
    pub start_time: DateTime<Utc>,
    pub end_time: Option<DateTime<Utc>>,
    pub status: AutomationExecutionStatus,
    pub error_message: Option<String>,
    pub retry_count: usize,
}

/// 自动化执行状态
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum AutomationExecutionStatus {
    Running,
    Completed,
    Failed,
    Timeout,
    Cancelled,
}

/// 自动化指标
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AutomationMetrics {
    pub total_rules: usize,
    pub active_rules: usize,
    pub total_executions: u64,
    pub successful_executions: u64,
    pub failed_executions: u64,
    pub average_execution_time_ms: f64,
    pub last_updated: DateTime<Utc>,
}

impl Default for AutomationMetrics {
    fn default() -> Self {
        Self {
            total_rules: 0,
            active_rules: 0,
            total_executions: 0,
            successful_executions: 0,
            failed_executions: 0,
            average_execution_time_ms: 0.0,
            last_updated: Utc::now(),
        }
    }
}

impl AutomationEngine {
    pub fn new(
        check_interval_seconds: u64,
        event_bus: Arc<dyn EventBusInterface>,
    ) -> Result<Self> {
        Ok(Self {
            rules: Arc::new(RwLock::new(HashMap::new())),
            executors: Arc::new(RwLock::new(HashMap::new())),
            event_bus,
            check_interval: Duration::from_secs(check_interval_seconds),
            is_running: Arc::new(RwLock::new(false)),
            execution_history: Arc::new(RwLock::new(Vec::new())),
            metrics: Arc::new(RwLock::new(AutomationMetrics::default())),
        })
    }
    
    /// 启动自动化引擎
    pub async fn start(&self) -> Result<()> {
        info!("启动自动化引擎...");
        
        // 设置运行状态
        {
            let mut is_running = self.is_running.write()
                .expect("运行状态写入锁获取失败：无法更新运行状态");
            *is_running = true;
        }
        
        // 注册事件监听器
        self.register_event_listeners().await?;
        
        // 启动定时检查任务
        self.start_periodic_checks().await?;
        
        // 初始化默认规则
        self.initialize_default_rules().await?;
        
        info!("自动化引擎启动完成");
        Ok(())
    }
    
    /// 停止自动化引擎
    pub async fn stop(&self) -> Result<()> {
        info!("停止自动化引擎...");
        
        {
            let mut is_running = self.is_running.write()
                .expect("运行状态写入锁获取失败：无法更新运行状态");
            *is_running = false;
        }
        
        info!("自动化引擎已停止");
        Ok(())
    }
    
    /// 注册动作执行器
    pub fn register_executor(&self, executor: Arc<dyn AutomationActionExecutor>) {
        let executor_type = executor.executor_type().to_string();
        let mut executors = self.executors.write()
            .expect("执行器写入锁获取失败：无法更新执行器");
        executors.insert(executor_type.clone(), executor);
        info!("注册自动化执行器: {}", executor_type);
    }
    
    /// 添加自动化规则
    pub fn add_rule(&self, rule: AutomationRule) -> Result<()> {
        let rule_id = rule.id.clone();
        let rule_name = rule.name.clone();
        
        // 验证规则
        self.validate_rule(&rule)?;
        
        // 添加规则
        {
            let mut rules = self.rules.write()
                .expect("规则写入锁获取失败：无法更新规则");
            rules.insert(rule_id.clone(), rule);
        }
        
        // 更新指标
        self.update_metrics();
        
        info!("添加自动化规则: {} ({})", rule_name, rule_id);
        Ok(())
    }
    
    /// 移除自动化规则
    pub fn remove_rule(&self, rule_id: &str) -> Result<()> {
        let mut rules = self.rules.write()
            .expect("规则写入锁获取失败：无法移除规则");
        if rules.remove(rule_id).is_some() {
            // 更新指标
            drop(rules);
            self.update_metrics();
            info!("移除自动化规则: {}", rule_id);
            Ok(())
        } else {
            Err(crate::Error::NotFound(format!("自动化规则不存在: {}", rule_id)))
        }
    }
    
    /// 获取自动化规则
    pub fn get_rule(&self, rule_id: &str) -> Option<AutomationRule> {
        let rules = self.rules.read()
            .expect("规则读取锁获取失败：无法读取规则");
        rules.get(rule_id).cloned()
    }
    
    /// 列出所有规则
    pub fn list_rules(&self) -> Vec<AutomationRule> {
        let rules = self.rules.read()
            .expect("规则读取锁获取失败：无法读取规则");
        rules.values().cloned().collect()
    }
    
    /// 启用/禁用规则
    pub fn set_rule_status(&self, rule_id: &str, status: AutomationRuleStatus) -> Result<()> {
        let mut rules = self.rules.write().unwrap();
        if let Some(rule) = rules.get_mut(rule_id) {
            rule.status = status;
            rule.updated_at = Utc::now();
            info!("更新规则状态: {} -> {:?}", rule_id, rule.status);
            Ok(())
        } else {
            Err(crate::Error::NotFound(format!("自动化规则不存在: {}", rule_id)))
        }
    }
    
    /// 手动执行规则
    pub async fn execute_rule(&self, rule_id: &str, context: Option<AutomationContext>) -> Result<()> {
        let rule = {
            let rules = self.rules.read()
            .expect("规则读取锁获取失败：无法读取规则");
            rules.get(rule_id).cloned()
        };
        
        if let Some(rule) = rule {
            let context = context.unwrap_or_else(|| AutomationContext::new(rule_id.to_string(), None));
            self.execute_rule_internal(rule, context).await
        } else {
            Err(crate::Error::NotFound(format!("自动化规则不存在: {}", rule_id)))
        }
    }
    
    /// 注册事件监听器
    async fn register_event_listeners(&self) -> Result<()> {
        let handler = AutomationEventHandler {
            engine: self.create_weak_ref(),
        };
        
        // 监听所有事件
        self.event_bus.subscribe("*", Box::new(handler)).await?;
        
        info!("自动化事件监听器已注册");
        Ok(())
    }
    
    /// 启动定时检查
    async fn start_periodic_checks(&self) -> Result<()> {
        let engine = Arc::new(self.clone_ref());
        let check_interval = self.check_interval;
        
        tokio::spawn(async move {
            let mut interval = interval(check_interval);
            loop {
                interval.tick().await;
                
                // 检查是否还在运行
                {
                    let is_running = engine.is_running.read()
                        .expect("运行状态读取锁获取失败：无法检查运行状态");
                    if !*is_running {
                        break;
                    }
                }
                
                // 执行定时检查
                if let Err(e) = engine.perform_periodic_checks().await {
                    error!("定时检查失败: {}", e);
                }
            }
        });
        
        info!("定时检查任务已启动，间隔: {:?}", check_interval);
        Ok(())
    }
    
    /// 执行定时检查
    async fn perform_periodic_checks(&self) -> Result<()> {
        debug!("执行自动化定时检查");
        
        // 检查基于时间调度的规则
        self.check_scheduled_rules().await?;
        
        // 检查基于阈值的规则
        self.check_threshold_rules().await?;
        
        // 清理历史记录
        self.cleanup_execution_history();
        
        // 更新指标
        self.update_metrics();
        
        Ok(())
    }
    
    /// 检查时间调度规则
    async fn check_scheduled_rules(&self) -> Result<()> {
        let rules = {
            let rules = self.rules.read()
            .expect("规则读取锁获取失败：无法读取规则");
            rules.values()
                .filter(|rule| rule.status == AutomationRuleStatus::Active)
                .filter(|rule| matches!(rule.trigger, AutomationTrigger::Schedule { .. }))
                .cloned()
                .collect::<Vec<_>>()
        };
        
        for rule in rules {
            if self.should_execute_scheduled_rule(&rule) {
                let context = AutomationContext::new(rule.id.clone(), None);
                if let Err(e) = self.execute_rule_internal(rule, context).await {
                    error!("定时规则执行失败: {}", e);
                }
            }
        }
        
        Ok(())
    }
    
    /// 检查阈值规则
    async fn check_threshold_rules(&self) -> Result<()> {
        let rules = {
            let rules = self.rules.read()
            .expect("规则读取锁获取失败：无法读取规则");
            rules.values()
                .filter(|rule| rule.status == AutomationRuleStatus::Active)
                .filter(|rule| matches!(rule.trigger, AutomationTrigger::Threshold { .. }))
                .cloned()
                .collect::<Vec<_>>()
        };
        
        for rule in rules {
            if self.should_execute_threshold_rule(&rule).await? {
                let context = AutomationContext::new(rule.id.clone(), None);
                if let Err(e) = self.execute_rule_internal(rule, context).await {
                    error!("阈值规则执行失败: {}", e);
                }
            }
        }
        
        Ok(())
    }
    
    /// 判断时间调度规则是否应该执行
    fn should_execute_scheduled_rule(&self, rule: &AutomationRule) -> bool {
        // 简化实现：每次检查都执行
        // 实际实现应该解析cron表达式
        match &rule.trigger {
            AutomationTrigger::Schedule { .. } => {
                // 检查是否距离上次执行足够长时间
                if let Some(last_executed) = rule.last_executed_at {
                    let elapsed = Utc::now().signed_duration_since(last_executed);
                    elapsed.num_seconds() >= self.check_interval.as_secs() as i64
                } else {
                    true
                }
            },
            _ => false,
        }
    }
    
    /// 判断阈值规则是否应该执行
    async fn should_execute_threshold_rule(&self, _rule: &AutomationRule) -> Result<bool> {
        // 简化实现：总是返回false
        // 实际实现应该检查指标值
        Ok(false)
    }
    
    /// 执行规则内部逻辑
    async fn execute_rule_internal(&self, mut rule: AutomationRule, context: AutomationContext) -> Result<()> {
        let execution_id = context.execution_id.clone();
        let start_time = Instant::now();
        
        info!("开始执行自动化规则: {} ({})", rule.name, rule.id);
        
        // 创建执行记录
        let mut execution = AutomationExecution {
            execution_id: execution_id.clone(),
            rule_id: rule.id.clone(),
            rule_name: rule.name.clone(),
            trigger_event: context.trigger_event.as_ref().map(|e| e.event_type.clone()),
            actions_executed: Vec::new(),
            start_time: context.start_time,
            end_time: None,
            status: AutomationExecutionStatus::Running,
            error_message: None,
            retry_count: 0,
        };
        
        // 添加到执行历史
        {
            let mut history = self.execution_history.write().unwrap();
            history.push(execution.clone());
        }
        
        // 执行动作
        let mut retry_count = 0;
        let mut last_error = None;
        
        while retry_count <= rule.retry_policy.max_retries {
            match self.execute_actions(&rule.actions, &context).await {
                Ok(executed_actions) => {
                    // 成功执行
                    execution.actions_executed = executed_actions;
                    execution.status = AutomationExecutionStatus::Completed;
                    execution.end_time = Some(Utc::now());
                    
                    // 更新规则统计
                    let rule_name = rule.name.clone();
                    rule.update_execution_stats(true);
                    
                    // 更新规则
                    {
                        let mut rules = self.rules.write()
                .expect("规则写入锁获取失败：无法更新规则");
                        rules.insert(rule.id.clone(), rule);
                    }
                    
                    // 更新执行记录
                    self.update_execution_record(execution);
                    
                    let duration = start_time.elapsed();
                    info!("自动化规则执行成功: {} (耗时: {:?})", rule_name, duration);
                    return Ok(());
                },
                Err(e) => {
                    let rule_name = rule.name.clone();
                    error!("自动化规则执行失败: {}, 错误: {}", rule_name, e);
                    last_error = Some(e);
                    retry_count += 1;
                    
                    if retry_count <= rule.retry_policy.max_retries {
                        let delay = self.calculate_retry_delay(&rule.retry_policy, retry_count);
                        let rule_name = rule.name.clone();
                        warn!("重试自动化规则: {} (第{}次重试，延迟{}秒)", rule_name, retry_count, delay.as_secs());
                        tokio::time::sleep(delay).await;
                    }
                }
            }
        }
        
        // 所有重试都失败了
        execution.status = AutomationExecutionStatus::Failed;
        let error_message = last_error.as_ref().map(|e| e.to_string());
        execution.error_message = error_message;
        execution.end_time = Some(Utc::now());
        execution.retry_count = retry_count;
        
        // 更新规则统计
        rule.update_execution_stats(false);
        
        // 更新规则
        {
            let mut rules = self.rules.write()
                .expect("规则写入锁获取失败：无法更新规则");
            rules.insert(rule.id.clone(), rule);
        }
        
        // 更新执行记录
        self.update_execution_record(execution);
        
        match last_error {
            Some(error) => Err(error),
            None => Err(crate::Error::ExecutionError("自动化规则执行失败，原因未知".to_string())),
        }
    }
    
    /// 执行动作列表
    async fn execute_actions(&self, actions: &[AutomationAction], context: &AutomationContext) -> Result<Vec<String>> {
        let mut executed_actions = Vec::new();
        
        for action in actions {
            let action_type = self.get_action_type(action);
            let action_type_clone = action_type.clone();
            
            // 查找对应的执行器
            let executor = {
                let executors = self.executors.read().unwrap();
                executors.get(&action_type_clone).cloned()
            };
            
            if let Some(executor) = executor {
                // 验证动作
                if !executor.validate(action).await? {
                    return Err(crate::Error::Validation(format!("动作验证失败: {:?}", action)));
                }
                
                // 执行动作
                executor.execute(action, context).await?;
                executed_actions.push(action_type_clone.clone());
                
                debug!("执行动作成功: {}", action_type_clone);
            } else {
                return Err(crate::Error::NotFound(format!("未找到动作执行器: {}", action_type_clone)));
            }
        }
        
        Ok(executed_actions)
    }
    
    /// 获取动作类型
    fn get_action_type(&self, action: &AutomationAction) -> String {
        match action {
            AutomationAction::StartTraining { .. } => "start_training".to_string(),
            AutomationAction::UpdateModel { .. } => "update_model".to_string(),
            AutomationAction::ExecuteAlgorithm { .. } => "execute_algorithm".to_string(),
            AutomationAction::SyncStorage { .. } => "sync_storage".to_string(),
            AutomationAction::SendNotification { .. } => "send_notification".to_string(),
            AutomationAction::CreateWorkflow { .. } => "create_workflow".to_string(),
            AutomationAction::ScaleResources { .. } => "scale_resources".to_string(),
        }
    }
    
    /// 计算重试延迟
    fn calculate_retry_delay(&self, policy: &RetryPolicy, retry_count: usize) -> Duration {
        let base_delay = Duration::from_secs(policy.retry_delay_seconds);
        
        if policy.exponential_backoff {
            let multiplier = 2_u64.pow(retry_count.saturating_sub(1) as u32);
            let delay_seconds = (base_delay.as_secs() * multiplier)
                .min(policy.max_delay_seconds);
            Duration::from_secs(delay_seconds)
        } else {
            base_delay
        }
    }
    
    /// 更新执行记录
    fn update_execution_record(&self, execution: AutomationExecution) {
        let mut history = self.execution_history.write().unwrap();
        if let Some(pos) = history.iter().position(|e| e.execution_id == execution.execution_id) {
            history[pos] = execution;
        }
    }
    
    /// 清理执行历史
    fn cleanup_execution_history(&self) {
        let mut history = self.execution_history.write().unwrap();
        let cutoff_time = Utc::now() - chrono::Duration::hours(24); // 保留24小时内的记录
        
        history.retain(|execution| execution.start_time > cutoff_time);
    }
    
    /// 更新指标
    fn update_metrics(&self) {
        let mut metrics = self.metrics.write().unwrap();
        
        // 统计规则数量
        let rules = self.rules.read()
            .expect("规则读取锁获取失败：无法读取规则");
        metrics.total_rules = rules.len();
        metrics.active_rules = rules.values()
            .filter(|rule| rule.status == AutomationRuleStatus::Active)
            .count();
        
        // 统计执行记录
        let history = self.execution_history.read().unwrap();
        metrics.total_executions = history.len() as u64;
        metrics.successful_executions = history.iter()
            .filter(|exec| exec.status == AutomationExecutionStatus::Completed)
            .count() as u64;
        metrics.failed_executions = history.iter()
            .filter(|exec| exec.status == AutomationExecutionStatus::Failed)
            .count() as u64;
        
        // 计算平均执行时间
        let total_duration: i64 = history.iter()
            .filter_map(|exec| {
                if let Some(end_time) = exec.end_time {
                    Some(end_time.signed_duration_since(exec.start_time).num_milliseconds())
                } else {
                    None
                }
            })
            .sum();
        
        if metrics.total_executions > 0 {
            metrics.average_execution_time_ms = total_duration as f64 / metrics.total_executions as f64;
        }
        
        metrics.last_updated = Utc::now();
    }
    
    /// 验证规则
    fn validate_rule(&self, rule: &AutomationRule) -> Result<()> {
        // 检查规则名称
        if rule.name.is_empty() {
            return Err(crate::Error::InvalidParameter("规则名称不能为空".to_string()));
        }
        
        // 检查动作列表
        if rule.actions.is_empty() {
            return Err(crate::Error::InvalidParameter("规则必须包含至少一个动作".to_string()));
        }
        
        // 检查超时设置
        if rule.timeout_seconds == 0 {
            return Err(crate::Error::InvalidParameter("超时时间必须大于0".to_string()));
        }
        
        Ok(())
    }
    
    /// 初始化默认规则
    async fn initialize_default_rules(&self) -> Result<()> {
        // 训练完成自动更新模型规则
        let training_to_model_rule = AutomationRule::new(
            "TrainingToModel".to_string(),
            AutomationRuleType::TrainingToModel,
            AutomationTrigger::Event {
                event_pattern: "training.completed".to_string(),
                conditions: HashMap::new(),
            },
            vec![
                AutomationAction::UpdateModel {
                    model_id: "${model_id}".to_string(),
                    parameters: HashMap::new(),
                },
                AutomationAction::SyncStorage {
                    source: "model".to_string(),
                    target: "storage".to_string(),
                    sync_type: "incremental".to_string(),
                },
            ],
        ).with_description("训练完成后自动更新模型参数".to_string());
        
        self.add_rule(training_to_model_rule)?;
        
        // 算法执行完成自动更新模型规则
        let algorithm_to_model_rule = AutomationRule::new(
            "AlgorithmToModel".to_string(),
            AutomationRuleType::AlgorithmToModel,
            AutomationTrigger::Event {
                event_pattern: "algorithm.executed".to_string(),
                conditions: HashMap::new(),
            },
            vec![
                AutomationAction::UpdateModel {
                    model_id: "${target_model}".to_string(),
                    parameters: HashMap::new(),
                },
            ],
        ).with_description("算法执行完成后自动更新目标模型".to_string());
        
        self.add_rule(algorithm_to_model_rule)?;
        
        info!("默认自动化规则初始化完成");
        Ok(())
    }
    
    /// 获取自动化指标
    pub fn get_metrics(&self) -> AutomationMetrics {
        let metrics = self.metrics.read().unwrap();
        metrics.clone()
    }
    
    /// 获取执行历史
    pub fn get_execution_history(&self, limit: Option<usize>) -> Vec<AutomationExecution> {
        let history = self.execution_history.read().unwrap();
        if let Some(limit) = limit {
            history.iter().rev().take(limit).cloned().collect()
        } else {
            history.clone()
        }
    }
    
    /// 创建引用副本
    fn clone_ref(&self) -> Self {
        Self {
            rules: self.rules.clone(),
            executors: self.executors.clone(),
            event_bus: self.event_bus.clone(),
            check_interval: self.check_interval,
            is_running: self.is_running.clone(),
            execution_history: self.execution_history.clone(),
            metrics: self.metrics.clone(),
        }
    }
    
    /// 创建弱引用
    fn create_weak_ref(&self) -> AutomationEngineRef {
        AutomationEngineRef {
            engine_id: "automation_engine".to_string(),
        }
    }
}

/// 自动化引擎弱引用
#[derive(Clone)]
pub struct AutomationEngineRef {
    pub engine_id: String,
}

/// 自动化事件处理器
pub struct AutomationEventHandler {
    pub engine: AutomationEngineRef,
}

#[async_trait]
impl crate::core::EventHandler for AutomationEventHandler {
    async fn handle(&self, event: &CoreEvent) -> Result<()> {
        debug!("处理自动化事件: {}", event.event_type);
        
        // 这里应该通过引擎引用来处理事件
        // 简化实现，实际应该通过适当的回调机制
        
        Ok(())
    }
}

/// 默认动作执行器
pub struct DefaultActionExecutor {
    executor_type: String,
}

impl DefaultActionExecutor {
    pub fn new(executor_type: String) -> Self {
        Self { executor_type }
    }
}

#[async_trait]
impl AutomationActionExecutor for DefaultActionExecutor {
    fn executor_type(&self) -> &str {
        &self.executor_type
    }
    
    async fn execute(&self, action: &AutomationAction, context: &AutomationContext) -> Result<()> {
        debug!("执行默认动作: {:?} (上下文: {})", action, context.execution_id);
        
        // 模拟动作执行
        tokio::time::sleep(Duration::from_millis(100)).await;
        
        match action {
            AutomationAction::StartTraining { model_id, .. } => {
                info!("启动模型训练: {}", model_id);
            },
            AutomationAction::UpdateModel { model_id, .. } => {
                info!("更新模型参数: {}", model_id);
            },
            AutomationAction::ExecuteAlgorithm { algorithm_id, target_model, .. } => {
                info!("执行算法 {} 到模型 {}", algorithm_id, target_model);
            },
            AutomationAction::SyncStorage { source, target, .. } => {
                info!("同步存储: {} -> {}", source, target);
            },
            AutomationAction::SendNotification { recipients, message, .. } => {
                info!("发送通知给 {:?}: {}", recipients, message);
            },
            AutomationAction::CreateWorkflow { workflow_name, .. } => {
                info!("创建工作流: {}", workflow_name);
            },
            AutomationAction::ScaleResources { resource_type, scale_factor } => {
                info!("缩放资源 {} 比例 {}", resource_type, scale_factor);
            },
        }
        
        Ok(())
    }
    
    async fn validate(&self, _action: &AutomationAction) -> Result<bool> {
        // 默认验证总是通过
        Ok(true)
    }
} 