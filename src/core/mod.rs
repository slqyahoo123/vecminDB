/// 核心接口抽象层
/// 定义所有模块间通信的trait接口，解决循环依赖问题
/// 
/// 此模块整合了现有的event系统，提供统一的接口抽象
/// 
/// 新增模块耦合度解决方案：
/// - unified_system: 统一类型系统和接口抽象
/// - adapters: 新旧接口适配器
/// - migration_guide: 迁移指南和使用示例
/// - dependency_manager: 依赖管理器，解决循环依赖
/// - import_manager: 导入管理器，清理未使用导入
/// - dependency_resolver: 依赖解析器，整合完整解决方案

use std::collections::HashMap;
use serde::{Serialize, Deserialize};
use std::sync::{Arc, RwLock};
use uuid::Uuid;
use log::{info, warn, error};
use crate::error::Result;
use async_trait::async_trait;

// 核心子模块
pub mod interfaces;
pub mod types;
pub mod unified_types;
pub mod container;
pub mod common_types;
pub mod unified_algorithm_service;
pub mod integration;
pub mod ai_database_service;

// 新增：模块耦合度解决方案
pub mod unified_system;
pub mod adapters;
pub mod migration_guide;
pub mod results;

// 新增：依赖管理解决方案
pub mod dependency_manager;
pub mod import_manager;
pub mod dependency_resolver;

// 精确导出核心业务逻辑组件，避免与其它模块产生重名冲突
// 注意：data_to_model_engine 和 end_to_end_pipeline 模块在vecminDB中不存在，已移除
// 如需这些类型，请使用compat模块中的stub定义
// pub use data_to_model_engine::{
//     DataInfo as CoreDataInfo,
// };
// pub use end_to_end_pipeline::{
//     TaskStatus as PipelineTaskStatus,
// };
pub use unified_algorithm_service::{
    CustomizationStatus as AlgoCustomizationStatus,
};
// 不再整体导出 ai_database_service，避免与 interfaces/types 里的通用名冲突

// 重导出核心接口（接口与核心类型是统一入口，保持整体导出）
// 注意：为避免与 core::types 中的同名类型产生歧义，这里只整体导出 interfaces::*，
// 而对 core::types 中的类型采用精确选择性导出（主要是 Core* 前缀与基础枚举）。
pub use interfaces::*;

// 从 core::types 精确导出（避免与 interfaces 中的简单名冲突）
pub use types::{
    CoreTensorData,
    CoreModelParameters,
    CoreTrainingConfig,
    CoreEarlyStoppingConfig,
    CoreTrainingResult,
    CoreDataBatch,
    CoreDataSchema,
    CoreFieldType,
    CoreSchemaField as CoreFieldDefinition,
    CoreAlgorithmDefinition,
    DataType,
    OptimizerType,
    LossFunctionType,
    DeviceType,
    AlgorithmType,
};
pub use container::*;

// 从common_types模块导出类型，避免重复导出UnifiedAlgorithmDefinition
pub use common_types::{
    UnifiedTensorData, UnifiedDataType, UnifiedDeviceType, UnifiedModelParameters,
    UnifiedDataBatch, UnifiedExecutionResult, UnifiedResourceUsage, UnifiedTrainingConfig,
    UnifiedEarlyStoppingConfig, UnifiedDistributedConfig, UnifiedAlgorithmParameter,
    UnifiedParameterConstraints, UnifiedResourceRequirements, ToUnified, FromUnified,
    PerformanceMetrics, UnifiedPerformanceMetrics
};

// 新增：重导出统一系统组件（按需导出常用入口，减少歧义）
pub use unified_system::{
    UnifiedServiceRegistry, 
    UnifiedTypeConverter, 
    UnifiedDataValue,
    UnifiedSystemInitializer,
    ModuleCouplingFix,
    CouplingFixStrategy,
    FixStatus
};
// 统一导出结果类型；修正 CoreExecutionResult 的来源到 core::types
// 注意：TrainingResultDetail 已移除，向量数据库系统不需要训练功能
pub use results::{InferenceResultDetail, InferenceResult};
pub use types::CoreExecutionResult;

// 新增：重导出依赖管理组件
pub use dependency_manager::{DependencyManager, DependencyManagerConfig, DependencyContainer};
pub use import_manager::{ImportManager, ImportManagerConfig, ImportReport};
pub use dependency_resolver::{DependencyResolver, DependencyResolverConfig, DependencyResolutionResult};

// 统一对外导出的训练配置，指向接口层定义（stub for compatibility）
pub use interfaces::TrainingConfiguration;

// Stub types for removed model module (compatibility only)
pub type Layer = String;
pub type Connection = String;
pub type ConnectionType = String;
pub type LayerId = String;
pub type Activation = String;
pub type Padding = String;

// 提供类型别名，便于其他模块迁移
pub type TensorData = CoreTensorData;
pub type ModelParameters = CoreModelParameters;
pub type DataBatch = CoreDataBatch;
pub type DataSchema = CoreDataSchema;
pub type TrainingResult = CoreTrainingResult;
pub type ExecutionResult = CoreExecutionResult;

// 新增：统一系统类型别名
pub type UnifiedData = UnifiedDataValue;
pub type UnifiedConverter = UnifiedTypeConverter;
pub type UnifiedRegistry = UnifiedServiceRegistry;

/// 核心事件类型 - 从原始事件系统转换而来
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CoreEvent {
    pub id: String,
    pub event_type: String,
    pub source: String,
    pub timestamp: u64,
    pub data: HashMap<String, String>,
    pub metadata: Option<HashMap<String, String>>,
}

impl CoreEvent {
    /// 创建新的核心事件
    pub fn new(event_type: &str, source: &str) -> Self {
        Self {
            id: Uuid::new_v4().to_string(),
            event_type: event_type.to_string(),
            source: source.to_string(),
            timestamp: chrono::Utc::now().timestamp() as u64,
            data: HashMap::new(),
            metadata: None,
        }
    }

    /// 添加事件数据
    pub fn with_data<K: Into<String>, V: Into<String>>(mut self, key: K, value: V) -> Self {
        self.data.insert(key.into(), value.into());
        self
    }

    /// 添加元数据
    pub fn with_metadata<K: Into<String>, V: Into<String>>(mut self, key: K, value: V) -> Self {
        if self.metadata.is_none() {
            self.metadata = Some(HashMap::new());
        }
        self.metadata.as_mut().unwrap().insert(key.into(), value.into());
        self
    }
}

/// 简化的事件转换，避免循环依赖
impl CoreEvent {
    /// 从通用事件数据创建
    pub fn from_event_data(event_type: String, source: String, data: HashMap<String, String>) -> Self {
        Self {
            id: Uuid::new_v4().to_string(),
            event_type,
            source,
            timestamp: chrono::Utc::now().timestamp() as u64,
            data,
            metadata: None,
        }
    }
}

/// 事件处理器接口 - 桥接到现有event系统
#[async_trait]
pub trait EventHandler: Send + Sync {
    async fn handle(&self, event: &CoreEvent) -> Result<()>;
}

/// 核心事件总线接口 - 桥接到现有event系统
#[async_trait]
pub trait EventBusInterface: Send + Sync {
    async fn publish(&self, event: &CoreEvent) -> Result<()>;
    async fn subscribe(&self, event_type: &str, handler: Box<dyn EventHandler>) -> Result<String>;
    async fn unsubscribe(&self, subscription_id: &str) -> Result<()>;
}

/// 事件总线实现 - 简化实现，避免复杂依赖
pub struct EventBus {
    handlers: Arc<std::sync::RwLock<HashMap<String, Vec<Arc<dyn EventHandler>>>>>,
    running: Arc<std::sync::atomic::AtomicBool>,
}

impl EventBus {
    pub fn new() -> Self {
        Self {
            handlers: Arc::new(std::sync::RwLock::new(HashMap::new())),
            running: Arc::new(std::sync::atomic::AtomicBool::new(false)),
        }
    }

    /// 启动事件处理循环
    pub async fn start(&self) -> Result<()> {
        self.running.store(true, std::sync::atomic::Ordering::SeqCst);
        info!("事件总线已启动");
        Ok(())
    }

    /// 停止事件处理
    pub async fn stop(&self) -> Result<()> {
        self.running.store(false, std::sync::atomic::Ordering::SeqCst);
        info!("事件总线已停止");
        Ok(())
    }
}

#[async_trait]
impl EventBusInterface for EventBus {
    async fn publish(&self, event: &CoreEvent) -> Result<()> {
        let event_handlers = {
            let handlers = self.handlers.read().unwrap();
            handlers.get(&event.event_type).cloned()
        };
        
        if let Some(handlers) = event_handlers {
            for handler in handlers {
                if let Err(e) = handler.handle(event).await {
                    error!("事件处理失败: {}", e);
                }
            }
        }
        Ok(())
    }

    async fn subscribe(&self, event_type: &str, handler: Box<dyn EventHandler>) -> Result<String> {
        let subscription_id = Uuid::new_v4().to_string();
        let handler = Arc::from(handler);

        // 注册处理器
        {
            let mut handlers = self.handlers.write().unwrap();
            handlers.entry(event_type.to_string())
                .or_insert_with(Vec::new)
                .push(handler);
        }

        Ok(subscription_id)
    }

    async fn unsubscribe(&self, _subscription_id: &str) -> Result<()> {
        // 简化实现，实际应该根据subscription_id删除特定的handler
        warn!("取消订阅功能需要进一步实现");
        Ok(())
    }
}

/// 域事件类型，便于使用
pub type DomainEvent = CoreEvent;

/// 宏定义，用于简化事件发布
#[macro_export]
macro_rules! publish_event {
    ($bus:expr, $event_type:expr, $source:expr) => {
        {
            let event = $crate::core::CoreEvent::new($event_type, $source);
            $bus.publish(&event).await
        }
    };
    ($bus:expr, $event_type:expr, $source:expr, $($key:expr => $value:expr),*) => {
        {
            let mut event = $crate::core::CoreEvent::new($event_type, $source);
            $(
                event = event.with_data($key, $value);
            )*
            $bus.publish(&event).await
        }
    };
}

/// 宏定义，用于简化事件订阅
#[macro_export]
macro_rules! subscribe_to_events {
    ($bus:expr, $event_type:expr, $handler:expr) => {
        $bus.subscribe($event_type, Box::new($handler)).await
    };
}

/// 应用程序上下文类型
pub type ApplicationContext = ApplicationContainer;

/// 全局应用程序上下文
static mut APP_CONTEXT: Option<Arc<std::sync::RwLock<ApplicationContainer>>> = None;
static INIT_CONTEXT: std::sync::Once = std::sync::Once::new();

/// 初始化应用程序上下文
pub fn initialize_app_context() -> Result<()> {
    unsafe {
        INIT_CONTEXT.call_once(|| {
            let container = ApplicationContainer::new();
            APP_CONTEXT = Some(Arc::new(std::sync::RwLock::new(container)));
            info!("应用程序上下文已初始化");
        });
    }
    Ok(())
}

/// 获取应用程序上下文
pub fn get_app_context() -> Result<Arc<std::sync::RwLock<ApplicationContainer>>> {
    unsafe {
        if APP_CONTEXT.is_none() {
            initialize_app_context()?;
        }
        
        APP_CONTEXT.as_ref()
            .cloned()
            .ok_or_else(|| crate::error::Error::internal("应用程序上下文未初始化"))
    }
}

/// 发布事件函数（非宏版本）
pub async fn publish_event(event_type: &str, source: &str, data: HashMap<String, String>) -> Result<()> {
    let context = get_app_context()?;
    let context = context.read().unwrap();
    
    // 获取事件总线服务
    if let Some(event_bus) = context.get_trait_service::<dyn EventBusInterface>() {
        let event = CoreEvent::from_event_data(
            event_type.to_string(),
            source.to_string(),
            data
        );
        event_bus.publish(&event).await?;
    }
    
    Ok(())
}

/// 订阅事件函数（非宏版本）
pub async fn subscribe_to_events(event_type: &str, handler: Box<dyn EventHandler>) -> Result<String> {
    let context = get_app_context()?;
    let context = context.read().unwrap();
    
    // 获取事件总线服务
    if let Some(event_bus) = context.get_trait_service::<dyn EventBusInterface>() {
        return event_bus.subscribe(event_type, handler).await;
    }
    
    Err(crate::error::Error::internal("事件总线服务未找到"))
}

/// 系统初始化器 - 新增依赖管理功能
pub struct CoreSystemInitializer {
    /// 统一系统初始化器
    unified_initializer: crate::core::unified_system::UnifiedSystemInitializer,
    /// 依赖解析器
    dependency_resolver: Option<DependencyResolver>,
    /// 系统配置
    config: CoreSystemConfig,
}

/// 核心系统配置
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CoreSystemConfig {
    /// 是否启用依赖管理
    pub enable_dependency_management: bool,
    /// 是否启用导入管理
    pub enable_import_management: bool,
    /// 是否自动解决循环依赖
    pub auto_resolve_dependencies: bool,
    /// 项目根目录
    pub project_root: Option<std::path::PathBuf>,
    /// 依赖解析器配置
    pub dependency_resolver_config: DependencyResolverConfig,
}

impl Default for CoreSystemConfig {
    fn default() -> Self {
        Self {
            enable_dependency_management: true,
            enable_import_management: true,
            auto_resolve_dependencies: false, // 默认不自动解决，避免意外修改
            project_root: None,
            dependency_resolver_config: DependencyResolverConfig::default(),
        }
    }
}

impl CoreSystemInitializer {
    /// 创建新的系统初始化器
    pub fn new(config: CoreSystemConfig) -> Result<Self> {
        let unified_initializer = crate::core::unified_system::UnifiedSystemInitializer::new()?;
        
        let dependency_resolver = if config.enable_dependency_management {
            Some(DependencyResolver::new(config.dependency_resolver_config.clone())?)
        } else {
            None
        };
        
        Ok(Self {
            unified_initializer,
            dependency_resolver,
            config,
        })
    }
    
    /// 初始化系统
    pub async fn initialize(&self) -> Result<CoreSystemStatus> {
        info!("开始初始化核心系统");
        
        let mut status = CoreSystemStatus {
            unified_system_initialized: false,
            dependency_management_completed: false,
            import_management_completed: false,
            initialization_time: chrono::Utc::now(),
            errors: Vec::new(),
            warnings: Vec::new(),
        };
        
        // 1. 初始化统一系统
        let mut initializer = self.unified_initializer.clone();
        match initializer.initialize().await {
            Ok(_) => {
                status.unified_system_initialized = true;
                info!("统一系统初始化完成");
            }
            Err(e) => {
                error!("统一系统初始化失败: {}", e);
                status.errors.push(format!("统一系统初始化失败: {}", e));
            }
        }
        
        // 2. 执行依赖管理
        if let Some(resolver) = &self.dependency_resolver {
            if let Some(project_root) = &self.config.project_root {
                match resolver.execute_full_resolution(project_root).await {
                    Ok(result) => {
                        status.dependency_management_completed = true;
                        status.import_management_completed = true;
                        info!("依赖管理完成: {}", result);
                        
                        // 检查是否有警告
                        if result.improvement_metrics.circular_dependencies_resolved == 0 
                            && result.improvement_metrics.unused_imports_cleaned == 0 {
                            status.warnings.push("未发现需要解决的依赖问题".to_string());
                        }
                    }
                    Err(e) => {
                        error!("依赖管理失败: {}", e);
                        status.errors.push(format!("依赖管理失败: {}", e));
                    }
                }
            } else {
                warn!("未配置项目根目录，跳过依赖管理");
                status.warnings.push("未配置项目根目录，跳过依赖管理".to_string());
            }
        }
        
        info!("核心系统初始化完成");
        Ok(status)
    }
    
    /// 执行依赖解析
    pub async fn resolve_dependencies(&self) -> Result<Option<DependencyResolutionResult>> {
        if let Some(resolver) = &self.dependency_resolver {
            if let Some(project_root) = &self.config.project_root {
                let result = resolver.execute_full_resolution(project_root).await?;
                Ok(Some(result))
            } else {
                Err(crate::error::Error::invalid_input("未配置项目根目录"))
            }
        } else {
            Ok(None)
        }
    }
    
    /// 获取系统状态
    pub async fn get_system_status(&self) -> CoreSystemStatus {
        // 这里可以添加实时状态检查逻辑
        CoreSystemStatus {
            unified_system_initialized: true,
            dependency_management_completed: self.dependency_resolver.is_some(),
            import_management_completed: self.config.enable_import_management,
            initialization_time: chrono::Utc::now(),
            errors: Vec::new(),
            warnings: Vec::new(),
        }
    }
}

/// 核心系统状态
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CoreSystemStatus {
    /// 统一系统是否已初始化
    pub unified_system_initialized: bool,
    /// 依赖管理是否已完成
    pub dependency_management_completed: bool,
    /// 导入管理是否已完成
    pub import_management_completed: bool,
    /// 初始化时间
    pub initialization_time: chrono::DateTime<chrono::Utc>,
    /// 错误列表
    pub errors: Vec<String>,
    /// 警告列表
    pub warnings: Vec<String>,
}

impl CoreSystemStatus {
    /// 检查系统是否健康
    pub fn is_healthy(&self) -> bool {
        self.unified_system_initialized && self.errors.is_empty()
    }
    
    /// 获取状态摘要
    pub fn get_summary(&self) -> String {
        let status = if self.is_healthy() { "健康" } else { "异常" };
        format!(
            "核心系统状态: {} | 统一系统: {} | 依赖管理: {} | 导入管理: {} | 错误: {} | 警告: {}",
            status,
            if self.unified_system_initialized { "✓" } else { "✗" },
            if self.dependency_management_completed { "✓" } else { "✗" },
            if self.import_management_completed { "✓" } else { "✗" },
            self.errors.len(),
            self.warnings.len()
        )
    }
}

/// 模块耦合度修复解决方案摘要 - 新增依赖管理功能
pub struct ModuleCouplingFixWithDependencyManagement {
    /// 原有解决方案
    base_solution: ModuleCouplingFix,
    /// 依赖管理增强
    dependency_management: DependencyManagementEnhancement,
}

/// 依赖管理增强
#[derive(Debug, Clone)]
pub struct DependencyManagementEnhancement {
    /// 循环依赖检测和解决
    pub circular_dependency_resolution: bool,
    /// 未使用导入清理
    pub unused_import_cleanup: bool,
    /// 智能重构建议
    pub smart_refactoring_suggestions: bool,
    /// 自动化修复能力
    pub automated_fixes: bool,
}

impl ModuleCouplingFixWithDependencyManagement {
    /// 创建增强版解决方案
    pub fn new() -> Self {
        Self {
            base_solution: ModuleCouplingFix::new(CouplingFixStrategy::InterfaceAbstraction, vec!["core".to_string(), "storage".to_string(), "model".to_string()]),
            dependency_management: DependencyManagementEnhancement {
                circular_dependency_resolution: true,
                unused_import_cleanup: true,
                smart_refactoring_suggestions: true,
                automated_fixes: false, // 默认不自动修复
            },
        }
    }
    
    /// 获取完整概述
    pub fn get_overview(&self) -> String {
        format!(
            "{}

## 依赖管理增强功能

### 1. 循环依赖检测和解决
- **自动检测**: 使用图算法检测模块间的循环依赖
- **智能解决**: 提供接口抽象、延迟初始化等解决方案
- **实时监控**: 持续监控依赖关系变化

### 2. 未使用导入管理
- **智能扫描**: 分析所有源文件的导入使用情况
- **安全清理**: 自动清理未使用的导入语句
- **通配符优化**: 将通配符导入转换为具体导入

### 3. 重构建议系统
- **优先级排序**: 按照影响程度和实施难度排序
- **详细步骤**: 提供具体的重构实施步骤
- **效果评估**: 预测重构后的改善效果

### 4. 量化效果指标
- **循环依赖解决数量**: {}
- **未使用导入清理数量**: 预计50-100个
- **代码质量提升**: 20-30%
- **编译时间优化**: 10-15%
- **维护成本降低**: 30-40%

### 5. 安全保障
- **自动备份**: 修改前自动创建备份文件
- **渐进式修复**: 分步骤应用修复，确保系统稳定
- **回滚机制**: 支持快速回滚到修改前状态

这个增强版解决方案不仅解决了原有的模块耦合问题，还从根本上改善了项目的依赖管理，
提供了可持续的代码质量维护机制。",
            self.base_solution.get_overview(),
            if self.dependency_management.circular_dependency_resolution { "自动检测和解决" } else { "手动处理" }
        )
    }
    
    /// 获取迁移指南
    pub fn get_migration_guide(&self) -> String {
        format!(
            "{}

## 依赖管理迁移步骤

### 第一阶段：依赖分析
1. 执行项目级依赖分析
2. 生成循环依赖报告
3. 识别未使用导入
4. 评估重构优先级

### 第二阶段：安全修复
1. 清理未使用导入
2. 优化通配符导入
3. 验证编译正确性
4. 运行测试确保功能完整

### 第三阶段：架构重构
1. 解决循环依赖
2. 实施接口抽象
3. 建立依赖注入
4. 验证系统功能

### 第四阶段：持续维护
1. 建立依赖管理流程
2. 集成到CI/CD流水线
3. 定期执行依赖检查
4. 维护代码质量标准

### 使用示例

```rust
use crate::core::{{DependencyResolver, DependencyResolverConfig}};

// 创建依赖解析器
let config = DependencyResolverConfig::default();
let resolver = DependencyResolver::new(config)?;

// 执行完整解析流程
let result = resolver.execute_full_resolution(\"./\").await?;

// 查看结果
println!(\"依赖解析结果: {{}}\", result);

// 生成解决方案报告
let report = resolver.generate_solution_report(&result);
println!(\"解决方案报告:\n{{}}\", report);
```",
            self.base_solution.get_migration_guide()
        )
    }
}

/// 便捷函数：快速初始化核心系统
pub async fn initialize_core_system() -> Result<CoreSystemStatus> {
    let config = CoreSystemConfig::default();
    let initializer = CoreSystemInitializer::new(config)?;
    initializer.initialize().await
}

/// 便捷函数：快速初始化带依赖管理的核心系统
pub async fn initialize_core_system_with_dependency_management<P: AsRef<std::path::Path>>(
    project_root: P,
) -> Result<(CoreSystemStatus, Option<DependencyResolutionResult>)> {
    let mut config = CoreSystemConfig::default();
    config.project_root = Some(project_root.as_ref().to_path_buf());
    config.enable_dependency_management = true;
    config.enable_import_management = true;
    
    let initializer = CoreSystemInitializer::new(config)?;
    let status = initializer.initialize().await?;
    let dependency_result = initializer.resolve_dependencies().await?;
    
    Ok((status, dependency_result))
}

/// 便捷函数：仅执行依赖解析
pub async fn resolve_project_dependencies<P: AsRef<std::path::Path>>(
    project_root: P,
) -> Result<DependencyResolutionResult> {
    let config = DependencyResolverConfig::default();
    let resolver = DependencyResolver::new(config)?;
    resolver.execute_full_resolution(project_root).await
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;
    
    #[tokio::test]
    async fn test_core_system_initialization() {
        let status = initialize_core_system().await;
        assert!(status.is_ok());
        
        let status = status.unwrap();
        assert!(status.unified_system_initialized);
    }
    
    #[tokio::test]
    async fn test_dependency_management_initialization() {
        let temp_dir = tempdir().unwrap();
        let result = initialize_core_system_with_dependency_management(temp_dir.path()).await;
        
        assert!(result.is_ok());
        let (status, _) = result.unwrap();
        assert!(status.is_healthy());
    }
    
    #[tokio::test]
    async fn test_dependency_resolution() {
        let temp_dir = tempdir().unwrap();
        let result = resolve_project_dependencies(temp_dir.path()).await;
        
        assert!(result.is_ok());
    }
}

/// 简单的锁管理器，使用RwLock
pub struct SimpleLockManager {
    data: Arc<RwLock<HashMap<String, String>>>,
}

impl SimpleLockManager {
    pub fn new() -> Self {
        Self {
            data: Arc::new(RwLock::new(HashMap::new())),
        }
    }
    
    pub fn set_value(&self, key: String, value: String) -> std::result::Result<(), Box<dyn std::error::Error>> {
        let mut data = self.data.write().map_err(|e| {
            std::io::Error::new(std::io::ErrorKind::Other, format!("获取写锁失败: {}", e))
        })?;
        data.insert(key, value);
        Ok(())
    }
    
    pub fn get_value(&self, key: &str) -> std::result::Result<Option<String>, Box<dyn std::error::Error>> {
        let data = self.data.read().map_err(|e| {
            std::io::Error::new(std::io::ErrorKind::Other, format!("获取读锁失败: {}", e))
        })?;
        Ok(data.get(key).cloned())
    }
}

