use std::sync::{Arc, RwLock};
use crate::error::Result;
use crate::algorithm::executor::sandbox::AlgorithmExecutor;
use crate::algorithm::enhanced_executor::{EnhancedAlgorithmExecutor, EnhancedExecutorConfig};
use crate::algorithm::manager::{AlgorithmManager, AlgorithmManagerConfig};
use crate::algorithm::status::AlgorithmStatusTracker;
use crate::algorithm::resource::{ResourceMonitor, ResourceMonitoringConfig, AlgorithmResourceLimits};
use crate::algorithm::security::{SecurityPolicyManager, SecurityPolicy};
use crate::algorithm::security_auto::AutoSecurityAdjuster;
use crate::algorithm::base_types::ExecutorType;
// ModelManagerInterface已移除，使用compat::manager::traits::ModelManager
use crate::compat::manager::traits::ModelManager;
use crate::storage::Storage;
use crate::event::EventSystem;

/// 创建增强算法管理器的工厂函数
pub fn create_enhanced_algorithm_manager(
    storage: Arc<Storage>,
    model_manager: Arc<dyn ModelManager>,
    event_system: Arc<dyn EventSystem>,
    config: Option<AlgorithmManagerConfig>,
    executor_type: Option<ExecutorType>,
) -> AlgorithmManager {
    // 创建基础执行器
    let executor = create_executor(executor_type.unwrap_or(ExecutorType::Local))
        .expect("Failed to create executor");
    
    // 创建增强执行器
    let enhanced_config = EnhancedExecutorConfig::default();
    let enhanced_executor = Arc::new(EnhancedAlgorithmExecutor::new(
        executor,
        Arc::clone(&model_manager),
        Some(enhanced_config),
    ));
    
    // 创建状态跟踪器
    let status_tracker = Arc::new(RwLock::new(AlgorithmStatusTracker::new()));
    
    // 创建安全策略管理器
    let security_policy = SecurityPolicy::default();
    let security_manager = Arc::new(RwLock::new(SecurityPolicyManager::new(
        vec![security_policy]
    )));
    
    // 创建自动安全调节器
    let auto_adjuster = Arc::new(RwLock::new(AutoSecurityAdjuster::new(
        crate::algorithm::security_auto::AutoAdjustConfig::default()
    )));
    
    // 创建算法管理器
    let manager = AlgorithmManager::new(
        storage,
        model_manager,
        enhanced_executor,
        status_tracker,
        event_system,
        config.unwrap_or_default(),
        security_manager,
        auto_adjuster,
    );
    
    manager
}

/// 创建基础执行器的工厂函数
pub fn create_executor(executor_type: ExecutorType) -> Result<Arc<dyn AlgorithmExecutor>> {
    match executor_type {
        ExecutorType::Local => {
            // 创建本地执行器
            let config = crate::algorithm::executor::config::ExecutorConfig::default();
            let executor = crate::algorithm::executor::AlgorithmExecutor::new(config);
            Ok(Arc::new(executor))
        },
        ExecutorType::Docker => {
            // 创建Docker执行器
            let mut config = crate::algorithm::executor::config::ExecutorConfig::default();
            config.security_level = crate::algorithm::types::SandboxSecurityLevel::High;
            let executor = crate::algorithm::executor::AlgorithmExecutor::new(config);
            Ok(Arc::new(executor))
        },
        ExecutorType::Wasm => {
            // 创建WebAssembly执行器
            let mut config = crate::algorithm::executor::config::ExecutorConfig::default();
            config.security_level = crate::algorithm::types::SandboxSecurityLevel::Maximum;
            let executor = crate::algorithm::executor::AlgorithmExecutor::new(config);
            Ok(Arc::new(executor))
        },
        ExecutorType::V8 => {
            // 创建V8执行器
            let mut config = crate::algorithm::executor::config::ExecutorConfig::default();
            config.security_level = crate::algorithm::types::SandboxSecurityLevel::Maximum;
            let executor = crate::algorithm::executor::AlgorithmExecutor::new(config);
            Ok(Arc::new(executor))
        },
        ExecutorType::Remote => {
            // 创建远程执行器
            let config = crate::algorithm::executor::config::ExecutorConfig::default();
            let executor = crate::algorithm::executor::AlgorithmExecutor::new(config);
            Ok(Arc::new(executor))
        },
    }
}

/// 创建增强执行器的工厂函数
pub fn create_enhanced_executor(
    executor_type: ExecutorType,
    model_manager: Arc<dyn ModelManager>,
    config: Option<EnhancedExecutorConfig>,
) -> Result<EnhancedAlgorithmExecutor> {
    let base_executor = create_executor(executor_type)?;
    Ok(EnhancedAlgorithmExecutor::new(
        base_executor,
        model_manager,
        config,
    ))
}

/// 创建资源监控器的工厂函数
pub fn create_resource_monitor(
    config: Option<ResourceMonitoringConfig>,
    limits: Option<AlgorithmResourceLimits>,
    enable_gpu: bool,
    gpu_index: Option<usize>,
) -> ResourceMonitor {
    let config = config.unwrap_or_default();
    let limits = limits.unwrap_or_default();
    
    if enable_gpu {
        ResourceMonitor::with_gpu_monitoring(
            config,
            limits,
            gpu_index.unwrap_or(0)
        )
    } else {
        ResourceMonitor::new(config, limits)
    }
}

/// 创建状态跟踪器的工厂函数
pub fn create_status_tracker() -> AlgorithmStatusTracker {
    AlgorithmStatusTracker::new()
}

/// 创建安全策略管理器的工厂函数
pub fn create_security_manager(policies: Option<Vec<SecurityPolicy>>) -> SecurityPolicyManager {
    let policies = policies.unwrap_or_else(|| vec![SecurityPolicy::default()]);
    SecurityPolicyManager::new(policies)
}

/// 创建自动安全调节器的工厂函数
pub fn create_auto_security_adjuster(
    config: Option<crate::algorithm::security_auto::AutoAdjustConfig>
) -> AutoSecurityAdjuster {
    let config = config.unwrap_or_default();
    AutoSecurityAdjuster::new(config)
}

/// 创建完整的算法执行环境
pub fn create_algorithm_execution_environment(
    storage: Arc<Storage>,
    model_manager: Arc<dyn ModelManager>,
    event_system: Arc<dyn EventSystem>,
    executor_type: Option<ExecutorType>,
    enhanced_config: Option<EnhancedExecutorConfig>,
    manager_config: Option<AlgorithmManagerConfig>,
) -> Result<(AlgorithmManager, Arc<EnhancedAlgorithmExecutor>)> {
    // 创建增强执行器
    let enhanced_executor = Arc::new(create_enhanced_executor(
        executor_type.unwrap_or(ExecutorType::Local),
        Arc::clone(&model_manager),
        enhanced_config,
    )?);
    
    // 创建状态跟踪器
    let status_tracker = Arc::new(RwLock::new(create_status_tracker()));
    
    // 创建安全管理器
    let security_manager = Arc::new(RwLock::new(create_security_manager(None)));
    
    // 创建自动安全调节器
    let auto_adjuster = Arc::new(RwLock::new(create_auto_security_adjuster(None)));
    
    // 创建算法管理器
    let manager = AlgorithmManager::new(
        storage,
        model_manager,
        Arc::clone(&enhanced_executor),
        status_tracker,
        event_system,
        manager_config.unwrap_or_default(),
        security_manager,
        auto_adjuster,
    );
    
    Ok((manager, enhanced_executor))
} 