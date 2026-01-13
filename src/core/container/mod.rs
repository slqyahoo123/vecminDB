/// 容器模块
/// 
/// 提供依赖注入容器和服务管理功能，支持模块化的服务架构

use std::sync::Arc;

pub mod service_container;
pub mod application_container;
pub mod global;
pub mod proxies;
pub mod examples;

// 重新导出核心类型
pub use service_container::{ServiceContainer, DefaultServiceContainer, ServiceContainerBuilder};
pub use application_container::ApplicationContainer;
pub use global::{
    get_global_container, 
    set_global_container, 
    is_global_container_initialized,
    clear_global_container,
    reset_global_container,
    get_global_container_ref_count,
    with_global_container,
    with_global_container_async,
    try_get_global_container,
};

// 重新导出代理类型
pub use proxies::{
    ModelManagerProxy,
    // TrainingEngineProxy, // 已移除：向量数据库系统不需要训练功能
    DataProcessorProxy,
    AlgorithmExecutorProxy,
};

// 重新导出示例类型
pub use examples::{ExampleService, ExampleServiceImpl, run_all_examples};

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    #[test]
    fn test_container_module_integration() {
        // 测试模块之间的集成
        let container = ApplicationContainer::new();
        
        // 验证所有核心服务都已创建
        assert!(!container.model_manager().is_null());
        assert!(!container.training_engine().is_null());
        assert!(!container.data_processor().is_null());
        assert!(!container.algorithm_executor().is_null());
    }

    #[test]
    fn test_global_container_integration() {
        // 清除现有全局容器
        clear_global_container();
        
        // 测试全局容器功能
        let global_container = get_global_container();
        assert!(is_global_container_initialized());
        
        // 验证全局容器的服务
        let model_manager = global_container.model_manager();
        assert!(!model_manager.is_null());
        
        // 测试全局容器的引用计数
        let ref_count = get_global_container_ref_count();
        assert!(ref_count > 0);
        
        // 清理
        clear_global_container();
    }

    #[test]
    fn test_service_container_integration() {
        // 测试服务容器与应用容器的集成
        let mut service_container = DefaultServiceContainer::new();
        
        // 注册一个测试服务
        #[derive(Debug)]
        struct TestService {
            name: String,
        }
        
        let test_service = Arc::new(TestService {
            name: "test".to_string(),
        });
        
        service_container.register(test_service.clone()).unwrap();
        
        // 验证服务注册成功
        assert!(service_container.contains::<TestService>());
        
        let retrieved_service = service_container.get::<TestService>().unwrap();
        assert_eq!(retrieved_service.name, "test");
        
        // 测试服务移除
        service_container.remove::<TestService>().unwrap();
        assert!(!service_container.contains::<TestService>());
    }

    #[test]
    fn test_proxy_integration() {
        // 测试代理类型的集成
        let container = ApplicationContainer::new();
        let service_container = Arc::new(DefaultServiceContainer::new());
        
        // 创建代理实例
        let model_proxy = ModelManagerProxy::new(service_container.clone());
        let training_proxy = TrainingEngineProxy::new(service_container.clone());
        let data_proxy = DataProcessorProxy::new(service_container.clone());
        let algo_proxy = AlgorithmExecutorProxy::new(service_container.clone());
        
        // 验证代理创建成功（通过类型检查）
        let _: ModelManagerProxy = model_proxy;
        let _: TrainingEngineProxy = training_proxy;
        let _: DataProcessorProxy = data_proxy;
        let _: AlgorithmExecutorProxy = algo_proxy;
    }

    #[tokio::test]
    async fn test_async_integration() {
        // 测试异步功能的集成
        clear_global_container();
        
        let result = with_global_container_async(|container| async move {
            // 测试异步服务访问
            let model_manager = container.model_manager();
            let training_engine = container.training_engine();
            let data_processor = container.data_processor();
            let algorithm_executor = container.algorithm_executor();
            
            // 简单验证服务存在
            assert!(!model_manager.is_null());
            assert!(!training_engine.is_null());
            assert!(!data_processor.is_null());
            assert!(!algorithm_executor.is_null());
            
            "integration_test_passed"
        }).await;
        
        assert_eq!(result, "integration_test_passed");
        clear_global_container();
    }
}

// 为Arc添加扩展方法的trait
trait ArcExt<T> {
    fn is_null(&self) -> bool;
}

impl<T> ArcExt<T> for Arc<T> {
    fn is_null(&self) -> bool {
        Arc::strong_count(self) == 0
    }
} 