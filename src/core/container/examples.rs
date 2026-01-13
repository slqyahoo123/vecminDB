/// 服务容器使用示例
/// 
/// 展示如何使用新的trait对象功能

use std::sync::Arc;
use crate::{Result, Error};
use crate::core::container::{DefaultServiceContainer, ServiceContainer, ServiceContainerBuilder};
use crate::core::interfaces::AlgorithmExecutorInterface;

// 定义一个示例trait
pub trait ExampleService: Send + Sync {
    fn get_name(&self) -> String;
    fn process_data(&self, data: &str) -> Result<String>;
}

// 实现示例trait的结构体
pub struct ExampleServiceImpl {
    name: String,
    version: String,
}

impl ExampleServiceImpl {
    pub fn new(name: String, version: String) -> Self {
        Self { name, version }
    }
}

impl ExampleService for ExampleServiceImpl {
    fn get_name(&self) -> String {
        format!("{} v{}", self.name, self.version)
    }

    fn process_data(&self, data: &str) -> Result<String> {
        if data.is_empty() {
            return Err(Error::InvalidInput("数据不能为空".to_string()));
        }
        
        Ok(format!("处理结果: {}", data.to_uppercase()))
    }
}

/// 展示基本服务注册和获取
pub fn example_basic_usage() -> Result<()> {
    println!("=== 基本服务使用示例 ===");
    
    let container = DefaultServiceContainer::new();
    
    // 注册普通服务
    let string_service = Arc::new("Hello World".to_string());
    container.register(string_service.clone())?;
    
    // 注册trait对象服务
    let example_service = Arc::new(ExampleServiceImpl::new("测试服务".to_string(), "1.0".to_string()));
    container.register_trait(example_service.clone())?;
    
    // 获取普通服务
    let retrieved_string = container.get::<String>()?;
    println!("获取的字符串服务: {}", *retrieved_string);
    
    // 获取trait对象服务
    let retrieved_trait = container.get_trait::<dyn ExampleService>()?;
    println!("获取的trait服务名称: {}", retrieved_trait.get_name());
    
    // 使用trait对象的方法
    let result = retrieved_trait.process_data("hello world")?;
    println!("处理结果: {}", result);
    
    // 检查服务存在性
    println!("字符串服务存在: {}", container.contains::<String>());
    println!("trait服务存在: {}", container.contains_trait::<dyn ExampleService>());
    
    // 列出所有服务
    let service_names = container.list_service_names();
    println!("所有服务: {:?}", service_names);
    
    println!("服务总数: {}", container.service_count());
    
    Ok(())
}

/// 展示构建器模式使用
pub fn example_builder_usage() -> Result<()> {
    println!("\n=== 构建器模式使用示例 ===");
    
    let string_service = Arc::new("构建器测试".to_string());
    let example_service = Arc::new(ExampleServiceImpl::new("构建器服务".to_string(), "2.0".to_string()));
    
    let container = ServiceContainerBuilder::new()
        .with_service(string_service)?
        .with_trait_service(example_service)?
        .build();
    
    println!("构建的服务数量: {}", container.service_count());
    println!("容器是否为空: {}", container.is_empty());
    
    // 验证服务是否正确注册
    assert!(container.contains::<String>());
    assert!(container.contains_trait::<dyn ExampleService>());
    
    // 获取并测试服务
    let string_val = container.get::<String>()?;
    println!("构建器注册的字符串: {}", *string_val);
    
    let trait_service = container.get_trait::<dyn ExampleService>()?;
    println!("构建器注册的trait服务: {}", trait_service.get_name());
    
    Ok(())
}

/// 展示服务移除和清理
pub fn example_removal_and_cleanup() -> Result<()> {
    println!("\n=== 服务移除和清理示例 ===");
    
    let container = DefaultServiceContainer::new();
    
    // 注册多个服务
    let string_service = Arc::new("要移除的字符串".to_string());
    let number_service = Arc::new(42i32);
    let example_service = Arc::new(ExampleServiceImpl::new("要移除的服务".to_string(), "1.0".to_string()));
    
    container.register(string_service)?;
    container.register(number_service)?;
    container.register_trait(example_service)?;
    
    println!("初始服务数量: {}", container.service_count());
    
    // 移除特定服务
    container.remove::<String>()?;
    println!("移除字符串服务后数量: {}", container.service_count());
    println!("字符串服务存在: {}", container.contains::<String>());
    
    // 移除trait服务
    container.remove_trait::<dyn ExampleService>()?;
    println!("移除trait服务后数量: {}", container.service_count());
    println!("trait服务存在: {}", container.contains_trait::<dyn ExampleService>());
    
    // 清空所有服务
    container.clear();
    println!("清空后服务数量: {}", container.service_count());
    println!("容器是否为空: {}", container.is_empty());
    
    Ok(())
}

/// 展示错误处理
pub fn example_error_handling() -> Result<()> {
    println!("\n=== 错误处理示例 ===");
    
    let container = DefaultServiceContainer::new();
    
    // 尝试获取不存在的服务
    match container.get::<String>() {
        Ok(_) => println!("意外成功获取了不存在的服务"),
        Err(e) => println!("预期的错误: {}", e),
    }
    
    match container.get_trait::<dyn ExampleService>() {
        Ok(_) => println!("意外成功获取了不存在的trait服务"),
        Err(e) => println!("预期的trait错误: {}", e),
    }
    
    // 尝试重复注册服务
    let service1 = Arc::new("服务1".to_string());
    let service2 = Arc::new("服务2".to_string());
    
    container.register(service1)?;
    match container.register(service2) {
        Ok(_) => println!("意外成功重复注册了服务"),
        Err(e) => println!("预期的重复注册错误: {}", e),
    }
    
    // 尝试重复注册trait服务
    let trait_service1 = Arc::new(ExampleServiceImpl::new("trait服务1".to_string(), "1.0".to_string()));
    let trait_service2 = Arc::new(ExampleServiceImpl::new("trait服务2".to_string(), "2.0".to_string()));
    
    container.register_trait(trait_service1)?;
    match container.register_trait(trait_service2) {
        Ok(_) => println!("意外成功重复注册了trait服务"),
        Err(e) => println!("预期的重复注册trait错误: {}", e),
    }
    
    Ok(())
}

/// 运行所有示例
pub async fn run_all_examples() -> Result<()> {
    println!("=== 运行所有服务容器示例 ===");
    
    example_basic_usage()?;
    example_builder_usage()?;
    example_removal_and_cleanup()?;
    example_error_handling()?;
    example_service_registration()?;
    example_service_migration()?;
    
    println!("=== 所有示例运行完成 ===");
    Ok(())
}

/// 服务注册示例
pub fn example_service_registration() -> Result<()> {
    println!("=== 服务注册示例 ===");
    
    let container = DefaultServiceContainer::new();
    
    // 创建具体的服务实例
    let algorithm_executor = Arc::new(crate::algorithm::executor::AlgorithmExecutor::new(
        crate::algorithm::executor::config::ExecutorConfig::default()
    ));
    
    // 注册为trait对象
    match container.register_trait::<dyn AlgorithmExecutorInterface + Send + Sync>(algorithm_executor) {
        Ok(_) => println!("✓ 成功注册 AlgorithmExecutor 为 trait 对象"),
        Err(e) => println!("✗ 注册 AlgorithmExecutor 失败: {}", e),
    }
    
    // 验证注册
    if container.contains_trait::<dyn AlgorithmExecutorInterface + Send + Sync>() {
        println!("✓ AlgorithmExecutor trait 对象已注册");
    } else {
        println!("✗ AlgorithmExecutor trait 对象未找到");
    }
    
    // 尝试获取trait对象
    match container.get_trait::<dyn AlgorithmExecutorInterface + Send + Sync>() {
        Ok(_) => println!("✓ 成功获取 AlgorithmExecutor trait 对象"),
        Err(e) => println!("✗ 获取 AlgorithmExecutor trait 对象失败: {}", e),
    }
    
    println!("=== 服务注册示例完成 ===");
    Ok(())
}

/// 服务迁移示例
pub fn example_service_migration() -> Result<()> {
    println!("=== 服务迁移示例 ===");
    
    let container = DefaultServiceContainer::new();
    
    // 旧方式：注册具体类型
    let old_algorithm_executor = Arc::new(crate::algorithm::executor::AlgorithmExecutor::new(
        crate::algorithm::executor::config::ExecutorConfig::default()
    ));
    
    println!("1. 旧方式：注册具体类型");
    match container.register::<crate::algorithm::executor::AlgorithmExecutor>(old_algorithm_executor) {
        Ok(_) => println!("✓ 成功注册具体类型"),
        Err(e) => println!("✗ 注册具体类型失败: {}", e),
    }
    
    // 新方式：注册trait对象
    let new_algorithm_executor = Arc::new(crate::algorithm::executor::AlgorithmExecutor::new(
        crate::algorithm::executor::config::ExecutorConfig::default()
    ));
    
    println!("2. 新方式：注册trait对象");
    match container.register_trait::<dyn AlgorithmExecutorInterface + Send + Sync>(new_algorithm_executor) {
        Ok(_) => println!("✓ 成功注册trait对象"),
        Err(e) => println!("✗ 注册trait对象失败: {}", e),
    }
    
    // 验证两种方式都可以获取
    println!("3. 验证获取方式");
    
    // 获取具体类型
    match container.get::<crate::algorithm::executor::AlgorithmExecutor>() {
        Ok(_) => println!("✓ 可以获取具体类型"),
        Err(_) => println!("✗ 无法获取具体类型"),
    }
    
    // 获取trait对象
    match container.get_trait::<dyn AlgorithmExecutorInterface + Send + Sync>() {
        Ok(_) => println!("✓ 可以获取trait对象"),
        Err(_) => println!("✗ 无法获取trait对象"),
    }
    
    // 显示服务列表
    println!("4. 当前注册的服务:");
    let service_names = container.list_service_names();
    for name in service_names {
        println!("  - {}", name);
    }
    
    println!("=== 服务迁移示例完成 ===");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_usage() {
        assert!(example_basic_usage().is_ok());
    }

    #[test]
    fn test_builder_usage() {
        assert!(example_builder_usage().is_ok());
    }

    #[test]
    fn test_removal_and_cleanup() {
        assert!(example_removal_and_cleanup().is_ok());
    }

    #[test]
    fn test_error_handling() {
        assert!(example_error_handling().is_ok());
    }

    #[test]
    fn test_all_examples() {
        assert!(run_all_examples().is_ok());
    }
} 