/// 全局容器管理模块
/// 
/// 提供全局级别的应用程序容器访问和管理功能

use std::sync::{Arc, RwLock};
use once_cell::sync::Lazy;
use log::{info, debug, warn};

use super::ApplicationContainer;

/// 全局应用程序容器
/// 
/// 使用RwLock保护的全局容器实例，支持多线程访问
static GLOBAL_CONTAINER: Lazy<RwLock<Option<Arc<ApplicationContainer>>>> = 
    Lazy::new(|| RwLock::new(None));

/// 获取全局容器
/// 
/// # 返回
/// 全局应用程序容器实例，如果未设置则创建默认实例
/// 
/// # 示例
/// ```rust
/// use crate::core::container::global::get_global_container;
/// 
/// let container = get_global_container();
/// let model_manager = container.model_manager();
/// ```
pub fn get_global_container() -> Arc<ApplicationContainer> {
    debug!("获取全局容器");
    
    // 首先尝试读取现有容器
    if let Ok(guard) = GLOBAL_CONTAINER.read() {
        if let Some(container) = guard.as_ref() {
            debug!("返回现有全局容器");
            return container.clone();
        }
    }
    
    // 如果没有现有容器，创建新的
    debug!("创建新的全局容器");
    let new_container = Arc::new(ApplicationContainer::new());
    
    // 设置为全局容器
    if let Ok(mut guard) = GLOBAL_CONTAINER.write() {
        *guard = Some(new_container.clone());
        info!("成功设置新的全局容器");
    } else {
        warn!("设置全局容器时获取写锁失败，返回临时容器");
    }
    
    new_container
}

/// 设置全局容器
/// 
/// # 参数
/// - container: 要设置的应用程序容器实例
/// 
/// # 示例
/// ```rust
/// use std::sync::Arc;
/// use crate::core::container::{ApplicationContainer, global::set_global_container};
/// 
/// let container = Arc::new(ApplicationContainer::new());
/// set_global_container(container);
/// ```
pub fn set_global_container(container: Arc<ApplicationContainer>) {
    info!("设置全局容器");
    
    if let Ok(mut guard) = GLOBAL_CONTAINER.write() {
        *guard = Some(container);
        info!("成功设置全局容器");
    } else {
        warn!("设置全局容器时获取写锁失败");
    }
}

/// 检查全局容器是否已初始化
/// 
/// # 返回
/// 如果全局容器已设置则返回true，否则返回false
/// 
/// # 示例
/// ```rust
/// use crate::core::container::global::is_global_container_initialized;
/// 
/// if is_global_container_initialized() {
///     println!("全局容器已初始化");
/// } else {
///     println!("全局容器未初始化");
/// }
/// ```
pub fn is_global_container_initialized() -> bool {
    if let Ok(guard) = GLOBAL_CONTAINER.read() {
        guard.is_some()
    } else {
        false
    }
}

/// 清除全局容器
/// 
/// 将全局容器设置为None，释放资源
/// 
/// # 示例
/// ```rust
/// use crate::core::container::global::clear_global_container;
/// 
/// clear_global_container();
/// ```
pub fn clear_global_container() {
    info!("清除全局容器");
    
    if let Ok(mut guard) = GLOBAL_CONTAINER.write() {
        if guard.is_some() {
            *guard = None;
            info!("成功清除全局容器");
        } else {
            debug!("全局容器已经为空");
        }
    } else {
        warn!("清除全局容器时获取写锁失败");
    }
}

/// 重置全局容器
/// 
/// 清除现有容器并创建新的默认容器
/// 
/// # 返回
/// 新创建的全局容器实例
/// 
/// # 示例
/// ```rust
/// use crate::core::container::global::reset_global_container;
/// 
/// let container = reset_global_container();
/// ```
pub fn reset_global_container() -> Arc<ApplicationContainer> {
    info!("重置全局容器");
    
    let new_container = Arc::new(ApplicationContainer::new());
    
    if let Ok(mut guard) = GLOBAL_CONTAINER.write() {
        *guard = Some(new_container.clone());
        info!("成功重置全局容器");
    } else {
        warn!("重置全局容器时获取写锁失败，返回临时容器");
    }
    
    new_container
}

/// 获取全局容器的引用计数
/// 
/// # 返回
/// 全局容器的引用计数，如果未设置则返回0
/// 
/// # 示例
/// ```rust
/// use crate::core::container::global::get_global_container_ref_count;
/// 
/// let ref_count = get_global_container_ref_count();
/// println!("全局容器引用计数: {}", ref_count);
/// ```
pub fn get_global_container_ref_count() -> usize {
    if let Ok(guard) = GLOBAL_CONTAINER.read() {
        if let Some(container) = guard.as_ref() {
            Arc::strong_count(container)
        } else {
            0
        }
    } else {
        0
    }
}

/// 使用全局容器执行操作
/// 
/// # 参数
/// - f: 要执行的操作，接受ApplicationContainer引用作为参数
/// 
/// # 返回
/// 操作的返回值
/// 
/// # 示例
/// ```rust
/// use crate::core::container::global::with_global_container;
/// 
/// let result = with_global_container(|container| {
///     container.model_manager().list_models()
/// }).await;
/// ```
pub fn with_global_container<F, R>(f: F) -> R
where
    F: FnOnce(&ApplicationContainer) -> R,
{
    let container = get_global_container();
    f(&container)
}

/// 异步使用全局容器执行操作
/// 
/// # 参数
/// - f: 要执行的异步操作，接受ApplicationContainer引用作为参数
/// 
/// # 返回
/// 操作的返回值
/// 
/// # 示例
/// ```rust
/// use crate::core::container::global::with_global_container_async;
/// 
/// let result = with_global_container_async(|container| async move {
///     container.model_manager().list_models().await
/// }).await;
/// ```
pub async fn with_global_container_async<F, Fut, R>(f: F) -> R
where
    F: FnOnce(Arc<ApplicationContainer>) -> Fut,
    Fut: std::future::Future<Output = R>,
{
    let container = get_global_container();
    f(container).await
}

/// 尝试获取全局容器（不自动创建）
/// 
/// # 返回
/// 全局容器实例（如果已设置），否则返回None
/// 
/// # 示例
/// ```rust
/// use crate::core::container::global::try_get_global_container;
/// 
/// if let Some(container) = try_get_global_container() {
///     println!("全局容器存在");
/// } else {
///     println!("全局容器不存在");
/// }
/// ```
pub fn try_get_global_container() -> Option<Arc<ApplicationContainer>> {
    if let Ok(guard) = GLOBAL_CONTAINER.read() {
        guard.clone()
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    #[test]
    fn test_global_container_lifecycle() {
        // 清除现有容器
        clear_global_container();
        assert!(!is_global_container_initialized());

        // 获取容器（应该自动创建）
        let container1 = get_global_container();
        assert!(is_global_container_initialized());

        // 获取相同的容器
        let container2 = get_global_container();
        assert!(Arc::ptr_eq(&container1, &container2));

        // 检查引用计数
        let ref_count = get_global_container_ref_count();
        assert!(ref_count >= 2);

        // 重置容器
        let container3 = reset_global_container();
        assert!(!Arc::ptr_eq(&container1, &container3));

        // 清除容器
        clear_global_container();
        assert!(!is_global_container_initialized());
    }

    #[test]
    fn test_try_get_global_container() {
        // 清除现有容器
        clear_global_container();
        
        // 尝试获取不存在的容器
        assert!(try_get_global_container().is_none());
        
        // 设置容器
        let container = Arc::new(ApplicationContainer::new());
        set_global_container(container.clone());
        
        // 尝试获取存在的容器
        let retrieved = try_get_global_container();
        assert!(retrieved.is_some());
        assert!(Arc::ptr_eq(&container, &retrieved.unwrap()));
    }

    #[test]
    fn test_with_global_container() {
        clear_global_container();
        
        let result = with_global_container(|container| {
            // 简单的测试操作
            container.model_manager();
            42
        });
        
        assert_eq!(result, 42);
        assert!(is_global_container_initialized());
    }

    #[tokio::test]
    async fn test_with_global_container_async() {
        clear_global_container();
        
        let result = with_global_container_async(|container| async move {
            // 简单的异步测试操作
            tokio::time::sleep(std::time::Duration::from_millis(1)).await;
            container.model_manager();
            "test_result"
        }).await;
        
        assert_eq!(result, "test_result");
        assert!(is_global_container_initialized());
    }
} 