/// 服务容器模块
/// 
/// 提供依赖注入容器的核心接口和实现

use std::sync::{Arc, RwLock};
use std::collections::HashMap;
use std::any::{Any, TypeId};
use log::{info, debug};

use crate::{Result, Error};

/// 服务容器特质
/// 
/// 定义了依赖注入容器的基本接口，支持服务的注册、获取、移除等操作
pub trait ServiceContainer: Send + Sync {
    /// 注册服务
    /// 
    /// # 参数
    /// - service: 要注册的服务实例
    /// 
    /// # 返回
    /// - Ok(()): 注册成功
    /// - Err: 注册失败的错误信息
    fn register<T: Any + Send + Sync>(&self, service: Arc<T>) -> Result<()>;
    
    /// 注册trait对象服务
    /// 
    /// # 参数
    /// - trait_id: trait的类型ID
    /// - service: trait对象实例
    /// 
    /// # 返回
    /// - Ok(()): 注册成功
    /// - Err: 注册失败的错误信息
    fn register_trait<T: ?Sized + Send + Sync + 'static>(&self, service: Arc<T>) -> Result<()>;
    
    /// 获取服务
    /// 
    /// # 返回
    /// - Ok(Arc<T>): 服务实例
    /// - Err: 服务不存在或类型转换失败
    fn get<T: Any + Send + Sync>(&self) -> Result<Arc<T>>;
    
    /// 获取trait对象服务
    /// 
    /// # 返回
    /// - Ok(Arc<T>): trait对象实例
    /// - Err: 服务不存在或类型转换失败
    fn get_trait<T: ?Sized + Send + Sync + 'static>(&self) -> Result<Arc<T>>;
    
    /// 通过trait对象获取服务
    /// 
    /// # 参数
    /// - trait_name: trait的名称
    /// 
    /// # 返回
    /// - Ok(Arc<dyn Any + Send + Sync>): 服务实例
    /// - Err: 服务不存在或类型转换失败
    fn get_by_trait(&self, trait_name: &str) -> Result<Arc<dyn Any + Send + Sync>>;
    
    /// 检查服务是否存在
    /// 
    /// # 返回
    /// - true: 服务存在
    /// - false: 服务不存在
    fn contains<T: Any + Send + Sync>(&self) -> bool;
    
    /// 检查trait对象服务是否存在
    /// 
    /// # 返回
    /// - true: 服务存在
    /// - false: 服务不存在
    fn contains_trait<T: ?Sized + Send + Sync + 'static>(&self) -> bool;
    
    /// 移除服务
    /// 
    /// # 返回
    /// - Ok(()): 移除成功
    /// - Err: 服务不存在
    fn remove<T: Any + Send + Sync>(&self) -> Result<()>;
    
    /// 移除trait对象服务
    /// 
    /// # 返回
    /// - Ok(()): 移除成功
    /// - Err: 服务不存在
    fn remove_trait<T: ?Sized + Send + Sync + 'static>(&self) -> Result<()>;
    
    /// 清空所有服务
    fn clear(&self);
}

/// 默认服务容器实现
/// 
/// 基于HashMap的线程安全服务容器实现，支持运行时类型检查和服务管理
pub struct DefaultServiceContainer {
    /// 服务存储：TypeId -> 服务实例
    services: RwLock<HashMap<TypeId, Arc<dyn Any + Send + Sync>>>,
    /// trait对象服务存储：TypeId -> Box<dyn Any>（内部装载 Arc<dyn Trait>）
    /// 说明：无法直接将 Arc<dyn Trait> 转为 Arc<dyn Any>，因此以 Box<dyn Any>
    /// 包装存储，再在读取时 downcast 到具体的 Arc<dyn Trait>。
    trait_services: RwLock<HashMap<TypeId, Box<dyn Any + Send + Sync>>>,
    /// 服务名称映射：TypeId -> 类型名称（用于调试）
    service_names: RwLock<HashMap<TypeId, String>>,
    /// trait服务名称映射：TypeId -> trait名称（用于调试）
    trait_service_names: RwLock<HashMap<TypeId, String>>,
}

impl DefaultServiceContainer {
    /// 创建新的服务容器实例
    pub fn new() -> Self {
        Self {
            services: RwLock::new(HashMap::new()),
            trait_services: RwLock::new(HashMap::new()),
            service_names: RwLock::new(HashMap::new()),
            trait_service_names: RwLock::new(HashMap::new()),
        }
    }

    /// 获取服务类型的名称
    /// 
    /// # 参数
    /// - T: 服务类型
    /// 
    /// # 返回
    /// 类型的字符串表示
    fn get_type_name<T: Any>() -> String {
        std::any::type_name::<T>().to_string()
    }

    /// 获取trait类型的名称
    /// 
    /// # 参数
    /// - T: trait类型
    /// 
    /// # 返回
    /// trait的字符串表示
    fn get_trait_name<T: ?Sized + 'static>() -> String {
        std::any::type_name::<T>().to_string()
    }

    /// 获取当前注册的服务数量
    pub fn service_count(&self) -> usize {
        let services = self.services.read().unwrap();
        let trait_services = self.trait_services.read().unwrap();
        services.len() + trait_services.len()
    }

    /// 列出所有已注册的服务类型名称
    pub fn list_service_names(&self) -> Vec<String> {
        let names = self.service_names.read().unwrap();
        let trait_names = self.trait_service_names.read().unwrap();
        let mut all_names = names.values().cloned().collect::<Vec<_>>();
        all_names.extend(trait_names.values().cloned());
        all_names
    }

    /// 检查容器是否为空
    pub fn is_empty(&self) -> bool {
        let services = self.services.read().unwrap();
        let trait_services = self.trait_services.read().unwrap();
        services.is_empty() && trait_services.is_empty()
    }

    /// 获取服务（get 方法的别名，用于向后兼容）
    /// 
    /// # 类型参数
    /// - T: 要获取的服务类型
    /// 
    /// # 返回
    /// 服务实例（如果存在）
    pub fn get_service<T: Any + Send + Sync>(&self) -> Result<Arc<T>> {
        <Self as ServiceContainer>::get::<T>(self)
    }

    /// 直接包装 trait 方法，便于通过 &DefaultServiceContainer 或 Arc<...> 调用
    pub fn get_trait<T: ?Sized + Send + Sync + 'static>(&self) -> Result<Arc<T>> {
        <Self as ServiceContainer>::get_trait::<T>(self)
    }

    /// 包装注册普通服务
    pub fn register<T: Any + Send + Sync>(&self, service: Arc<T>) -> Result<()> {
        <Self as ServiceContainer>::register::<T>(self, service)
    }

    /// 包装注册 trait 对象服务
    pub fn register_trait<T: ?Sized + Send + Sync + 'static>(&self, service: Arc<T>) -> Result<()> {
        <Self as ServiceContainer>::register_trait::<T>(self, service)
    }
}

impl Default for DefaultServiceContainer {
    fn default() -> Self {
        Self::new()
    }
}

impl ServiceContainer for DefaultServiceContainer {
    fn register<T: Any + Send + Sync>(&self, service: Arc<T>) -> Result<()> {
        let type_id = TypeId::of::<T>();
        let type_name = Self::get_type_name::<T>();
        
        // 首先检查服务是否已经存在
        {
            let services = self.services.read().unwrap();
            if services.contains_key(&type_id) {
                return Err(Error::InvalidInput(
                    format!("服务已存在: {}", type_name)
                ));
            }
        }
        
        // 然后插入新服务
        {
            let mut services = self.services.write().unwrap();
            let mut names = self.service_names.write().unwrap();
            
            services.insert(type_id, service);
            names.insert(type_id, type_name.clone());
        }
        
        debug!("已注册服务: {}", type_name);
        Ok(())
    }

    fn register_trait<T: ?Sized + Send + Sync + 'static>(&self, service: Arc<T>) -> Result<()> {
        let type_id = TypeId::of::<T>();
        let trait_name = Self::get_trait_name::<T>();

        // 检查是否已存在
        {
            let trait_services = self.trait_services.read().unwrap();
            if trait_services.contains_key(&type_id) {
                return Err(Error::InvalidInput(
                    format!("Trait服务已存在: {}", trait_name)
                ));
            }
        }

        // 以 Box<dyn Any> 包装 Arc<dyn Trait>
        // 注意：Arc<dyn Trait> 实现了 Any，所以可以直接包装
        {
            let mut trait_services = self.trait_services.write().unwrap();
            let mut trait_names = self.trait_service_names.write().unwrap();

            // Arc<dyn Trait> 实现了 Any，所以可以直接转换为 Box<dyn Any>
            let boxed: Box<dyn Any + Send + Sync> = Box::new(service);
            trait_services.insert(type_id, boxed);
            trait_names.insert(type_id, trait_name.clone());
        }

        debug!("已注册Trait服务: {}", trait_name);
        Ok(())
    }

    fn get<T: Any + Send + Sync + Sized>(&self) -> Result<Arc<T>> {
        let type_id = TypeId::of::<T>();
        let services = self.services.read().unwrap();
        
        match services.get(&type_id) {
            Some(service) => {
                service.clone()
                    .downcast::<T>()
                    .map_err(|_| Error::InvalidInput("服务类型转换失败".to_string()))
            },
            None => {
                let type_name = Self::get_type_name::<T>();
                Err(Error::InvalidInput(format!("服务未找到: {}", type_name)))
            }
        }
    }

    fn get_trait<T: ?Sized + Send + Sync + 'static>(&self) -> Result<Arc<T>> {
        let type_id = TypeId::of::<T>();
        let trait_services = self.trait_services.read().unwrap();

        match trait_services.get(&type_id) {
            Some(holder) => {
                // 从 Box<dyn Any> 中取出 Arc<dyn T>
                // Box<dyn Any> 中存储的是 Arc<dyn Trait>
                // 我们需要将其转换为 Arc<dyn Trait>
                // 由于 TypeId 匹配，我们可以安全地假设类型正确
                // 使用 unsafe 进行类型转换，但这是安全的，因为：
                // 1. TypeId 匹配确保类型正确
                // 2. Arc 是引用计数的，不会导致内存问题
                // 3. 我们只是进行类型转换，不改变内存布局
                unsafe {
                    // 从 Box<dyn Any> 中提取 Arc<dyn T>
                    // 由于 TypeId 匹配，我们可以安全地假设类型正确
                    // 使用 unsafe 进行类型转换，但这是安全的，因为：
                    // 1. TypeId 匹配确保类型正确
                    // 2. Arc 是引用计数的，不会导致内存问题
                    // 3. 我们只是进行类型转换，不改变内存布局
                    // 
                    // 注意：holder 是 &Box<dyn Any>，我们需要将其转换为 Arc<T>
                    // Box<dyn Any> 中存储的是 Arc<dyn Trait>（通过 Box::new(service) 创建）
                    // 所以 Box<dyn Any> 的内容实际上是 Arc<dyn Trait>
                    // 我们需要获取 Box 的内容（Arc<dyn Trait>），然后转换为 Arc<T>
                    // 
                    // 方法：Box<dyn Any> 中存储的是 Arc<dyn Trait>
                    // Box 的指针指向堆上的 Arc<dyn Trait>
                    // 我们可以获取 Box 的内容（Arc<dyn Trait>），然后转换为 Arc<T>
                    // 
                    // 由于 Box<Arc<dyn Trait>> 和 Box<dyn Any> 的内存布局相同（都是指向堆上的 Arc）
                    // 我们可以安全地将 Box<dyn Any> 转换为 Box<Arc<dyn Trait>>，然后解引用获取 Arc
                    let box_ptr: *const Box<dyn Any + Send + Sync> = holder;
                    // 将 Box<dyn Any> 指针转换为 Box<Arc<T>> 的指针
                    // 注意：这是 unsafe 的，因为我们假设类型匹配
                    let arc_box_ptr = box_ptr as *const Box<Arc<T>>;
                    // 解引用获取 Arc（通过双重解引用：*arc_box_ptr 得到 Box<Arc<T>>，**arc_box_ptr 得到 Arc<T>）
                    let arc_ref = unsafe { &**arc_box_ptr };
                    // 克隆 Arc
                    Ok(arc_ref.clone())
                }
            }
            None => {
                let trait_name = Self::get_trait_name::<T>();
                Err(Error::InvalidInput(format!("Trait服务未找到: {}", trait_name)))
            }
        }
    }

    fn get_by_trait(&self, trait_name: &str) -> Result<Arc<dyn Any + Send + Sync>> {
        let services = self.services.read().unwrap();
        let trait_services = self.trait_services.read().unwrap();
        let names = self.service_names.read().unwrap();
        let trait_names = self.trait_service_names.read().unwrap();
        
        // 首先在普通服务中查找
        for (type_id, service) in services.iter() {
            if let Some(name) = names.get(type_id) {
                if name.contains(trait_name) {
                    return Ok(service.clone());
                }
            }
        }
        
        // 然后在trait服务中查找
        for (type_id, _service) in trait_services.iter() {
            if let Some(name) = trait_names.get(type_id) {
                if name.contains(trait_name) {
                    // 返回一个空的占位 Any；名称匹配接口通常用于存在性检查，
                    // 这里返回错误以促使调用方改用 get_trait<T>() 检索具体类型
                    return Err(Error::InvalidInput("请使用 get_trait::<Trait>() 获取trait对象".to_string()));
                }
            }
        }
        
        Err(Error::InvalidInput(format!("未找到匹配trait的服务: {}", trait_name)))
    }

    fn contains<T: Any + Send + Sync + Sized>(&self) -> bool {
        let type_id = TypeId::of::<T>();
        let services = self.services.read().unwrap();
        services.contains_key(&type_id)
    }

    fn contains_trait<T: ?Sized + Send + Sync + 'static>(&self) -> bool {
        let type_id = TypeId::of::<T>();
        let trait_services = self.trait_services.read().unwrap();
        trait_services.contains_key(&type_id)
    }

    fn remove<T: Any + Send + Sync + Sized>(&self) -> Result<()> {
        let type_id = TypeId::of::<T>();
        let type_name = Self::get_type_name::<T>();
        
        let mut services = self.services.write().unwrap();
        let mut names = self.service_names.write().unwrap();
        
        if services.remove(&type_id).is_some() {
            names.remove(&type_id);
            debug!("已移除服务: {}", type_name);
            Ok(())
        } else {
            Err(Error::InvalidInput(format!("服务未找到: {}", type_name)))
        }
    }

    fn remove_trait<T: ?Sized + Send + Sync + 'static>(&self) -> Result<()> {
        let type_id = TypeId::of::<T>();
        let trait_name = Self::get_trait_name::<T>();

        let mut trait_services = self.trait_services.write().unwrap();
        let mut trait_names = self.trait_service_names.write().unwrap();

        if trait_services.remove(&type_id).is_some() {
            trait_names.remove(&type_id);
            debug!("已移除Trait服务: {}", trait_name);
            Ok(())
        } else {
            Err(Error::InvalidInput(format!("Trait服务未找到: {}", trait_name)))
        }
    }

    fn clear(&self) {
        let mut services = self.services.write().unwrap();
        let mut trait_services = self.trait_services.write().unwrap();
        let mut names = self.service_names.write().unwrap();
        let mut trait_names = self.trait_service_names.write().unwrap();
        
        let service_count = services.len();
        let trait_service_count = trait_services.len();
        
        services.clear();
        trait_services.clear();
        names.clear();
        trait_names.clear();
        
        info!("已清空所有服务，共清理了 {} 个普通服务和 {} 个trait服务", service_count, trait_service_count);
    }
}

/// 服务容器构建器
/// 
/// 提供流式API来构建和配置服务容器
pub struct ServiceContainerBuilder {
    container: DefaultServiceContainer,
}

impl ServiceContainerBuilder {
    /// 创建新的构建器
    pub fn new() -> Self {
        Self {
            container: DefaultServiceContainer::new(),
        }
    }

    /// 添加服务
    pub fn with_service<T: Any + Send + Sync + Sized>(mut self, service: Arc<T>) -> Result<Self> {
        self.container.register(service)?;
        Ok(self)
    }

    /// 添加trait对象服务
    pub fn with_trait_service<T: ?Sized + Send + Sync + 'static>(mut self, service: Arc<T>) -> Result<Self> {
        self.container.register_trait(service)?;
        Ok(self)
    }

    /// 构建容器
    pub fn build(self) -> DefaultServiceContainer {
        self.container
    }
}

impl Default for ServiceContainerBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    // 定义一个测试trait
    trait TestTrait: Send + Sync {
        fn get_value(&self) -> String;
    }

    // 实现测试trait的结构体
    struct TestTraitImpl {
        value: String,
    }

    impl TestTraitImpl {
        fn new(value: String) -> Self {
            Self { value }
        }
    }

    impl TestTrait for TestTraitImpl {
        fn get_value(&self) -> String {
            self.value.clone()
        }
    }

    #[test]
    fn test_service_container_basic_operations() {
        let container = DefaultServiceContainer::new();
        
        // 测试服务注册
        let test_string = Arc::new("test".to_string());
        assert!(container.register(test_string.clone()).is_ok());
        
        // 测试服务存在检查
        assert!(container.contains::<String>());
        assert!(!container.contains::<i32>());
        
        // 测试服务获取
        let retrieved = container.get::<String>().unwrap();
        assert_eq!(*retrieved, "test");
        
        // 测试服务数量
        assert_eq!(container.service_count(), 1);
        assert!(!container.is_empty());
        
        // 测试服务移除
        assert!(container.remove::<String>().is_ok());
        assert!(!container.contains::<String>());
        assert_eq!(container.service_count(), 0);
        assert!(container.is_empty());
    }

    #[test]
    fn test_service_container_trait_operations() {
        let container = DefaultServiceContainer::new();
        
        // 测试trait对象注册
        let test_trait_impl = Arc::new(TestTraitImpl::new("trait_test".to_string()));
        assert!(container.register_trait(test_trait_impl.clone()).is_ok());
        
        // 测试trait对象存在检查
        assert!(container.contains_trait::<dyn TestTrait>());
        assert!(!container.contains_trait::<dyn Send>());
        
        // 测试trait对象获取
        let retrieved = container.get_trait::<dyn TestTrait>().unwrap();
        assert_eq!(retrieved.get_value(), "trait_test");
        
        // 测试服务数量（包括trait对象）
        assert_eq!(container.service_count(), 1);
        assert!(!container.is_empty());
        
        // 测试trait对象移除
        assert!(container.remove_trait::<dyn TestTrait>().is_ok());
        assert!(!container.contains_trait::<dyn TestTrait>());
        assert_eq!(container.service_count(), 0);
        assert!(container.is_empty());
    }

    #[test]
    fn test_service_container_duplicate_registration() {
        let container = DefaultServiceContainer::new();
        
        let test_string1 = Arc::new("test1".to_string());
        let test_string2 = Arc::new("test2".to_string());
        
        // 第一次注册应该成功
        assert!(container.register(test_string1).is_ok());
        
        // 重复注册应该失败
        assert!(container.register(test_string2).is_err());
    }

    #[test]
    fn test_service_container_duplicate_trait_registration() {
        let container = DefaultServiceContainer::new();
        
        let test_trait_impl1 = Arc::new(TestTraitImpl::new("test1".to_string()));
        let test_trait_impl2 = Arc::new(TestTraitImpl::new("test2".to_string()));
        
        // 第一次注册应该成功
        assert!(container.register_trait(test_trait_impl1).is_ok());
        
        // 重复注册应该失败
        assert!(container.register_trait(test_trait_impl2).is_err());
    }

    #[test]
    fn test_service_container_builder() {
        let test_string = Arc::new("test".to_string());
        let test_number = Arc::new(42i32);
        
        let container = ServiceContainerBuilder::new()
            .with_service(test_string)
            .unwrap()
            .with_service(test_number)
            .unwrap()
            .build();
        
        assert_eq!(container.service_count(), 2);
        assert!(container.contains::<String>());
        assert!(container.contains::<i32>());
    }

    #[test]
    fn test_service_container_builder_with_traits() {
        let test_string = Arc::new("test".to_string());
        let test_trait_impl = Arc::new(TestTraitImpl::new("trait_test".to_string()));
        
        let container = ServiceContainerBuilder::new()
            .with_service(test_string)
            .unwrap()
            .with_trait_service(test_trait_impl)
            .unwrap()
            .build();
        
        assert_eq!(container.service_count(), 2);
        assert!(container.contains::<String>());
        assert!(container.contains_trait::<dyn TestTrait>());
    }

    #[test]
    fn test_service_container_list_services() {
        let container = DefaultServiceContainer::new();
        
        let test_string = Arc::new("test".to_string());
        let test_number = Arc::new(42i32);
        let test_trait_impl = Arc::new(TestTraitImpl::new("trait_test".to_string()));
        
        container.register(test_string).unwrap();
        container.register(test_number).unwrap();
        container.register_trait(test_trait_impl).unwrap();
        
        let service_names = container.list_service_names();
        assert_eq!(service_names.len(), 3);
        assert!(service_names.contains(&"alloc::string::String".to_string()));
        assert!(service_names.contains(&"i32".to_string()));
        assert!(service_names.contains(&"core::container::service_container::tests::TestTrait".to_string()));
    }

    #[test]
    fn test_service_container_clear() {
        let container = DefaultServiceContainer::new();
        
        let test_string = Arc::new("test".to_string());
        let test_number = Arc::new(42i32);
        let test_trait_impl = Arc::new(TestTraitImpl::new("trait_test".to_string()));
        
        container.register(test_string).unwrap();
        container.register(test_number).unwrap();
        container.register_trait(test_trait_impl).unwrap();
        assert_eq!(container.service_count(), 3);
        
        container.clear();
        assert_eq!(container.service_count(), 0);
        assert!(container.is_empty());
        assert!(!container.contains::<String>());
        assert!(!container.contains::<i32>());
        assert!(!container.contains_trait::<dyn TestTrait>());
    }

    #[test]
    fn test_service_container_mixed_operations() {
        let container = DefaultServiceContainer::new();
        
        // 注册普通服务和trait对象
        let test_string = Arc::new("test".to_string());
        let test_trait_impl = Arc::new(TestTraitImpl::new("trait_test".to_string()));
        
        container.register(test_string.clone()).unwrap();
        container.register_trait(test_trait_impl.clone()).unwrap();
        
        // 验证两种类型的服务都能正常工作
        assert_eq!(container.service_count(), 2);
        assert!(container.contains::<String>());
        assert!(container.contains_trait::<dyn TestTrait>());
        
        let retrieved_string = container.get::<String>().unwrap();
        let retrieved_trait = container.get_trait::<dyn TestTrait>().unwrap();
        
        assert_eq!(*retrieved_string, "test");
        assert_eq!(retrieved_trait.get_value(), "trait_test");
        
        // 测试通过trait名称获取服务
        let trait_service = container.get_by_trait("TestTrait").unwrap();
        assert!(trait_service.is::<TestTraitImpl>());
    }
} 