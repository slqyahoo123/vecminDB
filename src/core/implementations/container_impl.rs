/// 依赖注入容器的完整生产级实现
/// 提供服务注册、解析、生命周期管理等功能

use std::collections::HashMap;
use std::sync::{Arc, RwLock, Mutex};
use std::any::{Any, TypeId};
use async_trait::async_trait;
use chrono::{DateTime, Utc};

use crate::{Result, Error};

/// 生产级依赖注入容器实现
pub struct ProductionContainer {
    services: Arc<RwLock<HashMap<String, ServiceRegistration>>>,
    instances: Arc<RwLock<HashMap<String, Arc<dyn Any + Send + Sync>>>>,
    type_mappings: Arc<RwLock<HashMap<TypeId, String>>>,
    factories: Arc<RwLock<HashMap<String, Box<dyn ServiceFactory>>>>,
    lifecycle_callbacks: Arc<RwLock<HashMap<String, Vec<Box<dyn LifecycleCallback>>>>>,
    is_disposed: Arc<RwLock<bool>>,
}

impl ProductionContainer {
    pub fn new() -> Self {
        Self {
            services: Arc::new(RwLock::new(HashMap::new())),
            instances: Arc::new(RwLock::new(HashMap::new())),
            type_mappings: Arc::new(RwLock::new(HashMap::new())),
            factories: Arc::new(RwLock::new(HashMap::new())),
            lifecycle_callbacks: Arc::new(RwLock::new(HashMap::new())),
            is_disposed: Arc::new(RwLock::new(false)),
        }
    }

    /// 注册单例服务
    pub fn register_singleton<T>(&self, name: &str, instance: T) -> Result<()> 
    where
        T: Send + Sync + 'static,
    {
        self.ensure_not_disposed()?;
        
        let registration = ServiceRegistration {
            name: name.to_string(),
            service_type: std::any::type_name::<T>().to_string(),
            lifetime: ServiceLifetime::Singleton,
            registered_at: Utc::now(),
            dependencies: Vec::new(),
        };

        // 注册服务定义
        {
            let mut services = self.services.write().unwrap();
            services.insert(name.to_string(), registration);
        }

        // 存储实例
        {
            let mut instances = self.instances.write().unwrap();
            instances.insert(name.to_string(), Arc::new(instance));
        }

        // 注册类型映射
        {
            let mut mappings = self.type_mappings.write().unwrap();
            mappings.insert(TypeId::of::<T>(), name.to_string());
        }

        // 调用注册回调
        self.invoke_lifecycle_callbacks(name, LifecycleEvent::Registered)?;

        Ok(())
    }

    /// 注册临时服务
    pub fn register_transient<T, F>(&self, name: &str, factory: F) -> Result<()>
    where
        T: Send + Sync + 'static,
        F: Fn(&ProductionContainer) -> Result<T> + Send + Sync + 'static,
    {
        self.ensure_not_disposed()?;

        let registration = ServiceRegistration {
            name: name.to_string(),
            service_type: std::any::type_name::<T>().to_string(),
            lifetime: ServiceLifetime::Transient,
            registered_at: Utc::now(),
            dependencies: Vec::new(),
        };

        // 注册服务定义
        {
            let mut services = self.services.write().unwrap();
            services.insert(name.to_string(), registration);
        }

        // 注册工厂
        {
            let mut factories = self.factories.write().unwrap();
            factories.insert(
                name.to_string(),
                Box::new(TransientServiceFactory::new(factory)),
            );
        }

        // 注册类型映射
        {
            let mut mappings = self.type_mappings.write().unwrap();
            mappings.insert(TypeId::of::<T>(), name.to_string());
        }

        // 调用注册回调
        self.invoke_lifecycle_callbacks(name, LifecycleEvent::Registered)?;

        Ok(())
    }

    /// 注册作用域服务
    pub fn register_scoped<T, F>(&self, name: &str, factory: F) -> Result<()>
    where
        T: Send + Sync + 'static,
        F: Fn(&ProductionContainer) -> Result<T> + Send + Sync + 'static,
    {
        self.ensure_not_disposed()?;

        let registration = ServiceRegistration {
            name: name.to_string(),
            service_type: std::any::type_name::<T>().to_string(),
            lifetime: ServiceLifetime::Scoped,
            registered_at: Utc::now(),
            dependencies: Vec::new(),
        };

        // 注册服务定义
        {
            let mut services = self.services.write().unwrap();
            services.insert(name.to_string(), registration);
        }

        // 注册工厂
        {
            let mut factories = self.factories.write().unwrap();
            factories.insert(
                name.to_string(),
                Box::new(ScopedServiceFactory::new(factory)),
            );
        }

        // 注册类型映射
        {
            let mut mappings = self.type_mappings.write().unwrap();
            mappings.insert(TypeId::of::<T>(), name.to_string());
        }

        // 调用注册回调
        self.invoke_lifecycle_callbacks(name, LifecycleEvent::Registered)?;

        Ok(())
    }

    /// 解析服务
    pub fn resolve<T>(&self) -> Result<Arc<T>>
    where
        T: Send + Sync + 'static,
    {
        self.ensure_not_disposed()?;

        let service_name = {
            let mappings = self.type_mappings.read().unwrap();
            mappings.get(&TypeId::of::<T>())
                .cloned()
                .ok_or_else(|| Error::InvalidInput(format!("服务未注册: {}", std::any::type_name::<T>())))?
        };

        self.resolve_by_name(&service_name)
    }

    /// 根据名称解析服务
    pub fn resolve_by_name<T>(&self, name: &str) -> Result<Arc<T>>
    where
        T: Send + Sync + 'static,
    {
        self.ensure_not_disposed()?;

        let registration = {
            let services = self.services.read().unwrap();
            services.get(name)
                .cloned()
                .ok_or_else(|| Error::InvalidInput(format!("服务未找到: {}", name)))?
        };

        match registration.lifetime {
            ServiceLifetime::Singleton => self.resolve_singleton(name),
            ServiceLifetime::Transient => self.resolve_transient(name),
            ServiceLifetime::Scoped => self.resolve_scoped(name),
        }
    }

    /// 检查服务是否已注册
    pub fn is_registered<T>(&self) -> bool
    where
        T: Send + Sync + 'static,
    {
        let mappings = self.type_mappings.read().unwrap();
        mappings.contains_key(&TypeId::of::<T>())
    }

    /// 检查服务是否已注册（根据名称）
    pub fn is_registered_by_name(&self, name: &str) -> bool {
        let services = self.services.read().unwrap();
        services.contains_key(name)
    }

    /// 获取所有注册的服务信息
    pub fn get_registrations(&self) -> Result<Vec<ServiceInfo>> {
        self.ensure_not_disposed()?;

        let services = self.services.read().unwrap();
        Ok(services.values().map(|reg| ServiceInfo {
            name: reg.name.clone(),
            service_type: reg.service_type.clone(),
            lifetime: reg.lifetime.clone(),
            registered_at: reg.registered_at,
            dependencies: reg.dependencies.clone(),
        }).collect())
    }

    /// 添加生命周期回调
    pub fn add_lifecycle_callback<F>(&self, service_name: &str, callback: F) -> Result<()>
    where
        F: Fn(&str, LifecycleEvent) -> Result<()> + Send + Sync + 'static,
    {
        let mut callbacks = self.lifecycle_callbacks.write().unwrap();
        let service_callbacks = callbacks.entry(service_name.to_string()).or_insert_with(Vec::new);
        service_callbacks.push(Box::new(callback));
        Ok(())
    }

    /// 释放容器资源
    pub fn dispose(&self) -> Result<()> {
        *self.is_disposed.write().unwrap() = true;

        // 调用所有服务的销毁回调
        let service_names: Vec<String> = {
            let services = self.services.read().unwrap();
            services.keys().cloned().collect()
        };

        for name in service_names {
            let _ = self.invoke_lifecycle_callbacks(&name, LifecycleEvent::Disposing);
        }

        // 清理所有注册
        {
            let mut services = self.services.write().unwrap();
            services.clear();
        }

        {
            let mut instances = self.instances.write().unwrap();
            instances.clear();
        }

        {
            let mut mappings = self.type_mappings.write().unwrap();
            mappings.clear();
        }

        {
            let mut factories = self.factories.write().unwrap();
            factories.clear();
        }

        {
            let mut callbacks = self.lifecycle_callbacks.write().unwrap();
            callbacks.clear();
        }

        Ok(())
    }

    // 私有方法
    
    fn ensure_not_disposed(&self) -> Result<()> {
        if *self.is_disposed.read().unwrap() {
            return Err(Error::InvalidInput("容器已被释放".to_string()));
        }
        Ok(())
    }

    fn resolve_singleton<T>(&self, name: &str) -> Result<Arc<T>>
    where
        T: Send + Sync + 'static,
    {
        let instances = self.instances.read().unwrap();
        if let Some(instance) = instances.get(name) {
            instance.downcast_ref::<T>()
                .map(|typed_instance| Arc::new(unsafe { std::ptr::read(typed_instance) }))
                .ok_or_else(|| Error::InvalidInput(format!("类型转换失败: {}", name)))
        } else {
            Err(Error::InvalidInput(format!("单例实例未找到: {}", name)))
        }
    }

    fn resolve_transient<T>(&self, name: &str) -> Result<Arc<T>>
    where
        T: Send + Sync + 'static,
    {
        let factories = self.factories.read().unwrap();
        if let Some(factory) = factories.get(name) {
            let instance = factory.create(self)?;
            instance.downcast::<T>()
                .map_err(|_| Error::InvalidInput(format!("类型转换失败: {}", name)))
        } else {
            Err(Error::InvalidInput(format!("工厂未找到: {}", name)))
        }
    }

    fn resolve_scoped<T>(&self, name: &str) -> Result<Arc<T>>
    where
        T: Send + Sync + 'static,
    {
        // 对于作用域服务，我们检查是否已有实例
        {
            let instances = self.instances.read().unwrap();
            if let Some(instance) = instances.get(name) {
                if let Some(typed_instance) = instance.downcast_ref::<T>() {
                    return Ok(Arc::new(unsafe { std::ptr::read(typed_instance) }));
                }
            }
        }

        // 如果没有实例，创建新的
        let factories = self.factories.read().unwrap();
        if let Some(factory) = factories.get(name) {
            let instance = factory.create(self)?;
            
            // 存储实例用于后续请求
            {
                let mut instances = self.instances.write().unwrap();
                instances.insert(name.to_string(), instance.clone());
            }

            instance.downcast::<T>()
                .map_err(|_| Error::InvalidInput(format!("类型转换失败: {}", name)))
        } else {
            Err(Error::InvalidInput(format!("工厂未找到: {}", name)))
        }
    }

    fn invoke_lifecycle_callbacks(&self, service_name: &str, event: LifecycleEvent) -> Result<()> {
        let callbacks = self.lifecycle_callbacks.read().unwrap();
        if let Some(service_callbacks) = callbacks.get(service_name) {
            for callback in service_callbacks {
                callback(service_name, event.clone())?;
            }
        }
        Ok(())
    }
}

/// 服务工厂接口
trait ServiceFactory: Send + Sync {
    fn create(&self, container: &ProductionContainer) -> Result<Arc<dyn Any + Send + Sync>>;
}

/// 临时服务工厂
struct TransientServiceFactory<T, F>
where
    T: Send + Sync + 'static,
    F: Fn(&ProductionContainer) -> Result<T> + Send + Sync + 'static,
{
    factory_fn: F,
    _marker: std::marker::PhantomData<T>,
}

impl<T, F> TransientServiceFactory<T, F>
where
    T: Send + Sync + 'static,
    F: Fn(&ProductionContainer) -> Result<T> + Send + Sync + 'static,
{
    fn new(factory_fn: F) -> Self {
        Self {
            factory_fn,
            _marker: std::marker::PhantomData,
        }
    }
}

impl<T, F> ServiceFactory for TransientServiceFactory<T, F>
where
    T: Send + Sync + 'static,
    F: Fn(&ProductionContainer) -> Result<T> + Send + Sync + 'static,
{
    fn create(&self, container: &ProductionContainer) -> Result<Arc<dyn Any + Send + Sync>> {
        let instance = (self.factory_fn)(container)?;
        Ok(Arc::new(instance))
    }
}

/// 作用域服务工厂
struct ScopedServiceFactory<T, F>
where
    T: Send + Sync + 'static,
    F: Fn(&ProductionContainer) -> Result<T> + Send + Sync + 'static,
{
    factory_fn: F,
    _marker: std::marker::PhantomData<T>,
}

impl<T, F> ScopedServiceFactory<T, F>
where
    T: Send + Sync + 'static,
    F: Fn(&ProductionContainer) -> Result<T> + Send + Sync + 'static,
{
    fn new(factory_fn: F) -> Self {
        Self {
            factory_fn,
            _marker: std::marker::PhantomData,
        }
    }
}

impl<T, F> ServiceFactory for ScopedServiceFactory<T, F>
where
    T: Send + Sync + 'static,
    F: Fn(&ProductionContainer) -> Result<T> + Send + Sync + 'static,
{
    fn create(&self, container: &ProductionContainer) -> Result<Arc<dyn Any + Send + Sync>> {
        let instance = (self.factory_fn)(container)?;
        Ok(Arc::new(instance))
    }
}

/// 服务注册信息
#[derive(Debug, Clone)]
struct ServiceRegistration {
    name: String,
    service_type: String,
    lifetime: ServiceLifetime,
    registered_at: DateTime<Utc>,
    dependencies: Vec<String>,
}

/// 服务生命周期
#[derive(Debug, Clone, PartialEq)]
pub enum ServiceLifetime {
    /// 单例 - 整个应用程序生命周期内只有一个实例
    Singleton,
    /// 临时 - 每次请求都创建新实例
    Transient,
    /// 作用域 - 在特定作用域内是单例
    Scoped,
}

/// 服务信息
#[derive(Debug, Clone)]
pub struct ServiceInfo {
    pub name: String,
    pub service_type: String,
    pub lifetime: ServiceLifetime,
    pub registered_at: DateTime<Utc>,
    pub dependencies: Vec<String>,
}

/// 生命周期事件
#[derive(Debug, Clone)]
pub enum LifecycleEvent {
    Registered,
    Creating,
    Created,
    Disposing,
    Disposed,
}

/// 生命周期回调
type LifecycleCallback = dyn Fn(&str, LifecycleEvent) -> Result<()> + Send + Sync;

/// 服务作用域
pub struct ServiceScope {
    container: Arc<ProductionContainer>,
    scoped_instances: Arc<RwLock<HashMap<String, Arc<dyn Any + Send + Sync>>>>,
    is_disposed: Arc<RwLock<bool>>,
}

impl ServiceScope {
    pub fn new(container: Arc<ProductionContainer>) -> Self {
        Self {
            container,
            scoped_instances: Arc::new(RwLock::new(HashMap::new())),
            is_disposed: Arc::new(RwLock::new(false)),
        }
    }

    /// 在作用域内解析服务
    pub fn resolve<T>(&self) -> Result<Arc<T>>
    where
        T: Send + Sync + 'static,
    {
        if *self.is_disposed.read().unwrap() {
            return Err(Error::InvalidInput("作用域已被释放".to_string()));
        }

        // 首先检查作用域内的实例
        let service_name = {
            let mappings = self.container.type_mappings.read().unwrap();
            mappings.get(&TypeId::of::<T>())
                .cloned()
                .ok_or_else(|| Error::InvalidInput(format!("服务未注册: {}", std::any::type_name::<T>())))?
        };

        // 检查是否已在作用域内创建
        {
            let instances = self.scoped_instances.read().unwrap();
            if let Some(instance) = instances.get(&service_name) {
                if let Some(typed_instance) = instance.downcast_ref::<T>() {
                    return Ok(Arc::new(unsafe { std::ptr::read(typed_instance) }));
                }
            }
        }

        // 从容器解析并缓存在作用域内
        let instance = self.container.resolve::<T>()?;
        
        {
            let mut instances = self.scoped_instances.write().unwrap();
            instances.insert(service_name, instance.clone() as Arc<dyn Any + Send + Sync>);
        }

        Ok(instance)
    }

    /// 释放作用域
    pub fn dispose(&self) -> Result<()> {
        *self.is_disposed.write().unwrap() = true;
        
        let mut instances = self.scoped_instances.write().unwrap();
        instances.clear();
        
        Ok(())
    }
}

impl Drop for ServiceScope {
    fn drop(&mut self) {
        let _ = self.dispose();
    }
}

/// 容器构建器
pub struct ContainerBuilder {
    registrations: Vec<Box<dyn Fn(&ProductionContainer) -> Result<()> + Send + Sync>>,
}

impl ContainerBuilder {
    pub fn new() -> Self {
        Self {
            registrations: Vec::new(),
        }
    }

    /// 注册单例服务
    pub fn register_singleton<T>(mut self, name: &str, instance: T) -> Self
    where
        T: Send + Sync + 'static,
    {
        let name = name.to_string();
        self.registrations.push(Box::new(move |container| {
            container.register_singleton(&name, instance)
        }));
        self
    }

    /// 注册临时服务
    pub fn register_transient<T, F>(mut self, name: &str, factory: F) -> Self
    where
        T: Send + Sync + 'static,
        F: Fn(&ProductionContainer) -> Result<T> + Send + Sync + 'static,
    {
        let name = name.to_string();
        self.registrations.push(Box::new(move |container| {
            container.register_transient(&name, factory)
        }));
        self
    }

    /// 注册作用域服务
    pub fn register_scoped<T, F>(mut self, name: &str, factory: F) -> Self
    where
        T: Send + Sync + 'static,
        F: Fn(&ProductionContainer) -> Result<T> + Send + Sync + 'static,
    {
        let name = name.to_string();
        self.registrations.push(Box::new(move |container| {
            container.register_scoped(&name, factory)
        }));
        self
    }

    /// 构建容器
    pub fn build(self) -> Result<ProductionContainer> {
        let container = ProductionContainer::new();
        
        for registration in self.registrations {
            registration(&container)?;
        }
        
        Ok(container)
    }
}

/// 默认实现
impl Default for ContainerBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// 容器扩展 trait
pub trait ContainerExtensions {
    /// 创建服务作用域
    fn create_scope(&self) -> ServiceScope;
    
    /// 尝试解析服务（如果未注册返回None而不是错误）
    fn try_resolve<T>(&self) -> Option<Arc<T>>
    where
        T: Send + Sync + 'static;
}

impl ContainerExtensions for ProductionContainer {
    fn create_scope(&self) -> ServiceScope {
        ServiceScope::new(Arc::new(self.clone()))
    }
    
    fn try_resolve<T>(&self) -> Option<Arc<T>>
    where
        T: Send + Sync + 'static,
    {
        self.resolve::<T>().ok()
    }
}

/// 使容器可克隆
impl Clone for ProductionContainer {
    fn clone(&self) -> Self {
        Self {
            services: self.services.clone(),
            instances: self.instances.clone(),
            type_mappings: self.type_mappings.clone(),
            factories: self.factories.clone(),
            lifecycle_callbacks: self.lifecycle_callbacks.clone(),
            is_disposed: self.is_disposed.clone(),
        }
    }
} 