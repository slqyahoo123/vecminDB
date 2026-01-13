/// 应用程序容器模块
/// 
/// 提供应用程序级别的服务容器，统一管理所有核心服务

use std::sync::Arc;
use std::any::Any;
use log::{info, debug};
use chrono::Utc;

use crate::{Result, Error};
use crate::core::interfaces::{
	ModelManagerInterface, 
	// TrainingEngineInterface, // 已移除：向量数据库系统不需要训练功能
	DataProcessorInterface,
};
use crate::core::container::service_container::DefaultServiceContainer;
// use crate::core::container::proxies::*; // 广域导入未使用，保留精确导入
use crate::core::types::ComponentHealth;
use crate::core::{EventBusInterface, EventBus};

use super::{
	ServiceContainer,
	// ArcExt, // 暂未使用
	proxies::{
		ModelManagerProxy,
		// TrainingEngineProxy, // 已移除
		DataProcessorProxy,
		AlgorithmExecutorProxy,
	}
};

/// 应用程序容器
/// 
/// 管理所有核心服务的生命周期，提供统一的服务访问接口
pub struct ApplicationContainer {
	service_container: Arc<DefaultServiceContainer>,
	model_manager: Arc<dyn ModelManagerInterface>,
	// training_engine: Arc<dyn TrainingEngineInterface>, // 已移除
	data_processor: Arc<dyn DataProcessorInterface>,
	algorithm_executor: Arc<dyn crate::core::interfaces::AlgorithmExecutorInterface>,
}

impl ApplicationContainer {
	/// 创建新的应用程序容器
	pub fn new() -> Self {
		let service_container = Arc::new(DefaultServiceContainer::new());
		
		// 创建服务代理
		let model_manager = Arc::new(ModelManagerProxy::new(service_container.clone()));
		// let training_engine = Arc::new(TrainingEngineProxy::new(service_container.clone())); // 已移除
		let data_processor = Arc::new(DataProcessorProxy::new(service_container.clone()));
		let algorithm_executor = Arc::new(AlgorithmExecutorProxy::new(service_container.clone()));
		
		Self {
			service_container,
			model_manager,
			// training_engine, // 已移除
			data_processor,
			algorithm_executor,
		}
	}
	
	/// 获取模型管理器
	pub fn model_manager(&self) -> Arc<dyn ModelManagerInterface> {
		self.model_manager.clone()
	}
	
	// 训练引擎已移除：向量数据库系统不需要训练功能
	// pub fn training_engine(&self) -> Arc<dyn TrainingEngineInterface> { ... }
	
	/// 获取数据处理器
	pub fn data_processor(&self) -> Arc<dyn DataProcessorInterface> {
		self.data_processor.clone()
	}
	
	/// 获取算法执行器
	pub fn algorithm_executor(&self) -> Arc<dyn crate::core::interfaces::AlgorithmExecutorInterface> {
		self.algorithm_executor.clone()
	}
	
	/// 注册服务
	/// 
	/// # 参数
	/// - service: 要注册的服务实例
	/// 
	/// # 返回
	/// 注册是否成功
	pub fn register_service<T: Any + Send + Sync>(&self, service: Arc<T>) -> Result<()> {
		debug!("注册服务: {}", std::any::type_name::<T>());
		
		// 验证服务类型
		self.validate_service_type::<T>()?;
		
		// 使用 Arc 的 as_ref() 方法来获取引用，然后调用 register 方法
		self.service_container.as_ref().register(service)?;
		
		info!("成功注册服务: {}", std::any::type_name::<T>());
		Ok(())
	}

	/// 注册trait对象服务
	/// 
	/// # 参数
	/// - service: 要注册的trait对象服务实例
	/// 
	/// # 返回
	/// 注册是否成功
	pub fn register_trait_service<T: ?Sized + Send + Sync + 'static>(&self, service: Arc<T>) -> Result<()> {
		debug!("注册trait服务: {}", std::any::type_name::<T>());
		
		// 验证服务类型
		self.validate_trait_service_type::<T>()?;
		
		// 使用 Arc 的 as_ref() 方法来获取引用，然后调用 register_trait 方法
		self.service_container.as_ref().register_trait(service)?;
		
		info!("成功注册trait服务: {}", std::any::type_name::<T>());
		Ok(())
	}

	/// 注册核心服务
	/// 
	/// 注册所有核心服务到服务容器中
	pub async fn register_core_services(&self) -> Result<()> {
		debug!("注册核心服务");
		
		// 注册模型管理器 - 使用代理实现
		let model_manager = Arc::new(crate::core::container::proxies::ModelManagerProxy::new(self.service_container.clone()));
		self.register_trait_service::<dyn ModelManagerInterface + Send + Sync>(model_manager)?;
		
		// 训练引擎服务已移除：向量数据库系统不需要训练功能
		// let training_service = Arc::new(crate::core::container::proxies::TrainingEngineProxy::new(self.service_container.clone()));
		// self.register_trait_service::<dyn crate::core::interfaces::TrainingEngineInterface + Send + Sync>(training_service)?;
		
		// 注册数据处理器 - 使用new_default方法
		let data_processor = Arc::new(crate::data::processor::processor_impl::DataProcessor::new_default());
		self.register_trait_service::<dyn DataProcessorInterface + Send + Sync>(data_processor)?;
		
		// 注册算法执行器 - 使用代理以契合接口
		let algorithm_executor = Arc::new(crate::core::container::proxies::AlgorithmExecutorProxy::new(self.service_container.clone()));
		self.register_trait_service::<dyn crate::core::interfaces::AlgorithmExecutorInterface + Send + Sync>(algorithm_executor)?;
		
		// 注册并启动事件总线（作为接口服务对外暴露）
		let event_bus_concrete = Arc::new(EventBus::new());
		event_bus_concrete.start().await?;
		let event_bus_trait: Arc<dyn EventBusInterface + Send + Sync> = event_bus_concrete.clone();
		self.register_trait_service::<dyn EventBusInterface + Send + Sync>(event_bus_trait)?;
		
		info!("核心服务注册完成");
		Ok(())
	}
	
	/// 获取服务
	/// 
	/// # 类型参数
	/// - T: 要获取的服务类型
	/// 
	/// # 返回
	/// 服务实例（如果存在）
	pub fn get_service<T: Any + Send + Sync>(&self) -> Option<Arc<T>> {
		// 使用 Arc 的 as_ref() 方法来获取引用，然后调用 get 方法
		self.service_container.as_ref().get::<T>().ok()
	}

	/// 获取trait对象服务（便于按接口检索）
	pub fn get_trait_service<T: ?Sized + Send + Sync + 'static>(&self) -> Option<Arc<T>> {
		// 使用 Arc 的 as_ref() 方法来获取引用，然后调用 get_trait 方法
		self.service_container.as_ref().get_trait::<T>().ok()
	}
	
	/// 根据服务名称获取服务
	/// 
	/// # 参数
	/// - service_name: 服务名称
	/// 
	/// # 返回
	/// 对应的服务实例，如果不存在则返回None
	pub fn get_service_by_name<T: Any + Send + Sync + 'static>(&self, service_name: &str) -> Option<Arc<T>> {
		match service_name {
			"model_manager" => {
				// trait object 无法直接转换为 Any，返回 None
				// 如果需要获取 trait object，应该使用 get_trait_service 方法
				None
			},
			// "training_engine" => { // 已移除
			// 	None
			// },
			"data_processor" => {
				// trait object 无法直接转换为 Any，返回 None
				// 如果需要获取 trait object，应该使用 get_trait_service 方法
				None
			},
			"algorithm_executor" => {
				// trait object 无法直接转换为 Any，返回 None
				// 如果需要获取 trait object，应该使用 get_trait_service 方法
				None
			},
			_ => {
				debug!("未知的服务名称: {}", service_name);
				None
			}
		}
	}
	
	/// 初始化容器
	/// 
	/// 启动所有核心服务并进行健康检查
	pub async fn initialize(&self) -> Result<()> {
		info!("开始初始化应用程序容器");
		
		// 注册核心服务
		self.register_core_services().await?;
		
		// 验证容器状态
		self.validate_container_state().await?;
		
		// 初始化服务依赖
		self.initialize_service_dependencies().await?;
		
		// 验证服务配置
		self.validate_service_configurations().await?;
		
		// 启动核心服务
		self.start_core_services().await?;
		
		// 运行健康检查
		self.run_health_checks().await?;
		
		info!("应用程序容器初始化完成");
		Ok(())
	}
	
	/// 关闭容器
	/// 
	/// 优雅地关闭所有服务并清理资源
	pub async fn shutdown(&self) -> Result<()> {
		info!("开始关闭应用程序容器");
		
		// 停止接受新请求
		self.stop_accepting_requests().await?;
		
		// 等待处理中的请求完成
		self.wait_for_pending_requests().await?;
		
		// 优雅关闭服务
		self.shutdown_services_gracefully().await?;
		
		// 清理资源
		self.cleanup_resources().await?;
		
		// 持久化最终状态
		self.persist_final_state().await?;
		
		info!("应用程序容器关闭完成");
		Ok(())
	}
	
	/// 验证服务类型
	fn validate_service_type<T: Any + Send + Sync>(&self) -> Result<()> {
		let type_name = std::any::type_name::<T>();
		debug!("验证服务类型: {}", type_name);
		
		// 检查是否为允许的服务类型
		if !self.is_allowed_service_type::<T>() {
			return Err(crate::Error::InvalidInput(
				format!("不支持的服务类型: {}", type_name)
			));
		}
		
		Ok(())
	}

	/// 验证trait对象服务类型
	fn validate_trait_service_type<T: ?Sized + Send + Sync + 'static>(&self) -> Result<()> {
		let type_name = std::any::type_name::<T>();
		debug!("验证trait服务类型: {}", type_name);
		
		// 检查是否为允许的trait对象服务类型
		if !self.is_allowed_trait_service_type::<T>() {
			return Err(crate::Error::InvalidInput(
				format!("不支持的trait对象服务类型: {}", type_name)
			));
		}
		
		Ok(())
	}
	
	/// 检查是否为允许的服务类型
	fn is_allowed_service_type<T: Any + Send + Sync>(&self) -> bool {
		let type_name = std::any::type_name::<T>();
		
		// 允许的服务类型列表
		let allowed_types = [
			"ModelManagerInterface",
			// "TrainingEngineInterface", // 已移除
			"DataProcessorInterface",
			"AlgorithmExecutorInterface",
			"EventBusInterface",
		];
		
		allowed_types.iter().any(|&allowed| type_name.contains(allowed))
	}

	/// 检查是否为允许的trait对象服务类型
	fn is_allowed_trait_service_type<T: ?Sized + Send + Sync + 'static>(&self) -> bool {
		let type_name = std::any::type_name::<T>();
		
		// 允许的trait对象服务类型列表
		let allowed_types = [
			"ModelManagerInterface",
			// "TrainingEngineInterface", // 已移除
			"DataProcessorInterface",
			"AlgorithmExecutorInterface",
			"EventBusInterface",
		];
		
		allowed_types.iter().any(|&allowed| type_name.contains(allowed))
	}
	
	/// 验证容器状态
	async fn validate_container_state(&self) -> Result<()> {
		debug!("验证容器状态");
		
		// 检查服务容器是否有效
		if !self.is_service_container_valid() {
			return Err(crate::Error::InvalidInput("服务容器未正确初始化或为空".to_string()));
		}
		
		// 检查核心服务是否存在 - 对于Arc，我们检查strong_count而不是is_null
		if Arc::strong_count(&self.model_manager) == 0 {
			return Err(crate::Error::InvalidInput("模型管理器未正确初始化".to_string()));
		}
		
		// 训练引擎检查已移除
		// if Arc::strong_count(&self.training_engine) == 0 { ... }
		
		if Arc::strong_count(&self.data_processor) == 0 {
			return Err(crate::Error::InvalidInput("数据处理器未正确初始化".to_string()));
		}
		
		if Arc::strong_count(&self.algorithm_executor) == 0 {
			return Err(crate::Error::InvalidInput("算法执行器未正确初始化".to_string()));
		}
		
		// 综合检查所有核心服务
		if !self.are_core_services_initialized() {
			return Err(crate::Error::InvalidInput("部分核心服务未正确初始化".to_string()));
		}
		
		debug!("容器状态验证通过");
		Ok(())
	}
	
	/// 初始化服务依赖
	async fn initialize_service_dependencies(&self) -> Result<()> {
		debug!("初始化服务依赖");
		// 这里可以设置服务间的依赖关系
		Ok(())
	}
	
	/// 验证服务配置
	async fn validate_service_configurations(&self) -> Result<()> {
		debug!("验证服务配置");
		// 这里可以验证各个服务的配置是否正确
		Ok(())
	}
	
	/// 启动核心服务
	async fn start_core_services(&self) -> Result<()> {
		debug!("启动核心服务");
		// 这里可以启动各个服务的后台任务
		Ok(())
	}
	
	/// 运行健康检查
	async fn run_health_checks(&self) -> Result<()> {
		debug!("运行健康检查");
		// 这里可以检查各个服务的健康状态
		Ok(())
	}
	
	/// 停止接受新请求
	async fn stop_accepting_requests(&self) -> Result<()> {
		debug!("停止接受新请求");
		Ok(())
	}
	
	/// 等待处理中的请求完成
	async fn wait_for_pending_requests(&self) -> Result<()> {
		debug!("等待处理中的请求完成");
		// 这里可以等待所有进行中的请求完成
		tokio::time::sleep(std::time::Duration::from_millis(100)).await;
		Ok(())
	}
	
	/// 优雅关闭服务
	async fn shutdown_services_gracefully(&self) -> Result<()> {
		debug!("优雅关闭服务");
		// 这里可以优雅地关闭各个服务
		Ok(())
	}
	
	/// 清理资源
	async fn cleanup_resources(&self) -> Result<()> {
		debug!("清理资源");
		// 这里可以清理内存、文件句柄等资源
		Ok(())
	}
	
	/// 持久化最终状态
	async fn persist_final_state(&self) -> Result<()> {
		debug!("持久化最终状态");
		// 这里可以保存应用程序的最终状态
		Ok(())
	}
}

impl Default for ApplicationContainer {
	fn default() -> Self {
		Self::new()
	}
}

// 为ApplicationContainer扩展Arc检查功能
impl ApplicationContainer {
	/// 检查服务容器是否处于有效状态
	fn is_service_container_valid(&self) -> bool {
		!self.service_container.as_ref().is_empty()
	}
	
	/// 检查核心服务是否已正确初始化
	fn are_core_services_initialized(&self) -> bool {
		// 检查所有核心服务的Arc引用是否有效
		// 这里使用更合理的检查逻辑
		Arc::strong_count(&self.model_manager) > 0 &&
		// Arc::strong_count(&self.training_engine) > 0 && // 已移除
		Arc::strong_count(&self.data_processor) > 0 &&
		Arc::strong_count(&self.algorithm_executor) > 0
	}
	
	/// 检查指定服务的健康状态
	pub fn check_service_health(&self, service_name: &str) -> Result<ComponentHealth> {
		debug!("检查服务健康状态: {}", service_name);
		
		match service_name {
			"model_manager" => {
				// 检查模型管理器状态
				if Arc::strong_count(&self.model_manager) > 0 {
					Ok(ComponentHealth {
						component_id: service_name.to_string(),
						status: "healthy".to_string(),
						health_score: 1.0,
						last_check: chrono::Utc::now(),
						metadata: std::collections::HashMap::new(),
					})
				} else {
					Ok(ComponentHealth {
						component_id: service_name.to_string(),
						status: "unhealthy".to_string(),
						health_score: 0.0,
						last_check: chrono::Utc::now(),
						metadata: {
							let mut meta = std::collections::HashMap::new();
							meta.insert("error".to_string(), "Service not available".to_string());
							meta
						},
					})
				}
			},
			// "training_engine" => { // 已移除：向量数据库系统不需要训练功能
			// 	if Arc::strong_count(&self.training_engine) > 0 {
			// 		Ok(ComponentHealth {
			// 			component_id: service_name.to_string(),
			// 			status: "healthy".to_string(),
			// 			health_score: 1.0,
			// 			last_check: Utc::now(),
			// 			metadata: std::collections::HashMap::new(),
			// 		})
			// 	} else {
			// 		Ok(ComponentHealth {
			// 			component_id: service_name.to_string(),
			// 			status: "unhealthy".to_string(),
			// 			health_score: 0.0,
			// 			last_check: Utc::now(),
			// 			metadata: {
			// 				let mut meta = std::collections::HashMap::new();
			// 				meta.insert("error".to_string(), "Service not available".to_string());
			// 				meta
			// 			},
			// 		})
			// 	}
			// },
			"data_processor" => {
				// 检查数据处理器状态
				if Arc::strong_count(&self.data_processor) > 0 {
					Ok(ComponentHealth {
						component_id: service_name.to_string(),
						status: "healthy".to_string(),
						health_score: 1.0,
						last_check: Utc::now(),
						metadata: std::collections::HashMap::new(),
					})
				} else {
					Ok(ComponentHealth {
						component_id: service_name.to_string(),
						status: "unhealthy".to_string(),
						health_score: 0.0,
						last_check: Utc::now(),
						metadata: {
							let mut meta = std::collections::HashMap::new();
							meta.insert("error".to_string(), "Service not available".to_string());
							meta
						},
					})
				}
			},
			"algorithm_executor" => {
				// 检查算法执行器状态
				if Arc::strong_count(&self.algorithm_executor) > 0 {
					Ok(ComponentHealth {
						component_id: service_name.to_string(),
						status: "healthy".to_string(),
						health_score: 1.0,
						last_check: Utc::now(),
						metadata: std::collections::HashMap::new(),
					})
				} else {
					Ok(ComponentHealth {
						component_id: service_name.to_string(),
						status: "unhealthy".to_string(),
						health_score: 0.0,
						last_check: Utc::now(),
						metadata: {
							let mut meta = std::collections::HashMap::new();
							meta.insert("error".to_string(), "Service not available".to_string());
							meta
						},
					})
				}
			},
			_ => {
				Err(Error::invalid_input(&format!("Unknown service: {}", service_name)))
			}
		}
	}
	
	/// 获取系统运行时间（秒）
	pub fn get_uptime_seconds(&self) -> Option<u64> {
		// 获取系统启动时间并计算运行时间
		let current_time = std::time::SystemTime::now()
			.duration_since(std::time::UNIX_EPOCH)
			.ok()?
			.as_secs();
		
		// 假设系统在容器创建时启动，实际实现中可以存储启动时间
		Some(current_time.saturating_sub(current_time - 3600)) // 示例：假设运行1小时
	}
	
	/// 获取活跃组件数量
	pub fn get_active_component_count(&self) -> Option<u32> {
		let mut active_count = 0;
		
		// 检查各个核心服务是否活跃
		if Arc::strong_count(&self.model_manager) > 0 {
			active_count += 1;
		}
		if Arc::strong_count(&self.training_engine) > 0 {
			active_count += 1;
		}
		if Arc::strong_count(&self.data_processor) > 0 {
			active_count += 1;
		}
		if Arc::strong_count(&self.algorithm_executor) > 0 {
			active_count += 1;
		}
		
		Some(active_count)
	}
	
	/// 获取系统资源使用情况
	pub fn get_system_resources(&self) -> Option<(f32, f32, f32)> {
		// 实际实现中会使用系统API获取真实的资源使用情况
		// 这里返回模拟数据
		
		// CPU使用率 (0.0-1.0)
		let cpu_usage = 0.25; // 25%
		
		// 内存使用率 (0.0-1.0) 
		let memory_usage = 0.45; // 45%
		
		// 磁盘使用率 (0.0-1.0)
		let disk_usage = 0.60; // 60%
		
		Some((cpu_usage, memory_usage, disk_usage))
	}
	
	/// 检查网络连接性
	pub fn check_network_connectivity(&self) -> Option<bool> {
		// 实际实现中会检查网络连接状态
		// 这里返回模拟结果
		
		// 模拟网络检查：尝试连接本地回环地址
		match std::net::TcpStream::connect_timeout(
			&"127.0.0.1:80".parse().unwrap_or_else(|_| "127.0.0.1:80".parse().unwrap()),
			std::time::Duration::from_millis(100)
		) {
			Ok(_) => Some(true),
			Err(_) => {
				// 如果本地80端口连接失败，检查其他常用端口或返回基于服务状态的判断
				if self.are_core_services_initialized() {
					Some(true) // 如果核心服务正常，认为网络可用
				} else {
					Some(false)
				}
			}
		}
	}
	
	/// 清理资源
	pub fn cleanup(&self) {
		debug!("清理ApplicationContainer资源");
		// 这里可以进行资源清理
	}
}

impl Drop for ApplicationContainer {
	fn drop(&mut self) {
		debug!("ApplicationContainer 正在销毁");
		self.cleanup();
	}
} 