/// 代理模块
/// 
/// 提供各种服务的代理实现，通过服务容器获取真实服务或使用默认实现

pub mod model_manager;
// pub mod training_engine; // 已移除：向量数据库系统不需要训练功能
pub mod data_processor;
pub mod algorithm_executor;

// 重新导出代理类型
pub use model_manager::ModelManagerProxy;
// pub use training_engine::TrainingEngineProxy; // 已移除
pub use data_processor::DataProcessorProxy;
pub use algorithm_executor::AlgorithmExecutorProxy; 