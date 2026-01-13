//! 接口定义模块
//! 用于解决模块间循环依赖，提供清晰的接口抽象
//!
//! ## 模块依赖关系说明
//! 
//! 本模块的目的是解决循环依赖问题，特别是Storage、Model、Algorithm模块之间的依赖。
//! 通过定义抽象接口，使得各模块依赖于接口而非具体实现，从而打破循环依赖。
//! 
//! ## 依赖方向：
//! 1. storage模块 -> interfaces::model::ModelManagerInterface（接口）
//! 2. storage模块 -> interfaces::algorithm::AlgorithmManagerInterface（接口）
//! 3. model模块 -> interfaces::storage::StorageInterface（接口）
//! 4. algorithm模块 -> interfaces::storage::StorageInterface（接口）
//! 5. db模块 -> interfaces::storage::StorageInterface（接口）
//! 
//! 具体实现：
//! - storage::Storage实现StorageInterface
//! - model::manager::ModelManager实现ModelManagerInterface
//! - algorithm::manager::AlgorithmManager实现AlgorithmManagerInterface
//! - db::DBEngine实现DatabaseInterface
//! 
//! 这种方式确保了各模块不会直接互相依赖，而是通过接口解耦。

pub mod storage;
pub mod db;
pub mod algorithm;

// 重导出常用接口
pub use storage::StorageInterface;
pub use db::DatabaseInterface;
pub use algorithm::{
    AlgorithmManagerInterface, AlgorithmExecutorInterface,
    AlgorithmDefinitionInterface, AlgorithmParametersInterface, AlgorithmResultInterface,
    AlgorithmTypeInterface, AlgorithmTaskInterface, ValidationResultInterface,
    ResourceUsageInterface, ExecutionStatusInterface
}; 