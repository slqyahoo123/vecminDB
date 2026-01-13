// 共享代码模块 - 用于多个系统共享的核心功能
// Shared code module - Core functionality shared between multiple systems

pub mod processor_core;

// 重新导出常用组件，方便使用
// Re-export common components for convenience
pub use processor_core::{
    ProcessorStats,
    SharedProcessorContext,
    SharedProcessorResult,
}; 