// Vector Utils Module
// 这个模块包含向量模块的辅助工具和示例代码

// Re-export components
pub mod benchmark;
pub mod examples;
pub mod vector;  // 添加vector模块
pub mod parallel;  // 添加并行处理模块

// Re-export common functions
pub use benchmark::{BenchmarkResult, BenchmarkConfig, IndexBenchmark};
pub use examples::{run_vector_storage_example, vptree_example};
// 导出vector模块的feature_similarity函数
pub use self::vector::feature_similarity;
pub use self::vector::normalize_vector;

// 从benchmark.rs中新增以下函数导出 - 此修改基于benchmark.rs中检测到的函数
pub use self::benchmark::run_benchmark_example;

// Re-export important utilities
pub use parallel::ParallelVectorOps;  // 导出并行操作工具 