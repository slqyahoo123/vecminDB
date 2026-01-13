// Vector Core Module
// 这个模块包含向量的核心功能：距离计算、基础操作和批量操作

// Re-export components
pub mod distance;
pub mod ops;
pub mod operations;

// Re-export common types and functions
pub use distance::{Distance, DistanceMetric, EuclideanDistance, CosineDistance, ManhattanDistance};
pub use crate::vector::Vector;
pub use crate::vector::storage::storage::{VectorBatchRequest, VectorSearchResponse};
pub use operations::SimilarityMetric; 