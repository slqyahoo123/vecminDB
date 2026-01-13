//! 向量操作资源管理模块
//! 
//! 提供专门针对向量操作的资源分配和管理

use crate::Result;
use serde::{Serialize, Deserialize};
use super::{ResourceManager, ResourceRequest, ResourceAllocation, ResourceType};
use std::sync::Arc;

/// 向量操作类型
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum VectorOperationType {
    /// 向量插入
    Insert,
    /// 向量搜索
    Search,
    /// 索引构建
    IndexBuild,
    /// 索引优化
    IndexOptimize,
    /// 批量操作
    BatchOperation,
}

impl std::fmt::Display for VectorOperationType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            VectorOperationType::Insert => write!(f, "Insert"),
            VectorOperationType::Search => write!(f, "Search"),
            VectorOperationType::IndexBuild => write!(f, "IndexBuild"),
            VectorOperationType::IndexOptimize => write!(f, "IndexOptimize"),
            VectorOperationType::BatchOperation => write!(f, "BatchOperation"),
        }
    }
}

/// 向量操作资源需求
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VectorOperationResource {
    /// 操作类型
    pub operation_type: VectorOperationType,
    /// 向量数量
    pub vector_count: usize,
    /// 向量维度
    pub dimension: usize,
    /// 是否使用GPU
    pub use_gpu: bool,
    /// 并行度
    pub parallelism: usize,
}

impl VectorOperationResource {
    /// 创建新的向量操作资源需求
    pub fn new(
        operation_type: VectorOperationType,
        vector_count: usize,
        dimension: usize,
    ) -> Self {
        Self {
            operation_type,
            vector_count,
            dimension,
            use_gpu: false,
            parallelism: num_cpus::get(),
        }
    }

    /// 设置使用GPU
    pub fn with_gpu(mut self) -> Self {
        self.use_gpu = true;
        self
    }

    /// 设置并行度
    pub fn with_parallelism(mut self, parallelism: usize) -> Self {
        self.parallelism = parallelism;
        self
    }

    /// 估算内存需求（字节）
    pub fn estimate_memory(&self) -> usize {
        match self.operation_type {
            VectorOperationType::Insert => {
                // 向量数据 + 索引开销
                self.vector_count * self.dimension * 4 + self.vector_count * 64
            },
            VectorOperationType::Search => {
                // 查询向量 + 结果缓存
                self.dimension * 4 + self.vector_count * 128
            },
            VectorOperationType::IndexBuild => {
                // 向量数据 + 索引结构 + 临时缓冲区
                self.vector_count * self.dimension * 4 * 3
            },
            VectorOperationType::IndexOptimize => {
                // 索引数据 + 优化缓冲区
                self.vector_count * self.dimension * 4 * 2
            },
            VectorOperationType::BatchOperation => {
                // 批量数据 + 处理缓冲区
                self.vector_count * self.dimension * 4 * 2
            },
        }
    }

    /// 估算CPU需求（核心数）
    pub fn estimate_cpu(&self) -> usize {
        match self.operation_type {
            VectorOperationType::Insert => 1.max(self.parallelism / 2),
            VectorOperationType::Search => self.parallelism,
            VectorOperationType::IndexBuild => self.parallelism,
            VectorOperationType::IndexOptimize => self.parallelism,
            VectorOperationType::BatchOperation => self.parallelism,
        }
    }

    /// 估算GPU需求（如果使用）
    pub fn estimate_gpu(&self) -> usize {
        if self.use_gpu {
            match self.operation_type {
                VectorOperationType::Search | 
                VectorOperationType::IndexBuild => 1,
                _ => 0,
            }
        } else {
            0
        }
    }
}

/// 向量资源管理器
pub struct VectorResourceManager {
    /// 底层资源管理器
    resource_manager: Arc<ResourceManager>,
}

impl VectorResourceManager {
    /// 创建新的向量资源管理器
    pub fn new(resource_manager: Arc<ResourceManager>) -> Self {
        Self { resource_manager }
    }

    /// 为向量操作请求资源
    pub async fn request_for_operation(
        &self,
        operation: &VectorOperationResource,
    ) -> Result<VectorResourceAllocation> {
        let mut allocations = Vec::new();

        // 请求内存
        let memory_size = operation.estimate_memory();
        let memory_request = ResourceRequest {
            resource_type: ResourceType::Memory,
            amount: memory_size,
            priority: self.get_priority(operation.operation_type),
            timeout_ms: Some(30000), // 30秒超时
        };
        let memory_alloc = self.resource_manager.allocate(memory_request).await?;
        allocations.push(memory_alloc);

        // 请求CPU
        let cpu_count = operation.estimate_cpu();
        let cpu_request = ResourceRequest {
            resource_type: ResourceType::CPU,
            amount: cpu_count,
            priority: self.get_priority(operation.operation_type),
            timeout_ms: Some(30000),
        };
        let cpu_alloc = self.resource_manager.allocate(cpu_request).await?;
        allocations.push(cpu_alloc);

        // 如果需要GPU，请求GPU资源
        if operation.use_gpu {
            let gpu_count = operation.estimate_gpu();
            if gpu_count > 0 {
                let gpu_request = ResourceRequest {
                    resource_type: ResourceType::GPU,
                    amount: gpu_count,
                    priority: self.get_priority(operation.operation_type),
                    timeout_ms: Some(30000),
                };
                let gpu_alloc = self.resource_manager.allocate(gpu_request).await?;
                allocations.push(gpu_alloc);
            }
        }

        Ok(VectorResourceAllocation {
            operation_type: operation.operation_type,
            allocations,
        })
    }

    /// 释放向量操作资源
    pub async fn release(&self, allocation: VectorResourceAllocation) -> Result<()> {
        for alloc in allocation.allocations {
            self.resource_manager.release(alloc).await?;
        }
        Ok(())
    }

    // 内部辅助方法

    fn get_priority(&self, operation_type: VectorOperationType) -> u8 {
        match operation_type {
            VectorOperationType::Search => 8, // 搜索优先级高
            VectorOperationType::Insert => 6,
            VectorOperationType::BatchOperation => 5,
            VectorOperationType::IndexOptimize => 4,
            VectorOperationType::IndexBuild => 3, // 索引构建优先级低
        }
    }
}

/// 向量资源分配
pub struct VectorResourceAllocation {
    /// 操作类型
    pub operation_type: VectorOperationType,
    /// 底层资源分配
    pub allocations: Vec<ResourceAllocation>,
}

impl VectorResourceAllocation {
    /// 获取分配的内存大小
    pub fn memory_size(&self) -> usize {
        self.allocations
            .iter()
            .filter(|a| matches!(a.resource_type, ResourceType::Memory))
            .map(|a| a.amount)
            .sum()
    }

    /// 获取分配的CPU数量
    pub fn cpu_count(&self) -> usize {
        self.allocations
            .iter()
            .filter(|a| matches!(a.resource_type, ResourceType::CPU))
            .map(|a| a.amount)
            .sum()
    }

    /// 获取分配的GPU数量
    pub fn gpu_count(&self) -> usize {
        self.allocations
            .iter()
            .filter(|a| matches!(a.resource_type, ResourceType::GPU))
            .map(|a| a.amount)
            .sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vector_operation_resource() {
        let resource = VectorOperationResource::new(
            VectorOperationType::Search,
            1000,
            128,
        );

        assert_eq!(resource.operation_type, VectorOperationType::Search);
        assert_eq!(resource.vector_count, 1000);
        assert_eq!(resource.dimension, 128);
        assert!(!resource.use_gpu);
    }

    #[test]
    fn test_memory_estimation() {
        let insert_resource = VectorOperationResource::new(
            VectorOperationType::Insert,
            1000,
            128,
        );
        let memory = insert_resource.estimate_memory();
        assert!(memory > 0);

        let search_resource = VectorOperationResource::new(
            VectorOperationType::Search,
            10,
            128,
        );
        let search_memory = search_resource.estimate_memory();
        assert!(search_memory > 0);
        assert!(memory > search_memory); // 插入需要更多内存
    }

    #[test]
    fn test_cpu_estimation() {
        let resource = VectorOperationResource::new(
            VectorOperationType::IndexBuild,
            10000,
            256,
        ).with_parallelism(8);

        let cpu_count = resource.estimate_cpu();
        assert_eq!(cpu_count, 8);
    }

    #[test]
    fn test_gpu_estimation() {
        let resource_no_gpu = VectorOperationResource::new(
            VectorOperationType::Search,
            1000,
            128,
        );
        assert_eq!(resource_no_gpu.estimate_gpu(), 0);

        let resource_with_gpu = VectorOperationResource::new(
            VectorOperationType::Search,
            1000,
            128,
        ).with_gpu();
        assert_eq!(resource_with_gpu.estimate_gpu(), 1);
    }
}



