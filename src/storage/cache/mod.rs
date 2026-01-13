// 缓存模块
// 提供数据和模型缓存功能

mod implementation;
pub mod policy;
// pub mod eviction;
// pub mod metrics;

// 重新导出缓存管理组件
pub use self::implementation::{
    Cache,
    CacheResult,
    CacheEntry,
    MemoryCache,
};

// 导出策略组件
pub use policy::{
    CachePolicy,
    CacheManager,
    LRUPolicy,
    TTLPolicy,
    DefaultPolicy,
};

// pub use eviction::{
//     EvictionStrategy,
//     LRUEviction,
//     LFUEviction,
//     TTLEviction,
// };
