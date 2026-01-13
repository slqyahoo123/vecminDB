// 复制模块
//
// 提供数据库节点间的复制和同步功能

pub mod state_sync;
pub mod sync_protocol;

// 重新导出常用组件
pub use state_sync::{
    StateSyncManager,
    StateProvider,
    StateApplier,
    SyncPriority,
    SyncOperationType,
    VerificationLevel,
    NodeRole,
    NodeStatus,
};

// 重新导出同步协议组件
pub use sync_protocol::{
    SyncProtocol,
    SyncSession,
    SyncChunk,
    SyncStats,
    DefaultStateProvider,
    DefaultStateApplier,
}; 