//! 集合API模块
//! 提供集合管理的HTTP API接口

pub mod types;
pub mod handlers;

// 重新导出处理器函数
pub use handlers::{
    create_collection_actix,
    list_collections_actix,
    delete_collection_actix,
    get_collection_actix,
    get_collection_stats_actix,
    rebuild_collection_index_actix,
    add_vector_to_collection_actix,
    get_vector_from_collection_actix,
    delete_vector_from_collection_actix,
    search_collection_actix,
    batch_operations_actix,
};

// 重新导出类型
pub use types::*;

