//! API request handlers

use actix_web::web;
use std::sync::Arc;
use crate::api::server::ApiServerState;

/// 配置所有API路由
pub fn configure_routes(cfg: &mut web::ServiceConfig) {
    cfg.service(
        web::scope("/api/v1")
            .route("/health", web::get().to(health_check))
            // 向量API路由
            .service(
                web::scope("/vectors")
                    .route("", web::post().to(crate::api::vector::create_vector_actix))
                    .route("/batch", web::post().to(crate::api::vector::batch_create_vectors_actix))
                    .route("/{id}", web::get().to(crate::api::vector::get_vector_actix))
                    .route("/{id}", web::delete().to(crate::api::vector::delete_vector_actix))
                    .route("/batch/delete", web::post().to(crate::api::vector::batch_delete_vectors_actix))
                    .route("/search", web::post().to(crate::api::vector::search_vectors_actix))
                    .route("/count", web::get().to(crate::api::vector::count_vectors_actix))
                    .route("/rebuild-index", web::post().to(crate::api::vector::rebuild_index_actix))
            )
            // 集合管理API路由
            .service(
                web::scope("/collections")
                    .route("", web::post().to(crate::api::collections::create_collection_actix))
                    .route("", web::get().to(crate::api::collections::list_collections_actix))
                    .route("/{name}", web::get().to(crate::api::collections::get_collection_actix))
                    .route("/{name}", web::delete().to(crate::api::collections::delete_collection_actix))
                    .route("/{name}/stats", web::get().to(crate::api::collections::get_collection_stats_actix))
                    .route("/{name}/rebuild", web::post().to(crate::api::collections::rebuild_collection_index_actix))
                    // 集合级别的向量操作
                    .route("/{name}/vectors", web::post().to(crate::api::collections::add_vector_to_collection_actix))
                    .route("/{name}/vectors/{id}", web::get().to(crate::api::collections::get_vector_from_collection_actix))
                    .route("/{name}/vectors/{id}", web::delete().to(crate::api::collections::delete_vector_from_collection_actix))
                    .route("/{name}/search", web::post().to(crate::api::collections::search_collection_actix))
                    .route("/{name}/batch", web::post().to(crate::api::collections::batch_operations_actix))
            )
    );
}

/// 健康检查处理器
async fn health_check() -> actix_web::HttpResponse {
    actix_web::HttpResponse::Ok().json(serde_json::json!({
        "status": "ok",
        "version": env!("CARGO_PKG_VERSION")
    }))
}

/// 初始化API服务器状态
/// 这个函数应该在main.rs中调用，用于设置向量服务
pub fn init_api_server_state(
    vector_service: Arc<dyn crate::vector::VectorService>,
) -> crate::Result<Arc<ApiServerState>> {
    let mut state = ApiServerState::new();
    state.init_vector_api(vector_service)?;
    Ok(Arc::new(state))
}

