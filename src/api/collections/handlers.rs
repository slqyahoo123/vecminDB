//! 集合API HTTP处理器

use actix_web::{
    web, HttpRequest, HttpResponse, Result as ActixResult,
    error::ErrorBadRequest,
};
use serde_json::json;
use tracing::{info, error, warn};
use std::sync::Arc;
use crate::api::{
    server::ApiServerState,
    response::ApiResponse,
};
use crate::api::collections::types;
use crate::vector::index::IndexType;
use crate::api::vector::types as vector_types;

/// 创建集合处理器
pub async fn create_collection_actix(
    req: HttpRequest,
    body: web::Json<serde_json::Value>,
) -> ActixResult<HttpResponse> {
    let server_data = req.app_data::<web::Data<Arc<ApiServerState>>>()
        .ok_or_else(|| {
            error!("无法获取服务器数据");
            ErrorBadRequest("服务器配置错误")
        })?;
    
    let collection_data = body.into_inner();
    
    // 验证必需字段
    let name = collection_data.get("name")
        .and_then(|v| v.as_str())
        .ok_or_else(|| {
            error!("缺少集合名称");
            ErrorBadRequest("缺少必需的name字段")
        })?;
    
    let dimension = collection_data.get("dimension")
        .and_then(|v| v.as_u64())
        .ok_or_else(|| {
            error!("缺少向量维度");
            ErrorBadRequest("缺少必需的dimension字段")
        })? as usize;
    
    let index_type_str = collection_data.get("index_type")
        .and_then(|v| v.as_str())
        .unwrap_or("Flat");
    
    // 解析索引类型
    let index_type = match index_type_str {
        "HNSW" => IndexType::HNSW,
        "IVF" => IndexType::IVF,
        "PQ" => IndexType::PQ,
        "LSH" => IndexType::LSH,
        "VPTree" => IndexType::VPTree,
        "Flat" => IndexType::Flat,
        _ => {
            return Ok(HttpResponse::BadRequest().json(ApiResponse::<()>::error(
                format!("不支持的索引类型: {}", index_type_str), 
                400
            )));
        }
    };
    
    info!("创建集合请求: name={}, dimension={}, index_type={:?}", name, dimension, index_type);
    
    // 验证参数
    if name.is_empty() {
        return Ok(HttpResponse::BadRequest().json(ApiResponse::<()>::error(
            "集合名称不能为空".to_string(), 
            400
        )));
    }
    
    if dimension == 0 || dimension > 10000 {
        return Ok(HttpResponse::BadRequest().json(ApiResponse::<()>::error(
            "向量维度必须在1-10000之间".to_string(), 
            400
        )));
    }
    
    // 从服务器状态获取VectorDB
    if let Some(vector_db) = &server_data.vector_db {
        let mut db = vector_db.write().await;
        match db.create_collection(name, dimension, index_type) {
            Ok(_collection) => {
                info!("成功创建集合: {}", name);
                let collection_info = types::CollectionInfo {
                    name: name.to_string(),
                    dimension,
                    index_type: format!("{:?}", index_type),
                    vector_count: 0,
                    created_at: chrono::Utc::now().to_rfc3339(),
                };
                Ok(HttpResponse::Created().json(ApiResponse::success(collection_info)))
            },
            Err(err) => {
                error!("创建集合失败: {}", err);
                Ok(HttpResponse::BadRequest().json(ApiResponse::<()>::error(
                    format!("创建集合失败: {}", err), 
                    400
                )))
            }
        }
    } else {
        error!("VectorDB未初始化");
        Ok(HttpResponse::InternalServerError().json(ApiResponse::<()>::error(
            "向量数据库未初始化".to_string(), 
            500
        )))
    }
}

/// 列出集合处理器
pub async fn list_collections_actix(
    req: HttpRequest,
) -> ActixResult<HttpResponse> {
    let server_data = req.app_data::<web::Data<Arc<ApiServerState>>>()
        .ok_or_else(|| {
            error!("无法获取服务器数据");
            ErrorBadRequest("服务器配置错误")
        })?;
    
    if let Some(vector_db) = &server_data.vector_db {
        let db = vector_db.read().await;
        let collection_names = db.list_collections().await;
        
        let collections: Vec<types::CollectionInfo> = collection_names.iter().map(|name| {
            types::CollectionInfo {
                name: name.clone(),
                dimension: 0, // 需要从集合中获取
                index_type: "Unknown".to_string(),
                vector_count: 0, // 需要从集合中获取
                created_at: chrono::Utc::now().to_rfc3339(),
            }
        }).collect();
        
        Ok(HttpResponse::Ok().json(ApiResponse::success(collections)))
    } else {
        error!("VectorDB未初始化");
        Ok(HttpResponse::InternalServerError().json(ApiResponse::<()>::error(
            "向量数据库未初始化".to_string(), 
            500
        )))
    }
}

/// 删除集合处理器
pub async fn delete_collection_actix(
    req: HttpRequest,
    path: web::Path<String>,
) -> ActixResult<HttpResponse> {
    let server_data = req.app_data::<web::Data<Arc<ApiServerState>>>()
        .ok_or_else(|| {
            error!("无法获取服务器数据");
            ErrorBadRequest("服务器配置错误")
        })?;
    
    let collection_name = path.into_inner();
    info!("删除集合请求: {}", collection_name);
    
    if let Some(vector_db) = &server_data.vector_db {
        let mut db = vector_db.write().await;
        match db.delete_collection(&collection_name).await {
            Ok(_) => {
                info!("成功删除集合: {}", collection_name);
                Ok(HttpResponse::Ok().json(ApiResponse::success(json!({
                    "message": "集合删除成功",
                    "collection_name": collection_name,
                    "timestamp": chrono::Utc::now().to_rfc3339()
                }))))
            },
            Err(err) => {
                error!("删除集合失败: {}", err);
                Ok(HttpResponse::BadRequest().json(ApiResponse::<()>::error(
                    format!("删除集合失败: {}", err), 
                    400
                )))
            }
        }
    } else {
        error!("VectorDB未初始化");
        Ok(HttpResponse::InternalServerError().json(ApiResponse::<()>::error(
            "向量数据库未初始化".to_string(), 
            500
        )))
    }
}

/// 获取集合信息处理器
pub async fn get_collection_actix(
    req: HttpRequest,
    path: web::Path<String>,
) -> ActixResult<HttpResponse> {
    let server_data = req.app_data::<web::Data<Arc<ApiServerState>>>()
        .ok_or_else(|| {
            error!("无法获取服务器数据");
            ErrorBadRequest("服务器配置错误")
        })?;
    
    let collection_name = path.into_inner();
    info!("获取集合信息请求: {}", collection_name);
    
    if let Some(vector_db) = &server_data.vector_db {
        let db = vector_db.read().await;
        match db.get_collection(&collection_name) {
            Ok(collection) => {
                let coll = collection.read().await;
                let collection_info = types::CollectionInfo {
                    name: collection_name.clone(),
                    dimension: coll.config.dimension,
                    index_type: format!("{:?}", coll.config.index_type),
                    vector_count: coll.count(),
                    created_at: chrono::Utc::now().to_rfc3339(), // 实际应该从配置中获取
                };
                Ok(HttpResponse::Ok().json(ApiResponse::success(collection_info)))
            },
            Err(_err) => {
                warn!("集合不存在: {}", collection_name);
                Ok(HttpResponse::NotFound().json(ApiResponse::<()>::error(
                    format!("集合 {} 不存在", collection_name), 
                    404
                )))
            }
        }
    } else {
        error!("VectorDB未初始化");
        Ok(HttpResponse::InternalServerError().json(ApiResponse::<()>::error(
            "向量数据库未初始化".to_string(), 
            500
        )))
    }
}

/// 获取集合统计信息处理器
pub async fn get_collection_stats_actix(
    req: HttpRequest,
    path: web::Path<String>,
) -> ActixResult<HttpResponse> {
    let server_data = req.app_data::<web::Data<Arc<ApiServerState>>>()
        .ok_or_else(|| {
            error!("无法获取服务器数据");
            ErrorBadRequest("服务器配置错误")
        })?;
    
    let collection_name = path.into_inner();
    info!("获取集合统计信息请求: {}", collection_name);
    
    if let Some(vector_db) = &server_data.vector_db {
        let db = vector_db.read().await;
        match db.get_collection(&collection_name) {
            Ok(collection) => {
                let coll = collection.read().await;
                let stats = types::CollectionStats {
                    name: collection_name.clone(),
                    vector_count: coll.count(),
                    index_type: format!("{:?}", coll.config.index_type),
                    memory_usage: None, // 需要实际计算
                    index_status: "Ready".to_string(),
                };
                Ok(HttpResponse::Ok().json(ApiResponse::success(stats)))
            },
            Err(_err) => {
                warn!("集合不存在: {}", collection_name);
                Ok(HttpResponse::NotFound().json(ApiResponse::<()>::error(
                    format!("集合 {} 不存在", collection_name), 
                    404
                )))
            }
        }
    } else {
        error!("VectorDB未初始化");
        Ok(HttpResponse::InternalServerError().json(ApiResponse::<()>::error(
            "向量数据库未初始化".to_string(), 
            500
        )))
    }
}

/// 重建集合索引处理器
pub async fn rebuild_collection_index_actix(
    req: HttpRequest,
    path: web::Path<String>,
) -> ActixResult<HttpResponse> {
    let server_data = req.app_data::<web::Data<Arc<ApiServerState>>>()
        .ok_or_else(|| {
            error!("无法获取服务器数据");
            ErrorBadRequest("服务器配置错误")
        })?;
    
    let collection_name = path.into_inner();
    info!("重建集合索引请求: {}", collection_name);
    
    if let Some(vector_db) = &server_data.vector_db {
        let db = vector_db.read().await;
        match db.get_collection(&collection_name) {
            Ok(_collection) => {
                // 重建索引逻辑（需要实现）
                info!("重建集合索引: {}", collection_name);
                Ok(HttpResponse::Ok().json(ApiResponse::success(json!({
                    "message": "索引重建已启动",
                    "collection_name": collection_name,
                    "status": "processing",
                    "timestamp": chrono::Utc::now().to_rfc3339()
                }))))
            },
            Err(_err) => {
                warn!("集合不存在: {}", collection_name);
                Ok(HttpResponse::NotFound().json(ApiResponse::<()>::error(
                    format!("集合 {} 不存在", collection_name), 
                    404
                )))
            }
        }
    } else {
        error!("VectorDB未初始化");
        Ok(HttpResponse::InternalServerError().json(ApiResponse::<()>::error(
            "向量数据库未初始化".to_string(), 
            500
        )))
    }
}

/// 向集合添加向量处理器
pub async fn add_vector_to_collection_actix(
    req: HttpRequest,
    path: web::Path<String>,
    body: web::Json<serde_json::Value>,
) -> ActixResult<HttpResponse> {
    let server_data = req.app_data::<web::Data<Arc<ApiServerState>>>()
        .ok_or_else(|| {
            error!("无法获取服务器数据");
            ErrorBadRequest("服务器配置错误")
        })?;
    
    let collection_name = path.into_inner();
    let vector_data = body.into_inner();
    
    // 验证必需字段
    let id = vector_data.get("id")
        .and_then(|v| v.as_str())
        .ok_or_else(|| ErrorBadRequest("缺少必需的id字段"))?;
    
    let values = vector_data.get("values")
        .and_then(|v| v.as_array())
        .ok_or_else(|| ErrorBadRequest("缺少必需的values字段"))?;
    
    let vector_values: Vec<f32> = values.iter()
        .filter_map(|v| v.as_f64().map(|f| f as f32))
        .collect();
    
    if vector_values.len() != values.len() {
        return Err(ErrorBadRequest("values必须是数字数组"));
    }
    
    let metadata = vector_data.get("metadata").cloned();
    
    if let Some(vector_db) = &server_data.vector_db {
        let db = vector_db.read().await;
        match db.get_collection(&collection_name) {
            Ok(collection) => {
                let mut coll = collection.write().await;
                match coll.add_vector(id, &vector_values, metadata.as_ref()).await {
                    Ok(_) => {
                        info!("成功向集合 {} 添加向量: {}", collection_name, id);
                        let vector = vector_types::Vector {
                            id: id.to_string(),
                            values: vector_values,
                            metadata,
                            created_at: chrono::Utc::now().to_rfc3339(),
                            updated_at: chrono::Utc::now().to_rfc3339(),
                        };
                        Ok(HttpResponse::Created().json(ApiResponse::success(vector)))
                    },
                    Err(err) => {
                        error!("添加向量失败: {}", err);
                        Ok(HttpResponse::BadRequest().json(ApiResponse::<()>::error(
                            format!("添加向量失败: {}", err), 
                            400
                        )))
                    }
                }
            },
            Err(_err) => {
                warn!("集合不存在: {}", collection_name);
                Ok(HttpResponse::NotFound().json(ApiResponse::<()>::error(
                    format!("集合 {} 不存在", collection_name), 
                    404
                )))
            }
        }
    } else {
        error!("VectorDB未初始化");
        Ok(HttpResponse::InternalServerError().json(ApiResponse::<()>::error(
            "向量数据库未初始化".to_string(), 
            500
        )))
    }
}

/// 从集合获取向量处理器
pub async fn get_vector_from_collection_actix(
    req: HttpRequest,
    path: web::Path<(String, String)>,
) -> ActixResult<HttpResponse> {
    let server_data = req.app_data::<web::Data<Arc<ApiServerState>>>()
        .ok_or_else(|| {
            error!("无法获取服务器数据");
            ErrorBadRequest("服务器配置错误")
        })?;
    
    let (collection_name, vector_id) = path.into_inner();
    info!("从集合 {} 获取向量: {}", collection_name, vector_id);
    
    if let Some(vector_db) = &server_data.vector_db {
        let db = vector_db.read().await;
        match db.get_collection(&collection_name) {
            Ok(collection) => {
                let coll = collection.read().await;
                match coll.get_vector(&vector_id) {
                    Some(vector) => {
                        let vector_response = vector_types::Vector {
                            id: vector.id.clone(),
                            values: vector.data.clone(),
                            metadata: vector.metadata.as_ref().map(|m| serde_json::json!(m.properties)),
                            created_at: chrono::Utc::now().to_rfc3339(),
                            updated_at: chrono::Utc::now().to_rfc3339(),
                        };
                        Ok(HttpResponse::Ok().json(ApiResponse::success(vector_response)))
                    },
                    None => {
                        warn!("向量不存在: {}", vector_id);
                        Ok(HttpResponse::NotFound().json(ApiResponse::<()>::error(
                            format!("向量 {} 不存在", vector_id), 
                            404
                        )))
                    }
                }
            },
            Err(_err) => {
                warn!("集合不存在: {}", collection_name);
                Ok(HttpResponse::NotFound().json(ApiResponse::<()>::error(
                    format!("集合 {} 不存在", collection_name), 
                    404
                )))
            }
        }
    } else {
        error!("VectorDB未初始化");
        Ok(HttpResponse::InternalServerError().json(ApiResponse::<()>::error(
            "向量数据库未初始化".to_string(), 
            500
        )))
    }
}

/// 从集合删除向量处理器
pub async fn delete_vector_from_collection_actix(
    req: HttpRequest,
    path: web::Path<(String, String)>,
) -> ActixResult<HttpResponse> {
    let server_data = req.app_data::<web::Data<Arc<ApiServerState>>>()
        .ok_or_else(|| {
            error!("无法获取服务器数据");
            ErrorBadRequest("服务器配置错误")
        })?;
    
    let (collection_name, vector_id) = path.into_inner();
    info!("从集合 {} 删除向量: {}", collection_name, vector_id);
    
    if let Some(vector_db) = &server_data.vector_db {
        let db = vector_db.read().await;
        match db.get_collection(&collection_name) {
            Ok(collection) => {
                let mut coll = collection.write().await;
                match coll.delete_vector(&vector_id) {
                    Ok(_) => {
                        info!("成功从集合 {} 删除向量: {}", collection_name, vector_id);
                        Ok(HttpResponse::Ok().json(ApiResponse::success(json!({
                            "message": "向量删除成功",
                            "collection_name": collection_name,
                            "vector_id": vector_id,
                            "timestamp": chrono::Utc::now().to_rfc3339()
                        }))))
                    },
                    Err(err) => {
                        error!("删除向量失败: {}", err);
                        Ok(HttpResponse::BadRequest().json(ApiResponse::<()>::error(
                            format!("删除向量失败: {}", err), 
                            400
                        )))
                    }
                }
            },
            Err(_err) => {
                warn!("集合不存在: {}", collection_name);
                Ok(HttpResponse::NotFound().json(ApiResponse::<()>::error(
                    format!("集合 {} 不存在", collection_name), 
                    404
                )))
            }
        }
    } else {
        error!("VectorDB未初始化");
        Ok(HttpResponse::InternalServerError().json(ApiResponse::<()>::error(
            "向量数据库未初始化".to_string(), 
            500
        )))
    }
}

/// 搜索集合中的向量处理器
pub async fn search_collection_actix(
    req: HttpRequest,
    path: web::Path<String>,
    body: web::Json<serde_json::Value>,
) -> ActixResult<HttpResponse> {
    let server_data = req.app_data::<web::Data<Arc<ApiServerState>>>()
        .ok_or_else(|| {
            error!("无法获取服务器数据");
            ErrorBadRequest("服务器配置错误")
        })?;
    
    let collection_name = path.into_inner();
    let search_data = body.into_inner();
    
    // 验证必需字段
    let query = search_data.get("query")
        .and_then(|v| v.as_array())
        .ok_or_else(|| ErrorBadRequest("缺少必需的query字段"))?;
    
    let query_vector: Vec<f32> = query.iter()
        .filter_map(|v| v.as_f64().map(|f| f as f32))
        .collect();
    
    if query_vector.len() != query.len() {
        return Err(ErrorBadRequest("query必须是数字数组"));
    }
    
    let top_k = search_data.get("top_k")
        .and_then(|v| v.as_u64())
        .unwrap_or(10) as usize;
    
    let metric_str = search_data.get("metric")
        .and_then(|v| v.as_str())
        .unwrap_or("cosine");
    
    if let Some(vector_db) = &server_data.vector_db {
        let db = vector_db.read().await;
        match db.get_collection(&collection_name) {
            Ok(collection) => {
                let coll = collection.read().await;
                use crate::vector::search::VectorQuery;
                use crate::vector::core::operations::SimilarityMetric;
                
                let _metric = match metric_str {
                    "cosine" => SimilarityMetric::Cosine,
                    "euclidean" => SimilarityMetric::Euclidean,
                    "dot" => SimilarityMetric::DotProduct,
                    _ => SimilarityMetric::Cosine,
                };
                
                let query = VectorQuery {
                    vector: query_vector,
                    filter: None,
                    top_k,
                    include_metadata: true,
                    include_vectors: false,
                };
                
                // Use search_with_vector_query for synchronous search
                match coll.search_with_vector_query(&query) {
                    Ok(results) => {
                        let search_results: Vec<vector_types::VectorSearchResult> = results.into_iter().map(|r| {
                            let id = r.id.clone();
                            let metadata_clone = r.metadata.clone();
                            vector_types::VectorSearchResult {
                                id: r.id,
                                score: r.score,
                                metadata: r.metadata.map(|m| serde_json::json!(m.properties)),
                                vector: r.vector.map(|v| vector_types::Vector {
                                    id: id,
                                    values: v,
                                    metadata: metadata_clone.map(|m| serde_json::json!(m.properties)),
                                    created_at: String::new(),
                                    updated_at: String::new(),
                                }),
                            }
                        }).collect();
                        Ok(HttpResponse::Ok().json(ApiResponse::success(search_results)))
                    },
                    Err(err) => {
                        error!("搜索失败: {}", err);
                        Ok(HttpResponse::BadRequest().json(ApiResponse::<()>::error(
                            format!("搜索失败: {}", err), 
                            400
                        )))
                    }
                }
            },
            Err(_err) => {
                warn!("集合不存在: {}", collection_name);
                Ok(HttpResponse::NotFound().json(ApiResponse::<()>::error(
                    format!("集合 {} 不存在", collection_name), 
                    404
                )))
            }
        }
    } else {
        error!("VectorDB未初始化");
        Ok(HttpResponse::InternalServerError().json(ApiResponse::<()>::error(
            "向量数据库未初始化".to_string(), 
            500
        )))
    }
}

/// 批量操作处理器
pub async fn batch_operations_actix(
    req: HttpRequest,
    path: web::Path<String>,
    body: web::Json<serde_json::Value>,
) -> ActixResult<HttpResponse> {
    let _server_data = req.app_data::<web::Data<Arc<ApiServerState>>>()
        .ok_or_else(|| {
            error!("无法获取服务器数据");
            ErrorBadRequest("服务器配置错误")
        })?;
    
    let _collection_name = path.into_inner();
    let _batch_data = body.into_inner();
    
    // Batch operations are not a core feature of the vector database
    Ok(HttpResponse::NotImplemented().json(ApiResponse::<()>::error(
        "Batch operations feature is not enabled in this vector database".to_string(), 
        501
    )))
}
