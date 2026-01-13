//! 向量API HTTP处理器

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
use crate::api::vector::types;

/// 创建向量处理器
pub async fn create_vector_actix(
    req: HttpRequest,
    body: web::Json<serde_json::Value>,
) -> ActixResult<HttpResponse> {
    let server_data = req.app_data::<web::Data<Arc<ApiServerState>>>()
        .ok_or_else(|| {
            error!("无法获取服务器数据");
            ErrorBadRequest("服务器配置错误")
        })?;
    
    let vector_data = body.into_inner();
    
    // 验证必需字段
    let vector_id = vector_data.get("id")
        .and_then(|v| v.as_str())
        .ok_or_else(|| {
            error!("缺少向量ID");
            ErrorBadRequest("缺少必需的id字段")
        })?;
    
    let values = vector_data.get("values")
        .and_then(|v| v.as_array())
        .ok_or_else(|| {
            error!("缺少向量值");
            ErrorBadRequest("缺少必需的values字段")
        })?;
    
    info!("创建向量请求: id={}, dimension={}", vector_id, values.len());
    
    // 验证向量ID格式
    if vector_id.is_empty() || !vector_id.chars().all(|c| c.is_alphanumeric() || c == '-' || c == '_') {
        return Ok(HttpResponse::BadRequest().json(ApiResponse::<()>::error(
            "无效的向量ID格式".to_string(), 
            400
        )));
    }
    
    // 验证向量值
    if values.is_empty() || values.len() > 10000 {
        return Ok(HttpResponse::BadRequest().json(ApiResponse::<()>::error(
            "向量维度必须在1-10000之间".to_string(), 
            400
        )));
    }
    
    // 转换向量值
    let mut vector_values = Vec::new();
    for value in values {
        if let Some(f64_val) = value.as_f64() {
            vector_values.push(f64_val as f32);
        } else {
            return Ok(HttpResponse::BadRequest().json(ApiResponse::<()>::error(
                "向量值必须是数字".to_string(), 
                400
            )));
        }
    }
    
    // 构建创建向量请求
    let create_request = types::CreateVectorRequest {
        id: vector_id.to_string(),
        values: vector_values,
        metadata: vector_data.get("metadata").cloned(),
    };
    
    // 调用向量API创建向量
    if let Some(vector_api) = &server_data.vector_api {
        match vector_api.create_vector(create_request).await {
            Ok(vector) => {
                info!("成功创建向量: {}", vector.id);
                Ok(HttpResponse::Created().json(ApiResponse::success(vector)))
            },
            Err(err) => {
                error!("创建向量失败: {}", err);
                Ok(HttpResponse::BadRequest().json(ApiResponse::<()>::error(
                    format!("创建向量失败: {}", err), 
                    400
                )))
            }
        }
    } else {
        error!("向量API未初始化");
        Ok(HttpResponse::InternalServerError().json(ApiResponse::<()>::error(
            "向量服务未初始化".to_string(), 
            500
        )))
    }
}

/// 批量创建向量处理器
pub async fn batch_create_vectors_actix(
    req: HttpRequest,
    body: web::Json<serde_json::Value>,
) -> ActixResult<HttpResponse> {
    let server_data = req.app_data::<web::Data<Arc<ApiServerState>>>()
        .ok_or_else(|| {
            error!("无法获取服务器数据");
            ErrorBadRequest("服务器配置错误")
        })?;
    
    let batch_data = body.into_inner();
    
    // 验证必需字段
    let vectors = batch_data.get("vectors")
        .and_then(|v| v.as_array())
        .ok_or_else(|| {
            error!("缺少向量数组");
            ErrorBadRequest("缺少必需的vectors字段")
        })?;
    
    info!("批量创建向量请求: count={}", vectors.len());
    
    // 验证批量大小
    if vectors.is_empty() || vectors.len() > 1000 {
        return Ok(HttpResponse::BadRequest().json(ApiResponse::<()>::error(
            "批量向量数量必须在1-1000之间".to_string(), 
            400
        )));
    }
    
    // 转换向量数据
    let mut create_requests = Vec::new();
    for (index, vector_obj) in vectors.iter().enumerate() {
        if let Some(obj) = vector_obj.as_object() {
            let vector_id: String = obj.get("id")
                .and_then(|v| v.as_str())
                .map(|s| s.to_string())
                .unwrap_or_else(|| format!("vector_{}", index));
            
            let values = obj.get("values")
                .and_then(|v| v.as_array())
                .ok_or_else(|| {
                    error!("向量 {} 缺少values字段", vector_id);
                    ErrorBadRequest(format!("向量 {} 缺少values字段", vector_id))
                })?;
            
            // 验证向量值
            if values.is_empty() || values.len() > 10000 {
                return Ok(HttpResponse::BadRequest().json(ApiResponse::<()>::error(
                    format!("向量 {} 的维度必须在1-10000之间", vector_id), 
                    400
                )));
            }
            
            // 转换向量值
            let mut vector_values = Vec::new();
            for value in values {
                if let Some(f64_val) = value.as_f64() {
                    vector_values.push(f64_val as f32);
                } else {
                    return Ok(HttpResponse::BadRequest().json(ApiResponse::<()>::error(
                        format!("向量 {} 的值必须是数字", vector_id), 
                        400
                    )));
                }
            }
            
            let create_request = types::CreateVectorRequest {
                id: vector_id,
                values: vector_values,
                metadata: obj.get("metadata").cloned(),
            };
            
            create_requests.push(create_request);
        } else {
            return Ok(HttpResponse::BadRequest().json(ApiResponse::<()>::error(
                format!("第 {} 个向量数据格式错误", index), 
                400
            )));
        }
    }
    
    // 调用向量API批量创建向量
    if let Some(vector_api) = &server_data.vector_api {
        let batch_request = types::BatchCreateVectorsRequest {
            vectors: create_requests,
        };
        match vector_api.batch_create_vectors(batch_request).await {
            Ok(vectors) => {
                info!("成功批量创建向量: 共{}个向量", vectors.len());
                Ok(HttpResponse::Created().json(ApiResponse::success(vectors)))
            },
            Err(err) => {
                error!("批量创建向量失败: {}", err);
                Ok(HttpResponse::BadRequest().json(ApiResponse::<()>::error(
                    format!("批量创建向量失败: {}", err), 
                    400
                )))
            }
        }
    } else {
        error!("向量API未初始化");
        Ok(HttpResponse::InternalServerError().json(ApiResponse::<()>::error(
            "向量服务未初始化".to_string(), 
            500
        )))
    }
}

/// 获取向量处理器
pub async fn get_vector_actix(
    req: HttpRequest,
    path: web::Path<String>,
) -> ActixResult<HttpResponse> {
    let server_data = req.app_data::<web::Data<Arc<ApiServerState>>>()
        .ok_or_else(|| {
            error!("无法获取服务器数据");
            ErrorBadRequest("服务器配置错误")
        })?;
    
    let vector_id = path.into_inner();
    
    // 验证向量ID格式
    if vector_id.is_empty() || !vector_id.chars().all(|c| c.is_alphanumeric() || c == '-' || c == '_') {
        return Ok(HttpResponse::BadRequest().json(ApiResponse::<()>::error(
            "无效的向量ID格式".to_string(), 
            400
        )));
    }
    
    info!("获取向量请求: {}", vector_id);
    
    // 调用向量API获取向量
    if let Some(vector_api) = &server_data.vector_api {
        match vector_api.get_vector(&vector_id).await {
            Ok(Some(vector)) => {
                info!("成功获取向量: {}", vector_id);
                Ok(HttpResponse::Ok().json(ApiResponse::success(vector)))
            },
            Ok(None) => {
                warn!("向量不存在: {}", vector_id);
                Ok(HttpResponse::NotFound().json(ApiResponse::<()>::error(
                    format!("向量 {} 不存在", vector_id), 
                    404
                )))
            },
            Err(err) => {
                error!("获取向量失败: {}", err);
                Ok(HttpResponse::InternalServerError().json(ApiResponse::<()>::error(
                    format!("获取向量失败: {}", err), 
                    500
                )))
            }
        }
    } else {
        error!("向量API未初始化");
        Ok(HttpResponse::InternalServerError().json(ApiResponse::<()>::error(
            "向量服务未初始化".to_string(), 
            500
        )))
    }
}

/// 删除向量处理器
pub async fn delete_vector_actix(
    req: HttpRequest,
    path: web::Path<String>,
) -> ActixResult<HttpResponse> {
    let server_data = req.app_data::<web::Data<Arc<ApiServerState>>>()
        .ok_or_else(|| {
            error!("无法获取服务器数据");
            ErrorBadRequest("服务器配置错误")
        })?;
    
    let vector_id = path.into_inner();
    
    // 验证向量ID格式
    if vector_id.is_empty() || !vector_id.chars().all(|c| c.is_alphanumeric() || c == '-' || c == '_') {
        return Ok(HttpResponse::BadRequest().json(ApiResponse::<()>::error(
            "无效的向量ID格式".to_string(), 
            400
        )));
    }
    
    info!("删除向量请求: {}", vector_id);
    
    // 调用向量API删除向量
    if let Some(vector_api) = &server_data.vector_api {
        match vector_api.delete_vector(&vector_id).await {
            Ok(_) => {
                info!("成功删除向量: {}", vector_id);
                Ok(HttpResponse::Ok().json(ApiResponse::success(json!({
                    "message": "向量删除成功",
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
    } else {
        error!("向量API未初始化");
        Ok(HttpResponse::InternalServerError().json(ApiResponse::<()>::error(
            "向量服务未初始化".to_string(), 
            500
        )))
    }
}

/// 批量删除向量处理器
pub async fn batch_delete_vectors_actix(
    req: HttpRequest,
    body: web::Json<serde_json::Value>,
) -> ActixResult<HttpResponse> {
    let server_data = req.app_data::<web::Data<Arc<ApiServerState>>>()
        .ok_or_else(|| {
            error!("无法获取服务器数据");
            ErrorBadRequest("服务器配置错误")
        })?;
    
    let delete_data = body.into_inner();
    
    // 验证必需字段
    let vector_ids = delete_data.get("vector_ids")
        .and_then(|v| v.as_array())
        .ok_or_else(|| {
            error!("缺少向量ID数组");
            ErrorBadRequest("缺少必需的vector_ids字段")
        })?;
    
    info!("批量删除向量请求: count={}", vector_ids.len());
    
    // 验证批量大小
    if vector_ids.is_empty() || vector_ids.len() > 1000 {
        return Ok(HttpResponse::BadRequest().json(ApiResponse::<()>::error(
            "批量删除向量数量必须在1-1000之间".to_string(), 
            400
        )));
    }
    
    // 转换向量ID
    let mut ids = Vec::new();
    for (index, id_value) in vector_ids.iter().enumerate() {
        if let Some(id_str) = id_value.as_str() {
            if id_str.is_empty() || !id_str.chars().all(|c| c.is_alphanumeric() || c == '-' || c == '_') {
                return Ok(HttpResponse::BadRequest().json(ApiResponse::<()>::error(
                    format!("第 {} 个向量ID格式无效", index), 
                    400
                )));
            }
            ids.push(id_str.to_string());
        } else {
            return Ok(HttpResponse::BadRequest().json(ApiResponse::<()>::error(
                format!("第 {} 个向量ID必须是字符串", index), 
                400
            )));
        }
    }
    
    // 调用向量API批量删除向量
    if let Some(vector_api) = &server_data.vector_api {
        let batch_request = types::BatchDeleteVectorsRequest {
            ids: ids.clone(),
        };
        match vector_api.batch_delete_vectors(batch_request).await {
            Ok(deleted_count) => {
                info!("成功批量删除向量: 共{}个向量", deleted_count);
                Ok(HttpResponse::Ok().json(ApiResponse::success(deleted_count)))
            },
            Err(err) => {
                error!("批量删除向量失败: {}", err);
                Ok(HttpResponse::BadRequest().json(ApiResponse::<()>::error(
                    format!("批量删除向量失败: {}", err), 
                    400
                )))
            }
        }
    } else {
        error!("向量API未初始化");
        Ok(HttpResponse::InternalServerError().json(ApiResponse::<()>::error(
            "向量服务未初始化".to_string(), 
            500
        )))
    }
}

/// 搜索向量处理器
pub async fn search_vectors_actix(
    req: HttpRequest,
    body: web::Json<serde_json::Value>,
) -> ActixResult<HttpResponse> {
    let server_data = req.app_data::<web::Data<Arc<ApiServerState>>>()
        .ok_or_else(|| {
            error!("无法获取服务器数据");
            ErrorBadRequest("服务器配置错误")
        })?;
    
    let search_data = body.into_inner();
    
    // 验证必需字段
    let query_vector = search_data.get("query_vector")
        .and_then(|v| v.as_array())
        .ok_or_else(|| {
            error!("缺少查询向量");
            ErrorBadRequest("缺少必需的query_vector字段")
        })?;
    
    let top_k = search_data.get("top_k")
        .and_then(|v| v.as_u64())
        .unwrap_or(10) as usize;
    
    info!("搜索向量请求: dimension={}, top_k={}", query_vector.len(), top_k);
    
    // 验证查询向量
    if query_vector.is_empty() || query_vector.len() > 10000 {
        return Ok(HttpResponse::BadRequest().json(ApiResponse::<()>::error(
            "查询向量维度必须在1-10000之间".to_string(), 
            400
        )));
    }
    
    // 验证top_k参数
    if top_k == 0 || top_k > 1000 {
        return Ok(HttpResponse::BadRequest().json(ApiResponse::<()>::error(
            "top_k参数必须在1-1000之间".to_string(), 
            400
        )));
    }
    
    // 转换查询向量
    let mut query_values = Vec::new();
    for value in query_vector {
        if let Some(f64_val) = value.as_f64() {
            query_values.push(f64_val as f32);
        } else {
            return Ok(HttpResponse::BadRequest().json(ApiResponse::<()>::error(
                "查询向量值必须是数字".to_string(), 
                400
            )));
        }
    }
    
    // 构建搜索请求
    let search_request = types::SearchVectorsRequest {
        query_vector: query_values,
        top_k,
        metric: search_data.get("metric")
            .and_then(|v| v.as_str())
            .unwrap_or("cosine")
            .to_string(),
        filter: search_data.get("filter").cloned(),
    };
    
    // 调用向量API搜索向量
    if let Some(vector_api) = &server_data.vector_api {
        match vector_api.search_vectors(search_request).await {
            Ok(results) => {
                info!("成功搜索向量: 返回{}个结果", results.len());
                Ok(HttpResponse::Ok().json(ApiResponse::success(results)))
            },
            Err(err) => {
                error!("搜索向量失败: {}", err);
                Ok(HttpResponse::BadRequest().json(ApiResponse::<()>::error(
                    format!("搜索向量失败: {}", err), 
                    400
                )))
            }
        }
    } else {
        error!("向量API未初始化");
        Ok(HttpResponse::InternalServerError().json(ApiResponse::<()>::error(
            "向量服务未初始化".to_string(), 
            500
        )))
    }
}

/// 统计向量数量处理器
pub async fn count_vectors_actix(
    req: HttpRequest,
) -> ActixResult<HttpResponse> {
    let server_data = req.app_data::<web::Data<Arc<ApiServerState>>>()
        .ok_or_else(|| {
            error!("无法获取服务器数据");
            ErrorBadRequest("服务器配置错误")
        })?;
    
    if let Some(vector_api) = &server_data.vector_api {
        match vector_api.count_vectors().await {
            Ok(count) => {
                Ok(HttpResponse::Ok().json(ApiResponse::success(json!({
                    "count": count
                }))))
            },
            Err(err) => {
                error!("获取向量统计失败: {}", err);
                Ok(HttpResponse::InternalServerError().json(ApiResponse::<()>::error(
                    format!("获取向量统计失败: {}", err), 
                    500
                )))
            }
        }
    } else {
        error!("向量API未初始化");
        Ok(HttpResponse::InternalServerError().json(ApiResponse::<()>::error(
            "向量服务未初始化".to_string(), 
            500
        )))
    }
}

/// 重建索引处理器
pub async fn rebuild_index_actix(
    req: HttpRequest,
) -> ActixResult<HttpResponse> {
    let server_data = req.app_data::<web::Data<Arc<ApiServerState>>>()
        .ok_or_else(|| {
            error!("无法获取服务器数据");
            ErrorBadRequest("服务器配置错误")
        })?;
    
    if let Some(vector_api) = &server_data.vector_api {
        match vector_api.rebuild_index().await {
            Ok(_) => {
                info!("索引重建成功");
                Ok(HttpResponse::Ok().json(ApiResponse::success(json!({
                    "message": "索引重建成功",
                    "status": "completed",
                    "timestamp": chrono::Utc::now().to_rfc3339()
                }))))
            },
            Err(err) => {
                error!("索引重建失败: {}", err);
                Ok(HttpResponse::InternalServerError().json(ApiResponse::<()>::error(
                    format!("索引重建失败: {}", err), 
                    500
                )))
            }
        }
    } else {
        error!("向量API未初始化");
        Ok(HttpResponse::InternalServerError().json(ApiResponse::<()>::error(
            "向量服务未初始化".to_string(), 
            500
        )))
    }
}

