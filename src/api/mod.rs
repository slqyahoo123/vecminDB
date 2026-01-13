//! HTTP API module
//!
//! This module provides REST API endpoints for the vector database.

// API routes模块（包含algorithm等子模块）
pub mod routes;

#[cfg(feature = "http-server")]
pub mod handlers;

#[cfg(feature = "http-server")]
pub mod middleware;

// 向量API模块
#[cfg(feature = "http-server")]
pub mod vector;

// 集合API模块
#[cfg(feature = "http-server")]
pub mod collections;

// API响应类型
#[cfg(feature = "http-server")]
pub mod response;

// API服务器状态
#[cfg(feature = "http-server")]
pub mod server;

// 重新导出routes中的配置函数（如果http-server特性启用）
#[cfg(feature = "http-server")]
pub use crate::api::handlers::configure_routes;

// 重新导出常用类型
#[cfg(feature = "http-server")]
pub use response::ApiResponse;

