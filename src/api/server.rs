//! API服务器状态管理
//! 提供API服务器的状态结构和初始化逻辑

use std::sync::Arc;
use crate::vector::VectorService;
use crate::api::vector::VectorApi;
use crate::Result;

/// API服务器状态
/// 替代vecmind中的ProductionApiServer，专门为vecminDB设计
#[derive(Clone)]
pub struct ApiServerState {
    /// 向量API
    pub vector_api: Option<Arc<VectorApi>>,
    /// 向量服务（直接暴露给handlers使用）
    pub vector_service: Option<Arc<dyn VectorService>>,
    /// 向量数据库（用于集合管理）
    pub vector_db: Option<Arc<tokio::sync::RwLock<crate::vector::VectorDB>>>,
}

impl ApiServerState {
    /// 创建新的API服务器状态
    pub fn new() -> Self {
        Self {
            vector_api: None,
            vector_service: None,
            vector_db: None,
        }
    }
    
    /// 初始化向量API
    pub fn init_vector_api(&mut self, vector_service: Arc<dyn VectorService>) -> Result<()> {
        let vector_api = Arc::new(VectorApi::new(vector_service.clone()));
        self.vector_api = Some(vector_api);
        self.vector_service = Some(vector_service);
        Ok(())
    }
    
    /// 初始化向量数据库（用于集合管理）
    pub fn init_vector_db(&mut self, storage_path: &str) -> Result<()> {
        let db = crate::vector::VectorDB::new(storage_path)?;
        self.vector_db = Some(Arc::new(tokio::sync::RwLock::new(db)));
        Ok(())
    }
    
    /// 检查向量API是否已初始化
    pub fn is_vector_api_ready(&self) -> bool {
        self.vector_api.is_some()
    }
}

impl Default for ApiServerState {
    fn default() -> Self {
        Self::new()
    }
}

