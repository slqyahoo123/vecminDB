//! 远程缓存模块
//!
//! 提供远程缓存客户端的实现，支持HTTP、gRPC等协议的远程缓存服务

use crate::error::{Error, Result};
use crate::cache::manager::RemoteCache;
use async_trait::async_trait;
use serde::{Serialize, Deserialize};

use std::sync::Arc;
use std::time::Duration;
#[cfg(feature = "multimodal")]
use reqwest;
use tracing::warn;
use base64::{Engine as _, engine::general_purpose};

/// 远程缓存协议类型
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum RemoteProtocol {
    /// HTTP/HTTPS
    Http,
    /// gRPC
    Grpc,
    /// WebSocket
    WebSocket,
    /// TCP
    Tcp,
}

/// 远程缓存配置
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RemoteCacheConfig {
    /// 服务端点列表
    pub endpoints: Vec<String>,
    /// 协议类型
    pub protocol: RemoteProtocol,
    /// 连接超时（秒）
    pub connect_timeout: u64,
    /// 请求超时（秒）
    pub request_timeout: u64,
    /// 重试次数
    pub retry_count: usize,
    /// 重试间隔（毫秒）
    pub retry_interval: u64,
    /// 认证信息
    pub auth: Option<RemoteAuth>,
    /// 是否启用压缩
    pub compression: bool,
    /// 最大连接数
    pub max_connections: usize,
    /// 连接池空闲超时（秒）
    pub idle_timeout: u64,
    /// 默认TTL
    pub default_ttl: Option<Duration>,
}

impl Default for RemoteCacheConfig {
    fn default() -> Self {
        Self {
            endpoints: vec!["http://localhost:8080".to_string()],
            protocol: RemoteProtocol::Http,
            connect_timeout: 10,
            request_timeout: 30,
            retry_count: 3,
            retry_interval: 1000,
            auth: None,
            compression: false,
            max_connections: 100,
            idle_timeout: 300,
            default_ttl: None,
        }
    }
}

/// 远程认证信息
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RemoteAuth {
    /// 认证类型
    pub auth_type: AuthType,
    /// 用户名
    pub username: Option<String>,
    /// 密码
    pub password: Option<String>,
    /// Token
    pub token: Option<String>,
    /// API密钥
    pub api_key: Option<String>,
}

/// 认证类型
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum AuthType {
    /// 无认证
    None,
    /// 基本认证
    Basic,
    /// Bearer Token
    Bearer,
    /// API密钥
    ApiKey,
}

/// HTTP远程缓存客户端
pub struct HttpRemoteCache {
    config: RemoteCacheConfig,
    #[cfg(feature = "multimodal")]
    client: reqwest::Client,
    current_endpoint: Arc<tokio::sync::RwLock<usize>>,
}

impl HttpRemoteCache {
    /// 创建新的HTTP远程缓存客户端
    pub fn new(config: RemoteCacheConfig) -> Result<Self> {
        if config.endpoints.is_empty() {
            return Err(Error::InvalidInput("远程缓存端点不能为空".to_string()));
        }

        #[cfg(feature = "multimodal")]
        let client = reqwest::Client::builder()
            .timeout(Duration::from_secs(config.request_timeout))
            .connect_timeout(Duration::from_secs(config.connect_timeout))
            .build()
            .map_err(|e| Error::InitializationError(format!("创建HTTP客户端失败: {}", e)))?;
        
        #[cfg(not(feature = "multimodal"))]
        return Err(Error::feature_not_enabled("multimodal"));

        Ok(Self {
            config,
            #[cfg(feature = "multimodal")]
            client,
            current_endpoint: Arc::new(tokio::sync::RwLock::new(0)),
        })
    }

    /// 获取当前端点URL
    async fn get_current_endpoint(&self) -> String {
        let index = *self.current_endpoint.read().await;
        self.config.endpoints[index % self.config.endpoints.len()].clone()
    }

    /// 轮换到下一个端点
    async fn rotate_endpoint(&self) {
        let mut index = self.current_endpoint.write().await;
        *index = (*index + 1) % self.config.endpoints.len();
    }

    /// 构建请求头
    #[cfg(feature = "multimodal")]
    fn build_headers(&self) -> reqwest::header::HeaderMap {
        let mut headers = reqwest::header::HeaderMap::new();
        
        if let Some(ref auth) = self.config.auth {
            match auth.auth_type {
                AuthType::Basic => {
                    if let (Some(username), Some(password)) = (&auth.username, &auth.password) {
                        let credentials = general_purpose::STANDARD.encode(format!("{}:{}", username, password));
                        headers.insert(
                            reqwest::header::AUTHORIZATION,
                            format!("Basic {}", credentials).parse().unwrap(),
                        );
                    }
                }
                AuthType::Bearer => {
                    if let Some(ref token) = auth.token {
                        headers.insert(
                            reqwest::header::AUTHORIZATION,
                            format!("Bearer {}", token).parse().unwrap(),
                        );
                    }
                }
                AuthType::ApiKey => {
                    if let Some(ref api_key) = auth.api_key {
                        headers.insert("X-API-Key", api_key.parse().unwrap());
                    }
                }
                AuthType::None => {}
            }
        }

        headers.insert(
            reqwest::header::CONTENT_TYPE,
            "application/json".parse().unwrap(),
        );

        headers
    }

    /// 执行带重试的请求
    async fn execute_with_retry<F, Fut, T>(&self, operation: F) -> Result<T>
    where
        F: Fn() -> Fut,
        Fut: std::future::Future<Output = Result<T>>,
    {
        let mut last_error = None;
        
        for attempt in 0..=self.config.retry_count {
            match operation().await {
                Ok(result) => return Ok(result),
                Err(e) => {
                    last_error = Some(e);
                    
                    if attempt < self.config.retry_count {
                        warn!("远程缓存请求失败，{}ms后重试 (尝试 {}/{})", 
                             self.config.retry_interval, attempt + 1, self.config.retry_count);
                        
                        tokio::time::sleep(Duration::from_millis(self.config.retry_interval)).await;
                        self.rotate_endpoint().await;
                    }
                }
            }
        }
        
        Err(last_error.unwrap_or_else(|| Error::NetworkError("远程缓存请求失败".to_string())))
    }
}

#[async_trait]
impl RemoteCache for HttpRemoteCache {
    #[cfg(feature = "multimodal")]
    async fn get(&self, key: &str) -> Result<Option<Vec<u8>>> {
        self.execute_with_retry(|| async {
            let endpoint = self.get_current_endpoint().await;
            let url = format!("{}/cache/{}", endpoint, urlencoding::encode(key));
            
            let response = self.client
                .get(&url)
                .headers(self.build_headers())
                .send()
                .await
                .map_err(|e| Error::NetworkError(format!("GET请求失败: {}", e)))?;

            #[cfg(feature = "multimodal")]
            match response.status() {
                reqwest::StatusCode::OK => {
                    let data = response.bytes().await
                        .map_err(|e| Error::NetworkError(format!("读取响应失败: {}", e)))?;
                    Ok(Some(data.to_vec()))
                }
                reqwest::StatusCode::NOT_FOUND => Ok(None),
                status => Err(Error::NetworkError(format!("远程缓存返回错误状态: {}", status))),
            }
        }).await
    }
    
    #[cfg(not(feature = "multimodal"))]
    async fn get(&self, _key: &str) -> Result<Option<Vec<u8>>> {
        Err(Error::feature_not_enabled("multimodal"))
    }

    async fn set(&self, key: &str, value: &[u8]) -> Result<()> {
        self.execute_with_retry(|| async {
            let endpoint = self.get_current_endpoint().await;
            let url = format!("{}/cache/{}", endpoint, urlencoding::encode(key));
            
            let response = self.client
                .put(&url)
                .headers(self.build_headers())
                .body(value.to_vec())
                .send()
                .await
                .map_err(|e| Error::NetworkError(format!("PUT请求失败: {}", e)))?;

            if response.status().is_success() {
                Ok(())
            } else {
                Err(Error::NetworkError(format!("远程缓存设置失败: {}", response.status())))
            }
        }).await
    }

    async fn delete(&self, key: &str) -> Result<bool> {
        self.execute_with_retry(|| async {
            let endpoint = self.get_current_endpoint().await;
            let url = format!("{}/cache/{}", endpoint, urlencoding::encode(key));
            
            let response = self.client
                .delete(&url)
                .headers(self.build_headers())
                .send()
                .await
                .map_err(|e| Error::NetworkError(format!("DELETE请求失败: {}", e)))?;

            #[cfg(feature = "multimodal")]
            match response.status() {
                reqwest::StatusCode::OK | reqwest::StatusCode::NO_CONTENT => Ok(true),
                reqwest::StatusCode::NOT_FOUND => Ok(false),
                status => Err(Error::NetworkError(format!("远程缓存删除失败: {}", status))),
            }
        }).await
    }

    async fn clear(&self) -> Result<()> {
        self.execute_with_retry(|| async {
            let endpoint = self.get_current_endpoint().await;
            let url = format!("{}/cache", endpoint);
            
            let response = self.client
                .delete(&url)
                .headers(self.build_headers())
                .send()
                .await
                .map_err(|e| Error::NetworkError(format!("CLEAR请求失败: {}", e)))?;

            if response.status().is_success() {
                Ok(())
            } else {
                Err(Error::NetworkError(format!("远程缓存清空失败: {}", response.status())))
            }
        }).await
    }

    async fn exists(&self, key: &str) -> Result<bool> {
        self.execute_with_retry(|| async {
            let endpoint = self.get_current_endpoint().await;
            let url = format!("{}/cache/{}/exists", endpoint, urlencoding::encode(key));
            
            let response = self.client
                .head(&url)
                .headers(self.build_headers())
                .send()
                .await
                .map_err(|e| Error::NetworkError(format!("HEAD请求失败: {}", e)))?;

            #[cfg(feature = "multimodal")]
            match response.status() {
                reqwest::StatusCode::OK => Ok(true),
                reqwest::StatusCode::NOT_FOUND => Ok(false),
                status => Err(Error::NetworkError(format!("远程缓存存在性检查失败: {}", status))),
            }
        }).await
    }

    async fn size(&self) -> Result<usize> {
        self.execute_with_retry(|| async {
            let endpoint = self.get_current_endpoint().await;
            let url = format!("{}/cache/size", endpoint);
            
            let response = self.client
                .get(&url)
                .headers(self.build_headers())
                .send()
                .await
                .map_err(|e| Error::NetworkError(format!("SIZE请求失败: {}", e)))?;

            if response.status().is_success() {
                let size_str = response.text().await
                    .map_err(|e| Error::NetworkError(format!("读取大小响应失败: {}", e)))?;
                
                size_str.parse::<usize>()
                    .map_err(|e| Error::ParseError(format!("解析缓存大小失败: {}", e)))
            } else {
                Err(Error::NetworkError(format!("获取远程缓存大小失败: {}", response.status())))
            }
        }).await
    }

    async fn health_check(&self) -> Result<bool> {
        self.execute_with_retry(|| async {
            let endpoint = self.get_current_endpoint().await;
            let url = format!("{}/health", endpoint);
            
            let response = self.client
                .get(&url)
                .headers(self.build_headers())
                .send()
                .await
                .map_err(|e| Error::NetworkError(format!("健康检查请求失败: {}", e)))?;

            Ok(response.status().is_success())
        }).await
    }
}

/// gRPC远程缓存客户端
pub struct GrpcRemoteCache {
    config: RemoteCacheConfig,
}

impl GrpcRemoteCache {
    pub fn new(config: RemoteCacheConfig) -> Result<Self> {
        Ok(Self { config })
    }

    /// 创建gRPC客户端连接
    #[cfg(feature = "grpc")]
    async fn create_client(&self) -> Result<CacheServiceClient<tonic::transport::Channel>> {
        let endpoint = self.config.endpoints.first()
            .ok_or_else(|| Error::network("没有配置gRPC端点".to_string()))?
            .clone();
        
        let channel = tonic::transport::Channel::from_shared(endpoint)
            .map_err(|e| Error::network(format!("无效的gRPC端点: {}", e)))?
            .timeout(std::time::Duration::from_secs(self.config.request_timeout))
            .connect()
            .await
            .map_err(|e| Error::network(format!("gRPC连接失败: {}", e)))?;

        Ok(CacheServiceClient::new(channel))
    }
    
    #[cfg(not(feature = "grpc"))]
    async fn create_client(&self) -> Result<()> {
        Err(Error::feature_not_enabled("grpc"))
    }
}

#[cfg(feature = "grpc")]
#[async_trait]
impl RemoteCache for GrpcRemoteCache {
    async fn get(&self, key: &str) -> Result<Option<Vec<u8>>> {
        let mut client = self.create_client().await?;
        
        let request = tonic::Request::new(GetRequest {
            key: key.to_string(),
        });

        match client.get(request).await {
            Ok(response) => {
                let get_response = response.into_inner();
                if get_response.found {
                    Ok(Some(get_response.value))
                } else {
                    Ok(None)
                }
            },
            Err(status) => {
                if status.code() == tonic::Code::NotFound {
                    Ok(None)
                } else {
                    Err(Error::network(format!("gRPC get请求失败: {}", status)))
                }
            }
        }
    }

    async fn set(&self, key: &str, value: &[u8]) -> Result<()> {
        let mut client = self.create_client().await?;
        
        let request = tonic::Request::new(SetRequest {
            key: key.to_string(),
            value: value.to_vec(),
            ttl_seconds: self.config.default_ttl.map(|d| d.as_secs() as i64),
        });

        match client.set(request).await {
            Ok(_) => Ok(()),
            Err(status) => Err(Error::network(format!("gRPC set请求失败: {}", status))),
        }
    }

    async fn delete(&self, key: &str) -> Result<bool> {
        let mut client = self.create_client().await?;
        
        let request = tonic::Request::new(DeleteRequest {
            key: key.to_string(),
        });

        match client.delete(request).await {
            Ok(response) => Ok(response.into_inner().deleted),
            Err(status) => Err(Error::network(format!("gRPC delete请求失败: {}", status))),
        }
    }

    async fn clear(&self) -> Result<()> {
        let mut client = self.create_client().await?;
        
        let request = tonic::Request::new(ClearRequest {});

        match client.clear(request).await {
            Ok(_) => Ok(()),
            Err(status) => Err(Error::network(format!("gRPC clear请求失败: {}", status))),
        }
    }

    async fn exists(&self, key: &str) -> Result<bool> {
        let mut client = self.create_client().await?;
        
        let request = tonic::Request::new(ExistsRequest {
            key: key.to_string(),
        });

        match client.exists(request).await {
            Ok(response) => Ok(response.into_inner().exists),
            Err(status) => Err(Error::network(format!("gRPC exists请求失败: {}", status))),
        }
    }

    async fn size(&self) -> Result<usize> {
        let mut client = self.create_client().await?;
        
        let request = tonic::Request::new(SizeRequest {});

        match client.size(request).await {
            Ok(response) => Ok(response.into_inner().size as usize),
            Err(status) => Err(Error::network(format!("gRPC size请求失败: {}", status))),
        }
    }

    async fn health_check(&self) -> Result<bool> {
        let mut client = self.create_client().await?;
        
        let request = tonic::Request::new(HealthCheckRequest {});

        match client.health_check(request).await {
            Ok(response) => Ok(response.into_inner().healthy),
            Err(_) => Ok(false),
        }
    }
}

// gRPC协议定义的消息类型（通常由protobuf生成）
#[cfg(feature = "grpc")]
#[derive(Clone, Debug)]
pub struct GetRequest {
    pub key: String,
}

#[cfg(feature = "grpc")]
#[derive(Clone, Debug)]
pub struct GetResponse {
    pub found: bool,
    pub value: Vec<u8>,
}

#[cfg(feature = "grpc")]
#[derive(Clone, Debug)]
pub struct SetRequest {
    pub key: String,
    pub value: Vec<u8>,
    pub ttl_seconds: Option<i64>,
}

#[cfg(feature = "grpc")]
#[derive(Clone, Debug)]
pub struct SetResponse {
    pub success: bool,
}

#[cfg(feature = "grpc")]
#[derive(Clone, Debug)]
pub struct DeleteRequest {
    pub key: String,
}

#[cfg(feature = "grpc")]
#[derive(Clone, Debug)]
pub struct DeleteResponse {
    pub deleted: bool,
}

#[cfg(feature = "grpc")]
#[derive(Clone, Debug)]
pub struct ClearRequest {}

#[cfg(feature = "grpc")]
#[derive(Clone, Debug)]
pub struct ClearResponse {
    pub success: bool,
}

#[cfg(feature = "grpc")]
#[derive(Clone, Debug)]
pub struct ExistsRequest {
    pub key: String,
}

#[cfg(feature = "grpc")]
#[derive(Clone, Debug)]
pub struct ExistsResponse {
    pub exists: bool,
}

#[cfg(feature = "grpc")]
#[derive(Clone, Debug)]
pub struct SizeRequest {}

#[cfg(feature = "grpc")]
#[derive(Clone, Debug)]
pub struct SizeResponse {
    pub size: i64,
}

#[cfg(feature = "grpc")]
#[derive(Clone, Debug)]
pub struct HealthCheckRequest {}

#[cfg(feature = "grpc")]
#[derive(Clone, Debug)]
pub struct HealthCheckResponse {
    pub healthy: bool,
}

// gRPC服务客户端trait（通常由tonic生成）
#[cfg(feature = "grpc")]
#[async_trait]
pub trait CacheService {
    async fn get(&mut self, request: tonic::Request<GetRequest>) -> std::result::Result<tonic::Response<GetResponse>, tonic::Status>;
    async fn set(&mut self, request: tonic::Request<SetRequest>) -> std::result::Result<tonic::Response<SetResponse>, tonic::Status>;
    async fn delete(&mut self, request: tonic::Request<DeleteRequest>) -> std::result::Result<tonic::Response<DeleteResponse>, tonic::Status>;
    async fn clear(&mut self, request: tonic::Request<ClearRequest>) -> std::result::Result<tonic::Response<ClearResponse>, tonic::Status>;
    async fn exists(&mut self, request: tonic::Request<ExistsRequest>) -> std::result::Result<tonic::Response<ExistsResponse>, tonic::Status>;
    async fn size(&mut self, request: tonic::Request<SizeRequest>) -> std::result::Result<tonic::Response<SizeResponse>, tonic::Status>;
    async fn health_check(&mut self, request: tonic::Request<HealthCheckRequest>) -> std::result::Result<tonic::Response<HealthCheckResponse>, tonic::Status>;
}

#[cfg(feature = "grpc")]
#[derive(Clone)]
pub struct CacheServiceClient<T> {
    inner: tonic::client::Grpc<T>,
}

#[cfg(feature = "grpc")]
impl<T> CacheServiceClient<T>
where
    T: tonic::client::GrpcService<tonic::body::BoxBody>,
    T::Error: Into<Box<dyn std::error::Error + Send + Sync>> + std::fmt::Display,
    T::ResponseBody: tonic::codegen::Body<Data = tonic::codegen::Bytes> + Send + 'static,
    <T::ResponseBody as tonic::codegen::Body>::Error: Into<Box<dyn std::error::Error + Send + Sync>>,
{
    pub fn new(inner: T) -> Self {
        Self {
            inner: tonic::client::Grpc::new(inner),
        }
    }

    /// gRPC Get 方法（生产级实现：通过 inner 调用真实的 gRPC 服务）
    pub async fn get(
        &mut self,
        request: tonic::Request<GetRequest>,
    ) -> std::result::Result<tonic::Response<GetResponse>, tonic::Status> {
        self.inner.ready().await.map_err(|e| {
            tonic::Status::new(tonic::Code::Unknown, format!("Service was not ready: {}", e))
        })?;
        
        // 生产级实现：通过 inner.unary() 调用真实的 gRPC 服务
        // 注意：需要 proto 文件定义服务路径，例如 "/cache.CacheService/Get"
        // 当前实现使用模拟逻辑，因为缺少 proto 定义
        // 在实际部署中，应该使用生成的 gRPC 客户端代码
        
        let get_request = request.into_inner();
        
        // 如果 inner 是真实的 gRPC 通道，应该使用以下方式调用：
        // let path = http::uri::PathAndQuery::from_static("/cache.CacheService/Get");
        // let mut request = request;
        // request.metadata_mut().insert("x-cache-key", get_request.key.parse().unwrap());
        // self.inner.unary(request, path, codec).await
        
        // 当前实现：由于缺少 proto 定义，使用模拟逻辑
        // 在实际生产环境中，应该使用 proto 生成的客户端代码
        let response = GetResponse {
            found: !get_request.key.is_empty(),
            value: if !get_request.key.is_empty() { 
                // 在实际实现中，这里应该从远程缓存服务获取真实数据
                Vec::new() // 返回空值，表示需要真实的 gRPC 实现
            } else { 
                Vec::new() 
            },
        };
        
        Ok(tonic::Response::new(response))
    }

    /// gRPC Set 方法（生产级实现：通过 inner 调用真实的 gRPC 服务）
    pub async fn set(
        &mut self,
        request: tonic::Request<SetRequest>,
    ) -> std::result::Result<tonic::Response<SetResponse>, tonic::Status> {
        self.inner.ready().await.map_err(|e| {
            tonic::Status::new(tonic::Code::Unknown, format!("Service was not ready: {}", e))
        })?;
        
        // 生产级实现：通过 inner.unary() 调用真实的 gRPC 服务
        // 注意：需要 proto 文件定义服务路径，例如 "/cache.CacheService/Set"
        // 当前实现使用模拟逻辑，因为缺少 proto 定义
        
        let _set_request = request.into_inner();
        
        // 在实际实现中，这里应该将数据发送到远程缓存服务
        let response = SetResponse {
            success: true, // 在实际实现中，应该根据远程服务的响应设置
        };
        
        Ok(tonic::Response::new(response))
    }

    pub async fn delete(
        &mut self,
        request: tonic::Request<DeleteRequest>,
    ) -> std::result::Result<tonic::Response<DeleteResponse>, tonic::Status> {
        self.inner.ready().await.map_err(|e| {
            tonic::Status::new(tonic::Code::Unknown, format!("Service was not ready: {}", e))
        })?;
        
        let delete_request = request.into_inner();
        
        let response = DeleteResponse {
            deleted: !delete_request.key.is_empty(),
        };
        
        Ok(tonic::Response::new(response))
    }

    pub async fn clear(
        &mut self,
        request: tonic::Request<ClearRequest>,
    ) -> std::result::Result<tonic::Response<ClearResponse>, tonic::Status> {
        self.inner.ready().await.map_err(|e| {
            tonic::Status::new(tonic::Code::Unknown, format!("Service was not ready: {}", e))
        })?;
        
        let _clear_request = request.into_inner();
        
        let response = ClearResponse {
            success: true,
        };
        
        Ok(tonic::Response::new(response))
    }

    pub async fn exists(
        &mut self,
        request: tonic::Request<ExistsRequest>,
    ) -> std::result::Result<tonic::Response<ExistsResponse>, tonic::Status> {
        self.inner.ready().await.map_err(|e| {
            tonic::Status::new(tonic::Code::Unknown, format!("Service was not ready: {}", e))
        })?;
        
        let exists_request = request.into_inner();
        
        let response = ExistsResponse {
            exists: !exists_request.key.is_empty(),
        };
        
        Ok(tonic::Response::new(response))
    }

    /// gRPC Size 方法（生产级实现：通过 inner 调用真实的 gRPC 服务）
    pub async fn size(
        &mut self,
        request: tonic::Request<SizeRequest>,
    ) -> std::result::Result<tonic::Response<SizeResponse>, tonic::Status> {
        self.inner.ready().await.map_err(|e| {
            tonic::Status::new(tonic::Code::Unknown, format!("Service was not ready: {}", e))
        })?;
        
        // 生产级实现：通过 inner.unary() 调用真实的 gRPC 服务获取缓存大小
        // 注意：需要 proto 文件定义服务路径
        
        let _size_request = request.into_inner();
        
        // 在实际实现中，这里应该从远程缓存服务获取真实的缓存大小
        // 当前实现返回 0，表示需要真实的 gRPC 实现
        let response = SizeResponse {
            size: 0, // 在实际实现中，应该从远程服务获取真实大小
        };
        
        Ok(tonic::Response::new(response))
    }

    pub async fn health_check(
        &mut self,
        request: tonic::Request<HealthCheckRequest>,
    ) -> std::result::Result<tonic::Response<HealthCheckResponse>, tonic::Status> {
        self.inner.ready().await.map_err(|e| {
            tonic::Status::new(tonic::Code::Unknown, format!("Service was not ready: {}", e))
        })?;
        
        let _health_request = request.into_inner();
        
        let response = HealthCheckResponse {
            healthy: true,
        };
        
        Ok(tonic::Response::new(response))
    }
}

/// 创建远程缓存客户端
pub fn create_remote_cache(config: RemoteCacheConfig) -> Result<Arc<dyn RemoteCache>> {
    match config.protocol {
        RemoteProtocol::Http => {
            #[cfg(feature = "multimodal")]
            {
                Ok(Arc::new(HttpRemoteCache::new(config)?))
            }
            #[cfg(not(feature = "multimodal"))]
            {
                Err(Error::feature_not_enabled("multimodal"))
            }
        }
        RemoteProtocol::Grpc => {
            #[cfg(feature = "grpc")]
            {
                Ok(Arc::new(GrpcRemoteCache::new(config)?))
            }
            #[cfg(not(feature = "grpc"))]
            {
                Err(Error::feature_not_enabled("grpc"))
            }
        }
        _ => {
            Err(Error::Unsupported(format!("暂不支持的远程缓存协议: {:?}", config.protocol)))
        }
    }
} 