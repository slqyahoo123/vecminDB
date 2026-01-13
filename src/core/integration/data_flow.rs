/// 数据流管理模块
/// 
/// 提供清晰的跨模块数据流转路径和管理机制

use std::sync::{Arc, Mutex, RwLock};
use std::collections::{HashMap, VecDeque};
use async_trait::async_trait;
use chrono::{DateTime, Utc};
use uuid::Uuid;
use serde::{Serialize, Deserialize};
use log::{info, error, debug};
use anyhow::Result;

/// 数据流节点类型
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum DataFlowNodeType {
    Source,      // 数据源
    Processor,   // 数据处理器
    Transformer, // 数据转换器
    Sink,        // 数据接收器
    Junction,    // 数据分叉点
    Merger,      // 数据合并点
}

/// 数据流节点状态
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum DataFlowNodeStatus {
    Idle,
    Running,
    Blocked,
    Error,
    Completed,
}

/// 数据包
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataPacket {
    pub id: String,
    pub flow_id: String,
    pub source_node: String,
    pub target_node: Option<String>,
    pub data_type: String,
    pub payload: Vec<u8>,
    pub metadata: HashMap<String, String>,
    pub created_at: DateTime<Utc>,
    pub expires_at: Option<DateTime<Utc>>,
}

impl DataPacket {
    pub fn new(flow_id: String, source_node: String, data_type: String, payload: Vec<u8>) -> Self {
        Self {
            id: Uuid::new_v4().to_string(),
            flow_id,
            source_node,
            target_node: None,
            data_type,
            payload,
            metadata: HashMap::new(),
            created_at: Utc::now(),
            expires_at: None,
        }
    }
    
    pub fn with_target(mut self, target_node: String) -> Self {
        self.target_node = Some(target_node);
        self
    }
    
    pub fn with_metadata(mut self, key: String, value: String) -> Self {
        self.metadata.insert(key, value);
        self
    }
    
    pub fn with_expiry(mut self, expires_at: DateTime<Utc>) -> Self {
        self.expires_at = Some(expires_at);
        self
    }
}

/// 数据流节点
#[derive(Debug, Clone)]
pub struct DataFlowNode {
    pub id: String,
    pub name: String,
    pub node_type: DataFlowNodeType,
    pub status: DataFlowNodeStatus,
    pub input_ports: Vec<String>,
    pub output_ports: Vec<String>,
    pub properties: HashMap<String, String>,
    pub created_at: DateTime<Utc>,
    pub last_activity: Option<DateTime<Utc>>,
}

impl DataFlowNode {
    pub fn new(id: String, name: String, node_type: DataFlowNodeType) -> Self {
        Self {
            id,
            name,
            node_type,
            status: DataFlowNodeStatus::Idle,
            input_ports: Vec::new(),
            output_ports: Vec::new(),
            properties: HashMap::new(),
            created_at: Utc::now(),
            last_activity: None,
        }
    }
    
    pub fn add_input_port(mut self, port_name: String) -> Self {
        self.input_ports.push(port_name);
        self
    }
    
    pub fn add_output_port(mut self, port_name: String) -> Self {
        self.output_ports.push(port_name);
        self
    }
    
    pub fn with_property(mut self, key: String, value: String) -> Self {
        self.properties.insert(key, value);
        self
    }
}

/// 数据流连接
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataFlowConnection {
    pub id: String,
    pub source_node: String,
    pub source_port: String,
    pub target_node: String,
    pub target_port: String,
    pub connection_type: String,
    pub buffer_size: usize,
    pub is_active: bool,
    pub metadata: HashMap<String, String>,
}

impl DataFlowConnection {
    pub fn new(
        source_node: String,
        source_port: String,
        target_node: String,
        target_port: String,
    ) -> Self {
        Self {
            id: Uuid::new_v4().to_string(),
            source_node,
            source_port,
            target_node,
            target_port,
            connection_type: "direct".to_string(),
            buffer_size: 100,
            is_active: true,
            metadata: HashMap::new(),
        }
    }
    
    pub fn with_buffer_size(mut self, buffer_size: usize) -> Self {
        self.buffer_size = buffer_size;
        self
    }
    
    pub fn with_connection_type(mut self, connection_type: String) -> Self {
        self.connection_type = connection_type;
        self
    }
}

/// 数据流处理器接口
#[async_trait]
pub trait DataFlowProcessor: Send + Sync {
    /// 处理器ID
    fn processor_id(&self) -> &str;
    
    /// 处理数据包
    async fn process(&self, packet: DataPacket) -> Result<Vec<DataPacket>>;
    
    /// 获取处理器状态
    fn get_status(&self) -> DataFlowNodeStatus;
    
    /// 启动处理器
    async fn start(&mut self) -> Result<()>;
    
    /// 停止处理器
    async fn stop(&mut self) -> Result<()>;
}

/// 数据流管道
#[derive(Clone)]
pub struct DataFlowPipeline {
    pub id: String,
    pub name: String,
    pub nodes: HashMap<String, DataFlowNode>,
    pub connections: HashMap<String, DataFlowConnection>,
    pub processors: HashMap<String, Arc<dyn DataFlowProcessor>>,
    pub buffers: HashMap<String, VecDeque<DataPacket>>,
    pub is_running: bool,
    pub created_at: DateTime<Utc>,
    pub metadata: HashMap<String, String>,
}

impl DataFlowPipeline {
    pub fn new(name: String) -> Self {
        Self {
            id: Uuid::new_v4().to_string(),
            name,
            nodes: HashMap::new(),
            connections: HashMap::new(),
            processors: HashMap::new(),
            buffers: HashMap::new(),
            is_running: false,
            created_at: Utc::now(),
            metadata: HashMap::new(),
        }
    }
    
    /// 添加节点
    pub fn add_node(&mut self, node: DataFlowNode) -> Result<()> {
        let node_id = node.id.clone();
        
        // 为节点创建缓冲区
        self.buffers.insert(node_id.clone(), VecDeque::new());
        
        // 添加节点
        self.nodes.insert(node_id, node);
        
        Ok(())
    }
    
    /// 添加连接
    pub fn add_connection(&mut self, connection: DataFlowConnection) -> Result<()> {
        // 验证源节点和目标节点是否存在
        if !self.nodes.contains_key(&connection.source_node) {
            return Err(anyhow::anyhow!(
                "源节点不存在: {}", 
                connection.source_node
            ));
        }
        
        if !self.nodes.contains_key(&connection.target_node) {
            return Err(anyhow::anyhow!(
                "目标节点不存在: {}", 
                connection.target_node
            ));
        }
        
        let connection_id = connection.id.clone();
        self.connections.insert(connection_id, connection);
        
        Ok(())
    }
    
    /// 注册处理器
    pub fn register_processor(&mut self, processor: Arc<dyn DataFlowProcessor>) {
        self.processors.insert(processor.processor_id().to_string(), processor);
    }
    
    /// 发送数据包
    pub async fn send_packet(&mut self, packet: DataPacket) -> Result<()> {
        let source_node_id = packet.source_node.clone();
        
        // 将数据包添加到源节点的缓冲区
        if let Some(buffer) = self.buffers.get_mut(&source_node_id) {
            buffer.push_back(packet);
            debug!("数据包已添加到节点缓冲区: {}", source_node_id);
        } else {
            return Err(anyhow::anyhow!(
                "节点缓冲区不存在: {}", 
                source_node_id
            ));
        }
        
        // 触发数据流处理
        self.process_flow().await?;
        
        Ok(())
    }
    
    /// 处理数据流
    async fn process_flow(&mut self) -> Result<()> {
        let mut packets_to_route = Vec::new();
        
        // 从所有节点的缓冲区收集数据包
        for (node_id, buffer) in self.buffers.iter_mut() {
            while let Some(packet) = buffer.pop_front() {
                packets_to_route.push((node_id.clone(), packet));
            }
        }
        
        // 路由数据包
        for (source_node_id, packet) in packets_to_route {
            self.route_packet(source_node_id, packet).await?;
        }
        
        Ok(())
    }
    
    /// 路由数据包
    async fn route_packet(&mut self, source_node_id: String, packet: DataPacket) -> Result<()> {
        // 查找从源节点出发的连接
        let outgoing_connections: Vec<_> = self.connections.values()
            .filter(|conn| conn.source_node == source_node_id && conn.is_active)
            .cloned()
            .collect();
        
        if outgoing_connections.is_empty() {
            debug!("节点 {} 没有输出连接，数据包将被丢弃", source_node_id);
            return Ok(());
        }
        
        // 为每个输出连接创建数据包副本
        for connection in &outgoing_connections {
            let mut packet_copy = packet.clone();
            packet_copy.target_node = Some(connection.target_node.clone());
            
            // 如果目标节点有处理器，则处理数据包
            if let Some(processor) = self.processors.get(&connection.target_node) {
                match processor.process(packet_copy.clone()).await {
                    Ok(output_packets) => {
                        // 将输出数据包添加到目标节点的缓冲区
                        if let Some(buffer) = self.buffers.get_mut(&connection.target_node) {
                            for output_packet in output_packets {
                                buffer.push_back(output_packet);
                            }
                        }
                        debug!("数据包已处理: {} -> {}", source_node_id, connection.target_node);
                    },
                    Err(e) => {
                        error!("数据包处理失败: {} -> {}, 错误: {}", 
                               source_node_id, connection.target_node, e);
                    }
                }
            } else {
                // 没有处理器，直接转发到目标节点的缓冲区
                if let Some(buffer) = self.buffers.get_mut(&connection.target_node) {
                    buffer.push_back(packet_copy);
                    debug!("数据包已转发: {} -> {}", source_node_id, connection.target_node);
                }
            }
        }
        
        Ok(())
    }
    
    /// 启动管道
    pub async fn start(&mut self) -> Result<()> {
        if self.is_running {
            return Ok(());
        }
        
        info!("启动数据流管道: {}", self.name);
        
        // 启动所有处理器
        for processor in self.processors.values() {
            // 注意：这里需要可变引用，但我们通过Arc<dyn Trait>无法获得
            // 实际实现中需要重新设计接口或使用内部可变性
            debug!("启动处理器: {}", processor.processor_id());
        }
        
        self.is_running = true;
        info!("数据流管道启动完成: {}", self.name);
        
        Ok(())
    }
    
    /// 停止管道
    pub async fn stop(&mut self) -> Result<()> {
        if !self.is_running {
            return Ok(());
        }
        
        info!("停止数据流管道: {}", self.name);
        
        // 停止所有处理器
        for processor in self.processors.values() {
            debug!("停止处理器: {}", processor.processor_id());
        }
        
        self.is_running = false;
        info!("数据流管道停止完成: {}", self.name);
        
        Ok(())
    }
    
    /// 获取管道状态
    pub fn get_status(&self) -> HashMap<String, String> {
        let mut status = HashMap::new();
        status.insert("pipeline_id".to_string(), self.id.clone());
        status.insert("name".to_string(), self.name.clone());
        status.insert("is_running".to_string(), self.is_running.to_string());
        status.insert("nodes_count".to_string(), self.nodes.len().to_string());
        status.insert("connections_count".to_string(), self.connections.len().to_string());
        status.insert("processors_count".to_string(), self.processors.len().to_string());
        
        // 统计缓冲区状态
        let total_buffered = self.buffers.values().map(|b| b.len()).sum::<usize>();
        status.insert("buffered_packets".to_string(), total_buffered.to_string());
        
        status
    }
}

/// 数据流管理器
pub struct DataFlowManager {
    pipelines: Arc<RwLock<HashMap<String, DataFlowPipeline>>>,
    buffer_size: usize,
    cleanup_interval: std::time::Duration,
}

impl DataFlowManager {
    pub fn new(buffer_size: usize) -> Result<Self> {
        Ok(Self {
            pipelines: Arc::new(RwLock::new(HashMap::new())),
            buffer_size,
            cleanup_interval: std::time::Duration::from_secs(300), // 5分钟清理间隔
        })
    }
    
    /// 创建数据流管道
    pub fn create_pipeline(&self, name: String) -> Result<String> {
        let pipeline = DataFlowPipeline::new(name);
        let pipeline_id = pipeline.id.clone();
        
        let mut pipelines = self.pipelines.write()
            .expect("流水线列表写入锁获取失败：无法更新流水线");
        pipelines.insert(pipeline_id.clone(), pipeline);
        
        info!("创建数据流管道: {}", pipeline_id);
        Ok(pipeline_id)
    }
    
    /// 获取管道
    pub fn get_pipeline(&self, pipeline_id: &str) -> Option<DataFlowPipeline> {
        let pipelines = self.pipelines.read()
            .expect("流水线列表读取锁获取失败：无法读取流水线");
        pipelines.get(pipeline_id).cloned()
    }
    
    /// 删除管道
    pub async fn delete_pipeline(&self, pipeline_id: &str) -> Result<()> {
        // 首先停止管道
        {
            let mut pipelines = self.pipelines.write()
            .expect("流水线列表写入锁获取失败：无法更新流水线");
            if let Some(pipeline) = pipelines.get_mut(pipeline_id) {
                pipeline.stop().await?;
            }
        }
        
        // 然后删除管道
        {
            let mut pipelines = self.pipelines.write()
            .expect("流水线列表写入锁获取失败：无法更新流水线");
            pipelines.remove(pipeline_id);
        }
        
        info!("删除数据流管道: {}", pipeline_id);
        Ok(())
    }
    
    /// 列出所有管道
    pub fn list_pipelines(&self) -> Vec<String> {
        let pipelines = self.pipelines.read()
            .expect("流水线列表读取锁获取失败：无法读取流水线");
        pipelines.keys().cloned().collect()
    }
    
    /// 向管道发送数据包
    pub async fn send_to_pipeline(&self, pipeline_id: &str, packet: DataPacket) -> Result<()> {
        let mut pipelines = self.pipelines.write()
            .expect("流水线列表写入锁获取失败：无法更新流水线");
        if let Some(pipeline) = pipelines.get_mut(pipeline_id) {
            pipeline.send_packet(packet).await?;
            Ok(())
        } else {
            Err(anyhow::anyhow!("管道不存在: {}", pipeline_id))
        }
    }
    
    /// 获取管道状态
    pub fn get_pipeline_status(&self, pipeline_id: &str) -> Option<HashMap<String, String>> {
        let pipelines = self.pipelines.read()
            .expect("流水线列表读取锁获取失败：无法读取流水线");
        pipelines.get(pipeline_id).map(|p| p.get_status())
    }
    
    /// 启动所有管道
    pub async fn start_all_pipelines(&self) -> Result<()> {
        let mut pipelines = self.pipelines.write()
            .expect("流水线列表写入锁获取失败：无法更新流水线");
        for pipeline in pipelines.values_mut() {
            pipeline.start().await?;
        }
        info!("所有数据流管道已启动");
        Ok(())
    }
    
    /// 停止所有管道
    pub async fn stop_all_pipelines(&self) -> Result<()> {
        let mut pipelines = self.pipelines.write()
            .expect("流水线列表写入锁获取失败：无法更新流水线");
        for pipeline in pipelines.values_mut() {
            pipeline.stop().await?;
        }
        info!("所有数据流管道已停止");
        Ok(())
    }
    
    /// 清理过期数据包
    pub fn cleanup_expired_packets(&self) -> usize {
        let mut pipelines = self.pipelines.write()
            .expect("流水线列表写入锁获取失败：无法更新流水线");
        let mut cleaned_count = 0;
        let now = Utc::now();
        
        for pipeline in pipelines.values_mut() {
            for buffer in pipeline.buffers.values_mut() {
                let initial_len = buffer.len();
                buffer.retain(|packet| {
                    if let Some(expires_at) = packet.expires_at {
                        expires_at > now
                    } else {
                        true // 没有过期时间的数据包保留
                    }
                });
                cleaned_count += initial_len - buffer.len();
            }
        }
        
        if cleaned_count > 0 {
            info!("清理了 {} 个过期数据包", cleaned_count);
        }
        
        cleaned_count
    }
    
    /// 启动清理任务
    pub async fn start_cleanup_task(&self) {
        let manager = Arc::new(self.clone_ref());
        let cleanup_interval = self.cleanup_interval;
        
        tokio::spawn(async move {
            let mut interval = tokio::time::interval(cleanup_interval);
            loop {
                interval.tick().await;
                manager.cleanup_expired_packets();
            }
        });
        
        info!("数据流清理任务已启动，间隔: {:?}", cleanup_interval);
    }
    
    /// 创建引用副本（简化实现）
    fn clone_ref(&self) -> Self {
        Self {
            pipelines: self.pipelines.clone(),
            buffer_size: self.buffer_size,
            cleanup_interval: self.cleanup_interval,
        }
    }
}

/// 简单数据处理器示例
pub struct SimpleDataProcessor {
    id: String,
    processor_type: String,
    status: Arc<Mutex<DataFlowNodeStatus>>,
}

impl SimpleDataProcessor {
    pub fn new(id: String, processor_type: String) -> Self {
        Self {
            id,
            processor_type,
            status: Arc::new(Mutex::new(DataFlowNodeStatus::Idle)),
        }
    }
}

#[async_trait]
impl DataFlowProcessor for SimpleDataProcessor {
    fn processor_id(&self) -> &str {
        &self.id
    }
    
    async fn process(&self, mut packet: DataPacket) -> Result<Vec<DataPacket>> {
        debug!("处理数据包: {} (处理器: {})", packet.id, self.id);
        
        // 更新状态为运行中
        {
            let mut status = self.status.lock().unwrap();
            *status = DataFlowNodeStatus::Running;
        }
        
        // 模拟数据处理
        tokio::time::sleep(tokio::time::Duration::from_millis(10)).await;
        
        // 添加处理器信息到元数据
        packet.metadata.insert(
            "processed_by".to_string(), 
            self.id.clone()
        );
        packet.metadata.insert(
            "processor_type".to_string(), 
            self.processor_type.clone()
        );
        
        // 更新状态为完成
        {
            let mut status = self.status.lock().unwrap();
            *status = DataFlowNodeStatus::Completed;
        }
        
        Ok(vec![packet])
    }
    
    fn get_status(&self) -> DataFlowNodeStatus {
        let status = self.status.lock().unwrap();
        status.clone()
    }
    
    async fn start(&mut self) -> Result<()> {
        let mut status = self.status.lock().unwrap();
        *status = DataFlowNodeStatus::Idle;
        debug!("启动处理器: {}", self.id);
        Ok(())
    }
    
    async fn stop(&mut self) -> Result<()> {
        let mut status = self.status.lock().unwrap();
        *status = DataFlowNodeStatus::Idle;
        debug!("停止处理器: {}", self.id);
        Ok(())
    }
} 