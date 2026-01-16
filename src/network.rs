//! Network module - Local mode implementation
//!
//! Provides network-related types for vecminDB's embedded mode.

use serde::{Deserialize, Serialize};
use std::sync::Arc;
use crate::Result;

/// Node role
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum NodeRole {
    Master,
    Worker,
    Coordinator,
}

/// Node information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeInfo {
    pub id: String,
    pub role: NodeRole,
    pub address: String,
}

/// Network manager - Local mode implementation
///
/// vecminDB 作为嵌入式向量数据库，采用本地模式运行，不实现分布式网络层。
/// 为保证上层模块（事件总线、自动化等）的接口完整性，此实现提供：
/// - 本地模式下的网络管理接口
/// - 对外发送消息返回 `Error::not_implemented`
/// - 预留本地事件系统挂载接口（避免循环依赖）
#[derive(Debug, Clone, Default)]
pub struct NetworkManager {
    // In the current extracted vecminDB, the concrete local event system type
    // is not wired here to avoid cross‑module tight coupling. This field can be
    // extended in the future when a unified local event bus is stabilized.
}

impl NetworkManager {
    pub fn new() -> Result<Self> {
        Ok(Self::default())
    }
    
    /// 获取本地事件系统的引用
    ///
    /// 在当前本地模式下返回 None。使用 `dyn EventSystem` 作为接口类型，
    /// 保证与事件子系统的契约对齐，便于后续扩展。
    pub fn get_local_event_system(&self) -> Option<Arc<dyn crate::event::EventSystem>> {
        None
    }

    /// 广播消息到某个逻辑通道。
    ///
    /// 当前实现为占位的「本地模式」：直接返回未实现错误，避免静默丢包。
    pub async fn broadcast_message(&self, _channel: &str, _data: &[u8]) -> Result<()> {
        Err(crate::error::Error::not_implemented(
            "NetworkManager::broadcast_message: 分布式网络未在 vecminDB 中实现",
        ))
    }

    /// 向指定节点发送消息。
    ///
    /// 当前实现为占位的「本地模式」：直接返回未实现错误。
    pub async fn send_to(&self, _node: &str, _channel: &str, _data: &[u8]) -> Result<()> {
        Err(crate::error::Error::not_implemented(
            "NetworkManager::send_to: 分布式网络未在 vecminDB 中实现",
        ))
    }

    /// 注册消息处理器
    ///
    /// 当前实现为占位的「本地模式」：直接返回未实现错误。
    pub fn register_handler<F>(&self, _channel: &str, _handler: F) -> Result<()>
    where
        F: Fn(&[u8], &str) + Send + Sync + 'static,
    {
        Err(crate::error::Error::not_implemented(
            "NetworkManager::register_handler: 分布式网络未在 vecminDB 中实现",
        ))
    }

    /// 获取所有节点
    ///
    /// Local mode: returns empty list as vecminDB runs as a single-node embedded database.
    pub async fn get_all_nodes(&self) -> Vec<NodeInfo> {
        Vec::new()
    }

    /// 根据角色获取节点
    ///
    /// Local mode: returns empty list as vecminDB runs as a single-node embedded database.
    pub async fn get_nodes_by_role(&self, _role: &NodeRole) -> Vec<NodeInfo> {
        Vec::new()
    }

    /// 根据组获取节点
    ///
    /// Local mode: returns empty list as vecminDB runs as a single-node embedded database.
    pub async fn get_nodes_by_group(&self, _group: &str) -> Result<Vec<NodeInfo>> {
        Ok(Vec::new())
    }
}

