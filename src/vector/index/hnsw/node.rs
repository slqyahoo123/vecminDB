//! HNSW索引的节点实现
//!
//! 本模块定义了HNSW图中的节点结构及其操作方法。节点是HNSW索引的基本单位，
//! 包含向量数据、ID、元数据以及与其他节点的连接信息。

use crate::vector::index::hnsw::types::{NodeConnections, NodeConnection};
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use uuid::Uuid;

/// HNSW图中的节点，包含向量数据和连接信息
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HNSWNode {
    /// 节点在存储中的索引位置
    pub node_index: usize,
    /// 向量唯一标识符
    pub id: Uuid,
    /// 向量数据
    pub vector: Vec<f32>,
    /// 节点所在的最高层级
    pub level: usize,
    /// 节点是否已被标记为删除
    pub marked_deleted: bool,
    /// 各层的连接信息，每层包含多个连接
    pub connections: NodeConnections,
}

impl HNSWNode {
    /// 创建一个新的HNSW节点
    pub fn new(node_index: usize, id: Uuid, vector: Vec<f32>, level: usize) -> Self {
        let mut connections = Vec::with_capacity(level + 1);
        for _ in 0..=level {
            connections.push(Vec::new());
        }

        Self {
            node_index,
            id,
            vector,
            level,
            marked_deleted: false,
            connections,
        }
    }

    /// 获取指定层的连接（返回元组类型的连接）
    pub fn get_connections(&self, layer: usize) -> Option<&Vec<(usize, f32)>> {
        if layer <= self.level && layer < self.connections.len() {
            Some(&self.connections[layer])
        } else {
            None
        }
    }

    /// 获取指定层的可变连接（返回元组类型的连接）
    pub fn get_connections_mut(&mut self, layer: usize) -> Option<&mut Vec<(usize, f32)>> {
        if layer <= self.level && layer < self.connections.len() {
            Some(&mut self.connections[layer])
        } else {
            None
        }
    }

    /// 添加一个连接到指定层（使用元组类型）
    pub fn add_connection(&mut self, layer: usize, connection: (usize, f32)) -> bool {
        if layer > self.level {
            return false;
        }
        if layer >= self.connections.len() {
            // 扩展连接层
            while self.connections.len() <= layer {
                self.connections.push(Vec::new());
            }
        }
        self.connections[layer].push(connection);
        true
    }
    
    /// 添加一个 NodeConnection 到指定层（兼容旧接口）
    pub fn add_node_connection(&mut self, layer: usize, connection: NodeConnection) -> bool {
        self.add_connection(layer, (connection.node_id, connection.distance))
    }

    /// 设置指定层的所有连接（使用元组类型）
    pub fn set_connections(&mut self, layer: usize, connections: Vec<(usize, f32)>) -> bool {
        if layer > self.level {
            return false;
        }
        self.connections[layer] = connections;
        true
    }

    /// 标记节点为已删除
    pub fn mark_deleted(&mut self) {
        self.marked_deleted = true;
    }

    /// 检查节点是否已删除
    pub fn is_deleted(&self) -> bool {
        self.marked_deleted
    }

    /// 获取节点占用的内存估计(字节)
    pub fn memory_usage(&self) -> usize {
        // 基础结构大小
        let mut size = std::mem::size_of::<Self>();
        
        // 向量数据大小
        size += self.vector.len() * std::mem::size_of::<f32>();
        
        // 连接数据大小
        for layer_connections in &self.connections {
            size += layer_connections.len() * std::mem::size_of::<(usize, f32)>();
        }
        
        size
    }
}

/// 线程安全的包装节点类型
pub type SharedNode = Arc<RwLock<HNSWNode>>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_node_creation() {
        let id = Uuid::new_v4();
        let vector = vec![1.0, 2.0, 3.0];
        let node = HNSWNode::new(0, id, vector.clone(), 2);
        
        assert_eq!(node.node_index, 0);
        assert_eq!(node.id, id);
        assert_eq!(node.vector, vector);
        assert_eq!(node.level, 2);
        assert_eq!(node.is_deleted(), false);
        assert_eq!(node.connections.len(), 3); // 0, 1, 2层
    }

    #[test]
    fn test_connection_operations() {
        let id = Uuid::new_v4();
        let vector = vec![1.0, 2.0, 3.0];
        let mut node = HNSWNode::new(0, id, vector, 2);
        
        // 添加连接（使用元组）
        let conn = (5, 0.5);
        assert!(node.add_connection(1, conn));
        
        // 获取连接
        let connections = node.get_connections(1).unwrap();
        assert_eq!(connections.len(), 1);
        assert_eq!(connections[0].0, 5);  // 元组的第一个元素是 node_id
        assert_eq!(connections[0].1, 0.5); // 元组的第二个元素是 distance
        
        // 设置连接（使用元组）
        let new_connections = vec![
            (10, 0.3),
            (15, 0.7),
        ];
        assert!(node.set_connections(1, new_connections.clone()));
        
        let updated = node.get_connections(1).unwrap();
        assert_eq!(updated.len(), 2);
        assert_eq!(updated[0].0, 10);  // 元组的第一个元素是 node_id
        assert_eq!(updated[1].0, 15);  // 元组的第一个元素是 node_id
        
        // 超出层级的操作应当失败
        assert!(!node.add_connection(3, (20, 0.4)));
        assert!(!node.set_connections(3, vec![(20, 0.4)]));
        assert!(node.get_connections(3).is_none());
    }

    #[test]
    fn test_deletion() {
        let id = Uuid::new_v4();
        let vector = vec![1.0, 2.0, 3.0];
        let mut node = HNSWNode::new(0, id, vector, 1);
        
        assert!(!node.is_deleted());
        
        node.mark_deleted();
        assert!(node.is_deleted());
    }
} 