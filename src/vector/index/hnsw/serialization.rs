use std::io::{self, Read, Write};
use std::collections::HashMap;
use std::sync::Arc;
use parking_lot::RwLock;
use bincode::{serialize, deserialize};
use serde::{Serialize, Deserialize};
use uuid::Uuid;

use crate::vector::index::hnsw::build::HNSWIndex;
use crate::vector::index::hnsw::node::HNSWNode;
use crate::vector::index::hnsw::types::{Distance, DistanceType, NodeIndex, Vector, VectorId};

/// 用于序列化的HNSW节点数据
#[derive(Serialize, Deserialize)]
struct SerializedNode {
    node_index: NodeIndex,
    id: VectorId,
    vector: Vector,
    level: usize,
    marked_deleted: bool,
    connections: Vec<Vec<(NodeIndex, Distance)>>,
}

/// 用于序列化的HNSW索引元数据
#[derive(Serialize, Deserialize)]
struct SerializedHNSWMeta {
    dimension: usize,
    m: usize,
    m_max: usize,
    ef_construction: usize,
    max_level: usize,
    max_level_limit: usize,
    entry_point: Option<NodeIndex>,
    distance_type: DistanceType,
    index_id: String,
}

/// 用于序列化的HNSW索引完整数据
#[derive(Serialize, Deserialize)]
struct SerializedHNSW {
    meta: SerializedHNSWMeta,
    nodes: Vec<SerializedNode>,
    id_to_index: HashMap<VectorId, NodeIndex>,
}

impl HNSWIndex {
    /// 将HNSW索引序列化到writer
    pub fn serialize<W: Write>(&self, writer: &mut W) -> io::Result<()> {
        // 创建元数据
        let meta = SerializedHNSWMeta {
            dimension: self.dimension(),
            m: self.m(),
            m_max: self.m_max(),
            ef_construction: self.ef_construction(),
            max_level: self.max_level(),
            max_level_limit: self.max_level_limit(),
            entry_point: self.entry_point(),
            distance_type: self.distance_type(),
            index_id: self.index_id().to_string(),
        };
        
        // 序列化节点
        let mut nodes = Vec::with_capacity(self.nodes().len());
        let mut id_to_index = HashMap::new();
        
        for (node_idx, node_arc) in self.nodes().iter().enumerate() {
            let node_read = node_arc.read();
            // 将 Uuid 转换为 u64 (VectorId)
            let id_u64 = node_read.id.as_u128() as u64;
            let serialized_node = SerializedNode {
                node_index: node_idx,
                id: id_u64,
                vector: node_read.vector.clone(),
                level: node_read.level,
                marked_deleted: node_read.marked_deleted,
                connections: node_read.connections.clone(),
            };
            
            nodes.push(serialized_node);
            id_to_index.insert(id_u64, node_idx);
        }
        
        // 创建完整的序列化数据
        let serialized_hnsw = SerializedHNSW {
            meta,
            nodes,
            id_to_index,
        };
        
        // 序列化并写入
        let serialized_data = serialize(&serialized_hnsw)
            .map_err(|e| io::Error::new(io::ErrorKind::Other, e))?;
        
        writer.write_all(&serialized_data)
    }
    
    /// 从reader反序列化HNSW索引
    pub fn deserialize<R: Read>(reader: &mut R) -> io::Result<Self> {
        // 读取所有数据
        let mut buffer = Vec::new();
        reader.read_to_end(&mut buffer)?;
        
        // 反序列化数据
        let serialized_hnsw: SerializedHNSW = deserialize(&buffer)
            .map_err(|e| io::Error::new(io::ErrorKind::Other, e))?;
        
        // 创建新的索引
        let meta = &serialized_hnsw.meta;
        
        let mut index = HNSWIndex::new(
            meta.dimension,
            meta.m,
            meta.ef_construction,
            meta.distance_type,
            meta.max_level_limit as f32 / 2.0, // 尝试从max_level_limit恢复ml参数
            meta.max_level_limit,
            meta.index_id.clone(),
        );
        
        // 设置索引状态
        index.set_max_level(meta.max_level);
        index.set_entry_point(meta.entry_point);
        index.set_id_to_index(serialized_hnsw.id_to_index.clone());
        
        // 重建节点
        for node_data in serialized_hnsw.nodes {
            // 将 u64 (VectorId) 转换为 Uuid
            let id_uuid = Uuid::from_u128(node_data.id as u128);
            let mut node = HNSWNode::new(
                node_data.node_index,
                id_uuid,
                node_data.vector,
                node_data.level,
            );
            
            node.marked_deleted = node_data.marked_deleted;
            node.connections = node_data.connections;
            
            let shared_node = Arc::new(RwLock::new(node));
            index.add_node(shared_node);
        }
        
        Ok(index)
    }
    
    /// 将HNSW索引保存到文件
    pub fn save_to_file(&self, file_path: &str) -> io::Result<()> {
        let mut file = std::fs::File::create(file_path)?;
        self.serialize(&mut file)
    }
    
    /// 从文件加载HNSW索引
    pub fn load_from_file(file_path: &str) -> io::Result<Self> {
        let mut file = std::fs::File::open(file_path)?;
        Self::deserialize(&mut file)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;
    use uuid::Uuid;
    
    fn create_test_index() -> HNSWIndex {
        let mut index = HNSWIndex::new(
            4,   // dimension
            16,  // m
            200, // ef_construction
            DistanceType::Euclidean,
            1.0 / 2.0.ln(), // ml
            16,  // max_level_limit
            "test_index".to_string(),
        );
        
        // 添加一些测试数据
        for i in 0..10 {
            let vec = vec![i as f32, (i+1) as f32, (i+2) as f32, (i+3) as f32];
            index.add(Uuid::new_v4(), vec, None).unwrap();
        }
        
        index
    }
    
    #[test]
    fn test_serialization_deserialization() {
        let index = create_test_index();
        
        // 创建临时目录
        let dir = tempdir().unwrap();
        let file_path = dir.path().join("test_index.bin");
        
        // 保存索引
        index.save_to_file(file_path.to_str().unwrap()).unwrap();
        
        // 加载索引
        let loaded_index = HNSWIndex::load_from_file(file_path.to_str().unwrap()).unwrap();
        
        // 验证加载的索引
        assert_eq!(index.dimension(), loaded_index.dimension());
        assert_eq!(index.distance_type(), loaded_index.distance_type());
        assert_eq!(index.max_level(), loaded_index.max_level());
        assert_eq!(index.size(), loaded_index.size());
        
        // 验证搜索功能
        let query = vec![1.0, 2.0, 3.0, 4.0];
        let config = crate::vector::index::hnsw::types::SearchConfig::new()
            .with_limit(3);
            
        let original_results = index.search(&query, config.clone()).len();
        let loaded_results = loaded_index.search(&query, config).len();
        
        assert_eq!(original_results, loaded_results);
    }
}
