// 向量索引公共组件
// 包含共享的数据结构和类型，避免循环依赖

use serde::{Serialize, Deserialize};

/// HNSW索引节点
#[derive(Clone, Serialize, Deserialize, Debug)]
pub struct HNSWNode {
    pub id: String,
    pub vector: Vec<f32>,
    pub metadata: Option<serde_json::Value>,
    pub connections: Vec<Vec<String>>, // 每层的连接
} 