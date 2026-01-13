/// 类型定义模块
/// 
/// 包含算法执行器使用的各种内部类型定义

/// 解析模型参数结构
#[derive(Debug, Clone)]
pub(crate) struct ModelParameters {
    pub weights: Option<Vec<f32>>,
    pub bias: Option<f32>,
    pub network_config: Option<NetworkConfig>,
    pub tree_structure: Option<TreeStructure>,
    pub k_value: Option<usize>,
}

/// 神经网络配置
#[derive(Debug, Clone)]
pub(crate) struct NetworkConfig {
    pub layers: Vec<LayerConfig>,
}

/// 层配置
#[derive(Debug, Clone)]
pub(crate) struct LayerConfig {
    pub input_size: usize,
    pub output_size: usize,
    pub weights: Vec<f32>,
    pub bias: Vec<f32>,
    pub activation: String,
}

/// 决策树结构
#[derive(Debug, Clone)]
pub(crate) struct TreeStructure {
    pub nodes: Vec<TreeNode>,
}

/// 树节点
#[derive(Debug, Clone)]
pub(crate) struct TreeNode {
    pub feature_index: usize,
    pub threshold: f32,
    pub left_child: Option<usize>,
    pub right_child: Option<usize>,
    pub value: Option<f32>, // 叶子节点的值
}

