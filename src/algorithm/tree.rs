use std::collections::HashMap;
use serde::{Serialize, Deserialize};
use uuid::Uuid;
use std::fmt;
use chrono;
use crate::error::{Result, Error};
use crate::algorithm::{Algorithm, AlgorithmType};

/// 分割类型
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum SplitType {
    /// 数值型分割
    Numeric,
    /// 类别型分割
    Categorical,
}

impl fmt::Display for SplitType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SplitType::Numeric => write!(f, "numeric"),
            SplitType::Categorical => write!(f, "categorical"),
        }
    }
}

/// 决策树节点
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TreeNode {
    /// 节点ID
    pub id: String,
    /// 特征索引
    pub feature_index: Option<usize>,
    /// 特征名称
    pub feature_name: Option<String>,
    /// 分割类型
    pub split_type: Option<SplitType>,
    /// 分割值 (数值型)
    pub threshold: Option<f32>,
    /// 分割集合 (类别型)
    pub categories: Option<Vec<String>>,
    /// 预测值
    pub value: Option<f32>,
    /// 节点深度
    pub depth: usize,
    /// 样本数量
    pub samples: usize,
    /// 不纯度
    pub impurity: f32,
    /// 左子节点
    pub left: Option<Box<TreeNode>>,
    /// 右子节点
    pub right: Option<Box<TreeNode>>,
    /// 是否为叶节点
    pub is_leaf: bool,
    /// 类别分布 (分类树)
    pub class_distribution: Option<HashMap<String, usize>>,
}

impl TreeNode {
    /// 创建新的叶节点
    pub fn new_leaf(value: f32, samples: usize, depth: usize, impurity: f32) -> Self {
        Self {
            id: Uuid::new_v4().to_string(),
            feature_index: None,
            feature_name: None,
            split_type: None,
            threshold: None,
            categories: None,
            value: Some(value),
            depth,
            samples,
            impurity,
            left: None,
            right: None,
            is_leaf: true,
            class_distribution: None,
        }
    }
    
    /// 创建新的内部节点
    pub fn new_internal(
        feature_index: usize,
        feature_name: Option<String>,
        split_type: SplitType,
        samples: usize,
        depth: usize,
        impurity: f32,
    ) -> Self {
        Self {
            id: Uuid::new_v4().to_string(),
            feature_index: Some(feature_index),
            feature_name,
            split_type: Some(split_type),
            threshold: None,
            categories: None,
            value: None,
            depth,
            samples,
            impurity,
            left: None,
            right: None,
            is_leaf: false,
            class_distribution: None,
        }
    }
    
    /// 设置数值分割阈值
    pub fn with_threshold(mut self, threshold: f32) -> Self {
        self.threshold = Some(threshold);
        self
    }
    
    /// 设置类别分割集合
    pub fn with_categories(mut self, categories: Vec<String>) -> Self {
        self.categories = Some(categories);
        self
    }
    
    /// 设置子节点
    pub fn with_children(mut self, left: TreeNode, right: TreeNode) -> Self {
        self.left = Some(Box::new(left));
        self.right = Some(Box::new(right));
        self
    }
    
    /// 设置类别分布
    pub fn with_class_distribution(mut self, distribution: HashMap<String, usize>) -> Self {
        self.class_distribution = Some(distribution);
        self
    }
    
    /// 预测单个样本的值
    pub fn predict(&self, features: &[f32], feature_names: &[String]) -> f32 {
        if self.is_leaf {
            return self.value.unwrap_or(0.0);
        }
        
        let feature_index = self.feature_index.unwrap();
        let split_type = self.split_type.unwrap();
        
        match split_type {
            SplitType::Numeric => {
                let threshold = self.threshold.unwrap();
                let feature_value = features[feature_index];
                
                if feature_value <= threshold {
                    self.left.as_ref().unwrap().predict(features, feature_names)
                } else {
                    self.right.as_ref().unwrap().predict(features, feature_names)
                }
            },
            SplitType::Categorical => {
                // 在实际应用中，类别型特征需要编码为数值
                // 这里简化处理，假设特征值本身就是编码后的索引
                let feature_value = features[feature_index] as usize;
                
                // 根据所属类别决定路径
                if let Some(cats) = &self.categories {
                    if cats.contains(&feature_value.to_string()) {
                        self.left.as_ref().unwrap().predict(features, feature_names)
                    } else {
                        self.right.as_ref().unwrap().predict(features, feature_names)
                    }
                } else {
                    // 如果没有类别集合，则默认走左子树
                    self.left.as_ref().unwrap().predict(features, feature_names)
                }
            },
        }
    }
    
    /// 获取决策路径
    pub fn get_decision_path(&self, features: &[f32], feature_names: &[String]) -> Vec<String> {
        let mut path = Vec::new();
        
        if self.is_leaf {
            path.push(format!("预测值: {:.4}", self.value.unwrap_or(0.0)));
            return path;
        }
        
        let feature_index = self.feature_index.unwrap();
        let feature_name = if let Some(name) = &self.feature_name {
            name.clone()
        } else if feature_index < feature_names.len() {
            feature_names[feature_index].clone()
        } else {
            format!("特征_{}", feature_index)
        };
        
        let split_type = self.split_type.unwrap();
        
        match split_type {
            SplitType::Numeric => {
                let threshold = self.threshold.unwrap();
                let feature_value = features[feature_index];
                
                if feature_value <= threshold {
                    path.push(format!("{} <= {:.4}", feature_name, threshold));
                    let mut sub_path = self.left.as_ref().unwrap().get_decision_path(features, feature_names);
                    path.append(&mut sub_path);
                } else {
                    path.push(format!("{} > {:.4}", feature_name, threshold));
                    let mut sub_path = self.right.as_ref().unwrap().get_decision_path(features, feature_names);
                    path.append(&mut sub_path);
                }
            },
            SplitType::Categorical => {
                let feature_value = features[feature_index] as usize;
                
                if let Some(cats) = &self.categories {
                    if cats.contains(&feature_value.to_string()) {
                        path.push(format!("{} in {:?}", feature_name, cats));
                        let mut sub_path = self.left.as_ref().unwrap().get_decision_path(features, feature_names);
                        path.append(&mut sub_path);
                    } else {
                        path.push(format!("{} not in {:?}", feature_name, cats));
                        let mut sub_path = self.right.as_ref().unwrap().get_decision_path(features, feature_names);
                        path.append(&mut sub_path);
                    }
                } else {
                    path.push(format!("{} = ?", feature_name));
                    let mut sub_path = self.left.as_ref().unwrap().get_decision_path(features, feature_names);
                    path.append(&mut sub_path);
                }
            },
        }
        
        path
    }
    
    /// 计算节点数量
    pub fn count_nodes(&self) -> usize {
        let mut count = 1;
        
        if let Some(left) = &self.left {
            count += left.count_nodes();
        }
        
        if let Some(right) = &self.right {
            count += right.count_nodes();
        }
        
        count
    }
    
    /// 计算叶节点数量
    pub fn count_leaves(&self) -> usize {
        if self.is_leaf {
            return 1;
        }
        
        let mut count = 0;
        
        if let Some(left) = &self.left {
            count += left.count_leaves();
        }
        
        if let Some(right) = &self.right {
            count += right.count_leaves();
        }
        
        count
    }
    
    /// 计算最大深度
    pub fn max_depth(&self) -> usize {
        if self.is_leaf {
            return self.depth;
        }
        
        let left_depth = if let Some(left) = &self.left {
            left.max_depth()
        } else {
            self.depth
        };
        
        let right_depth = if let Some(right) = &self.right {
            right.max_depth()
        } else {
            self.depth
        };
        
        std::cmp::max(left_depth, right_depth)
    }
}

/// 树类型
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum TreeType {
    /// 分类树
    Classification,
    /// 回归树
    Regression,
}

impl fmt::Display for TreeType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TreeType::Classification => write!(f, "classification"),
            TreeType::Regression => write!(f, "regression"),
        }
    }
}

/// 划分标准
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum CriterionType {
    /// 基尼不纯度
    Gini,
    /// 信息增益
    Entropy,
    /// 均方误差
    MSE,
    /// 平均绝对误差
    MAE,
}

impl fmt::Display for CriterionType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CriterionType::Gini => write!(f, "gini"),
            CriterionType::Entropy => write!(f, "entropy"),
            CriterionType::MSE => write!(f, "mse"),
            CriterionType::MAE => write!(f, "mae"),
        }
    }
}

/// 决策树参数
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DecisionTreeParams {
    /// 树类型
    pub tree_type: TreeType,
    /// 根节点
    pub root: Option<TreeNode>,
    /// 特征名称
    pub feature_names: Vec<String>,
    /// 类别名称 (分类树)
    pub class_names: Option<Vec<String>>,
    /// 划分标准
    pub criterion: CriterionType,
    /// 最大深度
    pub max_depth: Option<usize>,
    /// 最小样本数量
    pub min_samples_split: usize,
    /// 最小叶节点样本数量
    pub min_samples_leaf: usize,
    /// 最大特征数量
    pub max_features: Option<usize>,
    /// 随机种子
    pub random_state: Option<u64>,
    /// 类别权重 (分类树)
    pub class_weight: Option<HashMap<String, f32>>,
    /// 特征重要性
    pub feature_importance: Option<Vec<f32>>,
    /// 节点数量
    pub node_count: usize,
    /// 叶节点数量
    pub leaf_count: usize,
    /// 最大深度
    pub tree_depth: usize,
}

impl Default for DecisionTreeParams {
    fn default() -> Self {
        Self {
            tree_type: TreeType::Regression,
            root: None,
            feature_names: Vec::new(),
            class_names: None,
            criterion: CriterionType::MSE,
            max_depth: None,
            min_samples_split: 2,
            min_samples_leaf: 1,
            max_features: None,
            random_state: None,
            class_weight: None,
            feature_importance: None,
            node_count: 0,
            leaf_count: 0,
            tree_depth: 0,
        }
    }
}

impl DecisionTreeParams {
    /// 创建新的决策树参数
    pub fn new(tree_type: TreeType) -> Self {
        Self {
            tree_type,
            criterion: match tree_type {
                TreeType::Classification => CriterionType::Gini,
                TreeType::Regression => CriterionType::MSE,
            },
            ..Default::default()
        }
    }
    
    /// 设置特征名称
    pub fn with_feature_names(mut self, feature_names: Vec<String>) -> Self {
        self.feature_names = feature_names;
        self
    }
    
    /// 设置类别名称
    pub fn with_class_names(mut self, class_names: Vec<String>) -> Self {
        self.class_names = Some(class_names);
        self
    }
    
    /// 设置划分标准
    pub fn with_criterion(mut self, criterion: CriterionType) -> Self {
        self.criterion = criterion;
        self
    }
    
    /// 设置最大深度
    pub fn with_max_depth(mut self, max_depth: usize) -> Self {
        self.max_depth = Some(max_depth);
        self
    }
    
    /// 设置根节点
    pub fn with_root(mut self, root: TreeNode) -> Self {
        self.root = Some(root);
        self.update_stats();
        self
    }
    
    /// 更新统计信息
    pub fn update_stats(&mut self) {
        if let Some(root) = &self.root {
            self.node_count = root.count_nodes();
            self.leaf_count = root.count_leaves();
            self.tree_depth = root.max_depth();
        } else {
            self.node_count = 0;
            self.leaf_count = 0;
            self.tree_depth = 0;
        }
    }
    
    /// 预测单个样本
    pub fn predict(&self, features: &[f32]) -> Result<f32> {
        if features.len() != self.feature_names.len() {
            return Err(Error::invalid_input(format!(
                "特征数量不匹配: 期望 {}, 实际 {}",
                self.feature_names.len(),
                features.len()
            )));
        }
        
        if let Some(root) = &self.root {
            Ok(root.predict(features, &self.feature_names))
        } else {
            Err(Error::invalid_state("决策树未训练"))
        }
    }
    
    /// 批量预测
    pub fn predict_batch(&self, features_batch: &[Vec<f32>]) -> Result<Vec<f32>> {
        let mut predictions = Vec::with_capacity(features_batch.len());
        
        for features in features_batch {
            predictions.push(self.predict(features)?);
        }
        
        Ok(predictions)
    }
    
    /// 获取决策路径
    pub fn get_decision_path(&self, features: &[f32]) -> Result<Vec<String>> {
        if features.len() != self.feature_names.len() {
            return Err(Error::invalid_input(format!(
                "特征数量不匹配: 期望 {}, 实际 {}",
                self.feature_names.len(),
                features.len()
            )));
        }
        
        if let Some(root) = &self.root {
            Ok(root.get_decision_path(features, &self.feature_names))
        } else {
            Err(Error::invalid_state("决策树未训练"))
        }
    }
    
    /// 获取参数映射
    pub fn get_params_map(&self) -> HashMap<String, String> {
        let mut params = HashMap::new();
        
        params.insert("tree_type".to_string(), self.tree_type.to_string());
        params.insert("criterion".to_string(), self.criterion.to_string());
        params.insert("min_samples_split".to_string(), self.min_samples_split.to_string());
        params.insert("min_samples_leaf".to_string(), self.min_samples_leaf.to_string());
        
        if let Some(max_depth) = self.max_depth {
            params.insert("max_depth".to_string(), max_depth.to_string());
        }
        
        if let Some(max_features) = self.max_features {
            params.insert("max_features".to_string(), max_features.to_string());
        }
        
        if let Some(random_state) = self.random_state {
            params.insert("random_state".to_string(), random_state.to_string());
        }
        
        // 添加树结构信息
        params.insert("node_count".to_string(), self.node_count.to_string());
        params.insert("leaf_count".to_string(), self.leaf_count.to_string());
        params.insert("tree_depth".to_string(), self.tree_depth.to_string());
        
        // 添加特征重要性
        if let Some(importances) = &self.feature_importance {
            for (i, &importance) in importances.iter().enumerate() {
                let feature_name = if i < self.feature_names.len() {
                    &self.feature_names[i]
                } else {
                    "unknown"
                };
                
                params.insert(
                    format!("importance_{}", feature_name),
                    format!("{:.6}", importance)
                );
            }
        }
        
        params
    }
    
    /// 导出为DOT格式 (用于Graphviz可视化)
    pub fn export_graphviz(&self) -> Result<String> {
        let mut result = String::new();
        result.push_str("digraph DecisionTree {\n");
        result.push_str("  node [shape=box, style=\"filled, rounded\", color=\"black\", fontname=helvetica];\n");
        result.push_str("  edge [fontname=helvetica];\n");
        
        if let Some(root) = &self.root {
            self.node_to_dot(root, &mut result, 0)?;
        } else {
            return Err(Error::invalid_state("决策树未训练"));
        }
        
        result.push_str("}\n");
        Ok(result)
    }
    
    /// 将节点转换为DOT格式
    fn node_to_dot(&self, node: &TreeNode, result: &mut String, node_id: usize) -> Result<usize> {
        let mut next_id = node_id + 1;
        
        if node.is_leaf {
            // 叶节点
            let value = node.value.unwrap_or(0.0);
            let label = format!("预测值: {:.4}", value);
            let color = format!("0.0 0.0 {:.3}", 0.5 + (0.5 * (node.depth as f32 / self.tree_depth as f32)));
            
            result.push_str(&format!(
                "  {} [label=\"{}\", fillcolor=\"{}\"];\n",
                node_id, label, color
            ));
        } else {
            // 内部节点
            let feature_index = node.feature_index.unwrap();
            let feature_name = if let Some(name) = &node.feature_name {
                name.clone()
            } else if feature_index < self.feature_names.len() {
                self.feature_names[feature_index].clone()
            } else {
                format!("特征_{}", feature_index)
            };
            
            let split_type = node.split_type.unwrap();
            let split_condition = match split_type {
                SplitType::Numeric => {
                    let threshold = node.threshold.unwrap();
                    format!("{} <= {:.4}\n不纯度: {:.4}\n样本数: {}", feature_name, threshold, node.impurity, node.samples)
                },
                SplitType::Categorical => {
                    let empty_vec = Vec::new();
                    let categories = node.categories.as_ref().unwrap_or(&empty_vec);
                    format!("{} in {:?}\n不纯度: {:.4}\n样本数: {}", feature_name, categories, node.impurity, node.samples)
                },
            };
            
            let color = format!("0.0 0.0 {:.3}", 0.9 - (0.5 * (node.depth as f32 / self.tree_depth as f32)));
            
            result.push_str(&format!(
                "  {} [label=\"{}\", fillcolor=\"{}\"];\n",
                node_id, split_condition, color
            ));
            
            // 左子节点
            if let Some(left) = &node.left {
                result.push_str(&format!("  {} -> {} [labeldistance=2.5, labelangle=45, headlabel=\"True\"];\n", node_id, next_id));
                next_id = self.node_to_dot(left, result, next_id)?;
            }
            
            // 右子节点
            if let Some(right) = &node.right {
                result.push_str(&format!("  {} -> {} [labeldistance=2.5, labelangle=-45, headlabel=\"False\"];\n", node_id, next_id));
                next_id = self.node_to_dot(right, result, next_id)?;
            }
        }
        
        Ok(next_id)
    }
}

/// 决策树算法实现
#[derive(Debug)]
pub struct DecisionTree {
    /// 算法ID
    id: String,
    /// 算法名称
    name: String,
    /// 参数
    params: DecisionTreeParams,
    /// 算法类型
    algorithm_type: AlgorithmType,
    /// 元数据
    metadata: HashMap<String, String>,
    /// 依赖
    dependencies: Vec<String>,
    /// 创建时间
    created_at: i64,
    /// 更新时间
    updated_at: i64,
}

impl DecisionTree {
    /// 创建新的决策树算法
    pub fn new(tree_type: TreeType) -> Self {
        let now = chrono::Utc::now().timestamp();
        Self {
            id: Uuid::new_v4().to_string(),
            name: format!("DecisionTree{}", tree_type),
            params: DecisionTreeParams::new(tree_type),
            algorithm_type: AlgorithmType::MachineLearning,
            metadata: HashMap::new(),
            dependencies: vec![
                "statistical_functions".to_string(),
                "tree_operations".to_string(),
                "splitting_algorithms".to_string(),
            ],
            created_at: now,
            updated_at: now,
        }
    }
    
    /// 从参数创建决策树
    pub fn from_params(params: DecisionTreeParams) -> Self {
        let now = chrono::Utc::now().timestamp();
        Self {
            id: Uuid::new_v4().to_string(),
            name: format!("DecisionTree{}", params.tree_type),
            params,
            algorithm_type: AlgorithmType::MachineLearning,
            metadata: HashMap::new(),
            dependencies: vec![
                "statistical_functions".to_string(),
                "tree_operations".to_string(),
                "splitting_algorithms".to_string(),
            ],
            created_at: now,
            updated_at: now,
        }
    }
    
    /// 获取参数
    pub fn get_params(&self) -> &DecisionTreeParams {
        &self.params
    }
    
    /// 获取可变参数引用
    pub fn get_params_mut(&mut self) -> &mut DecisionTreeParams {
        &mut self.params
    }
    
    /// 设置特征名称
    pub fn set_feature_names(&mut self, feature_names: Vec<String>) {
        self.params.feature_names = feature_names;
    }
    
    /// 预测
    pub fn predict(&self, features: &[f32]) -> Result<f32> {
        self.params.predict(features)
    }
    
    /// 批量预测
    pub fn predict_batch(&self, features_batch: &[Vec<f32>]) -> Result<Vec<f32>> {
        self.params.predict_batch(features_batch)
    }
    
    /// 获取元数据
    pub fn get_metadata(&self) -> HashMap<String, String> {
        let mut metadata = HashMap::new();
        metadata.insert("algorithm_type".to_string(), "decision_tree".to_string());
        metadata.insert("tree_type".to_string(), format!("{:?}", self.params.tree_type));
        metadata.insert("criterion".to_string(), format!("{:?}", self.params.criterion));
        metadata.insert("node_count".to_string(), self.params.node_count.to_string());
        metadata.insert("leaf_count".to_string(), self.params.leaf_count.to_string());
        metadata.insert("tree_depth".to_string(), self.params.tree_depth.to_string());
        
        metadata
    }
    
    /// 更新元数据
    fn update_metadata(&self) {
        let mut metadata = HashMap::new();
        
        // 添加算法特定的元数据
        metadata.insert("algorithm_type".to_string(), "decision_tree".to_string());
        metadata.insert("tree_type".to_string(), format!("{:?}", self.params.tree_type));
        metadata.insert("splitting_criterion".to_string(), format!("{:?}", self.params.criterion));
        metadata.insert("max_depth".to_string(), 
            self.params.max_depth.map_or("unlimited".to_string(), |d| d.to_string()));
        metadata.insert("min_samples_split".to_string(), self.params.min_samples_split.to_string());
        metadata.insert("min_samples_leaf".to_string(), self.params.min_samples_leaf.to_string());
        
        // 树结构信息
        metadata.insert("tree_depth".to_string(), self.params.tree_depth.to_string());
        metadata.insert("leaf_count".to_string(), self.params.leaf_count.to_string());
        metadata.insert("node_count".to_string(), self.params.node_count.to_string());
        
        // 训练信息
        metadata.insert("is_trained".to_string(), self.params.root.is_some().to_string());
        metadata.insert("feature_count".to_string(), self.params.feature_names.len().to_string());
        
        // 特征重要性（如果可用）
        if let Some(ref importance) = self.params.feature_importance {
            for (i, &imp) in importance.iter().enumerate() {
                if i < self.params.feature_names.len() {
                    metadata.insert(
                        format!("feature_importance_{}", self.params.feature_names[i]), 
                        imp.to_string()
                    );
                }
            }
        }
        
        // 使用unsafe来更新self.metadata，因为我们在不可变引用中
        unsafe {
            let metadata_ref = &mut *(self.metadata.as_ptr() as *mut HashMap<String, String>);
            *metadata_ref = metadata;
        }
    }
    
    /// 更新依赖列表
    fn update_dependencies(&self) {
        let mut deps = vec![
            "statistical_functions".to_string(),
            "tree_operations".to_string(),
            "splitting_algorithms".to_string(),
        ];
        
        // 根据分割准则添加依赖
        match self.params.criterion {
            CriterionType::Gini => deps.push("gini_impurity".to_string()),
            CriterionType::Entropy => deps.push("entropy_calculation".to_string()),
            CriterionType::MSE => deps.push("mse_calculation".to_string()),
            CriterionType::MAE => deps.push("mae_calculation".to_string()),
        }
        
        // 根据树类型添加依赖
        match self.params.tree_type {
            TreeType::Classification => {
                deps.push("classification_metrics".to_string());
                deps.push("class_probability".to_string());
            },
            TreeType::Regression => {
                deps.push("regression_metrics".to_string());
                deps.push("continuous_prediction".to_string());
            },
        }
        
        // 使用unsafe来更新self.dependencies
        unsafe {
            let deps_ref = &mut *(self.dependencies.as_ptr() as *mut Vec<String>);
            *deps_ref = deps;
        }
    }
    
    /// 计算预测置信度
    pub fn calculate_prediction_confidence(&self, features: &[f32]) -> Result<f32> {
        if let Some(ref root) = self.params.root {
            let mut current = root;
            
            // 遍历到叶节点
            while !current.is_leaf {
                let feature_idx = current.feature_index.ok_or_else(|| 
                    Error::algorithm_error("Internal node missing feature index"))?;
                
                if feature_idx >= features.len() {
                    return Err(Error::algorithm_error("Feature index out of bounds"));
                }
                
                let feature_value = features[feature_idx];
                
                match current.split_type {
                    Some(SplitType::Numeric) => {
                        let threshold = current.threshold.ok_or_else(|| 
                            Error::algorithm_error("Numeric split missing threshold"))?;
                        current = if feature_value <= threshold {
                            current.left.as_ref()
                        } else {
                            current.right.as_ref()
                        }.ok_or_else(|| Error::algorithm_error("Missing child node"))?;
                    },
                    Some(SplitType::Categorical) => {
                        // 简化的类别处理
                        current = current.left.as_ref().ok_or_else(|| 
                            Error::algorithm_error("Missing left child for categorical split"))?;
                    },
                    None => return Err(Error::algorithm_error("Node missing split type")),
                }
            }
            
            // 基于样本数量和不纯度计算置信度
            let confidence = match self.params.tree_type {
                TreeType::Classification => {
                    // 对于分类，基于类别分布计算置信度
                    if let Some(ref class_dist) = current.class_distribution {
                        let total_samples = class_dist.values().sum::<usize>() as f32;
                        let max_class_count = *class_dist.values().max().unwrap_or(&0) as f32;
                        max_class_count / total_samples
                    } else {
                        0.5 // 默认置信度
                    }
                },
                TreeType::Regression => {
                    // 对于回归，基于不纯度计算置信度
                    1.0 - current.impurity.min(1.0)
                },
            };
            
            Ok(confidence)
        } else {
            Err(Error::algorithm_error("Tree not trained"))
        }
    }
    
    /// 获取决策路径
    pub fn get_decision_path(&self, features: &[f32]) -> Result<Vec<String>> {
        if let Some(ref root) = self.params.root {
            Ok(root.get_decision_path(features, &self.params.feature_names))
        } else {
            Err(Error::algorithm_error("Tree not trained"))
        }
    }
    
    /// 获取特征重要性
    pub fn get_feature_importance(&self) -> Vec<f32> {
        self.params.feature_importance.clone().unwrap_or_else(|| 
            vec![0.0; self.params.feature_names.len()]
        )
    }
}

/// 决策树预测结果结构
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct TreePredictionResult {
    /// 预测值
    pub prediction: f32,
    /// 预测置信度
    pub confidence: f32,
    /// 决策路径
    pub path: Vec<String>,
    /// 特征重要性
    pub feature_importance: Vec<f32>,
}

impl Algorithm for DecisionTree {
    fn get_id(&self) -> &str {
        &self.id
    }
    
    fn get_name(&self) -> &str {
        &self.name
    }
    
    fn get_description(&self) -> Option<&str> {
        Some("High-performance decision tree implementation supporting both classification and regression with configurable splitting criteria, pruning strategies, and advanced tree optimization techniques.")
    }
    
    fn get_type(&self) -> AlgorithmType {
        match self.params.tree_type {
            TreeType::Classification => AlgorithmType::Classification,
            TreeType::Regression => AlgorithmType::Regression,
        }
    }
    
    fn get_code(&self) -> &str {
        match self.params.tree_type {
            TreeType::Classification => "decision_tree_classifier",
            TreeType::Regression => "decision_tree_regressor",
        }
    }
    
    fn get_version(&self) -> u32 {
        1
    }
    
    fn execute(&self, input: &[u8]) -> Result<Vec<u8>> {
        // 将输入字节转换为特征向量
        let features_data: Vec<f32> = bincode::deserialize(input)
            .map_err(|e| Error::serialization(&format!("Failed to deserialize input: {}", e)))?;
        
        // 执行预测
        let prediction = self.predict(&features_data)
            .map_err(|e| Error::prediction_error(&format!("Decision tree prediction failed: {}", e)))?;
        
        // 创建预测结果结构
        let prediction_result = TreePredictionResult {
            prediction,
            confidence: self.calculate_prediction_confidence(&features_data)?,
            path: self.get_decision_path(&features_data)?,
            feature_importance: self.get_feature_importance(),
        };
        
        // 序列化预测结果
        bincode::serialize(&prediction_result)
            .map_err(|e| Error::serialization(&format!("Failed to serialize prediction result: {}", e)))
    }
    
    fn get_algorithm_type(&self) -> &AlgorithmType {
        &self.algorithm_type
    }
    
    fn get_metadata(&self) -> &HashMap<String, String> {
        // 动态构建元数据，确保返回最新的信息
        self.update_metadata();
        &self.metadata
    }
    
    fn get_dependencies(&self) -> &[String] {
        // 动态构建依赖列表，确保返回最新的信息
        self.update_dependencies();
        &self.dependencies
    }
    
    fn get_created_at(&self) -> i64 {
        // 从元数据中获取创建时间，或使用当前时间
        if let Some(created_at_str) = self.params.metadata.get("created_at") {
            if let Ok(timestamp) = created_at_str.parse::<i64>() {
                return timestamp;
            }
        }
        self.created_at
    }
    
    fn get_updated_at(&self) -> i64 {
        // 从元数据中获取更新时间，或使用当前时间
        if let Some(updated_at_str) = self.params.metadata.get("updated_at") {
            if let Ok(timestamp) = updated_at_str.parse::<i64>() {
                return timestamp;
            }
        }
        self.updated_at
    }
    
    fn get_config(&self) -> HashMap<String, String> {
        self.params.get_params_map()
    }
    
    fn set_config(&mut self, config: HashMap<String, String>) {
        if let Some(min_samples_split) = config.get("min_samples_split") {
            if let Ok(mss) = min_samples_split.parse::<usize>() {
                self.params.min_samples_split = mss;
            }
        }
        
        if let Some(min_samples_leaf) = config.get("min_samples_leaf") {
            if let Ok(msl) = min_samples_leaf.parse::<usize>() {
                self.params.min_samples_leaf = msl;
            }
        }
        
        if let Some(max_depth) = config.get("max_depth") {
            if let Ok(md) = max_depth.parse::<usize>() {
                self.params.max_depth = Some(md);
            }
        }
        
        if let Some(max_features) = config.get("max_features") {
            if let Ok(mf) = max_features.parse::<usize>() {
                self.params.max_features = Some(mf);
            }
        }
        
        if let Some(random_state) = config.get("random_state") {
            if let Ok(rs) = random_state.parse::<u64>() {
                self.params.random_state = Some(rs);
            }
        }
    }
    
    fn apply(&self, params: &HashMap<String, String>) -> Result<serde_json::Value> {
        // 解析特征值
        let mut features = Vec::new();
        for feature_name in &self.params.feature_names {
            if let Some(value) = params.get(feature_name) {
                match value.parse::<f32>() {
                    Ok(val) => features.push(val),
                    Err(_) => return Err(Error::invalid_input(format!("无法解析特征 {}: {}", feature_name, value))),
                }
            } else {
                return Err(Error::invalid_input(format!("缺少特征: {}", feature_name)));
            }
        }
        
        // 预测
        let prediction = self.predict(&features)?;
        
        // 获取决策路径
        let decision_path = self.params.get_decision_path(&features)?;
        
        // 返回结果
        let mut result = serde_json::Map::new();
        
        // 分类树返回类别名称，回归树返回数值
        if let TreeType::Classification = self.params.tree_type {
            if let Some(class_names) = &self.params.class_names {
                // 找到最接近的类别索引
                let class_index = prediction.round() as usize;
                if class_index < class_names.len() {
                    result.insert("class".to_string(), serde_json::Value::String(class_names[class_index].clone()));
                }
            }
        }
        
        // 添加预测值
        result.insert(
            "prediction".to_string(), 
            serde_json::Value::Number(serde_json::Number::from_f64(prediction as f64).unwrap())
        );
        
        // 添加决策路径
        let path_array = decision_path.into_iter()
            .map(|s| serde_json::Value::String(s))
            .collect::<Vec<_>>();
        
        result.insert("decision_path".to_string(), serde_json::Value::Array(path_array));
        
        Ok(serde_json::Value::Object(result))
    }
    
    fn validate(&self) -> Result<()> {
        // 检查特征名称
        if self.params.feature_names.is_empty() {
            return Err(Error::invalid_state("未设置特征名称"));
        }
        
        // 检查类别名称 (分类树)
        if let TreeType::Classification = self.params.tree_type {
            if self.params.class_names.is_none() || self.params.class_names.as_ref().unwrap().is_empty() {
                return Err(Error::invalid_state("分类树未设置类别名称"));
            }
        }
        
        // 检查树是否已训练
        if self.params.root.is_none() {
            return Err(Error::invalid_state("决策树未训练"));
        }
        
        // 检查超参数
        if self.params.min_samples_split < 2 {
            return Err(Error::invalid_state(format!("min_samples_split必须至少为2: {}", self.params.min_samples_split)));
        }
        
        if self.params.min_samples_leaf < 1 {
            return Err(Error::invalid_state(format!("min_samples_leaf必须至少为1: {}", self.params.min_samples_leaf)));
        }
        
        if let Some(max_features) = self.params.max_features {
            if max_features == 0 || max_features > self.params.feature_names.len() {
                return Err(Error::invalid_state(format!(
                    "max_features必须在1和特征数量之间: {}, 特征数量: {}",
                    max_features,
                    self.params.feature_names.len()
                )));
            }
        }
        
        Ok(())
    }
    
    fn get_params(&self) -> Result<serde_json::Value> {
        Ok(serde_json::to_value(&self.params)?)
    }
} 