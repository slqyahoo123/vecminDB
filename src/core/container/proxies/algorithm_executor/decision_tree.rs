/// 决策树算法模块
/// 
/// 提供决策树预测算法的执行实现

use crate::{Result, Error};
use crate::core::types::CoreAlgorithmDefinition;
use crate::core::UnifiedDataType;

use super::model_parsing::parse_model_parameters;
use super::conversions::{convert_core_to_unified, convert_unified_to_core};

// 类型别名
type AlgorithmDefinition = CoreAlgorithmDefinition;

/// 计算特征数量（辅助函数）
fn calculate_n_features(input: &crate::core::UnifiedTensorData, n_samples: usize) -> Result<usize> {
    if n_samples == 0 {
        return Err(Error::InvalidInput("输入样本数不能为0".to_string()));
    }
    
    if input.shape.len() > 1 {
        Ok(input.shape[1])
    } else {
        // 从数据大小推断特征数量
        if input.data.len() % n_samples != 0 {
            return Err(Error::InvalidInput(
                format!("数据大小({})不能被样本数({})整除", input.data.len(), n_samples)
            ));
        }
        Ok(input.data.len() / n_samples)
    }
}

/// 执行决策树算法
pub(crate) async fn execute_decision_tree(
    algo_def: &AlgorithmDefinition, 
    inputs: &[crate::core::UnifiedTensorData]
) -> Result<Vec<crate::core::UnifiedTensorData>> {
    if inputs.is_empty() {
        return Err(Error::InvalidInput("决策树需要至少一个输入张量".to_string()));
    }
    
    let input = &inputs[0];
    let n_samples = input.shape[0];
    let n_features = calculate_n_features(input, n_samples)?;
    
    // 解析模型参数
    let params = parse_model_parameters(algo_def)?;
    
    // 如果有树结构，使用树进行预测
    if let Some(tree) = params.tree_structure {
        if tree.nodes.is_empty() {
            return Err(Error::InvalidInput("决策树结构为空".to_string()));
        }
        
        let mut predictions = Vec::with_capacity(n_samples);
        const MAX_TREE_DEPTH: usize = 1000; // 防止无限循环
        
        for i in 0..n_samples {
            // 从根节点开始遍历
            let mut node_idx = 0;
            let mut depth = 0;
            
            loop {
                if depth > MAX_TREE_DEPTH {
                    return Err(Error::InvalidInput(
                        format!("决策树遍历深度超过限制({})，可能存在循环引用", MAX_TREE_DEPTH)
                    ));
                }
                
                if node_idx >= tree.nodes.len() {
                    return Err(Error::InvalidInput(
                        format!("决策树节点索引越界: {}", node_idx)
                    ));
                }
                
                depth += 1;
                
                let node = &tree.nodes[node_idx];
                
                // 如果是叶子节点，返回预测值
                if let Some(value) = node.value {
                    predictions.push(value);
                    break;
                }
                
                // 根据特征值选择左子树或右子树
                let feature_value = if node.feature_index < n_features {
                    input.data[i * n_features + node.feature_index]
                } else {
                    return Err(Error::InvalidInput(
                        format!("特征索引越界: {} >= {}", node.feature_index, n_features)
                    ));
                };
                
                if feature_value <= node.threshold {
                    if let Some(left) = node.left_child {
                        node_idx = left;
                    } else {
                        return Err(Error::InvalidInput(
                            format!("决策树节点{}缺少左子树", node_idx)
                        ));
                    }
                } else {
                    if let Some(right) = node.right_child {
                        node_idx = right;
                    } else {
                        return Err(Error::InvalidInput(
                            format!("决策树节点{}缺少右子树", node_idx)
                        ));
                    }
                }
            }
        }
        
        return Ok(vec![crate::core::UnifiedTensorData {
            shape: vec![n_samples],
            data: predictions,
            dtype: UnifiedDataType::Float32,
            device: "cpu".to_string(),
        }]);
    }
    
    // 如果没有树结构，返回错误
    Err(Error::InvalidInput(
        "决策树需要树结构（节点、分裂条件、叶子值）".to_string()
    ))
}

/// 执行决策树算法（CoreTensorData版本）
pub(crate) async fn execute_decision_tree_core(
    algo_def: &AlgorithmDefinition,
    inputs: &[crate::core::types::CoreTensorData]
) -> Result<Vec<crate::core::types::CoreTensorData>> {
    // 转换为UnifiedTensorData
    let unified_inputs: Vec<crate::core::UnifiedTensorData> = inputs.iter()
        .map(|t| convert_core_to_unified(t))
        .collect();
    
    // 执行算法
    let unified_outputs = execute_decision_tree(algo_def, &unified_inputs).await?;
    
    // 转换回CoreTensorData
    let core_outputs: Vec<crate::core::types::CoreTensorData> = unified_outputs.iter()
        .map(|t| convert_unified_to_core(t))
        .collect();
    
    Ok(core_outputs)
}

