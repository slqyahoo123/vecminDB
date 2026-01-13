/// 分类算法模块
/// 
/// 提供分类算法的执行实现，支持多种分类模型类型

use crate::{Result, Error};
use crate::core::types::CoreAlgorithmDefinition;
use crate::core::UnifiedDataType;
use log::warn;

use super::model_parsing::parse_model_parameters;
use super::neural_network::execute_neural_network;
use super::decision_tree::execute_decision_tree;
use super::clustering::{kmeans_clustering};
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

/// 执行分类算法
pub(crate) async fn execute_classification(
    algo_def: &AlgorithmDefinition, 
    inputs: &[crate::core::UnifiedTensorData]
) -> Result<Vec<crate::core::UnifiedTensorData>> {
    if inputs.is_empty() {
        return Err(Error::InvalidInput("分类算法需要至少一个输入张量".to_string()));
    }
    
    let input = &inputs[0];
    let n_samples = input.shape[0];
    let n_features = calculate_n_features(input, n_samples)?;
    
    // 解析模型参数
    let params = parse_model_parameters(algo_def)?;
    
    // 检查模型类型（从metadata或parameters）
    let model_type = algo_def.metadata.get("model_type")
        .or_else(|| algo_def.parameters.iter()
            .find(|p| p.name == "model_type")
            .and_then(|p| p.default_value.as_ref()))
        .map(|s| s.as_str())
        .unwrap_or("linear");
    
    match model_type {
        "linear" | "logistic" => {
            // 线性分类器：使用权重和偏置
            if let (Some(weights), Some(bias)) = (params.weights, params.bias) {
                if weights.len() != n_features {
                    return Err(Error::InvalidInput(
                        format!("权重数量({})与特征数量({})不匹配", weights.len(), n_features)
                    ));
                }
                
                let mut predictions = Vec::with_capacity(n_samples);
                for i in 0..n_samples {
                    let mut score = bias;
                    for j in 0..n_features {
                        score += input.data[i * n_features + j] * weights[j];
                    }
                    
                    // 对于逻辑回归，应用sigmoid；对于线性分类，直接使用符号
                    if model_type == "logistic" {
                        score = 1.0 / (1.0 + (-score).exp());
                    }
                    
                    predictions.push(score);
                }
                
                return Ok(vec![crate::core::UnifiedTensorData {
                    shape: vec![n_samples],
                    data: predictions,
                    dtype: UnifiedDataType::Float32,
                    device: "cpu".to_string(),
                }]);
            }
        },
        "neural_network" => {
            // 神经网络分类器：使用神经网络前向传播
            return execute_neural_network(algo_def, inputs).await;
        },
        "decision_tree" => {
            // 决策树分类器：使用决策树预测
            return execute_decision_tree(algo_def, inputs).await;
        },
        _ => {
            // 未知模型类型，尝试使用线性分类器
            warn!("未知的分类模型类型: {}，使用线性分类器", model_type);
            if let (Some(weights), Some(bias)) = (params.weights, params.bias) {
                if weights.len() != n_features {
                    return Err(Error::InvalidInput(
                        format!("权重数量({})与特征数量({})不匹配", weights.len(), n_features)
                    ));
                }
                
                let mut predictions = Vec::with_capacity(n_samples);
                for i in 0..n_samples {
                    let mut score = bias;
                    for j in 0..n_features {
                        score += input.data[i * n_features + j] * weights[j];
                    }
                    predictions.push(score);
                }
                
                return Ok(vec![crate::core::UnifiedTensorData {
                    shape: vec![n_samples],
                    data: predictions,
                    dtype: UnifiedDataType::Float32,
                    device: "cpu".to_string(),
                }]);
            }
        }
    }
    
    // 如果没有模型参数，返回错误
    Err(Error::InvalidInput(
        format!("分类算法需要模型参数（模型类型: {}）", model_type)
    ))
}

/// 执行分类算法（CoreTensorData版本）
pub(crate) async fn execute_classification_core(
    algo_def: &AlgorithmDefinition,
    inputs: &[crate::core::types::CoreTensorData]
) -> Result<Vec<crate::core::types::CoreTensorData>> {
    // 转换为UnifiedTensorData
    let unified_inputs: Vec<crate::core::UnifiedTensorData> = inputs.iter()
        .map(|t| convert_core_to_unified(t))
        .collect();
    
    // 执行算法
    let unified_outputs = execute_classification(algo_def, &unified_inputs).await?;
    
    // 转换回CoreTensorData
    let core_outputs: Vec<crate::core::types::CoreTensorData> = unified_outputs.iter()
        .map(|t| convert_unified_to_core(t))
        .collect();
    
    Ok(core_outputs)
}

/// 执行聚类算法（包装函数，供主文件调用）
pub(crate) async fn execute_clustering(
    algo_def: &AlgorithmDefinition,
    inputs: &[crate::core::UnifiedTensorData]
) -> Result<Vec<crate::core::UnifiedTensorData>> {
    if inputs.is_empty() {
        return Err(Error::InvalidInput("聚类算法需要至少一个输入张量".to_string()));
    }
    
    let input = &inputs[0];
    let n_samples = input.shape[0];
    let n_features = calculate_n_features(input, n_samples)?;
    
    // 解析模型参数
    let params = parse_model_parameters(algo_def)?;
    
    // 获取K值（聚类数量）
    let k = params.k_value.unwrap_or_else(|| {
        // 使用启发式方法：K = sqrt(n_samples / 2)
        let k_heuristic = (n_samples as f32 / 2.0).sqrt() as usize;
        k_heuristic.max(2).min(n_samples)
    });
    
    if k > n_samples {
        return Err(Error::InvalidInput(
            format!("聚类数K({})不能大于样本数({})", k, n_samples)
        ));
    }
    
    // 执行K-means聚类
    let labels = kmeans_clustering(
        &input.data,
        n_samples,
        n_features,
        k,
        100 // 最大迭代次数
    )?;
    
    Ok(vec![crate::core::UnifiedTensorData {
        shape: vec![n_samples],
        data: labels.into_iter().map(|x| x as f32).collect(),
        dtype: UnifiedDataType::Float32,
        device: "cpu".to_string(),
    }])
}

/// 执行聚类算法（CoreTensorData版本）
pub(crate) async fn execute_clustering_core(
    algo_def: &AlgorithmDefinition,
    inputs: &[crate::core::types::CoreTensorData]
) -> Result<Vec<crate::core::types::CoreTensorData>> {
    // 转换为UnifiedTensorData
    let unified_inputs: Vec<crate::core::UnifiedTensorData> = inputs.iter()
        .map(|t| convert_core_to_unified(t))
        .collect();
    
    // 执行算法
    let unified_outputs = execute_clustering(algo_def, &unified_inputs).await?;
    
    // 转换回CoreTensorData
    let core_outputs: Vec<crate::core::types::CoreTensorData> = unified_outputs.iter()
        .map(|t| convert_unified_to_core(t))
        .collect();
    
    Ok(core_outputs)
}

