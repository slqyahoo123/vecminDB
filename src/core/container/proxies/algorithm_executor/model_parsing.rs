/// 模型参数解析模块
/// 
/// 提供从算法定义中解析模型参数的功能

use crate::{Result, Error};
use crate::core::types::CoreAlgorithmDefinition;
use super::types::{ModelParameters, NetworkConfig, LayerConfig, TreeStructure, TreeNode};

// 类型别名
type AlgorithmDefinition = CoreAlgorithmDefinition;

/// 从算法定义解析模型参数
pub(crate) fn parse_model_parameters(algo_def: &AlgorithmDefinition) -> Result<ModelParameters> {
    let mut params = ModelParameters {
        weights: None,
        bias: None,
        network_config: None,
        tree_structure: None,
        k_value: None,
    };

    // 1. 优先从 parameters 字段获取
    for param in &algo_def.parameters {
        match param.name.as_str() {
            "weights" | "model_weights" => {
                if let Some(default_val) = &param.default_value {
                    params.weights = parse_weights_from_string(default_val)?;
                }
            },
            "bias" | "model_bias" => {
                if let Some(default_val) = &param.default_value {
                    params.bias = default_val.parse::<f32>().ok();
                }
            },
            "k" | "n_clusters" => {
                if let Some(default_val) = &param.default_value {
                    params.k_value = default_val.parse::<usize>().ok();
                }
            },
            _ => {}
        }
    }

    // 2. 从 metadata 字段解析JSON格式
    for (key, value) in &algo_def.metadata {
        match key.as_str() {
            "model_weights" | "weights" => {
                if params.weights.is_none() {
                    params.weights = parse_weights_from_string(value)?;
                }
            },
            "model_bias" | "bias" => {
                if params.bias.is_none() {
                    params.bias = value.parse::<f32>().ok();
                }
            },
            "network_config" | "neural_network_config" => {
                params.network_config = parse_network_config(value)?;
            },
            "tree_structure" | "decision_tree" => {
                params.tree_structure = parse_tree_structure(value)?;
            },
            "k" | "n_clusters" | "num_clusters" => {
                if params.k_value.is_none() {
                    params.k_value = value.parse::<usize>().ok();
                }
            },
            _ => {}
        }
    }

    Ok(params)
}

/// 从字符串解析权重向量
pub(crate) fn parse_weights_from_string(s: &str) -> Result<Option<Vec<f32>>> {
    // 尝试解析为JSON数组
    if let Ok(json_value) = serde_json::from_str::<serde_json::Value>(s) {
        if let Some(arr) = json_value.as_array() {
            let weights: Result<Vec<f32>> = arr.iter()
                .map(|v| v.as_f64()
                    .ok_or_else(|| Error::InvalidInput("权重值必须是数字".to_string()))
                    .map(|f| f as f32))
                .collect();
            return Ok(Some(weights?));
        }
    }
    
    // 尝试解析为逗号分隔的字符串
    let weights: Result<Vec<f32>> = s.split(',')
        .map(|s| s.trim().parse::<f32>()
            .map_err(|_| Error::InvalidInput("无法解析权重值".to_string())))
        .collect();
    
    Ok(weights.ok())
}

/// 解析神经网络配置
pub(crate) fn parse_network_config(json_str: &str) -> Result<Option<NetworkConfig>> {
    let json: serde_json::Value = serde_json::from_str(json_str)
        .map_err(|e| Error::InvalidInput(format!("无法解析网络配置JSON: {}", e)))?;
    
    let layers_json = json.get("layers")
        .and_then(|v| v.as_array())
        .ok_or_else(|| Error::InvalidInput("网络配置缺少layers字段".to_string()))?;
    
    let mut layers = Vec::new();
    for layer_json in layers_json {
        let input_size = layer_json.get("input_size")
            .and_then(|v| v.as_u64())
            .map(|v| v as usize)
            .ok_or_else(|| Error::InvalidInput("层配置缺少input_size".to_string()))?;
        
        let output_size = layer_json.get("output_size")
            .and_then(|v| v.as_u64())
            .map(|v| v as usize)
            .ok_or_else(|| Error::InvalidInput("层配置缺少output_size".to_string()))?;
        
        let weights = layer_json.get("weights")
            .and_then(|v| v.as_array())
            .map(|arr| arr.iter()
                .filter_map(|v| v.as_f64().map(|f| f as f32))
                .collect::<Vec<f32>>())
            .ok_or_else(|| Error::InvalidInput("层配置缺少weights".to_string()))?;
        
        // 验证权重矩阵大小
        let expected_weights_size = input_size * output_size;
        if weights.len() != expected_weights_size {
            return Err(Error::InvalidInput(
                format!("权重矩阵大小不匹配: 期望 {} ({} x {}), 实际 {}", 
                       expected_weights_size, input_size, output_size, weights.len())
            ));
        }
        
        let bias = layer_json.get("bias")
            .and_then(|v| v.as_array())
            .map(|arr| arr.iter()
                .filter_map(|v| v.as_f64().map(|f| f as f32))
                .collect::<Vec<f32>>())
            .unwrap_or_else(|| vec![0.0; output_size]);
        
        // 验证偏置向量大小
        if bias.len() != output_size {
            return Err(Error::InvalidInput(
                format!("偏置向量大小不匹配: 期望 {}, 实际 {}", output_size, bias.len())
            ));
        }
        
        let activation = layer_json.get("activation")
            .and_then(|v| v.as_str())
            .unwrap_or("relu")
            .to_string();
        
        layers.push(LayerConfig {
            input_size,
            output_size,
            weights,
            bias,
            activation,
        });
    }
    
    Ok(Some(NetworkConfig { layers }))
}

/// 解析决策树结构
pub(crate) fn parse_tree_structure(json_str: &str) -> Result<Option<TreeStructure>> {
    let json: serde_json::Value = serde_json::from_str(json_str)
        .map_err(|e| Error::InvalidInput(format!("无法解析树结构JSON: {}", e)))?;
    
    let nodes_json = json.get("nodes")
        .and_then(|v| v.as_array())
        .ok_or_else(|| Error::InvalidInput("树结构缺少nodes字段".to_string()))?;
    
    let mut nodes = Vec::new();
    for node_json in nodes_json {
        let feature_index = node_json.get("feature_index")
            .and_then(|v| v.as_u64())
            .map(|v| v as usize)
            .unwrap_or(0);
        
        let threshold = node_json.get("threshold")
            .and_then(|v| v.as_f64())
            .map(|f| f as f32)
            .unwrap_or(0.0);
        
        let left_child = node_json.get("left_child")
            .and_then(|v| v.as_u64())
            .map(|v| v as usize);
        
        let right_child = node_json.get("right_child")
            .and_then(|v| v.as_u64())
            .map(|v| v as usize);
        
        let value = node_json.get("value")
            .and_then(|v| v.as_f64())
            .map(|f| f as f32);
        
        nodes.push(TreeNode {
            feature_index,
            threshold,
            left_child,
            right_child,
            value,
        });
    }
    
    Ok(Some(TreeStructure { nodes }))
}

