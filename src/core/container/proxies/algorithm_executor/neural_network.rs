/// 神经网络算法模块
/// 
/// 提供神经网络前向传播算法的执行实现

use crate::{Result, Error};
use crate::core::types::CoreAlgorithmDefinition;
use crate::core::UnifiedDataType;

use super::model_parsing::parse_model_parameters;
use super::conversions::{convert_core_to_unified, convert_unified_to_core};

// 类型别名
type AlgorithmDefinition = CoreAlgorithmDefinition;

/// 应用激活函数
pub(crate) fn apply_activation(data: &[f32], activation_type: &str) -> Result<Vec<f32>> {
    let mut result = data.to_vec();
    
    match activation_type {
        "relu" => {
            for value in &mut result {
                *value = value.max(0.0);
            }
        },
        "sigmoid" => {
            for value in &mut result {
                *value = 1.0 / (1.0 + (-*value).exp());
            }
        },
        "tanh" => {
            for value in &mut result {
                *value = value.tanh();
            }
        },
        _ => {
            return Err(Error::InvalidInput(
                format!("不支持的激活函数: {}", activation_type)
            ));
        }
    }
    
    Ok(result)
}

/// 执行神经网络算法
pub(crate) async fn execute_neural_network(
    algo_def: &AlgorithmDefinition, 
    inputs: &[crate::core::UnifiedTensorData]
) -> Result<Vec<crate::core::UnifiedTensorData>> {
    if inputs.is_empty() {
        return Err(Error::InvalidInput("神经网络需要至少一个输入张量".to_string()));
    }
    
    let input = &inputs[0];
    let input_size = input.data.len();
    
    // 解析模型参数
    let params = parse_model_parameters(algo_def)?;
    
    // 如果有网络配置，使用配置进行前向传播
    if let Some(network_config) = params.network_config {
        if network_config.layers.is_empty() {
            return Err(Error::InvalidInput("网络配置中没有层定义".to_string()));
        }
        
        let mut current_data = input.data.clone();
        let mut current_size = input_size;
        
        // 逐层前向传播
        for (layer_idx, layer) in network_config.layers.iter().enumerate() {
            if current_size != layer.input_size {
                return Err(Error::InvalidInput(
                    format!("第{}层输入大小不匹配: 期望 {}, 实际 {}", layer_idx, layer.input_size, current_size)
                ));
            }
            
            // 应用线性变换: output = input * weights + bias
            // 验证权重矩阵大小
            let expected_weights_size = layer.input_size * layer.output_size;
            if layer.weights.len() != expected_weights_size {
                return Err(Error::InvalidInput(
                    format!("第{}层权重矩阵大小不匹配: 期望 {} ({} x {}), 实际 {}", 
                           layer_idx, expected_weights_size, layer.input_size, layer.output_size, layer.weights.len())
                ));
            }
            
            // 验证偏置向量大小
            if layer.bias.len() != layer.output_size {
                return Err(Error::InvalidInput(
                    format!("第{}层偏置向量大小不匹配: 期望 {}, 实际 {}", 
                           layer_idx, layer.output_size, layer.bias.len())
                ));
            }
            
            let mut output = vec![0.0; layer.output_size];
            for i in 0..layer.output_size {
                output[i] = layer.bias[i];
                for j in 0..layer.input_size {
                    output[i] += current_data[j] * layer.weights[i * layer.input_size + j];
                }
            }
            
            // 应用激活函数
            current_data = apply_activation(&output, &layer.activation)?;
            current_size = layer.output_size;
        }
        
        return Ok(vec![crate::core::UnifiedTensorData {
            shape: vec![current_size],
            data: current_data,
            dtype: UnifiedDataType::Float32,
            device: "cpu".to_string(),
        }]);
    }
    
    // 如果没有网络配置，返回错误
    Err(Error::InvalidInput(
        "神经网络需要网络配置（层数、每层大小、权重和偏置）".to_string()
    ))
}

/// 执行神经网络算法（CoreTensorData版本）
pub(crate) async fn execute_neural_network_core(
    algo_def: &AlgorithmDefinition,
    inputs: &[crate::core::types::CoreTensorData]
) -> Result<Vec<crate::core::types::CoreTensorData>> {
    // 转换为UnifiedTensorData
    let unified_inputs: Vec<crate::core::UnifiedTensorData> = inputs.iter()
        .map(|t| convert_core_to_unified(t))
        .collect();
    
    // 执行算法
    let unified_outputs = execute_neural_network(algo_def, &unified_inputs).await?;
    
    // 转换回CoreTensorData
    let core_outputs: Vec<crate::core::types::CoreTensorData> = unified_outputs.iter()
        .map(|t| convert_unified_to_core(t))
        .collect();
    
    Ok(core_outputs)
}

