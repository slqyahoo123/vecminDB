/// 线性回归算法模块
/// 
/// 提供线性回归算法的执行实现

use crate::{Result, Error};
use crate::core::types::CoreAlgorithmDefinition;
use crate::core::UnifiedDataType;
use chrono::Utc;
use std::collections::HashMap;
// use crate::core::types::ResourceRequirements; // 不再需要，使用 ResourceRequirementsInterface

use super::model_parsing::parse_model_parameters;
use super::matrix_ops::least_squares;

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

/// 执行线性回归算法（带算法定义参数，用于获取模型参数）
pub(crate) async fn execute_linear_regression_with_params(
    algo_def: &AlgorithmDefinition,
    inputs: &[crate::core::UnifiedTensorData]
) -> Result<Vec<crate::core::UnifiedTensorData>> {
    if inputs.is_empty() {
        return Err(Error::InvalidInput("线性回归需要至少一个输入张量".to_string()));
    }
    
    let features = &inputs[0];
    let n_samples = features.shape[0];
    let n_features = calculate_n_features(features, n_samples)?;
    
    // 解析模型参数
    let params = parse_model_parameters(algo_def)?;
    
    // 如果有模型参数，使用参数进行预测
    if let (Some(weights), Some(bias)) = (params.weights, params.bias) {
        if weights.len() != n_features {
            return Err(Error::InvalidInput(
                format!("权重数量({})与特征数量({})不匹配", weights.len(), n_features)
            ));
        }
        
        let mut predictions = Vec::with_capacity(n_samples);
        for i in 0..n_samples {
            let mut pred = bias;
            for j in 0..n_features {
                pred += features.data[i * n_features + j] * weights[j];
            }
            predictions.push(pred);
        }
        
        return Ok(vec![crate::core::UnifiedTensorData {
            shape: vec![n_samples],
            data: predictions,
            dtype: UnifiedDataType::Float32,
            device: "cpu".to_string(),
        }]);
    }
    
    // 如果没有模型参数，尝试从训练数据计算
    if inputs.len() >= 2 {
        let targets = &inputs[1];
        if targets.data.len() == n_samples {
            // 使用训练数据计算权重
            let (weights, bias) = least_squares(
                &features.data,
                n_samples,
                n_features,
                &targets.data
            )?;
            
            // 使用计算出的权重进行预测
            let mut predictions = Vec::with_capacity(n_samples);
            for i in 0..n_samples {
                let mut pred = bias;
                for j in 0..n_features {
                    pred += features.data[i * n_features + j] * weights[j];
                }
                predictions.push(pred);
            }
            
            return Ok(vec![crate::core::UnifiedTensorData {
                shape: vec![n_samples],
                data: predictions,
                dtype: UnifiedDataType::Float32,
                device: "cpu".to_string(),
            }]);
        }
    }
    
    // 如果没有模型参数且没有训练数据，返回错误
    Err(Error::InvalidInput(
        "线性回归需要模型参数（权重和偏置）或训练数据（特征和目标）".to_string()
    ))
}

/// 执行线性回归算法（保持向后兼容）
pub(crate) async fn execute_linear_regression(
    inputs: &[crate::core::UnifiedTensorData]
) -> Result<Vec<crate::core::UnifiedTensorData>> {
    // 创建一个临时的算法定义用于调用带参数的方法
    let temp_algo_def = AlgorithmDefinition {
        id: "temp_linear_regression".to_string(),
        name: "Linear Regression".to_string(),
        algorithm_type: crate::core::types::AlgorithmType::Regression,
        parameters: Vec::new(),
        description: String::new(),
        version: "1.0.0".to_string(),
        source_code: String::new(),
        language: "rust".to_string(),
        input_schema: Vec::new(),
        output_schema: Vec::new(),
        resource_requirements: crate::core::interfaces::ResourceRequirementsInterface {
            max_memory_mb: 512,
            max_cpu_percent: 100.0,
            max_execution_time_seconds: 300,
            requires_gpu: false,
            max_gpu_memory_mb: None,
            network_access: false,
            file_system_access: Vec::new(),
        },
        metadata: HashMap::new(),
        created_at: Utc::now(),
        updated_at: Utc::now(),
    };
    
    execute_linear_regression_with_params(&temp_algo_def, inputs).await
}

/// 执行线性回归算法（CoreTensorData版本）
pub(crate) async fn execute_linear_regression_core(
    inputs: &[crate::core::types::CoreTensorData]
) -> Result<Vec<crate::core::types::CoreTensorData>> {
    use super::conversions::{convert_core_to_unified, convert_unified_to_core};
    
    // 转换为UnifiedTensorData
    let unified_inputs: Vec<crate::core::UnifiedTensorData> = inputs.iter()
        .map(|t| convert_core_to_unified(t))
        .collect();
    
    // 执行算法
    let unified_outputs = execute_linear_regression(&unified_inputs).await?;
    
    // 转换回CoreTensorData
    let core_outputs: Vec<crate::core::types::CoreTensorData> = unified_outputs.iter()
        .map(|t| convert_unified_to_core(t))
        .collect();
    
    Ok(core_outputs)
}

