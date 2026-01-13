/// 类型转换模块
/// 
/// 提供统一类型和核心类型之间的转换函数

use crate::Result;
use crate::core::types::{CoreTensorData, CoreAlgorithmDefinition};
use crate::core::unified_system::AlgorithmDefinition as UnifiedAlgorithmDefinition;
use chrono::Utc;
use uuid::Uuid;
use std::collections::HashMap;

// 类型别名
type AlgorithmDefinition = CoreAlgorithmDefinition;

/// 将统一算法定义转换为核心算法定义
pub(crate) fn convert_unified_to_core_algorithm_definition(
    algo_def: &UnifiedAlgorithmDefinition
) -> Result<CoreAlgorithmDefinition> {
    // 转换algorithm_type
    let algorithm_type = match algo_def.algorithm_type {
        crate::core::unified_system::AlgorithmType::Classification => crate::core::types::AlgorithmType::Classification,
        crate::core::unified_system::AlgorithmType::Regression => crate::core::types::AlgorithmType::Regression,
        crate::core::unified_system::AlgorithmType::Clustering => crate::core::types::AlgorithmType::Clustering,
        crate::core::unified_system::AlgorithmType::DataProcessing => crate::core::types::AlgorithmType::MachineLearning,
        crate::core::unified_system::AlgorithmType::FeatureExtraction => crate::core::types::AlgorithmType::MachineLearning,
        crate::core::unified_system::AlgorithmType::Custom => crate::core::types::AlgorithmType::MachineLearning,
        _ => crate::core::types::AlgorithmType::MachineLearning,
    };
    
    // 转换TensorSchema到TensorSchemaInterface
    let input_schema: Vec<crate::core::interfaces::TensorSchemaInterface> = algo_def.input_schema.iter().map(|s| {
        crate::core::interfaces::TensorSchemaInterface {
            name: s.name.clone(),
            shape: s.shape.iter().filter_map(|&x| x).collect(),
            data_type: match s.dtype {
                crate::core::unified_system::UnifiedDataType::Float32 => crate::core::types::DataType::Float32,
                crate::core::unified_system::UnifiedDataType::Float64 => crate::core::types::DataType::Float64,
                crate::core::unified_system::UnifiedDataType::Int32 => crate::core::types::DataType::Int32,
                crate::core::unified_system::UnifiedDataType::Int64 => crate::core::types::DataType::Int64,
                _ => crate::core::types::DataType::Float32,
            },
            optional: s.optional,
            description: None,
        }
    }).collect();
    
    let output_schema: Vec<crate::core::interfaces::TensorSchemaInterface> = algo_def.output_schema.iter().map(|s| {
        crate::core::interfaces::TensorSchemaInterface {
            name: s.name.clone(),
            shape: s.shape.iter().filter_map(|&x| x).collect(),
            data_type: match s.dtype {
                crate::core::unified_system::UnifiedDataType::Float32 => crate::core::types::DataType::Float32,
                crate::core::unified_system::UnifiedDataType::Float64 => crate::core::types::DataType::Float64,
                crate::core::unified_system::UnifiedDataType::Int32 => crate::core::types::DataType::Int32,
                crate::core::unified_system::UnifiedDataType::Int64 => crate::core::types::DataType::Int64,
                _ => crate::core::types::DataType::Float32,
            },
            optional: s.optional,
            description: None,
        }
    }).collect();
    
    // 转换ResourceRequirements到ResourceRequirementsInterface
    let resource_requirements = crate::core::interfaces::ResourceRequirementsInterface {
        max_memory_mb: algo_def.resource_requirements.max_memory_mb,
        max_cpu_percent: algo_def.resource_requirements.max_cpu_percent,
        max_execution_time_seconds: algo_def.resource_requirements.max_execution_time_seconds,
        requires_gpu: algo_def.resource_requirements.requires_gpu,
        max_gpu_memory_mb: None, // unified_system::ResourceRequirements没有这个字段
        network_access: false,
        file_system_access: Vec::new(),
    };
    
    Ok(CoreAlgorithmDefinition {
        id: algo_def.id.clone(),
        name: algo_def.name.clone(),
        version: "1.0.0".to_string(), // UnifiedAlgorithmDefinition 没有 version 字段，使用默认值
        algorithm_type,
        description: algo_def.description.clone(),
        parameters: Vec::new(),
        source_code: algo_def.source_code.clone(),
        language: algo_def.language.clone(),
        input_schema,
        output_schema,
        resource_requirements,
        metadata: HashMap::new(), // UnifiedAlgorithmDefinition 没有 metadata 字段
        created_at: Utc::now(), // UnifiedAlgorithmDefinition 没有 created_at 字段
        updated_at: Utc::now(), // UnifiedAlgorithmDefinition 没有 updated_at 字段
    })
}

/// 将核心算法定义转换为统一算法定义
pub(crate) fn convert_core_to_unified_algorithm_definition(
    core_def: &CoreAlgorithmDefinition
) -> Result<UnifiedAlgorithmDefinition> {
    // 转换algorithm_type
    let algorithm_type = match core_def.algorithm_type {
        crate::core::types::AlgorithmType::Classification => crate::core::unified_system::AlgorithmType::Classification,
        crate::core::types::AlgorithmType::Regression => crate::core::unified_system::AlgorithmType::Regression,
        crate::core::types::AlgorithmType::Clustering => crate::core::unified_system::AlgorithmType::Clustering,
        _ => crate::core::unified_system::AlgorithmType::DataProcessing,
    };
    
    // 转换TensorSchemaInterface到TensorSchema
    let input_schema: Vec<crate::core::unified_system::TensorSchema> = core_def.input_schema.iter().map(|s| {
        crate::core::unified_system::TensorSchema {
            name: s.name.clone(),
            shape: s.shape.iter().map(|&x| Some(x)).collect(),
            dtype: match s.data_type {
                crate::core::types::DataType::Float32 => crate::core::unified_system::UnifiedDataType::Float32,
                crate::core::types::DataType::Float64 => crate::core::unified_system::UnifiedDataType::Float64,
                crate::core::types::DataType::Int32 => crate::core::unified_system::UnifiedDataType::Int32,
                crate::core::types::DataType::Int64 => crate::core::unified_system::UnifiedDataType::Int64,
                _ => crate::core::unified_system::UnifiedDataType::Float32,
            },
            optional: s.optional,
        }
    }).collect();
    
    let output_schema: Vec<crate::core::unified_system::TensorSchema> = core_def.output_schema.iter().map(|s| {
        crate::core::unified_system::TensorSchema {
            name: s.name.clone(),
            shape: s.shape.iter().map(|&x| Some(x)).collect(),
            dtype: match s.data_type {
                crate::core::types::DataType::Float32 => crate::core::unified_system::UnifiedDataType::Float32,
                crate::core::types::DataType::Float64 => crate::core::unified_system::UnifiedDataType::Float64,
                crate::core::types::DataType::Int32 => crate::core::unified_system::UnifiedDataType::Int32,
                crate::core::types::DataType::Int64 => crate::core::unified_system::UnifiedDataType::Int64,
                _ => crate::core::unified_system::UnifiedDataType::Float32,
            },
            optional: s.optional,
        }
    }).collect();
    
    // 转换ResourceRequirementsInterface到ResourceRequirements
    let resource_requirements = crate::core::unified_system::ResourceRequirements {
        max_memory_mb: core_def.resource_requirements.max_memory_mb,
        max_cpu_percent: core_def.resource_requirements.max_cpu_percent,
        max_execution_time_seconds: core_def.resource_requirements.max_execution_time_seconds,
        requires_gpu: core_def.resource_requirements.requires_gpu,
    };
    
    Ok(UnifiedAlgorithmDefinition {
        id: core_def.id.clone(),
        name: core_def.name.clone(),
        description: core_def.description.clone(),
        algorithm_type,
        source_code: core_def.source_code.clone(),
        language: core_def.language.clone(),
        input_schema,
        output_schema,
        resource_requirements,
    })
}

/// 将算法定义转换为核心算法定义（同类型转换）
pub(crate) fn convert_to_core_algorithm_definition(
    algo_def: &AlgorithmDefinition
) -> Result<CoreAlgorithmDefinition> {
    Ok(CoreAlgorithmDefinition {
        id: algo_def.id.clone(),
        name: algo_def.name.clone(),
        version: algo_def.version.clone(),
        algorithm_type: algo_def.algorithm_type.clone(),
        description: algo_def.description.clone(),
        parameters: algo_def.parameters.clone(),
        source_code: algo_def.source_code.clone(),
        language: algo_def.language.clone(),
        input_schema: algo_def.input_schema.clone(),
        output_schema: algo_def.output_schema.clone(),
        resource_requirements: algo_def.resource_requirements.clone(),
        metadata: algo_def.metadata.clone(),
        created_at: algo_def.created_at,
        updated_at: algo_def.updated_at,
    })
}

/// 将核心算法定义转换为算法定义（同类型转换）
/// 注意：AlgorithmDefinition 是 CoreAlgorithmDefinition 的类型别名，所以直接返回克隆
pub(crate) fn convert_from_core_algorithm_definition(
    core_def: &CoreAlgorithmDefinition
) -> Result<AlgorithmDefinition> {
    // AlgorithmDefinition 是 CoreAlgorithmDefinition 的类型别名，直接返回克隆
    Ok(core_def.clone())
}

/// 将 UnifiedTensorData 转换为 CoreTensorData
pub(crate) fn convert_unified_to_core(
    unified: &crate::core::UnifiedTensorData
) -> CoreTensorData {
    CoreTensorData {
        id: Uuid::new_v4().to_string(),
        shape: unified.shape.clone(),
        data: unified.data.clone(),
        dtype: match unified.dtype {
            crate::core::UnifiedDataType::Float32 => "float32".to_string(),
            crate::core::UnifiedDataType::Float64 => "float64".to_string(),
            crate::core::UnifiedDataType::Int32 => "int32".to_string(),
            crate::core::UnifiedDataType::Int64 => "int64".to_string(),
            _ => "float32".to_string(),
        },
        device: unified.device.clone(),
        requires_grad: false,
        metadata: HashMap::new(),
        created_at: Utc::now(),
        updated_at: Utc::now(),
    }
}

/// 将 CoreTensorData 转换为 UnifiedTensorData
pub(crate) fn convert_core_to_unified(
    core: &CoreTensorData
) -> crate::core::UnifiedTensorData {
    crate::core::UnifiedTensorData {
        shape: core.shape.clone(),
        data: core.data.clone(),
        dtype: match core.dtype.as_str() {
            "float32" => crate::core::UnifiedDataType::Float32,
            "float64" => crate::core::UnifiedDataType::Float64,
            "int32" => crate::core::UnifiedDataType::Int32,
            "int64" => crate::core::UnifiedDataType::Int64,
            _ => crate::core::UnifiedDataType::Float32,
        },
        device: core.device.clone(),
    }
}

