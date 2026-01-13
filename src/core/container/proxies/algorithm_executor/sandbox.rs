/// 沙箱执行模块
/// 
/// 提供在沙箱环境中安全执行WASM代码的功能

use crate::{Result, Error};
use crate::algorithm::executor::sandbox::utils::execute_in_sandbox;
use crate::algorithm::executor::config::SandboxConfig;
use crate::algorithm::types::ResourceLimits;
#[cfg(feature = "wasmtime")]
use wasmtime::{Module, Engine as WasmEngine};
use std::time::Duration;
use std::sync::Arc;
use base64::engine::general_purpose::STANDARD as BASE64_ENGINE;
use base64::Engine as Base64Engine;
use crate::core::UnifiedDataType;

/// 在沙箱中安全执行代码
pub(crate) async fn execute_in_sandbox_safe(
    code: &str, 
    inputs: &[crate::core::UnifiedTensorData]
) -> Result<Vec<crate::core::UnifiedTensorData>> {
    if inputs.is_empty() {
        return Err(Error::InvalidInput("沙箱执行需要至少一个输入".to_string()));
    }
    
    if code.is_empty() {
        return Err(Error::InvalidInput("代码不能为空".to_string()));
    }
    
    #[cfg(feature = "wasmtime")]
    {
        // 1. 检测代码格式：检查是否为WASM二进制
        let code_bytes = code.as_bytes();
        let is_wasm = code_bytes.len() >= 4 && &code_bytes[0..4] == b"\0asm";
        
        if !is_wasm {
            // 如果不是WASM格式，尝试解析为字节数组
            // 可能是base64编码的WASM或其他格式
            if let Ok(decoded) = BASE64_ENGINE.decode(code) {
                let decoded_is_wasm = decoded.len() >= 4 && &decoded[0..4] == b"\0asm";
                if decoded_is_wasm {
                    return execute_wasm_in_sandbox(&decoded, inputs).await;
                }
            }
            
            // 如果不是WASM格式，返回错误，建议使用JSON配置
            return Err(Error::InvalidInput(
                "代码不是有效的WASM二进制格式。请使用JSON配置格式或提供WASM二进制代码".to_string()
            ));
        }
        
        // 2. 执行WASM代码
        execute_wasm_in_sandbox(code_bytes, inputs).await
    }
    
    #[cfg(not(feature = "wasmtime"))]
    {
        Err(Error::InvalidInput(
            "WASM沙箱功能需要启用 'wasmtime' feature。请使用 'cargo build --features wasmtime' 编译".to_string()
        ))
    }
}

/// 在沙箱中执行WASM代码
#[cfg(feature = "wasmtime")]
async fn execute_wasm_in_sandbox(
    wasm_code: &[u8],
    inputs: &[crate::core::UnifiedTensorData],
) -> Result<Vec<crate::core::UnifiedTensorData>> {
    // 1. 将 UnifiedTensorData 转换为输入字节数组
    let input_data = prepare_sandbox_input(inputs)?;
    
    // 2. 创建WASM模块
    let engine = WasmEngine::default();
    let module = Arc::new(
        Module::new(&engine, wasm_code)
            .map_err(|e| Error::InvalidInput(format!("无法编译WASM模块: {}", e)))?
    );
    
    // 3. 配置沙箱
    let sandbox_config = SandboxConfig::default();
    let mut resource_limits = ResourceLimits::default();
    resource_limits.max_memory_usage = 1024 * 1024 * 100; // 100MB
    resource_limits.max_memory_bytes = 1024 * 1024 * 100; // 100MB
    resource_limits.max_execution_time_ms = 300 * 1000; // 5分钟
    resource_limits.max_cpu_time_seconds = 300; // 5分钟
    let timeout = Duration::from_secs(300); // 5分钟超时
    
    // 4. 在沙箱中执行
    #[cfg(feature = "wasmtime")]
    let sandbox_result = execute_in_sandbox(
        &module,
        &input_data,
        &sandbox_config,
        resource_limits,
        timeout,
    )
    .await
    .map_err(|e| Error::InvalidInput(format!("沙箱执行失败: {}", e)))?;
    
    #[cfg(not(feature = "wasmtime"))]
    return Err(Error::InvalidInput(
        "WASM沙箱功能需要启用 'wasmtime' feature".to_string()
    ));
    
    // 5. 解析结果
    if !sandbox_result.stdout.is_empty() {
        // 尝试将stdout解析为JSON格式的张量数据
        if let Ok(json_value) = serde_json::from_str::<serde_json::Value>(&sandbox_result.stdout) {
            return parse_sandbox_output_json(&json_value);
        }
        
        // 如果不是JSON，尝试解析为二进制格式
        if let Ok(tensors) = bincode::deserialize::<Vec<crate::core::UnifiedTensorData>>(&sandbox_result.stdout.as_bytes()) {
            return Ok(tensors);
        }
        
        // 如果都失败，尝试将stdout作为单个张量的数据
        let output_data: Vec<f32> = sandbox_result.stdout
            .split_whitespace()
            .filter_map(|s| s.parse::<f32>().ok())
            .collect();
        
        if !output_data.is_empty() {
            let n_samples = inputs[0].shape[0];
            let n_features = output_data.len() / n_samples;
            return Ok(vec![crate::core::UnifiedTensorData {
                shape: vec![n_samples, n_features],
                data: output_data,
                dtype: UnifiedDataType::Float32,
                device: "CPU".to_string(),
            }]);
        }
    }
    
    // 6. 如果执行失败，返回错误
    if !sandbox_result.stderr.is_empty() {
        return Err(Error::InvalidInput(format!("沙箱执行错误: {}", sandbox_result.stderr)));
    }
    
    Err(Error::InvalidInput("沙箱执行未返回有效结果".to_string()))
}

/// 准备沙箱输入数据
fn prepare_sandbox_input(inputs: &[crate::core::UnifiedTensorData]) -> Result<Vec<u8>> {
    // 将 UnifiedTensorData 序列化为JSON格式，然后转换为字节数组
    let input_json = serde_json::json!({
        "inputs": inputs.iter().map(|t| {
            serde_json::json!({
                "shape": t.shape,
                "data": t.data,
                "dtype": format!("{:?}", t.dtype),
                "device": t.device,
            })
        }).collect::<Vec<_>>()
    });
    
    // 使用JSON字符串的字节数组作为输入
    Ok(input_json.to_string().into_bytes())
}

/// 解析沙箱输出的JSON结果
fn parse_sandbox_output_json(json_value: &serde_json::Value) -> Result<Vec<crate::core::UnifiedTensorData>> {
    let mut outputs = Vec::new();
    
    // 检查是否是数组格式
    if let Some(outputs_array) = json_value.as_array() {
        for output_item in outputs_array {
            if let Some(tensor) = parse_tensor_from_json(output_item)? {
                outputs.push(tensor);
            }
        }
    } else if let Some(tensor) = parse_tensor_from_json(json_value)? {
        // 单个张量
        outputs.push(tensor);
    } else {
        return Err(Error::InvalidInput("无法解析沙箱输出JSON格式".to_string()));
    }
    
    if outputs.is_empty() {
        return Err(Error::InvalidInput("沙箱输出为空".to_string()));
    }
    
    Ok(outputs)
}

/// 从JSON值解析张量
fn parse_tensor_from_json(json_value: &serde_json::Value) -> Result<Option<crate::core::UnifiedTensorData>> {
    let shape = json_value.get("shape")
        .and_then(|v| v.as_array())
        .and_then(|arr| {
            arr.iter()
                .map(|v| v.as_u64().map(|n| n as usize))
                .collect::<Option<Vec<_>>>()
        })
        .ok_or_else(|| Error::InvalidInput("缺少shape字段".to_string()))?;
    
    let data = json_value.get("data")
        .and_then(|v| v.as_array())
        .and_then(|arr| {
            arr.iter()
                .map(|v| v.as_f64().map(|f| f as f32))
                .collect::<Option<Vec<_>>>()
        })
        .ok_or_else(|| Error::InvalidInput("缺少data字段或数据格式无效".to_string()))?;
    
    let dtype_str = json_value.get("dtype")
        .and_then(|v| v.as_str())
        .unwrap_or("Float32");
    
    let dtype = match dtype_str {
        "Float32" => UnifiedDataType::Float32,
        "Float64" => UnifiedDataType::Float64,
        "Int32" => UnifiedDataType::Int32,
        "Int64" => UnifiedDataType::Int64,
        _ => UnifiedDataType::Float32,
    };
    
    let device = json_value.get("device")
        .and_then(|v| v.as_str())
        .unwrap_or("CPU")
        .to_string();
    
    Ok(Some(crate::core::UnifiedTensorData {
        shape,
        data,
        dtype,
        device,
    }))
}

