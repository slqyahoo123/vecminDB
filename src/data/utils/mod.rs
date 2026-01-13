// Utilities Module - Provides general utility functions and data processing tools

// Re-export public components
mod parallel_utils;

pub use parallel_utils::{
    ParallelProcessor,
    // 注释不存在的类型
    // ProcessingConfig,
    // TaskDispatcher,
    // DataChunk,
    // ProcessingResult,
    // ProcessingStats
};

// 引入DataValue类型
use crate::data::value::DataValue;

// Common utility functions
pub fn is_valid_json(input: &str) -> bool {
    serde_json::from_str::<serde_json::Value>(input).is_ok()
}

pub fn is_valid_csv(input: &str) -> bool {
    let mut reader = csv::Reader::from_reader(input.as_bytes());
    reader.records().next().is_some()
}

/// 合并两个数据值
pub fn merge_values(left: &DataValue, right: &DataValue) -> DataValue {
    match (left, right) {
        (DataValue::Object(left_map), DataValue::Object(right_map)) => {
            let mut result = left_map.clone();
            for (key, value) in right_map {
                result.insert(key.clone(), value.clone());
            }
            DataValue::Object(result)
        },
        (DataValue::Array(left_vec), DataValue::Array(right_vec)) => {
            let mut result = left_vec.clone();
            result.extend(right_vec.clone());
            DataValue::Array(result)
        },
        _ => right.clone(),
    }
}

/// 过滤数据值
pub fn filter_values(values: &[DataValue], predicate: impl Fn(&DataValue) -> bool) -> Vec<DataValue> {
    values.iter().filter(|v| predicate(v)).cloned().collect()
}

/// 转换数据值
pub fn map_values(values: &[DataValue], mapper: impl Fn(&DataValue) -> DataValue) -> Vec<DataValue> {
    values.iter().map(|v| mapper(v)).collect()
} 