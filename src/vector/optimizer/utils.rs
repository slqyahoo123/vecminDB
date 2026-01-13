// 工具模块
// 包含辅助函数和公共方法

use crate::vector::index::IndexConfig;
use std::collections::HashMap;
use super::parameter_space::ParameterRange;

/// 将参数应用到索引配置中
pub fn apply_params(config: &mut IndexConfig, params: &HashMap<String, String>) {
    for (key, value) in params {
        match key.as_str() {
            "hnsw_m" => {
                if let Ok(val) = value.parse::<usize>() {
                    config.hnsw_m = val;
                }
            },
            "hnsw_ef_construction" => {
                if let Ok(val) = value.parse::<usize>() {
                    config.hnsw_ef_construction = val;
                }
            },
            "hnsw_ef_search" => {
                if let Ok(val) = value.parse::<usize>() {
                    config.hnsw_ef_search = val;
                }
            },
            "ivf_nlist" => {
                if let Ok(val) = value.parse::<usize>() {
                    config.ivf_nlist = val;
                }
            },
            "ivf_nprobe" => {
                if let Ok(val) = value.parse::<usize>() {
                    config.ivf_nprobe = val;
                }
            },
            "pq_subvector_count" => {
                if let Ok(val) = value.parse::<usize>() {
                    config.pq_subvector_count = val;
                }
            },
            "pq_subvector_bits" => {
                if let Ok(val) = value.parse::<usize>() {
                    config.pq_subvector_bits = val;
                }
            },
            "lsh_hash_count" => {
                if let Ok(val) = value.parse::<usize>() {
                    config.lsh_hash_count = val;
                }
            },
            "lsh_hash_length" => {
                if let Ok(val) = value.parse::<usize>() {
                    config.lsh_hash_length = val;
                }
            },
            _ => {}
        }
    }
}

/// 将连续/整数参数取值应用到索引配置中（用于遗传算法等连续空间优化）
pub fn apply_params_to_config(config: &mut IndexConfig, params: &[f64], param_defs: &[ParameterRange]) {
    for (i, &value) in params.iter().enumerate() {
        if i >= param_defs.len() {
            break;
        }
        
        let param = &param_defs[i];
        let value_int = value.round() as usize;
        
        match param.name.as_str() {
            "hnsw_m" => config.hnsw_m = value_int,
            "hnsw_ef_construction" => config.hnsw_ef_construction = value_int,
            "hnsw_ef_search" => config.hnsw_ef_search = value_int,
            "ivf_nlist" => config.ivf_nlist = value_int,
            "ivf_nprobe" => config.ivf_nprobe = value_int,
            "pq_subvector_count" => config.pq_subvector_count = value_int,
            "pq_subvector_bits" => config.pq_subvector_bits = value_int,
            "lsh_hash_count" => config.lsh_hash_count = value_int,
            "lsh_hash_length" => config.lsh_hash_length = value_int,
            _ => {}
        }
    }
} 