// HNSW范围搜索实现
// 提供在指定半径内搜索向量的功能

use std::collections::{HashSet, BinaryHeap};
use std::cmp::Reverse;
use std::sync::{Arc, RwLock};
use std::collections::HashMap;
use crate::vector::index::types::SearchResult;
use crate::vector::index::common::HNSWNode;
use super::types::{FloatComparison, ReverseFloatComparison};

/// 在指定半径内搜索
pub fn range_search(
    query: &[f32],
    radius: f32,
    with_vectors: bool,
    with_metadata: bool,
    max_elements: Option<usize>,
    dynamic_ef: bool,
    entry_point: Option<usize>,
    max_layer: usize,
    ef_search: usize,
    calculate_distance_fn: impl Fn(&[f32], usize) -> Result<f32, String>,
    get_connections_fn: impl Fn(usize, usize) -> Result<Vec<usize>, String>,
    get_node_id_fn: impl Fn(usize) -> Result<String, String>,
    vectors: Arc<RwLock<HashMap<String, Vec<f32>>>>,
    nodes: Arc<RwLock<HashMap<String, HNSWNode>>>,
) -> Result<Vec<SearchResult>, String> {
    // 如果索引为空，返回空结果
    if entry_point.is_none() {
        return Ok(vec![]);
    }
    
    // 使用动态ef可能会提高结果质量
    let effective_ef = if dynamic_ef {
        // 使用较大的ef以确保不会错过潜在的结果
        ef_search.max(50)
    } else {
        ef_search
    };
    
    // 执行范围搜索
    let results = range_search_internal(
        query,
        radius,
        effective_ef,
        entry_point.unwrap(),
        max_layer,
        calculate_distance_fn,
        get_connections_fn,
    )?;
    
    // 限制结果数量
    let limited_results = if let Some(max) = max_elements {
        results.into_iter().take(max).collect()
    } else {
        results
    };
    
    // 转换结果为SearchResult
    let mut search_results = Vec::with_capacity(limited_results.len());
    for (node_id, distance) in limited_results {
        let id = get_node_id_fn(node_id)?;
        search_results.push(SearchResult {
            id,
            distance,
            metadata: None,
            node: None,
            vector: None,
        });
    }
    
    // 添加向量和元数据（如果需要）
    if with_vectors || with_metadata {
        let mut filtered_results = Vec::with_capacity(search_results.len());
        
        for mut result in search_results {
            if with_vectors {
                if let Ok(vectors_lock) = vectors.read() {
                    result.vector = vectors_lock.get(&result.id).cloned();
                }
            }
            
            if with_metadata {
                if let Ok(nodes_lock) = nodes.read() {
                    if let Some(node) = nodes_lock.get(&result.id) {
                        result.metadata = node.metadata.clone();
                        result.node = Some(node.clone());
                    }
                }
            }
            
            filtered_results.push(result);
        }
        
        Ok(filtered_results)
    } else {
        Ok(search_results)
    }
}

/// 内部范围搜索实现
fn range_search_internal(
    query: &[f32],
    radius: f32,
    ef: usize,
    entry_point: usize,
    max_layer: usize,
    calculate_distance_fn: impl Fn(&[f32], usize) -> Result<f32, String>,
    get_connections_fn: impl Fn(usize, usize) -> Result<Vec<usize>, String>,
) -> Result<Vec<(usize, f32)>, String> {
    let mut current_layer = max_layer;
    let mut ep = entry_point;
    
    // 在非零层找到最近邻节点
    while current_layer > 0 {
        let layer_search_result = search_layer(
            query,
            ep,
            1,
            current_layer,
            &calculate_distance_fn,
            &get_connections_fn,
        )?;
        
        if !layer_search_result.is_empty() {
            ep = layer_search_result[0].0;
        }
        
        current_layer -= 1;
    }
    
    // 在零层执行范围搜索
    let search_result = search_layer(
        query,
        ep,
        ef,
        0,
        &calculate_distance_fn,
        &get_connections_fn,
    )?;
    
    // 过滤掉超出半径的结果
    Ok(search_result
        .into_iter()
        .filter(|(_, dist)| *dist <= radius)
        .collect())
}

/// 在特定层进行搜索
fn search_layer(
    query: &[f32],
    entry_point: usize,
    ef: usize,
    layer: usize,
    calculate_distance_fn: &impl Fn(&[f32], usize) -> Result<f32, String>,
    get_connections_fn: &impl Fn(usize, usize) -> Result<Vec<usize>, String>,
) -> Result<Vec<(usize, f32)>, String> {
    let mut visited = HashSet::new();
    let mut candidates = BinaryHeap::new();
    let mut results = BinaryHeap::new();
    
    // 计算查询向量与入口点的距离
    let dist = calculate_distance_fn(query, entry_point)?;
    
    // 将入口点加入候选集和结果集
    candidates.push(Reverse(FloatComparison(dist, entry_point)));
    results.push(ReverseFloatComparison(dist, entry_point));
    visited.insert(entry_point);
    
    // 当候选集不为空时继续搜索
    while let Some(Reverse(FloatComparison(dist, node_id))) = candidates.pop() {
        // 如果结果中最远的距离小于当前候选节点的距离，停止搜索
        if let Some(ReverseFloatComparison(furthest_dist, _)) = results.peek() {
            if *furthest_dist < dist && results.len() >= ef {
                break;
            }
        }
        
        // 获取当前节点在指定层的连接
        let connections = get_connections_fn(node_id, layer)?;
        
        // 检查每个连接
        for &neighbor_id in &connections {
            if !visited.insert(neighbor_id) {
                continue; // 已访问过此节点
            }
            
            // 计算查询向量与邻居的距离
            let neighbor_dist = calculate_distance_fn(query, neighbor_id)?;
            
            // 如果结果集未满或邻居距离比结果中最远的距离更近
            if results.len() < ef || neighbor_dist < results.peek().unwrap().0 {
                candidates.push(Reverse(FloatComparison(neighbor_dist, neighbor_id)));
                results.push(ReverseFloatComparison(neighbor_dist, neighbor_id));
                
                // 如果结果集超过ef，移除最远的节点
                if results.len() > ef {
                    results.pop();
                }
            }
        }
    }
    
    // 将结果转换为向量
    let mut result_vec = Vec::with_capacity(results.len());
    while let Some(ReverseFloatComparison(dist, node_id)) = results.pop() {
        result_vec.push((node_id, dist));
    }
    
    // 按距离升序排序
    result_vec.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
    
    Ok(result_vec)
}

/// 批量范围搜索
pub fn batch_range_search(
    queries: &[Vec<f32>],
    radius: f32,
    with_vectors: bool,
    with_metadata: bool,
    max_elements: Option<usize>,
    dynamic_ef: bool,
    entry_point: Option<usize>,
    max_layer: usize,
    ef_search: usize,
    calculate_distance_fn: impl Fn(&[f32], usize) -> Result<f32, String> + Clone,
    get_connections_fn: impl Fn(usize, usize) -> Result<Vec<usize>, String> + Clone,
    get_node_id_fn: impl Fn(usize) -> Result<String, String> + Clone,
    vectors: Arc<RwLock<HashMap<String, Vec<f32>>>>,
    nodes: Arc<RwLock<HashMap<String, HNSWNode>>>,
) -> Result<Vec<Vec<SearchResult>>> {
    // 为每个查询执行范围搜索
    let mut all_results = Vec::with_capacity(queries.len());
    
    for query in queries {
        let results = range_search(
            query,
            radius,
            with_vectors,
            with_metadata,
            max_elements,
            dynamic_ef,
            entry_point,
            max_layer,
            ef_search,
            calculate_distance_fn.clone(),
            get_connections_fn.clone(),
            get_node_id_fn.clone(),
            Arc::clone(&vectors),
            Arc::clone(&nodes),
        )?;
        
        all_results.push(results);
    }
    
    Ok(all_results)
}

/// 动态范围搜索（根据时间预算自动调整参数）
pub fn dynamic_range_search(
    query: &[f32],
    radius: f32,
    max_elements: Option<usize>,
    time_budget_ms: u64,
    entry_point: Option<usize>,
    max_layer: usize,
    ef_search: usize,
    calculate_distance_fn: impl Fn(&[f32], usize) -> Result<f32, String>,
    get_connections_fn: impl Fn(usize, usize) -> Result<Vec<usize>, String>,
    get_node_id_fn: impl Fn(usize) -> Result<String, String>,
    range_search_fn: impl Fn(&[f32], f32, Option<usize>, usize) -> Result<Vec<(usize, f32)>, String>,
) -> Result<Vec<(usize, f32)>, String> {
    // 初始ef值
    let mut current_ef = ef_search;
    
    // 在有限时间内尝试不同的ef值
    let start_time = std::time::Instant::now();
    let time_budget = std::time::Duration::from_millis(time_budget_ms);
    
    // 第一次搜索使用默认ef
    let mut results = range_search_fn(query, radius, max_elements, current_ef)?;
    
    // 如果还有时间，尝试增加ef以提高质量
    while start_time.elapsed() < time_budget && current_ef < 1000 {
        // 增加ef
        current_ef = (current_ef * 2).min(1000);
        
        // 执行新的搜索
        let new_results = range_search_fn(query, radius, max_elements, current_ef)?;
        
        // 如果新结果更多，更新结果
        if new_results.len() > results.len() {
            results = new_results;
        }
        
        // 如果时间预算的80%已用完，停止
        if start_time.elapsed() > time_budget.mul_f32(0.8) {
            break;
        }
    }
    
    Ok(results)
}

/// 估计特定半径内的点密度
pub fn estimate_radius_density(
    query: &[f32],
    radius: f32,
    entry_point: Option<usize>,
    max_layer: usize,
    ef_search: usize,
    calculate_distance_fn: impl Fn(&[f32], usize) -> Result<f32, String>,
    get_connections_fn: impl Fn(usize, usize) -> Result<Vec<usize>, String>,
    total_vectors: usize,
) -> Result<f32, String> {
    // 如果索引为空，返回0密度
    if entry_point.is_none() || total_vectors == 0 {
        return Ok(0.0);
    }
    
    // 执行范围搜索
    let results = range_search_internal(
        query,
        radius,
        ef_search,
        entry_point.unwrap(),
        max_layer,
        calculate_distance_fn,
        get_connections_fn,
    )?;
    
    // 计算密度：范围内点数 / 总点数
    let density = results.len() as f32 / total_vectors as f32;
    
    Ok(density)
}
