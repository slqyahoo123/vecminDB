/// 聚类算法模块
/// 
/// 提供K-means聚类算法的实现

use crate::{Result, Error};
use rand::Rng;

/// K-means聚类算法
pub(crate) fn kmeans_clustering(
    data: &[f32], 
    n_samples: usize, 
    n_features: usize,
    k: usize, 
    max_iterations: usize
) -> Result<Vec<usize>> {
    if data.len() != n_samples * n_features {
        return Err(Error::InvalidInput(
            format!("数据大小不匹配: 期望 {}, 实际 {}", n_samples * n_features, data.len())
        ));
    }
    
    if k == 0 || k > n_samples {
        return Err(Error::InvalidInput(
            format!("聚类数K无效: {} (必须在1到{}之间)", k, n_samples)
        ));
    }
    
    // 初始化聚类中心（K-means++）
    let mut centers = kmeans_plus_plus_init(data, n_samples, n_features, k)?;
    
    // 迭代优化
    let mut labels = vec![0; n_samples];
    for _iteration in 0..max_iterations {
        // 分配点到最近的中心
        let mut changed = false;
        for i in 0..n_samples {
            let mut min_dist = f32::MAX;
            let mut best_cluster = 0;
            
            for j in 0..k {
                let mut dist = 0.0;
                for f in 0..n_features {
                    let diff = data[i * n_features + f] - centers[j * n_features + f];
                    dist += diff * diff;
                }
                dist = dist.sqrt();
                
                if dist < min_dist {
                    min_dist = dist;
                    best_cluster = j;
                }
            }
            
            if labels[i] != best_cluster {
                changed = true;
                labels[i] = best_cluster;
            }
        }
        
        if !changed {
            break;
        }
        
        // 更新中心
        let mut cluster_counts = vec![0; k];
        let mut cluster_sums = vec![0.0; k * n_features];
        
        for i in 0..n_samples {
            let cluster = labels[i];
            cluster_counts[cluster] += 1;
            for f in 0..n_features {
                cluster_sums[cluster * n_features + f] += data[i * n_features + f];
            }
        }
        
        for j in 0..k {
            if cluster_counts[j] > 0 {
                for f in 0..n_features {
                    centers[j * n_features + f] = cluster_sums[j * n_features + f] / cluster_counts[j] as f32;
                }
            } else {
                // 空聚类：重新随机初始化
                // 从数据中随机选择一个点作为新中心
                let mut rng = rand::thread_rng();
                let random_idx = rng.gen_range(0..n_samples);
                for f in 0..n_features {
                    centers[j * n_features + f] = data[random_idx * n_features + f];
                }
            }
        }
    }
    
    Ok(labels)
}

/// K-means++初始化
/// 使用真正的随机数生成器，而不是固定seed
pub(crate) fn kmeans_plus_plus_init(
    data: &[f32], 
    n_samples: usize, 
    n_features: usize,
    k: usize
) -> Result<Vec<f32>> {
    let mut centers = Vec::with_capacity(k * n_features);
    let mut rng = rand::thread_rng();
    
    // 第一个中心随机选择
    let first_idx = rng.gen_range(0..n_samples);
    for f in 0..n_features {
        centers.push(data[first_idx * n_features + f]);
    }
    
    // 选择剩余的中心
    for _center_idx in 1..k {
        let mut distances = vec![f32::MAX; n_samples];
        
        // 计算每个点到最近中心的距离
        for i in 0..n_samples {
            for c in 0..centers.len() / n_features {
                let mut dist = 0.0;
                for f in 0..n_features {
                    let diff = data[i * n_features + f] - centers[c * n_features + f];
                    dist += diff * diff;
                }
                dist = dist.sqrt();
                if dist < distances[i] {
                    distances[i] = dist;
                }
            }
        }
        
        // 根据距离平方的概率选择下一个中心
        let total_dist_sq: f32 = distances.iter().map(|&d| d * d).sum();
        if total_dist_sq < 1e-10 {
            // 如果所有点都很近，随机选择
            let idx = rng.gen_range(0..n_samples);
            for f in 0..n_features {
                centers.push(data[idx * n_features + f]);
            }
        } else {
            // 使用加权随机选择
            let rng_val: f32 = rng.gen_range(0.0..1.0);
            let mut cumsum = 0.0;
            let mut selected_idx = n_samples - 1; // 默认选择最后一个点
            
            for i in 0..n_samples {
                cumsum += distances[i] * distances[i] / total_dist_sq;
                if rng_val <= cumsum {
                    selected_idx = i;
                    break;
                }
            }
            
            for f in 0..n_features {
                centers.push(data[selected_idx * n_features + f]);
            }
        }
    }
    
    Ok(centers)
}

