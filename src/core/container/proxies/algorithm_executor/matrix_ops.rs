/// 矩阵运算模块
/// 
/// 提供矩阵乘法、转置、求逆、最小二乘法等数值计算功能

use crate::{Result, Error};
use rayon::prelude::*;

/// 矩阵乘法: C = A * B
/// A: [m, n], B: [n, p], C: [m, p]
/// 使用并行计算优化性能
pub(crate) fn matrix_multiply(
    a: &[f32], 
    a_rows: usize, 
    a_cols: usize,
    b: &[f32], 
    b_rows: usize, 
    b_cols: usize
) -> Result<Vec<f32>> {
    if a_cols != b_rows {
        return Err(Error::InvalidInput(
            format!("矩阵维度不匹配: A列数({}) != B行数({})", a_cols, b_rows)
        ));
    }
    
    if a.len() != a_rows * a_cols {
        return Err(Error::InvalidInput(
            format!("矩阵A数据大小不匹配: 期望 {}, 实际 {}", a_rows * a_cols, a.len())
        ));
    }
    
    if b.len() != b_rows * b_cols {
        return Err(Error::InvalidInput(
            format!("矩阵B数据大小不匹配: 期望 {}, 实际 {}", b_rows * b_cols, b.len())
        ));
    }
    
    // 对于小矩阵，使用串行计算；对于大矩阵，使用并行计算
    let threshold = 64; // 阈值：当矩阵行数超过64时使用并行计算
    
    if a_rows > threshold && b_cols > threshold {
        // 并行计算：并行处理结果矩阵的每一行
        let result: Vec<Vec<f32>> = (0..a_rows).into_par_iter()
            .map(|i| {
                (0..b_cols).map(|j| {
                    let mut sum = 0.0;
                    for k in 0..a_cols {
                        sum += a[i * a_cols + k] * b[k * b_cols + j];
                    }
                    sum
                }).collect()
            })
            .collect();
        
        // 展平二维向量为一维
        let flattened: Vec<f32> = result.into_iter().flatten().collect();
        Ok(flattened)
    } else {
        // 串行计算：适用于小矩阵
        let mut result = vec![0.0; a_rows * b_cols];
        
        for i in 0..a_rows {
            for j in 0..b_cols {
                let mut sum = 0.0;
                for k in 0..a_cols {
                    sum += a[i * a_cols + k] * b[k * b_cols + j];
                }
                result[i * b_cols + j] = sum;
            }
        }
        
        Ok(result)
    }
}

/// 矩阵转置
pub(crate) fn matrix_transpose(matrix: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    let mut result = vec![0.0; rows * cols];
    for i in 0..rows {
        for j in 0..cols {
            result[j * rows + i] = matrix[i * cols + j];
        }
    }
    result
}

/// 矩阵求逆（使用高斯消元法，仅支持小矩阵）
pub(crate) fn matrix_inverse(matrix: &[f32], n: usize) -> Result<Vec<f32>> {
    if matrix.len() != n * n {
        return Err(Error::InvalidInput(
            format!("矩阵必须是方阵: 期望大小 {}, 实际 {}", n * n, matrix.len())
        ));
    }
    
    // 创建增广矩阵 [A | I]
    let mut augmented = vec![0.0; n * (2 * n)];
    for i in 0..n {
        for j in 0..n {
            augmented[i * (2 * n) + j] = matrix[i * n + j];
            if i == j {
                augmented[i * (2 * n) + n + j] = 1.0;
            }
        }
    }
    
    // 高斯消元法
    for i in 0..n {
        // 找到主元
        let mut max_row = i;
        let mut max_val = augmented[i * (2 * n) + i].abs();
        for k in (i + 1)..n {
            let val = augmented[k * (2 * n) + i].abs();
            if val > max_val {
                max_val = val;
                max_row = k;
            }
        }
        
        if max_val < 1e-8 {
            return Err(Error::InvalidInput(
                format!("矩阵不可逆（奇异矩阵），主元值: {}", max_val)
            ));
        }
        
        // 交换行
        if max_row != i {
            for j in 0..(2 * n) {
                augmented.swap(i * (2 * n) + j, max_row * (2 * n) + j);
            }
        }
        
        // 归一化主元行
        let pivot = augmented[i * (2 * n) + i];
        for j in 0..(2 * n) {
            augmented[i * (2 * n) + j] /= pivot;
        }
        
        // 消元
        for k in 0..n {
            if k != i {
                let factor = augmented[k * (2 * n) + i];
                for j in 0..(2 * n) {
                    augmented[k * (2 * n) + j] -= factor * augmented[i * (2 * n) + j];
                }
            }
        }
    }
    
    // 提取逆矩阵
    let mut inverse = vec![0.0; n * n];
    for i in 0..n {
        for j in 0..n {
            inverse[i * n + j] = augmented[i * (2 * n) + n + j];
        }
    }
    
    Ok(inverse)
}

/// 最小二乘法求解: weights = (X^T * X)^(-1) * X^T * y
pub(crate) fn least_squares(
    x: &[f32], 
    n_samples: usize, 
    n_features: usize,
    y: &[f32]
) -> Result<(Vec<f32>, f32)> {
    if x.len() != n_samples * n_features {
        return Err(Error::InvalidInput(
            format!("特征矩阵大小不匹配: 期望 {}, 实际 {}", n_samples * n_features, x.len())
        ));
    }
    
    if y.len() != n_samples {
        return Err(Error::InvalidInput(
            format!("目标向量大小不匹配: 期望 {}, 实际 {}", n_samples, y.len())
        ));
    }
    
    // 计算 X^T * X
    let x_transposed = matrix_transpose(x, n_samples, n_features);
    let xtx = matrix_multiply(&x_transposed, n_features, n_samples, x, n_samples, n_features)?;
    
    // 计算 (X^T * X)^(-1)
    let xtx_inv = matrix_inverse(&xtx, n_features)?;
    
    // 计算 X^T * y
    let mut xty = vec![0.0; n_features];
    for i in 0..n_features {
        for j in 0..n_samples {
            xty[i] += x_transposed[i * n_samples + j] * y[j];
        }
    }
    
    // 计算 weights = (X^T * X)^(-1) * X^T * y
    let weights = matrix_multiply(&xtx_inv, n_features, n_features, &xty, n_features, 1)?;
    
    // 计算偏置: bias = mean(y) - mean(X * weights)
    let y_mean = y.iter().sum::<f32>() / n_samples as f32;
    let mut xw_mean = 0.0;
    for i in 0..n_samples {
        let mut sum = 0.0;
        for j in 0..n_features {
            sum += x[i * n_features + j] * weights[j];
        }
        xw_mean += sum;
    }
    xw_mean /= n_samples as f32;
    let bias = y_mean - xw_mean;
    
    Ok((weights, bias))
}

