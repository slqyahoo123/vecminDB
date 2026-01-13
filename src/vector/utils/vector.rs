use crate::Result;

/// 计算特征向量之间的相似度
/// 
/// 使用余弦相似度计算两个特征向量之间的相似度
pub fn feature_similarity(vec1: &[f32], vec2: &[f32]) -> Result<f32> {
    if vec1.len() != vec2.len() {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            format!("特征向量长度不同: {} vs {}", vec1.len(), vec2.len())
        ).into());
    }
    
    let mut dot_product = 0.0;
    let mut norm1 = 0.0;
    let mut norm2 = 0.0;
    
    for i in 0..vec1.len() {
        dot_product += vec1[i] * vec2[i];
        norm1 += vec1[i] * vec1[i];
        norm2 += vec2[i] * vec2[i];
    }
    
    norm1 = norm1.sqrt();
    norm2 = norm2.sqrt();
    
    if norm1 < 1e-6 || norm2 < 1e-6 {
        return Ok(0.0);
    }
    
    Ok(dot_product / (norm1 * norm2))
}

/// 规范化特征向量
pub fn normalize_vector(vec: &mut [f32]) -> Result<()> {
    let mut norm = 0.0;
    
    for &val in vec.iter() {
        norm += val * val;
    }
    
    norm = norm.sqrt();
    
    if norm < 1e-6 {
        return Ok(()); // 向量接近零向量，不进行规范化
    }
    
    for val in vec.iter_mut() {
        *val /= norm;
    }
    
    Ok(())
}

/// 向量平均
pub fn vector_average(vectors: &[Vec<f32>]) -> Result<Vec<f32>> {
    if vectors.is_empty() {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            "向量列表为空"
        ).into());
    }
    
    let dim = vectors[0].len();
    let mut result = vec![0.0; dim];
    
    for vec in vectors {
        if vec.len() != dim {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "向量维度不一致"
            ).into());
        }
        
        for (i, &val) in vec.iter().enumerate() {
            result[i] += val;
        }
    }
    
    let count = vectors.len() as f32;
    for val in &mut result {
        *val /= count;
    }
    
    Ok(result)
} 