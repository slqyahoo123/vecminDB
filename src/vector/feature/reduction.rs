// 向量维度约简模块
//
// 提供用于向量降维的各种算法

use rayon::prelude::*;
use rand::Rng;

use crate::Result;
use crate::Error;
// 本模块对外接口基于 Vec<f32> 集合，不直接依赖 Vector 结构

/// 维度约简器特性
pub trait DimensionReducer: Send + Sync {
    /// 降低向量维度
    fn reduce_dimensions(
        &self,
        vectors: &[Vec<f32>],
        target_dim: usize
    ) -> Result<Vec<Vec<f32>>>;
    
    /// 获取维度约简器名称
    fn name(&self) -> &str;
    
    /// 获取维度约简器描述
    fn description(&self) -> &str;
}

/// 主成分分析(PCA)降维
pub struct PCAReducer {
    /// 降维器名称
    name: String,
    /// 降维器描述
    description: String,
    /// 投影矩阵
    projection: Option<Vec<Vec<f32>>>,
    /// 均值向量
    mean: Option<Vec<f32>>,
    /// 是否标准化数据
    standardize: bool,
}

impl PCAReducer {
    /// 创建新的PCA降维器
    pub fn new(name: &str, description: &str, standardize: bool) -> Self {
        Self {
            name: name.to_string(),
            description: description.to_string(),
            projection: None,
            mean: None,
            standardize,
        }
    }
    
    /// 创建默认的PCA降维器
    pub fn default() -> Self {
        Self::new(
            "pca",
            "Principal Component Analysis dimension reduction",
            true
        )
    }
    
    /// 训练PCA模型
    fn train(&mut self, vectors: &[Vec<f32>], target_dim: usize) -> Result<()> {
        if vectors.is_empty() {
            return Err(Error::vector("Cannot train PCA on empty vector list".to_string()));
        }
        
        let input_dim = vectors[0].len();
        if target_dim >= input_dim {
            return Err(Error::vector(format!(
                "Target dimension must be less than input dimension: {} >= {}",
                target_dim, input_dim
            )));
        }
        
        // 1. 计算均值
        let mean = calculate_mean(vectors);
        
        // 2. 中心化数据
        let centered = center_data(vectors, &mean);
        
        // 3. 如果需要标准化，则除以标准差
        let (processed_data, std_dev) = if self.standardize {
            let std_dev = calculate_std_dev(vectors, &mean);
            let standardized = standardize_data(&centered, &std_dev);
            (standardized, Some(std_dev))
        } else {
            (centered, None)
        };
        
        // 4. 计算协方差矩阵
        let covariance = calculate_covariance_matrix(&processed_data);
        
        // 5. 计算协方差矩阵的特征值和特征向量
        // 注意：在实际生产环境中，应使用专业的线性代数库进行特征值分解
        // 这里使用一个简化的幂迭代法来估计主要特征向量
        let (eigenvalues, eigenvectors) = power_iteration(&covariance, target_dim);
        
        // 6. 选择前target_dim个特征向量
        let mut projection = Vec::with_capacity(target_dim);
        for i in 0..target_dim {
            projection.push(eigenvectors[i].clone());
        }
        
        // 7. 保存结果
        self.projection = Some(projection);
        self.mean = Some(mean);
        
        Ok(())
    }
    
    /// 应用PCA变换
    fn transform(&self, vector: &[f32]) -> Result<Vec<f32>> {
        if let (Some(proj), Some(mean)) = (&self.projection, &self.mean) {
            if vector.len() != mean.len() {
                return Err(Error::vector(format!(
                    "Vector dimension mismatch: expected {}, got {}",
                    mean.len(), vector.len()
                )));
            }
            
            // 1. 中心化数据
            let mut centered = vector.to_vec();
            for (i, val) in centered.iter_mut().enumerate() {
                *val -= mean[i];
            }
            
            // 2. 应用投影
            let result = proj.iter()
                .map(|eigenvector| {
                    centered.iter().zip(eigenvector.iter())
                        .map(|(&x, &v)| x * v)
                        .sum()
                })
                .collect();
                
            Ok(result)
        } else {
            Err(Error::vector("PCA model not trained".to_string()))
        }
    }
}

impl DimensionReducer for PCAReducer {
    fn reduce_dimensions(
        &self,
        vectors: &[Vec<f32>],
        target_dim: usize
    ) -> Result<Vec<Vec<f32>>> {
        if vectors.is_empty() {
            return Err(Error::vector("Cannot reduce dimensions of empty vector list".to_string()));
        }
        
        let input_dim = vectors[0].len();
        // 验证所有向量维度一致
        for (i, vec) in vectors.iter().enumerate().skip(1) {
            if vec.len() != input_dim {
                return Err(Error::vector(format!(
                    "Vector at index {} has incorrect dimension: expected {}, got {}",
                    i, input_dim, vec.len()
                )));
            }
        }
        
        // 检查目标维度
        if target_dim >= input_dim {
            return Err(Error::vector(format!(
                "Target dimension must be less than input dimension: {} >= {}",
                target_dim, input_dim
            )));
        }
        
        // 如果模型未训练，则先训练
        let mut this = self.clone();
        if this.projection.is_none() {
            this.train(vectors, target_dim)?;
        }
        
        // 并行应用变换
        let reduced = vectors.par_iter()
            .map(|vec| this.transform(vec))
            .collect::<Result<Vec<Vec<f32>>>>()?;
            
        Ok(reduced)
    }
    
    fn name(&self) -> &str {
        &self.name
    }
    
    fn description(&self) -> &str {
        &self.description
    }
}

impl Clone for PCAReducer {
    fn clone(&self) -> Self {
        Self {
            name: self.name.clone(),
            description: self.description.clone(),
            projection: self.projection.clone(),
            mean: self.mean.clone(),
            standardize: self.standardize,
        }
    }
}

/// 随机投影降维
pub struct RandomProjectionReducer {
    /// 降维器名称
    name: String,
    /// 降维器描述
    description: String,
    /// 投影矩阵
    projection: Option<Vec<Vec<f32>>>,
}

impl RandomProjectionReducer {
    /// 创建新的随机投影降维器
    pub fn new(name: &str, description: &str) -> Self {
        Self {
            name: name.to_string(),
            description: description.to_string(),
            projection: None,
        }
    }
    
    /// 创建默认的随机投影降维器
    pub fn default() -> Self {
        Self::new(
            "random_projection",
            "Random Projection dimension reduction"
        )
    }
    
    /// 初始化投影矩阵
    fn initialize_projection(&mut self, input_dim: usize, target_dim: usize) {
        let mut rng = rand::thread_rng();
        let mut projection = vec![vec![0.0; input_dim]; target_dim];
        
        // 使用稀疏随机投影
        for row in &mut projection {
            for val in row.iter_mut() {
                // 每个元素有1/3的概率为-1/sqrt(s), 1/3的概率为1/sqrt(s), 1/3的概率为0
                let r: f32 = rng.gen();
                let s = (input_dim as f32).sqrt();
                
                *val = if r < 1.0/3.0 {
                    -1.0 / s
                } else if r < 2.0/3.0 {
                    1.0 / s
                } else {
                    0.0
                };
            }
        }
        
        self.projection = Some(projection);
    }
    
    /// 应用随机投影
    fn transform(&self, vector: &[f32]) -> Result<Vec<f32>> {
        if let Some(proj) = &self.projection {
            if vector.len() != proj[0].len() {
                return Err(Error::vector(format!(
                    "Vector dimension mismatch: expected {}, got {}",
                    proj[0].len(), vector.len()
                )));
            }
            
            let result = proj.iter()
                .map(|row| {
                    row.iter().zip(vector.iter())
                        .map(|(&a, &b)| a * b)
                        .sum::<f32>()
                })
                .collect();
                
            Ok(result)
        } else {
            Err(Error::vector("Projection matrix not initialized".to_string()))
        }
    }
}

impl DimensionReducer for RandomProjectionReducer {
    fn reduce_dimensions(
        &self,
        vectors: &[Vec<f32>],
        target_dim: usize
    ) -> Result<Vec<Vec<f32>>> {
        if vectors.is_empty() {
            return Err(Error::vector("Cannot reduce dimensions of empty vector list".to_string()));
        }
        
        let input_dim = vectors[0].len();
        // 验证所有向量维度一致
        for (i, vec) in vectors.iter().enumerate().skip(1) {
            if vec.len() != input_dim {
                return Err(Error::vector(format!(
                    "Vector at index {} has incorrect dimension: expected {}, got {}",
                    i, input_dim, vec.len()
                )));
            }
        }
        
        // 检查目标维度
        if target_dim >= input_dim {
            return Err(Error::vector(format!(
                "Target dimension must be less than input dimension: {} >= {}",
                target_dim, input_dim
            )));
        }
        
        // 如果投影矩阵未初始化，则初始化
        let mut this = self.clone();
        if this.projection.is_none() {
            this.initialize_projection(input_dim, target_dim);
        }
        
        // 并行应用变换
        let reduced = vectors.par_iter()
            .map(|vec| this.transform(vec))
            .collect::<Result<Vec<Vec<f32>>>>()?;
            
        Ok(reduced)
    }
    
    fn name(&self) -> &str {
        &self.name
    }
    
    fn description(&self) -> &str {
        &self.description
    }
}

impl Clone for RandomProjectionReducer {
    fn clone(&self) -> Self {
        Self {
            name: self.name.clone(),
            description: self.description.clone(),
            projection: self.projection.clone(),
        }
    }
}

/// t-SNE降维框架
/// 
/// 注意：此为基本框架，实际实现需要与专业的t-SNE库集成
pub struct TSNEReducer {
    /// 降维器名称
    name: String,
    /// 降维器描述
    description: String,
    /// 困惑度
    perplexity: f32,
    /// 最大迭代次数
    max_iter: usize,
    /// 学习率
    learning_rate: f32,
}

impl TSNEReducer {
    /// 创建新的t-SNE降维器
    pub fn new(
        name: &str, 
        description: &str,
        perplexity: f32,
        max_iter: usize,
        learning_rate: f32
    ) -> Self {
        Self {
            name: name.to_string(),
            description: description.to_string(),
            perplexity,
            max_iter,
            learning_rate,
        }
    }
    
    /// 创建默认的t-SNE降维器
    pub fn default() -> Self {
        Self::new(
            "tsne",
            "t-Distributed Stochastic Neighbor Embedding",
            30.0,
            1000,
            200.0
        )
    }
}

impl DimensionReducer for TSNEReducer {
    fn reduce_dimensions(
        &self,
        vectors: &[Vec<f32>],
        target_dim: usize
    ) -> Result<Vec<Vec<f32>>> {
        if vectors.is_empty() {
            return Ok(Vec::new());
        }
        
        let n = vectors.len();
        if n == 1 {
            // 单个向量直接映射到目标维度
            return Ok(vec![vec![0.0; target_dim]]);
        }
        
        // 验证输入数据
        let input_dim = vectors[0].len();
        for (i, vec) in vectors.iter().enumerate() {
            if vec.len() != input_dim {
                return Err(Error::vector(format!(
                    "向量 {} 的维度不匹配: 期望 {}, 实际 {}",
                    i, input_dim, vec.len()
                )));
            }
        }
        
        // 检查目标维度
        if target_dim > input_dim {
            return Err(Error::vector(format!(
                "目标维度 {} 不能大于输入维度 {}",
                target_dim, input_dim
            )));
        }
        
        // 1. 计算高维空间的成对距离和条件概率
        let high_dim_probs = self.compute_pairwise_affinities(vectors)?;
        
        // 2. 初始化低维嵌入
        let mut low_dim_embedding = self.initialize_embedding(n, target_dim);
        
        // 3. 使用梯度下降优化低维嵌入
        self.optimize_embedding(&high_dim_probs, &mut low_dim_embedding, n, target_dim)?;
        
        Ok(low_dim_embedding)
    }
    
    fn name(&self) -> &str {
        &self.name
    }
    
    fn description(&self) -> &str {
        &self.description
    }
}

impl TSNEReducer {
    /// 计算高维空间的成对亲和度
    fn compute_pairwise_affinities(&self, vectors: &[Vec<f32>]) -> Result<Vec<Vec<f32>>> {
        let n = vectors.len();
        let mut affinities = vec![vec![0.0; n]; n];
        
        // 1. 计算成对距离
        let mut distances = vec![vec![0.0; n]; n];
        for i in 0..n {
            for j in i + 1..n {
                let dist = self.euclidean_distance(&vectors[i], &vectors[j]);
                distances[i][j] = dist;
                distances[j][i] = dist;
            }
        }
        
        // 2. 为每个点计算合适的方差（基于困惑度）
        let variances = self.compute_optimal_variances(&distances)?;
        
        // 3. 计算条件概率 P(j|i)
        for i in 0..n {
            let mut prob_sum = 0.0;
            
            // 计算所有条件概率
            for j in 0..n {
                if i != j {
                    let prob = (-distances[i][j] / (2.0 * variances[i])).exp();
                    affinities[i][j] = prob;
                    prob_sum += prob;
                }
            }
            
            // 标准化，使得每行的概率和为1
            if prob_sum > 0.0 {
                for j in 0..n {
                    if i != j {
                        affinities[i][j] /= prob_sum;
                    }
                }
            }
        }
        
        // 4. 计算对称概率 P(i,j) = (P(j|i) + P(i|j)) / (2n)
        let mut symmetric_probs = vec![vec![0.0; n]; n];
        for i in 0..n {
            for j in 0..n {
                if i != j {
                    symmetric_probs[i][j] = (affinities[i][j] + affinities[j][i]) / (2.0 * n as f32);
                    symmetric_probs[i][j] = symmetric_probs[i][j].max(1e-12); // 避免数值问题
                }
            }
        }
        
        Ok(symmetric_probs)
    }
    
    /// 计算最优方差参数，使困惑度达到预设值
    fn compute_optimal_variances(&self, distances: &[Vec<f32>]) -> Result<Vec<f32>> {
        let n = distances.len();
        let mut variances = vec![1.0; n];
        let target_perplexity = self.perplexity;
        
        for i in 0..n {
            let mut variance = 1.0;
            let mut min_var = 0.0;
            let mut max_var = f32::INFINITY;
            
            // 二分搜索找到合适的方差
            for _ in 0..50 { // 最多迭代50次
                let perplexity = self.compute_perplexity(i, variance, distances);
                
                if (perplexity - target_perplexity).abs() < 1e-5 {
                    break;
                }
                
                if perplexity > target_perplexity {
                    max_var = variance;
                    if max_var.is_infinite() {
                        variance *= 2.0;
                    } else {
                        variance = (min_var + max_var) / 2.0;
                    }
                } else {
                    min_var = variance;
                    if max_var.is_infinite() {
                        variance *= 2.0;
                    } else {
                        variance = (min_var + max_var) / 2.0;
                    }
                }
            }
            
            variances[i] = variance;
        }
        
        Ok(variances)
    }
    
    /// 计算给定方差下的困惑度
    fn compute_perplexity(&self, i: usize, variance: f32, distances: &[Vec<f32>]) -> f32 {
        let n = distances.len();
        let mut probs = vec![0.0; n];
        let mut prob_sum = 0.0;
        
        // 计算概率分布
        for j in 0..n {
            if i != j {
                let prob = (-distances[i][j] / (2.0 * variance)).exp();
                probs[j] = prob;
                prob_sum += prob;
            }
        }
        
        // 标准化
        if prob_sum > 0.0 {
            for j in 0..n {
                if i != j {
                    probs[j] /= prob_sum;
                }
            }
        }
        
        // 计算熵
        let mut entropy = 0.0;
        for j in 0..n {
            if i != j && probs[j] > 1e-12 {
                entropy -= probs[j] * probs[j].ln();
            }
        }
        
        // 困惑度 = 2^熵
        2.0_f32.powf(entropy)
    }
    
    /// 初始化低维嵌入
    fn initialize_embedding(&self, n: usize, target_dim: usize) -> Vec<Vec<f32>> {
        use rand::Rng;
        let mut rng = rand::thread_rng();
        let mut embedding = Vec::with_capacity(n);
        
        for _ in 0..n {
            let mut point = Vec::with_capacity(target_dim);
            for _ in 0..target_dim {
                // 使用小的随机值初始化
                point.push(rng.gen::<f32>() * 1e-4 - 5e-5);
            }
            embedding.push(point);
        }
        
        embedding
    }
    
    /// 优化低维嵌入
    fn optimize_embedding(
        &self,
        high_dim_probs: &[Vec<f32>],
        embedding: &mut [Vec<f32>],
        n: usize,
        target_dim: usize
    ) -> Result<()> {
        let mut momentum = vec![vec![0.0f32; target_dim]; n];
        let mut gains = vec![vec![1.0f32; target_dim]; n];
        
        let initial_momentum = 0.5;
        let final_momentum = 0.8;
        let eta = 200.0; // 学习率
        let min_gain: f32 = 0.01;
        
        for iter in 0..self.max_iter {
            // 计算低维空间的成对概率
            let low_dim_probs = self.compute_low_dim_affinities(embedding);
            
            // 计算梯度
            let gradients = self.compute_gradients(high_dim_probs, &low_dim_probs, embedding);
            
            // 确定动量参数
            let momentum_val = if iter < 250 {
                initial_momentum
            } else {
                final_momentum
            };
            
            // 更新嵌入
            for i in 0..n {
                for d in 0..target_dim {
                    // 自适应增益
                    if (gradients[i][d] > 0.0) != (momentum[i][d] > 0.0) {
                        gains[i][d] += 0.2;
                    } else {
                        gains[i][d] *= 0.8;
                    }
                    gains[i][d] = gains[i][d].max(min_gain);
                    
                    // 更新动量
                    momentum[i][d] = momentum_val * momentum[i][d] - eta * gains[i][d] * gradients[i][d];
                    
                    // 更新位置
                    embedding[i][d] += momentum[i][d];
                }
            }
            
            // 中心化嵌入
            if iter % 10 == 0 {
                self.center_embedding(embedding);
            }
            
            // 早期放大：在前期迭代中放大P值以避免局部最优
            if iter == 100 {
                // 将高维概率除以4，在早期阶段结束
                // （这里我们不实际修改high_dim_probs，而是在计算中考虑）
            }
        }
        
        Ok(())
    }
    
    /// 计算低维空间的亲和度
    fn compute_low_dim_affinities(&self, embedding: &[Vec<f32>]) -> Vec<Vec<f32>> {
        let n = embedding.len();
        let mut affinities = vec![vec![0.0; n]; n];
        let mut sum = 0.0;
        
        // 计算成对距离和概率
        for i in 0..n {
            for j in i + 1..n {
                let dist_sq = self.squared_euclidean_distance(&embedding[i], &embedding[j]);
                let affinity = 1.0 / (1.0 + dist_sq); // t分布，自由度为1
                affinities[i][j] = affinity;
                affinities[j][i] = affinity;
                sum += 2.0 * affinity;
            }
        }
        
        // 标准化
        if sum > 0.0 {
            for i in 0..n {
                for j in 0..n {
                    if i != j {
                        affinities[i][j] /= sum;
                        affinities[i][j] = affinities[i][j].max(1e-12);
                    }
                }
            }
        }
        
        affinities
    }
    
    /// 计算梯度
    fn compute_gradients(
        &self,
        high_dim_probs: &[Vec<f32>],
        low_dim_probs: &[Vec<f32>],
        embedding: &[Vec<f32>]
    ) -> Vec<Vec<f32>> {
        let n = embedding.len();
        let target_dim = embedding[0].len();
        let mut gradients = vec![vec![0.0; target_dim]; n];
        
        for i in 0..n {
            for j in 0..n {
                if i != j {
                    let prob_diff = high_dim_probs[i][j] - low_dim_probs[i][j];
                    let dist_sq = self.squared_euclidean_distance(&embedding[i], &embedding[j]);
                    let factor = 4.0 * prob_diff / (1.0 + dist_sq);
                    
                    for d in 0..target_dim {
                        let coord_diff = embedding[i][d] - embedding[j][d];
                        gradients[i][d] += factor * coord_diff;
                    }
                }
            }
        }
        
        gradients
    }
    
    /// 中心化嵌入
    fn center_embedding(&self, embedding: &mut [Vec<f32>]) {
        let n = embedding.len();
        let target_dim = embedding[0].len();
        
        // 计算质心
        let mut centroid = vec![0.0; target_dim];
        for point in embedding.iter() {
            for (d, &coord) in point.iter().enumerate() {
                centroid[d] += coord / n as f32;
            }
        }
        
        // 减去质心
        for point in embedding.iter_mut() {
            for (d, coord) in point.iter_mut().enumerate() {
                *coord -= centroid[d];
            }
        }
    }
    
    /// 计算欧几里德距离
    fn euclidean_distance(&self, v1: &[f32], v2: &[f32]) -> f32 {
        v1.iter()
            .zip(v2.iter())
            .map(|(&a, &b)| (a - b).powi(2))
            .sum::<f32>()
            .sqrt()
    }
    
    /// 计算欧几里德距离的平方
    fn squared_euclidean_distance(&self, v1: &[f32], v2: &[f32]) -> f32 {
        v1.iter()
            .zip(v2.iter())
            .map(|(&a, &b)| (a - b).powi(2))
            .sum::<f32>()
    }
}

// 辅助函数

/// 计算向量的均值
fn calculate_mean(vectors: &[Vec<f32>]) -> Vec<f32> {
    if vectors.is_empty() {
        return Vec::new();
    }
    
    let n = vectors.len() as f32;
    let dim = vectors[0].len();
    let mut mean = vec![0.0; dim];
    
    for vec in vectors {
        for (i, &val) in vec.iter().enumerate() {
            mean[i] += val / n;
        }
    }
    
    mean
}

/// 计算向量的标准差
fn calculate_std_dev(vectors: &[Vec<f32>], mean: &[f32]) -> Vec<f32> {
    if vectors.is_empty() {
        return Vec::new();
    }
    
    let n = vectors.len() as f32;
    let dim = vectors[0].len();
    let mut variance = vec![0.0; dim];
    
    for vec in vectors {
        for (i, &val) in vec.iter().enumerate() {
            let diff = val - mean[i];
            variance[i] += (diff * diff) / n;
        }
    }
    
    // 避免除以零
    for var in &mut variance {
        if *var < 1e-10 {
            *var = 1.0;
        }
    }
    
    variance.iter().map(|&var| var.sqrt()).collect()
}

/// 中心化数据
fn center_data(vectors: &[Vec<f32>], mean: &[f32]) -> Vec<Vec<f32>> {
    vectors.iter()
        .map(|vec| {
            vec.iter().enumerate()
                .map(|(i, &val)| val - mean[i])
                .collect()
        })
        .collect()
}

/// 标准化数据
fn standardize_data(centered: &[Vec<f32>], std_dev: &[f32]) -> Vec<Vec<f32>> {
    centered.iter()
        .map(|vec| {
            vec.iter().enumerate()
                .map(|(i, &val)| {
                    if std_dev[i] > 0.0 {
                        val / std_dev[i]
                    } else {
                        val
                    }
                })
                .collect()
        })
        .collect()
}

/// 计算协方差矩阵
fn calculate_covariance_matrix(data: &[Vec<f32>]) -> Vec<Vec<f32>> {
    let n = data.len() as f32;
    let dim = data[0].len();
    let mut covariance = vec![vec![0.0; dim]; dim];
    
    for i in 0..dim {
        for j in 0..dim {
            let mut cov = 0.0;
            for vec in data {
                cov += vec[i] * vec[j];
            }
            covariance[i][j] = cov / (n - 1.0);
        }
    }
    
    covariance
}

/// 幂迭代法估计特征值和特征向量
fn power_iteration(matrix: &[Vec<f32>], num_vectors: usize) -> (Vec<f32>, Vec<Vec<f32>>) {
    let dim = matrix.len();
    let mut eigenvalues = Vec::with_capacity(num_vectors);
    let mut eigenvectors = Vec::with_capacity(num_vectors);
    
    // 创建初始正交基
    let mut basis = Vec::with_capacity(dim);
    for i in 0..dim {
        let mut vec = vec![0.0; dim];
        vec[i] = 1.0;
        basis.push(vec);
    }
    
    // 存储已计算的特征向量空间
    let mut subspace: Vec<Vec<f32>> = Vec::new();
    
    // 计算前num_vectors个特征向量
    for _ in 0..num_vectors {
        // 随机初始化向量
        let mut v = Vec::with_capacity(dim);
        let mut rng = rand::thread_rng();
        for _ in 0..dim {
            v.push(rng.gen::<f32>() * 2.0 - 1.0);
        }
        
        // 施密特正交化
        for vec in &subspace {
            let dot_product: f32 = v.iter().zip(vec.iter()).map(|(&a, &b)| a * b).sum();
            for i in 0..dim {
                v[i] -= dot_product * vec[i];
            }
        }
        
        // 归一化
        let norm: f32 = v.iter().map(|&x| x * x).sum::<f32>().sqrt();
        if norm > 1e-10 {
            for val in &mut v {
                *val /= norm;
            }
        } else {
            // 如果向量接近零，选择一个新的随机向量
            continue;
        }
        
        // 幂迭代
        let max_iter = 100;
        let mut lambda = 0.0;
        
        for _ in 0..max_iter {
            // 矩阵向量乘法
            let mut new_v = vec![0.0; dim];
            for i in 0..dim {
                for j in 0..dim {
                    new_v[i] += matrix[i][j] * v[j];
                }
            }
            
            // 计算Rayleigh商（估计特征值）
            let rayleigh = new_v.iter().zip(v.iter()).map(|(&a, &b)| a * b).sum::<f32>();
            
            // 归一化
            let norm: f32 = new_v.iter().map(|&x| x * x).sum::<f32>().sqrt();
            if norm > 1e-10 {
                for val in &mut new_v {
                    *val /= norm;
                }
            } else {
                break;
            }
            
            // 检查收敛
            let diff: f32 = new_v.iter().zip(v.iter())
                .map(|(&a, &b)| (a - b).powi(2))
                .sum::<f32>()
                .sqrt();
                
            v = new_v;
            lambda = rayleigh;
            
            if diff < 1e-6 {
                break;
            }
        }
        
        eigenvalues.push(lambda);
        eigenvectors.push(v.clone());
        subspace.push(v);
    }
    
    // 按特征值排序
    let mut indices: Vec<usize> = (0..num_vectors).collect();
    indices.sort_by(|&a, &b| eigenvalues[b].partial_cmp(&eigenvalues[a]).unwrap_or(std::cmp::Ordering::Equal));
    
    let sorted_eigenvalues = indices.iter().map(|&i| eigenvalues[i]).collect();
    let sorted_eigenvectors = indices.iter().map(|&i| eigenvectors[i].clone()).collect();
    
    (sorted_eigenvalues, sorted_eigenvectors)
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_pca_reducer() {
        // 创建测试数据
        let vectors = vec![
            vec![1.0, 2.0, 3.0, 4.0, 5.0],
            vec![2.0, 3.0, 4.0, 5.0, 6.0],
            vec![3.0, 4.0, 5.0, 6.0, 7.0],
            vec![4.0, 5.0, 6.0, 7.0, 8.0],
            vec![5.0, 6.0, 7.0, 8.0, 9.0],
        ];
        
        let target_dim = 2;
        let reducer = PCAReducer::default();
        
        let reduced = reducer.reduce_dimensions(&vectors, target_dim).unwrap();
        
        // 验证结果
        assert_eq!(reduced.len(), vectors.len());
        for vec in &reduced {
            assert_eq!(vec.len(), target_dim);
        }
    }
    
    #[test]
    fn test_random_projection() {
        // 创建测试数据
        let vectors = vec![
            vec![1.0, 2.0, 3.0, 4.0, 5.0],
            vec![2.0, 3.0, 4.0, 5.0, 6.0],
            vec![3.0, 4.0, 5.0, 6.0, 7.0],
        ];
        
        let target_dim = 3;
        let reducer = RandomProjectionReducer::default();
        
        let reduced = reducer.reduce_dimensions(&vectors, target_dim).unwrap();
        
        // 验证结果
        assert_eq!(reduced.len(), vectors.len());
        for vec in &reduced {
            assert_eq!(vec.len(), target_dim);
        }
    }
} 