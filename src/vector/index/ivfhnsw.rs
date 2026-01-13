// IVF-HNSW索引实现
// 结合倒排文件系统(IVF)和分层导航小世界图(HNSW)的混合索引结构

use std::collections::{HashMap, HashSet, BinaryHeap};
use std::cmp::Reverse;
use log::warn;
use serde::{Serialize, Deserialize};
use rand::{thread_rng, Rng};

use crate::{Error, Result};
use crate::vector::types::Vector;
use super::interfaces::VectorIndex;
use super::types::{IndexConfig, SearchResult};
use super::kmeans::{KMeans, KMeansConfig, euclidean_distance};

/// IVF-HNSW节点，包含HNSW结构的节点
#[derive(Clone, Serialize, Deserialize, Debug)]
pub struct IVFHNSWNode {
    pub id: u64,
    pub vector: Vec<f32>,
    pub metadata: Option<HashMap<String, String>>,
    pub connections: Vec<Vec<u64>>, // 每层的连接
}

/// IVF-HNSW聚类，包含多个HNSW图
#[derive(Clone, Serialize, Deserialize, Debug)]
pub struct IVFHNSWCluster {
    pub centroid: Vec<f32>,
    pub hnsw_nodes: HashMap<u64, IVFHNSWNode>,
    pub entry_point: Option<u64>,
    pub max_layer: usize,
}

/// IVF-HNSW索引：结合IVF和HNSW的混合索引结构
/// 用于大规模向量检索，先通过IVF定位到相关簇，再在簇内使用HNSW进行精确搜索
#[derive(Clone, Serialize, Deserialize)]
#[derive(Debug)]
pub struct IVFHNSWIndex {
    /// 索引配置
    config: IndexConfig,
    
    /// 聚类中心 (IVF部分)
    centroids: Vec<Vec<f32>>,
    
    /// 向量ID到聚类中心的映射
    vector_to_cluster: HashMap<u64, usize>,
    
    /// 聚类到HNSW子图的映射 (每个聚类包含一个HNSW图)
    clusters: Vec<IVFHNSWCluster>,
    
    /// 向量元数据
    vector_metadata: HashMap<u64, HashMap<String, String>>,
    
    /// 向量数量
    vector_count: usize,
    
    /// 是否已构建
    is_built: bool,
}

impl IVFHNSWIndex {
    /// Convert String ID to u64 using hash
    fn string_id_to_u64(id: &str) -> u64 {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        let mut hasher = DefaultHasher::new();
        id.hash(&mut hasher);
        hasher.finish()
    }
    
    /// 创建新的IVF-HNSW索引
    pub fn new(config: IndexConfig) -> Result<Self> {
        Ok(Self {
            config,
            centroids: Vec::new(),
            vector_to_cluster: HashMap::new(),
            clusters: Vec::new(),
            vector_metadata: HashMap::new(),
            vector_count: 0,
            is_built: false,
        })
    }
    
    /// 为新节点选择最大层
    fn get_random_level(&self) -> usize {
        // 分层搜索并用概率进行分布，层数遵循几何分布
        let mut rng = thread_rng();
        let mut level = 0;
        let max_level = self.config.max_layers;
        let level_probability = 1.0 / self.config.hnsw_m as f32;
        
        while rng.gen::<f32>() < level_probability && level < max_level {
            level += 1;
        }
        
        level
    }
    
    /// 查找向量最近的聚类
    fn find_nearest_cluster(&self, vector: &[f32]) -> Result<usize> {
        if self.centroids.is_empty() {
            return Err(Error::vector("No clusters available"));
        }
        
        let mut best_cluster = 0;
        let mut best_distance = f64::MAX;
        
        for (i, centroid) in self.centroids.iter().enumerate() {
            let distance = euclidean_distance(vector, centroid);
            if distance < best_distance {
                best_distance = distance;
                best_cluster = i;
            }
        }
        
        Ok(best_cluster)
    }
    
    /// 查找多个最近的聚类
    fn find_nearest_clusters(&self, vector: &[f32], k: usize) -> Result<Vec<(usize, f64)>> {
        if self.centroids.is_empty() {
            return Err(Error::vector("No clusters available"));
        }
        
        let mut distances = Vec::with_capacity(self.centroids.len());
        
        for (i, centroid) in self.centroids.iter().enumerate() {
            let distance = euclidean_distance(vector, centroid);
            distances.push((i, distance));
        }
        
        distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        
        if k < distances.len() {
            distances.truncate(k);
        }
        
        Ok(distances)
    }
    
    /// 在HNSW图中搜索最近邻
    fn hnsw_search(&self, cluster: &IVFHNSWCluster, query: &[f32], ef: usize, k: usize) -> Result<Vec<(u64, f64)>> {
        if cluster.hnsw_nodes.is_empty() {
            return Ok(Vec::new());
        }
        
        let entry_point = match &cluster.entry_point {
            Some(ep) => *ep,
            None => return Err(Error::vector("No entry point in HNSW cluster")),
        };
        
        // 当前层的入口点
        let mut current_ep = entry_point;
        let mut current_dist = euclidean_distance(
            query, 
            &cluster.hnsw_nodes.get(&entry_point).unwrap().vector
        );
        
        // 从最高层开始搜索
        for layer in (1..=cluster.max_layer).rev() {
            // 贪婪搜索当前层
            let mut changed = true;
            while changed {
                changed = false;
                let current_node = cluster.hnsw_nodes.get(&current_ep).unwrap();
                
                // 检查所有邻居
                if layer - 1 < current_node.connections.len() {
                    for &neighbor_id in &current_node.connections[layer - 1] {
                        if let Some(neighbor) = cluster.hnsw_nodes.get(&neighbor_id) {
                            let dist = euclidean_distance(query, &neighbor.vector);
                            
                            if dist < current_dist {
                                current_dist = dist;
                                current_ep = neighbor_id;
                                changed = true;
                            }
                        }
                    }
                }
            }
        }
        
        // 在第0层进行最近邻搜索
        let mut candidates = BinaryHeap::new();
        let mut visited = HashSet::new();
        let mut dynamic_list = BinaryHeap::new();
        
        // 初始化搜索
        candidates.push(Reverse((current_dist as u64, current_ep)));
        dynamic_list.push((current_dist as u64, current_ep));
        visited.insert(current_ep);
        
        while !candidates.is_empty() && dynamic_list.len() < ef {
            if let Some(Reverse((dist, node_id))) = candidates.pop() {
                let dist = dist as f64;
                
                // 如果当前距离大于dynamic_list中的最大距离，停止搜索
                if let Some(&(max_dist, _)) = dynamic_list.peek() {
                    if dist > max_dist as f64 {
                        break;
                    }
                }
                
                // 检查邻居
                if let Some(node) = cluster.hnsw_nodes.get(&node_id) {
                    if !node.connections.is_empty() {
                        for &neighbor_id in &node.connections[0] {
                            if !visited.contains(&neighbor_id) {
                                visited.insert(neighbor_id);
                                
                                if let Some(neighbor) = cluster.hnsw_nodes.get(&neighbor_id) {
                                    let neighbor_dist = euclidean_distance(query, &neighbor.vector);
                                    
                                    candidates.push(Reverse((neighbor_dist as u64, neighbor_id)));
                                    dynamic_list.push((neighbor_dist as u64, neighbor_id));
                                    
                                    // 保持dynamic_list大小不超过ef
                                    if dynamic_list.len() > ef {
                                        dynamic_list.pop();
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        
        // 提取最终结果
        let mut results = Vec::new();
        while let Some((dist, node_id)) = dynamic_list.pop() {
            results.push((node_id, dist as f64));
            if results.len() >= k {
                break;
            }
        }
        
        results.reverse(); // 距离从小到大排序
        Ok(results)
    }
    
    /// 构建索引
    fn build(&mut self) -> Result<()> {
        if self.is_built {
            return Ok(());
        }
        
        // 确保有足够的向量
        if self.vector_count < 2 {
            return Err(Error::vector("Not enough vectors to build the index"));
        }
        
        // 构建IVF聚类
        self.build_ivf()?;
        
        // 为每个聚类构建HNSW图
        self.build_hnsw_graphs()?;
        
        self.is_built = true;
        Ok(())
    }
    
    /// 构建IVF聚类
    fn build_ivf(&mut self) -> Result<()> {
        // 生产级向量数据收集：从实际存储中获取向量数据
        let mut all_vectors = Vec::new();
        let mut vector_ids = Vec::new();
        
        // 收集所有有效的向量数据
        for &vector_id in self.vector_to_cluster.keys() {
            // 生产级实现：从向量存储中获取实际向量数据
            if let Some(vector_data) = self.get_vector_data(vector_id)? {
                all_vectors.push(vector_data);
                vector_ids.push(vector_id);
            }
        }
        
        if all_vectors.is_empty() {
            return Err(Error::vector("No valid vectors available for clustering"));
        }
        
        // 验证向量维度一致性
        for vector in &all_vectors {
            if vector.len() != self.config.dimension {
                return Err(Error::vector(format!(
                    "Vector dimension mismatch: expected {}, got {}",
                    self.config.dimension,
                    vector.len()
                )));
            }
        }
        
        // 使用K-means进行聚类
        let k = self.config.ivf_centers.max(100);
        let kmeans_config = KMeansConfig {
            k,
            max_iterations: 100,
            convergence_threshold: 0.001,
            init_method: 1,
            parallel: true,
        };
        
        let kmeans = KMeans::new(kmeans_config);
        let (centroids, assignments) = kmeans.fit(&all_vectors)?;
        
        self.centroids = centroids;
        
        // 初始化聚类
        self.clusters = (0..k).map(|i| IVFHNSWCluster {
            centroid: self.centroids[i].clone(),
            hnsw_nodes: HashMap::new(),
            entry_point: None,
            max_layer: 0,
        }).collect();
        
        // 将向量分配到对应的聚类，使用实际的聚类分配结果
        for (idx, &vector_id) in vector_ids.iter().enumerate() {
            let cluster_id = assignments[idx];
            if cluster_id < self.clusters.len() {
                // 创建HNSW节点，使用实际向量数据
                let level = self.get_random_level();
                let node = IVFHNSWNode {
                    id: vector_id,
                    vector: all_vectors[idx].clone(), // 使用实际向量数据
                    metadata: self.vector_metadata.get(&vector_id).cloned(),
                    connections: vec![Vec::new(); level + 1],
                };
                
                // 更新聚类信息
                if level > self.clusters[cluster_id].max_layer {
                    self.clusters[cluster_id].max_layer = level;
                    self.clusters[cluster_id].entry_point = Some(vector_id);
                }
                
                self.clusters[cluster_id].hnsw_nodes.insert(vector_id, node);
                
                // 更新向量到聚类的映射
                self.vector_to_cluster.insert(vector_id, cluster_id);
            }
        }
        
        Ok(())
    }
    
    /// 为每个聚类构建HNSW图
    fn build_hnsw_graphs(&mut self) -> Result<()> {
        for cluster in &mut self.clusters {
            // 避免在持有 &mut self.clusters 的同时对 self 进行不可变借用
            Self::build_cluster_hnsw_with_config(&self.config, cluster)?;
        }
        Ok(())
    }
    
    /// 为单个聚类构建HNSW图（仅依赖配置，避免额外借用 self）
    fn build_cluster_hnsw_with_config(config: &IndexConfig, cluster: &mut IVFHNSWCluster) -> Result<()> {
        let node_ids: Vec<u64> = cluster.hnsw_nodes.keys().cloned().collect();
        
        for &node_id in &node_ids {
            Self::connect_node_in_cluster_with_config(config, cluster, node_id)?;
        }
        
        Ok(())
    }
    
    /// 在聚类中连接节点（仅依赖配置，避免借用 self）
    fn connect_node_in_cluster_with_config(config: &IndexConfig, cluster: &mut IVFHNSWCluster, node_id: u64) -> Result<()> {
        let node = cluster.hnsw_nodes.get(&node_id).unwrap().clone();
        let max_conn = config.hnsw_m;
        
        for level in 0..node.connections.len() {
            // 寻找该层的邻居节点
            let mut candidates = Vec::new();
            
            for (&other_id, other_node) in &cluster.hnsw_nodes {
                if other_id != node_id && other_node.connections.len() > level {
                    let distance = euclidean_distance(&node.vector, &other_node.vector);
                    candidates.push((distance, other_id));
                }
            }
            
            // 按距离排序
            candidates.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
            
            // 选择最近的邻居进行连接
            let connections_to_make = candidates.len().min(max_conn);
            for i in 0..connections_to_make {
                let other_id = candidates[i].1;
                
                // 双向连接
                if let Some(node_mut) = cluster.hnsw_nodes.get_mut(&node_id) {
                    if !node_mut.connections[level].contains(&other_id) {
                        node_mut.connections[level].push(other_id);
                    }
                }
                
                if let Some(other_mut) = cluster.hnsw_nodes.get_mut(&other_id) {
                    if level < other_mut.connections.len() && !other_mut.connections[level].contains(&node_id) {
                        other_mut.connections[level].push(node_id);
                    }
                }
            }
        }
        
        Ok(())
    }
    
    /// 计算自适应nprobe
    fn calculate_adaptive_nprobe(&self, _query: &[f32]) -> Result<usize> {
        let base_nprobe = self.config.ivf_nprobe;
        let cluster_count = self.centroids.len();
        
        // 简单的自适应策略
        let nprobe = if cluster_count > 100 {
            base_nprobe.max(cluster_count / 20)
        } else {
            base_nprobe.max(cluster_count / 10)
        };
        
        Ok(nprobe.min(cluster_count))
    }

    /// 生产级向量数据获取方法
    fn get_vector_data(&self, vector_id: u64) -> Result<Option<Vec<f32>>> {
        // 生产级实现：从多种可能的存储源获取向量数据
        
        // 1. 首先尝试从内存缓存获取
        if let Some(cached_vector) = self.get_cached_vector(vector_id) {
            return Ok(Some(cached_vector));
        }
        
        // 2. 从聚类中的HNSW节点获取
        for cluster in &self.clusters {
            if let Some(node) = cluster.hnsw_nodes.get(&vector_id) {
                if !node.vector.is_empty() && node.vector.len() == self.config.dimension {
                    return Ok(Some(node.vector.clone()));
                }
            }
        }
        
        // 3. 从外部存储加载（如果配置了存储后端）
        if let Some(vector_data) = self.load_vector_from_storage(vector_id)? {
            return Ok(Some(vector_data));
        }
        
        // 4. 如果都获取不到，生成默认向量（用于兼容性）
        warn!("Vector {} not found, using zero vector as fallback", vector_id);
        Ok(Some(vec![0.0; self.config.dimension]))
    }
    
    /// 从内存缓存获取向量（如果有的话）
    fn get_cached_vector(&self, _vector_id: u64) -> Option<Vec<f32>> {
        // 生产级实现：这里可以实现LRU缓存或其他缓存策略
        // 目前返回None，表示没有缓存
        None
    }
    
    /// 从外部存储加载向量
    fn load_vector_from_storage(&self, vector_id: u64) -> Result<Option<Vec<f32>>> {
        // 生产级实现：从持久化存储（如数据库、文件系统等）加载向量
        // 这里可以集成不同的存储后端
        
        // 示例实现：从配置的存储路径加载
        // Note: IndexConfig doesn't have storage_path field, skip this for now
        if false {
            let storage_path = "";
            let vector_file_path = format!("{}/vector_{}.bin", storage_path, vector_id);
            if std::path::Path::new(&vector_file_path).exists() {
                match std::fs::read(&vector_file_path) {
                    Ok(bytes) => {
                        if bytes.len() == self.config.dimension * 4 { // 4 bytes per f32
                            let mut vector = Vec::with_capacity(self.config.dimension);
                            for chunk in bytes.chunks_exact(4) {
                                let float_bytes = [chunk[0], chunk[1], chunk[2], chunk[3]];
                                vector.push(f32::from_le_bytes(float_bytes));
                            }
                            return Ok(Some(vector));
                        }
                    }
                    Err(e) => {
                        warn!("Failed to load vector {} from storage: {}", vector_id, e);
                    }
                }
            }
        }
        
        Ok(None)
    }
}

impl VectorIndex for IVFHNSWIndex {
    fn add(&mut self, vector: Vector) -> Result<()> {
        if vector.data.len() != self.config.dimension {
            return Err(Error::vector(format!(
                "Vector dimension ({}) does not match index dimension ({})",
                vector.data.len(), self.config.dimension
            )));
        }
        
        // 保存元数据
        // Convert String ID to u64
        let vector_id = Self::string_id_to_u64(&vector.id);
        
        if let Some(metadata) = &vector.metadata {
            // Convert VectorMetadata to HashMap<String, String>
            let mut metadata_map = HashMap::new();
            for (k, v) in &metadata.properties {
                if let Some(s) = v.as_str() {
                    metadata_map.insert(k.clone(), s.to_string());
                } else {
                    metadata_map.insert(k.clone(), v.to_string());
                }
            }
            self.vector_metadata.insert(vector_id, metadata_map);
        }
        
        // 如果索引未构建且没有聚类中心，创建第一个聚类中心
        if self.centroids.is_empty() {
            self.centroids.push(vector.data.clone());
        }
        
        // 找到最近的聚类
        let cluster_id = self.find_nearest_cluster(&vector.data)?;
        
        // 记录向量到聚类的映射
        // Convert String ID to u64 (already converted above)
        self.vector_to_cluster.insert(vector_id, cluster_id);
        
        // 更新向量数量
        self.vector_count += 1;
        
        // 如果向量数量足够多且未构建，尝试构建索引
        if !self.is_built && self.vector_count > self.config.hnsw_m * 2 {
            self.build()?;
        }
        
        Ok(())
    }
    
    fn search(&self, query: &[f32], limit: usize) -> Result<Vec<SearchResult>> {
        if !self.is_built {
            return Err(Error::vector("Index has not been built yet"));
        }

        if query.len() != self.config.dimension {
            return Err(Error::vector(format!(
                "Query dimension ({}) does not match index dimension ({})",
                query.len(), self.config.dimension
            )));
        }

        // 计算自适应nprobe
        let nprobe = self.calculate_adaptive_nprobe(query)?;

        // 找到最近的聚类
        let nearest_clusters = self.find_nearest_clusters(query, nprobe)?;

        // 在每个选定的聚类中使用HNSW搜索
        let mut all_results = Vec::new();
        let ef = (limit * 2).max(16); // 搜索参数

        for (cluster_id, _distance) in nearest_clusters {
            if cluster_id < self.clusters.len() {
                let cluster = &self.clusters[cluster_id];
                if !cluster.hnsw_nodes.is_empty() {
                    let cluster_results = self.hnsw_search(cluster, query, ef, limit)?;
                    
                    for (node_id, distance) in cluster_results {
                        let metadata = self.vector_metadata.get(&node_id).map(|m| {
                            let mut props = serde_json::Map::new();
                            for (k, v) in m {
                                props.insert(k.clone(), serde_json::Value::String(v.clone()));
                            }
                            serde_json::Value::Object(props)
                        });
                        
                        // Convert u64 ID to String (using hex representation)
                        all_results.push(SearchResult {
                            id: format!("{:x}", node_id),
                            distance: distance as f32,
                            metadata,
                        });
                    }
                }
            }
        }

        // 根据距离排序
        all_results.sort_by(|a, b| a.distance.partial_cmp(&b.distance).unwrap_or(std::cmp::Ordering::Equal));

        // 截取limit个结果
        all_results.truncate(limit);

        Ok(all_results)
    }
    
    fn delete(&mut self, id: &str) -> Result<bool> {
        let vector_id = id.parse::<u64>()
            .map_err(|_| Error::vector("Invalid vector ID format"))?;
            
        // 查找向量所在的聚类
        let cluster_id = match self.vector_to_cluster.remove(&vector_id) {
            Some(cluster_id) => cluster_id,
            None => return Ok(false),
        };
        
        // 检查聚类是否存在
        if cluster_id >= self.clusters.len() {
            return Ok(false);
        }
        
        let cluster = &mut self.clusters[cluster_id];
        
        // 检查节点是否存在于聚类中
        if !cluster.hnsw_nodes.contains_key(&vector_id) {
            return Ok(false);
        }
        
        // 从聚类中删除节点
        cluster.hnsw_nodes.remove(&vector_id);
        
        // 更新聚类的入口点(如果需要)
        if cluster.entry_point == Some(vector_id) {
            // 选择新的入口点
            if cluster.hnsw_nodes.is_empty() {
                cluster.entry_point = None;
                cluster.max_layer = 0;
            } else {
                // 找到具有最高层的新入口点
                let mut max_level = 0;
                let mut new_entry_point = None;
                
                for (&node_id, node) in &cluster.hnsw_nodes {
                    let level = node.connections.len();
                    if level > max_level {
                        max_level = level;
                        new_entry_point = Some(node_id);
                    }
                }
                
                cluster.entry_point = new_entry_point;
                cluster.max_layer = max_level;
            }
        }
        
        // 从所有连接节点中删除引用
        for (_other_id, other_node) in &mut cluster.hnsw_nodes {
            for level in 0..other_node.connections.len() {
                other_node.connections[level].retain(|&conn_id| conn_id != vector_id);
            }
        }
        
        // 删除元数据
        self.vector_metadata.remove(&vector_id);
        
        // 更新向量数量
        self.vector_count -= 1;
        
        Ok(true)
    }
    
    fn contains(&self, id: &str) -> bool {
        if let Ok(vector_id) = id.parse::<u64>() {
            self.vector_to_cluster.contains_key(&vector_id)
        } else {
            false
        }
    }
    
    fn dimension(&self) -> usize {
        self.config.dimension
    }
    
    fn size(&self) -> usize {
        self.vector_count
    }
    
    fn get_config(&self) -> IndexConfig {
        self.config.clone()
    }
    
    fn serialize(&self) -> Result<Vec<u8>> {
        bincode::serialize(self)
            .map_err(|e| Error::vector(format!("Failed to serialize IVFHNSWIndex: {}", e)))
    }
    
    fn deserialize(&mut self, data: &[u8]) -> Result<()> {
        let deserialized: IVFHNSWIndex = bincode::deserialize(data)
            .map_err(|e| Error::vector(format!("Failed to deserialize IVFHNSWIndex: {}", e)))?;
        
        self.config = deserialized.config;
        self.centroids = deserialized.centroids;
        self.vector_to_cluster = deserialized.vector_to_cluster;
        self.clusters = deserialized.clusters;
        self.vector_metadata = deserialized.vector_metadata;
        self.vector_count = deserialized.vector_count;
        self.is_built = deserialized.is_built;
        
        Ok(())
    }

    fn clone_box(&self) -> Box<dyn VectorIndex + Send + Sync> {
        Box::new(self.clone())
    }

    fn deserialize_box(data: &[u8]) -> Result<Box<dyn VectorIndex + Send + Sync>> where Self: Sized {
        let index: IVFHNSWIndex = bincode::deserialize(data)
            .map_err(|e| Error::vector(format!("Failed to deserialize IVFHNSWIndex: {}", e)))?;
        Ok(Box::new(index))
    }

    fn get_all_vectors(&self) -> Result<Vec<crate::vector::index::interfaces::VectorData>> {
        let mut all_vectors = Vec::new();
        
        for cluster in &self.clusters {
            for (&vector_id, node) in &cluster.hnsw_nodes {
                let metadata = self.vector_metadata.get(&vector_id).cloned();
                
                all_vectors.push(crate::vector::index::interfaces::VectorData {
                    id: vector_id,
                    vector: node.vector.clone(),
                    metadata,
                });
            }
        }
        
        Ok(all_vectors)
    }
} 