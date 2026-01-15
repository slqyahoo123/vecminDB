use crate::{
    Result,
    vector::{
        storage::storage::VectorStorageManager as VectorStorage,
        index::{IndexType, IndexConfig},
        search::VectorIndexFactory,
        core::operations::SimilarityMetric,
        storage::storage::VectorSearchRequest,
        VectorMetadata, VectorQuery, VectorCollection, VectorCollectionConfig
    },
    vector::index::interfaces::VectorIndex
};
// use std::path::Path;
use rand::Rng;
use std::collections::HashMap;
use std::time::Instant;
use rand::{SeedableRng};
use rand::rngs::StdRng;
use crate::vector::types::Vector;

/// 运行向量存储示例
pub fn run_vector_storage_example() -> Result<()> {
    // 创建临时目录
    let temp_dir = tempfile::tempdir()?;
    let storage_path = temp_dir.path().join("vector_storage");
    
    // 创建向量存储
    let mut config = IndexConfig::default();
    config.index_type = IndexType::HNSW; // 使用HNSW索引
    config.dimension = 128; // 128维向量
    config.metric = SimilarityMetric::Cosine; // 使用余弦相似度
    
    let storage = VectorStorage::new(&storage_path, config)?;
    
    // 生成随机向量
    let vectors = generate_random_vectors(1000, 128)?;
    
    // 批量插入向量
    storage.batch_insert(&vectors)?;
    
    println!("已插入 {} 个向量", storage.count()?);
    
    // 这里原本演示 benchmark / compare / optimize 等高级功能，当前版本的 VectorStorageManager
    // 提供的 API 与示例不再完全匹配，因此我们保留核心存储与搜索示例，避免调用不存在的方法。

    // 搜索向量
    let query = &vectors[0].data;
    let search_request = VectorSearchRequest {
        query: query.clone(),
        top_k: 5,
        filter: None,
        include_metadata: false,
        include_vectors: false,
    };
    
    let search_results = storage.search(&search_request)?;
    
    println!("\n搜索结果 (用时: {}ms):", search_results.took_ms);
    for (i, result) in search_results.results.iter().enumerate() {
        println!(
            "  {}. ID: {}, 相似度: {:.4}",
            i + 1,
            result.id,
            result.distance
        );
    }
    
    // 导出和导入索引
    println!("\n导出索引...");
    let index_data = storage.export_index()?;
    println!("索引大小: {} 字节", index_data.len());
    
    println!("导入索引...");
    storage.import_index(&index_data)?;
    println!("索引导入成功");
    
    Ok(())
}

/// 生成随机向量
fn generate_random_vectors(count: usize, dimension: usize) -> Result<Vec<Vector>> {
    let mut rng = rand::thread_rng();
    let mut vectors = Vec::with_capacity(count);
    
    for i in 0..count {
        let id = format!("vec_{}", i);
        let data = (0..dimension).map(|_| rng.gen::<f32>()).collect();
        
        let mut vector = Vector {
            id,
            data,
            metadata: None,
        };
        // Normalize the vector
        let norm: f32 = vector.data.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 0.0 {
            for val in &mut vector.data {
                *val /= norm;
            }
        }
        
        vectors.push(vector);
    }
    
    Ok(vectors)
}

/// 生成随机查询向量
fn generate_random_queries(count: usize, dimension: usize) -> Vec<Vec<f32>> {
    let mut rng = rand::thread_rng();
    let mut queries = Vec::with_capacity(count);
    
    for _ in 0..count {
        let query = (0..dimension).map(|_| rng.gen::<f32>()).collect();
        queries.push(query);
    }
    
    queries
}

/// VP-Tree索引使用示例
pub fn vptree_example() -> Result<()> {
    println!("=== VP-Tree索引使用示例 ===");
    
    // 1. 创建VP-Tree索引配置
    let config = IndexConfig {
        index_type: IndexType::VPTree,
        metric: SimilarityMetric::Euclidean,
        dimension: 128,
        ..Default::default()
    };
    
    // 2. 创建VP-Tree索引
    let mut index = VectorIndexFactory::create_index(config.clone())?;
    println!("VP-Tree索引创建成功");
    
    // 3. 生成测试向量
    let num_vectors = 10_000;
    let dimensions = 128;
    let vectors = generate_random_vectors(num_vectors, dimensions)?;
    println!("生成 {} 个随机向量", num_vectors);
    
    // 4. 添加向量到索引
    let start = Instant::now();
    index.batch_insert(&vectors)?;
    let insert_duration = start.elapsed();
    println!("向量添加完成，耗时: {:?}", insert_duration);
    
    // 5. 执行向量搜索
    let mut rng = StdRng::seed_from_u64(42);
    let mut query = Vec::with_capacity(dimensions);
    for _ in 0..dimensions {
        query.push(rng.gen_range(0.0..1.0));
    }
    
    // 归一化查询向量
    let norm: f32 = query.iter().map(|x| x * x).sum::<f32>().sqrt();
    for val in &mut query {
        *val /= norm;
    }
    
    println!("执行向量搜索...");
    let start = Instant::now();
    let results = index.search(&query, 10)?;
    let search_duration = start.elapsed();
    println!("搜索完成，耗时: {:?}", search_duration);
    
    // 6. 输出搜索结果
    println!("搜索结果:");
    for (i, result) in results.iter().enumerate() {
        println!("  {}. ID: {}, 相似度: {:.4}", i + 1, result.id, result.distance);
    }
    
    // 7. 与其他索引类型比较
    println!("\n=== 与HNSW索引比较 ===");
    
    // 创建HNSW索引
    let hnsw_config = IndexConfig {
        index_type: IndexType::HNSW,
        metric: SimilarityMetric::Euclidean,
        dimension: 128,
        ..Default::default()
    };
    let mut hnsw_index = VectorIndexFactory::create_index(hnsw_config.clone())?;
    
    // 添加向量到HNSW索引
    let start = Instant::now();
    hnsw_index.batch_insert(&vectors)?;
    let hnsw_insert_duration = start.elapsed();
    println!("HNSW索引添加向量耗时: {:?}", hnsw_insert_duration);
    
    // 使用HNSW索引搜索
    let start = Instant::now();
    let hnsw_results = hnsw_index.search(&query, 10)?;
    let hnsw_search_duration = start.elapsed();
    println!("HNSW索引搜索耗时: {:?}", hnsw_search_duration);
    
    // 比较结果
    println!("\n性能比较:");
    println!("  插入性能: VP-Tree / HNSW = {:.2}", 
             insert_duration.as_secs_f64() / hnsw_insert_duration.as_secs_f64());
    println!("  查询性能: VP-Tree / HNSW = {:.2}", 
             search_duration.as_secs_f64() / hnsw_search_duration.as_secs_f64());
    
    // 8. 结果质量比较
    println!("\n结果质量比较:");
    let mut common_count = 0;
    for vp_result in &results {
        if hnsw_results.iter().any(|r| r.id == vp_result.id) {
            common_count += 1;
        }
    }
    println!("  前10个结果中共有项: {}/10", common_count);
    
    // 9. 内存占用比较
    let vp_memory = index.get_memory_usage()?;
    let hnsw_memory = hnsw_index.get_memory_usage()?;
    println!("\n内存占用比较:");
    println!("  VP-Tree: {:.2} MB", vp_memory as f64 / (1024.0 * 1024.0));
    println!("  HNSW: {:.2} MB", hnsw_memory as f64 / (1024.0 * 1024.0));
    println!("  比例: VP-Tree / HNSW = {:.2}", vp_memory as f64 / hnsw_memory as f64);
    
    Ok(())
}

/// 在向量集合中使用VP-Tree索引
pub async fn vptree_collection_example() -> Result<()> {
    println!("=== 在向量集合中使用VP-Tree索引 ===");
    
    // 1. 创建向量集合配置
    let config = VectorCollectionConfig {
        name: "vptree_collection".to_string(),
        dimension: 128,
        index_type: IndexType::VPTree,
        metadata_schema: None,
    };
    
    // 2. 创建索引配置
    let index_config = IndexConfig {
        index_type: IndexType::VPTree,
        metric: SimilarityMetric::Euclidean,
        dimension: 128,
        ..Default::default()
    };
    
    // 3. 创建索引
    let index = VectorIndexFactory::create_index(index_config)?;
    
    // 4. 创建向量集合
    let mut collection = VectorCollection::new(config, index);
    println!("向量集合创建成功");
    
    // 5. 生成测试向量
    let num_vectors = 1_000;
    let dimensions = 128;
    let mut rng = StdRng::seed_from_u64(42);
    
    println!("添加 {} 个向量到集合...", num_vectors);
    let start = Instant::now();
    
    // 6. 添加向量到集合
    for i in 0..num_vectors {
        // 生成随机向量
        let mut vector_data = Vec::with_capacity(dimensions);
        for _ in 0..dimensions {
            vector_data.push(rng.gen_range(0.0..1.0));
        }
        
        // 归一化向量
        let norm: f32 = vector_data.iter().map(|x| x * x).sum::<f32>().sqrt();
        for val in &mut vector_data {
            *val /= norm;
        }
        
        // 创建元数据
        let mut properties = HashMap::new();
        properties.insert("category".to_string(), 
                       serde_json::Value::String(["A", "B", "C", "D"][i % 4].to_string()));
        properties.insert("value".to_string(), 
                       serde_json::Value::Number(serde_json::Number::from(i)));
        
        // 创建向量
        let vector = Vector {
            id: format!("vector_{}", i),
            data: vector_data,
            metadata: Some(VectorMetadata { properties }),
        };
        
        // 添加到集合
        let metadata_json = vector.metadata.as_ref().map(|m| {
            serde_json::json!({
                "properties": m.properties
            })
        });
        collection.add_vector(&vector.id, &vector.data, metadata_json.as_ref()).await?;
    }
    
    let add_duration = start.elapsed();
    println!("向量添加完成，耗时: {:?}", add_duration);
    
    // 7. 创建查询向量
    let mut query_vector = Vec::with_capacity(dimensions);
    for _ in 0..dimensions {
        query_vector.push(rng.gen_range(0.0..1.0));
    }
    
    // 归一化查询向量
    let norm: f32 = query_vector.iter().map(|x| x * x).sum::<f32>().sqrt();
    for val in &mut query_vector {
        *val /= norm;
    }
    
    // 8. 创建查询
    let query = VectorQuery {
        vector: query_vector.clone(),
        filter: None,
        top_k: 5,
        include_metadata: true,
        include_vectors: false,
    };
    
    // 9. 执行搜索
    println!("执行向量搜索...");
    let start = Instant::now();
    let results = collection.search(&query.vector, query.top_k, SimilarityMetric::Cosine).await?;
    let search_duration = start.elapsed();
    println!("搜索完成，耗时: {:?}", search_duration);
    
    // 10. 输出搜索结果
    println!("搜索结果:");
    for (i, result) in results.iter().enumerate() {
        println!("  {}. ID: {}, 相似度: {:.4}", i + 1, result.id, result.score);
        if let Some(ref metadata) = result.metadata {
            println!("     类别: {}", metadata.properties.get("category").unwrap());
            println!("     值: {}", metadata.properties.get("value").unwrap());
        }
    }
    
    // 11. 使用过滤器搜索
    let mut filter_properties = HashMap::new();
    filter_properties.insert("category".to_string(), 
                           serde_json::Value::String("A".to_string()));
    
    let filter_query = VectorQuery {
        vector: query_vector.clone(),
        filter: Some(VectorMetadata { properties: filter_properties }),
        top_k: 5,
        include_metadata: true,
        include_vectors: false,
    };
    
    println!("\n使用过滤器搜索 (category = 'A')...");
    let start = Instant::now();
    let filter_results = collection.search(&filter_query.vector, filter_query.top_k, SimilarityMetric::Cosine).await?;
    let filter_search_duration = start.elapsed();
    println!("过滤搜索完成，耗时: {:?}", filter_search_duration);
    
    // 12. 输出过滤搜索结果
    println!("过滤搜索结果:");
    for (i, result) in filter_results.iter().enumerate() {
        println!("  {}. ID: {}, 相似度: {:.4}", i + 1, result.id, result.score);
        if let Some(ref metadata) = result.metadata {
            println!("     类别: {}", metadata.properties.get("category").unwrap());
            println!("     值: {}", metadata.properties.get("value").unwrap());
        }
    }
    
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_vector_storage_example() {
        let result = run_vector_storage_example();
        assert!(result.is_ok());
    }
} 