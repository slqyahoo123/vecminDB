# VecminDB

A high-performance vector database with multiple index algorithms, optimizers, and auto-tuning capabilities.

## Features

- **Multiple Index Algorithms**: HNSW, IVF, PQ, LSH, VPTree, ANNOY, NGT, and more
- **Multi-Objective Optimization**: NSGA-II, MOEAD, MOPSO for index tuning
- **Auto-Tuning**: Automatic parameter optimization for best performance
- **Parallel Processing**: Built-in support for parallel vector operations
- **Flexible Storage**: Sled-based persistent storage with transaction support
- **Caching System**: Multi-tier caching with Redis support
- **Resource Management**: Intelligent memory and CPU resource allocation
- **Monitoring**: Built-in performance metrics and query statistics
- **Dual API**: Use as a Rust library or standalone HTTP service

## Quick Start

### As a Library

```rust
use vecmindb::{VectorDB, IndexType, SimilarityMetric};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create a new vector database
    let db = VectorDB::new("./data")?;
    
    // Create a collection with 128-dimensional vectors using Flat index
    // Note: HNSW is not fully integrated yet, using Flat as default
    let collection = db.create_collection("my_vectors", 128, IndexType::Flat)?;
    
    // Add vectors (collection is Arc<RwLock<VectorCollection>>)
    // Note: RwLock is from tokio::sync, so we use .await for lock acquisition
    {
        let mut coll = collection.write().await;
        coll.add_vector("vec1", &vec![0.1; 128], None).await?;
        coll.add_vector("vec2", &vec![0.2; 128], None).await?;
    }
    
    // Search for similar vectors
    let results = {
        let coll = collection.read().await;
        coll.search(
            &vec![0.15; 128], 
            10, 
            SimilarityMetric::Cosine
        ).await?
    };
    
    for result in results {
        println!("ID: {}, Score: {}", result.id, result.score);
    }
    
    Ok(())
}
```

### As an HTTP Server

```bash
# Start the server
cargo run --bin vecmindb-server --features http-server

# Create a collection
curl -X POST http://localhost:8080/api/v1/collections \
  -H "Content-Type: application/json" \
  -d '{"name": "my_vectors", "dimension": 128, "index_type": "HNSW"}'

# Add a vector
curl -X POST http://localhost:8080/api/v1/collections/my_vectors/vectors \
  -H "Content-Type: application/json" \
  -d '{"id": "vec1", "values": [0.1, 0.2, ...], "metadata": {}}'

# Search vectors
curl -X POST http://localhost:8080/api/v1/collections/my_vectors/search \
  -H "Content-Type: application/json" \
  -d '{"query": [0.15, 0.25, ...], "top_k": 10, "metric": "cosine"}'
```

## Installation

Add this to your `Cargo.toml`:

```toml
[dependencies]
vecmindb = "0.1"

# Optional features
vecmindb = { version = "0.1", features = ["http-server", "distributed"] }
```

## Supported Index Types

- **HNSW**: Hierarchical Navigable Small World graphs
- **IVF**: Inverted File Index
- **PQ**: Product Quantization
- **LSH**: Locality-Sensitive Hashing
- **VPTree**: Vantage-Point Tree
- **ANNOY**: Approximate Nearest Neighbors Oh Yeah
- **NGT**: Neighborhood Graph and Tree
- **Flat**: Brute-force exact search

## Performance

VecminDB is designed for high performance:

- Parallel vector operations using Rayon
- SIMD optimizations (optional)
- Efficient memory management
- Smart caching strategies
- Auto-tuning for optimal parameters

## Documentation

For detailed documentation, see:

- [API Documentation](https://docs.rs/vecmindb)
- [User Guide](docs/user-guide.md)
- [Examples](examples/)

## License

Licensed under either of:

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE))
- MIT license ([LICENSE-MIT](LICENSE-MIT))

at your option.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

