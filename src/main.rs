//! VecminDB HTTP Server
//!
//! Standalone HTTP server for VecminDB

use vecmindb::{Config, Result};

#[cfg(feature = "http-server")]
#[actix_rt::main]
async fn main() -> Result<()> {
    // Initialize logger
    env_logger::init();
    
    log::info!("Starting VecminDB HTTP Server...");
    
    // Load configuration
    let config = Config::default();
    
    log::info!(
        "Server will listen on {}:{}",
        config.server.host,
        config.server.port
    );
    
    // Initialize vector service
    use std::sync::Arc;
    use vecmindb::vector::storage::storage::VectorStorageManager;
    use vecmindb::vector::index::{IndexConfig, IndexType};
    use vecmindb::vector::core::operations::SimilarityMetric;
    
    // 创建默认索引配置
    let index_config = IndexConfig {
        index_type: IndexType::Flat,
        metric: SimilarityMetric::Cosine,
        dimension: 128,
        ..Default::default()
    };
    
    let vector_storage = VectorStorageManager::new(&config.storage.path, index_config)?;
    let vector_service: Arc<dyn vecmindb::vector::VectorService> = Arc::new(vector_storage);
    
    // Initialize VectorDB for collection management
    let vector_db = vecmindb::vector::VectorDB::new(&config.storage.path)?;
    let vector_db_arc = Arc::new(tokio::sync::RwLock::new(vector_db));
    
    // Initialize API server state
    let mut api_state = vecmindb::api::handlers::init_api_server_state(vector_service.clone())?;
    // 使用Arc::get_mut需要确保这是唯一的引用
    if let Some(state) = Arc::get_mut(&mut api_state) {
        state.vector_db = Some(vector_db_arc.clone());
    } else {
        // 如果无法获取可变引用，创建一个新的状态
        let mut new_state = vecmindb::api::server::ApiServerState::new();
        new_state.init_vector_api(vector_service.clone())?;
        new_state.vector_db = Some(vector_db_arc.clone());
        api_state = Arc::new(new_state);
    }
    
    // Start HTTP server
    use actix_web::{App, HttpServer, web};
    
    let host = config.server.host.clone();
    let port = config.server.port;
    let workers = config.server.workers;
    let api_state_clone = api_state.clone();
    
    HttpServer::new(move || {
        App::new()
            .app_data(web::Data::new(api_state_clone.clone()))
            .configure(vecmindb::api::configure_routes)
    })
    .workers(workers)
    .bind((host.as_str(), port))?
    .run()
    .await?;
    
    log::info!("Server stopped.");
    
    Ok(())
}

#[cfg(not(feature = "http-server"))]
fn main() {
    eprintln!("HTTP server feature is not enabled.");
    eprintln!("Please compile with: cargo build --features http-server");
    std::process::exit(1);
}

