use super::*;
use std::collections::HashMap;
use std::sync::{Arc, RwLock};
use std::time::Duration;
use serde::{Serialize, Deserialize};
use chrono::{DateTime, Utc};

// ==================== API接口层实现 ====================

/// 文件上传管理器
pub struct FileUploadManager {
    pub(crate) upload_config: UploadConfig,
    pub(crate) upload_sessions: Arc<RwLock<HashMap<String, UploadSession>>>,
    pub(crate) file_processors: HashMap<String, Box<dyn FileProcessor>>,
}

/// 上传配置
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UploadConfig {
    pub max_file_size: usize,
    pub allowed_extensions: Vec<String>,
    pub upload_dir: String,
    pub chunk_size: usize,
    pub enable_resume: bool,
    pub max_concurrent_uploads: usize,
    pub cleanup_interval: Duration,
}

/// 上传会话
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UploadSession {
    pub session_id: String,
    pub file_name: String,
    pub file_size: usize,
    pub uploaded_chunks: Vec<usize>,
    pub total_chunks: usize,
    pub status: UploadStatus,
    pub created_at: DateTime<Utc>,
    pub last_activity: DateTime<Utc>,
    pub metadata: HashMap<String, String>,
}

/// 上传状态
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum UploadStatus {
    Pending,
    InProgress,
    Completed,
    Failed,
    Cancelled,
}

/// 文件处理器trait
pub trait FileProcessor: Send + Sync {
    fn process(&self, file_path: &str, metadata: &HashMap<String, String>) -> Result<ProcessedFile, crate::Error>;
    fn supported_extensions(&self) -> Vec<String>;
    fn validate_file(&self, file_path: &str) -> Result<bool, crate::Error>;
}

/// 处理后的文件
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProcessedFile {
    pub original_path: String,
    pub processed_path: String,
    pub file_type: String,
    pub size: usize,
    pub metadata: HashMap<String, String>,
    pub processing_time: Duration,
}

/// 图像文件处理器
pub struct ImageProcessor;

impl FileProcessor for ImageProcessor {
    fn process(&self, file_path: &str, metadata: &HashMap<String, String>) -> Result<ProcessedFile, crate::Error> {
        let start_time = std::time::Instant::now();
        let processed_path = format!("{}.processed", file_path);
        Ok(ProcessedFile {
            original_path: file_path.to_string(),
            processed_path,
            file_type: "image".to_string(),
            size: 1024 * 1024,
            metadata: metadata.clone(),
            processing_time: start_time.elapsed(),
        })
    }

    fn supported_extensions(&self) -> Vec<String> {
        vec![
            "jpg".to_string(),
            "jpeg".to_string(),
            "png".to_string(),
            "gif".to_string(),
            "bmp".to_string(),
        ]
    }

    fn validate_file(&self, file_path: &str) -> Result<bool, crate::Error> {
        Ok(file_path.ends_with(".jpg") || file_path.ends_with(".png"))
    }
}

/// 文本文件处理器
pub struct TextProcessor;

impl FileProcessor for TextProcessor {
    fn process(&self, file_path: &str, metadata: &HashMap<String, String>) -> Result<ProcessedFile, crate::Error> {
        let start_time = std::time::Instant::now();
        let processed_path = format!("{}.processed", file_path);
        Ok(ProcessedFile {
            original_path: file_path.to_string(),
            processed_path,
            file_type: "text".to_string(),
            size: 512 * 1024,
            metadata: metadata.clone(),
            processing_time: start_time.elapsed(),
        })
    }

    fn supported_extensions(&self) -> Vec<String> {
        vec!["txt".to_string(), "csv".to_string(), "json".to_string(), "xml".to_string()]
    }

    fn validate_file(&self, file_path: &str) -> Result<bool, crate::Error> {
        Ok(file_path.ends_with(".txt") || file_path.ends_with(".csv"))
    }
}

impl FileUploadManager {
    pub fn new(config: UploadConfig) -> Self {
        let mut file_processors: HashMap<String, Box<dyn FileProcessor>> = HashMap::new();
        file_processors.insert("image".to_string(), Box::new(ImageProcessor));
        file_processors.insert("text".to_string(), Box::new(TextProcessor));

        Self {
            upload_config: config,
            upload_sessions: Arc::new(RwLock::new(HashMap::new())),
            file_processors,
        }
    }

    pub async fn create_upload_session(&self, file_name: String, file_size: usize) -> Result<String, crate::Error> {
        if file_size > self.upload_config.max_file_size {
            return Err(crate::error::Error::Validation(format!(
                "File size {} exceeds maximum allowed size {}",
                file_size, self.upload_config.max_file_size
            )));
        }

        if let Some(extension) = file_name.split('.').last() {
            if !self
                .upload_config
                .allowed_extensions
                .contains(&extension.to_lowercase())
            {
                return Err(crate::error::Error::Validation(format!(
                    "File extension '{}' is not allowed",
                    extension
                )));
            }
        }

        let session_id = format!("upload_{}", Utc::now().timestamp_millis());
        let total_chunks = (file_size + self.upload_config.chunk_size - 1) / self.upload_config.chunk_size;

        let session = UploadSession {
            session_id: session_id.clone(),
            file_name,
            file_size,
            uploaded_chunks: Vec::new(),
            total_chunks,
            status: UploadStatus::Pending,
            created_at: Utc::now(),
            last_activity: Utc::now(),
            metadata: HashMap::new(),
        };

        let mut sessions = self.upload_sessions.write().unwrap();
        sessions.insert(session_id.clone(), session);

        Ok(session_id)
    }

    pub async fn upload_chunk(&self, session_id: &str, chunk_index: usize, chunk_data: Vec<u8>) -> Result<(), crate::Error> {
        let mut sessions = self.upload_sessions.write().unwrap();

        if let Some(session) = sessions.get_mut(session_id) {
            if chunk_index >= session.total_chunks {
                return Err(crate::error::Error::Validation(format!(
                    "Chunk index {} is out of range (0-{})",
                    chunk_index,
                    session.total_chunks - 1
                )));
            }

            if chunk_data.len() > self.upload_config.chunk_size {
                return Err(crate::error::Error::Validation(format!(
                    "Chunk size {} exceeds maximum chunk size {}",
                    chunk_data.len(), self.upload_config.chunk_size
                )));
            }

            if session.uploaded_chunks.contains(&chunk_index) {
                return Err(crate::error::Error::Validation(format!(
                    "Chunk {} has already been uploaded",
                    chunk_index
                )));
            }

            session.uploaded_chunks.push(chunk_index);
            session.last_activity = Utc::now();

            if session.uploaded_chunks.len() == session.total_chunks {
                session.status = UploadStatus::Completed;
            } else {
                session.status = UploadStatus::InProgress;
            }

            Ok(())
        } else {
            Err(crate::error::Error::NotFound(format!(
                "Upload session not found: {}",
                session_id
            )))
        }
    }

    pub async fn get_upload_status(&self, session_id: &str) -> Result<Option<UploadSession>, crate::Error> {
        let sessions = self.upload_sessions.read().unwrap();
        Ok(sessions.get(session_id).cloned())
    }

    pub async fn cancel_upload(&self, session_id: &str) -> Result<(), crate::Error> {
        let mut sessions = self.upload_sessions.write().unwrap();

        if let Some(session) = sessions.get_mut(session_id) {
            session.status = UploadStatus::Cancelled;
            session.last_activity = Utc::now();
            Ok(())
        } else {
            Err(crate::error::Error::NotFound(format!(
                "Upload session not found: {}",
                session_id
            )))
        }
    }

    pub async fn process_uploaded_file(&self, session_id: &str) -> Result<ProcessedFile, crate::Error> {
        let sessions = self.upload_sessions.read().unwrap();

        if let Some(session) = sessions.get(session_id) {
            if session.status != UploadStatus::Completed {
                return Err(crate::error::Error::Validation(format!(
                    "Upload session is not completed, current status: {:?}",
                    session.status
                )));
            }

            let file_extension = session
                .file_name
                .split('.')
                .last()
                .unwrap_or("")
                .to_lowercase();
            let processor = self.get_processor_for_extension(&file_extension)?;

            let file_path = format!("{}/{}", self.upload_config.upload_dir, session.file_name);

            processor.process(&file_path, &session.metadata)
        } else {
            Err(crate::error::Error::NotFound(format!(
                "Upload session not found: {}",
                session_id
            )))
        }
    }

    fn get_processor_for_extension(&self, extension: &str) -> Result<&Box<dyn FileProcessor>, crate::Error> {
        for (_file_type, processor) in &self.file_processors {
            if processor
                .supported_extensions()
                .contains(&extension.to_string())
            {
                return Ok(processor);
            }
        }

        Err(crate::error::Error::Validation(format!(
            "No processor found for file extension: {}",
            extension
        )))
    }

    pub async fn cleanup_expired_sessions(&self) -> Result<(), crate::Error> {
        let mut sessions = self.upload_sessions.write().unwrap();
        let now = Utc::now();
        let cleanup_threshold = now - chrono::Duration::hours(1);

        sessions.retain(|_, session| {
            session.last_activity > cleanup_threshold && session.status != UploadStatus::Completed
        });

        Ok(())
    }
}


