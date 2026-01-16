use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;
use log::{debug, error, info, warn};
use tokio::runtime::Runtime;
use tokio::sync::Semaphore;
use futures::future;

use crate::data::DataBatch;
use crate::error::{Error, Result};
use crate::data::loader::config::{ImportConfig, BatchImportConfig};
use crate::data::loader::result::{ImportResult, BatchImportResult};
use crate::data::loader::importer::DataImporter;
use crate::data::loader::types::DataFormat;

/// 批量导入器
pub struct BatchImporter {
    /// 批量导入配置
    config: BatchImportConfig,
    /// 缓存最后导入的批次
    last_imported_batches: Vec<DataBatch>,
}

impl BatchImporter {
    /// 创建新的批量导入器
    pub fn new(config: BatchImportConfig) -> Self {
        Self {
            config,
            last_imported_batches: Vec::new(),
        }
    }

    /// 从基础配置创建批量导入器
    pub fn from_base_config(base_config: ImportConfig) -> Self {
        let batch_config = BatchImportConfig::new(base_config);
        Self::new(batch_config)
    }
    
    /// 设置超时时间
    pub fn with_timeout(mut self, seconds: u64) -> Self {
        self.config.timeout_seconds = seconds;
        self
    }
    
    /// 设置最大并发数
    pub fn with_max_concurrent(mut self, max: usize) -> Self {
        self.config.max_concurrent = max;
        self
    }

    /// 设置匹配模式
    pub fn with_pattern<S: Into<String>>(mut self, pattern: S) -> Self {
        self.config.pattern = Some(pattern.into());
        self
    }

    /// 设置是否递归处理子目录
    pub fn with_recursive(mut self, recursive: bool) -> Self {
        self.config.recursive = recursive;
        self
    }
    
    /// 获取所有已导入的批次
    pub fn get_imported_batches(&self) -> &[DataBatch] {
        &self.last_imported_batches
    }
    
    /// 从目录导入多个文件
    pub async fn import_from_directory<P: AsRef<Path>>(&mut self, directory: P) -> Result<BatchImportResult> {
        let dir_path = directory.as_ref();
        info!("开始从目录导入数据: {:?}, 模式: {:?}", dir_path, self.config.pattern.as_ref());
        
        if !dir_path.exists() || !dir_path.is_dir() {
            return Err(Error::invalid_input(format!("指定的路径不是有效目录: {:?}", dir_path)));
        }
        
        // 扫描目录，获取匹配的文件
        let files = match crate::data::loader::utils::scan_directory(
            dir_path, 
            self.config.pattern.as_deref(), 
            self.config.recursive
        ) {
            Ok(files) => files,
            Err(e) => {
                error!("扫描目录失败: {}", e);
                return Err(e);
            }
        };
        
        if files.is_empty() {
            warn!("在目录 {:?} 中没有找到匹配的文件", dir_path);
            return Ok(BatchImportResult::new());
        }
        
        info!("在目录 {:?} 中找到 {} 个文件", dir_path, files.len());
        
        // 保存文件数量，避免借用问题
        let total_files = files.len();
        
        // 创建导入任务
        let mut tasks = Vec::with_capacity(files.len());
        let mut batch_result = BatchImportResult::new();
        let semaphore = Arc::new(Semaphore::new(self.config.max_concurrent));
        
        for file in files {
            // 创建该文件的导入配置
            let mut file_config = self.config.base_config.clone();
            file_config.source_path = file.to_string_lossy().to_string();
            
            // 检测文件格式（如果未指定）
            if file_config.format.is_none() {
                match crate::data::loader::utils::detect_file_format(&file) {
                    Ok(format) => {
                        debug!("检测到文件格式 {:?}: {:?}", file, format);
                        file_config.format = Some(format.to_string());
                    },
                    Err(e) => {
                        warn!("无法检测文件格式 {:?}: {}", file, e);
                    }
                }
            }
            
            // 创建导入器
            let importer = DataImporter::new(file_config)
                .with_timeout(self.config.timeout_seconds);
            
            // 获取信号量许可
            let permit = semaphore.clone();
            let file_path = file.clone();
            
            // 创建导入任务
            let task = async move {
                // 获取并释放许可
                let _permit = permit.acquire().await.unwrap();
                
                // 执行导入
                let file_path_str = file_path.to_string_lossy().to_string();
                match importer.import() {
                    Ok(result) => {
                        if result.success {
                            info!("成功导入文件 {:?}: {}", file_path, result.message);
                        } else {
                            warn!("导入文件 {:?} 失败: {}", file_path, result.message);
                        }
                        (file_path_str, result)
                    },
                    Err(e) => {
                        error!("导入文件 {:?} 时发生错误: {}", file_path, e);
                        let mut result = ImportResult::default();
                        result.success = false;
                        result.message = format!("导入出错: {}", e);
                        result.errors.push(e.to_string());
                        (file_path_str, result)
                    }
                }
            };
            
            tasks.push(task);
        }
        
        // 使用超时执行所有任务
        let timeout_duration = Duration::from_secs(self.config.timeout_seconds);
        match tokio::time::timeout(
            timeout_duration,
            future::join_all(tasks)
        ).await {
            Ok(task_results) => {
                // 处理每个导入结果
                self.last_imported_batches.clear();
                
                let mut success_count = 0;
                let mut failed_count = 0;
                let mut total_records = 0;
                
                for (file_path, result) in task_results {
                    if result.success {
                        success_count += 1;
                        total_records += result.records_processed;
                        
                        // 如果有批次摘要，添加到导入批次列表
                        if let Some(batch_summary) = &result.batch_summary {
                            // 创建一个新的批次对象
                            let mut batch = DataBatch::new("imported", 0, batch_summary.record_count);
                            batch.id = Some(batch_summary.batch_id.clone());
                            // 添加到导入批次列表
                            self.last_imported_batches.push(batch);
                        }
                    } else {
                        failed_count += 1;
                    }
                    
                    // 添加到批量导入结果
                    batch_result.add_file_result(&file_path, result);
                }
                
                info!("批量导入完成: 成功={}, 失败={}, 总记录数={}",
                     success_count, failed_count, total_records);
                
                batch_result.summary.add_metadata("total_files", &total_files.to_string());
                batch_result.summary.add_metadata("successful_files", &success_count.to_string());
                batch_result.summary.add_metadata("failed_files", &failed_count.to_string());
                
                Ok(batch_result)
            },
            Err(_) => {
                error!("批量导入超时，超过了{}秒", timeout_duration.as_secs());
                Err(Error::timeout(format!("批量导入超时，超过了{}秒", timeout_duration.as_secs())))
            }
        }
    }
    
    /// 从目录导入数据（同步版本）
    pub fn import_from_directory_sync<P: AsRef<Path>>(&mut self, directory: P) -> Result<BatchImportResult> {
        // 创建运行时
        let runtime = Runtime::new()
            .map_err(|e| Error::processing(format!("创建异步运行时失败: {}", e)))?;
        
        // 执行异步导入操作
        runtime.block_on(self.import_from_directory(directory))
    }
    
    /// 合并所有已导入的批次
    pub fn merge_imported_batches(&self) -> Result<DataBatch> {
        if self.last_imported_batches.is_empty() {
            return Err(Error::invalid_state("没有可合并的批次"));
        }
        
        let mut merged_batch = self.last_imported_batches[0].clone();
        
        for batch in &self.last_imported_batches[1..] {
            merged_batch.merge(batch)?;
        }
        
        info!("合并了{}个批次，共{}条记录",
             self.last_imported_batches.len(),
             merged_batch.records.len());
        
        Ok(merged_batch)
    }
    
    /// 清除已导入的批次
    pub fn clear_imported_batches(&mut self) {
        self.last_imported_batches.clear();
        debug!("已清除所有导入批次");
    }
    
    /// 获取导入的批次数量
    pub fn imported_batch_count(&self) -> usize {
        self.last_imported_batches.len()
    }
    
    /// 获取导入的总记录数
    pub fn imported_record_count(&self) -> usize {
        self.last_imported_batches.iter()
            .map(|batch| batch.records.len())
            .sum()
    }
    
    /// 导出所有批次到目标位置
    pub fn export_batches<P: AsRef<Path>>(&self, target_dir: P, format: DataFormat) -> Result<Vec<PathBuf>> {
        let target_path = target_dir.as_ref();
        
        // 确保目标目录存在
        if !target_path.exists() {
            std::fs::create_dir_all(target_path)
                .map_err(|e| Error::io_error(format!("创建目标目录失败: {}", e)))?;
        }
        
        let mut exported_files = Vec::new();
        
        // 导出每个批次
        for (i, batch) in self.last_imported_batches.iter().enumerate() {
            let filename = format!("batch_{}.{}", i, format.file_extension());
            let output_path = target_path.join(filename);
            
            let saved_path = crate::data::loader::utils::save_batch_to_file(batch, &output_path, format.clone())?;
            exported_files.push(saved_path);
        }
        
        info!("成功导出 {} 个批次到 {:?}", exported_files.len(), target_path);
        
        Ok(exported_files)
    }
} 