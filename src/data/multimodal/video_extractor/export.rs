//! 视频特征导出模块
//!
//! 本模块提供了将提取的视频特征导出为各种格式的功能

use std::path::{Path, PathBuf};
use std::fs::{self, File};
use std::io::{Write, BufWriter};
use std::collections::HashMap;
use serde::{Serialize, Deserialize};
use serde_json;
use csv;
use uuid;
use byteorder;
use flate2;
use log::{info, warn};
use serde_json::json;

use super::types::VideoFeatureResult;
use super::error::VideoExtractionError;

/// 导出格式
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ExportFormat {
    /// CSV格式
    CSV,
    /// JSON格式
    JSON,
    /// 二进制格式
    Binary,
    /// NumPy格式
    NumPy,
    /// TensorFlow格式
    TensorFlow,
    /// ONNX模型格式
    ONNX,
    /// HDF5格式
    HDF5,
}

impl ExportFormat {
    /// 获取格式对应的文件扩展名
    pub fn file_extension(&self) -> &'static str {
        match self {
            ExportFormat::CSV => "csv",
            ExportFormat::JSON => "json",
            ExportFormat::Binary => "bin",
            ExportFormat::NumPy => "npy",
            ExportFormat::TensorFlow => "tf",
            ExportFormat::ONNX => "onnx",
            ExportFormat::HDF5 => "h5",
        }
    }
}

/// 导出选项
#[derive(Debug, Clone)]
pub struct ExportOptions {
    /// 导出格式
    pub format: ExportFormat,
    /// 是否包含元数据
    pub include_metadata: bool,
    /// 是否包含处理信息
    pub include_processing_info: bool,
    /// 是否压缩
    pub compress: bool,
    /// 批处理大小
    pub batch_size: Option<usize>,
    /// 自定义选项
    pub custom_options: HashMap<String, String>,
}

impl Default for ExportOptions {
    fn default() -> Self {
        Self {
            format: ExportFormat::CSV,
            include_metadata: true,
            include_processing_info: false,
            compress: false,
            batch_size: None,
            custom_options: HashMap::new(),
        }
    }
}

/// 导出特征到指定路径
pub fn export_features<P: AsRef<Path>>(
    results: Vec<&VideoFeatureResult>,
    output_path: P,
    options: ExportOptions,
) -> Result<(), VideoExtractionError> {
    info!("导出 {} 条特征到 {}", results.len(), output_path.as_ref().display());
    
    if results.is_empty() {
        return Err(VideoExtractionError::ExportError("没有特征可导出".to_string()));
    }
    
    // 确保输出目录存在
    if let Some(parent) = output_path.as_ref().parent() {
        std::fs::create_dir_all(parent).map_err(|e| 
            VideoExtractionError::ExportError(format!("创建输出目录失败: {}", e))
        )?;
    }
    
    // 设置正确的文件扩展名
    let ext = options.format.file_extension();
    let output_path_with_ext = if output_path.as_ref().extension().is_none() {
        output_path.as_ref().with_extension(ext)
    } else {
        output_path.as_ref().to_path_buf()
    };
    
    // 根据格式选择导出方式
    match options.format {
        ExportFormat::CSV => export_to_csv(results, output_path_with_ext, &options),
        ExportFormat::JSON => export_to_json(results, output_path_with_ext, &options),
        ExportFormat::Binary => export_to_binary(results, output_path_with_ext, &options),
        ExportFormat::NumPy => export_to_numpy(results, output_path_with_ext, &options),
        ExportFormat::TensorFlow => export_to_tensorflow(results, output_path_with_ext, &options),
        ExportFormat::ONNX => export_to_onnx(results, output_path_with_ext, &options),
        ExportFormat::HDF5 => export_to_hdf5(results, output_path_with_ext, &options),
    }
}

/// 导出特征到CSV格式
fn export_to_csv<P: AsRef<Path>>(results: Vec<&VideoFeatureResult>, path: P, options: &ExportOptions) -> Result<(), VideoExtractionError> {
    info!("导出特征到CSV格式: {}", path.as_ref().display());
    
    if results.is_empty() {
        return Err(VideoExtractionError::ExportError("没有特征可导出".to_string()));
    }
    
    // 确保输出目录存在
    if let Some(parent) = path.as_ref().parent() {
        std::fs::create_dir_all(parent).map_err(|e| 
            VideoExtractionError::ExportError(format!("创建输出目录失败: {}", e))
        )?;
    }
    
    // 创建CSV文件
    let file = File::create(path.as_ref())
        .map_err(|e| VideoExtractionError::FileError(format!("无法创建CSV文件: {}", e)))?;
    
    let mut writer = csv::Writer::from_writer(file);
    
    // 写入头部行
    let mut header = vec!["video_id", "feature_type", "dimension"];
    
    // 如果包含元数据，添加相应的字段
    if options.include_metadata {
        header.extend(&["filename", "width", "height", "duration", "fps", "codec"]);
    }
    
    // 如果包含处理信息，添加相应的字段
    if options.include_processing_info {
        header.extend(&["processing_time", "created_at"]);
    }
    
    // 写入特征索引
    let feature_dim = results[0].features.len();
    for i in 0..feature_dim {
        header.push(&format!("feature_{}", i));
    }
    
    writer.write_record(&header)
        .map_err(|e| VideoExtractionError::ExportError(format!("写入CSV头部失败: {}", e)))?;
    
    // 写入每个结果的数据行
    for result in &results {
        let video_id = result.metadata.as_ref().map(|m| m.id.clone()).unwrap_or_default();
        let mut row = vec![
            video_id,
            format!("{:?}", result.feature_type),
            result.dimensions.to_string(),
        ];
        
        // 添加元数据
        if options.include_metadata {
            let file_path = result.metadata.as_ref().map(|m| m.file_path.clone()).unwrap_or_default();
            row.push(file_path);
            row.push(result.metadata.as_ref().map(|m| m.width.to_string()).unwrap_or_default());
            row.push(result.metadata.as_ref().map(|m| m.height.to_string()).unwrap_or_default());
            row.push(result.metadata.as_ref().map(|m| m.duration.to_string()).unwrap_or_default());
            row.push(result.metadata.as_ref().map(|m| m.fps.to_string()).unwrap_or_default());
            row.push(result.metadata.as_ref().map(|m| m.codec.clone()).unwrap_or_default());
        }
        
        // 添加处理信息
        if options.include_processing_info {
            let processing_time = result.processing_info.as_ref().map(|info| info.extraction_time_ms).unwrap_or(0);
            row.push(processing_time.to_string());
            row.push(result.timestamp.to_string());
        }
        
        // 添加特征数据
        for &value in &result.features {
            row.push(value.to_string());
        }
        
        writer.write_record(&row)
            .map_err(|e| VideoExtractionError::ExportError(format!("写入CSV数据行失败: {}", e)))?;
    }
    
    writer.flush()
        .map_err(|e| VideoExtractionError::ExportError(format!("写入CSV数据失败: {}", e)))?;
    
    info!("成功导出 {} 条记录到CSV格式", results.len());
    Ok(())
}

/// 导出特征到JSON格式
fn export_to_json<P: AsRef<Path>>(results: Vec<&VideoFeatureResult>, path: P, options: &ExportOptions) -> Result<(), VideoExtractionError> {
    info!("导出特征到JSON格式: {}", path.as_ref().display());
    
    if results.is_empty() {
        return Err(VideoExtractionError::ExportError("没有特征可导出".to_string()));
    }
    
    // 确保输出目录存在
    if let Some(parent) = path.as_ref().parent() {
        std::fs::create_dir_all(parent).map_err(|e| 
            VideoExtractionError::ExportError(format!("创建输出目录失败: {}", e))
        )?;
    }
    
    // 创建JSON文件
    let file = File::create(path.as_ref())
        .map_err(|e| VideoExtractionError::FileError(format!("无法创建JSON文件: {}", e)))?;
    let writer = BufWriter::new(file);
    
    if options.include_metadata {
        // 导出完整结果（包括元数据）
        let output: Vec<serde_json::Value> = if options.include_processing_info {
            // 包含完整信息
            results.iter().map(|r| {
                let video_id = r.metadata.as_ref().map(|m| m.id.clone()).unwrap_or_default();
                json!({
                    "video_id": video_id,
                    "feature_type": format!("{:?}", r.feature_type),
                    "dimension": r.dimensions,
                    "metadata": r.metadata,
                    "features": r.features,
                    "processing_info": r.processing_info,
                    "timestamp": r.timestamp
                })
            }).collect()
        } else {
            // 不包含处理信息
            results.iter().map(|r| {
                let video_id = r.metadata.as_ref().map(|m| m.id.clone()).unwrap_or_default();
                json!({
                    "video_id": video_id,
                    "feature_type": format!("{:?}", r.feature_type),
                    "dimension": r.dimensions,
                    "metadata": r.metadata,
                    "features": r.features
                })
            }).collect()
        };
        
        serde_json::to_writer_pretty(writer, &output)
            .map_err(|e| VideoExtractionError::ExportError(format!("序列化JSON数据失败: {}", e)))?;
    } else {
        // 只导出特征数据
        let output: Vec<serde_json::Value> = results.iter().map(|r| {
            let video_id = r.metadata.as_ref().map(|m| m.id.clone()).unwrap_or_default();
            json!({
                "video_id": video_id,
                "feature_type": format!("{:?}", r.feature_type),
                "dimension": r.dimensions,
                "features": r.features
            })
        }).collect();
        
        serde_json::to_writer_pretty(writer, &output)
            .map_err(|e| VideoExtractionError::ExportError(format!("序列化JSON数据失败: {}", e)))?;
    }
    
    info!("成功导出 {} 条记录到JSON格式", results.len());
    Ok(())
}

/// 导出特征到二进制格式
fn export_to_binary<P: AsRef<Path>>(results: Vec<&VideoFeatureResult>, path: P, options: &ExportOptions) -> Result<(), VideoExtractionError> {
    info!("导出特征到二进制格式: {}", path.as_ref().display());
    
    if results.is_empty() {
        return Err(VideoExtractionError::ExportError("没有特征可导出".to_string()));
    }
    
    // 确保输出目录存在
    if let Some(parent) = path.as_ref().parent() {
        std::fs::create_dir_all(parent).map_err(|e| 
            VideoExtractionError::ExportError(format!("创建输出目录失败: {}", e))
        )?;
    }
    
    // 创建二进制文件
    let file = File::create(path.as_ref())
        .map_err(|e| VideoExtractionError::FileError(format!("无法创建二进制文件: {}", e)))?;
    let mut writer = BufWriter::new(file);
    
    // 使用byteorder写入数据
    use byteorder::{LittleEndian, WriteBytesExt};
    
    // 写入文件魔数 "VFDB" (Video Feature Database)
    writer.write_all(b"VFDB")
        .map_err(|e| VideoExtractionError::ExportError(format!("写入魔数失败: {}", e)))?;
    
    // 写入版本号 (1.0)
    writer.write_u8(1)
        .map_err(|e| VideoExtractionError::ExportError(format!("写入版本号失败: {}", e)))?;
    writer.write_u8(0)
        .map_err(|e| VideoExtractionError::ExportError(format!("写入版本号失败: {}", e)))?;
    
    // 写入结果数量
    writer.write_u32::<LittleEndian>(results.len() as u32)
        .map_err(|e| VideoExtractionError::ExportError(format!("写入结果数量失败: {}", e)))?;
    
    // 写入特征维度
    let feature_dim = results[0].features.len() as u32;
    writer.write_u32::<LittleEndian>(feature_dim)
        .map_err(|e| VideoExtractionError::ExportError(format!("写入特征维度失败: {}", e)))?;
    
    // 写入元数据和处理信息标志
    let include_metadata = if options.include_metadata { 1u8 } else { 0u8 };
    let include_processing = if options.include_processing_info { 1u8 } else { 0u8 };
    
    writer.write_u8(include_metadata)
        .map_err(|e| VideoExtractionError::ExportError(format!("写入元数据标志失败: {}", e)))?;
    writer.write_u8(include_processing)
        .map_err(|e| VideoExtractionError::ExportError(format!("写入处理信息标志失败: {}", e)))?;
    
    // 写入每个结果的数据
    for result in &results {
        // 写入视频ID (长度+数据)
        let video_id = result.metadata.as_ref().map(|m| m.id.as_bytes()).unwrap_or_default();
        writer.write_u16::<LittleEndian>(video_id.len() as u16)
            .map_err(|e| VideoExtractionError::ExportError(format!("写入视频ID长度失败: {}", e)))?;
        writer.write_all(video_id)
            .map_err(|e| VideoExtractionError::ExportError(format!("写入视频ID失败: {}", e)))?;
        
        // 写入特征类型
        writer.write_u8(result.feature_type.as_u8())
            .map_err(|e| VideoExtractionError::ExportError(format!("写入特征类型失败: {}", e)))?;
        
        // 写入特征向量
        for &value in &result.features {
            writer.write_f32::<LittleEndian>(value)
                .map_err(|e| VideoExtractionError::ExportError(format!("写入特征数据失败: {}", e)))?;
        }
        
        // 如果需要，写入元数据
        if options.include_metadata {
            // 序列化元数据
            let metadata_json = serde_json::to_vec(&result.metadata)
                .map_err(|e| VideoExtractionError::ExportError(format!("序列化元数据失败: {}", e)))?;
            
            // 写入元数据长度和数据
            writer.write_u32::<LittleEndian>(metadata_json.len() as u32)
                .map_err(|e| VideoExtractionError::ExportError(format!("写入元数据长度失败: {}", e)))?;
            writer.write_all(&metadata_json)
                .map_err(|e| VideoExtractionError::ExportError(format!("写入元数据失败: {}", e)))?;
        }
        
        // 如果需要，写入处理信息
        if options.include_processing_info {
            // 写入处理时间
            let processing_time = result.processing_info.as_ref().map(|info| info.extraction_time_ms as f64).unwrap_or(0.0);
            writer.write_f64::<LittleEndian>(processing_time)
                .map_err(|e| VideoExtractionError::ExportError(format!("写入处理时间失败: {}", e)))?;
            
            // 写入创建时间
            writer.write_u64::<LittleEndian>(result.timestamp)
                .map_err(|e| VideoExtractionError::ExportError(format!("写入创建时间失败: {}", e)))?;
        }
    }
    
    // 如果需要压缩数据
    if options.compress {
        info!("正在压缩二进制数据...");
        
        // 创建压缩文件
        let compressed_path = path.as_ref().with_extension("bin.gz");
        
        // 获取原始文件内容
        let data = std::fs::read(path.as_ref())
            .map_err(|e| VideoExtractionError::FileError(format!("读取原始文件失败: {}", e)))?;
        
        // 创建压缩文件
        let file = File::create(&compressed_path)
            .map_err(|e| VideoExtractionError::FileError(format!("创建压缩文件失败: {}", e)))?;
        
        // 使用flate2进行gzip压缩
        let mut encoder = flate2::write::GzEncoder::new(file, flate2::Compression::default());
        encoder.write_all(&data)
            .map_err(|e| VideoExtractionError::ExportError(format!("写入压缩数据失败: {}", e)))?;
        
        // 完成压缩
        encoder.finish()
            .map_err(|e| VideoExtractionError::ExportError(format!("完成压缩失败: {}", e)))?;
        
        // 删除原始文件
        std::fs::remove_file(path.as_ref())
            .map_err(|e| VideoExtractionError::FileError(format!("删除原始文件失败: {}", e)))?;
        
        info!("二进制文件已压缩为: {}", compressed_path.display());
    }
    
    info!("成功导出 {} 条记录到二进制格式", results.len());
    Ok(())
}

/// 导出特征到NumPy格式
fn export_to_numpy<P: AsRef<Path>>(results: Vec<&VideoFeatureResult>, path: P, options: &ExportOptions) -> Result<(), VideoExtractionError> {
    #[cfg(feature = "numpy")]
    {
        use std::io::{BufWriter, Write};
        use byteorder::{LittleEndian, WriteBytesExt};
        
        info!("导出特征到NumPy格式: {}", path.as_ref().display());
        
        if results.is_empty() {
            return Err(VideoExtractionError::ExportError("没有特征可导出".to_string()));
        }
        
        // 确保输出目录存在
        if let Some(parent) = path.as_ref().parent() {
            std::fs::create_dir_all(parent).map_err(|e| 
                VideoExtractionError::ExportError(format!("创建输出目录失败: {}", e))
            )?;
        }
        
        // 获取特征维度
        let feature_dim = results[0].features.len();
        let num_features = results.len();
        
        // 创建文件
        let file = File::create(path.as_ref())
            .map_err(|e| VideoExtractionError::FileError(format!("无法创建NumPy文件: {}", e)))?;
        let mut writer = BufWriter::new(file);
        
        // 写入NumPy文件头
        // 1. 魔数: "\x93NUMPY"
        writer.write_all(b"\x93NUMPY")
            .map_err(|e| VideoExtractionError::ExportError(format!("写入NumPy魔数失败: {}", e)))?;
        
        // 2. 版本: 主版本1, 次版本0
        writer.write_u8(1)
            .map_err(|e| VideoExtractionError::ExportError(format!("写入NumPy版本失败: {}", e)))?;
        writer.write_u8(0)
            .map_err(|e| VideoExtractionError::ExportError(format!("写入NumPy版本失败: {}", e)))?;
        
        // 3. 构建头部信息
        let mut header = format!(
            "{{'descr': '<f4', 'fortran_order': False, 'shape': ({}, {}), }}",
            num_features, feature_dim
        );
        
        // 4. 计算头部长度并使其16字节对齐
        let header_len = header.len();
        let padding = (16 - ((header_len + 10) % 16)) % 16;
        for _ in 0..padding {
            header.push(' ');
        }
        header.push('\n');
        
        // 5. 写入头部长度(小端序)
        let header_len = header.len() as u16;
        writer.write_u16::<LittleEndian>(header_len)
            .map_err(|e| VideoExtractionError::ExportError(format!("写入NumPy头部长度失败: {}", e)))?;
        
        // 6. 写入头部
        writer.write_all(header.as_bytes())
            .map_err(|e| VideoExtractionError::ExportError(format!("写入NumPy头部失败: {}", e)))?;
        
        // 7. 写入数据 (小端序)
        for result in &results {
            for &value in &result.features {
                writer.write_f32::<LittleEndian>(value)
                    .map_err(|e| VideoExtractionError::ExportError(
                        format!("写入NumPy数据失败: {}", e)
                    ))?;
            }
        }
        
        // 如果需要压缩
        if options.compress {
            info!("正在压缩NumPy文件...");
            
            // 创建压缩文件
            let compressed_path = path.as_ref().with_extension("npz");
            
            // 使用zip创建压缩文件
            let file = File::create(&compressed_path)
                .map_err(|e| VideoExtractionError::FileError(format!("无法创建压缩文件: {}", e)))?;
            
            // 创建zip文件
            let mut zip = zip::ZipWriter::new(file);
            let options = zip::write::FileOptions::default()
                .compression_method(zip::CompressionMethod::Deflated)
                .unix_permissions(0o755);
            
            // 添加原始文件
            let file_name = path.as_ref().file_name().unwrap_or_default().to_string_lossy().to_string();
            zip.start_file(file_name, options)
                .map_err(|e| VideoExtractionError::ExportError(format!("添加文件到压缩包失败: {}", e)))?;
            
            // 写入NumPy文件内容
            let data = std::fs::read(path.as_ref())
                .map_err(|e| VideoExtractionError::FileError(format!("读取NumPy文件失败: {}", e)))?;
            zip.write_all(&data)
                .map_err(|e| VideoExtractionError::ExportError(format!("写入压缩数据失败: {}", e)))?;
            
            // 完成压缩
            zip.finish()
                .map_err(|e| VideoExtractionError::ExportError(format!("完成压缩文件失败: {}", e)))?;
            
            // 删除原始文件
            std::fs::remove_file(path.as_ref())
                .map_err(|e| VideoExtractionError::FileError(format!("删除原始文件失败: {}", e)))?;
            
            info!("NumPy文件已压缩为: {}", compressed_path.display());
        }
        
        // 如果需要元数据，写入单独的JSON文件
        if options.include_metadata {
            let metadata_path = path.as_ref().with_extension("meta.json");
            let metadata = results.iter()
                .map(|r| &r.metadata)
                .collect::<Vec<_>>();
            
            let json = serde_json::to_string_pretty(&metadata)
                .map_err(|e| VideoExtractionError::ExportError(format!("序列化元数据失败: {}", e)))?;
            
            std::fs::write(&metadata_path, json)
                .map_err(|e| VideoExtractionError::ExportError(format!("写入元数据文件失败: {}", e)))?;
            
            info!("元数据已保存到: {}", metadata_path.display());
        }
        
        info!("成功导出 {} 条记录到NumPy格式", results.len());
        Ok(())
    }
    
    #[cfg(not(feature = "numpy"))]
    {
        warn!("NumPy导出功能未启用，请在编译时启用'numpy'特性");
        Err(VideoExtractionError::ExportError(
            "NumPy导出功能未启用，请在编译时启用'numpy'特性".to_string()
        ))
    }
}

/// 导出特征数据为TensorFlow SavedModel格式
fn export_as_tensorflow(
    result: &VideoFeatureResult, 
    output_path: &Path, 
    options: &ExportOptions
) -> Result<(), VideoExtractionError> {
    info!("导出特征到TensorFlow格式: {}", output_path.display());
    
    // 创建输出目录
    if let Some(parent) = output_path.parent() {
        std::fs::create_dir_all(parent).map_err(|e| 
            VideoExtractionError::ExportError(format!("创建目录失败: {}", e))
        )?;
    }
    
    // 生成TensorFlow SavedModel格式
    let model_dir = output_path.with_extension("saved_model");
    std::fs::create_dir_all(&model_dir).map_err(|e| 
        VideoExtractionError::ExportError(format!("创建模型目录失败: {}", e))
    )?;
    
    // 创建模型配置文件
    let config = serde_json::json!({
        "model_info": {
            "name": "video_feature_model",
            "version": "1.0.0",
            "description": "Video feature extraction model",
            "created_at": chrono::Utc::now().to_rfc3339()
        },
        "feature_info": {
            "feature_dim": result.features.len(),
            "feature_type": format!("{:?}", result.feature_type),
            "video_path": result.metadata.as_ref().map(|m| m.file_path.clone()).unwrap_or_default(),
            "frame_count": result.metadata.as_ref().map(|m| m.frame_count).unwrap_or(0),
            "duration": result.metadata.as_ref().map(|m| m.duration).unwrap_or(0.0),
            "fps": result.metadata.as_ref().map(|m| m.fps).unwrap_or(0.0)
        },
        "export_options": {
            "include_metadata": options.include_metadata,
            "include_processing_info": options.include_processing_info,
            "compress": options.compress
        }
    });
    
    // 保存配置文件
    let config_path = model_dir.join("config.json");
    let config_file = std::fs::File::create(&config_path).map_err(|e| 
        VideoExtractionError::ExportError(format!("创建配置文件失败: {}", e))
    )?;
    serde_json::to_writer_pretty(config_file, &config).map_err(|e| 
        VideoExtractionError::ExportError(format!("写入配置文件失败: {}", e))
    )?;
    
    // 保存特征数据
    let features_path = model_dir.join("features.bin");
    let mut features_file = std::fs::File::create(&features_path).map_err(|e| 
        VideoExtractionError::ExportError(format!("创建特征文件失败: {}", e))
    )?;
    
    // 写入特征数据
    // 将 f32 特征转换为字节
    let features_bytes: Vec<u8> = result.features.iter()
        .flat_map(|&f| f.to_le_bytes().to_vec())
        .collect();
    features_file.write_all(&features_bytes).map_err(|e| 
        VideoExtractionError::ExportError(format!("写入特征数据失败: {}", e))
    )?;
    
    // 如果需要包含元数据
    if options.include_metadata {
        let metadata_path = model_dir.join("metadata.json");
        let metadata_file = std::fs::File::create(&metadata_path).map_err(|e| 
            VideoExtractionError::ExportError(format!("创建元数据文件失败: {}", e))
        )?;
        serde_json::to_writer_pretty(metadata_file, &result.metadata).map_err(|e| 
            VideoExtractionError::ExportError(format!("写入元数据失败: {}", e))
        )?;
    }
    
    // 创建推理脚本
    let inference_script = format!(
r#"#!/usr/bin/env python3
"""
TensorFlow视频特征推理脚本
生成时间: {}
"""

import numpy as np
import json
import sys
from pathlib import Path

class VideoFeatureInference:
    def __init__(self, model_path: str):
        self.model_path = Path(model_path)
        self.config = {{}}
        self.features = None
        self.metadata = {{}}
        
        self._load_model()
    
    def _load_model(self):
        """加载模型"""
        # 加载配置
        config_path = self.model_path / "config.json"
        if config_path.exists():
            with open(config_path, 'r') as f:
                self.config = json.load(f)
        
        # 加载特征数据
        features_path = self.model_path / "features.bin"
        if features_path.exists():
            with open(features_path, 'rb') as f:
                features_bytes = f.read()
                # 假设特征数据是float32格式
                self.features = np.frombuffer(features_bytes, dtype=np.float32)
        
        # 加载元数据
        metadata_path = self.model_path / "metadata.json"
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                self.metadata = json.load(f)
    
    def get_features(self) -> np.ndarray:
        """获取特征数据"""
        return self.features
    
    def get_metadata(self) -> dict:
        """获取元数据"""
        return self.metadata
    
    def get_config(self) -> dict:
        """获取配置信息"""
        return self.config
    
    def predict(self, input_data: np.ndarray) -> np.ndarray:
        """执行推理（这里只是返回存储的特征）"""
        # 在实际应用中，这里会加载TensorFlow模型并执行推理
        # 这里简化实现，直接返回存储的特征
        return self.features

def main():
    if len(sys.argv) < 2:
        print("用法: python script.py <model_path>")
        sys.exit(1)
    
    model_path = sys.argv[1]
    inference = VideoFeatureInference(model_path)
    
    # 打印模型信息
    config = inference.get_config()
    print("模型信息:")
    print(json.dumps(config, indent=2, ensure_ascii=False))
    
    # 获取特征
    features = inference.get_features()
    if features is not None:
        print(f"特征维度: {{features.shape}}")
        print(f"特征数据: {{features[:10]}}...")  # 只显示前10个值

if __name__ == "__main__":
    main()
"#,
        chrono::Utc::now().to_rfc3339()
    );
    
    let script_path = model_dir.join("inference.py");
    let mut script_file = std::fs::File::create(&script_path).map_err(|e| 
        VideoExtractionError::ExportError(format!("创建推理脚本失败: {}", e))
    )?;
    script_file.write_all(inference_script.as_bytes()).map_err(|e| 
        VideoExtractionError::ExportError(format!("写入推理脚本失败: {}", e))
    )?;
    
    info!("成功导出特征到TensorFlow格式: {}", model_dir.display());
    Ok(())
}

/// 导出特征到ONNX格式
fn export_to_onnx<P: AsRef<Path>>(results: Vec<&VideoFeatureResult>, path: P, options: &ExportOptions) -> Result<(), VideoExtractionError> {
    #[cfg(feature = "onnx")]
    {
        use tract_onnx::prelude::*;
        
        info!("导出特征到ONNX格式: {}", path.as_ref().display());
        
        if results.is_empty() {
            return Err(VideoExtractionError::ExportError("没有特征可导出".to_string()));
        }
        
        // 确保目录存在
        if let Some(parent) = path.as_ref().parent() {
            std::fs::create_dir_all(parent).map_err(|e| 
                VideoExtractionError::ExportError(format!("创建目录失败: {}", e))
            )?;
        }
        
        // 获取特征维度
        let feature_dim = results[0].features.len();
        let num_features = results.len();
        
        // 准备特征数据
        let mut features_data = Vec::with_capacity(num_features * feature_dim);
        for result in &results {
            features_data.extend_from_slice(&result.features);
        }
        
        // 创建特征张量
        let features_tensor = tract_ndarray::Array::from_shape_vec(
            (num_features, feature_dim),
            features_data
        ).map_err(|e| VideoExtractionError::ExportError(
            format!("创建特征张量失败: {}", e)
        ))?;
        
        // 创建模型
        let mut model = tract_onnx::prelude::SimplePlan::new(tract_onnx::prelude::TypedModel::default());
        
        // 创建输入节点
        let mut typed_model = tract_onnx::prelude::TypedModel::default();
        
        // 添加输入
        let input_shape = tvec!(num_features as i64, feature_dim as i64);
        let input = typed_model.add_source(
            "input", 
            tract_onnx::prelude::TypedFact::dt_shape(
                tract_onnx::prelude::f32::datum_type(), 
                input_shape
            )
        ).map_err(|e| VideoExtractionError::ExportError(
            format!("添加输入节点失败: {}", e)
        ))?;
        
        // 添加恒等节点，即简单地传递特征
        let output = typed_model.wire_node(
            "output",
            tract_onnx::prelude::tract_core::ops::identity::Identity::default(),
            &[input]
        ).map_err(|e| VideoExtractionError::ExportError(
            format!("添加输出节点失败: {}", e)
        ))?[0];
        
        // 标记输出
        typed_model.set_output_outlets(&[output]).map_err(|e| 
            VideoExtractionError::ExportError(format!("设置输出失败: {}", e))
        )?;
        
        // 优化模型
        let optimized_model = typed_model.into_optimized().map_err(|e| 
            VideoExtractionError::ExportError(format!("优化模型失败: {}", e))
        )?;
        
        // 保存模型
        let mut file = std::fs::File::create(path.as_ref()).map_err(|e| 
            VideoExtractionError::ExportError(format!("创建ONNX文件失败: {}", e))
        )?;
        
        // 导出为ONNX格式
        tract_onnx::onnx().with_onnx_10().serialize(&optimized_model, &mut file).map_err(|e| 
            VideoExtractionError::ExportError(format!("序列化ONNX模型失败: {}", e))
        )?;
        
        // 如果需要保存元数据
        if options.include_metadata {
            let metadata_path = path.as_ref().with_extension("metadata.json");
            
            // 收集元数据
            let metadata = results.iter().map(|r| {
                json!({
                    "file_path": r.metadata.as_ref().map(|m| m.file_path.clone()).unwrap_or_default(),
                    "duration": r.metadata.as_ref().map(|m| m.duration).unwrap_or(0.0),
                    "width": r.metadata.as_ref().map(|m| m.width).unwrap_or(0),
                    "height": r.metadata.as_ref().map(|m| m.height).unwrap_or(0),
                    "fps": r.metadata.as_ref().map(|m| m.fps).unwrap_or(0.0),
                    "feature_type": format!("{:?}", r.feature_type),
                    "processing_time": r.processing_info.as_ref().map(|info| info.extraction_time_ms).unwrap_or(0)
                })
            }).collect::<Vec<_>>();
            
            // 写入元数据JSON文件
            let metadata_json = serde_json::to_string_pretty(&metadata).map_err(|e| 
                VideoExtractionError::ExportError(format!("序列化元数据失败: {}", e))
            )?;
            
            std::fs::write(&metadata_path, metadata_json).map_err(|e| 
                VideoExtractionError::ExportError(format!("写入元数据文件失败: {}", e))
            )?;
            
            info!("元数据已保存到: {}", metadata_path.display());
        }
        
        // 创建实例数据文件，可用于测试模型
        let instance_path = path.as_ref().with_extension("instance.npz");
        
        // 使用numpy格式保存实例数据
        let mut instance_file = std::fs::File::create(&instance_path).map_err(|e| 
            VideoExtractionError::ExportError(format!("创建实例文件失败: {}", e))
        )?;
        
        // 使用npyz库保存为numpy格式
        npyz::write_iter(&mut instance_file, "features", 
            features_tensor.shape(), 
            features_tensor.iter().cloned()
        ).map_err(|e| VideoExtractionError::ExportError(
            format!("写入实例数据失败: {}", e)
        ))?;
        
        info!("实例数据已保存到: {}", instance_path.display());
        info!("成功导出 {} 条记录到ONNX格式", results.len());
        
        Ok(())
    }
    
    #[cfg(not(feature = "onnx"))]
    {
        warn!("ONNX导出功能未启用，请在编译时启用'onnx'特性");
        Err(VideoExtractionError::ExportError(
            "ONNX导出功能未启用，请在编译时启用'onnx'特性".to_string()
        ))
    }
}

/// 导出特征数据为HDF5格式
fn export_as_hdf5(
    result: &VideoFeatureResult, 
    output_path: &Path, 
    options: &ExportOptions
) -> Result<(), VideoExtractionError> {
    info!("导出特征到HDF5格式: {}", output_path.display());
    
    // 创建输出目录
    if let Some(parent) = output_path.parent() {
        std::fs::create_dir_all(parent).map_err(|e| 
            VideoExtractionError::ExportError(format!("创建目录失败: {}", e))
        )?;
    }
    
    // 生成HDF5格式文件
    let hdf5_path = output_path.with_extension("h5");
    
    // 创建HDF5文件结构（使用JSON作为占位符）
    let hdf5_structure = serde_json::json!({
        "file_info": {
            "format": "HDF5",
            "version": "1.0.0",
            "created_at": chrono::Utc::now().to_rfc3339(),
            "description": "Video feature data in HDF5 format"
        },
        "datasets": {
            "features": {
                "data": result.features,
                "shape": [result.features.len()],
                "dtype": "float32",
                "compression": "gzip",
                "compression_opts": 6
            },
            "metadata": {
                "video_path": result.metadata.as_ref().map(|m| m.file_path.clone()).unwrap_or_default(),
                "feature_type": format!("{:?}", result.feature_type),
                "frame_count": result.metadata.as_ref().map(|m| m.frame_count).unwrap_or(0),
                "duration": result.metadata.as_ref().map(|m| m.duration).unwrap_or(0.0),
                "fps": result.metadata.as_ref().map(|m| m.fps).unwrap_or(0.0),
                "extraction_time": result.processing_info.as_ref().map(|info| info.extraction_time_ms).unwrap_or(0),
                "processing_info": result.processing_info
            },
            "attributes": {
                "feature_dim": result.features.len(),
                "video_duration": result.metadata.as_ref().map(|m| m.duration).unwrap_or(0.0),
                "frame_rate": result.metadata.as_ref().map(|m| m.fps).unwrap_or(0.0),
                "feature_type": format!("{:?}", result.feature_type)
            }
        },
        "groups": {
            "video_info": {
                "path": result.metadata.as_ref().map(|m| m.file_path.clone()).unwrap_or_default(),
                "frame_count": result.metadata.as_ref().map(|m| m.frame_count).unwrap_or(0),
                "duration": result.metadata.as_ref().map(|m| m.duration).unwrap_or(0.0),
                "fps": result.metadata.as_ref().map(|m| m.fps).unwrap_or(0.0)
            },
            "feature_info": {
                "type": format!("{:?}", result.feature_type),
                "dimension": result.features.len(),
                "extraction_method": "video_feature_extractor"
            },
            "processing_info": {
                "extraction_time": result.processing_info.as_ref().map(|info| info.extraction_time_ms).unwrap_or(0),
                "processing_details": result.processing_info
            }
        }
    });
    
    // 保存HDF5结构文件
    let hdf5_file = std::fs::File::create(&hdf5_path).map_err(|e| 
        VideoExtractionError::ExportError(format!("创建HDF5文件失败: {}", e))
    )?;
    serde_json::to_writer_pretty(hdf5_file, &hdf5_structure).map_err(|e| 
        VideoExtractionError::ExportError(format!("写入HDF5结构失败: {}", e))
    )?;
    
    // 如果需要包含元数据
    if options.include_metadata {
        let metadata_path = hdf5_path.with_extension("metadata.json");
        let metadata_file = std::fs::File::create(&metadata_path).map_err(|e| 
            VideoExtractionError::ExportError(format!("创建元数据文件失败: {}", e))
        )?;
        serde_json::to_writer_pretty(metadata_file, &result.metadata).map_err(|e| 
            VideoExtractionError::ExportError(format!("写入元数据失败: {}", e))
        )?;
    }
    
    // 创建HDF5读取脚本
    let hdf5_script = format!(
r#"#!/usr/bin/env python3
"""
HDF5视频特征读取脚本
生成时间: {}
"""

import numpy as np
import json
import sys
from pathlib import Path

try:
    import h5py
    H5PY_AVAILABLE = True
except ImportError:
    H5PY_AVAILABLE = False
    print("警告: h5py未安装，将使用JSON格式读取")

class HDF5FeatureReader:
    def __init__(self, hdf5_path: str):
        self.hdf5_path = Path(hdf5_path)
        self.data = {{}}
        self.metadata = {{}}
        
        self._load_data()
    
    def _load_data(self):
        """加载数据"""
        if H5PY_AVAILABLE and self.hdf5_path.exists():
            try:
                with h5py.File(self.hdf5_path, 'r') as f:
                    # 读取特征数据
                    if 'features' in f:
                        self.data['features'] = f['features'][:]
                    
                    # 读取元数据
                    if 'metadata' in f:
                        metadata_group = f['metadata']
                        self.metadata = {{}}
                        for key in metadata_group.attrs:
                            self.metadata[key] = metadata_group.attrs[key]
                    
                    # 读取属性
                    if 'attributes' in f:
                        attr_group = f['attributes']
                        for key in attr_group.attrs:
                            self.data[key] = attr_group.attrs[key]
                            
            except Exception as e:
                print(f"HDF5读取失败: {{e}}")
                self._load_json_fallback()
        else:
            self._load_json_fallback()
    
    def _load_json_fallback(self):
        """JSON格式回退读取"""
        try:
            with open(self.hdf5_path, 'r') as f:
                json_data = json.load(f)
                
            if 'datasets' in json_data:
                datasets = json_data['datasets']
                if 'features' in datasets:
                    self.data['features'] = np.array(datasets['features']['data'])
                if 'metadata' in datasets:
                    self.metadata = datasets['metadata']
                    
        except Exception as e:
            print(f"JSON读取失败: {{e}}")
    
    def get_features(self) -> np.ndarray:
        """获取特征数据"""
        return self.data.get('features', np.array([]))
    
    def get_metadata(self) -> dict:
        """获取元数据"""
        return self.metadata
    
    def get_video_info(self) -> dict:
        """获取视频信息"""
        return {{
            'path': self.metadata.get('video_path', ''),
            'frame_count': self.metadata.get('frame_count', 0),
            'duration': self.metadata.get('duration', 0.0),
            'fps': self.metadata.get('fps', 0.0)
        }}
    
    def get_feature_info(self) -> dict:
        """获取特征信息"""
        return {{
            'type': self.metadata.get('feature_type', ''),
            'dimension': len(self.get_features()),
            'shape': self.get_features().shape
        }}

def main():
    if len(sys.argv) < 2:
        print("用法: python script.py <hdf5_path>")
        sys.exit(1)
    
    hdf5_path = sys.argv[1]
    reader = HDF5FeatureReader(hdf5_path)
    
    # 打印视频信息
    video_info = reader.get_video_info()
    print("视频信息:")
    print(json.dumps(video_info, indent=2, ensure_ascii=False))
    
    # 打印特征信息
    feature_info = reader.get_feature_info()
    print("特征信息:")
    print(json.dumps(feature_info, indent=2, ensure_ascii=False))
    
    # 获取特征数据
    features = reader.get_features()
    if len(features) > 0:
        print(f"特征维度: {{features.shape}}")
        print(f"特征数据: {{features[:10]}}...")  # 只显示前10个值

if __name__ == "__main__":
    main()
"#,
        chrono::Utc::now().to_rfc3339()
    );
    
    let script_path = hdf5_path.with_extension("py");
    let mut script_file = std::fs::File::create(&script_path).map_err(|e| 
        VideoExtractionError::ExportError(format!("创建HDF5读取脚本失败: {}", e))
    )?;
    script_file.write_all(hdf5_script.as_bytes()).map_err(|e| 
        VideoExtractionError::ExportError(format!("写入HDF5读取脚本失败: {}", e))
    )?;
    
    info!("成功导出特征到HDF5格式: {}", hdf5_path.display());
    Ok(())
}

/// 批量导出特征结果
/// 
/// 将多个特征结果批量导出到指定目录，每个结果保存为单独的文件
/// 
/// # 参数
/// - `results`: 要导出的特征结果数组
/// - `output_dir`: 输出目录
/// - `options`: 导出选项
/// 
/// # 返回
/// - 成功: 包含所有导出文件路径的列表
/// - 失败: 导出错误
pub fn export_features_batch<P: AsRef<Path>>(
    results: &[VideoFeatureResult],
    output_dir: P,
    options: &ExportOptions,
) -> Result<Vec<PathBuf>, VideoExtractionError> {
    info!("批量导出 {} 条特征到目录: {}", results.len(), output_dir.as_ref().display());
    
    if results.is_empty() {
        return Err(VideoExtractionError::ExportError("没有特征可导出".to_string()));
    }
    
    // 确保输出目录存在
    fs::create_dir_all(&output_dir).map_err(|e| 
        VideoExtractionError::ExportError(format!("创建输出目录失败: {}", e))
    )?;
    
    let mut output_paths = Vec::with_capacity(results.len());
    let ext = options.format.file_extension();
    
    // 根据批处理大小确定处理方式
    if let Some(batch_size) = options.batch_size {
        if batch_size > 1 && options.format == ExportFormat::HDF5 {
            // 对于HDF5格式批处理特殊处理
            #[cfg(feature = "hdf5")]
            {
                let batch_path = output_dir.as_ref().join(format!("features_batch.{}", ext));
                let refs: Vec<&VideoFeatureResult> = results.iter().collect();
                
                export_to_hdf5(refs, &batch_path, options)?;
                output_paths.push(batch_path);
            }
            
            #[cfg(not(feature = "hdf5"))]
            {
                return Err(VideoExtractionError::ExportError(
                    "HDF5批量导出功能未启用，请在编译时启用'hdf5'特性".to_string()
                ));
            }
        } else {
            // 进行批量处理
            for batch in results.chunks(batch_size) {
                let batch_id = uuid::Uuid::new_v4().to_string();
                let batch_path = output_dir.as_ref().join(format!("batch_{}_{}.{}", 
                                                               batch[0].feature_type, 
                                                               batch_id, 
                                                               ext));
                
                // 为批处理创建集合结果
                let refs: Vec<&VideoFeatureResult> = batch.iter().collect();
                
                match options.format {
                    ExportFormat::CSV => export_to_csv(refs, &batch_path, options)?,
                    ExportFormat::JSON => export_to_json(refs, &batch_path, options)?,
                    ExportFormat::Binary => export_to_binary(refs, &batch_path, options)?,
                    ExportFormat::NumPy => export_to_numpy(refs, &batch_path, options)?,
                    ExportFormat::TensorFlow => export_to_tensorflow(refs, &batch_path, options)?,
                    ExportFormat::ONNX => export_to_onnx(refs, &batch_path, options)?,
                    ExportFormat::HDF5 => {
                        #[cfg(feature = "hdf5")]
                        export_to_hdf5(refs, &batch_path, options)?;
                        
                        #[cfg(not(feature = "hdf5"))]
                        return Err(VideoExtractionError::ExportError(
                            "HDF5格式导出功能未启用，请在编译时启用'hdf5'特性".to_string()
                        ));
                    }
                }
                
                output_paths.push(batch_path);
            }
        }
    } else {
        // 单独处理每个结果
        for result in results {
            // 生成输出文件名，使用视频ID或生成唯一ID
            let filename = if let Some(ref metadata) = result.metadata {
                // 从原始文件名提取基本名称
                let path = Path::new(&metadata.file_path);
                let stem = path.file_stem()
                    .map(|s| s.to_string_lossy().to_string())
                    .unwrap_or_else(|| metadata.id.clone());
                
                format!("{}_{:?}.{}", stem, result.feature_type, ext)
            } else {
                // 没有元数据，使用特征类型和UUID
                format!("{}_{:?}.{}", uuid::Uuid::new_v4(), result.feature_type, ext)
            };
            
            let output_path = output_dir.as_ref().join(filename);
            
            // 导出单个特征
            match options.format {
                ExportFormat::CSV => export_to_csv(vec![result], &output_path, options)?,
                ExportFormat::JSON => export_to_json(vec![result], &output_path, options)?,
                ExportFormat::Binary => export_to_binary(vec![result], &output_path, options)?,
                ExportFormat::NumPy => export_to_numpy(vec![result], &output_path, options)?,
                ExportFormat::TensorFlow => export_to_tensorflow(vec![result], &output_path, options)?,
                ExportFormat::ONNX => export_to_onnx(vec![result], &output_path, options)?,
                ExportFormat::HDF5 => {
                    #[cfg(feature = "hdf5")]
                    export_to_hdf5(vec![result], &output_path, options)?;
                    
                    #[cfg(not(feature = "hdf5"))]
                    return Err(VideoExtractionError::ExportError(
                        "HDF5格式导出功能未启用，请在编译时启用'hdf5'特性".to_string()
                    ));
                }
            }
            
            output_paths.push(output_path);
        }
    }
    
    info!("成功导出 {} 个特征文件到 {}", output_paths.len(), output_dir.as_ref().display());
    
    Ok(output_paths)
}

/// 获取所有支持的导出格式
pub fn get_available_export_formats() -> Vec<ExportFormat> {
    let mut formats = vec![
        ExportFormat::CSV,
        ExportFormat::JSON,
        ExportFormat::Binary,
    ];
    
    #[cfg(feature = "numpy")]
    formats.push(ExportFormat::NumPy);
    
    #[cfg(feature = "tensorflow")]
    formats.push(ExportFormat::TensorFlow);
    
    #[cfg(feature = "onnx")]
    formats.push(ExportFormat::ONNX);
    
    #[cfg(feature = "hdf5")]
    formats.push(ExportFormat::HDF5);
    
    formats
}

/// 检查指定格式是否支持
pub fn is_format_supported(format: ExportFormat) -> bool {
    match format {
        ExportFormat::CSV | ExportFormat::JSON | ExportFormat::Binary => true,
        ExportFormat::NumPy => cfg!(feature = "numpy"),
        ExportFormat::TensorFlow => cfg!(feature = "tensorflow"),
        ExportFormat::ONNX => cfg!(feature = "onnx"),
        ExportFormat::HDF5 => cfg!(feature = "hdf5"),
    }
}

/// 获取格式的特性需求
pub fn get_format_feature_requirement(format: ExportFormat) -> &'static str {
    match format {
        ExportFormat::CSV | ExportFormat::JSON | ExportFormat::Binary => "",
        ExportFormat::NumPy => "numpy",
        ExportFormat::TensorFlow => "tensorflow",
        ExportFormat::ONNX => "onnx",
        ExportFormat::HDF5 => "hdf5",
    }
}

/// 创建导出选项
pub fn create_export_options(
    format: ExportFormat,
    include_metadata: bool,
    include_processing_info: bool,
    compress: bool
) -> ExportOptions {
    ExportOptions {
        format,
        include_metadata,
        include_processing_info,
        compress,
        batch_size: None,
        custom_options: HashMap::new(),
    }
}

/// 批量导出到tensorboard格式
pub fn export_to_tensorboard<P: AsRef<Path>>(
    results: &[VideoFeatureResult],
    log_dir: P,
    label_map: Option<HashMap<String, String>>
) -> Result<PathBuf, VideoExtractionError> {
    #[cfg(feature = "tensorboard")]
    {
        use tensorboard_rs::{SummaryWriter, FileWriter};
        
        info!("导出特征到TensorBoard: {}", log_dir.as_ref().display());
        
        // 确保目录存在
        fs::create_dir_all(&log_dir).map_err(|e| 
            VideoExtractionError::ExportError(format!("创建TensorBoard日志目录失败: {}", e))
        )?;
        
        // 创建SummaryWriter
        let log_path = log_dir.as_ref().to_path_buf();
        let writer = FileWriter::create(log_path.clone())
            .map_err(|e| VideoExtractionError::ExportError(
                format!("创建TensorBoard日志写入器失败: {}", e)
            ))?;
        
        let mut summary_writer = SummaryWriter::new(writer);
        
        // 收集所有特征类型
        let mut feature_types = std::collections::HashSet::new();
        for result in results {
            feature_types.insert(result.feature_type);
        }
        
        // 为每种特征类型创建投影
        for &feature_type in &feature_types {
            let metadata_path = log_path.join(format!("metadata_{}.tsv", feature_type.to_string()));
            let mut metadata_file = File::create(&metadata_path)
                .map_err(|e| VideoExtractionError::FileError(
                    format!("创建元数据文件失败: {}", e)
                ))?;
            
            // 写入元数据标题
            writeln!(metadata_file, "video_id\tlabel")
                .map_err(|e| VideoExtractionError::ExportError(
                    format!("写入元数据标题失败: {}", e)
                ))?;
            
            // 收集此特征类型的所有特征
            let type_results: Vec<_> = results.iter()
                .filter(|r| r.feature_type == feature_type)
                .collect();
            
            let mut tensor_data = Vec::new();
            let mut labels = Vec::new();
            
            for result in &type_results {
                tensor_data.push(result.features.clone());
                
                // 获取标签(如果有)
                let video_id = result.metadata.as_ref().map(|m| m.id.clone()).unwrap_or_default();
                let label = if let Some(label_map) = &label_map {
                    label_map.get(&video_id)
                        .cloned()
                        .unwrap_or_else(|| "unknown".to_string())
                } else {
                    video_id.clone()
                };
                
                labels.push(label.clone());
                
                // 写入元数据行
                writeln!(metadata_file, "{}\t{}", video_id, label)
                    .map_err(|e| VideoExtractionError::ExportError(
                        format!("写入元数据行失败: {}", e)
                    ))?;
            }
            
            // 创建嵌入
            summary_writer.add_embedding(
                &tensor_data,
                Some(&labels),
                Some(&metadata_path.to_string_lossy()),
                Some(&format!("features_{}", feature_type.to_string())),
                0
            ).map_err(|e| VideoExtractionError::ExportError(
                format!("添加嵌入到TensorBoard失败: {}", e)
            ))?;
        }
        
        info!("成功导出 {} 条特征到TensorBoard: {}", 
                   results.len(), log_path.display());
        
        Ok(log_path)
    }
    
    #[cfg(not(feature = "tensorboard"))]
    {
        warn!("TensorBoard导出功能未启用，请在编译时启用'tensorboard'特性");
        Err(VideoExtractionError::ExportError(
            "TensorBoard导出功能未启用，请在编译时启用'tensorboard'特性".to_string()
        ))
    }
}

/// 导出特征到HDF5格式
fn export_to_hdf5<P: AsRef<Path>>(results: Vec<&VideoFeatureResult>, path: P, options: &ExportOptions) -> Result<(), VideoExtractionError> {
    #[cfg(feature = "hdf5")]
    {
        use hdf5::{File, Group};
        use ndarray::{Array, Array2, Axis};
        
        info!("导出特征到HDF5格式: {}", path.as_ref().display());
        
        // 创建HDF5文件
        let file = File::create(path.as_ref()).map_err(|e| 
            VideoExtractionError::ExportError(format!("创建HDF5文件失败: {}", e))
        )?;
        
        // 创建包含基本信息的根组
        let root = file.create_group("features").map_err(|e| 
            VideoExtractionError::ExportError(format!("创建HDF5组失败: {}", e))
        )?;
        
        // 添加导出信息
        let info_group = root.create_group("info").map_err(|e| 
            VideoExtractionError::ExportError(format!("创建信息组失败: {}", e))
        )?;
        
        // 记录导出时间
        let now = chrono::Utc::now();
        let timestamp = now.to_rfc3339();
        info_group.new_dataset::<String>().create("export_time", ()).map_err(|e| 
            VideoExtractionError::ExportError(format!("写入导出时间失败: {}", e))
        )?.write_scalar(&timestamp).map_err(|e|
            VideoExtractionError::ExportError(format!("写入时间戳失败: {}", e))
        )?;
        
        // 记录特征数量
        info_group.new_dataset::<usize>().create("feature_count", ()).map_err(|e| 
            VideoExtractionError::ExportError(format!("创建特征计数数据集失败: {}", e))
        )?.write_scalar(&results.len()).map_err(|e|
            VideoExtractionError::ExportError(format!("写入特征计数失败: {}", e))
        )?;
        
        // 创建视频组
        let videos_group = root.create_group("videos").map_err(|e| 
            VideoExtractionError::ExportError(format!("创建视频组失败: {}", e))
        )?;
        
        // 收集所有文件名和特征尺寸
        let file_names: Vec<String> = results.iter()
            .map(|r| r.metadata.as_ref().map(|m| m.file_path.clone()).unwrap_or_else(|| "unknown".to_string()))
            .collect();
        
        // 获取特征维度
        let feature_dim = if !results.is_empty() {
            results[0].features.len()
        } else {
            0
        };
        
        // 创建特征数据集
        if !results.is_empty() {
            // 收集所有特征到一个大数组
            let mut features_array = Array2::<f32>::zeros((results.len(), feature_dim));
            
            for (i, result) in results.iter().enumerate() {
                if result.features.len() == feature_dim {
                    let row_slice = features_array.slice_mut(ndarray::s![i, ..]);
                    for (j, &value) in result.features.iter().enumerate() {
                        row_slice[j] = value;
                    }
                } else {
                    warn!("特征维度不一致，跳过: {:?}", result.metadata.as_ref().map(|m| &m.file_path));
                }
            }
            
            // 写入特征数据
            let features_dataset = root.new_dataset::<f32>()
                .shape((results.len(), feature_dim))
                .create("feature_data")
                .map_err(|e| VideoExtractionError::ExportError(
                    format!("创建特征数据集失败: {}", e))
                )?;
            
            features_dataset.write(&features_array).map_err(|e| 
                VideoExtractionError::ExportError(format!("写入特征数据失败: {}", e))
            )?;
            
            // 写入文件名
            let filenames_dataset = root.new_dataset::<String>()
                .shape((file_names.len(),))
                .create("filenames")
                .map_err(|e| VideoExtractionError::ExportError(
                    format!("创建文件名数据集失败: {}", e))
                )?;
            
            filenames_dataset.write(&Array::from_vec(file_names)).map_err(|e| 
                VideoExtractionError::ExportError(format!("写入文件名失败: {}", e))
            )?;
            
            // 如果需要，写入元数据
            if options.include_metadata {
                let metadata_group = root.create_group("metadata").map_err(|e| 
                    VideoExtractionError::ExportError(format!("创建元数据组失败: {}", e))
                )?;
                
                // 提取所有可能的元数据键
                let mut all_keys = Vec::new();
                for result in results.iter() {
                    if let Some(ref metadata) = result.metadata {
                        // 处理字段
                        if !metadata.file_path.is_empty() { all_keys.push("file_path".to_string()); }
                        all_keys.push("duration".to_string());
                        all_keys.push("width".to_string());
                        all_keys.push("height".to_string());
                        all_keys.push("fps".to_string());
                        if !metadata.codec.is_empty() { all_keys.push("codec".to_string()); }
                        if let Some(bitrate) = metadata.custom_metadata.as_ref().and_then(|m| m.get("bit_rate")) {
                            if !bitrate.is_empty() { all_keys.push("bitrate".to_string()); }
                        }
                        if metadata.audio_codec.is_some() { all_keys.push("audio_codec".to_string()); }
                        if metadata.audio_channels.is_some() { all_keys.push("audio_channels".to_string()); }
                        if metadata.audio_sample_rate.is_some() { all_keys.push("audio_sample_rate".to_string()); }
                    }
                }
                
                // 去重
                all_keys.sort();
                all_keys.dedup();
                
                // 为每个键创建数据集
                for key in all_keys {
                    match key.as_str() {
                        "filename" => {
                            let values: Vec<String> = results.iter()
                                .map(|r| r.metadata.as_ref().map(|m| m.file_path.clone()).unwrap_or_else(|| "".to_string()))
                                .collect();
                            let dataset = metadata_group.new_dataset::<String>()
                                .shape((values.len(),))
                                .create(&key)
                                .map_err(|e| VideoExtractionError::ExportError(
                                    format!("创建元数据数据集失败: {}", e))
                                )?;
                            dataset.write(&Array::from_vec(values)).map_err(|e| 
                                VideoExtractionError::ExportError(format!("写入元数据失败: {}", e))
                            )?;
                        },
                        "duration" => {
                            let values: Vec<f64> = results.iter()
                                .map(|r| r.metadata.as_ref().map(|m| m.duration).unwrap_or(0.0))
                                .collect();
                            let dataset = metadata_group.new_dataset::<f64>()
                                .shape((values.len(),))
                                .create(&key)
                                .map_err(|e| VideoExtractionError::ExportError(
                                    format!("创建元数据数据集失败: {}", e))
                                )?;
                            dataset.write(&Array::from_vec(values)).map_err(|e| 
                                VideoExtractionError::ExportError(format!("写入元数据失败: {}", e))
                            )?;
                        },
                        "width" | "height" => {
                            let values: Vec<u32> = if key == "width" {
                                results.iter().map(|r| r.metadata.as_ref().map(|m| m.width).unwrap_or(0)).collect()
                            } else {
                                results.iter().map(|r| r.metadata.as_ref().map(|m| m.height).unwrap_or(0)).collect()
                            };
                            let dataset = metadata_group.new_dataset::<u32>()
                                .shape((values.len(),))
                                .create(&key)
                                .map_err(|e| VideoExtractionError::ExportError(
                                    format!("创建元数据数据集失败: {}", e))
                                )?;
                            dataset.write(&Array::from_vec(values)).map_err(|e| 
                                VideoExtractionError::ExportError(format!("写入元数据失败: {}", e))
                            )?;
                        },
                        "fps" => {
                            let values: Vec<f32> = results.iter()
                                .map(|r| r.metadata.as_ref().map(|m| m.fps).unwrap_or(0.0))
                                .collect();
                            let dataset = metadata_group.new_dataset::<f32>()
                                .shape((values.len(),))
                                .create(&key)
                                .map_err(|e| VideoExtractionError::ExportError(
                                    format!("创建元数据数据集失败: {}", e))
                                )?;
                            dataset.write(&Array::from_vec(values)).map_err(|e| 
                                VideoExtractionError::ExportError(format!("写入元数据失败: {}", e))
                            )?;
                        },
                        // 处理其他元数据字段...
                        _ => {
                            warn!("跳过元数据字段: {}", key);
                        }
                    }
                }
            }
            
            // 如果需要，写入处理信息
            if options.include_processing_info {
                let processing_group = root.create_group("processing").map_err(|e| 
                    VideoExtractionError::ExportError(format!("创建处理信息组失败: {}", e))
                )?;
                
                // 收集处理时间
                let processing_times: Vec<f64> = results.iter()
                    .map(|r| r.processing_info.as_ref().map(|info| info.extraction_time_ms).unwrap_or(0))
                    .collect();
                
                let times_dataset = processing_group.new_dataset::<f64>()
                    .shape((processing_times.len(),))
                    .create("processing_times")
                    .map_err(|e| VideoExtractionError::ExportError(
                        format!("创建处理时间数据集失败: {}", e))
                    )?;
                
                times_dataset.write(&Array::from_vec(processing_times)).map_err(|e| 
                    VideoExtractionError::ExportError(format!("写入处理时间失败: {}", e))
                )?;
                
                // 写入特征类型信息
                if !results.is_empty() {
                    let feature_type = results[0].feature_type.to_string();
                    processing_group.new_dataset::<String>().create("feature_type", ()).map_err(|e| 
                        VideoExtractionError::ExportError(format!("创建特征类型数据集失败: {}", e))
                    )?.write_scalar(&feature_type).map_err(|e|
                        VideoExtractionError::ExportError(format!("写入特征类型失败: {}", e))
                    )?;
                }
            }
        }
        
        info!("成功导出 {} 条记录到HDF5文件", results.len());
        Ok(())
    }
    
    #[cfg(not(feature = "hdf5"))]
    {
        warn!("HDF5导出功能未启用，请在编译时启用'hdf5'特性");
        Err(VideoExtractionError::ExportError(
            "HDF5导出功能未启用，请在编译时启用'hdf5'特性".to_string()
        ))
    }
}

/// 导出特征到TensorFlow格式
fn export_to_tensorflow<P: AsRef<Path>>(results: Vec<&VideoFeatureResult>, path: P, options: &ExportOptions) -> Result<(), VideoExtractionError> {
    #[cfg(feature = "tensorflow")]
    {
        use tensorflow::{Graph, SavedModelBuilder, SessionOptions, Status, Tensor};
        
        info!("导出特征到TensorFlow格式: {}", path.as_ref().display());
        
        if results.is_empty() {
            return Err(VideoExtractionError::ExportError("没有特征可导出".to_string()));
        }
        
        // 确保目录存在
        let model_dir = if path.as_ref().extension().is_some() {
            // 如果路径有扩展名，则使用父目录
            path.as_ref().parent()
                .map(|p| p.to_path_buf())
                .unwrap_or_else(|| PathBuf::from("."))
        } else {
            // 否则使用路径作为目录
            path.as_ref().to_path_buf()
        };
        
        fs::create_dir_all(&model_dir).map_err(|e| 
            VideoExtractionError::ExportError(format!("创建TensorFlow模型目录失败: {}", e))
        )?;
        
        // 获取特征维度
        let feature_dim = results[0].features.len();
        let num_features = results.len();
        
        // 创建TensorFlow图
        let mut graph = Graph::new();
        
        // 创建输入张量
        let mut feature_data = Vec::with_capacity(num_features * feature_dim);
        for result in &results {
            feature_data.extend_from_slice(&result.features);
        }
        
        // 创建特征张量
        let feature_tensor = Tensor::new(&[num_features as u64, feature_dim as u64])
            .with_values(&feature_data)
            .map_err(|e| VideoExtractionError::ExportError(
                format!("创建TensorFlow特征张量失败: {}", e)
            ))?;
        
        // 创建SavedModel
        let mut builder = SavedModelBuilder::new();
        builder
            .add_tag("serve")
            .add_tag("features");
        
        // 添加特征变量
        let variable_name = "features";
        let op = graph.variable(variable_name, feature_tensor.shape(), tensorflow::DataType::Float)
                     .map_err(|e| VideoExtractionError::ExportError(
                         format!("创建TensorFlow变量失败: {}", e)
                     ))?;
        
        // 创建会话并保存模型
        let session_options = SessionOptions::new();
        let session = tensorflow::Session::new(&session_options, &graph)
                                 .map_err(|e| VideoExtractionError::ExportError(
                                     format!("创建TensorFlow会话失败: {}", e)
                                 ))?;
        
        // 初始化变量
        let init_op = graph.operation_by_name_required("init")
                          .map_err(|e| VideoExtractionError::ExportError(
                              format!("获取初始化操作失败: {}", e)
                          ))?;
        
        session.run(&[], &[], &[init_op])
              .map_err(|e| VideoExtractionError::ExportError(
                  format!("初始化TensorFlow变量失败: {}", e)
              ))?;
        
        // 保存模型
        builder.save(&session, &graph, &model_dir)
              .map_err(|e| VideoExtractionError::ExportError(
                  format!("保存TensorFlow模型失败: {}", e)
              ))?;
        
        // 如果需要保存元数据
        if options.include_metadata {
            let metadata_path = model_dir.join("metadata.json");
            
            // 收集元数据
            let metadata = results.iter().map(|r| {
                json!({
                    "file_path": r.metadata.as_ref().map(|m| m.file_path.clone()).unwrap_or_default(),
                    "duration": r.metadata.as_ref().map(|m| m.duration).unwrap_or(0.0),
                    "width": r.metadata.as_ref().map(|m| m.width).unwrap_or(0),
                    "height": r.metadata.as_ref().map(|m| m.height).unwrap_or(0),
                    "fps": r.metadata.as_ref().map(|m| m.fps).unwrap_or(0.0),
                    "feature_type": format!("{:?}", r.feature_type),
                    "processing_time": r.processing_info.as_ref().map(|info| info.extraction_time_ms).unwrap_or(0)
                })
            }).collect::<Vec<_>>();
            
            // 写入元数据JSON文件
            let metadata_json = serde_json::to_string_pretty(&metadata).map_err(|e| 
                VideoExtractionError::ExportError(format!("序列化元数据失败: {}", e))
            )?;
            
            std::fs::write(&metadata_path, metadata_json).map_err(|e| 
                VideoExtractionError::ExportError(format!("写入元数据文件失败: {}", e))
            )?;
            
            info!("元数据已保存到: {}", metadata_path.display());
        }
        
        info!("成功导出 {} 条记录到TensorFlow格式", results.len());
        Ok(())
    }
    
    #[cfg(not(feature = "tensorflow"))]
    {
        warn!("TensorFlow导出功能未启用，请在编译时启用'tensorflow'特性");
        Err(VideoExtractionError::ExportError(
            "TensorFlow导出功能未启用，请在编译时启用'tensorflow'特性".to_string()
        ))
    }
} 