//! 视频元数据提取模块
//! 
//! 本模块提供从视频文件中提取元数据的功能

use crate::data::multimodal::video_extractor::types::*;
use crate::data::multimodal::video_extractor::error::VideoExtractionError;
use std::path::Path;
use std::time::Instant;
use std::collections::HashMap;
use std::process::Command;
use std::str::FromStr;
use log::{debug, warn};
use super::types::{KeyframeInfo, SceneChange, Thumbnail};

/// 从视频文件中提取元数据
pub fn extract_video_metadata(
    video_path: &str
) -> std::result::Result<VideoMetadata, VideoExtractionError> {
    let path = Path::new(video_path);
    
    if !path.exists() {
        return Err(VideoExtractionError::FileError(
            format!("视频文件不存在: {}", video_path)
        ));
    }
    
    debug!("开始提取视频元数据: {}", video_path);
    let start_time = Instant::now();

    // 使用FFmpeg提取视频元数据
    let output = Command::new("ffprobe")
        .args(&[
            "-v", "quiet",
            "-print_format", "json",
            "-show_format",
            "-show_streams",
            video_path
        ])
        .output()
        .map_err(|e| VideoExtractionError::ProcessingError(
            format!("执行FFprobe失败: {}", e)
        ))?;

    if !output.status.success() {
        let error_msg = String::from_utf8_lossy(&output.stderr);
        return Err(VideoExtractionError::ProcessingError(
            format!("FFprobe返回错误: {}", error_msg)
        ));
    }

    let json_output = String::from_utf8_lossy(&output.stdout);
    
    // 解析JSON输出
    let parsed: serde_json::Value = serde_json::from_str(&json_output)
        .map_err(|e| VideoExtractionError::ProcessingError(
            format!("解析FFprobe输出失败: {}", e)
        ))?;

    // 提取视频流信息
    let mut width = 0;
    let mut height = 0;
    let mut frame_rate: Option<f64> = None;
    let mut frame_count: usize = 0;
    let mut video_codec = String::new();
    let mut audio_codec = None;
    let mut has_audio = false;
    let mut duration = 0.0;
    let mut bit_rate: Option<i32> = None;
    let mut additional_info: Option<HashMap<String, String>> = None;
    let mut format = String::new();

    // 获取格式信息
    if let Some(format_obj) = parsed.get("format") {
        // 获取格式名称
        if let Some(format_name) = format_obj.get("format_name") {
            format = format_name.as_str().unwrap_or("unknown").to_string();
        }

        // 获取持续时间
        if let Some(dur_str) = format_obj.get("duration") {
            if let Some(dur) = dur_str.as_str() {
                duration = f64::from_str(dur).unwrap_or(0.0);
            }
        }

        // 获取比特率
        if let Some(br_str) = format_obj.get("bit_rate") {
            if let Some(br) = br_str.as_str() {
                bit_rate = i32::from_str(br).ok();
            }
        }

        // 收集额外信息
        if let Some(tags) = format_obj.get("tags") {
            if let Some(obj) = tags.as_object() {
                let mut info = HashMap::new();
                for (key, value) in obj {
                    if let Some(val_str) = value.as_str() {
                        info.insert(key.clone(), val_str.to_string());
                    }
                }
                if !info.is_empty() {
                    additional_info = Some(info);
                }
            }
        }
    }

    // 处理流信息
    if let Some(streams) = parsed.get("streams") {
        if let Some(streams_array) = streams.as_array() {
            for stream in streams_array {
                if let Some(codec_type) = stream.get("codec_type") {
                    if codec_type.as_str() == Some("video") {
                        // 获取视频尺寸
                        if let Some(w) = stream.get("width") {
                            width = w.as_i64().unwrap_or(0) as usize;
                        }
                        if let Some(h) = stream.get("height") {
                            height = h.as_i64().unwrap_or(0) as usize;
                        }

                        // 获取视频编解码器
                        if let Some(codec) = stream.get("codec_name") {
                            video_codec = codec.as_str().unwrap_or("unknown").to_string();
                        }

                        // 获取帧率
                        if let Some(fps_str) = stream.get("r_frame_rate") {
                            if let Some(fps) = fps_str.as_str() {
                                let parts: Vec<&str> = fps.split('/').collect();
                                if parts.len() == 2 {
                                    let num = f64::from_str(parts[0]).unwrap_or(0.0);
                                    let den = f64::from_str(parts[1]).unwrap_or(1.0);
                                    if den > 0.0 {
                                        frame_rate = Some(num / den);
                                    }
                                }
                            }
                        }

                        // 获取帧数
                        if let Some(frames) = stream.get("nb_frames") {
                            if let Some(fc) = frames.as_str() {
                                frame_count = usize::from_str(fc).unwrap_or(0);
                            }
                        } else {
                            // 如果没有nb_frames字段，则从帧率和持续时间估算
                            if let Some(fps) = frame_rate {
                                frame_count = (fps * duration) as usize;
                            }
                        }
                    } else if codec_type.as_str() == Some("audio") {
                        has_audio = true;
                        if let Some(codec) = stream.get("codec_name") {
                            audio_codec = Some(codec.as_str().unwrap_or("unknown").to_string());
                        }
                    }
                }
            }
        }
    }

    // 如果无法直接获取格式，则从文件扩展名推断
    if format.is_empty() {
        format = match path.extension().and_then(|ext| ext.to_str()) {
            Some("mp4") => "mp4".to_string(),
            Some("avi") => "avi".to_string(),
            Some("mkv") => "matroska".to_string(),
            Some(ext) => ext.to_string(),
            None => "unknown".to_string(),
        };
    }

    let id = generate_video_id(video_path);

    // 构建自定义元数据
    let mut custom_metadata = HashMap::new();
    if !format.is_empty() {
        custom_metadata.insert("format".to_string(), format);
    }
    if let Some(bit_rate) = bit_rate {
        custom_metadata.insert("bit_rate".to_string(), bit_rate.to_string());
    }
    if let Some(ref info) = additional_info {
        for (k, v) in info {
            custom_metadata.insert(k.clone(), v.clone());
        }
    }
    
    let metadata = VideoMetadata {
        id,
        file_path: video_path.to_string(),
        file_size: std::fs::metadata(video_path).ok().map(|m| m.len()).unwrap_or(0),
        width: width.try_into().unwrap_or(0),
        height: height.try_into().unwrap_or(0),
        duration,
        fps: frame_rate.unwrap_or(0.0) as f32,
        frame_count: frame_count.try_into().unwrap_or(0),
        codec: if video_codec.is_empty() { "unknown".to_string() } else { video_codec },
        audio_codec,
        audio_sample_rate: None,
        audio_channels: None,
        created_at: get_file_creation_time(video_path).unwrap_or(0),
        custom_metadata: if custom_metadata.is_empty() { None } else { Some(custom_metadata) },
    };
    
    let elapsed = start_time.elapsed();
    debug!("视频元数据提取完成，耗时: {:?}", elapsed);
    
    Ok(metadata)
}

/// 获取文件创建时间
fn get_file_creation_time(file_path: &str) -> Option<u64> {
    match std::fs::metadata(file_path) {
        Ok(metadata) => {
            #[cfg(unix)]
            {
                #[cfg(unix)]
use std::os::unix::fs::MetadataExt;
                Some(metadata.ctime() as u64)
            }
            #[cfg(windows)]
            {
                use std::os::windows::fs::MetadataExt;
                {
                    let time = metadata.creation_time();
                    // Windows时间是从1601年1月1日开始的100纳秒间隔
                    // 转换为Unix时间戳（从1970年1月1日开始的秒数）
                    const WINDOWS_TICK: u64 = 10_000_000;
                    const SEC_TO_UNIX_EPOCH: u64 = 11_644_473_600;
                    Some(time / WINDOWS_TICK - SEC_TO_UNIX_EPOCH)
                }
            }
            #[cfg(not(any(unix, windows)))]
            {
                None
            }
        }
        Err(_) => None
    }
}

/// 生成视频ID
fn generate_video_id(video_path: &str) -> String {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};
    
    let path = Path::new(video_path);
    let file_name = path.file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("unknown");
    
    // 使用文件名和完整路径生成哈希
    let mut hasher = DefaultHasher::new();
    video_path.hash(&mut hasher);
    let hash = hasher.finish();
    
    format!("{}_{:x}", file_name, hash)
}

/// 从视频文件中提取关键帧信息
pub fn extract_keyframe_info(
    video_path: &str
) -> std::result::Result<Vec<KeyframeInfo>, VideoExtractionError> {
    let path = Path::new(video_path);
    
    if !path.exists() {
        return Err(VideoExtractionError::FileError(
            format!("视频文件不存在: {}", video_path)
        ));
    }
    
    debug!("开始提取视频关键帧信息: {}", video_path);
    let start_time = Instant::now();
    
    // 使用FFmpeg提取关键帧信息
    let output = Command::new("ffprobe")
        .args(&[
            "-v", "quiet",
            "-select_streams", "v",
            "-show_entries", "packet=pts_time,flags",
            "-of", "csv=print_section=0",
            video_path
        ])
        .output()
        .map_err(|e| VideoExtractionError::ProcessingError(
            format!("执行FFprobe失败: {}", e)
        ))?;

    if !output.status.success() {
        let error_msg = String::from_utf8_lossy(&output.stderr);
        return Err(VideoExtractionError::ProcessingError(
            format!("FFprobe返回错误: {}", error_msg)
        ));
    }

    let csv_output = String::from_utf8_lossy(&output.stdout);
    let mut keyframes = Vec::new();
    let mut frame_index = 0;
    
    // 解析CSV输出，找出关键帧
    for line in csv_output.lines() {
        let parts: Vec<&str> = line.split(',').collect();
        if parts.len() >= 2 {
            let is_keyframe = parts[1].contains("K");
            if is_keyframe {
                if let Ok(timestamp) = parts[0].parse::<f64>() {
                    keyframes.push(KeyframeInfo {
                        timestamp,
                        position: 0, // FFprobe输出中没有提供这个信息
                        frame_type: "I".to_string(),
                        frame_size: 0, // 同样没有这个信息
                        frame_index,
                    });
                }
            }
            frame_index += 1;
        }
    }
    
    let elapsed = start_time.elapsed();
    debug!("视频关键帧信息提取完成，数量: {}，耗时: {:?}", keyframes.len(), elapsed);
    
    Ok(keyframes)
}

/// 从视频文件中提取场景变化信息
pub fn extract_scene_changes(
    video_path: &str,
    threshold: f32
) -> std::result::Result<Vec<SceneChange>, VideoExtractionError> {
    let path = Path::new(video_path);
    
    if !path.exists() {
        return Err(VideoExtractionError::FileError(
            format!("视频文件不存在: {}", video_path)
        ));
    }
    
    debug!("开始提取视频场景变化信息: {}, 阈值: {}", video_path, threshold);
    let start_time = Instant::now();
    
    // 使用FFmpeg的scene检测滤镜来识别场景变化
    let output = Command::new("ffmpeg")
        .args(&[
            "-i", video_path,
            "-vf", &format!("select=gt(scene\\,{}),metadata=print:file=-", threshold),
            "-f", "null",
            "-"
        ])
        .output()
        .map_err(|e| VideoExtractionError::ProcessingError(
            format!("执行FFmpeg失败: {}", e)
        ))?;

    let stderr_output = String::from_utf8_lossy(&output.stderr);
    let mut scene_changes = Vec::new();
    let mut frame_index = 0;
    
    // 解析FFmpeg输出，寻找场景变化信息
    for line in stderr_output.lines() {
        if line.contains("lavfi.scene_score=") {
            let parts: Vec<&str> = line.split('=').collect();
            if parts.len() >= 2 {
                if let Ok(confidence) = parts[1].trim().parse::<f32>() {
                    // 从输出中提取时间戳
                    let time_parts: Vec<&str> = line.split("pts_time:").collect();
                    let mut timestamp = 0.0;
                    if time_parts.len() >= 2 {
                        let time_str = time_parts[1].split_whitespace().next().unwrap_or("0");
                        timestamp = time_str.parse::<f64>().unwrap_or(0.0);
                    }
                    
                    scene_changes.push(SceneChange {
                        start_timestamp: timestamp,
                        end_timestamp: timestamp + 0.1, // 假设场景变化发生在0.1秒内
                        confidence,
                        frame_index,
                        description: Some(format!("场景变化，置信度: {:.2}", confidence)),
                    });
                    
                    frame_index += 1;
                }
            }
        }
    }
    
    let elapsed = start_time.elapsed();
    debug!("视频场景变化信息提取完成，数量: {}，耗时: {:?}", scene_changes.len(), elapsed);
    
    Ok(scene_changes)
}

/// 为视频生成缩略图
pub fn generate_thumbnails(
    video_path: &str,
    count: usize,
    width: Option<usize>,
    height: Option<usize>
) -> std::result::Result<Vec<Thumbnail>, VideoExtractionError> {
    let path = Path::new(video_path);
    
    if !path.exists() {
        return Err(VideoExtractionError::FileError(
            format!("视频文件不存在: {}", video_path)
        ));
    }
    
    if count == 0 {
        return Err(VideoExtractionError::InputError(
            "缩略图数量不能为零".to_string()
        ));
    }
    
    debug!("开始为视频生成{}个缩略图: {}", count, video_path);
    let start_time = Instant::now();
    
    // 首先获取视频元数据以确定持续时间
    let metadata = extract_video_metadata(video_path)?;
    let duration = metadata.duration;
    let interval = duration / count as f64;
    
    let target_width = width.unwrap_or(320);
    let target_height = height.unwrap_or(240);
    
    let mut thumbnails = Vec::with_capacity(count);
    let temp_dir = std::env::temp_dir();
    
    for i in 0..count {
        let timestamp = i as f64 * interval;
        let output_file = temp_dir.join(format!("thumb_{}_{}.jpg", 
            path.file_name().unwrap_or_default().to_string_lossy(), i));
        let output_path = output_file.to_string_lossy();
        
        // 使用FFmpeg在指定时间点提取帧并调整大小
        let status = Command::new("ffmpeg")
            .args(&[
                "-ss", &timestamp.to_string(),
                "-i", video_path,
                "-vframes", "1",
                "-vf", &format!("scale={}:{}", target_width, target_height),
                "-q:v", "2", // 高质量
                "-y", // 覆盖已有文件
                &output_path
            ])
            .status()
            .map_err(|e| VideoExtractionError::ProcessingError(
                format!("执行FFmpeg失败: {}", e)
            ))?;
        
        if !status.success() {
            warn!("生成缩略图失败，时间戳: {}", timestamp);
            continue;
        }
        
        // 读取生成的图像文件
        let img_data = std::fs::read(&output_file)
            .map_err(|e| VideoExtractionError::FileError(
                format!("读取缩略图文件失败: {}", e)
            ))?;
        
        // 删除临时文件
        let _ = std::fs::remove_file(&output_file);
        
        thumbnails.push(Thumbnail {
            timestamp,
            file_path: output_file.clone(),
            width: target_width,
            height: target_height,
            quality: 85,
            data: Some(img_data),
        });
    }
    
    let elapsed = start_time.elapsed();
    debug!("视频缩略图生成完成，数量: {}，耗时: {:?}", thumbnails.len(), elapsed);
    
    Ok(thumbnails)
} 