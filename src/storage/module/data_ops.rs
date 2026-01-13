// 数据操作模块
// 实现DataOperations trait和相关数据操作方法

use std::path::Path;
use serde_json::Value;
use chrono::Utc;
use uuid::Uuid;

use crate::{Error, Result};
use crate::data::DataFormat;
use super::core::Storage;
use super::data::{DataOperations, DataInfo, DATA_RAW_PREFIX, DATA_PROCESSED_PREFIX};

impl DataOperations for Storage {
    fn store_raw_data(&self, data_id: &str, data: Vec<String>) -> Result<()> {
        let key = format!("{}{}", DATA_RAW_PREFIX, data_id);
        let value = serde_json::to_string(&data)
            .map_err(|e| Error::Serialization(e.to_string()))?;
        self.put(key.as_bytes(), value.as_bytes())?;
        Ok(())
    }
    
    fn get_raw_data(&self, data_id: &str) -> Result<Option<Vec<String>>> {
        let key = format!("{}{}", DATA_RAW_PREFIX, data_id);
        if let Some(value) = self.get(key.as_bytes())? {
            let data: Vec<String> = serde_json::from_slice(&value)
                .map_err(|e| Error::Deserialization(e.to_string()))?;
            Ok(Some(data))
        } else {
            Ok(None)
        }
    }
    
    fn store_processed_data(&self, data_id: &str, data: Vec<String>) -> Result<()> {
        let key = format!("{}{}", DATA_PROCESSED_PREFIX, data_id);
        let value = serde_json::to_string(&data)
            .map_err(|e| Error::Serialization(e.to_string()))?;
        self.put(key.as_bytes(), value.as_bytes())?;
        Ok(())
    }
    
    fn get_processed_data(&self, data_id: &str) -> Result<Option<Vec<String>>> {
        let key = format!("{}{}", DATA_PROCESSED_PREFIX, data_id);
        if let Some(value) = self.get(key.as_bytes())? {
            let data: Vec<String> = serde_json::from_slice(&value)
                .map_err(|e| Error::Deserialization(e.to_string()))?;
            Ok(Some(data))
        } else {
            Ok(None)
        }
    }
    
    fn import_data(&self, name: &str, path: &Path, format: DataFormat) -> Result<String> {
        // 检查文件是否存在
        if !path.exists() {
            return Err(Error::NotFound(format!("导入文件不存在: {:?}", path)));
        }
        
        if !path.is_file() {
            return Err(Error::InvalidInput(format!("路径不是文件: {:?}", path)));
        }
        
        // 读取文件内容
        let file_content = std::fs::read_to_string(path)
            .map_err(|e| Error::IoError(format!("读取文件失败: {}", e)))?;
        
        // 根据格式解析数据
        let data_lines = match format {
            DataFormat::CSV | DataFormat::Csv => {
                // CSV格式：每行是一条记录
                file_content.lines().map(|s| s.to_string()).collect::<Vec<String>>()
            },
            DataFormat::JSON | DataFormat::Json => {
                // JSON格式：解析为JSON数组，每行一个JSON对象
                let json_value: Value = serde_json::from_str(&file_content)
                    .map_err(|e| Error::Deserialization(format!("解析JSON失败: {}", e)))?;
                
                match json_value {
                    Value::Array(arr) => {
                        arr.into_iter().map(|v| serde_json::to_string(&v)
                            .unwrap_or_else(|_| v.to_string())).collect()
                    },
                    Value::Object(_) => {
                        vec![serde_json::to_string(&json_value)
                            .unwrap_or_else(|_| json_value.to_string())]
                    },
                    _ => vec![json_value.to_string()],
                }
            },
            DataFormat::TSV | DataFormat::Tsv => {
                // TSV格式：每行是一条记录
                file_content.lines().map(|s| s.to_string()).collect::<Vec<String>>()
            },
            DataFormat::Text => {
                // 文本格式：每行是一条记录
                file_content.lines().map(|s| s.to_string()).collect::<Vec<String>>()
            },
            _ => {
                // 其他格式：按行处理
                file_content.lines().map(|s| s.to_string()).collect::<Vec<String>>()
            },
        };
        
        // 生成数据ID
        let data_id = Uuid::new_v4().to_string();
        
        // 存储原始数据
        self.store_raw_data(&data_id, data_lines.clone())?;
        
        // 存储数据元数据信息
        let data_info = DataInfo {
            id: data_id.clone(),
            name: name.to_string(),
            format: format.clone(),
            size: file_content.len() as u64,
            created_at: Utc::now(),
        };
        
        let info_key = format!("{}info:{}", DATA_RAW_PREFIX, data_id);
        let info_value = serde_json::to_string(&data_info)
            .map_err(|e| Error::Serialization(format!("序列化数据信息失败: {}", e)))?;
        self.put(info_key.as_bytes(), info_value.as_bytes())?;
        
        Ok(data_id)
    }
    
    fn convert_data_format(&self, id: &str, from: DataFormat, to: DataFormat) -> Result<()> {
        // 如果格式相同，无需转换
        if from == to {
            return Ok(());
        }
        
        // 获取原始数据
        let raw_data = self.get_raw_data(id)?
            .ok_or_else(|| Error::NotFound(format!("数据不存在: {}", id)))?;
        
        // 根据源格式和目标格式进行转换
        let converted_data = match (from, to) {
            // CSV/TSV -> JSON
            (DataFormat::CSV | DataFormat::Csv | DataFormat::TSV | DataFormat::Tsv, DataFormat::JSON | DataFormat::Json) => {
                // 将CSV/TSV行转换为JSON数组
                let json_array: Vec<Value> = raw_data.iter()
                    .filter_map(|line| {
                        if line.is_empty() {
                            None
                        } else {
                            // 简单解析：假设第一行是表头，后续行是数据
                            Some(Value::String(line.clone()))
                        }
                    })
                    .collect();
                json_array.into_iter().map(|v| serde_json::to_string(&v)
                    .unwrap_or_else(|_| v.to_string())).collect()
            },
            // JSON -> CSV
            (DataFormat::JSON | DataFormat::Json, DataFormat::CSV | DataFormat::Csv) => {
                // 将JSON数组转换为CSV行
                raw_data.iter()
                    .filter_map(|line| {
                        if let Ok(json_value) = serde_json::from_str::<Value>(line) {
                            match json_value {
                                Value::Object(map) => {
                                    // 对象转CSV行：键值对用逗号分隔
                                    Some(map.values().map(|v| v.to_string()).collect::<Vec<_>>().join(","))
                                },
                                Value::Array(arr) => {
                                    Some(arr.iter().map(|v| v.to_string()).collect::<Vec<_>>().join(","))
                                },
                                _ => Some(json_value.to_string()),
                            }
                        } else {
                            Some(line.clone())
                        }
                    })
                    .collect()
            },
            // 其他转换：保持原样或简单转换
            _ => {
                // 对于不支持的转换，保持原数据
                raw_data
            },
        };
        
        // 存储转换后的数据
        self.store_raw_data(id, converted_data)?;
        
        // 更新数据信息中的格式
        let info_key = format!("{}info:{}", DATA_RAW_PREFIX, id);
        if let Some(info_bytes) = self.get(info_key.as_bytes())? {
            if let Ok(mut info) = serde_json::from_slice::<DataInfo>(&info_bytes) {
                info.format = to.clone();
                let updated_info = serde_json::to_string(&info)
                    .map_err(|e| Error::Serialization(format!("序列化数据信息失败: {}", e)))?;
                self.put(info_key.as_bytes(), updated_info.as_bytes())?;
            }
        }
        
        Ok(())
    }
    
    fn get_data_as_json(&self, id: &str) -> Result<Value> {
        // 将数据转换为JSON格式返回
        if let Some(raw_data) = self.get_raw_data(id)? {
            let value = serde_json::to_value(raw_data)
                    .map_err(|e| Error::Serialization(e.to_string()))?;
            Ok(value)
        } else {
            Err(Error::NotFound(format!("数据不存在: {}", id)))
        }
    }
    
    fn get_data_as_csv(&self, id: &str) -> Result<String> {
        // 获取原始数据
        let raw_data = self.get_raw_data(id)?
            .ok_or_else(|| Error::NotFound(format!("数据不存在: {}", id)))?;
        
        // 如果数据是JSON格式，需要转换为CSV
        let info_key = format!("{}info:{}", DATA_RAW_PREFIX, id);
        if let Some(info_bytes) = self.get(info_key.as_bytes())? {
            if let Ok(info) = serde_json::from_slice::<DataInfo>(&info_bytes) {
                if matches!(info.format, DataFormat::JSON | DataFormat::Json) {
                    // JSON转CSV：每行JSON对象转换为CSV行
                    let csv_lines: Vec<String> = raw_data.iter()
                        .filter_map(|line| {
                            if let Ok(json_value) = serde_json::from_str::<Value>(line) {
                                match json_value {
                                    Value::Object(map) => {
                                        let values: Vec<String> = map.values()
                                            .map(|v| {
                                                let s = v.to_string();
                                                // CSV转义：如果包含逗号或引号，需要用引号包裹
                                                if s.contains(',') || s.contains('"') || s.contains('\n') {
                                                    format!("\"{}\"", s.replace("\"", "\"\""))
                                                } else {
                                                    s
                                                }
                                            })
                                            .collect();
                                        Some(values.join(","))
                                    },
                                    Value::Array(arr) => {
                                        let values: Vec<String> = arr.iter()
                                            .map(|v| {
                                                let s = v.to_string();
                                                if s.contains(',') || s.contains('"') || s.contains('\n') {
                                                    format!("\"{}\"", s.replace("\"", "\"\""))
                                                } else {
                                                    s
                                                }
                                            })
                                            .collect();
                                        Some(values.join(","))
                                    },
                                    _ => Some(json_value.to_string()),
                                }
                            } else {
                                Some(line.clone())
                            }
                        })
                        .collect();
                    return Ok(csv_lines.join("\n"));
                }
            }
        }
        
        // 默认：直接返回原始数据，每行一条记录
        Ok(raw_data.join("\n"))
    }
    
    fn list_data(&self) -> Result<Vec<(String, String)>> {
        // 扫描所有以 DATA_RAW_PREFIX 开头的键
        let mut results = Vec::new();
        
        // 使用迭代器扫描前缀
        let iter = self.db().iterator(rocksdb::IteratorMode::From(
            DATA_RAW_PREFIX.as_bytes(),
            rocksdb::Direction::Forward
        ));
        
        for item in iter {
            match item {
                Ok((key, value)) => {
                    // 检查键是否以 DATA_RAW_PREFIX 开头
                    if key.starts_with(DATA_RAW_PREFIX.as_bytes()) {
                        // 跳过 info: 元数据键
                        if let Ok(key_str) = String::from_utf8(key.to_vec()) {
                            if !key_str.starts_with(&format!("{}info:", DATA_RAW_PREFIX)) {
                                // 提取数据ID（去掉前缀）
                                if let Some(data_id) = key_str.strip_prefix(DATA_RAW_PREFIX) {
                                    // 尝试获取数据名称（从元数据）
                                    let info_key = format!("{}info:{}", DATA_RAW_PREFIX, data_id);
                                    let name = if let Some(info_bytes) = self.get(info_key.as_bytes()).ok().flatten() {
                                        if let Ok(info) = serde_json::from_slice::<DataInfo>(&info_bytes) {
                                            info.name
                                        } else {
                                            format!("Data-{}", data_id)
                                        }
                                    } else {
                                        format!("Data-{}", data_id)
                                    };
                                    
                                    results.push((data_id.to_string(), name));
                                }
                            }
                        }
                    } else {
                        // 如果键不再以前缀开头，停止迭代
                        break;
                    }
                },
                Err(e) => {
                    return Err(Error::storage(format!("扫描数据列表失败: {}", e)));
                },
            }
        }
        
        Ok(results)
    }
    
    fn get_data_info(&self, id: &str) -> Result<DataInfo> {
        // 首先检查数据是否存在
        let raw_data = self.get_raw_data(id)?
            .ok_or_else(|| Error::NotFound(format!("数据不存在: {}", id)))?;
        
        // 尝试从元数据获取信息
        let info_key = format!("{}info:{}", DATA_RAW_PREFIX, id);
        if let Some(info_bytes) = self.get(info_key.as_bytes())? {
            if let Ok(mut info) = serde_json::from_slice::<DataInfo>(&info_bytes) {
                // 更新大小为实际数据大小
                let data_size: usize = raw_data.iter().map(|s| s.len()).sum();
                info.size = data_size as u64;
                return Ok(info);
            }
        }
        
        // 如果没有元数据，从实际数据推断信息
        let data_size: usize = raw_data.iter().map(|s| s.len()).sum();
        
        // 尝试推断格式
        let format = if !raw_data.is_empty() {
            // 检查第一行是否是JSON
            if let Ok(_) = serde_json::from_str::<Value>(&raw_data[0]) {
                DataFormat::JSON
            } else if raw_data[0].contains('\t') {
                DataFormat::TSV
            } else {
                DataFormat::CSV
            }
        } else {
            DataFormat::CSV
        };
        
        Ok(DataInfo {
            id: id.to_string(),
            name: format!("Data-{}", id),
            format,
            size: data_size as u64,
            created_at: Utc::now(),
        })
    }
}

