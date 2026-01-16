use std::fs::File;
use std::io::{BufReader, Read, Cursor, Seek};
use std::path::Path;
use std::path::PathBuf;
use std::collections::HashMap;
use byteorder::{LittleEndian, BigEndian, ReadBytesExt};
use crate::error::{Result, Error};
use crate::data::{DataBatch, DataFormat, DataSchema};
use crate::data::loader::{DataLoader, DataSource};
use async_trait::async_trait;

// 自定义二进制格式加载器
pub struct CustomBinaryLoader {
    endianness: Endianness,
    header_size: usize,
    data_format: CustomBinaryFormat,
}

// 字节序枚举
#[derive(Clone)]
pub enum Endianness {
    Little,
    Big,
}

// 自定义二进制格式类型
#[derive(Clone)]
pub enum CustomBinaryFormat {
    VecDB,     // 自定义向量数据库格式
    TensorRaw, // 原始张量格式
    RecordBatch, // 记录批次格式
    Custom(String), // 自定义格式，需提供解析器
}

impl CustomBinaryLoader {
    pub fn new(endianness: Endianness, header_size: usize, format: CustomBinaryFormat) -> Self {
        Self {
            endianness,
            header_size,
            data_format: format,
        }
    }
    
    // 解析VecDB格式
    fn parse_vecdb(&self, data: &[u8]) -> Result<DataBatch> {
        let mut cursor = Cursor::new(data);
        
        // 读取文件头部
        if data.len() < self.header_size {
            return Err(Error::invalid_input("Invalid VecDB file: header too small"));
        }
        
        // 跳过头部
        cursor.set_position(self.header_size as u64);
        
        // 读取记录数量
        let record_count = match self.endianness {
            Endianness::Little => cursor.read_u32::<LittleEndian>()?,
            Endianness::Big => cursor.read_u32::<BigEndian>()?,
        } as usize;
        
        // 读取维度
        let dimension = match self.endianness {
            Endianness::Little => cursor.read_u32::<LittleEndian>()?,
            Endianness::Big => cursor.read_u32::<BigEndian>()?,
        } as usize;
        
        // 初始化特征向量存储
        let mut features = Vec::with_capacity(record_count);
        
        // 读取向量数据
        for _ in 0..record_count {
            let mut vector = Vec::with_capacity(dimension);
            
            for _ in 0..dimension {
                let value = match self.endianness {
                    Endianness::Little => cursor.read_f32::<LittleEndian>()?,
                    Endianness::Big => cursor.read_f32::<BigEndian>()?,
                };
                vector.push(value);
            }
            
            features.push(vector);
        }
        
        // 构建元数据
        let mut metadata = HashMap::new();
        metadata.insert("format".to_string(), "vecdb".to_string());
        metadata.insert("record_count".to_string(), record_count.to_string());
        metadata.insert("dimension".to_string(), dimension.to_string());
        metadata.insert("endianness".to_string(), format!("{:?}", self.endianness));
        
        let mut batch = DataBatch::new("binary", 0, features.len()).with_features(features);
        for (k, v) in metadata.iter() {
            batch = batch.with_metadata(k, v);
        }
        Ok(batch)
    }
    
    // 解析TensorRaw格式
    fn parse_tensor_raw(&self, data: &[u8]) -> Result<DataBatch> {
        let mut cursor = Cursor::new(data);
        
        // 跳过头部
        cursor.set_position(self.header_size as u64);
        
        // 读取维度数量
        let rank = match self.endianness {
            Endianness::Little => cursor.read_u8()?,
            Endianness::Big => cursor.read_u8()?,
        } as usize;
        
        // 读取每个维度的大小
        let mut dimensions = Vec::with_capacity(rank);
        for _ in 0..rank {
            let dim_size = match self.endianness {
                Endianness::Little => cursor.read_u32::<LittleEndian>()?,
                Endianness::Big => cursor.read_u32::<BigEndian>()?,
            } as usize;
            dimensions.push(dim_size);
        }
        
        // 计算总元素数量
        let total_elements = dimensions.iter().product::<usize>();
        
        // 读取数据类型
        let data_type = match self.endianness {
            Endianness::Little => cursor.read_u8()?,
            Endianness::Big => cursor.read_u8()?,
        };
        
        // 根据数据类型读取数据
        let mut values = Vec::with_capacity(total_elements);
        match data_type {
            0 => { // f32
                for _ in 0..total_elements {
                    let value = match self.endianness {
                        Endianness::Little => cursor.read_f32::<LittleEndian>()?,
                        Endianness::Big => cursor.read_f32::<BigEndian>()?,
                    };
                    values.push(value);
                }
            },
            1 => { // f64
                for _ in 0..total_elements {
                    let value = match self.endianness {
                        Endianness::Little => cursor.read_f64::<LittleEndian>()? as f32,
                        Endianness::Big => cursor.read_f64::<BigEndian>()? as f32,
                    };
                    values.push(value);
                }
            },
            2 => { // i32
                for _ in 0..total_elements {
                    let value = match self.endianness {
                        Endianness::Little => cursor.read_i32::<LittleEndian>()? as f32,
                        Endianness::Big => cursor.read_i32::<BigEndian>()? as f32,
                    };
                    values.push(value);
                }
            },
            3 => { // i64
                for _ in 0..total_elements {
                    let value = match self.endianness {
                        Endianness::Little => cursor.read_i64::<LittleEndian>()? as f32,
                        Endianness::Big => cursor.read_i64::<BigEndian>()? as f32,
                    };
                    values.push(value);
                }
            },
            _ => return Err(Error::invalid_input(format!("Unsupported data type: {}", data_type))),
        }
        
        // 根据维度重组特征向量
        let features = if dimensions.len() >= 2 {
            // 假设第一个维度是批次大小，第二个维度是特征维度
            let batch_size = dimensions[0];
            let feature_dim = dimensions[1];
            
            let mut batch_features = Vec::with_capacity(batch_size);
            for i in 0..batch_size {
                let start = i * feature_dim;
                let end = start + feature_dim;
                batch_features.push(values[start..end].to_vec());
            }
            batch_features
        } else if dimensions.len() == 1 {
            // 单个特征向量
            vec![values]
        } else {
            return Err(Error::invalid_input("Invalid tensor dimensions"));
        };
        
        // 构建元数据
        let mut metadata = HashMap::new();
        metadata.insert("format".to_string(), "tensor_raw".to_string());
        metadata.insert("rank".to_string(), rank.to_string());
        for (i, dim) in dimensions.iter().enumerate() {
            metadata.insert(format!("dim_{}", i), dim.to_string());
        }
        metadata.insert("data_type".to_string(), data_type.to_string());
        metadata.insert("endianness".to_string(), format!("{:?}", self.endianness));
        
        let mut batch = DataBatch::new("tensor_raw", 0, features.len()).with_features(features);
        for (k, v) in metadata.iter() {
            batch = batch.with_metadata(k, v);
        }
        Ok(batch)
    }
    
    // 解析RecordBatch格式
    fn parse_record_batch(&self, data: &[u8]) -> Result<DataBatch> {
        let mut cursor = Cursor::new(data);
        
        // 跳过头部
        cursor.set_position(self.header_size as u64);
        
        // 读取列数量
        let column_count = match self.endianness {
            Endianness::Little => cursor.read_u16::<LittleEndian>()?,
            Endianness::Big => cursor.read_u16::<BigEndian>()?,
        } as usize;
        
        // 读取行数
        let row_count = match self.endianness {
            Endianness::Little => cursor.read_u32::<LittleEndian>()?,
            Endianness::Big => cursor.read_u32::<BigEndian>()?,
        } as usize;
        
        // 读取列名长度
        let mut column_names = Vec::with_capacity(column_count);
        for _ in 0..column_count {
            let name_len = match self.endianness {
                Endianness::Little => cursor.read_u16::<LittleEndian>()?,
                Endianness::Big => cursor.read_u16::<BigEndian>()?,
            } as usize;
            
            let mut name_buffer = vec![0u8; name_len];
            cursor.read_exact(&mut name_buffer)?;
            
            let column_name = String::from_utf8(name_buffer)
                .map_err(|e| Error::invalid_input(format!("Invalid UTF-8 sequence in column name: {}", e)))?;
                
            column_names.push(column_name);
        }
        
        // 读取列类型
        let mut column_types = Vec::with_capacity(column_count);
        for _ in 0..column_count {
            let type_code = match self.endianness {
                Endianness::Little => cursor.read_u8()?,
                Endianness::Big => cursor.read_u8()?,
            };
            column_types.push(type_code);
        }
        
        // 初始化特征向量存储
        let mut features = Vec::with_capacity(row_count);
        
        // 读取数据行
        for _ in 0..row_count {
            let mut row_values = Vec::with_capacity(column_count);
            
            for col_idx in 0..column_count {
                let type_code = column_types[col_idx];
                
                let value = match type_code {
                    0 => { // NULL
                        0.0
                    },
                    1 => { // FLOAT32
                        match self.endianness {
                            Endianness::Little => cursor.read_f32::<LittleEndian>()?,
                            Endianness::Big => cursor.read_f32::<BigEndian>()?,
                        }
                    },
                    2 => { // FLOAT64
                        let value = (match self.endianness {
                            Endianness::Little => cursor.read_f64::<LittleEndian>()?,
                            Endianness::Big => cursor.read_f64::<BigEndian>()?,
                        }) as f32; // 转换为f32
                        value
                    },
                    3 => { // INT32
                        let value = (match self.endianness {
                            Endianness::Little => cursor.read_i32::<LittleEndian>()?,
                            Endianness::Big => cursor.read_i32::<BigEndian>()?,
                        }) as f32; // 转换为f32
                        value
                    },
                    4 => { // INT64
                        let value = (match self.endianness {
                            Endianness::Little => cursor.read_i64::<LittleEndian>()?,
                            Endianness::Big => cursor.read_i64::<BigEndian>()?,
                        }) as f32; // 转换为f32
                        value
                    },
                    5 => { // BOOLEAN
                        let bool_val = match self.endianness {
                            Endianness::Little => cursor.read_u8()?,
                            Endianness::Big => cursor.read_u8()?,
                        };
                        if bool_val > 0 { 1.0 } else { 0.0 }
                    },
                    6 => { // STRING
                        // 字符串类型，读取长度
                        let str_len = match self.endianness {
                            Endianness::Little => cursor.read_u16::<LittleEndian>()?,
                            Endianness::Big => cursor.read_u16::<BigEndian>()?,
                        } as usize;
                        
                        // 读取字符串内容
                        let mut str_buffer = vec![0u8; str_len];
                        cursor.read_exact(&mut str_buffer)?;
                        
                        // 为了特征向量，我们使用字符串的简单哈希作为数值表示
                        let mut hash_value: u32 = 0;
                        for byte in str_buffer {
                            hash_value = hash_value.wrapping_add(byte as u32);
                            hash_value = hash_value.wrapping_mul(31);
                        }
                        // 归一化到 0-1 范围
                        (hash_value % 1000) as f32 / 1000.0
                    },
                    _ => return Err(Error::invalid_input(format!("Unsupported column type: {}", type_code))),
                };
                
                row_values.push(value);
            }
            
            // 添加到特征向量
            features.push(row_values);
        }
        
        // 构建元数据
        let mut metadata = HashMap::new();
        metadata.insert("format".to_string(), "record_batch".to_string());
        metadata.insert("column_count".to_string(), column_count.to_string());
        metadata.insert("row_count".to_string(), row_count.to_string());
        metadata.insert("columns".to_string(), column_names.join(","));
        metadata.insert("endianness".to_string(), format!("{:?}", self.endianness));
        
        let mut batch = DataBatch::new("record_batch", 0, features.len()).with_features(features);
        for (k, v) in metadata.iter() {
            batch = batch.with_metadata(k, v);
        }
        Ok(batch)
    }
    
    // 从文件加载二进制数据
    fn load_from_file(&self, path: &str) -> Result<Vec<u8>> {
        let file = File::open(path)
            .map_err(|e| Error::io_error(format!("Failed to open binary file: {}, error: {}", path, e)))?;
            
        let mut reader = BufReader::new(file);
        let mut buffer = Vec::new();
        
        reader.read_to_end(&mut buffer)
            .map_err(|e| Error::io_error(format!("Failed to read binary file: {}, error: {}", path, e)))?;
            
        Ok(buffer)
    }
}

#[async_trait]
impl DataLoader for CustomBinaryLoader {
    async fn load(&self, source: &DataSource, format: &crate::data::loader::types::DataFormat) -> Result<DataBatch> {
        match source {
            DataSource::File(path) => {
                // 加载文件内容
                let data = self.load_from_file(path)?;
                
                // 根据格式解析数据
                match &format {
                    crate::data::loader::types::DataFormat::CustomBinary(fmt) => {
                        match &self.data_format {
                            CustomBinaryFormat::VecDB => self.parse_vecdb(&data),
                            CustomBinaryFormat::TensorRaw => self.parse_tensor_raw(&data),
                            CustomBinaryFormat::RecordBatch => self.parse_record_batch(&data),
                            CustomBinaryFormat::Custom(name) => {
                                if name == fmt {
                                    // 使用自定义解析器
                                    self.parse_record_batch(&data) // 默认使用RecordBatch解析器
                                } else {
                                    Err(Error::invalid_argument(format!("Format mismatch: expected {}, got {}", name, fmt)))
                                }
                            }
                        }
                    },
                    _ => Err(Error::invalid_argument(format!("Unsupported format for binary loader: {:?}", format))),
                }
            },
            DataSource::Memory(data) => {
                // 解析内存数据
                match &format {
                    crate::data::loader::types::DataFormat::CustomBinary(fmt) => {
                        match self.data_format {
                            CustomBinaryFormat::VecDB => self.parse_vecdb(data),
                            CustomBinaryFormat::TensorRaw => self.parse_tensor_raw(data),
                            CustomBinaryFormat::RecordBatch => self.parse_record_batch(data),
                            CustomBinaryFormat::Custom(name) => {
                                if name == *fmt {
                                    self.parse_record_batch(data)
                                } else {
                                    Err(Error::invalid_argument(format!("格式不匹配: 期望 {}, 得到 {}", name, fmt)))
                                }
                            }
                        }
                    },
                    _ => Err(Error::invalid_argument(format!("二进制加载器不支持格式: {:?}", format))),
                }
            },
            _ => Err(Error::invalid_argument("二进制加载器只支持文件和内存数据源")),
        }
    }
    
    async fn get_schema(&self, source: &DataSource, format: &crate::data::loader::types::DataFormat) -> Result<DataSchema> {
        match source {
            DataSource::File(path) => {
                // 加载文件头部用于推断Schema
                let file = File::open(path)
                    .map_err(|e| Error::io_error(format!("Failed to open binary file for schema: {}, error: {}", path, e)))?;
                    
                let mut reader = BufReader::new(file);
                
                // 只读取头部数据
                let header_size = std::cmp::max(self.header_size, 1024); // 至少读取1KB
                let mut buffer = vec![0u8; header_size];
                let read_size = reader.read(&mut buffer)
                    .map_err(|e| Error::io_error(format!("Failed to read binary file header: {}, error: {}", path, e)))?;
                    
                buffer.truncate(read_size);
                
                // 创建一个基本的Schema
                let mut schema = DataSchema::default();
                
                // 根据格式推断Schema
                match &format {
                    crate::data::loader::types::DataFormat::CustomBinary(fmt) => {
                        // 设置格式信息
                        schema.metadata.insert("format".to_string(), fmt.clone());
                        schema.metadata.insert("endianness".to_string(), format!("{:?}", self.endianness));
                        
                        match self.data_format {
                            CustomBinaryFormat::VecDB => {
                                // 尝试从VecDB头部读取维度信息
                                if buffer.len() >= self.header_size + 8 {
                                    let mut cursor = Cursor::new(&buffer);
                                    cursor.set_position(self.header_size as u64);
                                    
                                    if let Ok(dimension) = match self.endianness {
                                        Endianness::Little => cursor.read_u32::<LittleEndian>().map(|v| v as usize),
                                        Endianness::Big => cursor.read_u32::<BigEndian>().map(|v| v as usize),
                                    } {
                                        schema.metadata.insert("dimension".to_string(), dimension.to_string());
                                        // feature_fields 是方法，不能直接赋值，使用 metadata 存储特征字段信息
                                        schema.metadata.insert("feature_fields".to_string(), serde_json::to_string(&vec!["vector".to_string()]).unwrap_or_default());
                                    }
                                }
                            },
                            CustomBinaryFormat::TensorRaw => {
                                // 尝试从TensorRaw头部读取维度信息
                                if buffer.len() >= self.header_size + 1 {
                                    let mut cursor = Cursor::new(&buffer);
                                    cursor.set_position(self.header_size as u64);
                                    
                                    if let Ok(rank) = cursor.read_u8() {
                                        schema.metadata.insert("rank".to_string(), rank.to_string());
                                        // feature_fields 是方法，不能直接赋值，使用 metadata 存储特征字段信息
                                        schema.metadata.insert("feature_fields".to_string(), serde_json::to_string(&vec!["tensor".to_string()]).unwrap_or_default());
                                    }
                                }
                            },
                            CustomBinaryFormat::RecordBatch => {
                                // 尝试从RecordBatch头部读取列信息
                                if buffer.len() >= self.header_size + 6 {
                                    let mut cursor = Cursor::new(&buffer);
                                    cursor.set_position(self.header_size as u64);
                                    
                                    if let Ok(column_count) = match self.endianness {
                                        Endianness::Little => cursor.read_u16::<LittleEndian>().map(|v| v as usize),
                                        Endianness::Big => cursor.read_u16::<BigEndian>().map(|v| v as usize),
                                    } {
                                        schema.metadata.insert("column_count".to_string(), column_count.to_string());
                                        
                                        // 尝试读取列名
                                        let mut column_names = Vec::new();
                                        let mut pos = cursor.position() + 4; // 跳过row_count
                                        
                                        for _ in 0..column_count {
                                            if pos + 2 <= buffer.len() as u64 {
                                                cursor.set_position(pos);
                                                if let Ok(name_len) = match self.endianness {
                                                    Endianness::Little => cursor.read_u16::<LittleEndian>().map(|v| v as usize),
                                                    Endianness::Big => cursor.read_u16::<BigEndian>().map(|v| v as usize),
                                                } {
                                                    pos += 2;
                                                    
                                                    if pos + name_len as u64 <= buffer.len() as u64 {
                                                        let mut name_buffer = vec![0u8; name_len];
                                                        cursor.set_position(pos);
                                                        if cursor.read_exact(&mut name_buffer).is_ok() {
                                                            if let Ok(name) = String::from_utf8(name_buffer) {
                                                                column_names.push(name);
                                                                pos += name_len as u64;
                                                            } else {
                                                                break;
                                                            }
                                                        } else {
                                                            break;
                                                        }
                                                    } else {
                                                        break;
                                                    }
                                                } else {
                                                    break;
                                                }
                                            } else {
                                                break;
                                            }
                                        }
                                        
                                        if !column_names.is_empty() {
                                            // feature_fields 是方法，不能直接赋值，使用 metadata 存储特征字段信息
                                            schema.metadata.insert("feature_fields".to_string(), serde_json::to_string(&column_names).unwrap_or_default());
                                        }
                                    }
                                }
                            },
                            CustomBinaryFormat::Custom(_) => {
                                // 自定义格式无法推断Schema
                                schema.metadata.insert("status".to_string(), "unknown".to_string());
                            }
                        }
                    },
                    _ => return Err(Error::invalid_argument(format!("Unsupported format for binary schema: {:?}", format))),
                }
                
                Ok(schema)
            },
            _ => Err(Error::invalid_argument("Binary loader only supports file sources for schema")),
        }
    }
    
    fn name(&self) -> &'static str {
        "CustomBinaryLoader"
    }
    
    async fn get_size(&self, path: &str) -> Result<usize> {
        // 获取文件大小
        let metadata = std::fs::metadata(path)
            .map_err(|e| Error::io_error(format!("无法获取文件元数据: {}, 错误: {}", path, e)))?;
            
        // 估算记录数量
        let file_size = metadata.len() as usize;
        if file_size <= self.header_size {
            return Ok(0);
        }
        
        // 从文件头部推断记录数量
        let file = File::open(path)
            .map_err(|e| Error::io_error(format!("无法打开文件: {}, 错误: {}", path, e)))?;
        let mut reader = BufReader::new(file);
        
        // 跳过头部
        reader.seek(std::io::SeekFrom::Start(self.header_size as u64))
            .map_err(|e| Error::io_error(format!("无法定位文件头部后: {}, 错误: {}", path, e)))?;
        
        // 尝试读取记录数量
        match self.data_format {
            CustomBinaryFormat::VecDB => {
                // VecDB格式直接存储记录数量
                let record_count = match self.endianness {
                    Endianness::Little => reader.read_u32::<LittleEndian>(),
                    Endianness::Big => reader.read_u32::<BigEndian>(),
                };
                
                match record_count {
                    Ok(count) => Ok(count as usize),
                    Err(_) => Ok(0),
                }
            },
            CustomBinaryFormat::RecordBatch => {
                // RecordBatch格式在列数量后存储行数
                // 跳过列数量
                match self.endianness {
                    Endianness::Little => reader.read_u16::<LittleEndian>(),
                    Endianness::Big => reader.read_u16::<BigEndian>(),
                }.ok();
                
                // 读取行数
                let row_count = match self.endianness {
                    Endianness::Little => reader.read_u32::<LittleEndian>(),
                    Endianness::Big => reader.read_u32::<BigEndian>(),
                };
                
                match row_count {
                    Ok(count) => Ok(count as usize),
                    Err(_) => Ok(0),
                }
            },
            _ => {
                // 对于其他格式，返回估计值
                Ok((file_size - self.header_size) / 100) // 假设每条记录平均100字节
            }
        }
    }
    
    async fn get_batch_at(&self, path: &str, index: usize, batch_size: usize) -> Result<DataBatch> {
        // 这个实现依赖于加载整个文件并截取部分，并不高效
        // 对于大文件，应该实现更高效的分段读取
        
        // 加载整个文件
        let data = self.load_from_file(path)?;
        
        // 根据格式解析数据
        let format = if path.ends_with(".vecdb") {
            DataFormat::CustomBinary("vecdb".to_string())
        } else if path.ends_with(".tensor") {
            DataFormat::CustomBinary("tensor_raw".to_string())
        } else if path.ends_with(".records") {
            DataFormat::CustomBinary("record_batch".to_string())
        } else {
            DataFormat::CustomBinary("custom".to_string())
        };
        
        let full_batch = match self.data_format {
            CustomBinaryFormat::VecDB => self.parse_vecdb(&data)?,
            CustomBinaryFormat::TensorRaw => self.parse_tensor_raw(&data)?,
            CustomBinaryFormat::RecordBatch => self.parse_record_batch(&data)?,
            CustomBinaryFormat::Custom(_) => self.parse_record_batch(&data)?,
        };
        
        // 截取指定范围的数据
        let start = index * batch_size;
        let total_size = full_batch.batch_size();
        
        if start >= total_size {
            return Err(Error::invalid_argument(format!("Batch start index {} exceeds total size {}", start, total_size)));
        }
        
        let end = std::cmp::min(start + batch_size, total_size);
        let sliced_batch = full_batch.slice(start, end)?;
        
        Ok(sliced_batch)
    }
    
    // 添加缺失的trait方法
    async fn load_batch(&self, source: &DataSource, format: &crate::data::loader::types::DataFormat, batch_size: usize, offset: usize) -> Result<DataBatch> {
        // 加载完整数据后进行批次分割
        let full_batch = self.load(source, format).await?;
        
        // 计算实际的开始和结束位置
        let total_size = full_batch.batch_size();
        let start = offset;
        let end = std::cmp::min(start + batch_size, total_size);
        
        if start >= total_size {
            // 超出范围，返回空批次
            return Ok(DataBatch::default());
        }
        
        // 切割批次
        full_batch.slice(start, end)
    }
    
    fn supports_format(&self, format: &crate::data::loader::types::DataFormat) -> bool {
        match format {
            crate::data::loader::types::DataFormat::CustomBinary(_) => true,
            _ => false,
        }
    }
    
    fn config(&self) -> &crate::data::loader::LoaderConfig {
        // 返回一个默认配置的引用，CustomBinaryLoader使用内部配置
        &crate::data::loader::LoaderConfig::default()
    }
    
    fn set_config(&mut self, config: crate::data::loader::LoaderConfig) {
        // CustomBinaryLoader有自己的内部配置，这里可以从LoaderConfig中提取相关设置
        if let Some(format) = config.format {
            // 根据格式调整内部设置
            match format {
                crate::data::loader::types::DataFormat::CustomBinary(fmt) => {
                    // 根据自定义格式调整内部设置
                    match fmt.as_str() {
                        "vecdb" => self.data_format = CustomBinaryFormat::VecDB,
                        "tensor" => self.data_format = CustomBinaryFormat::TensorRaw,
                        "record" => self.data_format = CustomBinaryFormat::RecordBatch,
                        _ => self.data_format = CustomBinaryFormat::Custom(fmt),
                    }
                },
                _ => {
                    // 其他格式不影响二进制加载器配置
                }
            }
        }
        
        // 从配置选项中提取其他设置
        if let Some(endianness_str) = config.options.get("endianness") {
            if let Ok(endianness) = endianness_str.parse::<Endianness>() {
                self.endianness = endianness;
            }
        }
        
        if let Some(header_size_str) = config.options.get("header_size") {
            if let Ok(header_size) = header_size_str.parse::<usize>() {
                self.header_size = header_size;
            }
        }
    }
}

impl std::str::FromStr for Endianness {
    type Err = Error;
    
    fn from_str(s: &str) -> Result<Self> {
        match s.to_lowercase().as_str() {
            "little" => Ok(Endianness::Little),
            "big" => Ok(Endianness::Big),
            _ => Err(Error::invalid_argument(format!("Invalid endianness: {}", s))),
        }
    }
}

impl std::str::FromStr for CustomBinaryFormat {
    type Err = Error;
    
    fn from_str(s: &str) -> Result<Self> {
        match s.to_lowercase().as_str() {
            "vecdb" => Ok(CustomBinaryFormat::VecDB),
            "tensor_raw" | "tensorraw" => Ok(CustomBinaryFormat::TensorRaw),
            "record_batch" | "recordbatch" => Ok(CustomBinaryFormat::RecordBatch),
            _ => Ok(CustomBinaryFormat::Custom(s.to_string())),
        }
    }
}

impl std::fmt::Debug for Endianness {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Endianness::Little => write!(f, "little"),
            Endianness::Big => write!(f, "big"),
        }
    }
}

impl std::fmt::Debug for CustomBinaryFormat {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CustomBinaryFormat::VecDB => write!(f, "vecdb"),
            CustomBinaryFormat::TensorRaw => write!(f, "tensor_raw"),
            CustomBinaryFormat::RecordBatch => write!(f, "record_batch"),
            CustomBinaryFormat::Custom(name) => write!(f, "custom({})", name),
        }
    }
}

/// 检查二进制文件是否存在且可读
pub fn validate_binary_file(path: &Path) -> Result<bool> {
    if !path.exists() {
        return Err(Error::not_found(path.to_string_lossy().to_string()));
    }
    
    if !path.is_file() {
        return Err(Error::invalid_input(format!("路径不是文件: {}", path.display())));
    }
    
    // 尝试打开文件，验证读取权限
    let _file = File::open(path)?;
    
    Ok(true)
}

/// 获取二进制文件大小
pub fn get_binary_file_size(path: &Path) -> Result<u64> {
    let metadata = std::fs::metadata(path)?;
    Ok(metadata.len())
}

/// 读取二进制文件头部数据以进行格式检测
pub fn read_binary_file_header(path: &Path, bytes: usize) -> Result<Vec<u8>> {
    let mut file = File::open(path)?;
    let mut buffer = vec![0; bytes];
    
    match file.read(&mut buffer) {
        Ok(n) if n < bytes => {
            // 文件比请求的字节数小，调整buffer大小
            buffer.truncate(n);
            Ok(buffer)
        },
        Ok(_) => Ok(buffer),
        Err(e) => Err(Error::io_error(format!("读取文件头部失败: {}", e))),
    }
}

/// 检测二进制文件格式
pub fn detect_binary_format(path: &Path) -> Result<String> {
    // 读取文件头部
    let header = read_binary_file_header(path, 16)?;
    
    // 检测常见二进制格式的魔数
    if header.starts_with(b"PAR1") {
        return Ok("parquet".to_string());
    } else if header.starts_with(&[0x4F, 0x62, 0x6A, 0x01]) {
        return Ok("avro".to_string());
    } else if header.starts_with(&[0x50, 0x4B, 0x03, 0x04]) {
        return Ok("zip".to_string());
    } else if header.starts_with(&[0x1F, 0x8B]) {
        return Ok("gzip".to_string());
    } else if header.starts_with(&[0x42, 0x5A, 0x68]) {
        return Ok("bzip2".to_string());
    } else if header.starts_with(&[0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A]) {
        return Ok("png".to_string());
    } else if header.starts_with(&[0xFF, 0xD8, 0xFF]) {
        return Ok("jpeg".to_string());
    }
    
    // 无法识别格式
    Ok("unknown".to_string())
}

/// 标准化二进制文件路径
pub fn normalize_binary_path(path: &str) -> Result<PathBuf> {
    let path_buf = Path::new(path).to_path_buf();
    
    // 验证路径
    if !path_buf.exists() {
        return Err(Error::not_found(path.to_string()));
    }
    
    Ok(path_buf)
} 