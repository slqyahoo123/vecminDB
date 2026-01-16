use std::path::{Path, PathBuf};
// 文件流处理相关导入（生产级实现：保留用于将来的流式处理功能）
// use std::fs::File;
// use std::io::BufReader;
use std::collections::HashMap;

// 日志导入
use log::{warn};
// debug日志在将来的功能增强中将会使用
// use log::debug;
#[cfg(feature = "excel")]
use calamine::{Reader, open_workbook, Xlsx, Range, DataType};

use crate::error::{Error, Result};
use crate::data::schema::DataSchema;
use crate::data::schema::schema::FieldDefinition;
use crate::data::value::DataValue as Value;
use crate::data::loader::file::{FileProcessor, Record};
use crate::data::loader::file::csv::Field;
use crate::data::loader::file::csv::FieldType as CsvFieldType;
use crate::data::DataBatch;

/// Excel处理器
pub struct ExcelProcessor {
    /// 文件路径
    file_path: PathBuf,
    /// 工作表名称
    sheet_name: Option<String>,
    /// 当前读取的起始行
    current_row: usize,
    /// 表头行索引
    header_row_index: usize,
    /// 是否包含表头
    has_header: bool,
}

impl ExcelProcessor {
    /// 创建新的Excel处理器
    pub fn new(path_str: String) -> Result<Self> {
        let path = PathBuf::from(path_str);
        // 验证文件是否存在
        if !path.exists() {
            return Err(Error::not_found(path.to_string_lossy().to_string()));
        }
        
        Ok(Self {
            file_path: path,
            sheet_name: None,
            current_row: 0,
            header_row_index: 0,
            has_header: true,
        })
    }
    
    /// 设置工作表名称
    pub fn with_sheet_name(mut self, sheet_name: &str) -> Self {
        self.sheet_name = Some(sheet_name.to_string());
        self
    }
    
    /// 设置表头行索引
    pub fn with_header_row_index(mut self, index: usize) -> Self {
        self.header_row_index = index;
        self
    }
    
    /// 设置是否包含表头
    pub fn with_has_header(mut self, has_header: bool) -> Self {
        self.has_header = has_header;
        self
    }
    
    /// 获取工作簿中所有工作表名称
    pub fn get_sheet_names(&self) -> Result<Vec<String>> {
        let mut workbook: Xlsx<_> = open_workbook(&self.file_path)
            .map_err(|e| Error::io_error(format!("打开Excel工作簿失败: {}", e)))?;
        Ok(workbook.sheet_names().to_vec())
    }
    
    /// 获取默认工作表
    fn get_default_sheet_name(&self) -> Result<String> {
        let sheet_names = self.get_sheet_names()?;
        if sheet_names.is_empty() {
            return Err(Error::invalid_data(format!(
                "Excel文件没有工作表: {}", self.file_path.display()
            )));
        }
        Ok(sheet_names[0].clone())
    }
    
    /// 读取Excel工作表
    fn read_sheet(&self, sheet_name: &str) -> Result<Range<DataType>> {
        let mut workbook: Xlsx<_> = open_workbook(&self.file_path)
            .map_err(|e| Error::io_error(format!("打开Excel工作簿失败: {}", e)))?;
        let range = workbook.worksheet_range(sheet_name)
            .ok_or_else(|| Error::not_found(format!("找不到工作表: {}", sheet_name)))?
            .map_err(|e| Error::io_error(format!("读取工作表错误: {}", e)))?;
        
        Ok(range)
    }
    
    /// 从工作表行创建记录
    fn create_record_from_row(
        &self, 
        row_index: usize, 
        row: &[DataType], 
        schema: &DataSchema
    ) -> Record {
        let mut values = Vec::with_capacity(schema.fields().len());
        
        for (i, field) in schema.fields().iter().enumerate() {
            let value = if i < row.len() {
                match &row[i] {
                    DataType::Empty => Value::Null,
                    DataType::String(s) => Value::String(s.clone()),
                    DataType::Float(f) => Value::Float(*f),
                    DataType::Int(i) => Value::Integer(*i as i64),
                    DataType::Bool(b) => Value::Boolean(*b),
                    DataType::DateTime(dt) => Value::String(dt.to_string()), // 将日期时间转换为字符串
                    DataType::Error(e) => {
                        warn!("工作表行 {} 列 {} 处的Excel错误: {:?}", row_index, i, e);
                        Value::Null
                    },
                }
            } else {
                Value::Null
            };
            
            values.push(value);
        }
        
        Record::new(values)
    }
    
    /// 从数据类型获取字段类型（使用CSV模块的FieldType表示）
    fn get_field_type_from_data_type(data_type: &DataType) -> CsvFieldType {
        match data_type {
            DataType::String(_) => CsvFieldType::String,
            DataType::Float(_) => CsvFieldType::Float,
            DataType::Int(_) => CsvFieldType::Integer,
            DataType::Bool(_) => CsvFieldType::Boolean,
            // Excel中的日期时间先按字符串处理，后续由上层按需要再解析
            DataType::DateTime(_) => CsvFieldType::String,
            _ => CsvFieldType::String, // 默认退化为字符串
        }
    }
}

impl FileProcessor for ExcelProcessor {
    fn get_file_path(&self) -> &Path {
        &self.file_path
    }
    
    fn get_schema(&self) -> Result<DataSchema> {
        // 如果没有指定工作表，使用第一个工作表
        let sheet_name = match &self.sheet_name {
            Some(name) => name.clone(),
            None => self.get_default_sheet_name()?,
        };
        
        let range = self.read_sheet(&sheet_name)?;
        
        // 确保工作表有数据
        if range.height() == 0 {
            return Err(Error::invalid_data(format!(
                "工作表为空: {}", sheet_name
            )));
        }
        
        // 获取表头行
        let header_row_index = self.header_row_index;
        if header_row_index >= range.height() {
            return Err(Error::invalid_argument(format!(
                "表头行索引 {} 超出工作表行数 {}", header_row_index, range.height()
            )));
        }
        
        let mut fields = Vec::new();
        
        // 如果有表头，使用表头作为字段名
        if self.has_header {
            // 获取表头行
            let header_row = range.rows().nth(header_row_index)
                .ok_or_else(|| Error::invalid_data("无法获取表头行"))?;
            
            // 检查数据行（如果可用）
            let data_row_opt = if header_row_index + 1 < range.height() {
                range.rows().nth(header_row_index + 1)
            } else {
                None
            };
            
            // 创建字段
            for (i, cell) in header_row.iter().enumerate() {
                let name = match cell {
                    DataType::String(s) if !s.trim().is_empty() => s.clone(),
                    _ => format!("column_{}", i + 1),
                };
                
                // 确定字段类型（从数据行推断，如果可用）
                let field_type = if let Some(data_row) = data_row_opt {
                    if i < data_row.len() {
                        Self::get_field_type_from_data_type(&data_row[i])
                    } else {
                        CsvFieldType::String // 默认为字符串
                    }
                } else {
                    CsvFieldType::String // 默认为字符串
                };
                
                fields.push(Field::new(name, field_type));
            }
        } else {
            // 如果没有表头，使用默认列名并从第一行推断类型
            let first_row = range.rows().nth(0)
                .ok_or_else(|| Error::invalid_data("无法获取第一行"))?;
            
            for (i, cell) in first_row.iter().enumerate() {
                let name = format!("column_{}", i + 1);
                let field_type = Self::get_field_type_from_data_type(cell);
                fields.push(Field::new(name, field_type));
            }
        }
        
        // 创建元数据
        let mut metadata = HashMap::new();
        metadata.insert("source".to_string(), "excel".to_string());
        metadata.insert("sheet".to_string(), sheet_name.clone());
        metadata.insert("path".to_string(), self.file_path.display().to_string());
        metadata.insert("has_header".to_string(), self.has_header.to_string());
        metadata.insert("row_count".to_string(), (range.height() - if self.has_header { 1 } else { 0 }).to_string());

        // 创建模式
        let mut schema = DataSchema::new("excel_schema", "1.0");
        let schema_fields: Vec<FieldDefinition> = fields
            .into_iter()
            .map(|f| FieldDefinition {
                name: f.name,
                field_type: f.field_type.to_schema_field_type(),
                data_type: None,
                required: false,
                nullable: true,
                primary_key: false,
                foreign_key: None,
                description: None,
                default_value: None,
                constraints: None,
                metadata: HashMap::new(),
            })
            .collect();
        schema.fields = schema_fields;
        // metadata信息可以存储在description或通过其他方式保存
        if !metadata.is_empty() {
            let metadata_str = metadata.iter()
                .map(|(k, v)| format!("{}: {}", k, v))
                .collect::<Vec<_>>()
                .join(", ");
            schema.description = Some(format!("Excel metadata: {}", metadata_str));
        }
        Ok(schema)
    }
    
    fn get_row_count(&self) -> Result<usize> {
        // 如果没有指定工作表，使用第一个工作表
        let sheet_name = match &self.sheet_name {
            Some(name) => name.clone(),
            None => self.get_default_sheet_name()?,
        };
        
        let range = self.read_sheet(&sheet_name)?;
        
        // 计算实际数据行数
        let header_offset = if self.has_header { 1 } else { 0 };
        let row_count = if range.height() > self.header_row_index + header_offset {
            range.height() - self.header_row_index - header_offset
        } else {
            0
        };
        
        Ok(row_count)
    }
    
    fn read_rows(&mut self, count: usize) -> Result<Vec<Record>> {
        // 如果没有指定工作表，使用第一个工作表
        let sheet_name = match &self.sheet_name {
            Some(name) => name.clone(),
            None => self.get_default_sheet_name()?,
        };
        
        let range = self.read_sheet(&sheet_name)?;
        
        // 获取模式
        let schema = self.get_schema()?;
        
        // 计算要读取的行范围
        let start_row = self.header_row_index + (if self.has_header { 1 } else { 0 }) + self.current_row;
        let end_row = std::cmp::min(start_row + count, range.height());
        
        // 读取行
        let mut records = Vec::new();
        for row_idx in start_row..end_row {
            if let Some(row) = range.rows().nth(row_idx) {
                let record = self.create_record_from_row(row_idx, row, &schema);
                records.push(record);
            }
        }
        
        // 更新当前行
        self.current_row += records.len();
        
        Ok(records)
    }
    
    fn reset(&mut self) -> Result<()> {
        self.current_row = 0;
        Ok(())
    }
}

/// 推断Excel文件模式
pub fn infer_excel_schema_impl(path: &Path) -> Result<DataSchema> {
    let processor = ExcelProcessor::new(path.to_string_lossy().to_string())?;
    processor.get_schema()
} 

/// Excel数据加载器
pub struct ExcelLoader {
    /// 文件路径
    file_path: PathBuf,
    /// 工作表名称
    sheet_name: Option<String>,
    /// 是否有标题行
    has_header: bool,
    /// 加载选项
    options: HashMap<String, String>,
}

// 取消特性门控的替代实现，统一使用主实现

impl ExcelLoader {
    /// 创建新的Excel加载器
    pub fn new(path: &str) -> Result<Self> {
        let file_path = PathBuf::from(path);
        
        // 验证文件是否存在
        if !file_path.exists() {
            return Err(Error::not_found(file_path.to_string_lossy().to_string()));
        }
        
        Ok(Self {
            file_path,
            sheet_name: None,
            has_header: true,
            options: HashMap::new(),
        })
    }
    
    /// 从文件加载数据
    pub fn load_from_file(&self, path: &str) -> Result<DataBatch> {
        let mut processor = ExcelProcessor::new(path.to_string())?;
        
        // 设置工作表名称（如果指定）
        if let Some(ref sheet_name) = self.sheet_name {
            processor = processor.with_sheet_name(sheet_name);
        }
        
        // 设置标题行选项
        processor = processor.with_has_header(self.has_header);
        
        // 推断模式
        let schema = processor.get_schema()?;
        
        // 获取行数
        let row_count = processor.get_row_count()?;
        
        // 读取所有行
        let records = processor.read_rows(row_count)?;
        
        // 将Record转换为HashMap格式
        let mut hashmap_records = Vec::new();
        let field_names: Vec<String> = schema.fields().iter().map(|f| f.name().to_string()).collect();
        
        for record in records {
            let mut hashmap_record = HashMap::new();
            for (i, value) in record.values().iter().enumerate() {
                if i < field_names.len() {
                    hashmap_record.insert(field_names[i].clone(), value.clone());
                }
            }
            hashmap_records.push(hashmap_record);
        }
        
        // 创建DataBatch
        let data_batch = DataBatch::from_records(hashmap_records, Some(schema))?;
        
        Ok(data_batch)
    }
    
    /// 设置工作表名称
    pub fn with_sheet_name(mut self, sheet_name: &str) -> Self {
        self.sheet_name = Some(sheet_name.to_string());
        self
    }
    
    /// 设置是否有标题行
    pub fn with_has_header(mut self, has_header: bool) -> Self {
        self.has_header = has_header;
        self
    }
    
    /// 设置选项
    pub fn with_option<K: Into<String>, V: Into<String>>(mut self, key: K, value: V) -> Self {
        self.options.insert(key.into(), value.into());
        self
    }
    
    /// 获取文件路径
    pub fn file_path(&self) -> &Path {
        &self.file_path
    }
} 