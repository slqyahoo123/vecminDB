//! 扩展数据格式支持模块
//! 提供对多种数据格式的读写支持，包括Avro、Arrow、ORC、HDF5等

use std::path::Path;
use std::fs::File;
use std::io::{BufReader, Read};
use std::collections::HashMap;

 
use serde::{Serialize, Deserialize};
 

use crate::error::{Result, Error};
use crate::data::value::{DataValue, UnifiedValue, ScalarValue, UnifiedToData};
 

/// 扩展数据格式类型
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ExtendedDataFormat {
    /// Apache Avro格式
    Avro,
    /// Apache Arrow格式
    Arrow,
    /// Apache ORC格式
    Orc,
    /// HDF5格式
    Hdf5,
    /// Apache Iceberg格式
    Iceberg,
    /// Delta Lake格式
    Delta,
    /// XML格式
    Xml,
    /// YAML格式
    Yaml,
    /// TOML格式
    Toml,
    /// MessagePack格式
    MessagePack,
    /// Protocol Buffers格式
    Protobuf,
    /// Apache Thrift格式
    Thrift,
    /// BSON格式
    Bson,
    /// CBOR格式
    Cbor,
    /// Excel格式
    Excel,
    /// TSV格式（Tab分隔）
    Tsv,
    /// 固定宽度格式
    FixedWidth,
    /// 自定义格式
    Custom(String),
}

/// 格式检测器
pub struct FormatDetector {
    /// 支持的格式映射
    format_map: HashMap<String, ExtendedDataFormat>,
    /// 文件头部签名
    signatures: HashMap<Vec<u8>, ExtendedDataFormat>,
}

impl FormatDetector {
    /// 创建新的格式检测器
    pub fn new() -> Self {
        let mut detector = Self {
            format_map: HashMap::new(),
            signatures: HashMap::new(),
        };
        
        detector.initialize_format_map();
        detector.initialize_signatures();
        detector
    }

    /// 初始化格式映射
    fn initialize_format_map(&mut self) {
        // 文件扩展名到格式的映射
        let extensions = vec![
            ("avro", ExtendedDataFormat::Avro),
            ("arrow", ExtendedDataFormat::Arrow),
            ("orc", ExtendedDataFormat::Orc),
            ("h5", ExtendedDataFormat::Hdf5),
            ("hdf5", ExtendedDataFormat::Hdf5),
            ("iceberg", ExtendedDataFormat::Iceberg),
            ("delta", ExtendedDataFormat::Delta),
            ("xml", ExtendedDataFormat::Xml),
            ("yaml", ExtendedDataFormat::Yaml),
            ("yml", ExtendedDataFormat::Yaml),
            ("toml", ExtendedDataFormat::Toml),
            ("msgpack", ExtendedDataFormat::MessagePack),
            ("pb", ExtendedDataFormat::Protobuf),
            ("proto", ExtendedDataFormat::Protobuf),
            ("thrift", ExtendedDataFormat::Thrift),
            ("bson", ExtendedDataFormat::Bson),
            ("cbor", ExtendedDataFormat::Cbor),
            ("xlsx", ExtendedDataFormat::Excel),
            ("xls", ExtendedDataFormat::Excel),
            ("tsv", ExtendedDataFormat::Tsv),
        ];

        for (ext, format) in extensions {
            self.format_map.insert(ext.to_string(), format);
        }
    }

    /// 初始化文件签名
    fn initialize_signatures(&mut self) {
        // 常见文件格式的魔数签名
        self.signatures.insert(b"Obj".to_vec(), ExtendedDataFormat::Avro);
        self.signatures.insert(b"ARROW1".to_vec(), ExtendedDataFormat::Arrow);
        self.signatures.insert(b"ORC".to_vec(), ExtendedDataFormat::Orc);
        self.signatures.insert(b"\x89HDF\r\n\x1a\n".to_vec(), ExtendedDataFormat::Hdf5);
        self.signatures.insert(b"PK".to_vec(), ExtendedDataFormat::Excel); // ZIP-based
        self.signatures.insert(b"<?xml".to_vec(), ExtendedDataFormat::Xml);
        self.signatures.insert(b"---".to_vec(), ExtendedDataFormat::Yaml);
    }

    /// 检测文件格式
    pub fn detect_format<P: AsRef<Path>>(&self, path: P) -> Result<ExtendedDataFormat> {
        let path = path.as_ref();
        
        // 首先尝试通过文件扩展名检测
        if let Some(extension) = path.extension().and_then(|e| e.to_str()) {
            if let Some(format) = self.format_map.get(&extension.to_lowercase()) {
                return Ok(format.clone());
            }
        }

        // 然后尝试通过文件头部签名检测
        if let Ok(mut file) = File::open(path) {
            let mut buffer = vec![0u8; 16];
            if let Ok(bytes_read) = file.read(&mut buffer) {
                buffer.truncate(bytes_read);
                
                for (signature, format) in &self.signatures {
                    if buffer.starts_with(signature) {
                        return Ok(format.clone());
                    }
                }
            }
        }

        Err(Error::not_implemented(format!("无法检测文件格式: {:?}", path)))
    }
}

/// 数据格式读取器特征，设计为dyn安全
pub trait DataFormatReader {
    /// 读取数据，使用具体路径类型
    fn read_data(&self, path: &Path) -> Result<Vec<DataValue>>;
    
    /// 获取支持的格式
    fn supported_formats(&self) -> Vec<ExtendedDataFormat>;
    
    /// 检查是否支持格式
    fn supports_format(&self, format: &ExtendedDataFormat) -> bool {
        self.supported_formats().contains(format)
    }
}

/// 数据格式写入器特征，设计为dyn安全
pub trait DataFormatWriter {
    /// 写入数据，使用具体路径类型
    fn write_data(&self, data: &[DataValue], path: &Path) -> Result<()>;
    
    /// 获取支持的格式
    fn supported_formats(&self) -> Vec<ExtendedDataFormat>;
    
    /// 检查是否支持格式
    fn supports_format(&self, format: &ExtendedDataFormat) -> bool {
        self.supported_formats().contains(format)
    }
}

/// XML格式读取器
pub struct XmlReader;

impl DataFormatReader for XmlReader {
    fn read_data(&self, path: &Path) -> Result<Vec<DataValue>> {
        let content = std::fs::read_to_string(path)
            .map_err(|e| Error::IoError(format!("读取XML文件失败: {}", e)))?;
        
        // XML解析需要完整的XML解析器，当前返回特征未启用错误
        // 如需使用XML格式，请集成XML解析库（如serde_xml_rs）或使用其他数据格式
        Err(Error::feature_not_enabled(
            "XML格式支持需要完整的XML解析器实现，当前未集成XML解析库。请使用其他数据格式（如JSON、CSV）或集成XML解析库。".to_string()
        ))
    }

    fn supported_formats(&self) -> Vec<ExtendedDataFormat> {
        vec![ExtendedDataFormat::Xml]
    }
}

/// YAML格式读取器
pub struct YamlReader;

impl DataFormatReader for YamlReader {
    fn read_data(&self, path: &Path) -> Result<Vec<DataValue>> {
        let content = std::fs::read_to_string(path)
            .map_err(|e| Error::IoError(format!("读取YAML文件失败: {}", e)))?;
        
        // 使用serde_yaml解析
        let yaml_value: serde_yaml::Value = serde_yaml::from_str(&content)
            .map_err(|e| Error::invalid_input(format!("解析YAML失败: {}", e)))?;
        
        let records = self.yaml_to_data_values(&yaml_value)?;
        Ok(records)
    }

    fn supported_formats(&self) -> Vec<ExtendedDataFormat> {
        vec![ExtendedDataFormat::Yaml]
    }
}

impl YamlReader {
    /// 将YAML值转换为DataValue
    fn yaml_to_data_values(&self, yaml: &serde_yaml::Value) -> Result<Vec<DataValue>> {
        match yaml {
            serde_yaml::Value::Sequence(seq) => {
                let mut records = Vec::new();
                for item in seq {
                    records.push(self.yaml_value_to_data_value(item)?);
                }
                Ok(records)
            },
            _ => Ok(vec![self.yaml_value_to_data_value(yaml)?]),
        }
    }

    /// 将单个YAML值转换为DataValue
    fn yaml_value_to_data_value(&self, yaml: &serde_yaml::Value) -> Result<DataValue> {
        match yaml {
            serde_yaml::Value::Null => Ok(DataValue::Null),
            serde_yaml::Value::Bool(b) => Ok(DataValue::Boolean(*b)),
            serde_yaml::Value::Number(n) => {
                if let Some(i) = n.as_i64() {
                    Ok(DataValue::Integer(i))
                } else if let Some(f) = n.as_f64() {
                    Ok(DataValue::Number(f))
                } else {
                    Ok(DataValue::String(n.to_string()))
                }
            },
            serde_yaml::Value::String(s) => Ok(DataValue::String(s.clone())),
            serde_yaml::Value::Sequence(seq) => {
                let mut array = Vec::new();
                for item in seq {
                    let unified_value = self.yaml_unified_value(item)?;
                    let data_value = unified_value.unified_to_data()
                        .map_err(|e| Error::data(format!("转换UnifiedValue失败: {}", e)))?;
                    array.push(data_value);
                }
                Ok(DataValue::Array(array))
            },
            serde_yaml::Value::Mapping(map) => {
                let mut object = HashMap::new();
                for (key, value) in map {
                    if let serde_yaml::Value::String(key_str) = key {
                        let unified_value = self.yaml_unified_value(value)?;
                        let data_value = unified_value.unified_to_data()
                            .map_err(|e| Error::data(format!("转换UnifiedValue失败: {}", e)))?;
                        object.insert(key_str.clone(), data_value);
                    }
                }
                Ok(DataValue::Object(object))
            },
            _ => {
                // 将YAML值转换为字符串
                let yaml_str = serde_yaml::to_string(yaml)
                    .map_err(|e| Error::data(format!("序列化YAML失败: {}", e)))?;
                Ok(DataValue::String(yaml_str))
            },
        }
    }

    /// 将YAML值转换为UnifiedValue
    fn yaml_unified_value(&self, yaml: &serde_yaml::Value) -> Result<UnifiedValue> {
        match yaml {
            serde_yaml::Value::Null => Ok(UnifiedValue::Text(String::new())),
            serde_yaml::Value::Bool(b) => Ok(UnifiedValue::Scalar(ScalarValue::Bool(*b))),
            serde_yaml::Value::Number(n) => {
                if let Some(i) = n.as_i64() {
                    Ok(UnifiedValue::Integer(i))
                } else if let Some(f) = n.as_f64() {
                    Ok(UnifiedValue::Float(f))
                } else {
                    Ok(UnifiedValue::Text(n.to_string()))
                }
            },
            serde_yaml::Value::String(s) => Ok(UnifiedValue::Text(s.clone())),
            _ => {
                // 将YAML值转换为字符串
                let yaml_str = serde_yaml::to_string(yaml)
                    .map_err(|e| Error::data(format!("序列化YAML失败: {}", e)))?;
                Ok(UnifiedValue::Text(yaml_str))
            },
        }
    }
}

/// TOML格式读取器
pub struct TomlReader;

impl DataFormatReader for TomlReader {
    fn read_data(&self, path: &Path) -> Result<Vec<DataValue>> {
        let content = std::fs::read_to_string(path)
            .map_err(|e| Error::IoError(format!("读取TOML文件失败: {}", e)))?;
        
        // 使用toml解析
        let toml_value: toml::Value = toml::from_str(&content)
            .map_err(|e| Error::invalid_input(format!("解析TOML失败: {}", e)))?;
        
        let record = self.toml_to_data_value(&toml_value)?;
        Ok(vec![record])
    }

    fn supported_formats(&self) -> Vec<ExtendedDataFormat> {
        vec![ExtendedDataFormat::Toml]
    }
}

impl TomlReader {
    /// 将TOML值转换为DataValue
    fn toml_to_data_value(&self, toml: &toml::Value) -> Result<DataValue> {
        match toml {
            toml::Value::String(s) => Ok(DataValue::String(s.clone())),
            toml::Value::Integer(i) => Ok(DataValue::Integer(*i)),
            toml::Value::Float(f) => Ok(DataValue::Number(*f)),
            toml::Value::Boolean(b) => Ok(DataValue::Boolean(*b)),
            toml::Value::Array(arr) => {
                let mut array = Vec::new();
                for item in arr {
                    let unified_value = self.toml_to_unified_value(item)?;
                    let data_value = unified_value.unified_to_data()
                        .map_err(|e| Error::data(format!("转换UnifiedValue失败: {}", e)))?;
                    array.push(data_value);
                }
                Ok(DataValue::Array(array))
            },
            toml::Value::Table(table) => {
                let mut object = HashMap::new();
                for (key, value) in table {
                    let unified_value = self.toml_to_unified_value(value)?;
                    let data_value = unified_value.unified_to_data()
                        .map_err(|e| Error::data(format!("转换UnifiedValue失败: {}", e)))?;
                    object.insert(key.clone(), data_value);
                }
                Ok(DataValue::Object(object))
            },
            toml::Value::Datetime(dt) => Ok(DataValue::String(dt.to_string())),
        }
    }

    /// 将TOML值转换为UnifiedValue
    fn toml_to_unified_value(&self, toml: &toml::Value) -> Result<UnifiedValue> {
        match toml {
            toml::Value::String(s) => Ok(UnifiedValue::Text(s.clone())),
            toml::Value::Integer(i) => Ok(UnifiedValue::Integer(*i)),
            toml::Value::Float(f) => Ok(UnifiedValue::Float(*f)),
            toml::Value::Boolean(b) => Ok(UnifiedValue::Scalar(ScalarValue::Bool(*b))),
            toml::Value::Datetime(dt) => Ok(UnifiedValue::Text(dt.to_string())),
            _ => Ok(UnifiedValue::Text(toml.to_string())),
        }
    }
}

/// TSV格式读取器
pub struct TsvReader;

impl DataFormatReader for TsvReader {
    fn read_data(&self, path: &Path) -> Result<Vec<DataValue>> {
        let file = File::open(path)
            .map_err(|e| Error::IoError(format!("打开TSV文件失败: {}", e)))?;
        
        let reader = BufReader::new(file);
        let mut csv_reader = csv::ReaderBuilder::new()
            .delimiter(b'\t')
            .has_headers(true)
            .from_reader(reader);
        
        let headers = csv_reader.headers()
            .map_err(|e| Error::invalid_input(format!("读取TSV标题失败: {}", e)))?
            .clone();
        
        let mut records = Vec::new();
        
        for result in csv_reader.records() {
            let record = result
                .map_err(|e| Error::invalid_input(format!("读取TSV行失败: {}", e)))?;
            
            let mut data_map: HashMap<String, DataValue> = HashMap::new();
            
            for (i, field) in record.iter().enumerate() {
                if let Some(header) = headers.get(i) {
                    let unified_value = self.parse_tsv_field(field);
                    let data_value = unified_value.unified_to_data()
                        .map_err(|e| Error::data(format!("转换UnifiedValue失败: {}", e)))?;
                    data_map.insert(header.to_string(), data_value);
                }
            }
            
            records.push(DataValue::Object(data_map));
        }
        
        Ok(records)
    }

    fn supported_formats(&self) -> Vec<ExtendedDataFormat> {
        vec![ExtendedDataFormat::Tsv]
    }
}

impl TsvReader {
    /// 解析TSV字段
    fn parse_tsv_field(&self, field: &str) -> UnifiedValue {
        if field.is_empty() {
            return UnifiedValue::Text(String::new());
        }
        
        // 尝试解析为数字
        if let Ok(i) = field.parse::<i64>() {
            return UnifiedValue::Integer(i);
        }
        
        if let Ok(f) = field.parse::<f64>() {
            return UnifiedValue::Float(f);
        }
        
        // 检查布尔值
        match field.to_lowercase().as_str() {
            "true" | "yes" | "1" => UnifiedValue::Scalar(ScalarValue::Bool(true)),
            "false" | "no" | "0" => UnifiedValue::Scalar(ScalarValue::Bool(false)),
            _ => UnifiedValue::Text(field.to_string()),
        }
    }
}

/// 固定宽度格式读取器
pub struct FixedWidthReader {
    /// 字段定义
    field_definitions: Vec<FixedWidthField>,
}

/// 固定宽度字段定义
#[derive(Debug, Clone)]
pub struct FixedWidthField {
    /// 字段名称
    pub name: String,
    /// 开始位置（0索引）
    pub start: usize,
    /// 字段长度
    pub length: usize,
    /// 数据类型
    pub data_type: FixedWidthDataType,
}

/// 固定宽度数据类型
#[derive(Debug, Clone)]
pub enum FixedWidthDataType {
    /// 字符串
    String,
    /// 整数
    Integer,
    /// 浮点数
    Float,
    /// 日期
    Date,
}

impl FixedWidthReader {
    /// 创建新的固定宽度读取器
    pub fn new(field_definitions: Vec<FixedWidthField>) -> Self {
        Self { field_definitions }
    }
}

impl DataFormatReader for FixedWidthReader {
    fn read_data(&self, path: &Path) -> Result<Vec<DataValue>> {
        let content = std::fs::read_to_string(path)
            .map_err(|e| Error::IoError(format!("读取固定宽度文件失败: {}", e)))?;
        
        let mut records = Vec::new();
        
        for line in content.lines() {
            if line.trim().is_empty() {
                continue;
            }
            
            let mut data_map: HashMap<String, DataValue> = HashMap::new();
            
            for field_def in &self.field_definitions {
                let end_pos = field_def.start + field_def.length;
                if line.len() >= end_pos {
                    let field_content = &line[field_def.start..end_pos].trim();
                    let unified_value = self.parse_fixed_width_field(field_content, &field_def.data_type);
                    let data_value = unified_value.unified_to_data()
                        .map_err(|e| Error::data(format!("转换UnifiedValue失败: {}", e)))?;
                    data_map.insert(field_def.name.clone(), data_value);
                }
            }
            
            records.push(DataValue::Object(data_map));
        }
        
        Ok(records)
    }

    fn supported_formats(&self) -> Vec<ExtendedDataFormat> {
        vec![ExtendedDataFormat::FixedWidth]
    }
}

impl FixedWidthReader {
    /// 解析固定宽度字段
    fn parse_fixed_width_field(&self, field: &str, data_type: &FixedWidthDataType) -> UnifiedValue {
        if field.is_empty() {
            return UnifiedValue::Text(String::new());
        }
        
        match data_type {
            FixedWidthDataType::String => UnifiedValue::Text(field.to_string()),
            FixedWidthDataType::Integer => {
                field.parse::<i64>()
                    .map(UnifiedValue::Integer)
                    .unwrap_or_else(|_| UnifiedValue::Text(field.to_string()))
            },
            FixedWidthDataType::Float => {
                field.parse::<f64>()
                    .map(UnifiedValue::Float)
                    .unwrap_or_else(|_| UnifiedValue::Text(field.to_string()))
            },
            FixedWidthDataType::Date => UnifiedValue::Text(field.to_string()),
        }
    }
}

/// 统一数据格式管理器
pub struct DataFormatManager {
    /// 格式检测器
    detector: FormatDetector,
    /// 读取器映射
    readers: HashMap<ExtendedDataFormat, Box<dyn DataFormatReader + Send + Sync>>,
    /// 写入器映射
    writers: HashMap<ExtendedDataFormat, Box<dyn DataFormatWriter + Send + Sync>>,
}

impl DataFormatManager {
    /// 创建新的数据格式管理器
    pub fn new() -> Self {
        let mut manager = Self {
            detector: FormatDetector::new(),
            readers: HashMap::new(),
            writers: HashMap::new(),
        };
        
        manager.register_default_readers();
        manager
    }

    /// 注册默认读取器
    fn register_default_readers(&mut self) {
        self.readers.insert(ExtendedDataFormat::Xml, Box::new(XmlReader));
        self.readers.insert(ExtendedDataFormat::Yaml, Box::new(YamlReader));
        self.readers.insert(ExtendedDataFormat::Toml, Box::new(TomlReader));
        self.readers.insert(ExtendedDataFormat::Tsv, Box::new(TsvReader));
    }

    /// 注册读取器
    pub fn register_reader(
        &mut self,
        format: ExtendedDataFormat,
        reader: Box<dyn DataFormatReader + Send + Sync>,
    ) {
        self.readers.insert(format, reader);
    }

    /// 注册写入器
    pub fn register_writer(
        &mut self,
        format: ExtendedDataFormat,
        writer: Box<dyn DataFormatWriter + Send + Sync>,
    ) {
        self.writers.insert(format, writer);
    }

    /// 读取数据文件
    pub fn read_file<P: AsRef<Path>>(&self, path: P) -> Result<Vec<DataValue>> {
        let format = self.detector.detect_format(&path)?;
        
        if let Some(reader) = self.readers.get(&format) {
            reader.read_data(path.as_ref())
        } else {
            Err(Error::not_implemented(format!("不支持的读取格式: {:?}", format)))
        }
    }

    /// 写入数据文件
    pub fn write_file<P: AsRef<Path>>(
        &self,
        data: &[DataValue],
        path: P,
        format: ExtendedDataFormat,
    ) -> Result<()> {
        if let Some(writer) = self.writers.get(&format) {
            writer.write_data(data, path.as_ref())
        } else {
            Err(Error::not_implemented(format!("不支持的写入格式: {:?}", format)))
        }
    }

    /// 获取支持的读取格式
    pub fn supported_read_formats(&self) -> Vec<ExtendedDataFormat> {
        self.readers.keys().cloned().collect()
    }

    /// 获取支持的写入格式
    pub fn supported_write_formats(&self) -> Vec<ExtendedDataFormat> {
        self.writers.keys().cloned().collect()
    }
}

/// 创建默认的数据格式管理器
pub fn create_data_format_manager() -> DataFormatManager {
    DataFormatManager::new()
}

/// 创建固定宽度读取器
pub fn create_fixed_width_reader(field_definitions: Vec<FixedWidthField>) -> FixedWidthReader {
    FixedWidthReader::new(field_definitions)
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;
    use std::io::Write;

    #[test]
    fn test_format_detection() {
        let detector = FormatDetector::new();
        
        // 测试扩展名检测
        let format = detector.detect_format("test.yaml").unwrap();
        assert_eq!(format, ExtendedDataFormat::Yaml);
        
        let format = detector.detect_format("data.toml").unwrap();
        assert_eq!(format, ExtendedDataFormat::Toml);
    }

    #[test]
    fn test_yaml_reader() {
        let dir = tempdir().unwrap();
        let file_path = dir.path().join("test.yaml");
        
        let yaml_content = r#"
- name: Alice
  age: 25
  active: true
- name: Bob
  age: 30
  active: false
"#;
        
        std::fs::write(&file_path, yaml_content).unwrap();
        
        let reader = YamlReader;
        let data = reader.read_data(&file_path).unwrap();
        
        assert_eq!(data.len(), 2);
    }

    #[test]
    fn test_tsv_reader() {
        let dir = tempdir().unwrap();
        let file_path = dir.path().join("test.tsv");
        
        let mut file = File::create(&file_path).unwrap();
        writeln!(file, "name\tage\tscore").unwrap();
        writeln!(file, "Alice\t25\t95.5").unwrap();
        writeln!(file, "Bob\t30\t87.2").unwrap();
        
        let reader = TsvReader;
        let data = reader.read_data(&file_path).unwrap();
        
        assert_eq!(data.len(), 2);
    }

    #[test]
    fn test_fixed_width_reader() {
        let dir = tempdir().unwrap();
        let file_path = dir.path().join("test.txt");
        
        let content = "Alice    25  95.5\nBob      30  87.2\n";
        std::fs::write(&file_path, content).unwrap();
        
        let field_definitions = vec![
            FixedWidthField {
                name: "name".to_string(),
                start: 0,
                length: 9,
                data_type: FixedWidthDataType::String,
            },
            FixedWidthField {
                name: "age".to_string(),
                start: 9,
                length: 4,
                data_type: FixedWidthDataType::Integer,
            },
            FixedWidthField {
                name: "score".to_string(),
                start: 13,
                length: 4,
                data_type: FixedWidthDataType::Float,
            },
        ];
        
        let reader = FixedWidthReader::new(field_definitions);
        let data = reader.read_data(&file_path).unwrap();
        
        assert_eq!(data.len(), 2);
    }

    #[test]
    fn test_data_format_manager() {
        let manager = DataFormatManager::new();
        
        let formats = manager.supported_read_formats();
        assert!(!formats.is_empty());
        assert!(formats.contains(&ExtendedDataFormat::Yaml));
        assert!(formats.contains(&ExtendedDataFormat::Toml));
    }
} 