// 文本规范化器桥接模块
// Text Normalizers Bridge Module
// 
// 重新导出 preprocessing/normalizer.rs 的功能以保持向后兼容性
// Re-exports functionality from preprocessing/normalizer.rs for backward compatibility
//
// 这个模块提供了各种文本规范化工具，包括大小写转换、重音移除、词干提取、词形还原等
// This module provides various text normalization tools, including case conversion,
// accent removal, stemming, lemmatization, etc.

// 从原始模块重新导出所有功能
pub use crate::data::text_features::preprocessing::normalizer::*;

// 从preprocessing模块中导入TextProcessor trait
use crate::data::text_features::preprocessing::TextProcessor;
use unicode_normalization::UnicodeNormalization;

/// 文本大小写规范化器 - 将文本转换为大写或小写
/// Text Case Normalizer - Converts text to uppercase or lowercase
pub struct TextCaseNormalizer {
    to_upper: bool,
    name: Option<String>,
}

impl TextCaseNormalizer {
    /// 创建新的大小写规范化器
    /// 
    /// # 参数
    /// * `to_upper` - 是否转换为大写，false则转换为小写
    /// * `name` - 处理器名称
    pub fn new(to_upper: bool, name: Option<String>) -> Self {
        Self {
            to_upper,
            name,
        }
    }
}

impl TextProcessor for TextCaseNormalizer {
    fn process(&self, text: &str) -> Result<String, crate::Error> {
        if self.to_upper {
            Ok(text.to_uppercase())
        } else {
            Ok(text.to_lowercase())
        }
    }

    fn name(&self) -> &str {
        match &self.name {
            Some(name) => name,
            None => if self.to_upper { "UppercaseNormalizer" } else { "LowercaseNormalizer" },
        }
    }
    
    fn processor_type(&self) -> crate::data::text_features::pipeline::ProcessingStageType {
        crate::data::text_features::pipeline::ProcessingStageType::Normalization
    }
    
    fn box_clone(&self) -> Box<dyn TextProcessor> {
        Box::new(Self {
            to_upper: self.to_upper,
            name: self.name.clone(),
        })
    }
}

// 确保TextCaseNormalizer也实现TextNormalizer trait，以便兼容性
impl TextNormalizer for TextCaseNormalizer {
    fn normalize(&self, text: &str) -> crate::Result<String> {
        self.process(text).map_err(|e| crate::Error::data(format!("文本规范化处理失败: {}", e)))
    }
    
    fn name(&self) -> &str {
        TextProcessor::name(self)
    }
}

/// 重音标记规范化器 - 移除文本中的重音标记
pub struct AccentNormalizer {
    name: Option<String>,
}

impl AccentNormalizer {
    /// 创建新的重音标记规范化器
    pub fn new(name: Option<String>) -> Self {
        Self { name }
    }
}

impl TextProcessor for AccentNormalizer {
    fn process(&self, text: &str) -> Result<String, crate::Error> {
        // 首先将文本转换成NFD形式，将重音符号与基本字符分离
        let nfd = text.nfd().collect::<String>();
        
        // 过滤掉所有非ASCII字符（包括重音符号）
        let result: String = nfd
            .chars()
            .filter(|&c| c.is_ascii())
            .collect();
            
        Ok(result)
    }

    fn name(&self) -> &str {
        match &self.name {
            Some(name) => name,
            None => "AccentNormalizer",
        }
    }
    
    fn processor_type(&self) -> crate::data::text_features::pipeline::ProcessingStageType {
        crate::data::text_features::pipeline::ProcessingStageType::Normalization
    }
    
    fn box_clone(&self) -> Box<dyn TextProcessor> {
        Box::new(Self {
            name: self.name.clone(),
        })
    }
}

// 确保AccentNormalizer也实现TextNormalizer trait，以便兼容性
impl TextNormalizer for AccentNormalizer {
    fn normalize(&self, text: &str) -> crate::Result<String> {
        self.process(text).map_err(|e| crate::Error::data(format!("文本规范化处理失败: {}", e)))
    }
    
    fn name(&self) -> &str {
        TextProcessor::name(self)
    }
}

/// 创建标准文本规范化器
/// 
/// 这是一个便捷函数，用于创建具有默认配置的StandardTextNormalizer
pub fn create_standard_normalizer() -> StandardTextNormalizer {
    StandardTextNormalizer::new()
}

/// 创建自定义文本规范化器
/// 
/// 根据指定的配置创建StandardTextNormalizer
pub fn create_custom_normalizer(config: NormalizerConfig) -> StandardTextNormalizer {
    StandardTextNormalizer::with_config(config)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_case_normalizer() {
        let lowercase = TextCaseNormalizer::new(false, None);
        assert_eq!(lowercase.process("Hello World").unwrap(), "hello world");
        
        let uppercase = TextCaseNormalizer::new(true, None);
        assert_eq!(uppercase.process("Hello World").unwrap(), "HELLO WORLD");
    }

    #[test]
    fn test_accent_normalizer() {
        let normalizer = AccentNormalizer::new(None);
        assert_eq!(normalizer.process("résumé").unwrap(), "resume");
        assert_eq!(normalizer.process("naïve").unwrap(), "naive");
    }
    
    #[test]
    fn test_normalizer_compatibility() {
        // 测试桥接模块中的规范化器是否同时兼容TextNormalizer和TextProcessor接口
        let case_normalizer = TextCaseNormalizer::new(false, None);
        let accent_normalizer = AccentNormalizer::new(None);
        
        // 使用TextNormalizer接口
        assert_eq!(case_normalizer.normalize("Hello World").unwrap(), "hello world");
        assert_eq!(accent_normalizer.normalize("résumé").unwrap(), "resume");
        
        // 使用TextProcessor接口
        assert_eq!(case_normalizer.process("Hello World").unwrap(), "hello world");
        assert_eq!(accent_normalizer.process("résumé").unwrap(), "resume");
    }
} 