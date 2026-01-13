use crate::Result;
use async_trait::async_trait;
use std::sync::Arc;

/// 文本预处理器特征
/// 
/// 该特征定义了文本预处理的基本接口，包括文本预处理、获取预处理器名称等功能。
/// 实现此特征的类型可以作为文本处理管道中的预处理组件使用。
#[async_trait]
pub trait TextPreprocessor: Send + Sync {
    /// 对输入文本进行预处理
    /// 
    /// # 参数
    /// * `text` - 输入的文本
    /// 
    /// # 返回
    /// 预处理后的文本
    fn preprocess(&self, text: &str) -> Result<String>;
    
    /// 异步对输入文本进行预处理
    /// 
    /// # 参数
    /// * `text` - 输入的文本
    /// 
    /// # 返回
    /// 预处理后的文本
    async fn preprocess_async(&self, text: &str) -> Result<String> {
        // 默认实现调用同步方法
        self.preprocess(text)
    }
    
    /// 批量处理多个文本
    /// 
    /// # 参数
    /// * `texts` - 多个输入文本
    /// 
    /// # 返回
    /// 预处理后的文本列表
    fn batch_preprocess(&self, texts: &[&str]) -> Result<Vec<String>> {
        texts.iter().map(|&text| self.preprocess(text)).collect()
    }
    
    /// 异步批量处理多个文本
    /// 
    /// # 参数
    /// * `texts` - 多个输入文本
    /// 
    /// # 返回
    /// 预处理后的文本列表
    async fn batch_preprocess_async(&self, texts: &[&str]) -> Result<Vec<String>> {
        // 默认实现调用同步方法
        self.batch_preprocess(texts)
    }
    
    /// 获取预处理器的名称
    ///
    /// # 返回
    /// 预处理器名称
    fn name(&self) -> &str;
    
    /// 创建一个克隆，用于线程安全共享
    ///
    /// # 返回
    /// 预处理器的Arc包装实例
    fn into_arc(self) -> Arc<dyn TextPreprocessor>
    where
        Self: Sized + 'static,
    {
        Arc::new(self)
    }
} 