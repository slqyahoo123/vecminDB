/// 多模态模型模块
/// 
/// 本模块提供了各种多模态特征提取的模型实现，用于从不同类型的数据中
/// 提取特征向量，以支持后续的检索、分类等任务。

// 图像模型
pub mod image_models;

// 重新导出关键类型
pub use image_models::{
    // 特征接口
    ImageFeatureModel,
    
    // 模型实现
    ResNetFeatureModel,
    VGGFeatureModel,
    EfficientNetFeatureModel,
    CLIPImageFeatureModel,
    
    // 工具函数
    calculate_tensor_hash,
    normalize_l2,
};

// 未来会添加其他模态的模型模块
// pub mod text_models;
// pub mod audio_models;
// pub mod video_models; 