// 图像处理 - 调整大小模块
//
// 该模块提供各种图像大小调整算法的实现，用于高效处理图像缩放。
// 支持最近邻、双线性、双三次和Lanczos等多种插值算法。

use std::io::{Error, ErrorKind, Result, Cursor};
#[cfg(feature = "multimodal")]
use image::{DynamicImage, GenericImageView, imageops, ImageBuffer, Rgba, Pixel};
use rayon::prelude::*;

use crate::data::image_processing::ImageFormat;

/// 调整大小时使用的插值算法
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum InterpolationMode {
    /// 最近邻插值，速度最快但质量最低
    Nearest,
    /// 双线性插值，平衡速度和质量
    Bilinear,
    /// 双三次插值，质量较高
    Bicubic,
    /// Lanczos算法，质量最高但速度最慢
    Lanczos,
}

impl Default for InterpolationMode {
    fn default() -> Self {
        InterpolationMode::Bilinear
    }
}

impl InterpolationMode {
    /// 将字符串转换为插值模式
    pub fn from_str(s: &str) -> Self {
        match s.to_lowercase().as_str() {
            "nearest" | "nearest_neighbor" => Self::Nearest,
            "bilinear" => Self::Bilinear,
            "bicubic" => Self::Bicubic,
            "lanczos" => Self::Lanczos,
            _ => Self::default(),
        }
    }
    
    /// 获取插值模式的描述
    pub fn description(&self) -> &'static str {
        match self {
            Self::Nearest => "最近邻插值 (速度最快,质量最低)",
            Self::Bilinear => "双线性插值 (平衡速度和质量)",
            Self::Bicubic => "双三次插值 (质量较高)",
            Self::Lanczos => "Lanczos插值 (质量最高,速度最慢)",
        }
    }
    
    /// 转换为image库的FilterType
    #[cfg(feature = "multimodal")]
    fn to_filter_type(&self) -> imageops::FilterType {
        match self {
            Self::Nearest => imageops::FilterType::Nearest,
            Self::Bilinear => imageops::FilterType::Triangle,
            Self::Bicubic => imageops::FilterType::CatmullRom,
            Self::Lanczos => imageops::FilterType::Lanczos3,
        }
    }
}

/// 调整图像大小
///
/// # 参数
/// * `image_data` - 图像数据字节数组
/// * `format` - 图像格式
/// * `target_width` - 目标宽度
/// * `target_height` - 目标高度
/// * `interpolation` - 插值算法
///
/// # 返回
/// * `Result<Vec<u8>>` - 调整大小后的图像数据
///
/// # 错误
/// 当图像解码失败或大小调整失败时返回错误
#[cfg(feature = "multimodal")]
pub fn resize_image(
    image_data: &[u8], 
    format: ImageFormat,
    target_width: u32, 
    target_height: u32,
    interpolation: InterpolationMode
) -> Result<Vec<u8>> {
    if image_data.is_empty() {
        return Err(Error::new(ErrorKind::InvalidInput, "Empty image data"));
    }
    
    if target_width == 0 || target_height == 0 {
        return Err(Error::new(ErrorKind::InvalidInput, "Invalid dimensions: width and height must be greater than 0"));
    }
    
    // 解码图像
    let img = image::load_from_memory(image_data)
        .map_err(|e| Error::new(ErrorKind::InvalidData, format!("Failed to decode image: {}", e)))?;
    
    // 根据选择的插值算法调整大小
    let resized = imageops::resize(
        &img.to_rgba8(), 
        target_width, 
        target_height, 
        interpolation.to_filter_type()
    );
    
    // 编码为适当的格式
    let mut output = Vec::new();
    let mut cursor = Cursor::new(&mut output);
    match format {
        ImageFormat::PNG => {
            let dynamic = DynamicImage::ImageRgba8(resized);
            dynamic.write_to(&mut cursor, image::ImageOutputFormat::Png)
                .map_err(|e| Error::new(ErrorKind::Other, format!("Failed to encode PNG: {}", e)))?;
        },
        ImageFormat::JPEG => {
            let dynamic = DynamicImage::ImageRgba8(resized);
            dynamic.write_to(&mut cursor, image::ImageOutputFormat::Jpeg(90))
                .map_err(|e| Error::new(ErrorKind::Other, format!("Failed to encode JPEG: {}", e)))?;
        },
        ImageFormat::WEBP => {
            let dynamic = DynamicImage::ImageRgba8(resized);
            dynamic.write_to(&mut cursor, image::ImageOutputFormat::WebP)
                .map_err(|e| Error::new(ErrorKind::Other, format!("Failed to encode WebP: {}", e)))?;
        },
        _ => {
            // 默认使用PNG格式
            let dynamic = DynamicImage::ImageRgba8(resized);
            dynamic.write_to(&mut cursor, image::ImageOutputFormat::Png)
                .map_err(|e| Error::new(ErrorKind::Other, format!("Failed to encode image: {}", e)))?;
        }
    }
    
    Ok(output)
}

#[cfg(not(feature = "multimodal"))]
pub fn resize_image(
    _image_data: &[u8], 
    _format: ImageFormat,
    _target_width: u32, 
    _target_height: u32,
    _interpolation: InterpolationMode
) -> Result<Vec<u8>> {
    Err(Error::new(ErrorKind::Unsupported, "Image processing requires 'multimodal' feature"))
}

/// 调整图像大小配置
#[derive(Debug, Clone)]
pub struct ResizeConfig {
    /// 目标宽度
    pub width: Option<u32>,
    /// 目标高度
    pub height: Option<u32>,
    /// 保持宽高比
    pub maintain_aspect_ratio: bool,
    /// 插值算法
    pub interpolation: InterpolationMode,
    /// 允许放大
    pub allow_upscaling: bool,
    /// 将目标尺寸作为最大值
    pub fit_within: bool,
    /// 目标背景颜色 (RGBA)
    pub background_color: Option<[u8; 4]>,
}

impl Default for ResizeConfig {
    fn default() -> Self {
        Self {
            width: None,
            height: None,
            maintain_aspect_ratio: true,
            interpolation: InterpolationMode::default(),
            allow_upscaling: false,
            fit_within: false,
            background_color: None,
        }
    }
}

impl ResizeConfig {
    /// 创建新的调整大小配置
    pub fn new() -> Self {
        Self::default()
    }
    
    /// 设置目标宽度
    pub fn with_width(mut self, width: u32) -> Self {
        self.width = Some(width);
        self
    }
    
    /// 设置目标高度
    pub fn with_height(mut self, height: u32) -> Self {
        self.height = Some(height);
        self
    }
    
    /// 设置目标尺寸
    pub fn with_dimensions(mut self, width: u32, height: u32) -> Self {
        self.width = Some(width);
        self.height = Some(height);
        self
    }
    
    /// 设置是否保持宽高比
    pub fn maintain_aspect_ratio(mut self, maintain: bool) -> Self {
        self.maintain_aspect_ratio = maintain;
        self
    }
    
    /// 设置插值算法
    pub fn with_interpolation(mut self, interpolation: InterpolationMode) -> Self {
        self.interpolation = interpolation;
        self
    }
    
    /// 设置是否允许放大
    pub fn allow_upscaling(mut self, allow: bool) -> Self {
        self.allow_upscaling = allow;
        self
    }
    
    /// 设置是否将目标尺寸作为最大值
    pub fn fit_within(mut self, fit: bool) -> Self {
        self.fit_within = fit;
        self
    }
    
    /// 设置背景颜色
    pub fn with_background_color(mut self, color: [u8; 4]) -> Self {
        self.background_color = Some(color);
        self
    }
    
    /// 验证配置是否有效
    pub fn validate(&self) -> Result<()> {
        if self.width.is_none() && self.height.is_none() {
            return Err(Error::new(ErrorKind::InvalidInput, "Either width or height must be specified"));
        }
        
        if let Some(width) = self.width {
            if width == 0 {
                return Err(Error::new(ErrorKind::InvalidInput, "Width must be greater than 0"));
            }
        }
        
        if let Some(height) = self.height {
            if height == 0 {
                return Err(Error::new(ErrorKind::InvalidInput, "Height must be greater than 0"));
            }
        }
        
        Ok(())
    }
    
    /// 根据原始尺寸计算调整后的尺寸
    pub fn calculate_dimensions(&self, original_width: u32, original_height: u32) -> Result<(u32, u32)> {
        if original_width == 0 || original_height == 0 {
            return Err(Error::new(ErrorKind::InvalidInput, "Original dimensions must be greater than 0"));
        }
        
        // 如果没有指定宽度或高度，返回原始尺寸
        if self.width.is_none() && self.height.is_none() {
            return Ok((original_width, original_height));
        }
        
        // 如果只指定了宽度
        if let Some(target_width) = self.width {
            if self.height.is_none() && self.maintain_aspect_ratio {
                let ratio = target_width as f64 / original_width as f64;
                let target_height = (original_height as f64 * ratio).round() as u32;
                return Ok((target_width, target_height));
            }
        }
        
        // 如果只指定了高度
        if let Some(target_height) = self.height {
            if self.width.is_none() && self.maintain_aspect_ratio {
                let ratio = target_height as f64 / original_height as f64;
                let target_width = (original_width as f64 * ratio).round() as u32;
                return Ok((target_width, target_height));
            }
        }
        
        // 如果同时指定了宽度和高度
        let target_width = self.width.unwrap_or(original_width);
        let target_height = self.height.unwrap_or(original_height);
        
        // 如果不需要保持宽高比，直接返回目标尺寸
        if !self.maintain_aspect_ratio {
            return Ok((target_width, target_height));
        }
        
        // 计算宽高比
        let width_ratio = target_width as f64 / original_width as f64;
        let height_ratio = target_height as f64 / original_height as f64;
        
        let ratio = if self.fit_within {
            // 取较小的比例，确保图像完全适应目标尺寸
            width_ratio.min(height_ratio)
        } else {
            // 取较大的比例，确保填充目标尺寸
            width_ratio.max(height_ratio)
        };
        
        // 如果不允许放大且比例大于1，则使用原始尺寸
        let ratio = if !self.allow_upscaling && ratio > 1.0 {
            1.0
        } else {
            ratio
        };
        
        let new_width = (original_width as f64 * ratio).round() as u32;
        let new_height = (original_height as f64 * ratio).round() as u32;
        
        Ok((new_width, new_height))
    }
}

/// 预设图像尺寸
#[derive(Debug, Clone, PartialEq)]
pub enum ResizeMode {
    /// 精确尺寸，无论原始宽高比
    Exact,
    /// 按比例调整，保持宽高比
    AspectRatio,
    /// 固定宽度，按比例调整高度
    FixedWidth,
    /// 固定高度，按比例调整宽度
    FixedHeight,
    /// 填充模式，确保覆盖整个目标区域
    Fill,
    /// 适应模式，确保图像完全适应目标区域
    Fit,
}

impl Default for ResizeMode {
    fn default() -> Self {
        ResizeMode::AspectRatio
    }
}

/// 使用高级配置调整图像大小
///
/// # 参数
/// * `image_data` - 图像数据字节数组
/// * `config` - 调整大小配置
///
/// # 返回
/// * `Result<Vec<u8>>` - 调整大小后的图像数据
///
/// # 错误
/// 当图像解码失败或大小调整失败时返回错误
#[cfg(feature = "multimodal")]
pub fn resize_image_with_config(image_data: &[u8], config: &ResizeConfig) -> Result<Vec<u8>> {
    // 验证配置
    config.validate()?;
    
    if image_data.is_empty() {
        return Err(Error::new(ErrorKind::InvalidInput, "Empty image data"));
    }
    
    // 解码图像
    let img = image::load_from_memory(image_data)
        .map_err(|e| Error::new(ErrorKind::InvalidData, format!("Failed to decode image: {}", e)))?;
    
    let (original_width, original_height) = img.dimensions();
    
    // 计算新尺寸
    let (new_width, new_height) = config.calculate_dimensions(original_width, original_height)?;
    
    // 调整大小
    let resized = imageops::resize(
        &img.to_rgba8(), 
        new_width, 
        new_height, 
        config.interpolation.to_filter_type()
    );
    
    // 如果需要填充背景（当维持宽高比且目标尺寸与计算尺寸不同时）
    if config.maintain_aspect_ratio && 
       (config.width.is_some() && config.width.unwrap() != new_width || 
        config.height.is_some() && config.height.unwrap() != new_height) {
        
        if let (Some(target_width), Some(target_height)) = (config.width, config.height) {
            let background_color = config.background_color.unwrap_or([0, 0, 0, 0]);
            
            // 创建一个背景图像
            let mut background = ImageBuffer::<Rgba<u8>, Vec<u8>>::new(target_width, target_height);
            
            // 填充背景色
            for pixel in background.pixels_mut() {
                *pixel = Rgba(background_color);
            }
            
            // 计算图像居中位置
            let x_offset = ((target_width as i32 - new_width as i32) / 2).max(0) as u32;
            let y_offset = ((target_height as i32 - new_height as i32) / 2).max(0) as u32;
            
            // 将调整后的图像复制到背景中
            let mut background_dynamic = DynamicImage::ImageRgba8(background);
            let resized_dynamic = DynamicImage::ImageRgba8(resized);
            imageops::overlay(&mut background_dynamic, &resized_dynamic, x_offset as i64, y_offset as i64);
            
            // 编码结果
            let mut output = Vec::new();
            let mut cursor = Cursor::new(&mut output);
            background_dynamic.write_to(&mut cursor, image::ImageOutputFormat::Png)
                .map_err(|e| Error::new(ErrorKind::Other, format!("Failed to encode image: {}", e)))?;
                
            return Ok(output);
        }
    }
    
    // 编码结果
    let dynamic = DynamicImage::ImageRgba8(resized);
    let mut output = Vec::new();
    let mut cursor = Cursor::new(&mut output);
    dynamic.write_to(&mut cursor, image::ImageOutputFormat::Png)
        .map_err(|e| Error::new(ErrorKind::Other, format!("Failed to encode image: {}", e)))?;
        
    Ok(output)
}

#[cfg(not(feature = "multimodal"))]
pub fn resize_image_with_config(_image_data: &[u8], _config: &ResizeConfig) -> Result<Vec<u8>> {
    Err(Error::new(ErrorKind::Unsupported, "Image processing requires 'multimodal' feature"))
}

/// 创建缩略图
///
/// 生成图像的缩略图，使用Fit模式和双线性插值，不允许放大。
///
/// # 参数
/// * `image_data` - 图像数据字节数组
/// * `max_width` - 最大宽度
/// * `max_height` - 最大高度
///
/// # 返回
/// * `Result<Vec<u8>>` - 缩略图数据
#[cfg(feature = "multimodal")]
pub fn create_thumbnail(image_data: &[u8], max_width: u32, max_height: u32) -> Result<Vec<u8>> {
    let config = ResizeConfig::new()
        .with_dimensions(max_width, max_height)
        .maintain_aspect_ratio(true)
        .fit_within(true)
        .allow_upscaling(false)
        .with_interpolation(InterpolationMode::Bilinear);
        
    resize_image_with_config(image_data, &config)
}

#[cfg(not(feature = "multimodal"))]
pub fn create_thumbnail(_image_data: &[u8], _max_width: u32, _max_height: u32) -> Result<Vec<u8>> {
    Err(Error::new(ErrorKind::Unsupported, "Image processing requires 'multimodal' feature"))
}

/// 批量调整图像大小
///
/// 使用多线程并行处理多个图像。
///
/// # 参数
/// * `images` - 图像数据和配置列表
///
/// # 返回
/// * `Result<Vec<Vec<u8>>>` - 调整大小后的图像数据列表
#[cfg(feature = "multimodal")]
pub fn batch_resize(images: Vec<(Vec<u8>, ResizeConfig)>) -> Result<Vec<Vec<u8>>> {
    // 使用Rayon进行并行处理
    images.into_par_iter()
        .map(|(data, config)| resize_image_with_config(&data, &config))
        .collect()
}

#[cfg(not(feature = "multimodal"))]
pub fn batch_resize(_images: Vec<(Vec<u8>, ResizeConfig)>) -> Result<Vec<Vec<u8>>> {
    Err(Error::new(ErrorKind::Unsupported, "Image processing requires 'multimodal' feature"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::path::Path;
    
    // 测试辅助函数，从文件加载测试图像
    fn load_test_image() -> Vec<u8> {
        // 注意：这个测试需要在测试资源目录中提供一个测试图像
        // 在实际项目中，请替换为有效的测试图像路径
        let test_image_path = "tests/resources/test_image.png";
        
        if Path::new(test_image_path).exists() {
            fs::read(test_image_path).unwrap_or_else(|_| {
                // 如果读取失败，创建一个1x1的黑色图像作为后备
                let img = ImageBuffer::<Rgba<u8>, Vec<u8>>::new(1, 1);
                let mut buffer = Vec::new();
                DynamicImage::ImageRgba8(img)
                    .write_to(&mut buffer, image::ImageOutputFormat::Png)
                    .expect("Failed to create test image");
                buffer
            })
        } else {
            // 创建一个1x1的黑色图像作为测试数据
            let img = ImageBuffer::<Rgba<u8>, Vec<u8>>::new(1, 1);
            let mut buffer = Vec::new();
            DynamicImage::ImageRgba8(img)
                .write_to(&mut buffer, image::ImageOutputFormat::Png)
                .expect("Failed to create test image");
            buffer
        }
    }
    
    #[test]
    fn test_interpolation_mode_from_str() {
        assert_eq!(InterpolationMode::from_str("nearest"), InterpolationMode::Nearest);
        assert_eq!(InterpolationMode::from_str("bilinear"), InterpolationMode::Bilinear);
        assert_eq!(InterpolationMode::from_str("bicubic"), InterpolationMode::Bicubic);
        assert_eq!(InterpolationMode::from_str("lanczos"), InterpolationMode::Lanczos);
        assert_eq!(InterpolationMode::from_str("unknown"), InterpolationMode::Bilinear); // 默认值
    }
    
    #[test]
    fn test_resize_config_calculate_dimensions() {
        // 测试不变大小
        let config = ResizeConfig::default();
        assert_eq!(config.calculate_dimensions(100, 100).unwrap(), (100, 100));
        
        // 测试指定宽度，保持比例
        let config = ResizeConfig::new().with_width(200).maintain_aspect_ratio(true);
        assert_eq!(config.calculate_dimensions(100, 100).unwrap(), (200, 200));
        
        // 测试指定高度，保持比例
        let config = ResizeConfig::new().with_height(200).maintain_aspect_ratio(true);
        assert_eq!(config.calculate_dimensions(100, 100).unwrap(), (200, 200));
        
        // 测试同时指定宽度和高度，不保持比例
        let config = ResizeConfig::new().with_dimensions(200, 300).maintain_aspect_ratio(false);
        assert_eq!(config.calculate_dimensions(100, 100).unwrap(), (200, 300));
        
        // 测试同时指定宽度和高度，保持比例，适应模式
        let config = ResizeConfig::new().with_dimensions(200, 400).maintain_aspect_ratio(true).fit_within(true);
        assert_eq!(config.calculate_dimensions(100, 100).unwrap(), (200, 200));
        
        // 测试同时指定宽度和高度，保持比例，填充模式
        let config = ResizeConfig::new().with_dimensions(200, 400).maintain_aspect_ratio(true).fit_within(false);
        assert_eq!(config.calculate_dimensions(100, 100).unwrap(), (400, 400));
        
        // 测试不允许放大
        let config = ResizeConfig::new().with_width(200).maintain_aspect_ratio(true).allow_upscaling(false);
        assert_eq!(config.calculate_dimensions(100, 100).unwrap(), (100, 100));
    }
    
    #[test]
    fn test_resize_image_with_config() {
        let test_image = load_test_image();
        
        // 测试基本调整大小
        let config = ResizeConfig::new().with_dimensions(50, 50);
        let result = resize_image_with_config(&test_image, &config);
        assert!(result.is_ok());
        
        // 测试保持宽高比
        let config = ResizeConfig::new().with_width(50).maintain_aspect_ratio(true);
        let result = resize_image_with_config(&test_image, &config);
        assert!(result.is_ok());
        
        // 测试创建缩略图功能
        let result = create_thumbnail(&test_image, 50, 50);
        assert!(result.is_ok());
    }
    
    #[test]
    fn test_invalid_inputs() {
        // 测试空数据
        let result = resize_image(&[], ImageFormat::PNG, 100, 100, InterpolationMode::Bilinear);
        assert!(result.is_err());
        
        // 测试无效尺寸
        let test_image = load_test_image();
        let result = resize_image(&test_image, ImageFormat::PNG, 0, 100, InterpolationMode::Bilinear);
        assert!(result.is_err());
        
        // 测试无效配置
        let config = ResizeConfig::new(); // 没有设置宽度或高度
        let result = resize_image_with_config(&test_image, &config);
        assert!(result.is_err());
    }
} 