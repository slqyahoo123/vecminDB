use image::{DynamicImage, GenericImageView, ImageBuffer, Rgba, RgbaImage};
use imageproc::contrast::{equalize_histogram, stretch_contrast};
use imageproc::filter::{gaussian_blur, sharpen3x3, unsharpen};
use rayon::prelude::*;
use thiserror::Error;

use std::io::Cursor;

/// 图像增强错误类型
#[derive(Error, Debug)]
pub enum EnhanceError {
    /// 图像解码错误
    #[error("Failed to decode image: {0}")]
    DecodeError(#[from] image::error::ImageError),
    
    /// 参数无效错误
    #[error("Invalid parameter: {0}")]
    InvalidParameter(String),
    
    /// 图像处理错误
    #[error("Failed to process image: {0}")]
    ProcessingError(String),
    
    /// 图像编码错误
    #[error("Failed to encode image: {0}")]
    EncodeError(String),
}

/// 图像增强结果类型
pub type EnhanceResult<T> = Result<T, EnhanceError>;

/// 图像增强配置
#[derive(Debug, Clone)]
pub struct EnhanceConfig {
    /// 亮度调整值 (-100 到 100)
    pub brightness: Option<i32>,
    
    /// 对比度调整值 (-100 到 100)
    pub contrast: Option<i32>,
    
    /// 饱和度调整值 (-100 到 100)
    pub saturation: Option<i32>,
    
    /// 锐化强度 (0.0 到 10.0)
    pub sharpness: Option<f32>,
    
    /// 降噪强度 (0.0 到 10.0)
    pub denoise: Option<f32>,
    
    /// 是否应用直方图均衡化
    pub histogram_equalization: bool,
    
    /// 是否应用对比度拉伸
    pub contrast_stretch: bool,
}

impl Default for EnhanceConfig {
    fn default() -> Self {
        Self {
            brightness: None,
            contrast: None,
            saturation: None,
            sharpness: None,
            denoise: None,
            histogram_equalization: false,
            contrast_stretch: false,
        }
    }
}

impl EnhanceConfig {
    /// 创建新的增强配置
    pub fn new() -> Self {
        Self::default()
    }
    
    /// 设置亮度调整值
    pub fn with_brightness(mut self, value: i32) -> EnhanceResult<Self> {
        if value < -100 || value > 100 {
            return Err(EnhanceError::InvalidParameter(
                "Brightness must be between -100 and 100".to_string()
            ));
        }
        self.brightness = Some(value);
        Ok(self)
    }
    
    /// 设置对比度调整值
    pub fn with_contrast(mut self, value: i32) -> EnhanceResult<Self> {
        if value < -100 || value > 100 {
            return Err(EnhanceError::InvalidParameter(
                "Contrast must be between -100 and 100".to_string()
            ));
        }
        self.contrast = Some(value);
        Ok(self)
    }
    
    /// 设置饱和度调整值
    pub fn with_saturation(mut self, value: i32) -> EnhanceResult<Self> {
        if value < -100 || value > 100 {
            return Err(EnhanceError::InvalidParameter(
                "Saturation must be between -100 and 100".to_string()
            ));
        }
        self.saturation = Some(value);
        Ok(self)
    }
    
    /// 设置锐化强度
    pub fn with_sharpness(mut self, value: f32) -> EnhanceResult<Self> {
        if value < 0.0 || value > 10.0 {
            return Err(EnhanceError::InvalidParameter(
                "Sharpness must be between 0.0 and 10.0".to_string()
            ));
        }
        self.sharpness = Some(value);
        Ok(self)
    }
    
    /// 设置降噪强度
    pub fn with_denoise(mut self, value: f32) -> EnhanceResult<Self> {
        if value < 0.0 || value > 10.0 {
            return Err(EnhanceError::InvalidParameter(
                "Denoise must be between 0.0 and 10.0".to_string()
            ));
        }
        self.denoise = Some(value);
        Ok(self)
    }
    
    /// 启用直方图均衡化
    pub fn with_histogram_equalization(mut self, enable: bool) -> Self {
        self.histogram_equalization = enable;
        self
    }
    
    /// 启用对比度拉伸
    pub fn with_contrast_stretch(mut self, enable: bool) -> Self {
        self.contrast_stretch = enable;
        self
    }
    
    /// 验证配置是否有效
    pub fn validate(&self) -> EnhanceResult<()> {
        // 已经在各个with_*方法中验证了参数范围
        Ok(())
    }
}

/// 使用指定配置增强图像
///
/// # 参数
///
/// * `image_data` - 输入图像的二进制数据
/// * `config` - 图像增强配置
///
/// # 返回
///
/// 返回增强后的图像二进制数据
pub fn enhance_image(image_data: &[u8], config: &EnhanceConfig) -> EnhanceResult<Vec<u8>> {
    // 验证配置
    config.validate()?;
    
    // 解码图像
    let mut img = image::load_from_memory(image_data)
        .map_err(EnhanceError::DecodeError)?;
    
    // 应用各种增强效果
    img = apply_enhancements(img, config)?;
    
    // 编码回二进制数据
    let mut output = Vec::new();
    let mut cursor = Cursor::new(&mut output);
    img.write_to(&mut cursor, image::ImageOutputFormat::Png)
        .map_err(|e| EnhanceError::EncodeError(e.to_string()))?;
    
    Ok(output)
}

/// 应用增强效果到图像
fn apply_enhancements(mut img: DynamicImage, config: &EnhanceConfig) -> EnhanceResult<DynamicImage> {
    // 亮度调整
    if let Some(brightness) = config.brightness {
        img = adjust_brightness(img, brightness)?;
    }
    
    // 对比度调整
    if let Some(contrast) = config.contrast {
        img = adjust_contrast(img, contrast)?;
    }
    
    // 饱和度调整
    if let Some(saturation) = config.saturation {
        img = adjust_saturation(img, saturation)?;
    }
    
    // 锐化处理
    if let Some(sharpness) = config.sharpness {
        img = apply_sharpening(img, sharpness)?;
    }
    
    // 降噪处理
    if let Some(denoise) = config.denoise {
        img = apply_denoising(img, denoise)?;
    }
    
    // 直方图均衡化
    if config.histogram_equalization {
        img = apply_histogram_equalization(img)?;
    }
    
    // 对比度拉伸
    if config.contrast_stretch {
        img = apply_contrast_stretch(img)?;
    }
    
    Ok(img)
}

/// 调整图像亮度
fn adjust_brightness(img: DynamicImage, value: i32) -> EnhanceResult<DynamicImage> {
    let rgba_img = img.to_rgba8();
    let (width, height) = rgba_img.dimensions();
    
    let factor = 1.0 + (value as f32 / 100.0);
    
    let adjusted = ImageBuffer::from_fn(width, height, |x, y| {
        let pixel = rgba_img.get_pixel(x, y);
        
        let r = (pixel[0] as f32 * factor).min(255.0).max(0.0) as u8;
        let g = (pixel[1] as f32 * factor).min(255.0).max(0.0) as u8;
        let b = (pixel[2] as f32 * factor).min(255.0).max(0.0) as u8;
        
        Rgba([r, g, b, pixel[3]])
    });
    
    Ok(DynamicImage::ImageRgba8(adjusted))
}

/// 调整图像对比度
fn adjust_contrast(img: DynamicImage, value: i32) -> EnhanceResult<DynamicImage> {
    let rgba_img = img.to_rgba8();
    let (width, height) = rgba_img.dimensions();
    
    let factor = 1.0 + (value as f32 / 100.0);
    let avg_luminance = calculate_average_luminance(&rgba_img);
    
    let adjusted = ImageBuffer::from_fn(width, height, |x, y| {
        let pixel = rgba_img.get_pixel(x, y);
        
        let r = ((pixel[0] as f32 - avg_luminance) * factor + avg_luminance).min(255.0).max(0.0) as u8;
        let g = ((pixel[1] as f32 - avg_luminance) * factor + avg_luminance).min(255.0).max(0.0) as u8;
        let b = ((pixel[2] as f32 - avg_luminance) * factor + avg_luminance).min(255.0).max(0.0) as u8;
        
        Rgba([r, g, b, pixel[3]])
    });
    
    Ok(DynamicImage::ImageRgba8(adjusted))
}

/// 计算图像的平均亮度
fn calculate_average_luminance(img: &RgbaImage) -> f32 {
    let mut sum = 0.0;
    let mut count = 0;
    
    for pixel in img.pixels() {
        // 使用相对亮度公式: 0.2126*R + 0.7152*G + 0.0722*B
        let luminance = 0.2126 * pixel[0] as f32 + 0.7152 * pixel[1] as f32 + 0.0722 * pixel[2] as f32;
        sum += luminance;
        count += 1;
    }
    
    if count > 0 {
        sum / count as f32
    } else {
        128.0 // 默认中间亮度
    }
}

/// 调整图像饱和度
fn adjust_saturation(img: DynamicImage, value: i32) -> EnhanceResult<DynamicImage> {
    let rgba_img = img.to_rgba8();
    let (width, height) = rgba_img.dimensions();
    
    let factor = 1.0 + (value as f32 / 100.0);
    
    let adjusted = ImageBuffer::from_fn(width, height, |x, y| {
        let pixel = rgba_img.get_pixel(x, y);
        
        // 转换为HSV颜色空间，调整饱和度，然后转回RGB
        let (h, s, v) = rgb_to_hsv(pixel[0], pixel[1], pixel[2]);
        let new_s = (s * factor).min(1.0).max(0.0);
        let (r, g, b) = hsv_to_rgb(h, new_s, v);
        
        Rgba([r, g, b, pixel[3]])
    });
    
    Ok(DynamicImage::ImageRgba8(adjusted))
}

/// RGB转HSV色彩空间
fn rgb_to_hsv(r: u8, g: u8, b: u8) -> (f32, f32, f32) {
    let r = r as f32 / 255.0;
    let g = g as f32 / 255.0;
    let b = b as f32 / 255.0;
    
    let max = r.max(g).max(b);
    let min = r.min(g).min(b);
    let delta = max - min;
    
    // 色相计算
    let h = if delta < 0.00001 {
        0.0 // 无色相
    } else if max == r {
        ((g - b) / delta).rem_euclid(6.0) / 6.0
    } else if max == g {
        ((b - r) / delta + 2.0) / 6.0
    } else {
        ((r - g) / delta + 4.0) / 6.0
    };
    
    // 饱和度计算
    let s = if max < 0.00001 {
        0.0
    } else {
        delta / max
    };
    
    // 明度
    let v = max;
    
    (h, s, v)
}

/// HSV转RGB色彩空间
fn hsv_to_rgb(h: f32, s: f32, v: f32) -> (u8, u8, u8) {
    if s <= 0.0 {
        return ((v * 255.0) as u8, (v * 255.0) as u8, (v * 255.0) as u8);
    }
    
    let h = if h >= 1.0 { 0.0 } else { h * 6.0 };
    let i = h.floor();
    let f = h - i;
    
    let p = v * (1.0 - s);
    let q = v * (1.0 - s * f);
    let t = v * (1.0 - s * (1.0 - f));
    
    let (r, g, b) = match i as i32 {
        0 => (v, t, p),
        1 => (q, v, p),
        2 => (p, v, t),
        3 => (p, q, v),
        4 => (t, p, v),
        _ => (v, p, q),
    };
    
    ((r * 255.0) as u8, (g * 255.0) as u8, (b * 255.0) as u8)
}

/// 应用锐化处理
fn apply_sharpening(img: DynamicImage, amount: f32) -> EnhanceResult<DynamicImage> {
    let rgba_img = img.to_rgba8();
    
    if amount < 0.01 {
        return Ok(img);
    }
    
    // 根据锐化强度选择处理方法
    let result = if amount < 1.0 {
        // 轻度锐化
        sharpen3x3(&rgba_img)
    } else {
        // 强力锐化（使用unsharp masking）
        let sigma = 1.0 + (amount - 1.0) * 0.3; // 根据输入参数调整sigma
        let threshold = 5;
        unsharpen(&rgba_img, sigma, threshold)
    };
    
    Ok(DynamicImage::ImageRgba8(result))
}

/// 应用降噪处理
fn apply_denoising(img: DynamicImage, amount: f32) -> EnhanceResult<DynamicImage> {
    let rgba_img = img.to_rgba8();
    
    // 使用高斯模糊实现简单降噪
    // 调整sigma参数控制强度，数值越大模糊越强
    let sigma = 0.5 + (amount * 0.2);
    let result = gaussian_blur(&rgba_img, sigma);
    
    Ok(DynamicImage::ImageRgba8(result))
}

/// 应用直方图均衡化
fn apply_histogram_equalization(img: DynamicImage) -> EnhanceResult<DynamicImage> {
    let rgba_img = img.to_rgba8();
    let result = equalize_histogram(&rgba_img);
    Ok(DynamicImage::ImageRgba8(result))
}

/// 应用对比度拉伸
fn apply_contrast_stretch(img: DynamicImage) -> EnhanceResult<DynamicImage> {
    let rgba_img = img.to_rgba8();
    let result = stretch_contrast(&rgba_img, 0, 255);
    Ok(DynamicImage::ImageRgba8(result))
}

/// 批量处理多张图像
///
/// 使用并行处理提高性能
///
/// # 参数
///
/// * `images` - 图像数据列表
/// * `config` - 增强配置
///
/// # 返回
///
/// 返回处理后的图像列表和任何失败的处理结果
pub fn batch_enhance(
    images: Vec<Vec<u8>>, 
    config: &EnhanceConfig
) -> (Vec<Vec<u8>>, Vec<EnhanceError>) {
    let results: Vec<Result<Vec<u8>, EnhanceError>> = images
        .par_iter()
        .map(|image_data| enhance_image(image_data, config))
        .collect();
    
    let mut successful = Vec::new();
    let mut errors = Vec::new();
    
    for result in results {
        match result {
            Ok(enhanced) => successful.push(enhanced),
            Err(err) => errors.push(err),
        }
    }
    
    (successful, errors)
}

/// 自动增强图像
///
/// 应用一组预设的增强效果以改善图像质量
///
/// # 参数
///
/// * `image_data` - 输入图像数据
///
/// # 返回
///
/// 返回自动增强后的图像数据
pub fn auto_enhance(image_data: &[u8]) -> EnhanceResult<Vec<u8>> {
    // 创建一个平衡的自动增强配置
    let config = EnhanceConfig::new()
        .with_contrast(15)?
        .with_brightness(5)?
        .with_saturation(10)?
        .with_sharpness(1.2)?
        .with_contrast_stretch(true);
    
    enhance_image(image_data, &config)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs::File;
    use std::io::Read;
    use std::path::Path;
    
    // 辅助函数：从测试资源加载图像数据
    fn load_test_image() -> Vec<u8> {
        let test_image_path = Path::new("tests/resources/test_image.jpg");
        if test_image_path.exists() {
            let mut file = File::open(test_image_path).expect("Failed to open test image");
            let mut buffer = Vec::new();
            file.read_to_end(&mut buffer).expect("Failed to read test image");
            buffer
        } else {
            // 如果测试图像不存在，生成一个1x1像素的测试图像
            let img = RgbaImage::from_pixel(1, 1, Rgba([255, 0, 0, 255]));
            let mut buffer = Vec::new();
            let mut cursor = Cursor::new(&mut buffer);
            img.write_to(&mut cursor, image::ImageOutputFormat::Png)
                .expect("Failed to create test image");
            buffer
        }
    }
    
    #[test]
    fn test_enhance_config_validation() {
        // 测试有效配置
        let valid_config = EnhanceConfig::new()
            .with_brightness(50).unwrap()
            .with_contrast(30).unwrap()
            .with_saturation(20).unwrap()
            .with_sharpness(2.0).unwrap()
            .with_denoise(1.0).unwrap();
        
        assert!(valid_config.validate().is_ok());
        
        // 测试无效亮度
        let result = EnhanceConfig::new().with_brightness(101);
        assert!(result.is_err());
        
        // 测试无效对比度
        let result = EnhanceConfig::new().with_contrast(-101);
        assert!(result.is_err());
        
        // 测试无效锐化强度
        let result = EnhanceConfig::new().with_sharpness(-0.1);
        assert!(result.is_err());
    }
    
    #[test]
    fn test_enhance_image_basics() {
        let image_data = load_test_image();
        
        // 测试基本增强
        let config = EnhanceConfig::new()
            .with_brightness(10).unwrap()
            .with_contrast(10).unwrap();
        
        let result = enhance_image(&image_data, &config);
        assert!(result.is_ok());
        
        // 确保结果不为空且大小合理
        let enhanced = result.unwrap();
        assert!(!enhanced.is_empty());
        
        // 验证输出是有效的图像数据
        let result = image::load_from_memory(&enhanced);
        assert!(result.is_ok());
    }
    
    #[test]
    fn test_auto_enhance() {
        let image_data = load_test_image();
        let result = auto_enhance(&image_data);
        
        assert!(result.is_ok());
        let enhanced = result.unwrap();
        
        // 验证结果是有效的图像
        let img_result = image::load_from_memory(&enhanced);
        assert!(img_result.is_ok());
    }
    
    #[test]
    fn test_batch_processing() {
        let image_data = load_test_image();
        let images = vec![image_data.clone(), image_data.clone(), image_data];
        
        let config = EnhanceConfig::new()
            .with_contrast_stretch(true)
            .with_sharpness(1.0).unwrap();
        
        let (successful, errors) = batch_enhance(images, &config);
        
        assert_eq!(successful.len(), 3);
        assert!(errors.is_empty());
    }
    
    #[test]
    fn test_invalid_image_data() {
        // 测试无效的图像数据
        let invalid_data = vec![0, 1, 2, 3]; // 不是有效的图像格式
        
        let config = EnhanceConfig::default();
        let result = enhance_image(&invalid_data, &config);
        
        assert!(result.is_err());
        match result {
            Err(EnhanceError::DecodeError(_)) => {}, // 预期错误类型
            _ => panic!("Expected DecodeError but got different error"),
        }
    }
} 