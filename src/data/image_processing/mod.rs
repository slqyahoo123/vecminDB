// 图像处理模块
//
// 该模块提供了图像处理的各种功能，包括图像缩放、增强、转换等

pub mod resize;
pub mod augmentation;
pub mod color;

use std::io::{Result, Error, ErrorKind};
#[cfg(feature = "multimodal")]
use std::io::Cursor;
use std::path::Path;

// 图像格式枚举
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ImageFormat {
    PNG,
    JPEG,
    WEBP,
    TIFF,
    BMP,
    GIF,
    Grayscale,
    RGB,
    RGBA,
    Unknown,
}

impl ImageFormat {
    // 从文件扩展名推断图像格式
    pub fn from_extension(extension: &str) -> Self {
        match extension.to_lowercase().as_str() {
            "png" => ImageFormat::PNG,
            "jpg" | "jpeg" => ImageFormat::JPEG,
            "webp" => ImageFormat::WEBP,
            "tiff" | "tif" => ImageFormat::TIFF,
            "bmp" => ImageFormat::BMP,
            "gif" => ImageFormat::GIF,
            _ => ImageFormat::Unknown,
        }
    }
    
    // 从文件路径推断图像格式
    pub fn from_path<P: AsRef<Path>>(path: P) -> Self {
        path.as_ref()
            .extension()
            .and_then(|e| e.to_str())
            .map(ImageFormat::from_extension)
            .unwrap_or(ImageFormat::Unknown)
    }
    
    // 转换为MIME类型字符串
    pub fn to_mime_type(&self) -> &'static str {
        match self {
            ImageFormat::PNG => "image/png",
            ImageFormat::JPEG => "image/jpeg",
            ImageFormat::WEBP => "image/webp",
            ImageFormat::TIFF => "image/tiff",
            ImageFormat::BMP => "image/bmp",
            ImageFormat::GIF => "image/gif",
            ImageFormat::Grayscale | ImageFormat::RGB | ImageFormat::RGBA => "application/octet-stream",
            ImageFormat::Unknown => "application/octet-stream",
        }
    }
}

// 图像尺寸结构
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ImageDimensions {
    pub width: u32,
    pub height: u32,
}

impl ImageDimensions {
    pub fn new(width: u32, height: u32) -> Self {
        Self { width, height }
    }
    
    pub fn aspect_ratio(&self) -> f32 {
        if self.height == 0 {
            return 0.0;
        }
        self.width as f32 / self.height as f32
    }
    
    pub fn is_empty(&self) -> bool {
        self.width == 0 || self.height == 0
    }
    
    pub fn pixel_count(&self) -> u32 {
        self.width * self.height
    }
}

// 图像元数据结构
#[derive(Debug, Clone)]
pub struct ImageMetadata {
    pub dimensions: ImageDimensions,
    pub format: ImageFormat,
    pub color_depth: u8,
    pub has_alpha: bool,
    pub dpi: Option<(u32, u32)>,
}

// 图像插值算法枚举
/// 图像处理中使用的插值模式
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
    /// 从字符串创建插值模式
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
}

// 从图像数据中检查图像格式
pub fn detect_image_format(image_data: &[u8]) -> ImageFormat {
    if image_data.len() < 8 {
        return ImageFormat::Unknown;
    }

    // 检查PNG签名
    if image_data.starts_with(&[0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A]) {
        return ImageFormat::PNG;
    }
    
    // 检查JPEG签名
    if image_data.starts_with(&[0xFF, 0xD8, 0xFF]) {
        return ImageFormat::JPEG;
    }
    
    // 检查GIF签名
    if image_data.starts_with(b"GIF87a") || image_data.starts_with(b"GIF89a") {
        return ImageFormat::GIF;
    }
    
    // 检查BMP签名
    if image_data.starts_with(b"BM") {
        return ImageFormat::BMP;
    }
    
    // 检查WEBP签名
    if image_data.len() >= 12 && &image_data[0..4] == b"RIFF" && &image_data[8..12] == b"WEBP" {
        return ImageFormat::WEBP;
    }
    
    // 检查TIFF签名
    if image_data.starts_with(&[0x49, 0x49, 0x2A, 0x00]) || 
       image_data.starts_with(&[0x4D, 0x4D, 0x00, 0x2A]) {
        return ImageFormat::TIFF;
    }
    
    ImageFormat::Unknown
}

// 读取图像元数据
#[cfg(feature = "multimodal")]
pub fn read_image_metadata(image_data: &[u8]) -> Result<ImageMetadata> {
    if image_data.is_empty() {
        return Err(Error::new(ErrorKind::InvalidData, "Empty image data"));
    }
    
    // 使用image库解码图像头部获取元数据
    let format = detect_image_format(image_data);
    
    if format == ImageFormat::Unknown {
        return Err(Error::new(ErrorKind::InvalidData, "Unknown image format"));
    }
    
    let img = match image::load_from_memory(image_data) {
        Ok(img) => img,
        Err(e) => return Err(Error::new(ErrorKind::InvalidData, format!("Failed to decode image: {}", e))),
    };
    
    use image::GenericImageView;
    let (width, height) = img.dimensions();
    let color_type = img.color();
    
    let has_alpha = match color_type {
        image::ColorType::Rgba8 | image::ColorType::Rgba16 | image::ColorType::Rgba32F => true,
        _ => false,
    };
    
    let color_depth = match color_type {
        image::ColorType::L8 | image::ColorType::Rgb8 | image::ColorType::Rgba8 | image::ColorType::La8 => 8,
        image::ColorType::L16 | image::ColorType::Rgb16 | image::ColorType::Rgba16 | image::ColorType::La16 => 16,
        image::ColorType::Rgb32F | image::ColorType::Rgba32F => 32,
        _ => 8,
    };
    
    Ok(ImageMetadata {
        dimensions: ImageDimensions::new(width, height),
        format,
        color_depth,
        has_alpha,
        dpi: None, // image库不直接提供DPI信息
    })
}

#[cfg(not(feature = "multimodal"))]
pub fn read_image_metadata(_image_data: &[u8]) -> Result<ImageMetadata> {
    Err(Error::new(ErrorKind::Unsupported, "Image processing requires 'multimodal' feature"))
}

// 转换图像格式
#[cfg(feature = "multimodal")]
pub fn convert_format(image_data: &[u8], target_format: ImageFormat) -> Result<Vec<u8>> {
    if image_data.is_empty() {
        return Err(Error::new(ErrorKind::InvalidData, "Empty image data"));
    }
    
    let source_format = detect_image_format(image_data);
    if source_format == ImageFormat::Unknown {
        return Err(Error::new(ErrorKind::InvalidData, "Unknown source image format"));
    }
    
    if source_format == target_format {
        return Ok(image_data.to_vec());
    }
    
    // 解码图像
    let img = match image::load_from_memory(image_data) {
        Ok(img) => img,
        Err(e) => return Err(Error::new(ErrorKind::InvalidData, format!("Failed to decode image: {}", e))),
    };
    
    // 根据目标格式编码
    let mut output = Vec::new();
    let mut cursor = Cursor::new(&mut output);
    match target_format {
        ImageFormat::PNG => {
            img.write_to(&mut cursor, image::ImageOutputFormat::Png)
                .map_err(|e| Error::new(ErrorKind::Other, format!("Failed to encode PNG: {}", e)))?;
        },
        ImageFormat::JPEG => {
            img.write_to(&mut cursor, image::ImageOutputFormat::Jpeg(90))
                .map_err(|e| Error::new(ErrorKind::Other, format!("Failed to encode JPEG: {}", e)))?;
        },
        ImageFormat::WEBP => {
            img.write_to(&mut cursor, image::ImageOutputFormat::WebP)
                .map_err(|e| Error::new(ErrorKind::Other, format!("Failed to encode WebP: {}", e)))?;
        },
        ImageFormat::TIFF => {
            // 当前image库不直接支持TIFF输出，可考虑使用其他库如tiff
            return Err(Error::new(ErrorKind::Unsupported, "TIFF encoding not supported"));
        },
        ImageFormat::BMP => {
            img.write_to(&mut cursor, image::ImageOutputFormat::Bmp)
                .map_err(|e| Error::new(ErrorKind::Other, format!("Failed to encode BMP: {}", e)))?;
        },
        ImageFormat::GIF => {
            // 如果需要GIF支持，可能需要额外设置
            return Err(Error::new(ErrorKind::Unsupported, "GIF encoding not supported"));
        },
        _ => {
            return Err(Error::new(ErrorKind::InvalidInput, "Unsupported target format"));
        }
    }
    
    Ok(output)
}

#[cfg(not(feature = "multimodal"))]
pub fn convert_format(_image_data: &[u8], _target_format: ImageFormat) -> Result<Vec<u8>> {
    Err(Error::new(ErrorKind::Unsupported, "Image processing requires 'multimodal' feature"))
}

// 公共工具函数，判断是否为有效的图像文件路径
pub fn is_image_file<P: AsRef<Path>>(path: P) -> bool {
    let format = ImageFormat::from_path(path);
    format != ImageFormat::Unknown
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_image_format_from_extension() {
        assert_eq!(ImageFormat::from_extension("png"), ImageFormat::PNG);
        assert_eq!(ImageFormat::from_extension("jpg"), ImageFormat::JPEG);
        assert_eq!(ImageFormat::from_extension("jpeg"), ImageFormat::JPEG);
        assert_eq!(ImageFormat::from_extension("webp"), ImageFormat::WEBP);
        assert_eq!(ImageFormat::from_extension("gif"), ImageFormat::GIF);
        assert_eq!(ImageFormat::from_extension("bmp"), ImageFormat::BMP);
        assert_eq!(ImageFormat::from_extension("tiff"), ImageFormat::TIFF);
        assert_eq!(ImageFormat::from_extension("unknown"), ImageFormat::Unknown);
    }
    
    #[test]
    fn test_image_dimensions() {
        let dim = ImageDimensions::new(800, 600);
        assert_eq!(dim.width, 800);
        assert_eq!(dim.height, 600);
        assert_eq!(dim.aspect_ratio(), 800.0 / 600.0);
        assert_eq!(dim.pixel_count(), 800 * 600);
        assert!(!dim.is_empty());
        
        let empty_dim = ImageDimensions::new(0, 0);
        assert!(empty_dim.is_empty());
        assert_eq!(empty_dim.aspect_ratio(), 0.0);
    }
    
    #[test]
    fn test_interpolation_mode() {
        assert_eq!(InterpolationMode::from_str("nearest"), InterpolationMode::Nearest);
        assert_eq!(InterpolationMode::from_str("bilinear"), InterpolationMode::Bilinear);
        assert_eq!(InterpolationMode::from_str("unknown"), InterpolationMode::default());
        
        let mode = InterpolationMode::Bicubic;
        assert!(mode.description().contains("质量较高"));
    }
} 