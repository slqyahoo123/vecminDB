use std::sync::{Arc, Mutex};
use std::collections::HashMap;
use std::fmt::{Debug, Formatter, Result as FmtResult};

#[cfg(feature = "multimodal")]
use image::{GenericImageView, DynamicImage, imageops};
#[cfg(feature = "multimodal")]
use image::imageops::FilterType;
use base64::{Engine as _, engine::general_purpose::STANDARD as BASE64_STANDARD};
use rayon::prelude::*;
// use serde::{Serialize, Deserialize}; // derive used inline as serde::Serialize / serde::Deserialize

use crate::{Error, Result};
use crate::compat::tensor::TensorData;
use crate::data::multimodal::extractors::interface::FeatureExtractor;
use crate::data::multimodal::*;
use crate::data::multimodal::models::image_models::ImageFeatureModel;

/// 图像特征提取和处理的配置参数
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ImageProcessingConfig {
    /// 输出特征维度
    pub feature_dim: usize,
    /// 图像预处理尺寸
    pub resize_dim: u32,
    /// 是否启用自动增强
    pub auto_enhance: bool,
    /// 是否应用标准化
    pub normalize: bool,
    /// 色彩调整参数
    pub color_adjustment: Option<ColorAdjustment>,
    /// 质量提升参数
    pub quality_enhancement: Option<QualityEnhancement>,
    /// 缓存大小
    pub cache_size: usize,
}

/// 色彩调整参数
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ColorAdjustment {
    /// 亮度调整因子 (-1.0 到 1.0)
    pub brightness: f32,
    /// 对比度调整因子 (0.0 到 2.0)
    pub contrast: f32,
    /// 饱和度调整因子 (0.0 到 2.0)
    pub saturation: f32,
    /// 色调调整 (-180 到 180 度)
    pub hue: i32,
}

/// 图像质量提升参数
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct QualityEnhancement {
    /// 锐化强度 (0.0 到 1.0)
    pub sharpness: f32,
    /// 降噪强度 (0.0 到 1.0)
    pub denoise: f32,
    /// 超分辨率因子 (1 到 4)
    pub super_resolution_factor: u8,
}

/// 图像特征提取器
pub struct ImageFeatureExtractor {
    /// 配置参数
    config: ImageProcessingConfig,
    /// 特征模型
    model: Box<dyn ImageFeatureModel + Send + Sync>,
    /// 处理结果缓存
    cache: Arc<Mutex<lru::LruCache<u64, Vec<f32>>>>,
}

impl ImageFeatureExtractor {
    /// 创建新的图像特征提取器
    pub fn new(config: ImageProcessingConfig, model_type: &str) -> Result<Self> {
        use crate::data::multimodal::models::image_models::{ResNetConfig, VGGConfig, EfficientNetConfig, CLIPImageConfig};
        
        // 创建模型
        let model: Box<dyn ImageFeatureModel + Send + Sync> = match model_type.to_lowercase().as_str() {
            "resnet" => {
                let resnet_config = ResNetConfig {
                    version: 50, // 默认版本
                    input_size: (config.resize_dim, config.resize_dim),
                    pretrained: true,
                    feature_layer: "avgpool".to_string(),
                    model_path: None,
                };
                Box::new(crate::data::multimodal::models::image_models::ResNetFeatureModel::new(resnet_config)?)
            },
            "vgg" => {
                let vgg_config = VGGConfig {
                    version: 16, // 默认版本
                    input_size: (config.resize_dim, config.resize_dim),
                    pretrained: true,
                    feature_layer: "fc7".to_string(),
                    model_path: None,
                };
                Box::new(crate::data::multimodal::models::image_models::VGGFeatureModel::new(vgg_config)?)
            },
            "efficientnet" => {
                let efficientnet_config = EfficientNetConfig {
                    version: "B0".to_string(), // 默认版本
                    input_size: (config.resize_dim, config.resize_dim),
                    pretrained: true,
                    model_path: None,
                };
                Box::new(crate::data::multimodal::models::image_models::EfficientNetFeatureModel::new(efficientnet_config)?)
            },
            "clip" => {
                let clip_config = CLIPImageConfig {
                    version: "ViT-B/32".to_string(), // 默认版本
                    input_size: (config.resize_dim, config.resize_dim),
                    pretrained: true,
                    model_path: None,
                };
                Box::new(crate::data::multimodal::models::image_models::CLIPImageFeatureModel::new(clip_config)?)
            },
            _ => return Err(Error::invalid_argument(format!("不支持的图像特征模型类型: {}", model_type))),
        };
        
        // 创建LRU缓存
        use std::num::NonZeroUsize;
        let cache_size = NonZeroUsize::new(config.cache_size.max(1))
            .ok_or_else(|| Error::invalid_argument("缓存大小必须大于0".to_string()))?;
        let cache = Arc::new(Mutex::new(lru::LruCache::new(cache_size)));
        
        Ok(Self {
            config,
            model,
            cache,
        })
    }
    
    /// 使用默认配置创建图像特征提取器
    pub fn default() -> Result<Self> {
        let config = ImageProcessingConfig {
            feature_dim: 512,
            resize_dim: 224,
            auto_enhance: true,
            normalize: true,
            color_adjustment: None,
            quality_enhancement: None,
            cache_size: 100,
        };
        
        Self::new(config, "resnet")
    }
    
    /// 处理Base64编码的图像
    #[cfg(feature = "multimodal")]
    pub fn process_base64_image(&self, base64_data: &str) -> Result<DynamicImage> {
        // 去除base64前缀（如果有）
        let base64_clean = if base64_data.contains(";base64,") {
            base64_data.split(";base64,").nth(1).unwrap_or(base64_data)
        } else {
            base64_data
        };
        
        // 解码Base64
        let image_data = BASE64_STANDARD.decode(base64_clean)
            .map_err(|e| Error::data(format!("无法解码Base64图像数据: {}", e)))?;
        
        // 解析为图像
        let img = image::load_from_memory(&image_data)
            .map_err(|e| Error::data(format!("无法加载图像: {}", e)))?;
        
        Ok(img)
    }
    
    /// 预处理图像
    #[cfg(feature = "multimodal")]
    pub fn preprocess_image(&self, img: &DynamicImage) -> Result<DynamicImage> {
        let mut processed = img.clone();
        
        // 1. 调整大小
        processed = DynamicImage::ImageRgba8(
            imageops::resize(
                &processed.to_rgba8(), 
                self.config.resize_dim, 
                self.config.resize_dim, 
                FilterType::Lanczos3
            )
        );
        
        // 2. 自动增强（如果启用）
        if self.config.auto_enhance {
            processed = self.auto_enhance_image(&processed)?;
        }
        
        // 3. 应用色彩调整（如果配置）
        if let Some(ref color_adj) = self.config.color_adjustment {
            processed = self.adjust_colors(&processed, color_adj)?;
        }
        
        // 4. 应用质量提升（如果配置）
        if let Some(ref quality) = self.config.quality_enhancement {
            processed = self.enhance_quality(&processed, quality)?;
        }
        
        Ok(processed)
    }
    
    /// 自动增强图像
    #[cfg(feature = "multimodal")]
    fn auto_enhance_image(&self, img: &DynamicImage) -> Result<DynamicImage> {
        // 转换为RGB
        let mut rgb_img = img.to_rgb8();
        
        // 计算直方图
        let mut histogram = [0u32; 256];
        for pixel in rgb_img.pixels() {
            let luminance = ((pixel[0] as u32 * 299 + pixel[1] as u32 * 587 + pixel[2] as u32 * 114) / 1000) as u8;
            histogram[luminance as usize] += 1;
        }
        
        // 计算累积直方图
        let total_pixels = img.width() * img.height();
        let mut cumulative_hist = [0u32; 256];
        let mut sum = 0;
        for i in 0..256 {
            sum += histogram[i];
            cumulative_hist[i] = sum;
        }
        
        // 直方图均衡化 LUT
        let mut lut = [0u8; 256];
        for i in 0..256 {
            lut[i] = ((cumulative_hist[i] as f32 / total_pixels as f32) * 255.0) as u8;
        }
        
        // 应用均衡化
        for pixel in rgb_img.pixels_mut() {
            let r = lut[pixel[0] as usize];
            let g = lut[pixel[1] as usize];
            let b = lut[pixel[2] as usize];
            *pixel = image::Rgb([r, g, b]);
        }
        
        Ok(DynamicImage::ImageRgb8(rgb_img))
    }
    
    /// 调整图像色彩
    #[cfg(feature = "multimodal")]
    fn adjust_colors(&self, img: &DynamicImage, adjustment: &ColorAdjustment) -> Result<DynamicImage> {
        let mut rgb_img = img.to_rgb8();
        
        for pixel in rgb_img.pixels_mut() {
            // 转换为HSL色彩空间
            let (h, s, l) = rgb_to_hsl(pixel[0], pixel[1], pixel[2]);
            
            // 应用调整
            let h_new = (h + adjustment.hue as f32) % 360.0;
            let s_new = (s * adjustment.saturation).clamp(0.0, 1.0);
            let l_new = (l * adjustment.contrast + adjustment.brightness).clamp(0.0, 1.0);
            
            // 转回RGB
            let (r, g, b) = hsl_to_rgb(h_new, s_new, l_new);
            *pixel = image::Rgb([r, g, b]);
        }
        
        Ok(DynamicImage::ImageRgb8(rgb_img))
    }
    
    /// 增强图像质量
    #[cfg(feature = "multimodal")]
    fn enhance_quality(&self, img: &DynamicImage, enhancement: &QualityEnhancement) -> Result<DynamicImage> {
        let mut processed = img.clone();
        
        // 锐化处理
        if enhancement.sharpness > 0.0 {
            processed = self.apply_sharpening(&processed, enhancement.sharpness)?;
        }
        
        // 降噪处理
        if enhancement.denoise > 0.0 {
            processed = self.apply_denoising(&processed, enhancement.denoise)?;
        }
        
        // 超分辨率处理（简化实现）
        if enhancement.super_resolution_factor > 1 {
            let factor = enhancement.super_resolution_factor as u32;
            let new_width = processed.width() * factor;
            let new_height = processed.height() * factor;
            
            processed = DynamicImage::ImageRgba8(
                imageops::resize(
                    &processed.to_rgba8(),
                    new_width,
                    new_height,
                    FilterType::Lanczos3
                )
            );
            
            // 如果设置了resize_dim，确保不超过
            if new_width > self.config.resize_dim || new_height > self.config.resize_dim {
                processed = DynamicImage::ImageRgba8(
                    imageops::resize(
                        &processed.to_rgba8(),
                        self.config.resize_dim,
                        self.config.resize_dim,
                        FilterType::Lanczos3
                    )
                );
            }
        }
        
        Ok(processed)
    }
    
    /// 应用锐化
    #[cfg(feature = "multimodal")]
    fn apply_sharpening(&self, img: &DynamicImage, strength: f32) -> Result<DynamicImage> {
        // 创建卷积核 (3x3 拉普拉斯锐化核)
        let kernel: Vec<f32> = vec![
            -strength, -strength, -strength,
            -strength, 1.0 + 8.0 * strength, -strength,
            -strength, -strength, -strength
        ];
        
        // 应用卷积
        let filtered = imageops::filter3x3(&img.to_rgb8(), &kernel);
        
        Ok(DynamicImage::ImageRgb8(filtered))
    }
    
    /// 应用降噪（简化的高斯模糊）
    #[cfg(feature = "multimodal")]
    fn apply_denoising(&self, img: &DynamicImage, strength: f32) -> Result<DynamicImage> {
        // 根据强度确定高斯模糊半径
        let sigma = strength * 2.0;
        
        // 应用高斯模糊
        let blurred = imageops::blur(&img.to_rgb8(), sigma);
        
        Ok(DynamicImage::ImageRgb8(blurred))
    }
    
    /// 将图像转换为特征向量
    #[cfg(feature = "multimodal")]
    fn image_to_tensor(&self, img: &DynamicImage) -> Result<TensorData> {
        // 确保图像是RGB格式
        let rgb_img = img.to_rgb8();
        let width = rgb_img.width() as usize;
        let height = rgb_img.height() as usize;
        
        // 转换为浮点数据
        let mut data = Vec::with_capacity(width * height * 3);
        
        for pixel in rgb_img.pixels() {
            let mut r = pixel[0] as f32 / 255.0;
            let mut g = pixel[1] as f32 / 255.0;
            let mut b = pixel[2] as f32 / 255.0;
            
            // 应用标准化（如果启用）
            if self.config.normalize {
                // ImageNet标准化均值和方差
                r = (r - 0.485) / 0.229;
                g = (g - 0.456) / 0.224;
                b = (b - 0.406) / 0.225;
            }
            
            data.push(r);
            data.push(g);
            data.push(b);
        }
        
        // 创建张量数据
        use crate::compat::tensor::{TensorValues, DataType};
        let tensor = TensorData {
            data: TensorValues::F32(data),
            shape: vec![1, 3, height, width], // NCHW格式
            dtype: DataType::Float32,
            metadata: HashMap::new(),
        };
        
        Ok(tensor)
    }
    
    /// 提取图像特征
    #[cfg(feature = "multimodal")]
    pub fn extract_features_from_image(&self, img: &DynamicImage) -> Result<Vec<f32>> {
        // 预处理图像
        let processed_img = self.preprocess_image(img)?;
        
        // 转换为tensor
        let tensor = self.image_to_tensor(&processed_img)?;
        
        // 使用模型提取特征
        let features = self.model.extract_features(&tensor)?;
        
        Ok(features)
    }
    
    /// 从批量图像中提取特征
    #[cfg(feature = "multimodal")]
    pub fn batch_extract_features(&self, images: &[DynamicImage]) -> Result<Vec<Vec<f32>>> {
        // 使用Rayon并行处理
        let results: Vec<Result<Vec<f32>>> = images.par_iter()
            .map(|img| self.extract_features_from_image(img))
            .collect();
        
        // 处理结果
        let mut features = Vec::with_capacity(images.len());
        for result in results {
            features.push(result?);
        }
        
        Ok(features)
    }
    
    /// 生成图像的指纹（特征哈希）
    #[cfg(feature = "multimodal")]
    pub fn generate_image_fingerprint(&self, img: &DynamicImage) -> Result<String> {
        // 调整大小为8x8灰度图像
        let small_img = img.resize_exact(8, 8, FilterType::Lanczos3).to_luma8();
        
        // 计算平均值
        let mut sum = 0;
        for pixel in small_img.pixels() {
            sum += pixel[0] as u32;
        }
        let avg = sum / 64;
        
        // 生成哈希值
        let mut hash = String::with_capacity(64);
        for pixel in small_img.pixels() {
            if pixel[0] as u32 >= avg {
                hash.push('1');
            } else {
                hash.push('0');
            }
        }
        
        Ok(hash)
    }
    
    /// 比较两个图像指纹的相似度
    pub fn compare_fingerprints(&self, fp1: &str, fp2: &str) -> Result<f32> {
        if fp1.len() != fp2.len() {
            return Err(Error::invalid_argument("指纹长度不匹配".to_string()));
        }
        
        // 计算汉明距离
        let mut distance = 0;
        for (c1, c2) in fp1.chars().zip(fp2.chars()) {
            if c1 != c2 {
                distance += 1;
            }
        }
        
        // 计算相似度
        let similarity = 1.0 - (distance as f32 / fp1.len() as f32);
        
        Ok(similarity)
    }
}

impl FeatureExtractor for ImageFeatureExtractor {
    fn extract_features(&self, data: &[u8], metadata: Option<&HashMap<String, String>>) -> Result<Vec<f32>> {
        // 计算数据哈希值作为缓存键
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        std::hash::Hash::hash_slice(data, &mut hasher);
        let hash = std::hash::Hasher::finish(&hasher);
        
        // 检查缓存
        if let Ok(mut cache) = self.cache.lock() {
            if let Some(cached_features) = cache.get(&hash) {
                return Ok(cached_features.clone());
            }
        }
        
        // 解析图像数据
        let img = image::load_from_memory(data)
            .map_err(|e| Error::data(format!("无法加载图像数据: {}", e)))?;
        
        // 提取特征
        let features = self.extract_features_from_image(&img)?;
        
        // 更新缓存
        if let Ok(mut cache) = self.cache.lock() {
            cache.put(hash, features.clone());
        }
        
        Ok(features)
    }
    
    fn batch_extract(&self, data_batch: &[Vec<u8>], metadata_batch: Option<&[HashMap<String, String>]>) -> Result<Vec<Vec<f32>>> {
        // 加载所有图像
        let mut images = Vec::with_capacity(data_batch.len());
        for data in data_batch {
            let img = image::load_from_memory(data)
                .map_err(|e| Error::data(format!("无法加载图像数据: {}", e)))?;
            images.push(img);
        }
        
        // 批量提取特征
        self.batch_extract_features(&images)
    }
    
    fn get_output_dim(&self) -> usize {
        self.config.feature_dim
    }
    
    fn get_extractor_type(&self) -> String {
        "image".to_string()
    }
}

/// 将RGB值转换为HSL
fn rgb_to_hsl(r: u8, g: u8, b: u8) -> (f32, f32, f32) {
    let r_f = r as f32 / 255.0;
    let g_f = g as f32 / 255.0;
    let b_f = b as f32 / 255.0;
    
    let max = r_f.max(g_f).max(b_f);
    let min = r_f.min(g_f).min(b_f);
    
    let mut h = 0.0;
    let mut s = 0.0;
    let l = (max + min) / 2.0;
    
    if max != min {
        let d = max - min;
        s = if l > 0.5 { d / (2.0 - max - min) } else { d / (max + min) };
        
        if max == r_f {
            h = (g_f - b_f) / d + (if g_f < b_f { 6.0 } else { 0.0 });
        } else if max == g_f {
            h = (b_f - r_f) / d + 2.0;
        } else {
            h = (r_f - g_f) / d + 4.0;
        }
        
        h /= 6.0;
    }
    
    (h * 360.0, s, l)
}

/// 将HSL值转换为RGB
fn hsl_to_rgb(h: f32, s: f32, l: f32) -> (u8, u8, u8) {
    let h = h / 360.0;
    
    let r;
    let g;
    let b;
    
    if s == 0.0 {
        r = l;
        g = l;
        b = l;
    } else {
        let q = if l < 0.5 { l * (1.0 + s) } else { l + s - l * s };
        let p = 2.0 * l - q;
        
        r = hue_to_rgb(p, q, h + 1.0/3.0);
        g = hue_to_rgb(p, q, h);
        b = hue_to_rgb(p, q, h - 1.0/3.0);
    }
    
    ((r * 255.0) as u8, (g * 255.0) as u8, (b * 255.0) as u8)
}

/// HSL内部计算函数
fn hue_to_rgb(p: f32, q: f32, mut t: f32) -> f32 {
    if t < 0.0 { t += 1.0; }
    if t > 1.0 { t -= 1.0; }
    
    if t < 1.0/6.0 {
        return p + (q - p) * 6.0 * t;
    }
    if t < 1.0/2.0 {
        return q;
    }
    if t < 2.0/3.0 {
        return p + (q - p) * (2.0/3.0 - t) * 6.0;
    }
    
    p
}

impl Debug for ImageFeatureExtractor {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        f.debug_struct("ImageFeatureExtractor")
            .field("config", &self.config)
            .field("model", &"<model implementation>")
            .field("cache", &"<cache implementation>")
            .finish()
    }
} 