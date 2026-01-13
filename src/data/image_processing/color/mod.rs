// 图像颜色处理模块
//
// 该模块提供了颜色调整、变换和分析功能

use std::io::{Result, Error, ErrorKind};

/// 颜色空间枚举
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ColorSpace {
    /// RGB颜色空间
    RGB,
    /// RGBA颜色空间(含透明通道)
    RGBA,
    /// 灰度
    Grayscale,
    /// HSV色彩空间
    HSV,
    /// CMYK色彩空间(印刷)
    CMYK,
    /// Lab色彩空间
    Lab,
}

/// 颜色信息结构体
#[derive(Debug, Clone)]
pub struct ColorInfo {
    /// 主色调 (RGB)
    pub dominant_color: [u8; 3],
    /// 色彩数量
    pub color_count: usize,
    /// 平均亮度 (0-255)
    pub average_brightness: u8,
    /// 对比度值
    pub contrast: f32,
    /// 是否包含透明通道
    pub has_alpha: bool,
    /// 颜色直方图
    pub histogram: Vec<u32>,
}

impl ColorInfo {
    /// 创建新的颜色信息实例
    pub fn new() -> Self {
        Self {
            dominant_color: [0, 0, 0],
            color_count: 0,
            average_brightness: 0,
            contrast: 0.0,
            has_alpha: false,
            histogram: vec![0; 256], // 默认灰度直方图
        }
    }
    
    /// 检查图像是否为暗色调
    pub fn is_dark(&self) -> bool {
        self.average_brightness < 128
    }
    
    /// 检查图像是否为亮色调
    pub fn is_bright(&self) -> bool {
        self.average_brightness >= 128
    }
    
    /// 检查图像是否为高对比度
    pub fn is_high_contrast(&self) -> bool {
        self.contrast > 50.0
    }
    
    /// 检查图像是否为低对比度
    pub fn is_low_contrast(&self) -> bool {
        self.contrast < 10.0
    }
    
    /// 获取颜色直方图
    pub fn get_histogram(&self) -> &[u32] {
        &self.histogram
    }
    
    /// 获取主色调的十六进制表示
    pub fn dominant_hex(&self) -> String {
        format!("#{:02x}{:02x}{:02x}", 
                self.dominant_color[0], 
                self.dominant_color[1], 
                self.dominant_color[2])
    }
}

impl Default for ColorInfo {
    fn default() -> Self {
        Self::new()
    }
}

/// 颜色调整配置
#[derive(Debug, Clone)]
pub struct ColorAdjustment {
    /// 亮度调整 (-100 到 100)
    pub brightness: i8,
    /// 对比度调整 (-100 到 100)
    pub contrast: i8,
    /// 饱和度调整 (-100 到 100)
    pub saturation: i8,
    /// 色相调整 (-180 到 180 度)
    pub hue: i16,
    /// 伽马调整 (0.1 到 5.0)
    pub gamma: f32,
    /// 颜色温度调整 (2000K 到 15000K)
    pub temperature: u16,
}

impl Default for ColorAdjustment {
    fn default() -> Self {
        Self {
            brightness: 0,
            contrast: 0,
            saturation: 0,
            hue: 0,
            gamma: 1.0,
            temperature: 6500, // 标准日光
        }
    }
}

impl ColorAdjustment {
    /// 创建新的颜色调整配置
    pub fn new() -> Self {
        Self::default()
    }
    
    /// 设置亮度
    pub fn with_brightness(mut self, value: i8) -> Self {
        self.brightness = value.max(-100).min(100);
        self
    }
    
    /// 设置对比度
    pub fn with_contrast(mut self, value: i8) -> Self {
        self.contrast = value.max(-100).min(100);
        self
    }
    
    /// 设置饱和度
    pub fn with_saturation(mut self, value: i8) -> Self {
        self.saturation = value.max(-100).min(100);
        self
    }
    
    /// 设置色相
    pub fn with_hue(mut self, value: i16) -> Self {
        self.hue = value.max(-180).min(180);
        self
    }
    
    /// 设置伽马
    pub fn with_gamma(mut self, value: f32) -> Self {
        self.gamma = value.max(0.1).min(5.0);
        self
    }
    
    /// 设置色温
    pub fn with_temperature(mut self, value: u16) -> Self {
        self.temperature = value.max(2000).min(15000);
        self
    }
    
    /// 应用所有调整
    pub fn apply_all(&self) -> bool {
        self.brightness != 0 || 
        self.contrast != 0 || 
        self.saturation != 0 || 
        self.hue != 0 || 
        (self.gamma - 1.0).abs() > 0.01 || 
        self.temperature != 6500
    }
    
    /// 重置所有调整
    pub fn reset(&mut self) {
        *self = Self::default();
    }
    
    /// 验证配置是否有效
    pub fn validate(&self) -> Result<()> {
        if self.brightness < -100 || self.brightness > 100 {
            return Err(Error::new(ErrorKind::InvalidInput, "亮度值应在-100到100之间"));
        }
        
        if self.contrast < -100 || self.contrast > 100 {
            return Err(Error::new(ErrorKind::InvalidInput, "对比度值应在-100到100之间"));
        }
        
        if self.saturation < -100 || self.saturation > 100 {
            return Err(Error::new(ErrorKind::InvalidInput, "饱和度值应在-100到100之间"));
        }
        
        if self.hue < -180 || self.hue > 180 {
            return Err(Error::new(ErrorKind::InvalidInput, "色相值应在-180到180之间"));
        }
        
        if self.gamma < 0.1 || self.gamma > 5.0 {
            return Err(Error::new(ErrorKind::InvalidInput, "伽马值应在0.1到5.0之间"));
        }
        
        if self.temperature < 2000 || self.temperature > 15000 {
            return Err(Error::new(ErrorKind::InvalidInput, "色温值应在2000K到15000K之间"));
        }
        
        Ok(())
    }
}

/// 分析图像颜色信息
pub fn analyze_colors(_image_data: &[u8]) -> Result<ColorInfo> {
    // 实际实现中需要解码图像并分析像素数据
    // 这里仅提供接口定义，返回默认值
    Ok(ColorInfo::default())
}

/// 调整图像颜色
pub fn adjust_colors(_image_data: &[u8], _settings: &ColorAdjustment) -> Result<Vec<u8>> {
    // 实际实现中需要解码图像、应用色彩调整并重新编码
    // 这里仅提供接口定义，返回原始数据
    Ok(_image_data.to_vec())
}

/// 将图像转换为灰度
pub fn convert_to_grayscale(_image_data: &[u8]) -> Result<Vec<u8>> {
    // 实际实现中需要解码图像、应用灰度转换并重新编码
    // 这里仅提供接口定义，返回原始数据
    Ok(_image_data.to_vec())
}

/// 从RGB值计算亮度
pub fn calculate_brightness(r: u8, g: u8, b: u8) -> u8 {
    // 使用加权平均法计算亮度，考虑人眼对不同颜色的敏感度
    // 公式: Y = 0.299*R + 0.587*G + 0.114*B
    ((0.299 * r as f32) + (0.587 * g as f32) + (0.114 * b as f32)) as u8
}

/// RGB转HSV
pub fn rgb_to_hsv(r: u8, g: u8, b: u8) -> (f32, f32, f32) {
    let r_f = r as f32 / 255.0;
    let g_f = g as f32 / 255.0;
    let b_f = b as f32 / 255.0;
    
    let max = r_f.max(g_f).max(b_f);
    let min = r_f.min(g_f).min(b_f);
    let delta = max - min;
    
    // 色相计算
    let hue = if delta < 0.001 {
        0.0 // 灰色没有色相
    } else if (max - r_f).abs() < 0.001 {
        60.0 * (((g_f - b_f) / delta) % 6.0)
    } else if (max - g_f).abs() < 0.001 {
        60.0 * (((b_f - r_f) / delta) + 2.0)
    } else {
        60.0 * (((r_f - g_f) / delta) + 4.0)
    };
    
    // 饱和度计算
    let saturation = if max < 0.001 { 0.0 } else { delta / max };
    
    // 明度就是最大值
    let value = max;
    
    (hue, saturation, value)
}

/// HSV转RGB
pub fn hsv_to_rgb(h: f32, s: f32, v: f32) -> (u8, u8, u8) {
    let h = if h >= 360.0 { h - 360.0 } else { h };
    
    let c = v * s;
    let x = c * (1.0 - ((h / 60.0) % 2.0 - 1.0).abs());
    let m = v - c;
    
    let (r, g, b) = if h < 60.0 {
        (c, x, 0.0)
    } else if h < 120.0 {
        (x, c, 0.0)
    } else if h < 180.0 {
        (0.0, c, x)
    } else if h < 240.0 {
        (0.0, x, c)
    } else if h < 300.0 {
        (x, 0.0, c)
    } else {
        (c, 0.0, x)
    };
    
    (
        ((r + m) * 255.0) as u8,
        ((g + m) * 255.0) as u8,
        ((b + m) * 255.0) as u8
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_color_adjustment_builder() {
        let adjustment = ColorAdjustment::new()
            .with_brightness(20)
            .with_contrast(15)
            .with_saturation(-10)
            .with_hue(30)
            .with_gamma(1.2)
            .with_temperature(5500);
        
        assert_eq!(adjustment.brightness, 20);
        assert_eq!(adjustment.contrast, 15);
        assert_eq!(adjustment.saturation, -10);
        assert_eq!(adjustment.hue, 30);
        assert_eq!(adjustment.gamma, 1.2);
        assert_eq!(adjustment.temperature, 5500);
        assert!(adjustment.apply_all());
    }
    
    #[test]
    fn test_color_adjustment_validation() {
        let valid = ColorAdjustment::new()
            .with_brightness(20)
            .with_contrast(15);
        assert!(valid.validate().is_ok());
        
        let invalid_brightness = ColorAdjustment::new()
            .with_brightness(120); // 超出范围
        assert!(invalid_brightness.validate().is_err());
    }
    
    #[test]
    fn test_brightness_calculation() {
        assert_eq!(calculate_brightness(255, 255, 255), 255); // 白色
        assert_eq!(calculate_brightness(0, 0, 0), 0); // 黑色
        assert_eq!(calculate_brightness(255, 0, 0), 76); // 红色
        assert_eq!(calculate_brightness(0, 255, 0), 149); // 绿色
        assert_eq!(calculate_brightness(0, 0, 255), 29); // 蓝色
    }
    
    #[test]
    fn test_rgb_hsv_conversion() {
        // 红色
        let (h, s, v) = rgb_to_hsv(255, 0, 0);
        assert!((h - 0.0).abs() < 0.1);
        assert!((s - 1.0).abs() < 0.01);
        assert!((v - 1.0).abs() < 0.01);
        
        // 绿色
        let (h, s, v) = rgb_to_hsv(0, 255, 0);
        assert!((h - 120.0).abs() < 0.1);
        assert!((s - 1.0).abs() < 0.01);
        assert!((v - 1.0).abs() < 0.01);
        
        // 往返转换测试
        let original = (128, 64, 192);
        let (h, s, v) = rgb_to_hsv(original.0, original.1, original.2);
        let (r, g, b) = hsv_to_rgb(h, s, v);
        
        // 考虑到舍入误差，允许一定的偏差
        assert!((r as i32 - original.0 as i32).abs() <= 1);
        assert!((g as i32 - original.1 as i32).abs() <= 1);
        assert!((b as i32 - original.2 as i32).abs() <= 1);
    }
} 