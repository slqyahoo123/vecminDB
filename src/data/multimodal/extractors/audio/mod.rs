// Copyright (c) 2023 Vecmind Labs. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//! 音频特征提取模块
//!
//! 提供从各种音频源（文件、URL、Base64等）提取特征向量的能力，
//! 用于相似性检索和机器学习任务

// 子模块声明
pub mod error;
pub mod source;
pub mod config;
pub mod types;
pub mod extractor;
pub mod extractors;
pub mod utils;

// 新增的实用工具子模块
pub mod conversion;
pub mod filter;
pub mod generators;
pub mod spectral;
pub mod temporal;

// 公开重要类型
pub use error::AudioError;
pub use source::AudioSource;
pub use config::AudioProcessingConfig;
pub use types::AudioFeatureType;
pub use extractor::AudioFeatureExtractor;

// 公开重要函数
pub use extractor::create_default_audio_extractor;

// 公开频谱分析函数
#[cfg(feature = "multimodal")]
pub use spectral::{
    compute_mfcc,
    compute_melspectrogram,
    compute_chroma,
    compute_spectral_centroid,
    compute_spectral_rolloff,
    compute_zero_crossing_rate,
    compute_energy,
};

// 公开时域分析函数
pub use temporal::{
    zero_crossing_rate,
    signal_energy,
    root_mean_square,
    crest_factor,
    kurtosis,
    skewness,
    envelope,
    temporal_features,
};

// 公开信号转换函数
#[cfg(feature = "multimodal")]
pub use conversion::{
    hz_to_mel,
    mel_to_hz,
    amplitude_to_db,
    spectrogram_to_db,
    frame_signal,
    compute_fft,
    dct,
    stft,
    magnitude_spectrum,
};

// 公开信号生成器函数（主要用于测试）
pub use generators::{
    generate_sine_wave,
    generate_square_wave,
    generate_sawtooth_wave,
    generate_triangle_wave,
    generate_white_noise,
    generate_uniform_noise,
    generate_impulse,
    generate_multitone,
    generate_audio_events,
};

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_create_default_audio_extractor() {
        // 测试创建默认音频提取器
        let extractor = create_default_audio_extractor();
        
        // 默认维度应为128
        assert_eq!(extractor.dimension, 128);
    }
    
    #[test]
    fn test_process_audio() {
        use source::AudioSource;
        use std::sync::Arc;
        
        // 创建提取器
        let extractor = create_default_audio_extractor();
        
        // 创建模拟音频源（采样率为44100Hz的1秒正弦波）
        let sample_rate = 44100;
        let data = generators::generate_sine_wave(440.0, 1.0, sample_rate, None, None);
        let audio_data = Arc::new(data);
        
        let audio_source = AudioSource::InMemory {
            data: audio_data,
            sample_rate,
        };
        
        // 提取特征
        let features = extractor.process_audio(&audio_source).unwrap();
        
        // 检查结果维度
        assert_eq!(features.len(), extractor.dimension);
    }
} 