// 缓存实现模块
// 提供存储引擎的缓存功能实现

use std::collections::HashMap;
use std::sync::RwLock;
use chrono::{DateTime, Utc};
use serde::{Serialize, Deserialize};

/// Cache replacement policies
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum CachePolicy {
    LRU,  // Least Recently Used
    MRU,  // Most Recently Used 
    FIFO, // First In First Out
}

/// Cache entry structure
#[derive(Debug, Clone)]
pub struct CacheEntry {
    pub data: Vec<u8>,
    pub timestamp: DateTime<Utc>,
}

/// Cache manager for storage engine
pub struct CacheManager {
    entries: RwLock<HashMap<String, CacheEntry>>,
    size: RwLock<usize>,
    max_size: usize,
    policy: CachePolicy,
}

impl CacheManager {
    /// Create a new cache manager
    pub fn new(max_size: usize, policy: CachePolicy) -> Self {
        Self {
            entries: RwLock::new(HashMap::new()),
            size: RwLock::new(0),
            max_size,
            policy,
        }
    }
    
    /// Get entry from cache
    pub fn get(&self, key: &str) -> Option<Vec<u8>> {
        let entries = self.entries.read().ok()?;
        entries.get(key).map(|entry| {
            let mut data = entry.data.clone();
            
            // For LRU, we need to update the timestamp
            if self.policy == CachePolicy::LRU {
                drop(entries);
                if let Ok(mut entries) = self.entries.write() {
                    if let Some(entry) = entries.get_mut(key) {
                        entry.timestamp = Utc::now();
                    }
                }
            }
            
            data
        })
    }
    
    /// Put entry into cache
    pub fn put(&self, key: String, data: Vec<u8>) {
        let entry_size = data.len();
        let mut entries = match self.entries.write() {
            Ok(e) => e,
            Err(_) => return, // 如果无法获取写锁，直接返回
        };
        let mut size = match self.size.write() {
            Ok(s) => s,
            Err(_) => return, // 如果无法获取写锁，直接返回
        };
        
        // If key already exists, update size delta
        let size_delta = if let Some(old_entry) = entries.get(&key) {
            entry_size as isize - old_entry.data.len() as isize
        } else {
            entry_size as isize
        };
        
        // Ensure we have enough space
        if *size as isize + size_delta > self.max_size as isize {
            self.evict(&mut entries, &mut size, entry_size);
        }
        
        // Update cache entry
        entries.insert(key, CacheEntry {
            data,
            timestamp: Utc::now(),
        });
        
        // Update size
        *size = (*size as isize + size_delta) as usize;
    }
    
    /// Evict entries based on cache policy
    fn evict(&self, entries: &mut HashMap<String, CacheEntry>, size: &mut usize, needed_space: usize) {
        if entries.is_empty() {
            return;
        }
        
        match self.policy {
            CachePolicy::LRU => {
                // Evict least recently used entries
                while *size + needed_space > self.max_size && !entries.is_empty() {
                    if let Some((key, entry)) = entries.iter()
                        .min_by_key(|(_, e)| e.timestamp) {
                        let key = key.clone();
                        let entry_size = entry.data.len();
                        entries.remove(&key);
                        *size = size.saturating_sub(entry_size);
                    } else {
                        break;
                    }
                }
            },
            CachePolicy::MRU => {
                // Evict most recently used entries
                while *size + needed_space > self.max_size && !entries.is_empty() {
                    if let Some((key, entry)) = entries.iter()
                        .max_by_key(|(_, e)| e.timestamp) {
                        let key = key.clone();
                        let entry_size = entry.data.len();
                        entries.remove(&key);
                        *size = size.saturating_sub(entry_size);
                    } else {
                        break;
                    }
                }
            },
            CachePolicy::FIFO => {
                // Evict oldest entries first (First In First Out)
                while *size + needed_space > self.max_size && !entries.is_empty() {
                    if let Some((key, entry)) = entries.iter()
                        .min_by_key(|(_, e)| e.timestamp) {
                        let key = key.clone();
                        let entry_size = entry.data.len();
                        entries.remove(&key);
                        *size = size.saturating_sub(entry_size);
                    } else {
                        break;
                    }
                }
            }
        }
    }
    
    /// Clear all entries from cache
    pub fn clear(&self) {
        if let Ok(mut entries) = self.entries.write() {
            entries.clear();
            if let Ok(mut size) = self.size.write() {
                *size = 0;
            }
        }
    }
    
    /// Get current cache size
    pub fn get_size(&self) -> usize {
        self.size.read().map(|s| *s).unwrap_or(0)
    }
    
    /// Get number of entries in cache
    pub fn get_count(&self) -> usize {
        self.entries.read().map(|e| e.len()).unwrap_or(0)
    }
}