use std::path::Path;
use std::fs;
use crate::error::Result;

/// 确保目录存在，如果不存在则创建
pub fn ensure_directory<P: AsRef<Path>>(path: P) -> Result<()> {
    let path = path.as_ref();
    if !path.exists() {
        fs::create_dir_all(path)?;
    } else if !path.is_dir() {
        return Err(crate::Error::io_error(format!("路径{}存在但不是目录", path.display())));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;
    use tempfile::tempdir;

    #[test]
    fn test_ensure_directory_creates_new_dir() {
        let temp = tempdir().unwrap();
        let test_dir = temp.path().join("test_dir");
        
        assert!(!test_dir.exists());
        ensure_directory(&test_dir).unwrap();
        assert!(test_dir.exists());
        assert!(test_dir.is_dir());
    }

    #[test]
    fn test_ensure_directory_existing_dir() {
        let temp = tempdir().unwrap();
        
        ensure_directory(temp.path()).unwrap();
        assert!(temp.path().exists());
        assert!(temp.path().is_dir());
    }
} 