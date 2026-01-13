#![cfg(test)]

use std::sync::Arc;
use crate::core::interfaces::storage_interface::{StorageService as CoreStorageService, StorageTransaction as CoreStorageTx};
use crate::storage::engine::core::StorageEngineImpl;

#[tokio::test]
async fn transaction_commit_persists_changes() {
	let engine = StorageEngineImpl::new_in_memory().expect("engine");
	let key = "tx:test:commit";
	let value = b"hello";

	let mut tx = CoreStorageService::transaction(&engine).expect("tx");
	tx.store(key, value).expect("store");
	tx.commit().expect("commit");

	let got = CoreStorageService::retrieve(&engine, key).await.expect("retrieve");
	assert_eq!(got.as_deref(), Some(value));
}

#[tokio::test]
async fn transaction_rollback_discards_changes() {
	let engine = StorageEngineImpl::new_in_memory().expect("engine");
	let key = "tx:test:rollback";
	let value = b"world";

	let mut tx = CoreStorageService::transaction(&engine).expect("tx");
	tx.store(key, value).expect("store");
	tx.rollback().expect("rollback");

	let got = CoreStorageService::retrieve(&engine, key).await.expect("retrieve");
	assert!(got.is_none());
}
