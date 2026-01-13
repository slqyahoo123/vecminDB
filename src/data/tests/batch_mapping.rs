#![cfg(test)]

use chrono::Utc;
use std::collections::HashMap;

#[test]
fn data_batch_to_core_and_back_round_trip() {
	let mut db = crate::data::exports::DataBatch::with_data(vec![vec![1.0, 2.0], vec![3.0, 4.0]], vec![0.0, 1.0]);
	db.metadata.insert("k".into(), "v".into());

	let core: crate::core::types::CoreDataBatch = (&db).into();
	let back: crate::data::exports::DataBatch = core.try_into().expect("try_into");

	assert_eq!(back.data, vec![vec![1.0, 2.0], vec![3.0, 4.0]]);
	assert_eq!(back.labels, vec![0.0, 1.0]);
	assert_eq!(back.batch_size, 2);
	assert_eq!(back.feature_dim, 2);
	assert_eq!(back.metadata.get("k"), Some(&"v".to_string()));
}

#[test]
fn processor_batch_to_processed_data_and_back() {
	let features = crate::core::types::CoreTensorData {
		id: Some("x".into()),
		shape: vec![2, 2],
		data: vec![1.0, 2.0, 3.0, 4.0],
		dtype: crate::core::types::DataType::Float32,
		device: crate::core::types::DeviceType::CPU,
		requires_grad: false,
		gradient: None,
		metadata: HashMap::new(),
	};
	let pb = crate::data::processor::types::ProcessorBatch {
		id: "pb1".into(),
		features,
		labels: None,
		metadata: HashMap::new(),
		format: crate::data::DataFormat::CSV,
		field_names: vec!["a".into(), "b".into()],
		records: Vec::new(),
	};

	let pd: crate::core::interfaces::ProcessedData = pb.clone().into();
	assert_eq!(pd.shape, vec![2, 2]);
	assert_eq!(pd.processed_content, vec![1.0, 2.0, 3.0, 4.0]);

	let back: crate::data::processor::types::ProcessorBatch = pd.into();
	assert_eq!(back.features.shape, vec![2, 2]);
	assert_eq!(back.features.data, vec![1.0, 2.0, 3.0, 4.0]);
}
