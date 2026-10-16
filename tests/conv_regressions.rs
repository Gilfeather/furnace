//! Regression tests for the patched CPU backend in vendor/burn-flex.

use burn::tensor::{Device, Tensor, TensorData, module, ops::ConvOptions};

/// A 3D conv with depth padding and a stride that skips every real depth slice must only see
/// padding: the output is the bias alone. The 2D fast path once ignored depth padding here.
#[test]
fn conv3d_depth_padding_is_not_taken_by_2d_fast_path() {
    let device = Device::flex();
    let x = Tensor::<5>::from_data(TensorData::new(vec![1.0f32; 4], [1, 1, 1, 2, 2]), &device);
    let weight = Tensor::<5>::from_data(TensorData::new(vec![1.0f32], [1, 1, 1, 1, 1]), &device);
    let bias = Tensor::<1>::from_data(TensorData::new(vec![0.5f32], [1]), &device);
    // out_d = (1 + 2 * 1 - 1) / 3 + 1 = 1, and that slice reads depth -1 (padding)
    let options = ConvOptions::new([3, 1, 1], [1, 0, 0], [1, 1, 1], 1);

    let out = module::conv3d(x, weight, Some(bias), options);

    assert_eq!(out.dims(), [1, 1, 1, 2, 2]);
    assert_eq!(out.into_data().try_to_vec::<f32>().unwrap(), vec![0.5; 4]);
}
