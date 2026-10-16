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

/// 450 input channels per group is more than one im2col tile holds, so the 2D path multiplies
/// each tile in input-channel slices (here 404 + 46 channels) and accumulates. Inputs are
/// constant over each plane and weights constant over each kernel, so every output is
/// `(b + 1) * taps * S_g + bias_g`: `taps` counts the kernel positions inside the padded 3x3 input
/// (4, 6 or 9) and `S_g` sums `(c % 7 + 1) * (c_g % 5 - 1)` over the group's channels (1785, 1810).
#[test]
fn conv2d_input_channel_slices_with_groups_padding_and_batch() {
    let (batch, groups, c_in_group) = (2, 2, 450);
    let c_in = groups * c_in_group;
    let device = Device::flex();
    let x: Vec<f32> = (0..batch * c_in)
        .flat_map(|i| [((i % c_in % 7 + 1) * (i / c_in + 1)) as f32; 9])
        .collect();
    let weight: Vec<f32> = (0..c_in)
        .flat_map(|i| [(i % c_in_group % 5) as f32 - 1.0; 9])
        .collect();
    let x = Tensor::<4>::from_data(TensorData::new(x, [batch, c_in, 3, 3]), &device);
    let weight =
        Tensor::<4>::from_data(TensorData::new(weight, [groups, c_in_group, 3, 3]), &device);
    let bias = Tensor::<1>::from_data(TensorData::new(vec![0.5f32, -0.25], [2]), &device);

    let out = module::conv2d(
        x,
        weight,
        Some(bias),
        ConvOptions::new([1, 1], [1, 1], [1, 1], groups),
    );

    let taps = [4.0, 6.0, 4.0, 6.0, 9.0, 6.0, 4.0, 6.0, 4.0];
    let mut expected = Vec::new();
    for b in [1.0, 2.0] {
        for (s, bias) in [(1785.0, 0.5), (1810.0, -0.25)] {
            expected.extend(taps.iter().map(|t| b * t * s + bias));
        }
    }
    assert_eq!(out.dims(), [2, 2, 3, 3]);
    // Integer-valued sums, exact in f32 whatever the summation order
    assert_eq!(out.into_data().try_to_vec::<f32>().unwrap(), expected);
}
