//! The Burn models served in this benchmark, written as a Burn user would write them: plain
//! `Module`s, nothing furnace specific. Weights are random; speed does not depend on them.

use burn::module::Module;
use burn::nn::conv::{Conv2d, Conv2dConfig};
use burn::nn::pool::{AdaptiveAvgPool2d, AdaptiveAvgPool2dConfig, MaxPool2d, MaxPool2dConfig};
use burn::nn::transformer::{
    TransformerEncoder, TransformerEncoderConfig, TransformerEncoderInput,
};
use burn::nn::{
    BatchNorm, BatchNormConfig, Embedding, EmbeddingConfig, Linear, LinearConfig, PaddingConfig2d,
    Relu,
};
use burn::tensor::{Device, Int, Tensor};

/// A small CNN: `[batch, 1, 28, 28]` -> `[batch, 10]`, a copy of `DigitNet` in
/// `furnace-burn/examples/serve_burn_model/models.rs` (this crate cannot use the example's):
/// change both together
#[derive(Module, Debug)]
pub struct DigitNet {
    conv: Conv2d,
    pool: AdaptiveAvgPool2d,
    relu: Relu,
    head: Linear,
}

impl DigitNet {
    pub fn new(device: &Device) -> Self {
        Self {
            conv: Conv2dConfig::new([1, 8], [3, 3]).init(device),
            pool: AdaptiveAvgPool2dConfig::new([4, 4]).init(),
            relu: Relu::new(),
            head: LinearConfig::new(8 * 4 * 4, 10).init(device),
        }
    }

    pub fn forward(&self, images: Tensor<4>) -> Tensor<2> {
        let x = self.relu.forward(self.conv.forward(images));
        let x = self.pool.forward(x).flatten::<2>(1, 3);
        self.head.forward(x)
    }
}

fn conv(c_in: usize, c_out: usize, kernel: usize, stride: usize, device: &Device) -> Conv2d {
    let pad = kernel / 2;
    Conv2dConfig::new([c_in, c_out], [kernel, kernel])
        .with_stride([stride, stride])
        .with_padding(PaddingConfig2d::Explicit(pad, pad, pad, pad))
        .with_bias(false)
        .init(device)
}

/// ResNet's basic block: two 3x3 convolutions and a shortcut
#[derive(Module, Debug)]
struct Block {
    conv1: Conv2d,
    bn1: BatchNorm,
    conv2: Conv2d,
    bn2: BatchNorm,
    down_conv: Option<Conv2d>,
    down_bn: Option<BatchNorm>,
    relu: Relu,
}

impl Block {
    fn new(c_in: usize, c_out: usize, stride: usize, device: &Device) -> Self {
        let down = stride != 1 || c_in != c_out;
        Self {
            conv1: conv(c_in, c_out, 3, stride, device),
            bn1: BatchNormConfig::new(c_out).init(device),
            conv2: conv(c_out, c_out, 3, 1, device),
            bn2: BatchNormConfig::new(c_out).init(device),
            down_conv: down.then(|| conv(c_in, c_out, 1, stride, device)),
            down_bn: down.then(|| BatchNormConfig::new(c_out).init(device)),
            relu: Relu::new(),
        }
    }

    fn forward(&self, x: Tensor<4>) -> Tensor<4> {
        let shortcut = match (&self.down_conv, &self.down_bn) {
            (Some(conv), Some(bn)) => bn.forward(conv.forward(x.clone())),
            _ => x.clone(),
        };
        let y = self.relu.forward(self.bn1.forward(self.conv1.forward(x)));
        let y = self.bn2.forward(self.conv2.forward(y));
        self.relu.forward(y + shortcut)
    }
}

/// ResNet-18: `[batch, 3, 224, 224]` -> `[batch, 1000]`
#[derive(Module, Debug)]
pub struct ResNet18 {
    stem: Conv2d,
    stem_bn: BatchNorm,
    relu: Relu,
    max_pool: MaxPool2d,
    blocks: Vec<Block>,
    avg_pool: AdaptiveAvgPool2d,
    fc: Linear,
}

impl ResNet18 {
    pub fn new(device: &Device) -> Self {
        let stages = [
            (64, 64, 1),
            (64, 64, 1),
            (64, 128, 2),
            (128, 128, 1),
            (128, 256, 2),
            (256, 256, 1),
            (256, 512, 2),
            (512, 512, 1),
        ];
        Self {
            stem: conv(3, 64, 7, 2, device),
            stem_bn: BatchNormConfig::new(64).init(device),
            relu: Relu::new(),
            max_pool: MaxPool2dConfig::new([3, 3])
                .with_strides([2, 2])
                .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
                .init(),
            blocks: stages
                .iter()
                .map(|&(c_in, c_out, stride)| Block::new(c_in, c_out, stride, device))
                .collect(),
            avg_pool: AdaptiveAvgPool2dConfig::new([1, 1]).init(),
            fc: LinearConfig::new(512, 1000).init(device),
        }
    }

    pub fn forward(&self, images: Tensor<4>) -> Tensor<2> {
        let x = self
            .relu
            .forward(self.stem_bn.forward(self.stem.forward(images)));
        let mut x = self.max_pool.forward(x);
        for block in &self.blocks {
            x = block.forward(x);
        }
        self.fc.forward(self.avg_pool.forward(x).flatten::<2>(1, 3))
    }
}

/// A small BERT-like text classifier: token and position embeddings, a 4-layer transformer
/// encoder (width 256, 4 heads), mean pooling under the mask, a linear head.
/// `[batch, seq]` ids and mask -> `[batch, 2]` logits
#[derive(Module, Debug)]
pub struct TextEncoder {
    tokens: Embedding,
    positions: Embedding,
    encoder: TransformerEncoder,
    head: Linear,
}

impl TextEncoder {
    pub const VOCAB: usize = 30_522;
    pub const SEQ: usize = 128;
    const WIDTH: usize = 256;

    pub fn new(device: &Device) -> Self {
        Self {
            tokens: EmbeddingConfig::new(Self::VOCAB, Self::WIDTH).init(device),
            positions: EmbeddingConfig::new(Self::SEQ, Self::WIDTH).init(device),
            encoder: TransformerEncoderConfig::new(Self::WIDTH, 4 * Self::WIDTH, 4, 4).init(device),
            head: LinearConfig::new(Self::WIDTH, 2).init(device),
        }
    }

    pub fn forward(&self, ids: Tensor<2, Int>, mask: Tensor<2, Int>) -> Tensor<2> {
        let [batch, seq] = ids.dims();
        let device = ids.device();
        let positions = Tensor::<1, Int>::arange(0..seq as i64, &device)
            .unsqueeze::<2>()
            .repeat_dim(0, batch);
        let x = self.tokens.forward(ids) + self.positions.forward(positions);
        let padding = mask.clone().equal_elem(0);
        let hidden = self
            .encoder
            .forward(TransformerEncoderInput::new(x).mask_pad(padding));
        let mask = mask.float().unsqueeze_dim::<3>(2); // [batch, seq, 1]
        let summed = (hidden * mask.clone()).sum_dim(1).squeeze_dim::<2>(1);
        let count = mask.sum_dim(1).squeeze_dim::<2>(1).clamp_min(1.0);
        self.head.forward(summed / count)
    }
}
