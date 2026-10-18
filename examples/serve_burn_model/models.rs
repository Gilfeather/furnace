//! The Burn models served by the `serve_burn_model` example, shared with its integration test
//! (tests/serve_burn_model.rs)

use burn::module::Module;
use burn::nn::conv::{Conv2d, Conv2dConfig};
use burn::nn::pool::{AdaptiveAvgPool2d, AdaptiveAvgPool2dConfig};
use burn::nn::{Embedding, EmbeddingConfig, Linear, LinearConfig, Relu};
use burn::tensor::{Device, Int, Tensor};
use furnace::{BackendKind, BurnModel, DataType, HostTensor, IoSpec};

/// A small CNN: `[batch, 1, 28, 28]` -> `[batch, 10]` logits
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

/// Mean of the token embeddings under the mask, then a linear layer: `[batch, seq]` ids and mask
/// -> `[batch, 2]` logits
#[derive(Module, Debug)]
pub struct SentimentNet {
    embedding: Embedding,
    head: Linear,
}

impl SentimentNet {
    pub const VOCAB: usize = 30_522;
    const DIM: usize = 16;

    pub fn new(device: &Device) -> Self {
        Self {
            embedding: EmbeddingConfig::new(Self::VOCAB, Self::DIM).init(device),
            head: LinearConfig::new(Self::DIM, 2).init(device),
        }
    }

    pub fn forward(&self, ids: Tensor<2, Int>, mask: Tensor<2, Int>) -> Tensor<2> {
        let embedded = self.embedding.forward(ids); // [batch, seq, dim]
        let mask = mask.float().unsqueeze_dim::<3>(2); // [batch, seq, 1]
        let summed = (embedded * mask.clone()).sum_dim(1).squeeze_dim::<2>(1);
        let count = mask.sum_dim(1).squeeze_dim::<2>(1).clamp_min(1.0);
        self.head.forward(summed / count)
    }
}

/// `SentimentNet` served with named tensors
#[derive(Debug)]
pub struct Sentiment {
    pub net: SentimentNet,
    pub device: Device,
    pub seq_len: usize,
}

impl BurnModel for Sentiment {
    fn name(&self) -> &str {
        "sentiment"
    }

    fn backend(&self) -> BackendKind {
        BackendKind::Cpu
    }

    fn inputs(&self) -> Vec<IoSpec> {
        let seq = [Some(self.seq_len)];
        vec![
            IoSpec::new("input_ids", DataType::Int64, &seq),
            IoSpec::new("attention_mask", DataType::Int64, &seq),
        ]
    }

    fn outputs(&self) -> Vec<IoSpec> {
        vec![IoSpec::fp32("logits", &[2])]
    }

    fn run(&self, inputs: Vec<HostTensor>) -> furnace::Result<Vec<HostTensor>> {
        // The server has checked the names, types and shapes, and orders the inputs as above
        let mut inputs = inputs.into_iter();
        let (Some(ids), Some(mask)) = (inputs.next(), inputs.next()) else {
            unreachable!("checked against inputs()");
        };
        let logits = self.net.forward(
            ids.into_int::<2>(&self.device)?,
            mask.into_int::<2>(&self.device)?,
        );
        Ok(vec![HostTensor::from_float(logits)?])
    }
}
