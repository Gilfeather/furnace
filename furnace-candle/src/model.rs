//! The llama family's forward pass over GGUF weights (llama, and qwen2 with its attention
//! biases), as in candle-transformers' `quantized_llama` and `quantized_qwen2`, but with the KV
//! cache outside the model: the weights are shared by every sequence, and each sequence owns
//! its cache, which furnace drops when the request ends.

use candle_core::quantized::{QMatMul, gguf_file};
use candle_core::{DType, Device, IndexOp, Module, Result, Tensor};
use candle_nn::{Embedding, RmsNorm};

/// What differs between the architectures this file serves
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Architecture {
    /// `llama`: Llama, SmolLM, Mistral and the others written as llama
    Llama,
    /// `qwen2`: Qwen2 and Qwen2.5, with biases on the attention projections
    Qwen2,
}

impl Architecture {
    fn from_gguf(name: &str) -> Result<Self> {
        match name {
            "llama" => Ok(Self::Llama),
            "qwen2" => Ok(Self::Qwen2),
            other => candle_core::bail!(
                "GGUF architecture '{other}' is not supported; furnace-candle serves llama and qwen2"
            ),
        }
    }

    /// The metadata key prefix
    fn key(self) -> &'static str {
        match self {
            Self::Llama => "llama",
            Self::Qwen2 => "qwen2",
        }
    }

    /// Rotary embedding on pairs (i, i + d/2) ("NEOX"), or on (2i, 2i + 1)
    fn rope_is_neox(self) -> bool {
        matches!(self, Self::Qwen2)
    }
}

#[derive(Debug)]
struct Layer {
    attention_norm: RmsNorm,
    wq: QMatMul,
    wk: QMatMul,
    wv: QMatMul,
    /// qwen2's attention biases
    biases: Option<[Tensor; 3]>,
    wo: QMatMul,
    ffn_norm: RmsNorm,
    gate: QMatMul,
    up: QMatMul,
    down: QMatMul,
}

/// The weights of a model, shared by all its sequences
#[derive(Debug)]
pub struct Weights {
    /// Which forward pass the file needs
    pub architecture: Architecture,
    embedding: Embedding,
    layers: Vec<Layer>,
    norm: RmsNorm,
    output: QMatMul,
    n_head: usize,
    n_kv_head: usize,
    head_dim: usize,
    /// Rotary tables, `[context, head_dim / 2]`
    cos: Tensor,
    sin: Tensor,
    /// The context the GGUF file was trained for
    pub context_length: usize,
    /// Tokens in the vocabulary: the length of every logits vector
    pub vocab_size: usize,
    device: Device,
}

/// One sequence's KV cache and position
#[derive(Debug)]
pub struct Cache {
    /// `(k, v)` per layer, each `[1, kv_heads, position, head_dim]`
    layers: Vec<Option<(Tensor, Tensor)>>,
    position: usize,
}

impl Cache {
    /// Tokens in the cache
    pub fn position(&self) -> usize {
        self.position
    }
}

impl Weights {
    /// Load the weights of a llama or qwen2 GGUF file, with rotary tables for `max_context`
    /// tokens (at most the trained context)
    pub fn from_gguf<R: std::io::Seek + std::io::Read>(
        content: &gguf_file::Content,
        reader: &mut R,
        max_context: Option<usize>,
        device: &Device,
    ) -> Result<Self> {
        let metadata = |key: &str| match content.metadata.get(key) {
            Some(value) => Ok(value),
            None => candle_core::bail!("the GGUF file has no {key}"),
        };
        let architecture = Architecture::from_gguf(metadata("general.architecture")?.to_string()?)?;
        let key = |name: &str| format!("{}.{name}", architecture.key());
        let usize_of =
            |name: &str| -> Result<usize> { Ok(metadata(&key(name))?.to_u32()? as usize) };
        let experts = metadata(&key("expert_count"))
            .and_then(|v| v.to_u32())
            .unwrap_or(0);
        if experts > 1 {
            candle_core::bail!("mixture-of-experts models are not supported");
        }
        let n_head = usize_of("attention.head_count")?;
        let n_kv_head = usize_of("attention.head_count_kv")?;
        let embedding_length = usize_of("embedding_length")?;
        let block_count = usize_of("block_count")?;
        let context_length = usize_of("context_length")?;
        let eps = f64::from(metadata(&key("attention.layer_norm_rms_epsilon"))?.to_f32()?);
        let freq_base = metadata(&key("rope.freq_base"))
            .and_then(|v| v.to_f32())
            .unwrap_or(10_000.0);
        let head_dim = embedding_length / n_head;
        if let Ok(rope_dim) = usize_of("rope.dimension_count")
            && rope_dim != head_dim
        {
            candle_core::bail!(
                "partial rotary embeddings ({rope_dim} of {head_dim}) are not supported"
            );
        }

        let tensor = |name: &str, reader: &mut R| content.tensor(reader, name, device);
        let norm = |name: &str, reader: &mut R| -> Result<RmsNorm> {
            Ok(RmsNorm::new(tensor(name, reader)?.dequantize(device)?, eps))
        };
        let embedding_q = tensor("token_embd.weight", reader)?;
        let vocab_size = embedding_q.shape().dims()[0];
        let embedding = Embedding::new(embedding_q.dequantize(device)?, embedding_length);
        let output_norm = norm("output_norm.weight", reader)?;
        // Without an output matrix, the embeddings are tied
        let output = match tensor("output.weight", reader) {
            Ok(output) => QMatMul::from_qtensor(output)?,
            Err(_) => QMatMul::from_qtensor(embedding_q)?,
        };
        // Llama 3's long-context frequency factors, when the file has them
        let freq_factors = match tensor("rope_freqs.weight", reader) {
            Ok(factors) => Some(factors.dequantize(device)?.to_vec1::<f32>()?),
            Err(_) => None,
        };

        let mut layers = Vec::with_capacity(block_count);
        for i in 0..block_count {
            let name = |part: &str| format!("blk.{i}.{part}");
            let matmul = |part: &str, reader: &mut R| -> Result<QMatMul> {
                QMatMul::from_qtensor(tensor(&name(part), reader)?)
            };
            let biases = match architecture {
                Architecture::Qwen2 => {
                    let mut bias = |part: &str| tensor(&name(part), reader)?.dequantize(device);
                    Some([
                        bias("attn_q.bias")?,
                        bias("attn_k.bias")?,
                        bias("attn_v.bias")?,
                    ])
                }
                Architecture::Llama => None,
            };
            layers.push(Layer {
                attention_norm: norm(&name("attn_norm.weight"), reader)?,
                wq: matmul("attn_q.weight", reader)?,
                wk: matmul("attn_k.weight", reader)?,
                wv: matmul("attn_v.weight", reader)?,
                biases,
                wo: matmul("attn_output.weight", reader)?,
                ffn_norm: norm(&name("ffn_norm.weight"), reader)?,
                gate: matmul("ffn_gate.weight", reader)?,
                up: matmul("ffn_up.weight", reader)?,
                down: matmul("ffn_down.weight", reader)?,
            });
        }

        let context = max_context.map_or(context_length, |m| m.min(context_length));
        let (cos, sin) = rotary_tables(
            head_dim,
            freq_base,
            freq_factors.as_deref(),
            context,
            device,
        )?;
        Ok(Self {
            architecture,
            embedding,
            layers,
            norm: output_norm,
            output,
            n_head,
            n_kv_head,
            head_dim,
            cos,
            sin,
            context_length,
            vocab_size,
            device: device.clone(),
        })
    }

    /// Tokens the rotary tables cover: the most a sequence can hold
    pub fn max_context(&self) -> usize {
        self.cos.dims()[0]
    }

    /// An empty KV cache for a new sequence
    pub fn new_cache(&self) -> Cache {
        Cache {
            layers: vec![None; self.layers.len()],
            position: 0,
        }
    }

    /// Append `tokens` to the sequence; the logits after the last of them
    pub fn forward(&self, cache: &mut Cache, tokens: &[u32]) -> Result<Vec<f32>> {
        let seq_len = tokens.len();
        let position = cache.position;
        if position + seq_len > self.max_context() {
            candle_core::bail!(
                "{} tokens exceed the context of {}",
                position + seq_len,
                self.max_context()
            );
        }
        let input = Tensor::new(tokens, &self.device)?.unsqueeze(0)?;
        let mask = (seq_len > 1)
            .then(|| causal_mask(seq_len, position, &self.device))
            .transpose()?;
        let mut hidden = self.embedding.forward(&input)?;
        for (layer, kv) in self.layers.iter().zip(&mut cache.layers) {
            let residual = &hidden;
            let x = layer.attention_norm.forward(residual)?;
            let x = (self.attention(layer, &x, mask.as_ref(), position, kv)? + residual)?;
            let residual = &x;
            let y = layer.ffn_norm.forward(residual)?;
            let y = candle_nn::ops::silu(&layer.gate.forward(&y)?)? * layer.up.forward(&y)?;
            hidden = (layer.down.forward(&y?)? + residual)?;
        }
        let hidden = self.norm.forward(&hidden)?.i((.., seq_len - 1, ..))?;
        let logits = self.output.forward(&hidden)?.squeeze(0)?;
        cache.position += seq_len;
        logits.to_dtype(DType::F32)?.to_vec1()
    }

    fn attention(
        &self,
        layer: &Layer,
        x: &Tensor,
        mask: Option<&Tensor>,
        position: usize,
        kv: &mut Option<(Tensor, Tensor)>,
    ) -> Result<Tensor> {
        let (batch, seq_len, n_embd) = x.dims3()?;
        let mut q = layer.wq.forward(x)?;
        let mut k = layer.wk.forward(x)?;
        let mut v = layer.wv.forward(x)?;
        if let Some([bq, bk, bv]) = &layer.biases {
            q = q.broadcast_add(bq)?;
            k = k.broadcast_add(bk)?;
            v = v.broadcast_add(bv)?;
        }
        let heads = |t: Tensor, n: usize| -> Result<Tensor> {
            t.reshape((batch, seq_len, n, self.head_dim))?
                .transpose(1, 2)?
                .contiguous()
        };
        let q = self.rotate(&heads(q, self.n_head)?, position)?;
        let k = self.rotate(&heads(k, self.n_kv_head)?, position)?;
        let v = heads(v, self.n_kv_head)?;
        let (k, v) = match kv.take() {
            Some((k_cache, v_cache)) => (
                Tensor::cat(&[&k_cache, &k], 2)?,
                Tensor::cat(&[&v_cache, &v], 2)?,
            ),
            None => (k, v),
        };
        *kv = Some((k.clone(), v.clone()));

        let repeats = self.n_head / self.n_kv_head;
        let k = repeat_kv(k, repeats)?;
        let v = repeat_kv(v, repeats)?;
        let scores = (q.matmul(&k.t()?)? / (self.head_dim as f64).sqrt())?;
        let scores = match mask {
            Some(mask) => {
                let mask = mask.broadcast_as(scores.shape())?;
                let blocked = Tensor::new(f32::NEG_INFINITY, &self.device)?
                    .broadcast_as(scores.shape().dims())?;
                mask.where_cond(&blocked, &scores)?
            }
            None => scores,
        };
        let weights = candle_nn::ops::softmax_last_dim(&scores)?;
        let y = weights.matmul(&v.contiguous()?)?;
        let y = y.transpose(1, 2)?.reshape((batch, seq_len, n_embd))?;
        layer.wo.forward(&y)
    }

    fn rotate(&self, x: &Tensor, position: usize) -> Result<Tensor> {
        let seq_len = x.dim(2)?;
        let cos = self.cos.narrow(0, position, seq_len)?;
        let sin = self.sin.narrow(0, position, seq_len)?;
        if self.architecture.rope_is_neox() {
            candle_nn::rotary_emb::rope(x, &cos, &sin)
        } else {
            candle_nn::rotary_emb::rope_i(x, &cos, &sin)
        }
    }
}

/// `cos` and `sin` of every position's rotation angles, `[context, head_dim / 2]`
fn rotary_tables(
    head_dim: usize,
    freq_base: f32,
    freq_factors: Option<&[f32]>,
    context: usize,
    device: &Device,
) -> Result<(Tensor, Tensor)> {
    let theta: Vec<f32> = (0..head_dim)
        .step_by(2)
        .enumerate()
        .map(|(j, i)| {
            let theta = 1.0 / freq_base.powf(i as f32 / head_dim as f32);
            theta / freq_factors.and_then(|f| f.get(j)).copied().unwrap_or(1.0)
        })
        .collect();
    let theta = Tensor::new(theta.as_slice(), device)?;
    let angles = Tensor::arange(0, context as u32, device)?
        .to_dtype(DType::F32)?
        .reshape((context, 1))?
        .matmul(&theta.reshape((1, theta.elem_count()))?)?;
    Ok((angles.cos()?, angles.sin()?))
}

/// 1 where a query may not look: keys after its own position. `[seq_len, position + seq_len]`
fn causal_mask(seq_len: usize, position: usize, device: &Device) -> Result<Tensor> {
    let total = position + seq_len;
    let mask: Vec<u8> = (0..seq_len)
        .flat_map(|i| (0..total).map(move |j| u8::from(j > position + i)))
        .collect();
    Tensor::from_slice(&mask, (seq_len, total), device)
}

/// Grouped-query attention: each KV head serves `repeats` query heads
fn repeat_kv(x: Tensor, repeats: usize) -> Result<Tensor> {
    if repeats == 1 {
        return Ok(x);
    }
    let (batch, kv_heads, seq_len, head_dim) = x.dims4()?;
    Tensor::cat(&vec![&x; repeats], 2)?.reshape((batch, kv_heads * repeats, seq_len, head_dim))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_mask_blocks_only_later_keys() {
        let mask = causal_mask(3, 2, &Device::Cpu).unwrap();
        let values: Vec<u8> = mask.flatten_all().unwrap().to_vec1().unwrap();
        #[rustfmt::skip]
        assert_eq!(values, [
            0, 0, 0, 1, 1,
            0, 0, 0, 0, 1,
            0, 0, 0, 0, 0,
        ]);
    }

    #[test]
    fn frequency_factors_slow_down_rotations() {
        let (plain, _) = rotary_tables(4, 10_000.0, None, 2, &Device::Cpu).unwrap();
        let (scaled, _) = rotary_tables(4, 10_000.0, Some(&[1.0, 4.0]), 2, &Device::Cpu).unwrap();
        let plain: Vec<Vec<f32>> = plain.to_vec2().unwrap();
        let scaled: Vec<Vec<f32>> = scaled.to_vec2().unwrap();
        assert_eq!(plain[1][0], scaled[1][0]);
        let angle = 1.0f32 / 100.0;
        assert!((plain[1][1] - angle.cos()).abs() < 1e-6);
        assert!((scaled[1][1] - (angle / 4.0).cos()).abs() < 1e-6);
    }

    #[test]
    fn grouped_heads_repeat_in_order() {
        let x = Tensor::arange(0f32, 4.0, &Device::Cpu)
            .unwrap()
            .reshape((1, 2, 1, 2))
            .unwrap();
        let repeated: Vec<f32> = repeat_kv(x, 2)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1()
            .unwrap();
        assert_eq!(repeated, [0.0, 1.0, 0.0, 1.0, 2.0, 3.0, 2.0, 3.0]);
    }

    #[test]
    fn an_unknown_architecture_is_refused() {
        assert!(Architecture::from_gguf("mamba").is_err());
    }
}
