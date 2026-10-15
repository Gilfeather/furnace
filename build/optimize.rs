// Build-time ONNX graph optimizations applied before burn-onnx code generation.
//
// burn-onnx 0.22 emits one Burn op per ONNX node and does not fold inference-mode
// BatchNormalization into the preceding convolution, so a ResNet runs ~20 extra
// elementwise passes over its activations. `fold_conv_batchnorm` rewrites
//
//   y = BN(Conv(x, W, b))  with  BN(z) = gamma * (z - mean) / sqrt(var + eps) + beta
//
// into a single `Conv(x, W', b')` where, per output channel c,
//
//   s_c   = gamma_c / sqrt(var_c + eps)
//   W'_c  = W_c * s_c
//   b'_c  = (b_c - mean_c) * s_c + beta_c        (b_c = 0 when the Conv has no bias)
//
// Shared by build.rs and the crate's tests (via `#[path]`), so it only uses onnx-ir's
// re-exported protobuf types.

use onnx_ir::{ModelProto, NodeProto, TensorProto};
use std::collections::{HashMap, HashSet};

/// ONNX `TensorProto.DataType.FLOAT`
const FLOAT: i32 = 1;

/// What `fold_conv_batchnorm` did, for build logs
#[derive(Debug, Default, PartialEq)]
pub struct FoldReport {
    pub folded: usize,
    /// `(BatchNormalization node name, reason)` for pairs left alone
    pub skipped: Vec<(String, String)>,
}

/// Fold every eligible `Conv -> BatchNormalization` pair in `model` in place.
pub fn fold_conv_batchnorm(model: &mut ModelProto) -> FoldReport {
    let mut report = FoldReport::default();
    let graph = model.graph.mut_or_insert_default();

    // How many node inputs read each value (initializer sharing and fan-out checks)
    let mut readers: HashMap<String, usize> = HashMap::new();
    for node in &graph.node {
        for input in node.input.iter().filter(|i| !i.is_empty()) {
            *readers.entry(input.clone()).or_default() += 1;
        }
    }
    let graph_outputs: HashSet<String> = graph.output.iter().map(|o| o.name.clone()).collect();
    let producer: HashMap<String, usize> = graph
        .node
        .iter()
        .enumerate()
        .flat_map(|(i, n)| n.output.iter().map(move |o| (o.clone(), i)))
        .collect();
    let init_index: HashMap<String, usize> = graph
        .initializer
        .iter()
        .enumerate()
        .map(|(i, t)| (t.name.clone(), i))
        .collect();

    // Plan every fold against the unmodified graph, then apply them
    let mut plans = Vec::new();
    for (bn_idx, bn) in graph.node.iter().enumerate() {
        if bn.op_type != "BatchNormalization" || !bn.domain.is_empty() {
            continue;
        }
        let label = if bn.name.is_empty() {
            bn.output.first().cloned().unwrap_or_default()
        } else {
            bn.name.clone()
        };
        match plan_fold(
            bn,
            &graph.node,
            &producer,
            &readers,
            &graph_outputs,
            &init_index,
            &graph.initializer,
        ) {
            Ok(plan) => plans.push((bn_idx, plan)),
            Err(reason) => report.skipped.push((label, reason)),
        }
    }

    let mut remove_nodes = HashSet::new();
    let mut remove_inits = HashSet::new();
    for (bn_idx, plan) in plans {
        remove_nodes.insert(bn_idx);
        remove_inits.extend(plan.dropped_inits.iter().cloned());
        set_floats(
            &mut graph.initializer[init_index[&plan.weight_name]],
            &plan.new_weight,
        );
        set_floats(
            &mut graph.initializer[init_index[&plan.bias_name]],
            &plan.new_bias,
        );
        let conv = &mut graph.node[plan.conv_idx];
        if conv.input.len() < 3 {
            conv.input.resize(3, String::new());
        }
        conv.input[2] = plan.bias_name;
        conv.output[0] = plan.new_output;
        report.folded += 1;
    }

    // Drop folded BN nodes and the BN parameters nothing reads any more
    let mut idx = 0;
    graph.node.retain(|_| {
        let keep = !remove_nodes.contains(&idx);
        idx += 1;
        keep
    });
    graph
        .initializer
        .retain(|t| !remove_inits.contains(&t.name));
    // IR < 4 models also list initializers as graph inputs
    graph.input.retain(|i| !remove_inits.contains(&i.name));
    // Intermediate Conv outputs that no longer exist
    let live: HashSet<String> = graph
        .node
        .iter()
        .flat_map(|n| n.output.iter().cloned())
        .collect();
    graph
        .value_info
        .retain(|v| live.contains(&v.name) || graph_outputs.contains(&v.name));

    report
}

struct FoldPlan {
    conv_idx: usize,
    weight_name: String,
    /// Initializer that becomes the Conv bias (the existing bias, or BN's beta reused)
    bias_name: String,
    new_weight: Vec<f32>,
    new_bias: Vec<f32>,
    new_output: String,
    dropped_inits: Vec<String>,
}

#[allow(clippy::too_many_arguments)]
fn plan_fold(
    bn: &NodeProto,
    nodes: &[NodeProto],
    producer: &HashMap<String, usize>,
    readers: &HashMap<String, usize>,
    graph_outputs: &HashSet<String>,
    init_index: &HashMap<String, usize>,
    inits: &[TensorProto],
) -> Result<FoldPlan, String> {
    if bn.input.len() != 5 {
        return Err(format!("expected 5 inputs, found {}", bn.input.len()));
    }
    if bn.output.iter().skip(1).any(|o| !o.is_empty()) {
        return Err("training-mode outputs are used".to_string());
    }
    let x = &bn.input[0];
    let &conv_idx = producer.get(x).ok_or("input is not produced by a node")?;
    let conv = &nodes[conv_idx];
    if conv.op_type != "Conv" || !conv.domain.is_empty() {
        return Err(format!("input comes from {}, not Conv", conv.op_type));
    }
    if readers.get(x).copied().unwrap_or(0) != 1 || graph_outputs.contains(x) {
        return Err("Conv output is used elsewhere".to_string());
    }
    if conv.input.len() < 2 {
        return Err("Conv has no weight input".to_string());
    }

    let tensor = |name: &str| -> Result<&TensorProto, String> {
        init_index
            .get(name)
            .map(|&i| &inits[i])
            .ok_or_else(|| format!("'{name}' is not an initializer"))
    };
    let unshared = |name: &str| -> Result<(), String> {
        match readers.get(name).copied().unwrap_or(0) {
            1 => Ok(()),
            n => Err(format!("'{name}' is read by {n} nodes")),
        }
    };

    let weight_name = conv.input[1].clone();
    let weight = tensor(&weight_name)?;
    unshared(&weight_name)?;
    let out_channels = *weight.dims.first().ok_or("Conv weight has no dims")? as usize;

    let [gamma, beta, mean, var] = [1, 2, 3, 4].map(|i| bn.input[i].as_str());
    let gamma_v = floats(tensor(gamma)?)?;
    let beta_v = floats(tensor(beta)?)?;
    let mean_v = floats(tensor(mean)?)?;
    let var_v = floats(tensor(var)?)?;
    for (name, v) in [
        (gamma, &gamma_v),
        (beta, &beta_v),
        (mean, &mean_v),
        (var, &var_v),
    ] {
        if v.len() != out_channels {
            return Err(format!(
                "'{name}' has {} values for {out_channels} output channels",
                v.len()
            ));
        }
    }
    let epsilon = bn
        .attribute
        .iter()
        .find(|a| a.name == "epsilon")
        .map_or(1e-5, |a| a.f);

    let w = floats(weight)?;
    if w.is_empty() || !w.len().is_multiple_of(out_channels) {
        return Err("Conv weight size is not a multiple of its output channels".to_string());
    }
    let per_channel = w.len() / out_channels;

    let has_bias = conv.input.get(2).is_some_and(|b| !b.is_empty());
    let (bias_name, bias_v) = if has_bias {
        let name = conv.input[2].clone();
        unshared(&name)?;
        let v = floats(tensor(&name)?)?;
        if v.len() != out_channels {
            return Err("Conv bias size does not match its output channels".to_string());
        }
        (name, v)
    } else {
        // Reuse beta's initializer (already shaped [C]) as the new bias
        unshared(beta)?;
        (beta.to_string(), vec![0.0; out_channels])
    };

    let scale: Vec<f32> = (0..out_channels)
        .map(|c| gamma_v[c] / (var_v[c] + epsilon).sqrt())
        .collect();
    let new_weight: Vec<f32> = w
        .iter()
        .enumerate()
        .map(|(i, v)| v * scale[i / per_channel])
        .collect();
    let new_bias: Vec<f32> = (0..out_channels)
        .map(|c| (bias_v[c] - mean_v[c]) * scale[c] + beta_v[c])
        .collect();

    let mut dropped_inits = Vec::new();
    for name in [gamma, mean, var] {
        unshared(name)?;
        dropped_inits.push(name.to_string());
    }
    if has_bias {
        unshared(beta)?;
        dropped_inits.push(beta.to_string());
    }

    Ok(FoldPlan {
        conv_idx,
        weight_name,
        bias_name,
        new_weight,
        new_bias,
        new_output: bn.output[0].clone(),
        dropped_inits,
    })
}

/// Values of a FLOAT initializer stored either as `float_data` or little-endian `raw_data`
pub fn floats(t: &TensorProto) -> Result<Vec<f32>, String> {
    if t.data_type != FLOAT {
        return Err(format!(
            "'{}' is not float32 (data type {})",
            t.name, t.data_type
        ));
    }
    if !t.external_data.is_empty() {
        return Err(format!("'{}' uses external data", t.name));
    }
    if !t.float_data.is_empty() {
        return Ok(t.float_data.clone());
    }
    let (chunks, rest) = t.raw_data.as_chunks::<4>();
    if !rest.is_empty() {
        return Err(format!(
            "'{}' raw data is not a whole number of f32",
            t.name
        ));
    }
    Ok(chunks.iter().map(|b| f32::from_le_bytes(*b)).collect())
}

fn set_floats(t: &mut TensorProto, values: &[f32]) {
    t.float_data.clear();
    t.raw_data = values
        .iter()
        .flat_map(|v| v.to_le_bytes())
        .collect::<Vec<u8>>()
        .into();
}
