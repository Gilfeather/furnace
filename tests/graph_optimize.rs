//! Tests for the build-time ONNX graph optimizations in build/optimize.rs.

#[path = "../build/optimize.rs"]
mod optimize;

use onnx_ir::{AttributeProto, ModelProto, NodeProto, TensorProto};
use optimize::{floats, fold_conv_batchnorm};

const FLOAT: i32 = 1;

fn tensor(name: &str, dims: &[i64], values: &[f32]) -> TensorProto {
    let mut t = TensorProto::new();
    t.name = name.to_string();
    t.dims = dims.to_vec();
    t.data_type = FLOAT;
    t.float_data = values.to_vec();
    t
}

fn node(op: &str, inputs: &[&str], outputs: &[&str]) -> NodeProto {
    let mut n = NodeProto::new();
    n.op_type = op.to_string();
    n.name = format!("{op}_{}", outputs[0]);
    n.input = inputs.iter().map(|s| s.to_string()).collect();
    n.output = outputs.iter().map(|s| s.to_string()).collect();
    n
}

const GAMMA: [f32; 2] = [1.5, -0.5];
const BETA: [f32; 2] = [0.25, 2.0];
const MEAN: [f32; 2] = [0.1, -0.3];
const VAR: [f32; 2] = [0.8, 2.5];
const EPS: f32 = 1e-3;
// 1x1 conv, 2 input channels -> 2 output channels
const W: [f32; 4] = [0.2, -1.0, 0.7, 0.4];
const B: [f32; 2] = [0.05, -0.2];

/// Conv(x, W[, B]) -> BatchNormalization -> y
fn conv_bn_model(with_bias: bool) -> ModelProto {
    let mut model = ModelProto::new();
    let graph = model.graph.mut_or_insert_default();
    let conv_inputs: &[&str] = if with_bias {
        &["x", "w", "b"]
    } else {
        &["x", "w"]
    };
    graph.node.push(node("Conv", conv_inputs, &["conv_out"]));
    let mut bn = node(
        "BatchNormalization",
        &["conv_out", "gamma", "beta", "mean", "var"],
        &["y"],
    );
    let mut eps = AttributeProto::new();
    eps.name = "epsilon".to_string();
    eps.f = EPS;
    bn.attribute.push(eps);
    graph.node.push(bn);

    graph.initializer.push(tensor("w", &[2, 2, 1, 1], &W));
    if with_bias {
        graph.initializer.push(tensor("b", &[2], &B));
    }
    graph.initializer.push(tensor("gamma", &[2], &GAMMA));
    graph.initializer.push(tensor("beta", &[2], &BETA));
    graph.initializer.push(tensor("mean", &[2], &MEAN));
    graph.initializer.push(tensor("var", &[2], &VAR));
    model
}

/// Reference: BN(Conv(x)) for one pixel with input channels `x`
fn reference(x: [f32; 2], with_bias: bool) -> [f32; 2] {
    std::array::from_fn(|c| {
        let conv = W[c * 2] * x[0] + W[c * 2 + 1] * x[1] + if with_bias { B[c] } else { 0.0 };
        GAMMA[c] * (conv - MEAN[c]) / (VAR[c] + EPS).sqrt() + BETA[c]
    })
}

fn init<'a>(model: &'a ModelProto, name: &str) -> &'a TensorProto {
    model
        .graph
        .initializer
        .iter()
        .find(|t| t.name == name)
        .unwrap_or_else(|| panic!("initializer '{name}' missing"))
}

#[test]
fn folds_batchnorm_into_conv() {
    for with_bias in [false, true] {
        let mut model = conv_bn_model(with_bias);
        let report = fold_conv_batchnorm(&mut model);
        assert_eq!(report.folded, 1, "with_bias={with_bias}");
        assert!(report.skipped.is_empty());

        let graph = &model.graph;
        assert_eq!(graph.node.len(), 1);
        let conv = &graph.node[0];
        assert_eq!(conv.op_type, "Conv");
        assert_eq!(conv.output, vec!["y".to_string()]);
        assert_eq!(conv.input.len(), 3);

        // BN parameters that are no longer read are gone
        let names: Vec<&str> = graph.initializer.iter().map(|t| t.name.as_str()).collect();
        for gone in ["gamma", "mean", "var"] {
            assert!(
                !names.contains(&gone),
                "{gone} should be removed: {names:?}"
            );
        }

        let w = floats(init(&model, &conv.input[1])).unwrap();
        let b = floats(init(&model, &conv.input[2])).unwrap();
        for x in [[1.0, 0.0], [0.0, 1.0], [0.3, -2.0], [4.0, 1.5]] {
            let expected = reference(x, with_bias);
            for c in 0..2 {
                let folded = w[c * 2] * x[0] + w[c * 2 + 1] * x[1] + b[c];
                assert!(
                    (folded - expected[c]).abs() < 1e-5,
                    "with_bias={with_bias} x={x:?} c={c}: {folded} != {}",
                    expected[c]
                );
            }
        }
    }
}

#[test]
fn reads_raw_data_initializers() {
    let mut model = conv_bn_model(false);
    // Store the weight as little-endian raw bytes instead of float_data
    let w = model
        .graph
        .mut_or_insert_default()
        .initializer
        .iter_mut()
        .find(|t| t.name == "w")
        .unwrap();
    w.raw_data = W
        .iter()
        .flat_map(|v| v.to_le_bytes())
        .collect::<Vec<u8>>()
        .into();
    w.float_data.clear();

    assert_eq!(fold_conv_batchnorm(&mut model).folded, 1);
}

#[test]
fn keeps_batchnorm_when_conv_output_is_shared() {
    let mut model = conv_bn_model(false);
    // Another node also reads the Conv output, so it can't be rewritten
    model
        .graph
        .mut_or_insert_default()
        .node
        .push(node("Relu", &["conv_out"], &["other"]));

    let report = fold_conv_batchnorm(&mut model);
    assert_eq!(report.folded, 0);
    assert_eq!(report.skipped.len(), 1);
    assert_eq!(model.graph.node.len(), 3);
}

#[test]
fn keeps_batchnorm_not_preceded_by_conv() {
    let mut model = ModelProto::new();
    let graph = model.graph.mut_or_insert_default();
    graph.node.push(node(
        "BatchNormalization",
        &["x", "gamma", "beta", "mean", "var"],
        &["y"],
    ));
    for (name, values) in [
        ("gamma", GAMMA),
        ("beta", BETA),
        ("mean", MEAN),
        ("var", VAR),
    ] {
        graph.initializer.push(tensor(name, &[2], &values));
    }

    let report = fold_conv_batchnorm(&mut model);
    assert_eq!(report.folded, 0);
    assert_eq!(model.graph.node.len(), 1);
}
