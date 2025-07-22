# Design Document

## Overview

この設計では、Burnの公式ドキュメントに従って正しいONNX importを実装します。現在の実装は`.mpk`ファイルを直接読み込もうとしていますが、正しいアプローチは：

1. `build.rs`でONNXファイルからRustコードを生成
2. 生成されたコードを`include!`マクロでモジュールに取り込み
3. `Model::<Backend>::default()`で生成されたモデルを使用

## Architecture

### 現在のアーキテクチャ（問題あり）
```
ONNX → build.rs → .mpk生成 → ファイル読み込み → エラー
```

### 正しいアーキテクチャ
```
ONNX → build.rs → Rustコード生成 → include!マクロ → 生成モデル使用
```

## Components and Interfaces

### 1. build.rs の修正

```rust
use burn_import::onnx::ModelGen;

fn main() {
    // ResNet18モデルの生成
    ModelGen::new()
        .input("models/resnet18.onnx")
        .out_dir("models/")
        .run_from_script();
}
```

### 2. モジュール構造

```rust
// src/models/mod.rs
pub mod resnet18 {
    include!(concat!(env!("OUT_DIR"), "/models/resnet18.rs"));
}
```

### 3. モデル使用方法

```rust
use burn::backend::ndarray::NdArray;
use crate::models::resnet18::Model;

type Backend = NdArray<f32>;

// モデルの初期化
let model: Model<Backend> = Model::default();

// 推論の実行
let input = Tensor::zeros([1, 3, 224, 224], &device);
let output = model.forward(input);
```

### 4. 組み込みモデル管理

```rust
pub enum BuiltInModel {
    ResNet18,
}

impl BuiltInModel {
    pub fn from_name(name: &str) -> Result<Self> {
        match name {
            "resnet18" => Ok(Self::ResNet18),
            _ => Err(ModelError::UnknownModel(name.to_string())),
        }
    }
    
    pub fn create_model(&self) -> Result<Box<dyn BurnModel>> {
        match self {
            Self::ResNet18 => {
                let model = crate::models::resnet18::Model::<Backend>::default();
                Ok(Box::new(GeneratedBurnModel::new(model, "resnet18")))
            }
        }
    }
}
```

## Data Models

### GeneratedBurnModel

```rust
pub struct GeneratedBurnModel<M> {
    model: M,
    name: String,
    input_shape: Vec<usize>,
    output_shape: Vec<usize>,
}

impl<M> GeneratedBurnModel<M> 
where 
    M: Module<Backend> + Send + Sync + std::fmt::Debug,
{
    pub fn new(model: M, name: &str) -> Self {
        Self {
            model,
            name: name.to_string(),
            input_shape: vec![1, 3, 224, 224], // ResNet18の場合
            output_shape: vec![1000],           // ImageNet classes
        }
    }
}

impl<M> BurnModel for GeneratedBurnModel<M> 
where 
    M: Module<Backend> + Send + Sync + std::fmt::Debug,
{
    fn predict(&self, input: Tensor<Backend, 2>) -> Result<Tensor<Backend, 2>> {
        // 2D → 4D変換
        let [batch_size, _] = input.dims();
        let input_4d = input.reshape([batch_size, 3, 224, 224]);
        
        // 推論実行
        let output_4d = self.model.forward(input_4d);
        
        // 4D → 2D変換
        let output_shape = output_4d.dims();
        let output_2d = output_4d.reshape([output_shape[0], output_shape[1]]);
        
        Ok(output_2d)
    }
    
    fn get_input_shape(&self) -> &[usize] {
        &self.input_shape
    }
    
    fn get_output_shape(&self) -> &[usize] {
        &self.output_shape
    }
    
    fn get_name(&self) -> &str {
        &self.name
    }
    
    fn get_backend_info(&self) -> String {
        "burn-generated".to_string()
    }
}
```

## Error Handling

### 新しいエラー型

```rust
#[derive(Debug, thiserror::Error)]
pub enum ModelError {
    #[error("Unknown built-in model: {0}")]
    UnknownModel(String),
    
    #[error("Generated model not available: {0}")]
    GeneratedModelNotAvailable(String),
    
    #[error("Model inference failed: {0}")]
    InferenceFailed(String),
}
```

## Implementation Plan

### Phase 1: build.rs修正
1. 正しいModelGen設定
2. ONNXファイルからRustコード生成

### Phase 2: モジュール構造作成
1. `src/models/mod.rs`作成
2. 生成されたコードの取り込み

### Phase 3: モデル統合
1. `GeneratedBurnModel`実装
2. `BuiltInModel`enum実装
3. 既存コードとの統合

### Phase 4: テストと検証
1. 推論APIのテスト
2. エラーハンドリングのテスト
3. 既存機能の互換性確認

## Testing Strategy

### 単体テスト
- 生成されたモデルの初期化
- 推論の実行
- エラーハンドリング

### 統合テスト
- APIエンドポイントのテスト
- モデル切り替えのテスト
- フォールバック動作のテスト

### 実際のモデルテスト
- ResNet18での画像分類
- 入力形状の検証
- 出力形状の検証