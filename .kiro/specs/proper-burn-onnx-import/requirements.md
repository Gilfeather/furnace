# Requirements Document

## Introduction

現在の実装はBurnの公式ONNX importドキュメントに準拠していません。正しいBurnのアプローチに従って、ONNXモデルからRustコードを生成し、生成されたモデルを適切に使用する機能を実装します。

## Requirements

### Requirement 1

**User Story:** 開発者として、Burnの公式ドキュメントに従ってONNXモデルを正しくインポートしたい。これにより、標準的なBurnのワークフローでモデルを使用できる。

#### Acceptance Criteria

1. WHEN build.rsでModelGenを使用する THEN ONNXファイルからRustコードが生成される SHALL
2. WHEN 生成されたコードをinclude!マクロで取り込む THEN モジュールとして使用可能になる SHALL
3. WHEN Model::<Backend>::default()を使用する THEN 生成されたモデルが正しく初期化される SHALL
4. WHEN model.forward()を呼び出す THEN 正しい推論結果が返される SHALL

### Requirement 2

**User Story:** システム管理者として、組み込みモデル名を指定してサーバーを起動したい。これにより、外部ファイルに依存せずにモデルを使用できる。

#### Acceptance Criteria

1. WHEN --model-name resnet18を指定する THEN 生成されたResNet18モデルが使用される SHALL
2. WHEN 生成されたモデルが利用可能でない THEN 適切なエラーメッセージが表示される SHALL
3. WHEN モデル読み込みが失敗する THEN ダミーモデルにフォールバックする SHALL
4. WHEN サーバーが起動する THEN APIエンドポイントが正常に動作する SHALL

### Requirement 3

**User Story:** 開発者として、生成されたモデルの推論APIが正しく動作することを確認したい。これにより、実際のアプリケーションで使用できる。

#### Acceptance Criteria

1. WHEN /model/info APIを呼び出す THEN 正しいモデル情報が返される SHALL
2. WHEN /predict APIを呼び出す THEN 正しい推論結果が返される SHALL
3. WHEN 入力形状が正しい THEN 推論が成功する SHALL
4. WHEN 入力形状が間違っている THEN 適切なエラーが返される SHALL

### Requirement 4

**User Story:** 開発者として、既存のBurnモデルとダミーモデルの動作が維持されることを確認したい。これにより、既存の機能が破損しない。

#### Acceptance Criteria

1. WHEN Burnモデル（.mpk/.json）を読み込む THEN 既存の動作が維持される SHALL
2. WHEN ダミーモデルを使用する THEN 既存の動作が維持される SHALL
3. WHEN 既存のAPIエンドポイント THEN 互換性が保たれる SHALL
4. WHEN 既存のテスト THEN 正常に動作する SHALL