//! The text side of an LLM for [`Scheduled`](super::Scheduled): the chat template, the
//! tokenizer, end-of-sequence tokens and incremental detokenization

use std::fmt;
use std::path::Path;

use minijinja::{Environment, ErrorKind};
pub use tokenizers::Tokenizer;

use super::{BoxError, EngineError, Input, Message};

const CHAT_TEMPLATE: &str = "chat";

/// A tokenizer with what it takes to turn requests into prompt tokens and tokens back into text
pub struct TextPipeline {
    tokenizer: Tokenizer,
    /// The model's chat template, as Hugging Face's `chat_template` (Jinja), compiled into an
    /// environment that holds it alone
    chat_template: Option<Environment<'static>>,
    bos_token: Option<String>,
    eos_token: Option<String>,
    /// Tokens that end generation
    eos_ids: Vec<u32>,
}

impl fmt::Debug for TextPipeline {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("TextPipeline")
            .field("vocab_size", &self.tokenizer.get_vocab_size(true))
            .field("chat_template", &self.chat_template.is_some())
            .field("eos_ids", &self.eos_ids)
            .finish()
    }
}

impl TextPipeline {
    /// A tokenizer without a chat template or end-of-sequence tokens; add them with the
    /// `with_*` methods
    pub fn new(tokenizer: Tokenizer) -> Self {
        Self {
            tokenizer,
            chat_template: None,
            bos_token: None,
            eos_token: None,
            eos_ids: Vec::new(),
        }
    }

    /// A Hugging Face `tokenizer.json`, and optionally the `tokenizer_config.json` next to it
    /// for the chat template and the BOS and EOS tokens
    pub fn from_files(
        tokenizer: impl AsRef<Path>,
        config: Option<&Path>,
    ) -> Result<Self, BoxError> {
        let tokenizer = Tokenizer::from_file(tokenizer)?;
        let mut pipeline = Self::new(tokenizer);
        if let Some(config) = config {
            let config: serde_json::Value = serde_json::from_slice(&std::fs::read(config)?)?;
            pipeline = pipeline.with_tokenizer_config(&config)?;
        }
        Ok(pipeline)
    }

    /// Take the chat template and the BOS and EOS tokens from a `tokenizer_config.json`
    pub fn with_tokenizer_config(mut self, config: &serde_json::Value) -> Result<Self, BoxError> {
        // A token is a string, or an object with its `content`
        let token = |key: &str| {
            let value = config.get(key)?;
            value
                .as_str()
                .or_else(|| value.get("content")?.as_str())
                .map(str::to_string)
        };
        if let Some(bos) = token("bos_token") {
            self.bos_token = Some(bos);
        }
        if let Some(eos) = token("eos_token") {
            self = self.with_eos_token(&eos)?;
        }
        // One template, or a list of named ones of which `default` is the chat template
        let template = match config.get("chat_template") {
            Some(serde_json::Value::String(template)) => Some(template.as_str()),
            Some(serde_json::Value::Array(named)) => named
                .iter()
                .find(|t| t.get("name").and_then(|n| n.as_str()) == Some("default"))
                .or_else(|| named.first())
                .and_then(|t| t.get("template")?.as_str()),
            _ => None,
        };
        if let Some(template) = template {
            self = self.with_chat_template(template)?;
        }
        Ok(self)
    }

    /// Use a Jinja chat template, as in Hugging Face's `chat_template`. Python string methods
    /// (`.strip()`, `.startswith()`, ...) and `raise_exception` work as they do there.
    pub fn with_chat_template(mut self, template: &str) -> Result<Self, BoxError> {
        let mut env = Environment::new();
        // As in transformers: blocks don't leave their newlines and indentation behind
        env.set_trim_blocks(true);
        env.set_lstrip_blocks(true);
        minijinja_contrib::add_to_environment(&mut env);
        env.set_unknown_method_callback(minijinja_contrib::pycompat::unknown_method_callback);
        env.add_function("raise_exception", |message: String| -> Result<String, _> {
            Err(minijinja::Error::new(ErrorKind::InvalidOperation, message))
        });
        env.add_function("strftime_now", |format: String| {
            use std::fmt::Write as _;
            let mut out = String::new();
            write!(out, "{}", chrono::Local::now().format(&format))
                .map(|()| out)
                .map_err(|_| {
                    minijinja::Error::new(
                        ErrorKind::InvalidOperation,
                        format!("invalid strftime format {format:?}"),
                    )
                })
        });
        env.add_template_owned(CHAT_TEMPLATE, template.to_string())?;
        self.chat_template = Some(env);
        Ok(self)
    }

    /// The beginning-of-sequence token a chat template writes as `bos_token`. It only reaches
    /// the template: raw text gets the tokenizer's own (see [`encode`](Self::encode)).
    pub fn with_bos_token(mut self, token: &str) -> Self {
        self.bos_token = Some(token.to_string());
        self
    }

    /// End generation at `token` (several can be added); it must be in the vocabulary
    pub fn with_eos_token(mut self, token: &str) -> Result<Self, BoxError> {
        let id = self
            .tokenizer
            .token_to_id(token)
            .ok_or_else(|| format!("the EOS token {token:?} is not in the vocabulary"))?;
        if self.eos_token.is_none() {
            self.eos_token = Some(token.to_string());
        }
        Ok(self.with_eos_id(id))
    }

    /// End generation at the token `id`
    pub fn with_eos_id(mut self, id: u32) -> Self {
        if !self.eos_ids.contains(&id) {
            self.eos_ids.push(id);
        }
        self
    }

    /// The tokenizer, for detokenizing and for the vocabulary size
    pub fn tokenizer(&self) -> &Tokenizer {
        &self.tokenizer
    }

    /// The tokens that end generation, in the order they were added
    pub fn eos_ids(&self) -> &[u32] {
        &self.eos_ids
    }

    /// Whether chat requests can be served; without a template only raw text can
    pub fn has_chat_template(&self) -> bool {
        self.chat_template.is_some()
    }

    /// The prompt text for a chat, ending where the assistant's answer begins
    pub fn apply_chat_template(&self, messages: &[Message]) -> Result<String, EngineError> {
        let Some(env) = &self.chat_template else {
            return Err(EngineError::Invalid(
                "this model has no chat template; use /v1/completions".to_string(),
            ));
        };
        let template = env
            .get_template(CHAT_TEMPLATE)
            .map_err(|e| EngineError::Failed(e.to_string()))?;
        template
            .render(minijinja::context! {
                messages => messages,
                add_generation_prompt => true,
                bos_token => self.bos_token.as_deref().unwrap_or(""),
                eos_token => self.eos_token.as_deref().unwrap_or(""),
            })
            .map_err(|e| {
                // `raise_exception` in the template: the request is what it rejects
                let message = match e.detail() {
                    Some(detail) if e.kind() == ErrorKind::InvalidOperation => detail.to_string(),
                    _ => e.to_string(),
                };
                EngineError::Invalid(format!("chat template: {message}"))
            })
    }

    /// The prompt tokens of a request: a chat through the chat template, raw text as it is.
    /// See [`encode`](Self::encode) for the special tokens.
    pub fn encode_input(&self, input: &Input) -> Result<Vec<u32>, EngineError> {
        let (text, add_special_tokens) = match input {
            Input::Chat(messages) => (self.apply_chat_template(messages)?, false),
            Input::Text(text) => (text.clone(), true),
        };
        self.encode(&text, add_special_tokens)
    }

    /// The tokens of `text`.
    ///
    /// `add_special_tokens` runs the tokenizer's post-processor, which adds the special tokens
    /// the model was trained to see around a text: a BOS token, for many models. A chat's
    /// template writes its own (`bos_token` and the role markers), so a rendered chat is encoded
    /// without them, or the prompt would start with two BOS tokens; raw text is encoded with them.
    pub fn encode(&self, text: &str, add_special_tokens: bool) -> Result<Vec<u32>, EngineError> {
        self.tokenizer
            .encode_fast(text, add_special_tokens)
            .map(|encoding| encoding.get_ids().to_vec())
            .map_err(|e| EngineError::Failed(format!("tokenizer: {e}")))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::llm::mock;

    #[test]
    fn a_hugging_face_style_template_renders() {
        // The shape of Qwen2.5's template, Python string methods included
        let template = "{%- if messages[0]['role'] == 'system' %}\
            {{- '<|im_start|>system\\n' + messages[0]['content'].strip() + '<|im_end|>\\n' }}\
            {%- else %}{{- '<|im_start|>system\\nYou are helpful.<|im_end|>\\n' }}{%- endif %}\
            {%- for message in messages %}{%- if message.role != 'system' %}\
            {{- '<|im_start|>' + message.role + '\\n' + message.content + '<|im_end|>\\n' }}\
            {%- endif %}{%- endfor %}\
            {%- if add_generation_prompt %}{{- '<|im_start|>assistant\\n' }}{%- endif %}";
        let pipeline = TextPipeline::new(mock::byte_tokenizer())
            .with_chat_template(template)
            .unwrap();
        let text = pipeline
            .apply_chat_template(&[
                Message::new("system", "  Be brief. "),
                Message::new("user", "Hi"),
            ])
            .unwrap();
        assert_eq!(
            text,
            "<|im_start|>system\nBe brief.<|im_end|>\n<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n"
        );
    }

    #[test]
    fn raise_exception_rejects_the_request() {
        let pipeline = TextPipeline::new(mock::byte_tokenizer())
            .with_chat_template(
                "{% if messages[0].role != 'user' %}{{ raise_exception('Conversations must start with a user message') }}{% endif %}",
            )
            .unwrap();
        let err = pipeline
            .apply_chat_template(&[Message::new("assistant", "x")])
            .unwrap_err();
        assert!(
            matches!(&err, EngineError::Invalid(m) if m.contains("must start with a user")),
            "{err:?}"
        );
    }

    #[test]
    fn a_chat_without_a_template_is_refused() {
        let pipeline = TextPipeline::new(mock::byte_tokenizer());
        assert!(matches!(
            pipeline.encode_input(&Input::Chat(vec![Message::new("user", "Hi")])),
            Err(EngineError::Invalid(_))
        ));
    }

    #[test]
    fn tokenizer_config_gives_the_template_and_tokens() {
        let config = serde_json::json!({
            "bos_token": {"content": "<s>", "lstrip": false},
            "eos_token": mock::EOS_TOKEN,
            "chat_template": [
                {"name": "tool_use", "template": "tools"},
                {"name": "default", "template": "{{ bos_token }}{% for m in messages %}{{ m.content }}{% endfor %}{{ eos_token }}"}
            ]
        });
        let pipeline = TextPipeline::new(mock::byte_tokenizer())
            .with_tokenizer_config(&config)
            .unwrap();
        assert_eq!(pipeline.eos_ids(), [mock::EOS_ID]);
        let text = pipeline
            .apply_chat_template(&[Message::new("user", "Hi")])
            .unwrap();
        assert_eq!(text, format!("<s>Hi{}", mock::EOS_TOKEN));
    }

    #[test]
    fn an_unknown_eos_token_is_an_error() {
        assert!(
            TextPipeline::new(mock::byte_tokenizer())
                .with_eos_token("<nope>")
                .is_err()
        );
    }
}
