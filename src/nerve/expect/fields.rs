//! What the matcher sees of a signal: a template and named fields, extracted from a log line
//! or a JSON body.
//!
//! - A log line's template is the line with every token holding a digit masked as `<*>`;
//!   the masked tokens are the fields `v0`, `v1`, ... in order, and a `key=value` token is
//!   also the field `key`.
//! - A JSON object's fields are its members, nested objects joined with dots (`cpu.load`);
//!   its template is the sorted list of its field names, so bodies of one shape share one.

use serde_json::{Map, Value};

use super::language::template_key;
use crate::nerve::signal::{Payload, Signal};

/// How a signal is read
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Extraction {
    /// Text as a log line, a JSON body as JSON: whatever the payload is
    #[default]
    Auto,
    /// Always as a log line
    LogLines,
    /// As JSON, text included; as a log line if it is not a JSON object
    Json,
}

/// A signal as the matcher sees it
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Observation {
    /// The template's key ([`template_key`] of [`Observation::template_text`])
    pub template: Option<u64>,
    /// The template as text
    pub template_text: Option<String>,
    /// The named fields
    pub fields: Map<String, Value>,
    /// The line, or the body as text, for the brain's context
    pub text: Option<String>,
}

impl Observation {
    /// The field `name`
    pub fn field(&self, name: &str) -> Option<&Value> {
        self.fields.get(name)
    }

    /// A field as a number: a JSON number, or text that parses as one
    pub fn number(&self, name: &str) -> Option<f64> {
        match self.fields.get(name)? {
            Value::Number(n) => n.as_f64(),
            Value::String(s) => s.trim_end_matches(['%', ',', ';']).parse().ok(),
            _ => None,
        }
    }

    /// For a surprise's payload
    pub fn to_json(&self) -> Value {
        serde_json::json!({
            "template": self.template_text,
            "fields": self.fields,
            "text": self.text,
        })
    }
}

/// Read `signal` as `how` says
pub fn observe(signal: &Signal, how: Extraction) -> Observation {
    let json = match (&signal.payload, how) {
        (_, Extraction::LogLines) => None,
        (Payload::Json(v), _) => Some((**v).clone()),
        (Payload::Text(_) | Payload::Bytes(_), Extraction::Json) => signal.payload.to_json(),
        (Payload::Text(t), Extraction::Auto) if t.trim_start().starts_with('{') => {
            signal.payload.to_json()
        }
        _ => None,
    };
    match json {
        Some(Value::Object(object)) => json_fields(&object),
        _ => match signal.payload.as_text() {
            Some(line) => log_line(line),
            None => Observation::default(),
        },
    }
}

/// A log line's template and fields
pub fn log_line(line: &str) -> Observation {
    let mut fields = Map::new();
    let mut template = String::with_capacity(line.len());
    let mut n = 0;
    for token in line.split_whitespace() {
        if !template.is_empty() {
            template.push(' ');
        }
        if let Some((key, value)) = token.split_once('=')
            && !key.is_empty()
            && !key.bytes().any(|b| b.is_ascii_digit())
        {
            fields.insert(key.to_string(), Value::String(value.to_string()));
            if value.bytes().any(|b| b.is_ascii_digit()) {
                template.push_str(key);
                template.push_str("=<*>");
                fields.insert(format!("v{n}"), Value::String(value.to_string()));
                n += 1;
                continue;
            }
        }
        if token.bytes().any(|b| b.is_ascii_digit()) {
            template.push_str("<*>");
            fields.insert(format!("v{n}"), Value::String(token.to_string()));
            n += 1;
        } else {
            template.push_str(token);
        }
    }
    Observation {
        template: Some(template_key(&template)),
        template_text: Some(template),
        fields,
        text: Some(line.to_string()),
    }
}

/// A JSON object's fields, nested objects flattened with dots, and its shape as template
pub fn json_fields(object: &Map<String, Value>) -> Observation {
    let mut fields = Map::new();
    flatten("", object, &mut fields);
    let mut names: Vec<&str> = fields.keys().map(String::as_str).collect();
    names.sort_unstable();
    let template = format!("{{{}}}", names.join(","));
    Observation {
        template: Some(template_key(&template)),
        template_text: Some(template),
        text: Some(Value::Object(object.clone()).to_string()),
        fields,
    }
}

fn flatten(prefix: &str, object: &Map<String, Value>, out: &mut Map<String, Value>) {
    for (key, value) in object {
        let name = if prefix.is_empty() {
            key.clone()
        } else {
            format!("{prefix}.{key}")
        };
        match value {
            Value::Object(inner) => flatten(&name, inner, out),
            other => {
                out.insert(name, other.clone());
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nerve::signal::Feature;

    #[test]
    fn log_lines_become_a_template_and_fields() {
        let o = log_line(
            "081109 203615 148 INFO dfs.DataNode: Receiving block blk_-16 src: /10.250.19.102:54106 size=67108864 user=root",
        );
        assert_eq!(
            o.template_text.as_deref(),
            Some("<*> <*> <*> INFO dfs.DataNode: Receiving block <*> src: <*> size=<*> user=root")
        );
        assert_eq!(o.field("v0"), Some(&Value::from("081109")));
        assert_eq!(o.field("user"), Some(&Value::from("root")));
        assert_eq!(o.number("size"), Some(67_108_864.0));
        // Other numbers, same template
        let again = log_line(
            "081110 000001 7 INFO dfs.DataNode: Receiving block blk_99 src: /10.0.0.1:1 size=1 user=root",
        );
        assert_eq!(again.template, o.template);
    }

    #[test]
    fn json_bodies_become_fields_and_a_shape() {
        let signal = Signal::new(
            Feature::NONE,
            Payload::json(serde_json::json!({"svc": "a", "cpu": {"load": 0.5}})),
        );
        let o = observe(&signal, Extraction::Auto);
        assert_eq!(o.template_text.as_deref(), Some("{cpu.load,svc}"));
        assert_eq!(o.number("cpu.load"), Some(0.5));
        let text = Signal::new(
            Feature::NONE,
            Payload::text(r#"{"svc": "b", "cpu": {"load": 1}}"#),
        );
        assert_eq!(observe(&text, Extraction::Auto).template, o.template);
        // Read as a log line when asked to
        assert!(
            observe(&text, Extraction::LogLines)
                .fields
                .contains_key("v0")
        );
    }
}
