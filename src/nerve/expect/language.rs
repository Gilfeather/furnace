//! The expectation language (v1): declarative, schema-checked, no code. What a brain writes
//! with an `expect` command; each expectation compiles to a small matcher with a few words of
//! state ([`super::Expectations`]).

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use super::fields::Observation;
use crate::nerve::signal::{Command, sketch64};

/// Which signals an expectation is about. Every part given must hold.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Selector {
    /// A log template, as the extraction writes it (numbers and ids masked as `<*>`)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub template: Option<String>,
    /// Fields that must be equal (as text) to these values
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub fields: BTreeMap<String, serde_json::Value>,
}

impl Selector {
    /// The template's sketch, the index the matcher looks expectations up by
    pub fn template_key(&self) -> Option<u64> {
        self.template.as_deref().map(template_key)
    }

    /// Whether this selects nothing in particular: no template and no field
    pub fn is_wildcard(&self) -> bool {
        self.template.is_none() && self.fields.is_empty()
    }

    /// Whether `observed` is selected
    pub fn matches(&self, observed: &Observation) -> bool {
        if let Some(t) = self.template_key()
            && observed.template != Some(t)
        {
            return false;
        }
        self.fields
            .iter()
            .all(|(k, v)| observed.field(k).is_some_and(|o| same(o, v)))
    }

    /// Whether this is at least as specific as `other`: it names the same template if `other`
    /// does, and every field `other` names, with the same value
    pub fn at_least_as_specific_as(&self, other: &Selector) -> bool {
        (other.template.is_none() || self.template == other.template)
            && other
                .fields
                .iter()
                .all(|(k, v)| self.fields.get(k).is_some_and(|mine| same(mine, v)))
    }
}

/// The index of a template text
pub fn template_key(template: &str) -> u64 {
    sketch64(template.as_bytes())
}

/// Equal as text: `42` and `"42"` select the same field value
fn same(a: &serde_json::Value, b: &serde_json::Value) -> bool {
    a == b || text(a) == text(b)
}

pub(crate) fn text(v: &serde_json::Value) -> String {
    match v {
        serde_json::Value::String(s) => s.clone(),
        other => other.to_string(),
    }
}

/// What holds, and the state it needs
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "lowercase")]
pub enum Kind {
    /// A matching signal arrives; with `on_match` commands, act (the cache case)
    Match,
    /// Matching signals per `window_ms` stay within `[min, max]`
    Rate {
        /// The window, in milliseconds
        window_ms: u64,
        /// Fewest signals per window, 0 by default
        #[serde(default)]
        min: u64,
        /// Most signals per window, unbounded by default
        #[serde(default = "unbounded")]
        max: u64,
    },
    /// One matching signal at least every `period_ms`, give or take `tolerance_ms`
    Every {
        /// The expected gap between signals, in milliseconds
        period_ms: u64,
        /// Allowed deviation from `period_ms`, in milliseconds, 0 by default
        #[serde(default)]
        tolerance_ms: u64,
    },
    /// The numeric `field` of every matching signal stays within `[min, max]`
    Range {
        /// The field's name
        field: String,
        /// Lowest value allowed
        min: f64,
        /// Highest value allowed
        max: f64,
    },
    /// A matching signal is followed by one matching `then` within `within_ms`
    Then {
        /// The signals that must follow
        then: Selector,
        /// How soon they must, in milliseconds
        within_ms: u64,
    },
    /// No matching signal at all
    Never,
}

fn unbounded() -> u64 {
    u64::MAX
}

impl Kind {
    /// The kind as written in an expectation: `"rate"`, `"never"`, ...
    pub fn name(&self) -> &'static str {
        match self {
            Self::Match => "match",
            Self::Rate { .. } => "rate",
            Self::Every { .. } => "every",
            Self::Range { .. } => "range",
            Self::Then { .. } => "then",
            Self::Never => "never",
        }
    }

    /// The parameters [`Learn`] may name for this kind
    /// The fields this kind takes besides `kind`
    fn keys(&self) -> &'static [&'static str] {
        match self {
            Self::Rate { .. } => &["window_ms", "min", "max"],
            Self::Every { .. } => &["period_ms", "tolerance_ms"],
            Self::Range { .. } => &["field", "min", "max"],
            Self::Then { .. } => &["then", "within_ms"],
            Self::Match | Self::Never => &[],
        }
    }

    fn tunable(&self) -> &'static [&'static str] {
        match self {
            Self::Rate { .. } => &["min", "max"],
            Self::Every { .. } => &["period_ms", "tolerance_ms"],
            Self::Range { .. } => &["min", "max"],
            Self::Then { .. } => &["within_ms"],
            Self::Match | Self::Never => &[],
        }
    }
}

/// What to do with a matching signal that keeps the expectation
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum OnMatch {
    /// Nothing reaches the brain
    #[default]
    Absorb,
    /// These commands, at once; nothing reaches the brain
    Act(Vec<Command>),
}

/// What to do when the expectation breaks
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum OnViolation {
    /// Send the surprise to the brain
    #[default]
    Escalate,
    /// These commands at once, and send the surprise to the brain too
    Act(Vec<Command>),
}

/// Parameters the periphery may tune itself, each within `[low, high]` set by the brain: the
/// brain sets the structure, statistics set the numbers
pub type Learn = BTreeMap<String, [f64; 2]>;

/// One expectation, as a brain writes it. Unknown fields are refused ([`Expectation::parse`]).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Expectation {
    /// Unique among a neuron's expectations; a write with the same id replaces it
    pub id: String,
    /// Which signals it is about; none: every signal
    #[serde(default)]
    pub selector: Selector,
    /// What holds, flattened into the expectation (`"kind": "rate"`, ...)
    #[serde(flatten)]
    pub kind: Kind,
    /// What a matching signal that keeps it does
    #[serde(default)]
    pub on_match: OnMatch,
    /// What breaking it does
    #[serde(default)]
    pub on_violation: OnViolation,
    /// Parameters the periphery may tune within the brain's bounds
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub learn: Learn,
    /// Forgotten after this long without a match; none: the store's default
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ttl_s: Option<u64>,
    /// External state it depends on: retired when any of it changes
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub depends_on: Vec<String>,
    /// Why it exists: the surprise that led to it, as the brain cites it
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub provenance: Option<serde_json::Value>,
}

/// Why an expectation was refused
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Refused(pub String);

impl std::fmt::Display for Refused {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl Expectation {
    /// Fields every expectation may have
    const KEYS: [&'static str; 9] = [
        "id",
        "kind",
        "selector",
        "on_match",
        "on_violation",
        "learn",
        "ttl_s",
        "depends_on",
        "provenance",
    ];

    /// Parse one, refusing fields it does not know (serde cannot, with the kind flattened)
    pub fn parse(value: &serde_json::Value) -> Result<Self, Refused> {
        let e =
            Self::deserialize(value).map_err(|e| Refused(format!("not an expectation: {e}")))?;
        if let Some(object) = value.as_object()
            && let Some(unknown) = object
                .keys()
                .find(|k| !Self::KEYS.contains(&k.as_str()) && !e.kind.keys().contains(&k.as_str()))
        {
            return Err(Refused(format!(
                "not an expectation: unknown field `{unknown}` for {}",
                e.kind.name()
            )));
        }
        Ok(e)
    }

    /// Whether it absorbs (or acts on) matching signals without the brain
    pub fn absorbs(&self) -> bool {
        !matches!(self.kind, Kind::Never)
    }

    /// The commands it may fire
    pub fn commands(&self) -> impl Iterator<Item = &Command> {
        let on_match = match &self.on_match {
            OnMatch::Act(c) => c.as_slice(),
            OnMatch::Absorb => &[],
        };
        let on_violation = match &self.on_violation {
            OnViolation::Act(c) => c.as_slice(),
            OnViolation::Escalate => &[],
        };
        on_match.iter().chain(on_violation)
    }

    /// The checks that need no state: well-formed numbers, tunable parameters, the scope
    /// rules on selectors and effectors
    pub fn check(&self, allowed: &[String], surprise: Option<&Selector>) -> Result<(), Refused> {
        if self.id.is_empty() || self.id.len() > 128 {
            return Err(Refused("an id of 1 to 128 bytes".into()));
        }
        match &self.kind {
            Kind::Rate {
                window_ms,
                min,
                max,
            } if *window_ms == 0 || min > max => {
                return Err(Refused("rate: a window above 0 and min <= max".into()));
            }
            Kind::Every { period_ms: 0, .. } => {
                return Err(Refused("every: a period above 0".into()));
            }
            Kind::Range { min, max, .. } if min > max || min.is_nan() || max.is_nan() => {
                return Err(Refused("range: min <= max".into()));
            }
            Kind::Then { within_ms: 0, .. } => {
                return Err(Refused("then: within above 0".into()));
            }
            Kind::Then { then, .. } if then.is_wildcard() => {
                return Err(Refused("then: a selector for what follows".into()));
            }
            _ => {}
        }
        for (param, [low, high]) in &self.learn {
            if !self.kind.tunable().contains(&param.as_str()) {
                return Err(Refused(format!(
                    "learn: {} has no parameter '{param}'",
                    self.kind.name()
                )));
            }
            if low > high || low.is_nan() || high.is_nan() {
                return Err(Refused(format!("learn: {param} needs low <= high")));
            }
        }
        // No wildcard absorb: what absorbs must say what it absorbs, and at least as
        // specifically as the surprise it answers
        if self.absorbs() {
            if self.selector.is_wildcard() {
                return Err(Refused(
                    "an absorbing expectation needs a template or fields".into(),
                ));
            }
            if let Some(surprise) = surprise
                && !self.selector.at_least_as_specific_as(surprise)
            {
                return Err(Refused(
                    "an absorbing expectation must be at least as specific as the surprise it answers"
                        .into(),
                ));
            }
        }
        if let Some(c) = self.commands().find(|c| !allowed.contains(&c.effector)) {
            return Err(Refused(format!(
                "effector '{}' is not allowed for expectations",
                c.effector
            )));
        }
        Ok(())
    }

    /// A parameter's value, for learning and for `GET /nerve`
    pub(crate) fn param(&self, name: &str) -> Option<f64> {
        Some(match (&self.kind, name) {
            (Kind::Rate { min, .. }, "min") => *min as f64,
            (Kind::Rate { max, .. }, "max") => *max as f64,
            (Kind::Every { period_ms, .. }, "period_ms") => *period_ms as f64,
            (Kind::Every { tolerance_ms, .. }, "tolerance_ms") => *tolerance_ms as f64,
            (Kind::Range { min, .. }, "min") => *min,
            (Kind::Range { max, .. }, "max") => *max,
            (Kind::Then { within_ms, .. }, "within_ms") => *within_ms as f64,
            _ => return None,
        })
    }

    /// Set a learnable parameter, clamped to its `learn` bounds; `false` if not learnable
    pub(crate) fn tune(&mut self, name: &str, value: f64) -> bool {
        let Some(&[low, high]) = self.learn.get(name) else {
            return false;
        };
        let v = value.clamp(low, high);
        let whole = v.max(0.0).round() as u64;
        match (&mut self.kind, name) {
            (Kind::Rate { min, .. }, "min") => *min = whole,
            (Kind::Rate { max, .. }, "max") => *max = whole,
            (Kind::Every { period_ms, .. }, "period_ms") => *period_ms = whole.max(1),
            (Kind::Every { tolerance_ms, .. }, "tolerance_ms") => *tolerance_ms = whole,
            (Kind::Range { min, .. }, "min") => *min = v,
            (Kind::Range { max, .. }, "max") => *max = v,
            (Kind::Then { within_ms, .. }, "within_ms") => *within_ms = whole.max(1),
            _ => return false,
        }
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse(v: serde_json::Value) -> Result<Expectation, Refused> {
        Expectation::parse(&v)
    }

    #[test]
    fn expectations_parse_and_unknown_fields_are_refused() {
        let e = parse(serde_json::json!({
            "id": "hb", "kind": "every", "period_ms": 5000, "tolerance_ms": 500,
            "selector": {"template": "heartbeat <*>", "fields": {"svc": "a"}},
            "learn": {"period_ms": [4000, 8000]}
        }))
        .unwrap();
        assert_eq!(
            e.kind,
            Kind::Every {
                period_ms: 5000,
                tolerance_ms: 500
            }
        );
        assert_eq!(e.on_match, OnMatch::Absorb);
        assert!(parse(serde_json::json!({"id": "x", "kind": "match", "sudo": true})).is_err());
        assert!(parse(serde_json::json!({"id": "x", "kind": "teleport"})).is_err());
        let acting = parse(serde_json::json!({
            "id": "t", "kind": "match", "selector": {"template": "disk <*> full"},
            "on_match": {"act": [{"effector": "ticket", "args": {"q": "ops"}}]}
        }))
        .unwrap();
        assert_eq!(acting.commands().count(), 1);
    }

    #[test]
    fn scope_rules() {
        let allowed = vec!["ticket".to_string()];
        let wildcard = parse(serde_json::json!({"id": "all", "kind": "match"})).unwrap();
        assert!(
            wildcard.check(&allowed, None).is_err(),
            "no wildcard absorb"
        );
        // Never absorbs nothing, so it may select broadly
        let never = parse(serde_json::json!({"id": "n", "kind": "never", "selector": {}})).unwrap();
        assert!(never.check(&allowed, None).is_ok());
        let pay = parse(serde_json::json!({
            "id": "p", "kind": "match", "selector": {"template": "invoice <*>"},
            "on_match": {"act": [{"effector": "pay"}]}
        }))
        .unwrap();
        assert!(pay.check(&allowed, None).is_err(), "only allowed effectors");
        // Narrower than the surprise it answers: refused; as narrow: accepted
        let surprise = Selector {
            template: Some("login failed for <*>".into()),
            fields: BTreeMap::from([("user".into(), serde_json::json!("root"))]),
        };
        let broad = parse(serde_json::json!({
            "id": "b", "kind": "match", "selector": {"template": "login failed for <*>"}
        }))
        .unwrap();
        assert!(broad.check(&allowed, Some(&surprise)).is_err());
        let narrow = parse(serde_json::json!({
            "id": "b", "kind": "match",
            "selector": {"template": "login failed for <*>", "fields": {"user": "root"}}
        }))
        .unwrap();
        assert!(narrow.check(&allowed, Some(&surprise)).is_ok());
        let bad_learn = parse(serde_json::json!({
            "id": "l", "kind": "match", "selector": {"template": "x"}, "learn": {"max": [0, 1]}
        }))
        .unwrap();
        assert!(bad_learn.check(&allowed, None).is_err());
    }

    #[test]
    fn learning_stays_within_the_brain_s_bounds() {
        let mut e = parse(serde_json::json!({
            "id": "r", "kind": "range", "field": "t", "min": 18.0, "max": 27.0,
            "selector": {"template": "temp <*>"}, "learn": {"max": [27.0, 30.0]}
        }))
        .unwrap();
        assert!(e.tune("max", 29.0));
        assert_eq!(e.param("max"), Some(29.0));
        assert!(e.tune("max", 45.0));
        assert_eq!(e.param("max"), Some(30.0));
        assert!(!e.tune("min", 0.0), "min was not given to learn");
    }
}
