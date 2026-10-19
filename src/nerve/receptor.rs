//! Receptors: the afferent side. Each samples the world on its own task or thread, extracts
//! a feature, stamps the urgency and hands the signal to the fabric; habituation lives in
//! its gate.

use std::sync::Weak;

use super::fabric::{Fabric, Inner, Outcome};
use super::neuron::Neuron;
use super::signal::{NeuronId, Signal};

/// A receptor's way into the fabric, given to it by [`Fabric::start`]. It holds the fabric
/// weakly: once the fabric is gone, [`Afferent::emit`] returns `None` and the receptor's
/// task should stop.
#[derive(Clone)]
pub struct Afferent {
    inner: Weak<Inner>,
    id: NeuronId,
}

impl Afferent {
    pub(crate) fn new(inner: Weak<Inner>, id: NeuronId) -> Self {
        Self { inner, id }
    }

    /// This receptor's id
    pub fn id(&self) -> NeuronId {
        self.id
    }

    /// The fabric, while it lives
    pub fn fabric(&self) -> Option<Fabric> {
        self.inner.upgrade().map(Fabric)
    }

    /// Stimulate the fabric through this receptor; the whole synchronous chain runs before
    /// this returns. `None` once the fabric is gone.
    pub fn emit(&self, signal: Signal) -> Option<Outcome> {
        let fabric = self.fabric()?;
        Some(fabric.stimulate(self.id, signal))
    }

    /// The receptive side's runtime, for receptors that read the network
    pub fn runtime(&self) -> Option<tokio::runtime::Handle> {
        self.fabric().map(|f| f.afferent().clone())
    }
}

/// A neuron that is also a source: started with the fabric, and possibly mounted as a route
pub trait Receptor: Neuron {
    /// Start sampling (a thread or a task on [`Afferent::runtime`]); return at once
    fn start(&self, _afferent: Afferent) {}

    /// Mount this receptor as an HTTP route: the fabric's router (`Server::nerve`) calls
    /// [`Receptor::ingest`] with each request's body
    fn mount(&self) -> Option<HttpMount> {
        None
    }

    /// Make the signal for an HTTP request (see [`Receptor::mount`]); `None` ignores it
    fn ingest(&self, _request: Ingest<'_>) -> Option<Signal> {
        None
    }
}

/// Where a receptor is mounted
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct HttpMount {
    /// An axum route path, e.g. `/events`
    pub path: String,
    /// `POST` (the default) or `GET`
    pub method: HttpMethod,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HttpMethod {
    Get,
    Post,
}

/// What an HTTP receptor sees of a request
pub struct Ingest<'a> {
    pub body: bytes::Bytes,
    pub content_type: Option<&'a str>,
    pub headers: &'a axum::http::HeaderMap,
}
