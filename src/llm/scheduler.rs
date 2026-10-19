//! Furnace's scheduler for step-wise models: it turns a [`StepModel`] and a [`TextPipeline`] into
//! an [`LlmEngine`].
//!
//! Worker threads hold the running sequences. Each iteration a worker drops the sequences whose
//! client went away, starts queued requests (their prefills) up to a budget of prompt tokens,
//! then runs one decode step over all of them: at batch 1 that is a round over them, token by
//! token, so concurrent streams interleave and a long request never holds the model for the
//! others. Sampling runs on the
//! worker, on the host logits. Detokenization and stop strings run on the async side, one task
//! per request, which ends the request by dropping its channel, exactly as a disconnect does.

use std::collections::VecDeque;
use std::fmt;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::time::Instant;

use tokio::sync::mpsc;
use tracing::{debug, warn};

use super::{
    BoxError, EngineError, Event, FinishReason, GenerateRequest, LlmEngine, Sampler, StopStrings,
    TextPipeline,
};
use crate::telemetry::Histogram;

/// Next-token logits over the vocabulary, on the host
#[derive(Debug, Clone, PartialEq)]
pub struct Logits(pub Vec<f32>);

/// A model that can prefill a prompt and decode one token at a time.
///
/// `Sequence` holds a request's state (KV cache, position); furnace drops it when the request
/// ends or its client disconnects, which frees that memory. The methods block: furnace calls them
/// on its own worker threads, never on the async runtime.
pub trait StepModel: Send + Sync + fmt::Debug + 'static {
    type Sequence: Send + 'static;

    /// The name clients ask for in `model`
    fn name(&self) -> &str;

    /// Tokens of prompt and output together
    fn max_context(&self) -> usize;

    /// Sequences the model can hold at once (KV memory); furnace never runs more
    fn max_sequences(&self) -> usize {
        1
    }

    /// Run the prompt; the logits are for the token after it
    fn prefill(&self, prompt: &[u32]) -> Result<(Self::Sequence, Logits), BoxError>;

    /// Append `token` to the sequence; the logits are for the token after it
    fn decode(&self, sequence: &mut Self::Sequence, token: u32) -> Result<Logits, BoxError>;

    /// One decode step for several sequences. The default runs them one by one; a model with a
    /// ragged-batch forward pass overrides it, and that alone gives continuous batching.
    fn decode_batch(
        &self,
        sequences: &mut [&mut Self::Sequence],
        tokens: &[u32],
    ) -> Vec<Result<Logits, BoxError>> {
        sequences
            .iter_mut()
            .zip(tokens)
            .map(|(sequence, &token)| self.decode(sequence, token))
            .collect()
    }
}

/// How [`Scheduled`] runs a model
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SchedulerConfig {
    /// Worker threads; each runs its own sequences. On a CPU, one worker usually suffices, since
    /// the model's kernels use every core already.
    pub workers: usize,
    /// Requests waiting for a sequence slot before new ones get 503
    pub max_waiting: usize,
    /// Prompt tokens a worker prefills between two decode steps, so a burst of arrivals starts
    /// together while the running streams wait at most about this much prefill per token. A
    /// longer prompt still starts, alone.
    pub max_prefill_tokens: usize,
}

impl Default for SchedulerConfig {
    fn default() -> Self {
        Self {
            workers: 1,
            max_waiting: 256,
            max_prefill_tokens: 512,
        }
    }
}

/// Model time per step, from furnace's scheduler
#[derive(Default)]
pub struct SchedulerStats {
    /// One prompt's prefill
    pub prefill: Histogram,
    /// One decode step: one call of `decode_batch`, over every running sequence of a worker
    pub decode_step: Histogram,
}

impl fmt::Debug for SchedulerStats {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SchedulerStats")
            .field("prefill_ms", &self.prefill.percentiles_ms())
            .field("decode_step_ms", &self.decode_step.percentiles_ms())
            .finish()
    }
}

/// What a worker tells a request's relay task
enum Step {
    Started { prompt_tokens: usize },
    Token(u32),
    Done(FinishReason),
    Failed(String),
}

/// A request waiting for a sequence slot
struct Job {
    prompt: Vec<u32>,
    sampler: Sampler,
    max_new_tokens: usize,
    steps: mpsc::UnboundedSender<Step>,
}

struct Shared<M: StepModel> {
    model: M,
    eos_ids: Vec<u32>,
    queue: Mutex<VecDeque<Job>>,
    /// Signalled on a new job, a freed sequence slot, and shutdown
    wake: Condvar,
    /// Sequences running, over every worker
    running: AtomicUsize,
    max_sequences: usize,
    max_prefill_tokens: usize,
    stop: AtomicBool,
    stats: SchedulerStats,
}

/// A [`StepModel`] served through furnace's scheduler, with a [`TextPipeline`] for the text: the
/// [`LlmEngine`] to give [`Server::llm`](crate::Server::llm).
pub struct Scheduled<M: StepModel> {
    shared: Arc<Shared<M>>,
    pipeline: Arc<TextPipeline>,
    name: String,
    max_context: usize,
    max_waiting: usize,
}

impl<M: StepModel> fmt::Debug for Scheduled<M> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Scheduled")
            .field("model", &self.shared.model)
            .field("pipeline", &self.pipeline)
            .finish()
    }
}

impl<M: StepModel> Scheduled<M> {
    /// Serve `model` with [`SchedulerConfig::default`]: one worker thread
    pub fn new(model: M, pipeline: TextPipeline) -> Self {
        Self::with_config(model, pipeline, SchedulerConfig::default())
    }

    pub fn with_config(model: M, pipeline: TextPipeline, config: SchedulerConfig) -> Self {
        let name = model.name().to_string();
        let max_context = model.max_context();
        let shared = Arc::new(Shared {
            max_sequences: model.max_sequences().max(1),
            max_prefill_tokens: config.max_prefill_tokens,
            model,
            eos_ids: pipeline.eos_ids().to_vec(),
            queue: Mutex::new(VecDeque::new()),
            wake: Condvar::new(),
            running: AtomicUsize::new(0),
            stop: AtomicBool::new(false),
            stats: SchedulerStats::default(),
        });
        for i in 0..config.workers.max(1) {
            let shared = shared.clone();
            std::thread::Builder::new()
                .name(format!("furnace-llm-{i}"))
                .spawn(move || work(&shared))
                .expect("spawning an LLM worker thread");
        }
        Self {
            shared,
            pipeline: Arc::new(pipeline),
            name,
            max_context,
            max_waiting: config.max_waiting,
        }
    }

    /// The model, for whatever it offers besides [`StepModel`]
    pub fn model(&self) -> &M {
        &self.shared.model
    }

    pub fn pipeline(&self) -> &TextPipeline {
        &self.pipeline
    }

    /// Sequences running now, over every worker
    pub fn running(&self) -> usize {
        self.shared.running.load(Ordering::Acquire)
    }
}

impl<M: StepModel> Drop for Scheduled<M> {
    fn drop(&mut self) {
        self.shared.stop.store(true, Ordering::Release);
        self.shared.wake.notify_all();
    }
}

impl<M: StepModel> LlmEngine for Scheduled<M> {
    fn name(&self) -> &str {
        &self.name
    }

    fn max_context(&self) -> usize {
        self.max_context
    }

    fn generate(&self, request: GenerateRequest) -> Result<mpsc::Receiver<Event>, EngineError> {
        let prompt = self.pipeline.encode_input(&request.input)?;
        if prompt.is_empty() {
            return Err(EngineError::Invalid("the prompt is empty".to_string()));
        }
        if prompt.len() >= self.max_context {
            return Err(EngineError::TooLong {
                prompt_tokens: prompt.len(),
                max_context: self.max_context,
            });
        }
        let room = self.max_context - prompt.len();
        let max_new_tokens = match request.params.max_tokens {
            Some(0) => {
                return Err(EngineError::Invalid(
                    "max_tokens must be at least 1".to_string(),
                ));
            }
            Some(max) => max.min(room),
            None => room,
        };
        let (steps, step_receiver) = mpsc::unbounded_channel();
        let job = Job {
            prompt,
            sampler: Sampler::new(&request.params),
            max_new_tokens,
            steps,
        };
        {
            let mut queue = lock(&self.shared.queue);
            // Requests whose client left already don't count
            queue.retain(|job| !job.steps.is_closed());
            if queue.len() >= self.max_waiting {
                return Err(EngineError::Busy(format!(
                    "{} requests are waiting for {}",
                    queue.len(),
                    self.name
                )));
            }
            queue.push_back(job);
        }
        self.shared.wake.notify_one();

        let (events, receiver) = mpsc::channel(64);
        tokio::spawn(relay(
            self.pipeline.clone(),
            StopStrings::new(&request.stop),
            step_receiver,
            events,
        ));
        Ok(receiver)
    }

    fn scheduler_stats(&self) -> Option<&SchedulerStats> {
        Some(&self.shared.stats)
    }
}

/// One request's text: tokens from the worker become `Text` events, through incremental
/// detokenization and the stop strings. Returning drops `steps`, which tells the worker to drop
/// the sequence: when the client is gone (`events` closed) or a stop string matched.
async fn relay(
    pipeline: Arc<TextPipeline>,
    mut stop: StopStrings,
    mut steps: mpsc::UnboundedReceiver<Step>,
    events: mpsc::Sender<Event>,
) {
    let mut decoder = pipeline.tokenizer().decode_stream(true);
    let mut completion_tokens = 0;
    loop {
        let step = tokio::select! {
            step = steps.recv() => step,
            () = events.closed() => return,
        };
        let event = match step {
            Some(Step::Started { prompt_tokens }) => Event::Started { prompt_tokens },
            Some(Step::Token(token)) => {
                completion_tokens += 1;
                let piece = match decoder.step(token) {
                    Ok(Some(piece)) => piece,
                    // An incomplete character: it comes with a later token
                    Ok(None) => continue,
                    Err(e) => {
                        let _ = events
                            .send(Event::Error(EngineError::Failed(format!(
                                "detokenizing: {e}"
                            ))))
                            .await;
                        return;
                    }
                };
                let (text, matched) = stop.push(&piece);
                if !text.is_empty() && events.send(Event::Text(text)).await.is_err() {
                    return;
                }
                if matched {
                    let _ = events
                        .send(Event::Finished {
                            reason: FinishReason::Stop,
                            completion_tokens,
                        })
                        .await;
                    return;
                }
                continue;
            }
            Some(Step::Done(reason)) => {
                let rest = stop.finish();
                if !rest.is_empty() && events.send(Event::Text(rest)).await.is_err() {
                    return;
                }
                let _ = events
                    .send(Event::Finished {
                        reason,
                        completion_tokens,
                    })
                    .await;
                return;
            }
            Some(Step::Failed(message)) => {
                let _ = events
                    .send(Event::Error(EngineError::Failed(message)))
                    .await;
                return;
            }
            None => {
                let _ = events
                    .send(Event::Error(EngineError::Failed(
                        "the model stopped".to_string(),
                    )))
                    .await;
                return;
            }
        };
        if events.send(event).await.is_err() {
            return;
        }
    }
}

/// A sequence a worker is running
struct Running<S> {
    sequence: S,
    sampler: Sampler,
    /// The token to decode next
    last: u32,
    generated: usize,
    max_new_tokens: usize,
    steps: mpsc::UnboundedSender<Step>,
    _slot: Slot,
}

/// One of the model's `max_sequences`, given back on drop
struct Slot {
    running: Arc<dyn Fn() + Send + Sync>,
}

impl Drop for Slot {
    fn drop(&mut self) {
        (self.running)();
    }
}

fn lock<T>(mutex: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    mutex.lock().unwrap_or_else(|e| e.into_inner())
}

/// A model call, with a panic turned into an error so the worker and its other sequences live on
fn guarded<T>(call: impl FnOnce() -> Result<T, BoxError>) -> Result<T, String> {
    match catch_unwind(AssertUnwindSafe(call)) {
        Ok(Ok(value)) => Ok(value),
        Ok(Err(e)) => Err(e.to_string()),
        Err(_) => Err("the model panicked".to_string()),
    }
}

/// A worker's loop, until the engine is dropped
fn work<M: StepModel>(shared: &Arc<Shared<M>>) {
    let release: Arc<dyn Fn() + Send + Sync> = {
        let shared = Arc::downgrade(shared);
        Arc::new(move || {
            if let Some(shared) = shared.upgrade() {
                shared.running.fetch_sub(1, Ordering::AcqRel);
                shared.wake.notify_all();
            }
        })
    };
    let mut running: Vec<Running<M::Sequence>> = Vec::new();
    loop {
        // Drop the sequences whose client went away, or whose stop string matched
        running.retain(|r| !r.steps.is_closed());

        let jobs = {
            let mut queue = lock(&shared.queue);
            loop {
                if shared.stop.load(Ordering::Acquire) {
                    return;
                }
                queue.retain(|job| !job.steps.is_closed());
                let free = shared.running.load(Ordering::Acquire) < shared.max_sequences;
                if !running.is_empty() || (free && !queue.is_empty()) {
                    break;
                }
                queue = shared.wake.wait(queue).unwrap_or_else(|e| e.into_inner());
            }
            // Within the prefill budget; the first one in any case
            let mut jobs = Vec::new();
            let mut budget = shared.max_prefill_tokens;
            while let Some(job) = queue.front() {
                if (!jobs.is_empty() && job.prompt.len() > budget) || !reserve(shared) {
                    break;
                }
                budget = budget.saturating_sub(job.prompt.len());
                jobs.extend(queue.pop_front());
            }
            jobs
        };

        // The new sequences join this iteration's decode step, so their second token comes one
        // step after their first, as everyone's does
        for job in jobs {
            let slot = Slot {
                running: release.clone(),
            };
            running.extend(start(shared, job, slot));
        }

        if !running.is_empty() {
            let tokens: Vec<u32> = running.iter().map(|r| r.last).collect();
            let start = Instant::now();
            let results = {
                let mut sequences: Vec<&mut M::Sequence> =
                    running.iter_mut().map(|r| &mut r.sequence).collect();
                match catch_unwind(AssertUnwindSafe(|| {
                    shared.model.decode_batch(&mut sequences, &tokens)
                })) {
                    Ok(results) if results.len() == tokens.len() => results
                        .into_iter()
                        .map(|r| r.map_err(|e| e.to_string()))
                        .collect(),
                    Ok(results) => {
                        let message = format!(
                            "decode_batch returned {} results for {} sequences",
                            results.len(),
                            tokens.len()
                        );
                        vec![Err(message); tokens.len()]
                    }
                    Err(_) => vec![Err("the model panicked".to_string()); tokens.len()],
                }
            };
            shared.stats.decode_step.observe(start.elapsed());
            let mut results = results.into_iter();
            running.retain_mut(|r| match results.next() {
                Some(Ok(Logits(mut logits))) => advance(shared, r, &mut logits),
                Some(Err(message)) => {
                    warn!("Decoding failed: {message}");
                    let _ = r.steps.send(Step::Failed(message));
                    false
                }
                None => false,
            });
        }
    }
}

/// Take a sequence slot if one is free
fn reserve<M: StepModel>(shared: &Shared<M>) -> bool {
    let mut running = shared.running.load(Ordering::Acquire);
    while running < shared.max_sequences {
        match shared.running.compare_exchange_weak(
            running,
            running + 1,
            Ordering::AcqRel,
            Ordering::Acquire,
        ) {
            Ok(_) => return true,
            Err(now) => running = now,
        }
    }
    false
}

/// Prefill a job and sample its first token; `None` when it ended already
fn start<M: StepModel>(shared: &Shared<M>, job: Job, slot: Slot) -> Option<Running<M::Sequence>> {
    let Job {
        prompt,
        sampler,
        max_new_tokens,
        steps,
    } = job;
    steps
        .send(Step::Started {
            prompt_tokens: prompt.len(),
        })
        .ok()?;
    let begin = Instant::now();
    let (sequence, Logits(mut logits)) = match guarded(|| shared.model.prefill(&prompt)) {
        Ok(prefilled) => prefilled,
        Err(message) => {
            warn!("Prefill failed: {message}");
            let _ = steps.send(Step::Failed(message));
            return None;
        }
    };
    shared.stats.prefill.observe(begin.elapsed());
    debug!("Prefilled {} tokens in {:?}", prompt.len(), begin.elapsed());
    let mut running = Running {
        sequence,
        sampler,
        last: 0,
        generated: 0,
        max_new_tokens,
        steps,
        _slot: slot,
    };
    advance(shared, &mut running, &mut logits).then_some(running)
}

/// Sample the next token and send it; whether the sequence goes on
fn advance<M: StepModel, S>(
    shared: &Shared<M>,
    running: &mut Running<S>,
    logits: &mut [f32],
) -> bool {
    let token = running.sampler.sample(logits);
    if shared.eos_ids.contains(&token) {
        let _ = running.steps.send(Step::Done(FinishReason::Stop));
        return false;
    }
    if running.steps.send(Step::Token(token)).is_err() {
        return false;
    }
    running.generated += 1;
    running.last = token;
    if running.generated >= running.max_new_tokens {
        let _ = running.steps.send(Step::Done(FinishReason::Length));
        return false;
    }
    true
}
