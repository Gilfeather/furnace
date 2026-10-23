# TLA+ specifications of the nervous system

The design of furnace's nervous system ([docs/NERVE.md](../docs/NERVE.md)), written in TLA+
and model-checked with TLC: the motor side's hierarchy, the nerves' lanes, and how a signal is
routed. Each spec follows the Rust code it names closely enough to catch its bugs, and stays
small and bounded, so that TLC explores every state in seconds.
[`.github/workflows/tla.yml`](../.github/workflows/tla.yml) checks every model on the pull
requests that touch `specs/` or `src/nerve/`, and on main.

| Spec | Models | The Rust code |
| --- | --- | --- |
| [`Motor.tla`](Motor.tla) | [`MCMotor.cfg`](MCMotor.cfg), [`MCMotorPreempt.cfg`](MCMotorPreempt.cfg) | `src/nerve/motor/mod.rs` (`Motor::stimulate`, `Started::run`, `Motion::{command, together}`), `src/nerve/effector/local_loop.rs` (`run`), `Fabric::stop` |
| [`Lanes.tla`](Lanes.tla) | [`MCLanesCross.cfg`](MCLanesCross.cfg), [`MCLanesSelf.cfg`](MCLanesSelf.cfg), [`MCLanesOrder.cfg`](MCLanesOrder.cfg) | `src/nerve/fabric/nerve.rs` (`Fabric::{enter, release}`, `Ticket`, `Held::drop`) |
| [`Routing.tla`](Routing.tla) | [`MCRouting.cfg`](MCRouting.cfg) | `src/nerve/fabric/route.rs` (`Fabric::{drive, route, select, emit}`), `summing.rs` (`fire_summed`) |

The `MC*.tla` modules give the models their constants (the parts, their wiring, the programs).

## What is checked

**Motor hierarchy** (`Motor.tla`). A brain commands actions; a motor neuron's program breaks
its command down, step by step, into commands to the neurons below (one, or several with
`together`), and an effector acts and retries. `MCMotor` sends one command down a three-level
tree (`dress` → `arm` through a deliberative queue, then `sleeve` and `hand` together; `arm` →
`hand`), with a deadline or none, a `stop`, retries and a non-action; `MCMotorPreempt` sends
two commands to `dress`, the second preempting the first.

| Property | Kind | Says |
| --- | --- | --- |
| `OneOutcome` | action | A command ends once, in one of done, failed, timed_out, cancelled (or unrouted), and its commander hears at most once |
| `EveryCommandEnds` | liveness | No command is lost: every one ends |
| `HeardIsTrue` | invariant | What a commander hears is how the command really ended |
| `OutcomeGoesBack` | liveness | A program waiting for a command hears its outcome, unless it stops waiting first |
| `ReportGoesUp` | liveness | A brain's command that started a program is reported up the motor neuron's pathways |
| `NoWorkAfterCommanderOrDeadline` | invariant | No program takes a step and no effector tries again once its commander is gone (stopped, preempted, ended, or let it go) or its deadline has passed |
| `NoActForNobody` | invariant | No effector starts on a command whose commander is gone |
| `OneProgram` | invariant | One program runs per motor neuron |
| `TogetherStopsTheRest` | invariant | After a failure, `together` waits for none of the rest (which then stop, by the two above) |
| `ActionsOnly` | invariant | A brain's command reaches declared actions only |

**Nerve lanes** (`Lanes.tla`). Parts in two nerves, A and B, with one place each.
`MCLanesCross`: each nerve's part commands into the other, both under a deadline.
`MCLanesSelf`: a part commands into its own nerve, directly and through a motor neuron in no
nerve, without a deadline. `MCLanesOrder`: commands of different urgency queue for one part.

| Property | Kind | Says |
| --- | --- | --- |
| `Bounded` | invariant | No more places taken than `in_flight`, no more commands waiting than the lane holds |
| `CountIsTrue` | invariant | The lane's count is the places held by tickets not yet given back |
| `NoIdlePlace` | invariant | No place stays free while a command waits |
| `InsideNeverWaits` | invariant | A part's command to its own nerve never takes the lane |
| `WaitingHoldsNoPlace` | invariant | A waiting command holds no place anywhere: a flooded nerve never keeps another nerve's places taken |
| `UrgentFirst` | action | A freed place goes to the most urgent waiting command, never to one past its deadline |
| `LeastUrgentDropped` | action | A full lane drops its least urgent command, the newcomer included |
| `EveryCommandEnds`, `PlacesComeBack` | liveness | No deadlock: every command ends and every place is given back (in `MCLanesCross` thanks to the deadlines; see below) |

**Routing** (`Routing.tla`). A receptor feeds a reflex junction that loops to itself, acts,
or asks a brain through a queue; the brain commands its one action, a motor neuron, whose
reports go back to the brain through a queue. Every neuron may be quiet or fire a plain
signal or a command to any name, including an unknown one; every combination of measured
p99s and remaining times is tried.

| Property | Kind | Says |
| --- | --- | --- |
| `Terminates` | liveness | Every signal ends: the hop cut holds through queues, lanes and programs, which start the synchronous depth again but not the trace |
| `WithinHops` | invariant | No signal goes past `MAX_HOPS` |
| `HeaviestThatFits` | invariant | The pathway taken is the heaviest whose p99 fits the remaining time; none only when none fits |
| `CommandsReachActuators` | invariant | A command is delivered to a motor neuron or an effector only; to anything else it is unrouted |
| `BrainReachesActionsOnly` | invariant | A brain's command is delivered to a declared action only |

## State counts

TLC 2.19 (`tla2tools.jar` v1.7.4), every state explored:

| Model | Distinct states | Generated | Depth |
| --- | ---: | ---: | ---: |
| `MCMotor` | 4,321 | 9,622 | 26 |
| `MCMotorPreempt` | 48,824 | 110,933 | 24 |
| `MCLanesCross` | 798 | 1,294 | 17 |
| `MCLanesSelf` | 202 | 316 | 27 |
| `MCLanesOrder` | 9,984 | 18,738 | 17 |
| `MCRouting` | 69,604 | 143,072 | 10 |

## Running

With Java 11 or later and [`tla2tools.jar`](https://github.com/tlaplus/tlaplus/releases/tag/v1.7.4):

```sh
cd specs
java -cp tla2tools.jar tlc2.TLC -workers auto -lncheck final -config MCLanesCross.cfg MCLanes
```

The workflow runs the six models this way; TLC exits non-zero on a violation and prints the
behavior that leads to it.

## What the models abstract

- Time is a small clock that ticks; a deadline is a tick. Latencies are a few p99 values.
- Constants are small (one or two places per nerve, two or three commands, `MAX_HOPS` 4
  instead of 32): the properties do not depend on the sizes, and TLC sees every interleaving
  of the steps.
- A program's command that takes no queue is delivered within the program's step, as
  `Motion::send` runs the route synchronously.
- An effector's act in flight runs to its end (it is a future the effector awaits); what stops
  with the commander are the tries after it.
- Fired signals derive from the one received (`Signal::derive`), as every neuron in furnace
  does. A neuron that makes a fresh signal starts a fresh trace, and a loop of such neurons
  through a queue is not cut.
- Apalache is not used: the models are finite and TLC exhausts them; Apalache's symbolic
  checking would pay off for unbounded parameters, which these properties do not need.

## Found

The checks found these in the design as it was. The fixed ones have a regression test each.

**Fixed**

1. *A command skipped by a deliberative queue was heard as done* (`HeardIsTrue`,
   `NoActForNobody`). A deliberative worker skipped every signal whose request was no longer
   waited for (`Summing::wanted`), commands included, and dropped it without a report;
   `Motion::command` reads a closed channel as "done with nothing to say". Trace: `dress`
   commands `arm` through a queue; the HTTP client that sent the stimulus goes away; the worker
   skips the command, and `dress` hears "done" for a command never carried out. The other way
   round, a command whose program had stopped waiting was still carried out by the queue's
   worker. Now a command is skipped exactly when nobody waits for it any more.
   Test: `a_command_in_a_deliberative_queue_is_carried_out_after_the_request_is_gone`.
2. *Effectors retried after the deadline* (`NoWorkAfterCommanderOrDeadline`). The local loop
   checked whether the commander still waited before a retry, not the deadline: a brain's
   command (nobody waits on it) kept retrying past its deadline, and a program's command could
   retry between its deadline and the program's own timeout. Trace: `dress` (deadline 1) →
   `arm` → `hand` acting; the clock reaches 1; `hand` fails in a retryable way and tries
   again. Now no try starts after the deadline; the command times out.
   Test: `no_try_starts_after_the_deadline`.
3. *A waiting command kept its own nerve's place taken* (`EveryCommandEnds`, `PlacesComeBack`
   in `MCLanesCross`). A command from a part in nerve A to a part in nerve B inherits A's
   ticket, and kept it while it waited in B's lane. Trace: the hand (A) commands the foot (B)
   while the leg (B) commands the finger (A); both commands wait; both programs time out, but
   the waiting commands still hold A's and B's places, so neither lane frees again: both nerves
   are stuck for good, deadlines or not. B's flood held up A, too. Now a command entering a
   nerve from outside leaves the place it holds elsewhere behind. The loom models found the
   same change from another side, with the signals that leave a lane (dropped,
   expired, given up on) now dropped outside its lock, where freeing a place takes a lane's
   lock again; `Lanes.tla` releases a place in one step, so it does not model the lock.
   Test: `a_part_waiting_at_another_nerve_holds_no_place_in_its_own`.
4. *A part's own work coming back through a part outside the nerve waited for itself*
   (`EveryCommandEnds` in `MCLanesSelf`). Trace: the hand (A, one place) commands `spine`, a
   motor neuron in no nerve, which commands the hand's finger: the command enters A from
   outside and waits for the place the hand itself holds, for ever, or until a deadline
   (the loom models bounded it by the deadline). Now a command whose signal holds a place in the nerve
   already goes straight on, as a command from inside it does, deadline or not.
   Test: `a_command_that_leaves_a_nerve_and_comes_back_never_wedges_it` (from the loom models, now
   expecting no wait).
5. *A half-open circuit's trial given up on stayed a trial for good*, found reading the local
   loop for `Motor.tla` and fixed with the loom models for an abandoned command; a command that times out
   before its next try (2. above) clears the trial the same way.

**Open** (design decisions, not changed here)

6. *Preemption stops at a full lane* (`PreemptedAtTheLane` in `Lanes.tla`, not checked in CI).
   A newer command to a motor neuron preempts its program when it arrives, but in a nerve whose
   places are all taken it waits in the lane, behind the very program it should preempt; with
   `in_flight(1)` it can never preempt. Trace: the brain commands the hand (A, one place), and
   commands it again: the second command waits until the first program ends. Stopping the
   program when the newer command starts to wait is not enough: two commands waiting for the
   same motor neuron then run one after the other, the older first.
7. *Two parts commanding into each other deadlock without deadlines* (`EveryCommandEnds` in
   `MCLanesCross` with no deadline). The hand (A) holds A's place and waits for the foot (B);
   the leg (B) holds B's place and waits for the finger (A). With deadlines both time out and
   the lanes recover (3. above); without, nothing ends.
8. *A brain is not told when a lane drops its command*. A brain's command to a motor neuron
   that a full lane drops, or whose deadline passes while it waits, ends without a report: the
   report up the pathways comes from the program, which never started. (Not modelled: found
   reading `nerve.rs` for `Lanes.tla`.)
