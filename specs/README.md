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
| [`Lanes.tla`](Lanes.tla) | [`MCLanesCross.cfg`](MCLanesCross.cfg), [`MCLanesCycle.cfg`](MCLanesCycle.cfg), [`MCLanesFull.cfg`](MCLanesFull.cfg), [`MCLanesOrder.cfg`](MCLanesOrder.cfg) | `src/nerve/fabric/nerve.rs` (`Fabric::{enter, release}`, `Ticket`, `Held::drop`), with `Motor::stimulate` |
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

**Nerve lanes** (`Lanes.tla`). A lane counts effector acts only: every act entering a nerve
takes a place, from inside the part or outside, and an act never waits for another command.
Motor programs never take a place; the latest command to a motor neuron stops its program at
once. No model has a deadline but `MCLanesOrder`, so nothing ends because time ran out.
`MCLanesCross`: two parts in nerves A and B, one place each, command each other's acts while
the brain floods both lanes. `MCLanesCycle`: two motor neurons command each other round and
round. `MCLanesFull`: commands to a part whose lane is full. `MCLanesOrder`: acts of different
urgency queue for one effector.

| Property | Kind | Says |
| --- | --- | --- |
| `Bounded` | invariant | No more places taken than `in_flight`, no more acts waiting than the lane holds |
| `CountIsTrue` | invariant | The lane's count is the acts holding a place |
| `NoIdlePlace` | invariant | No place stays free while an act waits |
| `ProgramsNeverWait` | invariant | Only acts wait in a lane or hold a place, never a program |
| `LatestWins` | invariant | One program per motor neuron, the latest command's: it starts at once, full lane or not, stopping the one before |
| `HeardIsTrue` | invariant | What a commander hears is how the command really ended |
| `OneOutcome` | action | A command ends once, and its commander hears at most once |
| `UrgentFirst` | action | A freed place goes to the most urgent waiting act, never to one past its deadline |
| `LeastUrgentDropped` | action | A full lane drops its least urgent act, the newcomer included, which fails, and its program hears it |
| `EveryCommandEnds`, `PlacesComeBack` | liveness | No deadlock, without deadlines: every command ends and every place comes back |
| `ReportGoesUp` | liveness | Every brain command to a motor neuron reports up how it ended |
| `DropReachesBrain` | liveness | A failed act's failure (a dropped one's among them) goes up to the brain through the program that commanded it: as `failed`, or `cancelled` if a newer command preempted the program first |

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
| `MCLanesCross` | 49,884 | 116,914 | 17 |
| `MCLanesCycle` | 698 | 1,126 | 13 |
| `MCLanesFull` | 38,152 | 70,955 | 22 |
| `MCLanesOrder` | 27,195 | 54,204 | 13 |
| `MCRouting` | 69,604 | 143,072 | 10 |

## Running

With Java 11 or later and [`tla2tools.jar`](https://github.com/tlaplus/tlaplus/releases/tag/v1.7.4):

```sh
cd specs
java -cp tla2tools.jar tlc2.TLC -workers auto -lncheck final -config MCLanesCross.cfg MCLanes
```

The workflow runs the seven models this way; TLC exits non-zero on a violation and prints the
behavior that leads to it.

## What the models abstract

- Time is a small clock that ticks; a deadline is a tick. Latencies are a few p99 values.
- Constants are small (one or two places per nerve, a few commands, `MAX_HOPS` 4
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
3. *Lanes counted motor programs, which deadlocked and blocked preemption*. A command taken
   held its nerve's place until its program ended, everything it commanded inside the nerve
   included. The models found, in turn:
   - a command waiting in another nerve's lane kept its own nerve's place: two parts
     commanding each other's nerves stuck both lanes for good, deadlines or not (the loom
     models found the same from another side, and moved the drops out of the lane's lock);
   - a part's own work coming back to its nerve through a motor neuron in no nerve waited for
     the place the part itself held;
   - two parts commanding into each other deadlocked without deadlines: each program held
     its nerve's place and waited for the other's;
   - a newer command to a motor neuron in a full nerve waited behind the program it should
     preempt (with `in_flight(1)` it never could), and stopping the program when the newer
     command started to wait still ran two waiting commands oldest first;
   - a brain's command to a motor neuron that a lane dropped, or that timed out waiting,
     ended with no report: the report comes from the program, which never started.

   Now a lane counts effector acts only, from inside the part or outside, and an act waits for
   no other command and always ends. Programs never wait in a lane: the latest command stops
   the running program at once. A dropped act fails, and the program that commanded it reports
   the failure up like any outcome. `MCLanesCross`, `MCLanesCycle` and `MCLanesFull` check it
   without deadlines.
   Tests: `two_parts_commanding_each_other_never_deadlock`,
   `motor_neurons_commanding_each_other_in_a_cycle_end`,
   `a_newer_command_preempts_at_once_when_the_lane_is_full`,
   `an_act_a_full_lane_drops_fails_and_the_brain_hears_it`,
   `inside_a_nerve_acts_take_the_lane_and_programs_do_not`,
   `a_part_waiting_at_another_nerve_holds_no_place_in_its_own`,
   `a_command_that_leaves_a_nerve_and_comes_back_never_wedges_it`.
4. *A half-open circuit's trial given up on stayed a trial for good*, found reading the local
   loop for `Motor.tla` and fixed with the loom models for an abandoned command; a command
   that times out before its next try (2. above) clears the trial the same way.
**Not covered**

- An effector a brain commands directly reports to nobody, whether its act is done, failed or
  dropped by its lane: only a program hears an act's outcome. Declare a motor neuron as the
  action for an act whose outcome the brain should hear.
