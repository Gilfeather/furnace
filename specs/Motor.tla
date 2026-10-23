------------------------------- MODULE Motor -------------------------------
(***************************************************************************)
(* The motor hierarchy (src/nerve/motor/, src/nerve/effector/local_loop.rs) *)
(*                                                                         *)
(* A brain commands actions. A command to a motor neuron starts its        *)
(* program, which breaks the command down step by step: each step commands *)
(* one neuron below (Motion::command) or several at once (Motion::together) *)
(* and waits for their outcomes. A command to an effector is an act, tried *)
(* again with backoff while it fails in a retryable way.                   *)
(*                                                                         *)
(* One program runs per motor neuron: a newer command preempts it, and     *)
(* Fabric::stop stops it. A program ends when it is done or fails, when it *)
(* is stopped, when its commander stops waiting, or when its deadline      *)
(* passes (Started::run's biased select, in that order). A command below   *)
(* keeps its commander's deadline.                                         *)
(*                                                                         *)
(* A command between two neurons travels directly, or through a            *)
(* deliberative pathway's queue (Queued), whose worker may run it later.   *)
(***************************************************************************)
EXTENDS Naturals, Sequences, FiniteSets

CONSTANTS
    Motors,       \* motor neuron names
    Effectors,    \* effector names
    Script,       \* [Motors -> Seq(SUBSET (Motors \cup Effectors))]: each step's targets
    Actions,      \* the neurons a brain may command (FabricBuilder::action)
    Commanded,    \* the names the brain sends commands to: actions, and others
    Queued,       \* pairs <<from, to>> joined by a deliberative pathway
    MaxBrain,     \* commands the brain sends, at most
    MaxStops,     \* Fabric::stop calls, at most
    MaxAttempts,  \* an effector's attempts per command (Retry::attempts)
    MaxTime,      \* the clock runs 0..MaxTime
    Deadlines,    \* deadlines the brain may give, in 1..MaxTime
    MaxCmds       \* bound on the commands created

Neurons == Motors \cup Effectors
Brain   == 0            \* the commander of the brain's commands
Never   == MaxTime + 1  \* no deadline

\* How a command ends. "unrouted" is a brain's command to a non-action: it is never
\* delivered (MotionError::Unrouted, a failure).
Outcomes == {"done", "failed", "timed_out", "cancelled"}
Ended    == Outcomes \cup {"unrouted"}

VARIABLES
    now,        \* the clock
    cmd,        \* [1..MaxCmds -> command record]
    n,          \* commands created
    running,    \* [Motors -> the command whose program runs there, or 0]
    sent,       \* commands the brain sent
    stops,      \* stops the brain made
    upward,     \* brain commands to motor neurons whose report went up the pathways
    work,       \* the command that just did work: a program's step or an effector's retry
    first       \* the command whose first attempt an effector just made

vars == <<now, cmd, n, running, sent, stops, upward, work, first>>

Ids == 1..MaxCmds

(* A command: its target, its commander (Brain or the command whose program sent it), *)
(* its deadline, its state, and what its commander heard of its outcome.              *)
(*   st:      "free" | "sent" | "running" (a program) | "acting" | "backoff" (an act)  *)
(*            | an outcome in Ended                                                   *)
(*   pc:      the next step of a program                                              *)
(*   waits:   the commands below that a program waits for in its current step         *)
(*   tries:   an act's attempts so far                                                *)
(*   heard:   "none" or the outcome the commanding program received                  *)
Free == [target |-> "", parent |-> 0, deadline |-> Never, st |-> "free",
         pc |-> 1, waits |-> {}, tries |-> 0, heard |-> "none"]

TypeOK ==
    /\ now \in 0..MaxTime
    /\ n \in 0..MaxCmds
    /\ running \in [Motors -> 0..MaxCmds]
    /\ upward \subseteq Ids
    /\ work \in 0..MaxCmds
    /\ first \in 0..MaxCmds
    /\ \A c \in Ids :
        /\ cmd[c].st \in {"free", "sent", "running", "acting", "backoff"} \cup Ended
        /\ cmd[c].waits \subseteq Ids
        /\ cmd[c].heard \in {"none"} \cup Outcomes

Init ==
    /\ now = 0
    /\ cmd = [c \in Ids |-> Free]
    /\ n = 0
    /\ running = [m \in Motors |-> 0]
    /\ sent = 0
    /\ stops = 0
    /\ upward = {}
    /\ work = 0
    /\ first = 0

Target(c) == cmd[c].target
Parent(c) == cmd[c].parent

\* Whoever commanded c still waits for its outcome: the brain (it has no way to stop
\* waiting), or a program still running and waiting for c in its current step. Once
\* the program has ended or let c go (Motion::together after a failure), the receiver
\* is dropped and the command is abandoned.
Waited(c) ==
    IF Parent(c) = Brain THEN TRUE
    ELSE cmd[Parent(c)].st = "running" /\ c \in cmd[Parent(c)].waits

Abandoned(c) == ~Waited(c)

From(c) == IF Parent(c) = Brain THEN "brain" ELSE Target(Parent(c))

ViaQueue(c) == <<From(c), Target(c)>> \in Queued

-----------------------------------------------------------------------------
(* Ending a command *)

\* c ends with outcome o: its commander hears it if it still waits (Signal::report,
\* Started::run); a program that ends drops its receivers, abandoning what it waits for.
EndCmd(c, o) ==
    [cmd EXCEPT ![c].st = o,
                ![c].waits = {},
                ![c].heard = IF Parent(c) # Brain /\ Waited(c) THEN o ELSE "none"]

\* A program ends: the motor neuron is free again, and a brain's command reports up.
EndProgram(c, o) ==
    /\ cmd' = EndCmd(c, o)
    /\ running' = [running EXCEPT ![Target(c)] = IF @ = c THEN 0 ELSE @]
    /\ upward' = IF Parent(c) = Brain THEN upward \cup {c} ELSE upward

-----------------------------------------------------------------------------
(* The brain *)

\* The brain commands any neuron by name: routing lets through the declared actions only.
BrainCommands(t, d) ==
    /\ sent < MaxBrain
    /\ n < MaxCmds
    /\ cmd' = [cmd EXCEPT ![n + 1] = [Free EXCEPT !.target = t, !.parent = Brain,
                                                  !.deadline = d, !.st = "sent"]]
    /\ n' = n + 1
    /\ sent' = sent + 1
    /\ work' = 0 /\ first' = 0
    /\ UNCHANGED <<now, running, stops, upward>>

\* Fabric::stop: the program running at m is stopped, and so everything it waits for.
Stop(m) ==
    /\ stops < MaxStops
    /\ running[m] # 0
    /\ EndProgram(running[m], "cancelled")
    /\ stops' = stops + 1
    /\ work' = 0 /\ first' = 0
    /\ UNCHANGED <<now, n, sent>>

Tick ==
    /\ now < MaxTime
    /\ now' = now + 1
    /\ work' = 0 /\ first' = 0
    /\ UNCHANGED <<cmd, n, running, sent, stops, upward>>

-----------------------------------------------------------------------------
(* Delivery: route.rs, and the deliberative worker (summing.rs) *)

\* A deliberative worker skips the commands nobody waits for any more (Summing::wanted).
\* Whether the request behind the stimulus is still waited for does not matter to a
\* command, so it is not modelled.
Skipped(c) == ViaQueue(c) /\ Abandoned(c)

Deliver(c) ==
    /\ cmd[c].st = "sent"
    /\ work' = 0
    /\ UNCHANGED <<now, n, sent, stops>>
    /\ IF Parent(c) = Brain /\ Target(c) \notin Actions
       THEN \* route.rs: a brain commands the actions it was given, never the parts below
            /\ cmd' = [cmd EXCEPT ![c].st = "unrouted"]
            /\ first' = 0
            /\ UNCHANGED <<running, upward>>
       ELSE IF Skipped(c)
       THEN \* Dropped: nobody hears of it
            /\ cmd' = EndCmd(c, "cancelled")
            /\ first' = 0
            /\ UNCHANGED <<running, upward>>
       ELSE IF Target(c) \in Motors
       THEN \* Motor::stimulate: the program running is preempted and the new one starts
            LET m == Target(c)
                old == running[m]
                stopped == IF old # 0 THEN EndCmd(old, "cancelled") ELSE cmd
            IN /\ cmd' = [stopped EXCEPT ![c].st = "running", ![c].pc = 1]
               /\ running' = [running EXCEPT ![m] = c]
               /\ upward' = IF old # 0 /\ Parent(old) = Brain
                            THEN upward \cup {old} ELSE upward
               /\ first' = 0
       ELSE \* Effector::stimulate: the first attempt
            /\ cmd' = [cmd EXCEPT ![c].st = "acting", ![c].tries = 1]
            /\ first' = c
            /\ UNCHANGED <<running, upward>>

-----------------------------------------------------------------------------
(* A motor program (Started::run) *)

\* The program may take a step: not stopped (still running), its commander waits, and
\* its deadline has not passed. Otherwise the select ends it first.
Live(c) ==
    /\ cmd[c].st = "running"
    /\ Waited(c)
    /\ now < cmd[c].deadline

\* The next step: command its targets, all at once (one for Motion::command).
Step(c) ==
    LET m == Target(c)
        targets == Script[m][cmd[c].pc]
        k == Cardinality(targets)
        \* the new commands' ids, n+1..n+k, one per target
        order == CHOOSE f \in [1..k -> targets] : \A i, j \in 1..k : i # j => f[i] # f[j]
    IN
    /\ Live(c)
    /\ cmd[c].waits = {}
    /\ cmd[c].pc <= Len(Script[m])
    /\ n + k <= MaxCmds
    /\ cmd' = [d \in Ids |->
                 IF d = c THEN [cmd[c] EXCEPT !.pc = @ + 1, !.waits = (n + 1)..(n + k)]
                 ELSE IF d \in (n + 1)..(n + k)
                      THEN [Free EXCEPT !.target = order[d - n], !.parent = c,
                                        !.deadline = cmd[c].deadline, !.st = "sent"]
                      ELSE cmd[d]]
    /\ n' = n + k
    /\ work' = c /\ first' = 0
    /\ UNCHANGED <<now, running, sent, stops, upward>>

\* An outcome arrives. Done: wait for the rest. Otherwise the program returns that
\* error (the `?` after command and together); together drops the rest.
Hear(c) ==
    /\ Live(c)
    /\ \E d \in cmd[c].waits :
        /\ cmd[d].heard # "none"
        /\ IF cmd[d].heard = "done"
           THEN /\ cmd' = [cmd EXCEPT ![c].waits = @ \ {d}]
                /\ UNCHANGED <<running, upward>>
           ELSE EndProgram(c, cmd[d].heard)
    /\ work' = 0 /\ first' = 0
    /\ UNCHANGED <<now, n, sent, stops>>

\* Every step done: the program returns its value.
Finish(c) ==
    /\ Live(c)
    /\ cmd[c].waits = {}
    /\ cmd[c].pc > Len(Script[Target(c)])
    /\ EndProgram(c, "done")
    /\ work' = 0 /\ first' = 0
    /\ UNCHANGED <<now, n, sent, stops>>

\* The program gives up on its own.
GiveUp(c) ==
    /\ Live(c)
    /\ EndProgram(c, "failed")
    /\ work' = 0 /\ first' = 0
    /\ UNCHANGED <<now, n, sent, stops>>

\* Started::run's select: the commander stopped waiting, or the deadline passed.
Abandon(c) ==
    /\ cmd[c].st = "running"
    /\ Abandoned(c)
    /\ EndProgram(c, "cancelled")
    /\ work' = 0 /\ first' = 0
    /\ UNCHANGED <<now, n, sent, stops>>

Expire(c) ==
    /\ cmd[c].st = "running"
    /\ Waited(c)
    /\ now >= cmd[c].deadline
    /\ EndProgram(c, "timed_out")
    /\ work' = 0 /\ first' = 0
    /\ UNCHANGED <<now, n, sent, stops>>

-----------------------------------------------------------------------------
(* An effector's local loop (local_loop.rs::run) *)

\* An attempt ends: done, failed for good, or worth another try.
Act(c) ==
    /\ cmd[c].st = "acting"
    /\ \/ cmd' = EndCmd(c, "done")
       \/ cmd' = EndCmd(c, "failed")
       \/ cmd' = IF cmd[c].tries < MaxAttempts
                 THEN [cmd EXCEPT ![c].st = "backoff"]
                 ELSE EndCmd(c, "failed")
    /\ work' = 0 /\ first' = 0
    /\ UNCHANGED <<now, n, running, sent, stops, upward>>

\* After the backoff: nobody waits any more (it ends without a report), the deadline
\* has passed (it times out), or try again.
Retry(c) ==
    /\ cmd[c].st = "backoff"
    /\ first' = 0
    /\ IF Abandoned(c)
       THEN /\ cmd' = EndCmd(c, "cancelled")
            /\ work' = 0
       ELSE IF now >= cmd[c].deadline
       THEN /\ cmd' = EndCmd(c, "timed_out")
            /\ work' = 0
       ELSE /\ cmd' = [cmd EXCEPT ![c].st = "acting", ![c].tries = @ + 1]
            /\ work' = c
    /\ UNCHANGED <<now, n, running, sent, stops, upward>>

-----------------------------------------------------------------------------

\* A program's command that takes no queue is delivered within the program's step
\* (Motion::send runs the route synchronously): nothing happens in between.
Direct(c) == cmd[c].st = "sent" /\ Parent(c) # Brain /\ ~ViaQueue(c)

Settled == \A c \in Ids : ~Direct(c)

Next ==
    \/ \E c \in Ids : (Settled \/ Direct(c)) /\ Deliver(c)
    \/ /\ Settled
       /\ \/ \E t \in Commanded, d \in Deadlines \cup {Never} : BrainCommands(t, d)
          \/ \E m \in Motors : Stop(m)
          \/ Tick
          \/ \E c \in Ids :
              \/ Step(c) \/ Hear(c) \/ Finish(c) \/ GiveUp(c) \/ Abandon(c) \/ Expire(c)
              \/ Act(c) \/ Retry(c)

\* Whatever can move eventually does. Every action is bounded (commands, stops, the
\* clock, attempts), so a behavior ends in a state where nothing can move: weak fairness
\* on Next is fairness to every part, and the properties below hold in those end states.
\* The clock runs to MaxTime in them, but a command without a deadline ends anyway.
Spec == Init /\ [][Next]_vars /\ WF_vars(Next)

-----------------------------------------------------------------------------
(* Properties *)

IsEnded(c) == cmd[c].st \in Ended

\* An ended command stays as it ended, and its commander hears at most once.
OneOutcome ==
    [][\A c \in Ids :
         /\ IsEnded(c) => cmd'[c].st = cmd[c].st
         /\ cmd[c].heard # "none" => cmd'[c].heard = cmd[c].heard]_vars

\* What a commander hears is how the command really ended.
HeardIsTrue ==
    \A c \in Ids : cmd[c].heard # "none" => cmd[c].heard = cmd[c].st

\* Every command ends (no command is lost).
EveryCommandEnds ==
    \A c \in Ids : cmd[c].st # "free" ~> IsEnded(c)

\* A program that waits for a command hears its outcome, unless it stops waiting first.
InWaits(c) == IF Parent(c) = Brain THEN FALSE ELSE c \in cmd[Parent(c)].waits

OutcomeGoesBack ==
    \A c \in Ids : (cmd[c].st # "free" /\ InWaits(c)) ~> (cmd[c].heard # "none" \/ ~InWaits(c))

\* A brain's command that started a program is reported up the pathways when it ends.
ReportGoesUp ==
    \A c \in Ids :
        (cmd[c].st = "running" /\ Parent(c) = Brain) ~> (c \in upward)

\* No program takes a step, and no effector tries again, once its commander is gone
\* (stopped, preempted, ended, or let it go) or once its deadline has passed.
NoWorkAfterCommanderOrDeadline ==
    work # 0 => /\ Waited(work)
                /\ now < cmd[work].deadline

\* No effector starts on a command whose commander is gone.
NoActForNobody ==
    first # 0 => Waited(first)

\* One program per motor neuron, and it is the one running there.
OneProgram ==
    \A c \in Ids : cmd[c].st = "running" <=> (Target(c) \in Motors /\ running[Target(c)] = c)

\* Motion::together: after a failure, the program waits for none of the rest.
TogetherStopsTheRest ==
    \A c \in Ids : cmd[c].st \in Outcomes \ {"done"} => cmd[c].waits = {}

\* A brain's command reaches declared actions only.
ActionsOnly ==
    \A c \in Ids :
        (Parent(c) = Brain /\ cmd[c].st \notin {"free", "sent"}) =>
            (Target(c) \in Actions \/ cmd[c].st = "unrouted")

=============================================================================
