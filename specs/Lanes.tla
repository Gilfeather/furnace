------------------------------- MODULE Lanes -------------------------------
(***************************************************************************)
(* Nerves and their lanes (src/nerve/fabric/nerve.rs), with the motor      *)
(* programs that hold their places (src/nerve/motor/mod.rs).               *)
(*                                                                         *)
(* Every part (a motor neuron or an effector) is in at most one nerve. A   *)
(* command that enters a nerve from outside it takes a place in the        *)
(* nerve's lane: at most Limit[x] places are taken at once, and the        *)
(* commands that find none wait, at most Capacity of them, the nearest     *)
(* deadline first. A place is held by a ticket that the command's signal   *)
(* carries and every signal derived from it inherits: the place is free    *)
(* once none of them is left (Held::drop), and the most urgent waiting     *)
(* command takes it (Fabric::release). A command from inside the nerve     *)
(* never takes the lane, nor one whose ticket is already the nerve's: that *)
(* is the nerve's own work coming back through a part outside it.          *)
(*                                                                         *)
(* A program here commands one part at a time and waits for its outcome;  *)
(* an effector's act ends on its own. Motor.tla has the rest of a program. *)
(***************************************************************************)
EXTENDS Naturals, Sequences, FiniteSets, TLC

CONSTANTS
    Nerves,         \* nerve names
    NerveOf,        \* [Parts -> Nerves \cup {Outside}]
    Outside,        \* the nerve of a part in none (a model value)
    Motors,         \* the parts that are motor neurons; the others are effectors
    Script,         \* [Motors -> Seq(Parts)]: what each program commands, in turn
    Limit,          \* [Nerves -> 1..]: Nerve::in_flight
    Capacity,       \* commands that may wait in a lane (WAITING)
    Commanded,      \* the parts the brain commands
    BrainDeadlines, \* the deadlines the brain gives (Never, MaxTime + 1, for none)
    MaxBrain,       \* commands the brain sends, at most
    MaxTime,        \* the clock runs 0..MaxTime
    MaxCmds         \* bound on the commands created

Parts == DOMAIN NerveOf
Never == MaxTime + 1
Brain == 0

Outcomes == {"done", "failed", "timed_out", "cancelled"}
Live     == {"waiting", "admitted", "running", "acting"}

VARIABLES
    now,       \* the clock
    cmd,       \* [1..MaxCmds -> command record]
    n,         \* commands created
    sent,      \* commands the brain sent
    running,   \* [Motors -> the command whose program runs there, or 0]
    inflight,  \* [Nerves -> places taken]: LaneState::running
    places,    \* [ticket -> its nerve]: the tickets handed out; a ticket is named by
               \* the command that took it
    freed,     \* the tickets whose place was given back
    admitted,  \* the command a release just let in from a lane, or 0
    dropped    \* the command a full lane just dropped, or 0

vars == <<now, cmd, n, sent, running, inflight, places, freed, admitted, dropped>>

Ids == 1..MaxCmds

(* A command: target, commander (Brain, or the command whose program sent it),      *)
(* deadline, state, the next step of its program (pc), the command it waits for     *)
(* (waits, 0 for none), what its commander heard (heard), and the ticket its signal *)
(* carries (holds, 0 for none). The id is the order of arrival.                      *)
(*   st: "free" | "waiting" (in a lane) | "admitted" (let in from the lane, on its   *)
(*       way: the spawned Fabric::deliver) | "running" (a program) | "acting" (an    *)
(*       effector's act) | an outcome                                                *)
Free == [target |-> "", parent |-> Brain, deadline |-> Never, st |-> "free",
         pc |-> 1, waits |-> 0, heard |-> "none", holds |-> 0]

TypeOK ==
    /\ now \in 0..MaxTime
    /\ n \in 0..MaxCmds
    /\ running \in [Motors -> 0..MaxCmds]
    /\ inflight \in [Nerves -> Nat]
    /\ DOMAIN places \subseteq Ids
    /\ freed \subseteq DOMAIN places
    /\ admitted \in 0..MaxCmds
    /\ dropped \in 0..MaxCmds
    /\ \A c \in Ids :
        /\ cmd[c].st \in {"free"} \cup Live \cup Outcomes
        /\ cmd[c].holds \in 0..MaxCmds

Init ==
    /\ now = 0
    /\ cmd = [c \in Ids |-> Free]
    /\ n = 0
    /\ sent = 0
    /\ running = [m \in Motors |-> 0]
    /\ inflight = [x \in Nerves |-> 0]
    /\ places = <<>>
    /\ freed = {}
    /\ admitted = 0
    /\ dropped = 0

-----------------------------------------------------------------------------
(* Reading a commands function f (the state's cmd, or one being built in a step) *)

Target(f, c) == f[c].target
Parent(f, c) == f[c].parent
NerveOfCmd(f, c) == NerveOf[f[c].target]
Waiting(f, x) == {c \in Ids : f[c].st = "waiting" /\ NerveOfCmd(f, c) = x}

\* Its commander still waits for it (see Motor.tla)
Waited(f, c) ==
    IF Parent(f, c) = Brain THEN TRUE
    ELSE f[Parent(f, c)].st = "running" /\ f[Parent(f, c)].waits = c

\* Urgency order (nerve.rs Key): a deadline first, the nearest first, then none; each
\* in order of arrival
Before(f, c, d) ==
    \/ f[c].deadline < f[d].deadline
    \/ f[c].deadline = f[d].deadline /\ c < d

\* The most and the least urgent of a nonempty set
First(f, S) == CHOOSE c \in S : \A d \in S : d = c \/ Before(f, c, d)
Last(f, S)  == CHOOSE c \in S : \A d \in S : d = c \/ Before(f, d, c)

\* The nerve of a ticket, Outside for none
TicketNerve(t) == IF t \in DOMAIN places THEN places[t] ELSE Outside

-----------------------------------------------------------------------------
(* Steps on a commands function f and the running map r, as pairs <<f, r>> *)

\* c ends with o; its commander hears it if it still waits
End(f, c, o) ==
    [f EXCEPT ![c].st = o, ![c].waits = 0,
              ![c].heard = IF Parent(f, c) # Brain /\ Waited(f, c) THEN o ELSE "none"]

\* c, a program, ends: its motor neuron is free
EndRun(fr, c, o) ==
    <<End(fr[1], c, o),
      [fr[2] EXCEPT ![Target(fr[1], c)] = IF @ = c THEN 0 ELSE @]>>

\* c goes on into its part (Fabric::drive): a motor neuron's program starts and
\* preempts the one running there; an effector acts
Arrive(fr, c) ==
    LET f == fr[1]
        r == fr[2]
        t == Target(f, c)
    IN IF t \in Motors
       THEN LET g == IF r[t] # 0 THEN End(f, r[t], "cancelled") ELSE f
            IN <<[g EXCEPT ![c].st = "running", ![c].pc = 1], [r EXCEPT ![t] = c]>>
       ELSE <<[f EXCEPT ![c].st = "acting"], r>>

\* Fabric::enter for the new command c, already in f: sent from a part in nerve `from`
\* whose signal held ticket `held` (the brain: Outside and 0). Sets cmd', running',
\* inflight', places' and dropped'.
Enter(f0, c, from, held) ==
    LET x == NerveOfCmd(f0, c)
        f == [f0 EXCEPT ![c].holds = held]
    IN
    IF x = Outside \/ x = from \/ TicketNerve(held) = x
    THEN \* In no nerve, from inside it, or the nerve's own work coming back: on
         LET a == Arrive(<<f, running>>, c)
         IN /\ cmd' = a[1] /\ running' = a[2]
            /\ UNCHANGED <<inflight, places>>
            /\ dropped' = 0
    ELSE IF inflight[x] < Limit[x] /\ Waiting(f, x) = {}
    THEN \* A place is free: it takes it, with a ticket of its own
         LET a == Arrive(<<[f EXCEPT ![c].holds = c], running>>, c)
         IN /\ cmd' = a[1] /\ running' = a[2]
            /\ inflight' = [inflight EXCEPT ![x] = @ + 1]
            /\ places' = places @@ (c :> x)
            /\ dropped' = 0
    ELSE \* It waits, and holds no place anywhere while it waits
         LET w == [f EXCEPT ![c].holds = 0, ![c].st = "waiting"]
             full == Cardinality(Waiting(f, x)) >= Capacity
             \* A full lane drops the least urgent, the newcomer included
             least == Last(w, Waiting(w, x))
         IN /\ cmd' = IF full THEN End(w, least, "failed") ELSE w
            /\ dropped' = IF full THEN least ELSE 0
            /\ UNCHANGED <<running, inflight, places>>

-----------------------------------------------------------------------------
(* The brain, and the clock *)

BrainCommands(t, d) ==
    /\ sent < MaxBrain
    /\ n < MaxCmds
    /\ Enter([cmd EXCEPT ![n + 1] = [Free EXCEPT !.target = t, !.deadline = d]],
             n + 1, Outside, 0)
    /\ n' = n + 1
    /\ sent' = sent + 1
    /\ admitted' = 0
    /\ UNCHANGED <<now, freed>>

Tick ==
    /\ now < MaxTime
    /\ now' = now + 1
    /\ admitted' = 0 /\ dropped' = 0
    /\ UNCHANGED <<cmd, n, sent, running, inflight, places, freed>>

-----------------------------------------------------------------------------
(* A program (Started::run): it may step while it runs, its commander waits and *)
(* its deadline has not passed                                                  *)

LiveRun(c) ==
    /\ cmd[c].st = "running"
    /\ Waited(cmd, c)
    /\ now < cmd[c].deadline

\* Command the next part (Motion::command): the command keeps this one's deadline,
\* and its signal this one's ticket
Step(c) ==
    LET m == Target(cmd, c)
        d == n + 1
    IN
    /\ LiveRun(c)
    /\ cmd[c].waits = 0
    /\ cmd[c].pc <= Len(Script[m])
    /\ n < MaxCmds
    /\ Enter([cmd EXCEPT ![c].pc = @ + 1, ![c].waits = d,
                         ![d] = [Free EXCEPT !.target = Script[m][cmd[c].pc],
                                             !.parent = c,
                                             !.deadline = cmd[c].deadline]],
             d, NerveOf[m], cmd[c].holds)
    /\ n' = n + 1
    /\ admitted' = 0
    /\ UNCHANGED <<now, sent, freed>>

\* The outcome arrives: done, go on; otherwise return it
Hear(c) ==
    LET d == cmd[c].waits IN
    /\ LiveRun(c)
    /\ d # 0
    /\ cmd[d].heard # "none"
    /\ IF cmd[d].heard = "done"
       THEN cmd' = [cmd EXCEPT ![c].waits = 0] /\ UNCHANGED running
       ELSE LET e == EndRun(<<cmd, running>>, c, cmd[d].heard)
            IN cmd' = e[1] /\ running' = e[2]
    /\ admitted' = 0 /\ dropped' = 0
    /\ UNCHANGED <<now, n, sent, inflight, places, freed>>

Finish(c) ==
    /\ LiveRun(c)
    /\ cmd[c].waits = 0
    /\ cmd[c].pc > Len(Script[Target(cmd, c)])
    /\ LET e == EndRun(<<cmd, running>>, c, "done") IN cmd' = e[1] /\ running' = e[2]
    /\ admitted' = 0 /\ dropped' = 0
    /\ UNCHANGED <<now, n, sent, inflight, places, freed>>

\* The select: abandoned before expired
Abandon(c) ==
    /\ cmd[c].st = "running"
    /\ ~Waited(cmd, c)
    /\ LET e == EndRun(<<cmd, running>>, c, "cancelled") IN cmd' = e[1] /\ running' = e[2]
    /\ admitted' = 0 /\ dropped' = 0
    /\ UNCHANGED <<now, n, sent, inflight, places, freed>>

Expire(c) ==
    /\ cmd[c].st = "running"
    /\ Waited(cmd, c)
    /\ now >= cmd[c].deadline
    /\ LET e == EndRun(<<cmd, running>>, c, "timed_out") IN cmd' = e[1] /\ running' = e[2]
    /\ admitted' = 0 /\ dropped' = 0
    /\ UNCHANGED <<now, n, sent, inflight, places, freed>>

\* An effector's act ends
Act(c) ==
    /\ cmd[c].st = "acting"
    /\ cmd' = End(cmd, c, "done")
    /\ admitted' = 0 /\ dropped' = 0
    /\ UNCHANGED <<now, n, sent, running, inflight, places, freed>>

\* The deliver spawned by a release runs
Deliver(c) ==
    /\ cmd[c].st = "admitted"
    /\ LET a == Arrive(<<cmd, running>>, c) IN cmd' = a[1] /\ running' = a[2]
    /\ admitted' = 0 /\ dropped' = 0
    /\ UNCHANGED <<now, n, sent, inflight, places, freed>>

-----------------------------------------------------------------------------
(* Held::drop and Fabric::release *)

\* Who keeps ticket t's place: the commands whose signals carry it, until they end
Holders(t) == {c \in Ids : cmd[c].holds = t /\ cmd[c].st \in Live}

\* The last holder of t is gone: the place is free, and the most urgent command
\* waiting takes it. On the way, a command whose deadline passed times out and one
\* nobody waits for any more is forgotten.
RECURSIVE Admit(_, _)
Admit(f, x) ==
    IF Waiting(f, x) = {} THEN <<f, 0>>
    ELSE LET c == First(f, Waiting(f, x)) IN
         IF now >= f[c].deadline THEN Admit(End(f, c, "timed_out"), x)
         ELSE IF ~Waited(f, c) THEN Admit(End(f, c, "cancelled"), x)
         ELSE <<[f EXCEPT ![c].st = "admitted", ![c].holds = c], c>>

Release(t) ==
    LET x == places[t]
        a == Admit(cmd, x)
    IN
    /\ t \in DOMAIN places \ freed
    /\ Holders(t) = {}
    /\ freed' = freed \cup {t}
    /\ cmd' = a[1]
    /\ admitted' = a[2]
    /\ IF a[2] # 0
       THEN /\ places' = places @@ (a[2] :> x)
            /\ UNCHANGED inflight
       ELSE /\ inflight' = [inflight EXCEPT ![x] = @ - 1]
            /\ UNCHANGED places
    /\ dropped' = 0
    /\ UNCHANGED <<now, n, sent, running>>

-----------------------------------------------------------------------------

Next ==
    \/ \E t \in Commanded, d \in BrainDeadlines : BrainCommands(t, d)
    \/ Tick
    \/ \E c \in Ids :
        Step(c) \/ Hear(c) \/ Finish(c) \/ Abandon(c) \/ Expire(c) \/ Act(c) \/ Deliver(c)
    \/ \E t \in DOMAIN places : Release(t)

\* Everything is bounded: weak fairness on Next is fairness to every part (Motor.tla)
Spec == Init /\ [][Next]_vars /\ WF_vars(Next)

-----------------------------------------------------------------------------
(* Properties *)

\* No more places taken than the nerve has, and no more commands waiting than the lane holds
Bounded ==
    \A x \in Nerves :
        /\ inflight[x] <= Limit[x]
        /\ Cardinality(Waiting(cmd, x)) <= Capacity

\* The count is the places held: one per ticket not given back
CountIsTrue ==
    \A x \in Nerves :
        inflight[x] = Cardinality({t \in DOMAIN places \ freed : places[t] = x})

\* No place is idle while a command waits
NoIdlePlace ==
    \A x \in Nerves : Waiting(cmd, x) # {} => inflight[x] = Limit[x]

\* A part's command to its own nerve never waits there
InsideNeverWaits ==
    \A c \in Ids :
        cmd[c].st = "waiting" /\ Parent(cmd, c) # Brain =>
            NerveOf[Target(cmd, Parent(cmd, c))] # NerveOfCmd(cmd, c)

\* A command waiting in a lane holds no place: so a flooded nerve, whose commands wait,
\* never keeps another nerve's places taken
WaitingHoldsNoPlace ==
    \A c \in Ids : cmd[c].st = "waiting" => cmd[c].holds = 0

\* The place that frees goes to the most urgent command waiting, unless its deadline
\* passed
UrgentFirst ==
    [][admitted' # 0 =>
        LET c == admitted'
            x == NerveOfCmd(cmd, c)
        IN /\ now < cmd[c].deadline
           /\ \A d \in Waiting(cmd', x) : Before(cmd, c, d)]_vars

\* A full lane drops the least urgent, the newcomer included
LeastUrgentDropped ==
    [][dropped' # 0 =>
        LET c == dropped'
            x == NerveOfCmd(cmd', c)
        IN \A d \in Waiting(cmd', x) : Before(cmd', d, c)]_vars

\* A newer command to a motor neuron preempts the program running there. Not checked:
\* it does not hold when the nerve is full (see README.md, "Found").
PreemptedAtTheLane ==
    \A c \in Ids :
        (cmd[c].st = "waiting" /\ Target(cmd, c) \in Motors) =>
            running[Target(cmd, c)] \in {0} \cup (c + 1)..MaxCmds

\* No deadlock: every command ends, and every place is given back
EveryCommandEnds == \A c \in Ids : cmd[c].st \in Live ~> cmd[c].st \in Outcomes

PlacesComeBack == \A x \in Nerves : <>[](inflight[x] = 0)

=============================================================================
