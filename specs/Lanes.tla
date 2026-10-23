------------------------------- MODULE Lanes -------------------------------
(***************************************************************************)
(* Nerves and their lanes (src/nerve/fabric/nerve.rs), with the motor      *)
(* programs that command through them (src/nerve/motor/mod.rs).            *)
(*                                                                         *)
(* A nerve's lane counts effector acts, nothing else. Every act entering a *)
(* nerve takes a place in its lane, whoever sent it, from inside the part  *)
(* or outside: at most Limit[x] acts are carried out at once, and the ones *)
(* that find no place wait, at most Capacity of them, the nearest deadline *)
(* first. A full lane drops its least urgent act, which fails: its         *)
(* commander hears it as any failure. An act never waits for another       *)
(* command and always ends, so the places it holds always come back.       *)
(*                                                                         *)
(* A motor program never takes a place and never waits in a lane. One     *)
(* program runs per motor neuron and the latest command wins: it stops the *)
(* program running there at once (its commander hears "cancelled") and    *)
(* starts its own. A program commands one neuron at a time and waits for   *)
(* its outcome; Motor.tla has the rest of a program. A command is cut      *)
(* after MaxHops neurons, as route.rs cuts every signal.                   *)
(***************************************************************************)
EXTENDS Naturals, Sequences, FiniteSets

CONSTANTS
    Nerves,         \* nerve names
    NerveOf,        \* [Parts -> Nerves \cup {Outside}]
    Outside,        \* the nerve of a part in none (a model value)
    Motors,         \* the parts that are motor neurons; the others are effectors
    Script,         \* [Motors -> Seq(Parts)]: what each program commands, in turn
    Limit,          \* [Nerves -> 1..]: Nerve::in_flight
    Capacity,       \* acts that may wait in a lane (WAITING)
    Commanded,      \* the parts the brain commands
    BrainDeadlines, \* the deadlines the brain gives (Never, MaxTime + 1, for none)
    MaxBrain,       \* commands the brain sends, at most
    MaxTime,        \* the clock runs 0..MaxTime
    MaxHops,        \* MAX_HOPS
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
    upward,    \* the brain's commands to motor neurons whose outcome went up to it
    admitted,  \* the act a freed place just went to, or 0
    dropped    \* the act a full lane just dropped, or 0

vars == <<now, cmd, n, sent, running, inflight, upward, admitted, dropped>>

Ids == 1..MaxCmds

(* A command: target, commander (Brain, or the command whose program sent it),      *)
(* deadline, neurons passed (hops), state, its program's next step (pc), the command *)
(* it waits for (waits, 0 for none), and what its commander heard (heard). The id is *)
(* the order of arrival.                                                             *)
(*   st: "free" | "waiting" (an act in a lane) | "admitted" (an act a freed place    *)
(*       went to, on its way: the spawned Fabric::deliver) | "acting" | "running"    *)
(*       (a program) | an outcome                                                    *)
Free == [target |-> "", parent |-> Brain, deadline |-> Never, hops |-> 0,
         st |-> "free", pc |-> 1, waits |-> 0, heard |-> "none"]

TypeOK ==
    /\ now \in 0..MaxTime
    /\ n \in 0..MaxCmds
    /\ running \in [Motors -> 0..MaxCmds]
    /\ inflight \in [Nerves -> Nat]
    /\ upward \subseteq Ids
    /\ admitted \in 0..MaxCmds
    /\ dropped \in 0..MaxCmds
    /\ \A c \in Ids : cmd[c].st \in {"free"} \cup Live \cup Outcomes

Init ==
    /\ now = 0
    /\ cmd = [c \in Ids |-> Free]
    /\ n = 0
    /\ sent = 0
    /\ running = [m \in Motors |-> 0]
    /\ inflight = [x \in Nerves |-> 0]
    /\ upward = {}
    /\ admitted = 0
    /\ dropped = 0

-----------------------------------------------------------------------------
(* Reading a commands function f (the state's cmd, or one being built in a step) *)

Target(f, c) == f[c].target
Parent(f, c) == f[c].parent
NerveOfCmd(f, c) == NerveOf[f[c].target]
Waiting(f, x) == {c \in Ids : f[c].st = "waiting" /\ NerveOfCmd(f, c) = x}
\* The acts holding a place in x: carried out, or on their way to be
Holding(f, x) == {c \in Ids : f[c].st \in {"acting", "admitted"} /\ NerveOfCmd(f, c) = x}

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

-----------------------------------------------------------------------------
(* Steps on a state s = [f |-> commands, r |-> running, u |-> upward] *)

\* c ends with o; its commander hears it if it still waits
End(f, c, o) ==
    [f EXCEPT ![c].st = o, ![c].waits = 0,
              ![c].heard = IF Parent(f, c) # Brain /\ Waited(f, c) THEN o ELSE "none"]

\* A program ends: its motor neuron is free, and a brain's command reports up
EndRun(s, c, o) ==
    [f |-> End(s.f, c, o),
     r |-> [s.r EXCEPT ![Target(s.f, c)] = IF @ = c THEN 0 ELSE @],
     u |-> IF Parent(s.f, c) = Brain THEN s.u \cup {c} ELSE s.u]

\* A program starts at its motor neuron and stops the one running there
Start(s, c) ==
    LET m == Target(s.f, c)
        t == IF s.r[m] # 0 THEN EndRun(s, s.r[m], "cancelled") ELSE s
    IN [t EXCEPT !.f = [@ EXCEPT ![c].st = "running", ![c].pc = 1], !.r = [@ EXCEPT ![m] = c]]

Commit(s) == cmd' = s.f /\ running' = s.r /\ upward' = s.u

\* The new command c, already in s.f, is routed: cut after MaxHops, a program started,
\* an act carried out, or an act that waits for a place, which a full lane may drop
Enter(s, c) ==
    LET x == NerveOfCmd(s.f, c) IN
    IF s.f[c].hops > MaxHops
    THEN Commit([s EXCEPT !.f = End(@, c, "failed")])
         /\ UNCHANGED inflight /\ dropped' = 0
    ELSE IF Target(s.f, c) \in Motors
    THEN Commit(Start(s, c)) /\ UNCHANGED inflight /\ dropped' = 0
    ELSE IF x = Outside
    THEN Commit([s EXCEPT !.f = [@ EXCEPT ![c].st = "acting"]])
         /\ UNCHANGED inflight /\ dropped' = 0
    ELSE IF inflight[x] < Limit[x] /\ Waiting(s.f, x) = {}
    THEN Commit([s EXCEPT !.f = [@ EXCEPT ![c].st = "acting"]])
         /\ inflight' = [inflight EXCEPT ![x] = @ + 1]
         /\ dropped' = 0
    ELSE LET w == [s.f EXCEPT ![c].st = "waiting"]
             full == Cardinality(Waiting(s.f, x)) >= Capacity
             least == Last(w, Waiting(w, x))
         IN /\ Commit([s EXCEPT !.f = IF full THEN End(w, least, "failed") ELSE w])
            /\ dropped' = IF full THEN least ELSE 0
            /\ UNCHANGED inflight

-----------------------------------------------------------------------------
(* The brain, and the clock *)

Now == [f |-> cmd, r |-> running, u |-> upward]

BrainCommands(t, d) ==
    /\ sent < MaxBrain
    /\ n < MaxCmds
    /\ Enter([Now EXCEPT !.f = [cmd EXCEPT ![n + 1] =
                 [Free EXCEPT !.target = t, !.deadline = d, !.hops = 1]]], n + 1)
    /\ n' = n + 1
    /\ sent' = sent + 1
    /\ admitted' = 0
    /\ UNCHANGED now

Tick ==
    /\ now < MaxTime
    /\ now' = now + 1
    /\ admitted' = 0 /\ dropped' = 0
    /\ UNCHANGED <<cmd, n, sent, running, inflight, upward>>

-----------------------------------------------------------------------------
(* A program (Started::run): it may step while it runs, its commander waits and *)
(* its deadline has not passed                                                  *)

LiveRun(c) ==
    /\ cmd[c].st = "running"
    /\ Waited(cmd, c)
    /\ now < cmd[c].deadline

\* Command the next neuron (Motion::command): the command keeps this one's deadline
Step(c) ==
    LET m == Target(cmd, c)
        d == n + 1
    IN
    /\ LiveRun(c)
    /\ cmd[c].waits = 0
    /\ cmd[c].pc <= Len(Script[m])
    /\ n < MaxCmds
    /\ Enter([Now EXCEPT !.f = [cmd EXCEPT ![c].pc = @ + 1, ![c].waits = d,
                 ![d] = [Free EXCEPT !.target = Script[m][cmd[c].pc], !.parent = c,
                                     !.deadline = cmd[c].deadline,
                                     !.hops = cmd[c].hops + 1]]], d)
    /\ n' = n + 1
    /\ admitted' = 0
    /\ UNCHANGED <<now, sent>>

\* The outcome arrives: done, go on; otherwise return it
Hear(c) ==
    LET d == cmd[c].waits IN
    /\ LiveRun(c)
    /\ d # 0
    /\ cmd[d].heard # "none"
    /\ IF cmd[d].heard = "done"
       THEN cmd' = [cmd EXCEPT ![c].waits = 0] /\ UNCHANGED <<running, upward>>
       ELSE Commit(EndRun(Now, c, cmd[d].heard))
    /\ admitted' = 0 /\ dropped' = 0
    /\ UNCHANGED <<now, n, sent, inflight>>

Finish(c) ==
    /\ LiveRun(c)
    /\ cmd[c].waits = 0
    /\ cmd[c].pc > Len(Script[Target(cmd, c)])
    /\ Commit(EndRun(Now, c, "done"))
    /\ admitted' = 0 /\ dropped' = 0
    /\ UNCHANGED <<now, n, sent, inflight>>

\* The select: abandoned before expired
Abandon(c) ==
    /\ cmd[c].st = "running"
    /\ ~Waited(cmd, c)
    /\ Commit(EndRun(Now, c, "cancelled"))
    /\ admitted' = 0 /\ dropped' = 0
    /\ UNCHANGED <<now, n, sent, inflight>>

Expire(c) ==
    /\ cmd[c].st = "running"
    /\ Waited(cmd, c)
    /\ now >= cmd[c].deadline
    /\ Commit(EndRun(Now, c, "timed_out"))
    /\ admitted' = 0 /\ dropped' = 0
    /\ UNCHANGED <<now, n, sent, inflight>>

-----------------------------------------------------------------------------
(* An act, and the place it frees (Held::drop, Fabric::release) *)

\* The most urgent waiting act takes the freed place. On the way, one whose deadline
\* passed times out and one nobody waits for any more is forgotten.
RECURSIVE Admit(_, _)
Admit(f, x) ==
    IF Waiting(f, x) = {} THEN <<f, 0>>
    ELSE LET c == First(f, Waiting(f, x)) IN
         IF now >= f[c].deadline THEN Admit(End(f, c, "timed_out"), x)
         ELSE IF ~Waited(f, c) THEN Admit(End(f, c, "cancelled"), x)
         ELSE <<[f EXCEPT ![c].st = "admitted"], c>>

\* An act ends, its tries done or failed for good (the effector's own loop; the tries
\* are bounded), and its place is free
Act(c) ==
    LET x == NerveOfCmd(cmd, c) IN
    /\ cmd[c].st = "acting"
    /\ \E o \in {"done", "failed"} :
        LET f == End(cmd, c, o)
            a == IF x = Outside THEN <<f, 0>> ELSE Admit(f, x)
        IN /\ cmd' = a[1]
           /\ admitted' = a[2]
           /\ inflight' = IF x # Outside /\ a[2] = 0
                          THEN [inflight EXCEPT ![x] = @ - 1] ELSE inflight
    /\ dropped' = 0
    /\ UNCHANGED <<now, n, sent, running, upward>>

\* The deliver spawned by a release runs
Deliver(c) ==
    /\ cmd[c].st = "admitted"
    /\ cmd' = [cmd EXCEPT ![c].st = "acting"]
    /\ admitted' = 0 /\ dropped' = 0
    /\ UNCHANGED <<now, n, sent, running, inflight, upward>>

-----------------------------------------------------------------------------

Next ==
    \/ \E t \in Commanded, d \in BrainDeadlines : BrainCommands(t, d)
    \/ Tick
    \/ \E c \in Ids :
        Step(c) \/ Hear(c) \/ Finish(c) \/ Abandon(c) \/ Expire(c) \/ Act(c) \/ Deliver(c)

\* Everything is bounded: weak fairness on Next is fairness to every part (Motor.tla)
Spec == Init /\ [][Next]_vars /\ WF_vars(Next)

-----------------------------------------------------------------------------
(* Properties *)

\* No more places taken than the nerve has, and no more acts waiting than the lane holds
Bounded ==
    \A x \in Nerves :
        /\ inflight[x] <= Limit[x]
        /\ Cardinality(Waiting(cmd, x)) <= Capacity

\* The count is the acts holding places
CountIsTrue == \A x \in Nerves : inflight[x] = Cardinality(Holding(cmd, x))

\* No place is idle while an act waits
NoIdlePlace ==
    \A x \in Nerves : Waiting(cmd, x) # {} => inflight[x] = Limit[x]

\* Only acts wait, and only acts hold places: a program never does
ProgramsNeverWait ==
    \A c \in Ids : cmd[c].st \in {"waiting", "admitted", "acting"} => Target(cmd, c) \notin Motors

\* One program per motor neuron, the latest command's: a command to a motor neuron
\* starts at once, whatever the lanes (with ProgramsNeverWait), stopping what ran there
LatestWins ==
    \A c \in Ids : cmd[c].st = "running" <=>
        (Target(cmd, c) \in Motors /\ running[Target(cmd, c)] = c)

\* What a commander hears is how the command really ended
HeardIsTrue == \A c \in Ids : cmd[c].heard # "none" => cmd[c].heard = cmd[c].st

\* A command ends once, and its commander hears at most once
OneOutcome ==
    [][\A c \in Ids :
         /\ cmd[c].st \in Outcomes => cmd'[c].st = cmd[c].st
         /\ cmd[c].heard # "none" => cmd'[c].heard = cmd[c].heard]_vars

\* The place that frees goes to the most urgent act waiting, unless its deadline passed
UrgentFirst ==
    [][admitted' # 0 =>
        LET c == admitted'
            x == NerveOfCmd(cmd, c)
        IN /\ now < cmd[c].deadline
           /\ \A d \in Waiting(cmd', x) : Before(cmd, c, d)]_vars

\* A full lane drops the least urgent act, the newcomer included, and the act fails:
\* its commanding program hears it
LeastUrgentDropped ==
    [][dropped' # 0 =>
        LET c == dropped'
            x == NerveOfCmd(cmd', c)
        IN /\ \A d \in Waiting(cmd', x) : Before(cmd', d, c)
           /\ cmd'[c].st = "failed"
           /\ (Parent(cmd', c) # Brain /\ Waited(cmd, c)) => cmd'[c].heard = "failed"]_vars

\* No deadlock, deadlines or not: every command ends, every place comes back
EveryCommandEnds == \A c \in Ids : cmd[c].st \in Live ~> cmd[c].st \in Outcomes

PlacesComeBack == \A x \in Nerves : <>[](inflight[x] = 0)

\* Every brain command to a motor neuron reports up how it ended. A failed act's
\* failure, a dropped one's among them, goes up through the program that commanded it:
\* as "failed", or as "cancelled" when a newer command preempted the program first
ReportGoesUp ==
    \A c \in Ids :
        (cmd[c].st \in Live /\ Parent(cmd, c) = Brain /\ Target(cmd, c) \in Motors)
            ~> c \in upward

Grand(c) == IF Parent(cmd, c) = Brain THEN MaxCmds + 1 ELSE Parent(cmd, Parent(cmd, c))

DropReachesBrain ==
    \A c \in Ids :
        (cmd[c].st = "failed" /\ cmd[c].heard = "failed" /\ Grand(c) = Brain)
            ~> (Parent(cmd, c) \in upward /\ cmd[Parent(cmd, c)].st \in {"failed", "cancelled"})

=============================================================================
