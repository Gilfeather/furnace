------------------------------ MODULE Routing ------------------------------
(***************************************************************************)
(* How a signal moves (src/nerve/fabric/route.rs, summing.rs, nerve.rs).   *)
(*                                                                         *)
(* A stimulus enters at a receptor. Every neuron it reaches may be quiet   *)
(* (its gate, or nothing to say) or fire one signal derived from it: a     *)
(* plain signal, or a command naming a neuron. A plain signal takes the    *)
(* pathway its timing selects: among the pathways leaving the neuron whose *)
(* measured p99 fits in the signal's remaining time, the heaviest. A       *)
(* command goes to the motor neuron or effector it names, along the        *)
(* pathway to it or directly; a brain's command only to a declared action. *)
(*                                                                         *)
(* A reflex pathway calls the next neuron at once; a deliberative one      *)
(* queues the signal for a worker, and so does a nerve's lane and a motor  *)
(* neuron's program (Fabric::emit). After any of them the depth of the     *)
(* synchronous chain starts again at 1, but the trace keeps counting: a    *)
(* signal is cut once either reaches MaxHops.                              *)
(*                                                                         *)
(* Firing more than one signal splits the chain into chains of the same    *)
(* kind, so one signal at a time is modelled. Fired signals derive from    *)
(* the one received (Signal::derive), as every neuron in furnace's own     *)
(* code does: a neuron that makes a fresh signal starts a new trace.       *)
(***************************************************************************)
EXTENDS Naturals, Sequences, FiniteSets

CONSTANTS
    Kind,       \* [Neurons -> {"receptor", "reflex", "brain", "motor", "effector"}]
    Actions,    \* the neurons declared actions
    Paths,      \* Seq of [from, to, kind |-> "reflex" | "deliberative", weight], in order
    Unknown,    \* a name no neuron has
    Latencies,  \* the p99s a pathway may have measured; None for not yet
    Remainders, \* the remaining times a signal may have; None for no deadline
    None,       \* a model value
    MaxHops     \* MAX_HOPS

Neurons == DOMAIN Kind
Names == Neurons \cup {Unknown}
PathIds == 1..Len(Paths)

\* The pathways leaving f, in the order they were declared (Cell::out)
Out(f) == SelectSeq([i \in PathIds |-> i], LAMBDA i : Paths[i].from = f)
OutSet(f) == {i \in PathIds : Paths[i].from = f}

VARIABLES
    p99,       \* [PathIds -> Latencies]: what each pathway has measured
    sig,       \* the signal in flight, or None: [at, depth, len, rem, cmd, how]
    log,       \* what happened to it, once: "terminal", "no_pathway", "unrouted", "cut",
               \* "quiet"
    chosen,    \* the last plain routing: <<from, remaining, pathway or 0, p99>>, or <<>>
    delivered  \* commands delivered: <<the kind of the neuron that sent it, its target>>

vars == <<p99, sig, log, chosen, delivered>>

(* A signal: `at` is the neuron about to fire it (how = "fire") or about to receive  *)
(* it (how = "drive", or "queued" for a worker, a lane or a program to carry on);    *)
(* depth is the synchronous chain's, len the trace's; rem its remaining time; cmd    *)
(* the name it commands, or "" for a plain signal.                                   *)
Signal(at, depth, len, rem, cmd, how) ==
    [at |-> at, depth |-> depth, len |-> len, rem |-> rem, cmd |-> cmd, how |-> how]

Init ==
    /\ p99 \in [PathIds -> Latencies]
    /\ \E r \in {x \in Neurons : Kind[x] = "receptor"}, rem \in Remainders :
         sig = Signal(r, 0, 0, rem, "", "drive")
    /\ log = ""
    /\ chosen = <<>>
    /\ delivered = {}

-----------------------------------------------------------------------------
(* Pathway selection (Fabric::select), as route.rs runs it: in the order declared, *)
(* a fitting pathway replaces the best so far only when strictly heavier           *)

FitsWith(lat, i, rem) == rem = None \/ lat[i] = None \/ lat[i] <= rem
Fits(i, rem) == FitsWith(p99, i, rem)

RECURSIVE Fold(_, _, _)
Fold(s, rem, best) ==
    IF s = <<>> THEN best
    ELSE LET i == Head(s)
             next == IF Fits(i, rem) /\ (best = 0 \/ Paths[i].weight > Paths[best].weight)
                     THEN i ELSE best
         IN Fold(Tail(s), rem, next)

Select(f, rem) == Fold(Out(f), rem, 0)

-----------------------------------------------------------------------------
(* Steps *)

\* Where a signal goes next over pathway i: at once, or queued
Over(i, s) ==
    IF Paths[i].kind = "reflex"
    THEN [s EXCEPT !.at = Paths[i].to, !.how = "drive"]
    ELSE [s EXCEPT !.at = Paths[i].to, !.how = "queued"]

\* Fabric::route: `s` was fired by s.at at depth s.depth
Route(s) ==
    LET f == s.at IN
    IF s.depth >= MaxHops \/ s.len >= MaxHops
    THEN sig' = None /\ log' = "cut" /\ UNCHANGED <<chosen, delivered>>
    ELSE IF s.cmd # ""
    THEN \* A command names its effector or motor neuron
         LET t == s.cmd
             ok == /\ t \in Neurons
                   /\ Kind[t] \in {"motor", "effector"}
                   /\ (Kind[f] = "brain" => t \in Actions)
             via == {i \in OutSet(f) : Paths[i].to = t}
             i == CHOOSE i \in via : \A j \in via : i <= j
         IN IF ~ok
            THEN sig' = None /\ log' = "unrouted" /\ UNCHANGED <<chosen, delivered>>
            ELSE /\ delivered' = delivered \cup {<<Kind[f], t>>}
                 /\ \* Along the first pathway to it, or directly; a nerve's lane
                    \* may make it wait (and then it is carried on as from a queue)
                    \/ /\ via # {}
                       /\ sig' = Over(i, s)
                    \/ /\ via = {}
                       /\ sig' \in {[s EXCEPT !.at = t, !.how = "drive"],
                                    [s EXCEPT !.at = t, !.how = "queued"]}
                 /\ UNCHANGED <<log, chosen>>
    ELSE \* A plain signal takes the pathway its timing selects
         LET i == Select(f, s.rem) IN
         /\ chosen' = <<f, s.rem, i, p99>>
         /\ IF i # 0
            THEN sig' = Over(i, s) /\ UNCHANGED log
            ELSE /\ sig' = None
                 /\ log' = IF Out(f) = <<>> THEN "terminal" ELSE "no_pathway"
         /\ UNCHANGED delivered

\* A neuron receives the signal: its gate may stop it; else it passes (the trace grows)
\* and the neuron is quiet or fires one signal derived from it. A motor neuron's
\* program fires later, from its own task (Fabric::emit, depth 1).
Drive ==
    /\ sig # None
    /\ sig.how \in {"drive", "queued"}
    /\ LET s == sig
           \* fire_summed and Fabric::deliver drive at depth 1; a program's signals
           \* route at depth 1 (Fabric::emit)
           d == IF s.how = "queued" THEN 1 ELSE s.depth
       IN \/ sig' = None /\ log' = "quiet"
          \/ /\ \E cmd \in {""} \cup Names :
                  sig' = [s EXCEPT !.len = @ + 1, !.cmd = cmd,
                                   !.depth = IF Kind[s.at] = "motor" THEN 1 ELSE d + 1,
                                   !.how = "fire"]
             /\ UNCHANGED log
    /\ UNCHANGED <<p99, chosen, delivered>>

Fire ==
    /\ sig # None
    /\ sig.how = "fire"
    /\ Route(sig)
    /\ UNCHANGED p99

\* Time passes while a signal waits: its remaining time shrinks
Wait ==
    /\ sig # None
    /\ sig.how = "queued"
    /\ sig.rem # None
    /\ \E r \in Remainders \ {None} :
         /\ r < sig.rem
         /\ sig' = [sig EXCEPT !.rem = r]
    /\ UNCHANGED <<p99, log, chosen, delivered>>

\* The p99s are chosen at the start, every combination of them: a selection reads them
\* as they are at that moment, so this covers latencies that change in between.
Next == Drive \/ Fire \/ Wait

Spec == Init /\ [][Next]_vars /\ WF_vars(Drive) /\ WF_vars(Fire)

-----------------------------------------------------------------------------
(* Properties *)

\* Every signal ends, loops through queues, lanes and programs included
Terminates == <>(sig = None)

\* And it never goes past the cut
WithinHops == sig # None => sig.len <= MaxHops /\ sig.depth <= MaxHops

\* The pathway taken is the heaviest of those whose p99 fits the remaining time, and
\* none is taken only when none fits
HeaviestThatFits ==
    chosen # <<>> =>
        LET f == chosen[1]
            rem == chosen[2]
            i == chosen[3]
            fitting == {j \in OutSet(f) : FitsWith(chosen[4], j, rem)}
        IN IF i = 0
           THEN fitting = {}
           ELSE /\ i \in fitting
                /\ \A j \in fitting : Paths[j].weight <= Paths[i].weight

\* Commands reach motor neurons and effectors only, and a brain's reach actions only:
\* a command to anything else is unrouted, never delivered
CommandsReachActuators ==
    \A d \in delivered : Kind[d[2]] \in {"motor", "effector"}

BrainReachesActionsOnly ==
    \A d \in delivered : d[1] = "brain" => d[2] \in Actions

=============================================================================
