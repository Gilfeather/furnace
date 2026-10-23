----------------------------- MODULE MCRouting -----------------------------
(* The model TLC checks: a receptor R feeds a reflex junction X, which loops back  *)
(* to itself, acts through the effector E, or asks the brain B through a queue;    *)
(* the brain commands the motor neuron M, its one action, whose reports go back to *)
(* the brain through a queue: a loop through queues and a program.                 *)
EXTENDS Routing, TLC

Graph == ("R" :> "receptor") @@ ("X" :> "reflex") @@ ("B" :> "brain")
            @@ ("M" :> "motor") @@ ("E" :> "effector")

P(from, to, kind, weight) == [from |-> from, to |-> to, kind |-> kind, weight |-> weight]

Wiring == << P("R", "X", "reflex", 1), P("X", "X", "reflex", 1),
             P("X", "E", "reflex", 1), P("X", "B", "deliberative", 2),
             P("B", "M", "deliberative", 2), P("M", "B", "deliberative", 2) >>
=============================================================================
