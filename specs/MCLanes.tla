------------------------------ MODULE MCLanes ------------------------------
(* The models TLC checks: two nerves (the right arm "A", the left leg "B"). None *)
(* has a deadline but MCLanesOrder: nothing ends because a deadline passed.     *)
EXTENDS Lanes, TLC

\* MCLanesCross.cfg: two parts command each other's acts while the brain floods both
\* lanes. The hand (A) moves the toe (B); the leg (B) moves the finger (A).
CrossNerves == ("hand" :> "A") @@ ("toe" :> "B") @@ ("leg" :> "B") @@ ("finger" :> "A")
CrossScript == ("hand" :> <<"toe">>) @@ ("leg" :> <<"finger">>)

\* MCLanesCycle.cfg: two motor neurons command each other, round and round.
CycleNerves == ("m1" :> "A") @@ ("m2" :> "B")
CycleScript == ("m1" :> <<"m2">>) @@ ("m2" :> <<"m1">>)

\* MCLanesFull.cfg: the hand commands its finger while the arm's lane is full: newer
\* commands to the hand preempt at once, and a dropped act's failure goes up.
FullNerves == ("hand" :> "A") @@ ("finger" :> "A")
FullScript == ("hand" :> <<"finger">>)

\* MCLanesOrder.cfg: acts of different urgency queue for the finger.
OrderNerves == ("finger" :> "A")
NoScript == [m \in {} |-> <<>>]

OneEach == [x \in {"A", "B"} |-> 1]
=============================================================================
