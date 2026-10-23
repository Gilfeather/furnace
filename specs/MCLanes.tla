------------------------------ MODULE MCLanes ------------------------------
(* The models TLC checks: two nerves (the right arm "A", the left leg "B") and *)
(* parts in them or outside both.                                              *)
EXTENDS Lanes

\* MCLanesCross.cfg: each part commands into the other nerve. The hand (in A) moves
\* the foot (in B); the leg (in B) moves the finger (in A).
CrossNerves == ("hand" :> "A") @@ ("foot" :> "B") @@ ("leg" :> "B") @@ ("finger" :> "A")
CrossScript == ("hand" :> <<"foot">>) @@ ("leg" :> <<"finger">>)

\* MCLanesSelf.cfg: the hand moves its finger, then a motor neuron outside every nerve,
\* the spine, which moves the same finger: the arm's own work, twice.
SelfNerves == ("hand" :> "A") @@ ("finger" :> "A") @@ ("spine" :> Outside)
SelfScript == ("hand" :> <<"finger", "spine">>) @@ ("spine" :> <<"finger">>)

\* MCLanesOrder.cfg: commands of different urgency queue for the finger and the hand.
OrderNerves == ("hand" :> "A") @@ ("finger" :> "A")
OrderScript == ("hand" :> <<>>)

OneEach == [x \in {"A", "B"} |-> 1]
=============================================================================
