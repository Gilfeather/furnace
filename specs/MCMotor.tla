------------------------------ MODULE MCMotor ------------------------------
(* The models TLC checks, after examples/get_dressed.rs. The brain may call "dress" *)
(* and the effector "sleeve"; "arm" and "hand" are parts below.                     *)
EXTENDS Motor

\* MCMotor.cfg: one command down the whole tree. dress commands arm through a
\* deliberative pathway, then sleeve and hand together; arm commands hand.
Tree == [m \in {"dress", "arm"} |->
            IF m = "dress" THEN << {"arm"}, {"sleeve", "hand"} >> ELSE << {"hand"} >>]

\* MCMotorPreempt.cfg: two commands to dress, the second preempting the first.
Short == [m \in {"dress", "arm"} |->
             IF m = "dress" THEN << {"arm"} >> ELSE << {"hand"} >>]

DressToArm == {<<"dress", "arm">>}
=============================================================================
