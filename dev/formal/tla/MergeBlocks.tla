\* Copyright contributors to the oneDAL project
\*
\* Licensed under the Apache License, Version 2.0 (the "License");
\* you may not use this file except in compliance with the License.
\* You may obtain a copy of the License at
\*
\*     http://www.apache.org/licenses/LICENSE-2.0
\*
\* Unless required by applicable law or agreed to in writing, software
\* distributed under the License is distributed on an "AS IS" BASIS,
\* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
\* See the License for the specific language governing permissions and
\* limitations under the License.
---------------------------- MODULE MergeBlocks ----------------------------
(* Execution protocol of basic_statistics GPU compute_by_blocks -> merge_blocks
   (cpp/oneapi/dal/algo/basic_statistics/backend/gpu/compute_kernel_dense_impl_dpc.cpp).
   Host: submit K1 (process_blocks, writes ndbuf), submit K2 (merge_blocks_kernel,
   depends_on K1, reads ndbuf), wait_and_throw(K2), ndbuf destroyed (sycl::free,
   no implicit wait). Device: one work-group of L items of K2 = one column.
   Mutations are selected through the Mut constant. *)
EXTENDS Naturals, FiniteSets

CONSTANTS L,      \* work-group size (device_max_sg_size), power of two
          B,      \* block_count (get_row_block_count)
          Mut     \* subset of {"nodep", "nowait", "nobarrier"}

Items  == 0..(L-1)
Blocks == 0..(B-1)
Min(a, b) == IF a < b THEN a ELSE b
\* stride sequence: min(L,B)/2, /2, ... ,1 ; Round r has stride Stride(r)
RECURSIVE Log2(_)
Log2(n) == IF n <= 1 THEN 0 ELSE 1 + Log2(n \div 2)
NRounds == Log2(Min(L, B))
RECURSIVE Pow2(_)
Pow2(n) == IF n = 0 THEN 1 ELSE 2 * Pow2(n - 1)
Stride(r) == (Min(L, B) \div 2) \div Pow2(r - 1)   \* r in 1..NRounds
MyBlocks(i) == {b \in Blocks : b % L = i}            \* for (i = id; i < B; i += L)

VARIABLES host,     \* "k1" | "k2" | "wait" | "free" | "done"
          k1,       \* "none" | "running" | "done"
          written,  \* blocks of ndbuf written by K1
          alive,    \* ndbuf allocation alive
          k2,       \* "none" | "submitted" | "done"
          pc,       \* per item: "load" | "bar" | "red" | "fin" | "exit"
          todo,     \* per item: blocks still to load in the first loop
          acc,      \* per item: private merged set (mrg* registers)
          dup,      \* per item: some block merged twice
          loc,      \* per item: local memory slot l*_ptr[id]
          round,    \* per item: current tree round (0 = first loop)
          err       \* set of violated protocol facts

vars == <<host, k1, written, alive, k2, pc, todo, acc, dup, loc, round, err>>

Init == /\ host = "k1" /\ k1 = "none" /\ written = {} /\ alive = TRUE
        /\ k2 = "none"
        /\ pc = [i \in Items |-> "load"] /\ todo = [i \in Items |-> MyBlocks(i)]
        /\ acc = [i \in Items |-> {}] /\ dup = [i \in Items |-> FALSE]
        /\ loc = [i \in Items |-> {}] /\ round = [i \in Items |-> 0]
        /\ err = {}

\* ---- Host ----
SubmitK1 == host = "k1" /\ host' = "k2" /\ k1' = "running"
            /\ UNCHANGED <<written, alive, k2, pc, todo, acc, dup, loc, round, err>>
SubmitK2 == host = "k2" /\ host' = "wait" /\ k2' = "submitted"
            /\ UNCHANGED <<k1, written, alive, pc, todo, acc, dup, loc, round, err>>
\* last_event.wait_and_throw(): blocks until K2 is complete
Wait == host = "wait" /\ ("nowait" \in Mut \/ k2 = "done") /\ host' = "free"
        /\ UNCHANGED <<k1, written, alive, k2, pc, todo, acc, dup, loc, round, err>>
\* merge_blocks returns: ndbuf (rvalue parameter) is destroyed -> sycl::free
Free == host = "free" /\ host' = "done" /\ alive' = FALSE
        /\ UNCHANGED <<k1, written, k2, pc, todo, acc, dup, loc, round, err>>

\* ---- K1 ----
K1Write(b) == k1 = "running" /\ b \notin written /\ written' = written \cup {b}
              /\ k1' = (IF written' = Blocks THEN "done" ELSE "running")
              /\ UNCHANGED <<host, alive, k2, pc, todo, acc, dup, loc, round, err>>

\* ---- K2 ----
\* cgh.depends_on(deps): no work-item of K2 starts before K1 completes
K2Ready == k2 = "submitted" /\ ("nodep" \in Mut \/ k1 = "done")

Merge(i, S) == /\ acc' = [acc EXCEPT ![i] = acc[i] \cup S]
               /\ dup' = [dup EXCEPT ![i] = dup[i] \/ (acc[i] \cap S # {})]

Load(i) == /\ K2Ready /\ pc[i] = "load"
           /\ (IF todo[i] = {}
                THEN /\ pc' = [pc EXCEPT ![i] = (IF NRounds = 0 THEN "fin" ELSE "bar")]
                     /\ UNCHANGED <<todo, acc, dup, loc, err>>
                ELSE \E b \in todo[i] :
                       /\ \A c \in todo[i] : b <= c
                       /\ todo' = [todo EXCEPT ![i] = todo[i] \ {b}]
                       /\ Merge(i, {b})
                       /\ loc' = [loc EXCEPT ![i] = acc'[i]]
                       /\ err' = err \cup (IF ~alive THEN {"UseAfterFree"} ELSE {})
                                     \cup (IF b \notin written THEN {"ReadBeforeWrite"} ELSE {})
                       /\ UNCHANGED pc)
           /\ UNCHANGED <<host, k1, written, alive, k2, round>>

\* sycl::group_barrier at the top of every tree round: item waits until all
\* items have finished the previous round (round[j] >= round[i] and not mid-round)
Barrier(i) == /\ K2Ready /\ pc[i] = "bar"
              /\ "nobarrier" \in Mut \/
                 \A j \in Items : round[j] > round[i] \/ (round[j] = round[i] /\ pc[j] = "bar")
              /\ round' = [round EXCEPT ![i] = round[i] + 1]
              /\ pc' = [pc EXCEPT ![i] = "red"]
              /\ UNCHANGED <<host, k1, written, alive, k2, todo, acc, dup, loc, err>>

\* if (stride > id) { merge l*_ptr[id + stride]; if (0 < id) store own slot }
Reduce(i) == /\ K2Ready /\ pc[i] = "red"
             /\ (LET s == Stride(round[i]) IN
                IF s > i
                  THEN /\ Merge(i, loc[i + s])
                       /\ loc' = (IF i > 0 THEN [loc EXCEPT ![i] = acc'[i]] ELSE loc)
                       \* partner slot must hold the partner's value of the previous round
                       /\ err' = err \cup (IF pc[i + s] = "load" \/ round[i + s] < round[i] - 1
                                               \/ (round[i + s] = round[i] - 1 /\ pc[i + s] = "red")
                                            THEN {"StaleLocalRead"} ELSE {})
                  ELSE UNCHANGED <<acc, dup, loc, err>>)
             /\ pc' = [pc EXCEPT ![i] = (IF round[i] = NRounds THEN "fin" ELSE "bar")]
             /\ UNCHANGED <<host, k1, written, alive, k2, todo, round>>

\* if (0 == id) write r*_ptr[group_id]; kernel completes when all items exit
Finish(i) == /\ K2Ready /\ pc[i] = "fin"
             /\ pc' = [pc EXCEPT ![i] = "exit"]
             /\ k2' = (IF \A j \in Items \ {i} : pc[j] = "exit" THEN "done" ELSE k2)
             /\ UNCHANGED <<host, k1, written, alive, todo, acc, dup, loc, round, err>>

Done == host = "done" /\ k2 = "done" /\ UNCHANGED vars

Next == \/ SubmitK1 \/ SubmitK2 \/ Wait \/ Free \/ Done
        \/ \E b \in Blocks : K1Write(b)
        \/ \E i \in Items : Load(i) \/ Barrier(i) \/ Reduce(i) \/ Finish(i)

Spec == Init /\ [][Next]_vars /\ WF_vars(Next)

\* ---- Properties ----
NoUseAfterFree   == "UseAfterFree" \notin err
NoReadBeforeWrite == "ReadBeforeWrite" \notin err
NoStaleLocalRead == "StaleLocalRead" \notin err
\* work-item 0 has merged every block exactly once when it writes the result
ResultCoversAllBlocks == pc[0] \in {"fin", "exit"} => (acc[0] = Blocks /\ ~dup[0])
\* finalize() reads ndres only after K2 completed
ResultReadAfterK2 == host = "done" => k2 = "done" \/ "nowait" \in Mut
Termination == <>(host = "done" /\ k2 = "done")
=============================================================================
