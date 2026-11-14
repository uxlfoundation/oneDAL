/-
Copyright contributors to the oneDAL project

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-/
/-
Exact-arithmetic semantics of the count/mean/M2 merge used by basic_statistics on GPU
(cpp/oneapi/dal/algo/basic_statistics/backend/gpu/compute_kernel_dense_impl_dpc.cpp):
`merge_blocks_kernel` (pairwise merge, lines ~559-573 and ~634-646) and the distributed
branch of `finalize` (k-way formula, lines ~1199-1219).
Everything here is over Real. Nothing is claimed about IEEE-754 float/double arithmetic.
-/
import Mathlib.Tactic

namespace OnedalStats

open List

/-- Reference statistics of a sample. -/
noncomputable def cnt (xs : List Real) : Real := xs.length
noncomputable def sum (xs : List Real) : Real := xs.sum
noncomputable def sumSq (xs : List Real) : Real := (xs.map (fun x => x ^ 2)).sum
noncomputable def mean (xs : List Real) : Real := sum xs / cnt xs
/-- M2 = sum of squared deviations from the sample mean (`sum2cent` in oneDAL). -/
noncomputable def m2 (xs : List Real) : Real := (xs.map (fun x => (x - mean xs) ^ 2)).sum

/-- Partial result kept per block: row count, mean, M2. -/
structure Stats where
  n : Real
  mean : Real
  m2 : Real

noncomputable def ofList (xs : List Real) : Stats := Stats.mk (cnt xs) (mean xs) (m2 xs)

/-- Contract for an empty block: count 0, mean 0, M2 0 (what `ofList []` gives). -/
def empty : Stats := Stats.mk 0 0 0

/-- A partial result that can come from a block: n >= 0, and n = 0 forces mean = M2 = 0. -/
def Valid (s : Stats) : Prop := 0 <= s.n /\ (s.n = 0 -> s.mean = 0 /\ s.m2 = 0)

/-- Pairwise merge exactly as written in merge_blocks_kernel:
    sum_n1n2 = n1 + n2, delta_scale = n1*n2/sum_n1n2, mean_scale = 1/sum_n1n2,
    delta = mean - mrgmean, m2 = m2a + m2b + delta^2*delta_scale,
    mean = (mrgmean*n1 + mean*n2)*mean_scale. (`a` is the accumulator.) -/
noncomputable def merge (a b : Stats) : Stats :=
  let N := a.n + b.n
  let delta := b.mean - a.mean
  Stats.mk N ((a.mean * a.n + b.mean * b.n) * (1 / N)) (a.m2 + b.m2 + delta * delta * (a.n * b.n / N))

/-! ### Power-sum form -/

lemma cnt_append (xs ys : List Real) : cnt (xs ++ ys) = cnt xs + cnt ys := by
  simp [cnt]
lemma sum_append' (xs ys : List Real) : sum (xs ++ ys) = sum xs + sum ys := by
  simp [sum]
lemma sumSq_append (xs ys : List Real) : sumSq (xs ++ ys) = sumSq xs + sumSq ys := by
  simp [sumSq]

lemma shifted_sq_sum (xs : List Real) (c : Real) :
    (xs.map (fun x => (x - c) ^ 2)).sum = sumSq xs - 2 * c * sum xs + cnt xs * c ^ 2 := by
  induction xs with
  | nil => simp [sumSq, sum, cnt]
  | cons x xs ih =>
    simp only [List.map_cons, List.sum_cons, ih, sumSq, sum, cnt, List.length_cons]
    push_cast; ring

lemma cnt_nonneg (xs : List Real) : 0 <= cnt xs := by simp [cnt]

lemma eq_nil_of_cnt (xs : List Real) (h : cnt xs = 0) : xs = [] := by
  simpa [cnt] using h

/-- M2 in power-sum form: M2 = Q - S^2/n (also true for the empty sample, both sides 0). -/
lemma m2_eq (xs : List Real) : m2 xs = sumSq xs - sum xs ^ 2 / cnt xs := by
  unfold m2
  rw [shifted_sq_sum]
  by_cases h : cnt xs = 0
  next =>
    have := eq_nil_of_cnt xs h; subst this; simp [sumSq, sum, cnt, mean]
  next =>
    unfold mean; field_simp; ring

/-- Deviation of a block around an arbitrary centre c (used by the k-way formula). -/
lemma shift_identity (xs : List Real) (c : Real) :
    (xs.map (fun x => (x - c) ^ 2)).sum = m2 xs + cnt xs * (mean xs - c) ^ 2 := by
  rw [shifted_sq_sum, m2_eq]
  by_cases h : cnt xs = 0
  next =>
    have := eq_nil_of_cnt xs h; subst this; simp [sumSq, sum, cnt]
  next =>
    unfold mean; field_simp; ring

/-! ### Theorem 1: merging block statistics gives the statistics of the concatenated sample -/

theorem merge_ofList (xs ys : List Real) : merge (ofList xs) (ofList ys) = ofList (xs ++ ys) := by
  have hx := cnt_nonneg xs
  have hy := cnt_nonneg ys
  simp only [merge, ofList, Stats.mk.injEq]
  refine And.intro (cnt_append xs ys).symm (And.intro ?_ ?_)
  next =>
    unfold mean; rw [cnt_append, sum_append']
    rcases eq_or_ne (cnt xs) 0 with h0 | h0
    next =>
      have := eq_nil_of_cnt xs h0; subst this
      simp only [cnt, sum, List.length_nil, List.sum_nil, Nat.cast_zero, zero_add, mul_zero]
      rcases eq_or_ne (ys.length : Real) 0 with h1 | h1
      next =>
        simp [h1]
      next =>
        field_simp
    next =>
      rcases eq_or_ne (cnt ys) 0 with h1 | h1
      next =>
        have := eq_nil_of_cnt ys h1; subst this
        simp only [cnt, sum, List.length_nil, List.sum_nil, Nat.cast_zero, add_zero,
          mul_zero] at h0
        simp only [cnt, sum, List.length_nil, List.sum_nil, Nat.cast_zero, add_zero,
          mul_zero]
        field_simp
      next =>
        have hN : Ne (cnt xs + cnt ys) 0 :=
          ne_of_gt (by have := lt_of_le_of_ne hx (Ne.symm h0); linarith)
        field_simp
  next =>
    rw [m2_eq, m2_eq, m2_eq, cnt_append, sum_append', sumSq_append]
    unfold mean
    rcases (hx.eq_or_lt.imp Eq.symm id) with h0 | h0
    next =>
      have := eq_nil_of_cnt xs h0; subst this; simp [cnt, sum, sumSq]
    next =>
      rcases (hy.eq_or_lt.imp Eq.symm id) with h1 | h1
      next =>
        have := eq_nil_of_cnt ys h1; subst this; simp [cnt, sum, sumSq]
      next =>
        have hN : Ne (cnt xs + cnt ys) 0 := by positivity
        field_simp; ring

/-! ### Theorem 2: empty blocks are neutral (under the contract `empty = Stats.mk 0 0 0`) -/

theorem merge_empty_left (s : Stats) (hs : Valid s) : merge empty s = s := by
  have hn := (hs).1; have hz := (hs).2
  cases s; rename_i n mu q
  simp only [merge, empty, Stats.mk.injEq] at *
  by_cases h : n = 0
  next =>
    have h1 := (hz h).1; have h2 := (hz h).2; subst h; subst h1; subst h2; simp
  next =>
    refine And.intro (by ring) (And.intro ?_ ?_) <;> field_simp

theorem merge_empty_right (s : Stats) (hs : Valid s) : merge s empty = s := by
  have hn := (hs).1; have hz := (hs).2
  cases s; rename_i n mu q
  simp only [merge, empty, Stats.mk.injEq] at *
  by_cases h : n = 0
  next =>
    have h1 := (hz h).1; have h2 := (hz h).2; subst h; subst h1; subst h2; simp
  next =>
    refine And.intro (by ring) (And.intro ?_ ?_) <;> field_simp

/-- Without the contract the "empty" block is not neutral: a zero-count block whose
    mean field holds garbage changes nothing here only because n = 0 kills it,
    but a block with n = 0 and m2 != 0 leaks its m2 into the result. -/
example : Ne (merge (Stats.mk 0 0 1) (Stats.mk 1 5 0)) (Stats.mk 1 5 0) := by
  simp [merge]

theorem merge_valid (a b : Stats) (ha : Valid a) (hb : Valid b) : Valid (merge a b) := by
  have han := (ha).1; have haz := (ha).2
  have hbn := (hb).1; have hbz := (hb).2
  refine And.intro (by simp [merge]; linarith) (fun h => ?_)
  simp only [merge] at h; simp only [merge]
  have ha0 : a.n = 0 := by linarith
  have hb0 : b.n = 0 := by linarith
  have h1 := (haz ha0).1; have h2 := (haz ha0).2
  have h3 := (hbz hb0).1; have h4 := (hbz hb0).2
  simp [ha0, hb0, h2, h4]

/-! ### Theorem 3: associativity in exact arithmetic (any valid partial results) -/

theorem merge_assoc (a b c : Stats) (ha : Valid a) (hb : Valid b) (hc : Valid c) :
    merge (merge a b) c = merge a (merge b c) := by
  have han := (ha).1; have haz := (ha).2
  have hbn := (hb).1; have hbz := (hb).2
  have hcn := (hc).1; have hcz := (hc).2
  cases a; cases b; cases c; rename_i na ma qa nb mb qb nc mc qc
  simp only at *
  simp only [merge, Stats.mk.injEq]
  rcases han.eq_or_lt with h0 | h0
  next =>
    have r1 := (haz h0.symm).1; have r2 := (haz h0.symm).2; subst r1 r2; rw [<- h0]
    rcases hbn.eq_or_lt with h1 | h1
    next =>
      have r3 := (hbz h1.symm).1; have r4 := (hbz h1.symm).2; subst r3 r4; rw [<- h1]
      rcases hcn.eq_or_lt with h2 | h2
      next =>
        have r5 := (hcz h2.symm).1; have r6 := (hcz h2.symm).2; subst r5 r6; rw [<- h2]; simp
      next =>
        refine And.intro (by ring) (And.intro ?_ ?_) <;> field_simp
    next =>
      refine And.intro (by ring) (And.intro ?_ ?_) <;> field_simp
  next =>
    rcases hbn.eq_or_lt with h1 | h1
    next =>
      have r3 := (hbz h1.symm).1; have r4 := (hbz h1.symm).2; subst r3 r4; rw [<- h1]
      rcases hcn.eq_or_lt with h2 | h2
      next =>
        have r5 := (hcz h2.symm).1; have r6 := (hcz h2.symm).2; subst r5 r6; rw [<- h2]
        refine And.intro (by ring) (And.intro ?_ ?_) <;> field_simp
      next =>
        refine And.intro (by ring) (And.intro ?_ ?_) <;> field_simp
    next =>
      rcases hcn.eq_or_lt with h2 | h2
      next =>
        have r5 := (hcz h2.symm).1; have r6 := (hcz h2.symm).2; subst r5 r6; rw [<- h2]
        refine And.intro (by ring) (And.intro ?_ ?_) <;> field_simp
      next =>
        have : Ne (na + nb) 0 := by positivity
        have : Ne (nb + nc) 0 := by positivity
        have : Ne (na + nb + nc) 0 := by positivity
        have : Ne (na + (nb + nc)) 0 := by positivity
        refine And.intro (by ring) (And.intro ?_ ?_) <;> field_simp <;> ring

/-- Merge is commutative (no hypotheses needed); with `merge_assoc` any merge order
    and any tree shape give the same exact result. -/
theorem merge_comm (a b : Stats) : merge a b = merge b a := by
  simp only [merge, Stats.mk.injEq]
  refine And.intro (by ring) (And.intro (by ring) (by ring))

/-! ### Theorem 4: the k-way formula of distributed `finalize`
  Each rank r holds (n_r, mean_r, M2_r) and the global sum; finalize computes
  M2_r + n_r * (mean - mean_r)^2 per rank and allreduce-sums it. -/

noncomputable def kwayM2 (blocks : List (List Real)) : Real :=
  let all := blocks.flatten
  (blocks.map (fun b => m2 b + cnt b * (mean all - mean b) ^ 2)).sum

theorem kwayM2_correct (blocks : List (List Real)) : kwayM2 blocks = m2 blocks.flatten := by
  unfold kwayM2
  simp only
  generalize hc : mean blocks.flatten = c
  have key : forall bs : List (List Real),
      (bs.map (fun b => m2 b + cnt b * (c - mean b) ^ 2)).sum
        = (bs.flatten.map (fun x => (x - c) ^ 2)).sum := by
    intro bs
    induction bs with
    | nil => simp
    | cons b bs ih =>
      rw [List.map_cons, List.sum_cons, ih, List.flatten_cons, List.map_append,
        List.sum_append, shift_identity b c]
      ring
  rw [key, <- hc]; rfl

/-! ### Negative variant: drop the `/N` in delta_scale -/

noncomputable def mergeBad (a b : Stats) : Stats :=
  let N := a.n + b.n
  let delta := b.mean - a.mean
  Stats.mk N ((a.mean * a.n + b.mean * b.n) * (1 / N)) (a.m2 + b.m2 + delta * delta * (a.n * b.n))

/-- Concrete counterexample: blocks [0] and [2,4]. Correct M2 of [0,2,4] is 8;
    the mutated formula gives 2 + 9*2 = 20. -/
theorem mergeBad_wrong :
    Ne (mergeBad (ofList [0]) (ofList [2, 4])) (ofList ([0] ++ [2, 4])) := by
  simp only [mergeBad, ofList, ne_eq, Stats.mk.injEq, not_and]
  intro _ _
  simp [m2, mean, sum, cnt]
  norm_num

/-! ### The DAAL CPU online/distributed merge term
  `low_order_moments_impl.i` and `low_order_moments_estimates_online.i` add
  `coeff * (mean1*mean1 + mean2*mean2 - 2*mean1*mean2)`. Over Real this is the same as
  `coeff * (mean2 - mean1)^2`, so the proofs above hold for both spellings; the
  difference between them is only in floating point (see dev/formal/numeric). -/
theorem daal_term_eq (coeff m1 m2 : Real) :
    coeff * (m1 * m1 + m2 * m2 - 2 * m1 * m2) = coeff * ((m2 - m1) * (m2 - m1)) := by
  ring

end OnedalStats
