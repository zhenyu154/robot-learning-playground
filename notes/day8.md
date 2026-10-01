# Day8: Y-Active Loss Weighting (2026-10-01)

## Starting point

Day7's corrected-camera v2 dataset passes QA (20 successful demonstrations,
2,037 frames). The 2,000-update ACT policy has useful offline Y direction
predictions, but online seen control fails: Y displacement is too small, then
the policy descends before alignment. The main evaluation is `0/6` with
`n_action_steps=10`; the tighter-feedback diagnostic is `0/2` with
`n_action_steps=1`. Neither result establishes the root cause by itself.

## Correction to the initial plan

Increasing keyboard XY speed does NOT make Y supervision temporally denser.
For a fixed movement distance and frame rate, larger per-frame displacement
needs fewer active frames. The proposed `XY=0.50` recollection was withdrawn.
Keep the existing Day7 v2 dataset and `XY=0.25`, and change one training factor.

## Read-only dataset/window analysis

The existing frame labels contain:

| Raw Y action | Frames |
|---|---:|
| `-0.25` | 183 |
| `0` | 1671 |
| `+0.25` | 183 |

- Active-Y frame ratio: `366/2037 = 18.0%`; signed label counts are balanced.
- No simultaneous active-Y and downward-Z label frames were found. The expert
  action stages are lateral movement, then descent, close, and lift.
- Initial all-motion-neutral periods total 261 frames; first Y activity starts
  at frame 10--21. Such waiting can create timing ambiguity, but it is not proof
  of the cause of online failure and is not automatically removed.
- For every frame as a possible chunk start, `873/2037 = 42.9%` of 50-step
  windows contain at least one active-Y target. Excluding episode-end padding,
  active Y occurs at `8729/77350 = 11.3%` of valid target time positions. This
  describes supervision coverage, not a measured sampler frequency or a proven
  causal explanation. Each valid time position has four scalar action targets.

## Hypothesis and single-factor intervention

Test whether emphasizing active-Y action errors improves lateral control.
Use the SAME v2 demonstrations, normalization, sampler, model, seed, batch size,
2,000 updates, `chunk_size=50`, and `n_action_steps=10` as the Day7 v2 recipe.
Start from scratch; do not fine-tune the Day7 checkpoint.

For batch element b, chunk time t, and action channel d:

```text
active_y[b,t] = abs(raw_action[b,t,1]) > 1e-6
w[b,t,d] = 4 if d==1 and active_y[b,t], otherwise 1
w[b,t,d] = 0 for action_is_pad[b,t]

L1_weighted = sum(w * abs(normalized_prediction - normalized_target)) / sum(w)
loss = L1_weighted + unchanged_upstream_KL_term
```

The activity mask is captured BEFORE normalization and applies to every valid
position in the predicted chunk, not just its first action. Neutral-Y, X, Z,
and gripper scalar errors remain weight 1. The common weighted denominator
changes their absolute gradient scale too; it is the **relative** active-Y
coefficient that is four times larger. KL computation/coefficient is unchanged.
This does not guarantee that descent/gripper behavior will remain unchanged.
Monitor those channels as possible side effects.

Weight 1 is a numerical control: it returns the upstream loss exactly. Weight
4 is one prespecified trial, not a tuned optimum. The operator's movement speed
and the policy's inference actions are not multiplied or otherwise assisted.

## Implementation and provenance

- `scripts/day8_y_weighted_loss.py`: raw-mask capture, valid weighted reduction,
  and temporary, process-local adaptation of the ordinary training functions.
- `scripts/day8_train_y_weighted.py`: validation, integration, and provenance.
- `scripts/day8_act_y_weighted_xpu.sh`: controlled XPU launcher with log capture.
- `tests/test_day8_y_weighted_loss.py`: CPU unit/integration tests.

No installed LeRobot files are edited. The upstream forward runs once per batch;
the adapter replaces its L1 term while retaining the original computation graph
for KL. Hooks are removed after use. Saved models retain the ordinary ACT class
and tensor keys, so existing audit/evaluation scripts load them without a custom
policy implementation.

`day8_loss_recipe.json` records the loss settings, dataset info/stats hashes,
source hashes/revisions, and training settings in the run directory AND each
`pretrained_model` directory. The standard `train_config.json` alone does not
describe the custom loss. Resume/fine-tuning, distributed runs, compilation,
activation checkpointing, AMP, EMA, and combined sample weighting are rejected
by this first implementation. Do not resume using vanilla `lerobot-train`: it
would silently revert to ordinary ACT loss.

## Validation and commands

Code validation completed on CPU; no training experiment or online rollout has
been run by the coding agent. All 21 loss/adapter tests passed, including:

- Weight 1 loss AND gradients match the real upstream ACT forward.
- Weight 4 has the intended relative scalar-error gradient coefficients.
- Neutral-Y, future chunk positions, padded positions, and empty activity.
- Raw activity survives real processor normalization with a nonzero mean.
- A single model forward and intact KL contribution.
- Hook restoration and standard ACT checkpoint save/load/inference.

Repeat tests in the LeRobot environment:

```bash
python -m unittest discover -s tests -p 'test_day8*.py' -v
```

First run a five-update pipeline smoke test, not the full experiment:

```bash
STEPS=5 LOG_FREQ=1 \
OUTPUT_DIR="$PWD/outputs/act_panda_day8_y_weighted_w4_xpu_smoke_v1" \
JOB_NAME=act_panda_day8_y_weighted_w4_xpu_smoke_v1 \
bash scripts/day8_act_y_weighted_xpu.sh
```

Check: XPU selected, five updates complete, finite losses, extra metrics appear,
standard checkpoint saved, and `day8_loss_recipe.json` says weight 4. Loss
`y_active_target_ratio` describes valid target positions in the sampled chunks,
not the original dataset's 18% frame ratio. It may be zero in some batches.

The launcher uses PyAV, as in Day7. A previously observed optional TorchCodec
load/fallback warning is separate from this loss change; stop for a fatal
traceback or nonfinite training loss, not merely a successful PyAV fallback.

If smoke passes, run the full, fresh 2,000-update experiment:

```bash
bash scripts/day8_act_y_weighted_xpu.sh
```

Default final checkpoint:
`outputs/act_panda_day8_y_weighted_w4_xpu_v1/checkpoints/002000/pretrained_model`

The full terminal log is preserved alongside the run directory:
`outputs/act_panda_day8_y_weighted_w4_xpu_v1.log`.
Existing output directories/logs are never overwritten. A rerun needs a fresh
`OUTPUT_DIR`. The launcher's dataset paths, job name, output path, weight, number
of steps, and log frequency can be overridden via environment variables.

## Evaluation after training

1. Use the existing offline audit on the SAME v2 dataset and a new CSV name.
2. Compare positive/negative Y prediction magnitude and sign-and-magnitude hit
   rates at the unchanged `0.05` threshold, and neutral false-motion.
3. Also monitor gripper close hit/false-close: different loss values alone do
   not establish improvement. `l1_loss` now reports weighted L1;
   `unweighted_l1_loss` reports the ordinary reconstruction metric for reference.
4. Run two policy-only seen diagnostics (one per Y side), `n_action_steps=10`,
   `max_steps=150`, geometry enabled. The key question is lateral alignment
   BEFORE descent/close, not merely whether close is emitted.
5. If promising, evaluate all six seen positions. Do not test interpolation
   yet if seen control still fails.

## Experiment results

All training and online evaluations below were run by the operator. The coding
agent inspected the saved configurations, CSV, and tee logs.

### Training and offline audit

- Five-update XPU smoke: passed; custom loss recipe and checkpoint saved.
- Full weighted run: 2,000 updates, batch size 8, nominal epochs `7.85`.
- Runtime: about 5m07s. Final logged weighted L1 `0.328`, unweighted L1 `0.308`,
  KL `0.081`, total loss `1.140`. The weighted total is not directly comparable
  to the Day7 unweighted training objective.

| Offline metric, SAME Day7 v2 observations | Unweighted Day7 v2 | Weighted Day8 |
|---|---:|---:|
| Active Y sign-and-magnitude hit, threshold 0.05 | 70.5% | 91.5% |
| Negative-side active Y prediction mean | -0.046 | -0.214 |
| Positive-side active Y prediction mean | +0.070 | +0.230 |
| Neutral Y false-motion, threshold 0.05 | 0.4% | 14.7% |
| Gripper close hit, threshold 1.5 | 66.7% | 44.2% |
| Gripper hold false-close, threshold 1.5 | 0.0% | 0.0% |

Of the weighted model's 245 neutral-frame Y false motions, 201 occur before
the first demonstrated Y movement, 36 in pauses within the Y-movement span,
and 8 after the final demonstrated Y movement. No threshold-crossing Y false
motion occurs on expert descent, close, or lift label frames. This is an
offline stage analysis, not evidence that online motion is always safe or
precisely aligned. The gripper threshold is an audit convention, not a binary
physical closing threshold.

### Online control ablation

All conditions use the same two seen positions `x=0.48, y=+/-0.10`, policy-only
gripper execution, and a 150-step maximum. The two trained models use the same
dataset, initialization seed 1000, model recipe, and update count.

| Training loss | Evaluation n_action_steps | Seen result | Evidence |
|---|---:|---:|---|
| Unweighted | 10 | 0/6 | Day7 experiment record |
| Unweighted | 5 | 0/6 | `results/day8_unweighted_n5_seen_eval.log` |
| Y-active weight 4 | 10 | 0/2 | Operator-provided pilot log |
| Y-active weight 4 | 5 | 6/6 | `results/day8_y_weighted_w4_n5_seen_eval.log` |

The weighted/n5 pilot also succeeded 2/2; keep it separate from the formal 6/6
result. At step 20 in the logged n5 seen runs, unweighted TCP Y is about
`-0.0055` or `+0.0078`, while weighted TCP Y is `-0.1026` or `+0.1136`. The
weighted policy approaches the cube rather than descending far off-target.

This comparison supports an advantage of the weighted training recipe at the
same n5 evaluation setting. Reducing the execution block alone did not solve
the baseline, and weighting alone at n10 did not yield success in its two
pilot episodes. Do not present either change as universally necessary or
sufficient: there is one training seed, a small set of positions, unequal
rollout counts, and repeated essentially deterministic resets.

### Held-out midpoint

Weighted/n5 at `x=0.48, y=0.00`: **5/5**, with complete log in
`results/day8_y_weighted_w4_n5_interpolation_eval.log`. This cube initial
position is absent from the Day7 v2 training schedule. All five episodes repeat
the same midpoint; they are not five distinct unseen positions. Since the home
TCP Y is already near zero, this test needs little lateral movement and does
not establish reliable interpolation throughout the Y interval.

### Held-out nonzero interpolation

The prespecified weighted/n5 evaluation at `x=0.48, y=-0.05/+0.05` completed
**3/6**, but the aggregate hides a directional failure:

| Scheduled Y | Successes | Behavior |
|---|---:|---|
| `-0.05` | 3/3 | Approach, overshoot, partial correction, then successful lift |
| `+0.05` | 0/3 | Overshoot toward approximately +0.10, close off-target, no lift |

Full log: `results/day8_y_weighted_w4_n5_nonzero_interpolation_eval.log`.

For negative Y, TCP Y is about `-0.0468` at step 10, `-0.0675` at step 20,
then `-0.0616` at step 30; the cube also moves during contact. Success here does
not mean millimeter-level alignment without contact-induced displacement.

For positive Y, TCP Y is `+0.0408` at step 10 (9.2 mm short of the cube), but
continues to `+0.0935` at step 20 with a target at `+0.05`. By step 30 it
is near `+0.10`, with the cube at about `+0.0582` and the gripper closing control
saturated. This leaves roughly 42 mm of Y error. All three positive-side
repetitions show the same early trajectory and fail; this is not three
independent random misses or evidence that more episode time will fix it.

Conclusion: the recipe solves the tested seen positions and exhibits partial,
asymmetric interpolation. It does not establish a reliable continuous mapping
from visual target distance to movement magnitude. Attraction toward the
trained positive endpoint and stale chunk execution are hypotheses, not proven
internal mechanisms. No extrapolation, randomized-position, or multi-seed claim
is justified.

### Step-by-step replanning diagnostic

The SAME weighted checkpoint was tested for one episode per `+/-0.05` with
`n_action_steps=1`: **1/2**, negative side successful, positive side unsuccessful.
Log: `results/day8_y_weighted_w4_n1_nonzero_interpolation_diagnostic.log`.

On the positive side, at step 10 the TCP is already at Y `+0.0508`, but the
logged action commands positive Y movement. Geometry is logged after applying
the action, so this line alone does not establish what the model saw before
that action. At step 20 TCP Y is `+0.0730`
and cube Y is `+0.0587`, reducing the lateral error compared with n5's roughly
43.5 mm error at the same step. At step 30 the TCP and cube Y are approximately
`+0.0741/+0.0740`: the cube has moved through contact, so this is not clean
alignment with an undisturbed target. The policy then remains near the same
configuration with closing control saturated and small negative Z commands;
the cube stays at Z about `0.0186` instead of lifting. Closing control is not a
measurement of finger spacing or proof of a grasp.

Thus n1 changes/improves parts of the approach but does not complete the
positive-side task. Stale chunk execution alone is not a sufficient explanation
or fix; the observed problem also involves the stopping/approach-to-grasp/lift
transitions in this unseen scene. This single diagnostic does not isolate
their internal causes. Do not replace the n5 reference setting with n1 or
continue arbitrary queue-length sweeps based on these few trials.

Retain the n5 benchmark unchanged and document the n1 test separately. Any
configuration adjusted using these observed failures is a development
diagnostic, not a fresh held-out benchmark; use additional untouched positions
for a later generalization claim. Do not recollect at the test positions and
still describe them as held out. Day8 establishes a controlled improvement on
seen positions plus limited/asymmetric interpolation, not general spatial
manipulation. A subsequent investigation should compare successful expert
states and failing positive-side states around stopping, closing, and lifting,
rather than blindly increasing training steps or execution-time limits.

## Saved evidence and session closeout

- `results/day8_summary.json`: structured results extracted from the five saved
  rollout logs, grouped by position and control setting. Diagnostic runs are
  separate from reference evaluations; no mixed aggregate success rate is used.
- `results/day8_loss_recipe.json`: the actual full-run custom-loss recipe from
  the 2,000-update checkpoint (source hashes included).
- `results/day8_training_tail.log`: final metrics/save evidence from the full
  training log; the complete training log remains in the ignored local outputs.
- `results/offline_gripper_audit_day8_y_weighted_w4_xpu_2000.csv`: all 2,037
  independent expert-frame predictions. It is an in-dataset diagnostic, not a
  held-out action accuracy measurement.
- Five `results/day8_*.log` files: online evaluation/diagnostic evidence, with
  terminal color escapes removed for readable diffs. Numeric values are intact.

Datasets, videos, checkpoints, and the five-step smoke output remain local.
The selected artifacts and code are intended for the Day8 GitHub closeout.
No new training run or online rollout was executed during closeout.
