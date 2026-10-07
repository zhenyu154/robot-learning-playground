# Day14: Staged 2D Demonstrations and Weight Ablation

**Experiment date:** October 6, 2026

## Research question

Can explicit temporal structure in the demonstrations improve the transition
that Day13 failed to execute reliably?

The target sequence was made explicit:

```text
XY approach -> neutral hover -> descent -> close -> closed stabilization -> lift
```

A second question was whether the active-motion loss multiplier of 4 was too
strong relative to neutral transition states.

## Dataset and staged protocol

Dataset:

```text
/home/wusanggg/robotics/data/panda_pick_cube_day14_xy_staged_v1
```

Training positions:

```text
(0.32,-0.075), (0.32,+0.075),
(0.48,-0.075), (0.48,+0.075)
```

The final dataset used 10 successful demonstrations per corner:

```text
40 episodes, 3,765 frames
```

Each demonstration included a neutral hover after XY motion, a descent,
stationary closed-gripper frames before lifting, and 10 positive-Z lift frames.
The automated QA initially used an 8-frame close threshold; 11 episodes had
6--7 thresholded close frames. Since all demonstrations succeeded, had at least
3 stationary closed frames, and had 10 lift frames, the dataset was accepted
with `--min-close-frames 6`. Manual video inspection found no issue.

QA script:

```text
scripts/audit_day14_xy_staged_dataset.py
```

Dataset QA summary:

```text
40/40 reward-positive demonstrations
40/40 done-positive episodes
10 episodes per corner
no simultaneous X/Y activity
10 positive-Z lift frames per episode
no X/Y activity during lift
```

## Common training recipe

All three models used the same dataset, seed, 7,000 updates, batch size 8,
`chunk_size=50`, and close-frame weight 4. The active weight was varied:

```text
active_weight in {4, 2, 1}
close_weight = 4
```

The `active_weight=4` model was trained first, followed by the two ablations.
Training at 7,000 updates corresponds to approximately 14.87 dataset passes,
comparable to Day13's 14.53 passes.

## Offline metrics

### Active weight 4

| Metric | Result |
|---|---:|
| X sign-and-magnitude hit | 99.4% |
| Y sign-and-magnitude hit | 93.7% |
| X neutral false motion | 6.3% |
| Y neutral false motion | 5.8% |
| Gripper close hit | 97.5% |
| Hold false-close | 5.3% |
| Active Z hit | 99.8% |
| Neutral Z false motion | 15.4% |

### Active weight 2

| Metric | Result |
|---|---:|
| X sign-and-magnitude hit | 97.2% |
| Y sign-and-magnitude hit | 90.9% |
| X neutral false motion | 1.9% |
| Y neutral false motion | 3.3% |
| Gripper close hit | 98.8% |
| Hold false-close | 4.2% |

The active-weight-2 model had lower active-action accuracy but substantially
better neutral-motion behavior.

## Policy-only seen evaluation

The same 12-episode schedule was used for each checkpoint.

| Model | Result |
|---|---:|
| Day13 XYZ weight 4, close 4 | 2/12 |
| Day14 XYZ weight 4, close 4 | 3/12 |
| Day14 XYZ weight 2, close 4 | **8/12** |
| Day14 XYZ weight 1, close 4 | 1/12 |

The active-weight-2 result by corner was:

| Corner | Result |
|---|---:|
| `(0.32,-0.075)` | 3/3 |
| `(0.32,+0.075)` | 3/3 |
| `(0.48,-0.075)` | 0/3 |
| `(0.48,+0.075)` | 2/3 |
| **Total** | **8/12** |

The active-weight-1 model often failed to complete the Y approach and was
therefore too weak. The active-weight-4 model predicted active actions very
accurately offline but remained less reliable online, consistent with
overemphasis of active movement relative to neutral transition states.

## Interpretation

The staged dataset and weight reduction produced a substantial improvement:

```text
Day14 active 4: 3/12
Day14 active 2: 8/12
```

The best current setting is therefore:

```text
active X/Y/Z weight = 2
close-frame weight = 4
```

This is not yet a complete 2D generalization result. All evaluations so far
use seen training corners. The persistent weakness is the far negative-Y
corner `(0.48,-0.075)`, where the policy may descend and close but still fail
to initiate a sufficient lift.

## Grid-search plan for the next session

A targeted rather than exhaustive search is appropriate. The completed shared
active-weight results are:

```text
active=1 -> 1/12
active=2 -> 8/12
active=4 -> 3/12
```

The next candidates should be:

```text
active in {1.5, 2.5, 3.0}, close=4
```

Use four-corner screening first, then run the full 12-episode evaluation only
for the best candidate(s). If shared active weighting does not improve beyond
8/12, test separate XY and Z weights instead of continuing a blind scalar
search.

## Evidence

```text
results/offline_gripper_audit_day14_xy_staged_xyz4_closew4_xpu_7000.csv
results/offline_gripper_audit_day14_xy_staged_xyz2_closew4_xpu_7000.csv
results/day14_xy_staged_xyz4_closew4_7000_n5_seen_eval.log
results/day14_xy_staged_xyz2_closew4_7000_n5_seen_eval.log
results/day14_xy_staged_xyz1_closew4_7000_n5_seen_eval.log
```
