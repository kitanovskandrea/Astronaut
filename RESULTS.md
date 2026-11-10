# Measured results

All values below use the official course rewards: -1 for collision, +1 for
an oxygen pickup without collision, and 0 otherwise. No reward shaping is
used. Means +/- sample SD are across independent training policies, after
averaging the episodes for each policy. Curriculum chains are independent;
stages inside a chain inherit learned values. The checkpoint for each run is
selected using validation only; the selected policy is then frozen for testing.

## Selected-policy test performance

| Variant | Policies | Test episodes / policy | Frames | Full survival (%) | Tanks / episode | Return |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| Flight | 5 | 1 | 1200.0 +/- 0.0 | 100.0 +/- 0.0 | 0.00 +/- 0.00 | 0.00 +/- 0.00 |
| + Asteroids | 5 | 100 | 1193.0 +/- 13.1 | 98.6 +/- 2.6 | 0.00 +/- 0.00 | -0.01 +/- 0.03 |
| + Oxygen | 5 | 100 | 1185.0 +/- 12.5 | 96.8 +/- 3.1 | 5.32 +/- 0.61 | 5.29 +/- 0.61 |

Flight has no stochastic reset/spawn variations: its episode seeds do not
create different scenarios. Training-seed variability still measures the
independently explored policies listed in the table.

## Random baseline

| Variant | Frames | Full survival (%) | Tanks / episode |
| --- | ---: | ---: | ---: |
| Flight | 225.0 +/- 0.0 | 0.0 +/- 0.0 | 0.00 +/- 0.00 |
| + Asteroids | 133.9 +/- 0.0 | 0.0 +/- 0.0 | 0.00 +/- 0.00 |
| + Oxygen | 133.6 +/- 0.0 | 0.0 +/- 0.0 | 0.07 +/- 0.00 |

The random policy uses the same test seeds. Its repeated evaluations across
training-seed records are the same baseline, not additional independent samples.

## Allocated stage cost, cumulative cost and selected checkpoints

| Variant | Q decisions | Physics steps | Incremental seconds | Cumulative seconds | Selected Q decision | Qualified chains |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Flight | 1000000 +/- 0 | 5978066 +/- 259 | 40.66 +/- 0.16 | 40.66 +/- 0.16 | 100000 +/- 0 | 5/5 |
| + Asteroids | 1500000 +/- 0 | 8991983 +/- 278 | 183.71 +/- 0.23 | 224.37 +/- 0.32 | 1040000 +/- 167332 | 5/5 |
| + Oxygen | 2000000 +/- 0 | 11993966 +/- 1146 | 432.57 +/- 0.60 | 656.94 +/- 0.73 | 1180000 +/- 396232 | 5/5 |

Each completed stage continues to its allocation even if an earlier checkpoint is selected.
Flight and later stages have different Q-decision budgets; decisions execute up
to six actual physics frames. Cumulative costs include full prerequisite training
within each chain. This is not an equal-budget, independent-from-zero comparison.
A stage that fails its behavioral gate is recorded as unqualified and is not
promoted to the next stage. The main model file holds the selected policy; the separate final
checkpoint records the end-of-budget learner. The timer covers training
interaction and Q updates, excluding checkpoint validation, model export and rendering.

Timing protocol: independent parallel chains; sequential stages within each chain; active per-process wall time under concurrent load excludes initialization, validation, test, export and progress IO.

## Oxygen and collision diagnostics

| Variant | Tanks / 1,000 steps | Complete with >=3 tanks (%) | Ground (%) | Ceiling (%) | Asteroid (%) |
| --- | ---: | ---: | ---: | ---: | ---: |
| Flight | 0.00 +/- 0.00 | 0.00 +/- 0.00 | 0.00 +/- 0.00 | 0.00 +/- 0.00 | 0.00 +/- 0.00 |
| + Asteroids | 0.00 +/- 0.00 | 0.00 +/- 0.00 | 0.40 +/- 0.89 | 0.00 +/- 0.00 | 1.00 +/- 1.73 |
| + Oxygen | 4.49 +/- 0.51 | 93.20 +/- 2.86 | 0.00 +/- 0.00 | 1.00 +/- 1.73 | 2.20 +/- 1.92 |

Tank rate is calculated separately for each policy as 1,000 times total tanks
divided by total survived physics steps, then aggregated across training seeds.
Collision percentages describe outcomes; longer survival can expose a policy
to more asteroids, so they must be read alongside frames and completion.

## Coverage and sample efficiency

| Variant | Inherited rows | Stored rows | Updated states | Updated pairs | Updated pairs with <10 updates (%) | Absent test rows (%) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Flight | 0 +/- 0 | 68 +/- 1 | 67.6 +/- 0.5 | 134.4 +/- 1.1 | 4.9 +/- 0.6 | 0.00 +/- 0.00 |
| + Asteroids | 2696 +/- 22 | 2707 +/- 12 | 1620.2 +/- 9.7 | 3103.8 +/- 16.2 | 24.3 +/- 0.5 | 0.00 +/- 0.00 |
| + Oxygen | 97430 +/- 447 | 97436 +/- 442 | 11955.6 +/- 219.1 | 20521.0 +/- 519.0 | 48.3 +/- 1.0 | 0.00 +/- 0.00 |

Coverage columns count actual child-stage updates, not copied parent values.
Stored inherited rows can have zero child visits. Unseen test decisions
refer to absent rows in the selected table; a present inherited row need not have
received any child update. The complete
coverage statistics for both tables are in the run records; `aggregate.json`
contains means/SD for all numeric full-budget coverage fields.

- **Flight**
  First target checkpoint: 5/5 policies; 100000 +/- 0 Q decisions among reached runs.
  Consecutive gate stability: 5/5 policies; 200000 +/- 0 Q decisions among reached runs.
- **+ Asteroids**
  First target checkpoint: 5/5 policies; 380000 +/- 130384 Q decisions among reached runs.
  Consecutive gate stability: 5/5 policies; 680000 +/- 109545 Q decisions among reached runs.
- **+ Oxygen**
  First target checkpoint: 5/5 policies; 340000 +/- 313050 Q decisions among reached runs.
  Consecutive gate stability: 5/5 policies; 620000 +/- 521536 Q decisions among reached runs.

Behavioral promotion gates are frozen engineering requirements, not professor-set
grading thresholds. Oxygen collection is ranked only after the survival floor
is met, and two consecutive checks are required for promotion. Runs that never
reach a target are reported explicitly, not replaced by zero steps.

## Learning curves and reproduction

`figures/loss.svg` shows pre-update squared TD residuals in fixed-decision blocks.
Flight and avoidance residuals decrease strongly. Oxygen residuals persist and rise later;
this run demonstrates better collection behavior, not zero-loss convergence for oxygen.
`survival.svg`, `completion.svg` and `oxygen.svg` show frozen validation
performance at checkpoint zero and later checkpoints; `coverage.svg` shows updated states.
The x axis is Q decisions; phase budgets differ and actual physics steps are reported separately.
Lines are policy means and bands are +/- one sample SD (clipped at the
physical metric bounds when needed). Checkpoint curves use validation,
while the tables above use the separate final test set.

`training_time.svg` and `performance.svg` compare costs and selected-policy
test performance. Raw episode CSVs, validation CSVs and model hashes are
retained. Regenerate this report with `python -m astronaut_rl.report`.

Validation seeds are 9590000..9590049 for stochastic stages; final test seeds are
9600000..9600099. Flight uses one deterministic episode at each evaluation.

The figures show finite-budget behavior. Sparse zero rewards can produce
small TD residuals even for weak policies, so low loss alone is not evidence
of success or theoretical convergence. Compact state bins still merge
continuous situations. Sample SD across chains is not a confidence interval.
