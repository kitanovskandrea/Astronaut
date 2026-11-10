# Methods and reproducibility

Group 7: Andrea Kitanovska (89262058) and Hristijan Kochovski (89262125).

## Environment and implementation

The simulator is the unmodified official `famnit_gym/Astronaut-v1`, with the course `Asteroids` and then `Oxygen` wrappers added for the harder stages. The pinned upstream commit is `0db0652ad3b8cece05d476c56637a3dcb56557cd`. The original wheel, MIT license, provenance and critical installed-source hashes are bundled. Upstream reused version 0.2, so its version number alone does not identify the source.

The current defaults use 60 physics steps per second, horizontal scroll speed 500 pixels/second, gravity 400 pixels/second squared and net upward acceleration 600 pixels/second squared while thrusting. An episode ends on collision or the 1,200-step time limit. We do not edit the package or override its physics or reward callback. Default rewards are -1 on collision, +1 on oxygen pickup without collision, and 0 otherwise. Collision takes priority.

The Q learner, feature encoding, experiment controller, aggregation, SVG plots and vector PDF writer are original Python standard-library code. The course package uses Gymnasium, NumPy and Pygame. Its environment-owned random generator supplies spawning. Training seeds the environment once, then resets with `None` so its sequence continues; each evaluation episode resets an independent environment with its designated seed.

## Why the training design changed

The assessment-driven revision changed multiple effective learning conditions together: it removed earlier shaped rewards, updated the upstream physics, changed action repeat from six to one, coarsened the state, shortened the physical discount horizon and reduced actual interaction from about 5.95 million to two million physics frames per run. Its consistency checks passed, but its asteroid and oxygen behavior weakened. Those changes were not isolated, so the data cannot identify one causal effect. The error was promoting a methodological revision without requiring the desired behavior to survive the changes.

The recovery keeps the current official rewards and physics, restores a suitable six-frame decision interval, uses a longer physical discount horizon, and trains a sequence that preserves already learned skills. Its checkpoints must satisfy survival constraints before promotion. The assessment's suggested compact bin counts and independent-from-zero stages were engineering choices, not a requirement to discard a useful learned skill. The curriculum is disclosed rather than compared as if it were an independent-from-scratch experiment. `ASSESSMENT_RESPONSE.md` explains the regression and scope of the comparison.

## Public-observation state

Each encoded state begins with astronaut height and vertical velocity. Asteroid states append arrival time and a projected vertical offset. Oxygen states retain that entire hazard-aware prefix and append tank arrival time and projected vertical offset, or a unique absent-tank code. The oxygen stage never drops the asteroid features merely because a tank is closer.

Bins use `bisect_right`, so equality with a threshold belongs to the higher bin. Thresholds and state type are saved in every model; finite outer bins bound the representation.

| Feature | Thresholds | Bins |
| --- | --- | ---: |
| Astronaut height | `160,230,310,400,490,570,640` | 8 |
| Vertical velocity | `-300,-180,-90,-30,30,90,180,300` | 9 |
| Object arrival time | `0.18,0.4,0.7,1.2` seconds | 5 |
| Projected asteroid offset | `-240,-160,-100,0,100,160,240` pixels | 8 |
| Projected tank offset | `-220,-120,-50,50,120,220` pixels | 7 |

For object center `(x_o,y_o)` and astronaut height/velocity `(y,v)`:

```text
tau = max(0, (x_o - 300)/500)
projected_dy = y_o - y - v*min(tau, cap)
cap = 0.75 seconds for asteroids; 0.60 seconds for oxygen
```

This is an agent-side, first-order feature of public observations. It accounts for current motion when grouping states; it is not a collision oracle or action rule, and it does not alter physics. Acceleration and future actions are not predicted.

The public tank coordinates persist after collection. Availability starts true on reset, becomes false on `oxygen_picked == "yes"`, and becomes true after its x position jumps more than 100 pixels to the right on respawn. This bookkeeping runs every physics frame. An absent tank uses a separate code outside the five available arrival bins.

| Stage | State tuple | Capacity | State-action capacity |
| --- | --- | ---: | ---: |
| Flight | `(height, velocity)` | `8*9 = 72` | 144 |
| Asteroids | Flight prefix + asteroid `(time, projected_dy)` | `72*5*8 = 2,880` | 5,760 |
| Oxygen | Asteroid prefix + tank `(time, projected_dy)` or absent code | `2,880*(5*7+1) = 103,680` | 207,360 |

These are capacities, not counts of reachable or updated states. Coarse bins still merge physically different situations, so the representation is not claimed to be exactly Markov. Actual stored-row and visit counts are reported separately.

## Learned curriculum and initialization

There are five independent training chains, using seeds `11,23,37,51,79`. Within each chain:

1. Flight starts with zero Q values and learns from the official environment.
2. The selected qualified flight table initializes the asteroid table: each learned flight row is copied across the new asteroid suffixes. Initial greedy actions therefore reproduce the flight policy before asteroid learning.
3. The selected qualified asteroid table initializes the oxygen table across the new tank suffixes. This preserves its hazard-aware policy before tank-specific Q updates begin.

The parent comes from the same chain's validation-selected checkpoint, not another chain or a final-test search. Copied entries contain learned action values; they are not heuristic actions. Training then uses the same classical Q-learning implementation and epsilon-greedy actions. Parent provenance and hashes are recorded.

Child visit counts start at zero. Copied rows can therefore be stored without having received any child-stage updates. A count of stored rows must not be described as a count of visited states; visitation statistics use actual child updates. Similarly, a test state can have inherited Q values without having been visited during child training. The report distinguishes these cases.

The five chains are independent replicates. Stages within a chain are dependent by design. This experiment answers whether a learned sequence preserves flight and avoidance while adding collection; it does not estimate the cost or quality of training each task from zero.

## Q-learning with six-frame decisions

Actions are 0 (coast) and 1 (thrust). Each selected action executes for up to six physics frames, stopping immediately on collision or truncation. The Bellman update uses the number of frames actually executed, including shortened final decisions.

With physical-frame discount `g = 0.9995`, executed frame count `k`, and official rewards `r_0,...,r_(k-1)`:

```text
R = sum(g**j * r_j for j in range(k))
target = R                               if truly terminated
target = R + g**k * max_a Q(s_next,a)      otherwise
delta = target - Q(s,a)
Q(s,a) = Q(s,a) + alpha*delta
```

The nominal six-frame discount is `g**6`, approximately 0.997004. This is a semi-Markov form of classical Q-learning: rewards within the repeated action and the bootstrap use consistent physical discounting. A collision has no bootstrap. The external time limit is a truncation and still bootstraps. No survival, distance, missed-tank or shaped rewards are added.

Alpha is 0.03 for flight and 0.08 for later stages. Flight exploration starts at 1.0 and decays to 0.05 over 650,000 decisions. Asteroids starts at 0.25 and decays to 0.01 over 650,000 decisions. Oxygen starts at 0.25 and decays to 0.05 over 1,300,000 decisions. Each then remains at its floor. Exact values are frozen in the supplied configurations. Exploration and training ties use the chain's seeded Python RNG. Evaluation has no exploration, resolves ties deterministically and does not update Q values, visits, epsilon or the training RNG.

## Budgets, checkpoints and promotion

| Stage | Allocated Q decisions | Maximum physics steps | Allocated cumulative decisions |
| --- | ---: | ---: | ---: |
| Flight | 1,000,000 | 6,000,000 | 1,000,000 |
| Asteroids | 1,500,000 | 9,000,000 | 2,500,000 |
| Oxygen | 2,000,000 | 12,000,000 | 4,500,000 |

Actual physics-step totals are slightly smaller when an episode ends during a repeated action. Each stage completes its full allocation. An initialization checkpoint at decision zero records inherited behavior; subsequent checks occur every 100,000 Q decisions. Initialization is not counted as newly trained behavior.

A checkpoint must meet the frozen survival/collection gate for two consecutive checks before it may be promoted. Flight requires full survival on its deterministic validation episode; asteroids requires at least 90% survival. Oxygen's survival floor is the larger of 90% and its inherited checkpoint-zero survival minus five percentage points. Its pickup floor is the larger of three tanks per episode and checkpoint-zero collection plus 0.25. These are engineering acceptance gates, not professor-specified seminar grading criteria.

For oxygen, tank collection is ranked only after the survival constraint is met. A higher pickup count cannot rescue a checkpoint that violates the survival floor. Checkpoint zero remains an available inherited reference, so new learning is not automatically presented as an improvement over the parent. If no checkpoint qualifies for promotion, the run records that failure, retains its safest available table and the curriculum does not advance. A short smoke budget failing a gate is not a software error or evidence of successful training.

The model selected for evaluation is identified separately from the final-budget checkpoint. A later learner can regress; finishing more updates does not make its final table the best one. Selection uses validation only, and ties retain the earlier checkpoint according to the declared rule. Final tests do not alter the selected policy.

## Validation, final tests and metrics

Development pilots and feature tuning use separate development episodes. Frozen recovery validation uses `9590000..9590049` for asteroid and oxygen tasks. Final testing uses previously unused seeds `9600000..9600099`. Flight has a deterministic reset and no spawns, so it uses one episode at each evaluation; changing its seed does not create a new flight scenario. Test counts and exact seed lists are recorded in configuration and run metadata.

The same scenario seeds are shared across policies for paired comparisons. A uniform-random-action baseline uses the same action interval and environment. Any parent-policy baseline is identified explicitly; its copied initialization is not described as a separately trained oxygen policy.

Per-policy metrics include mean survived physics steps, full 1,200-step completion, oxygen tanks per episode, official undiscounted episode return, collision distribution, oxygen per 1,000 survived physics steps, completion with at least three tanks, and state/visit diagnostics. Oxygen rate is `1000*sum(tanks)/sum(physics_steps)` within each policy, then aggregated across chains. Raw return and discounted Q-training reward are distinct quantities.

The report first averages evaluation episodes within each learned policy, then reports means and sample standard deviations across the five independent chains. It does not pool episodes as independent training replicates. SD bands are descriptive, not confidence intervals. A finite validation gate does not guarantee success on every possible spawn sequence.

## Loss and visitation diagnostics

TD loss is the squared residual before updating the corresponding Q entry. Fixed blocks of Q decisions log its mean. Curves show chain means with one sample SD; validation survival, completion and collection curves show behavior separately. The x axis is Q decisions, not physics frames, and phases have different allocated lengths.

Official rewards are usually zero. Small TD residuals can coexist with repeated collisions, so low loss alone does not prove successful play or theoretical convergence. Child-stage inherited values can also change initial loss behavior without adding child updates.

Visitation counters count each state-action update. Summaries distinguish stored/inherited rows, actually updated states/pairs, unvisited actions, positive-count pairs with fewer than ten updates and larger-count thresholds. Unreachable state combinations can explain low capacity coverage, so coverage alone is not an optimization target. Budget-interrupted episodes are marked explicitly rather than counted as collisions or complete episodes.

## Training cost and reproducibility

The submitted batch runs five independent chains in parallel, with stages inside each chain running sequentially. Reported seconds are active wall-clock time per process under that concurrent workload, not isolated CPU time or total batch elapsed time. Hardware load affects these values. The active training timer includes environment interaction, feature encoding, Q updates and within-stage resets, while excluding validation, final testing, checkpoint exports, progress-file IO and rendering. Each record reports the stage's incremental training cost. Cumulative cost also includes the full allocated training of its prerequisite stages, even when an earlier parent checkpoint was selected. Actual physics steps and Q decisions are both recorded.

Flight and the later stages have different budgets, and later stages inherit training. Runtime therefore compares measured stage costs; it is not an equal-budget from-zero benchmark and not time to convergence. A sequential rerun produces the same learned tables under the same settings but different timings. Wall time remains dependent on the machine and other activity.

Models contain state specifications, reward/environment provenance, discounts, initialization, gates, selected/final steps and hashes. Raw CSVs retain training episodes, loss blocks, validation outcomes and final evaluations. `astronaut_rl.verify` checks artifacts and recomputes evaluations without learning. `astronaut_rl.report`, `plots` and `poster` read those records. Tests establish implementation consistency; the measured survival and collection evidence establishes behavior.

## Display

Python runs the actual game at 60 physics frames per second and chooses Q actions at the saved decision interval. Scene snapshots stream to a local browser, which draws official sprites using a short interpolation buffer. Its animation contains no game physics or policy. Watching a model does not update it. Training runs in a separate process and the saved selected policy is replayed when ready.

## References

- Watkins, C. J. C. H. and Dayan, P. Q-learning. Machine Learning 8, 279-292 (1992). https://www.gatsby.ucl.ac.uk/~dayan/papers/cjch.pdf . The update is independently implemented here.
- Domen Soberl. Official famnit-gym course environment, source commit `0db0652ad3b8cece05d476c56637a3dcb56557cd`. https://github.com/DomenSoberlFamnit/famnit-gym . The unchanged package and license are bundled.
