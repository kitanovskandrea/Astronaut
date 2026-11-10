# Ten-minute, two-speaker presentation

Group 7: Andrea Kitanovska (89262058) and Hristijan Kochovski (89262125).

Use the current poster and `RESULTS.md`. Quote final recorded measurements, not development pilots or the earlier shaped-reward benchmark. Both speakers should understand the entire learner and be able to explain the curriculum tradeoff.

| Time | Speaker | Content |
| --- | --- | --- |
| 0:00-0:45 | Andrea | Introduce flight, asteroid avoidance and oxygen collection. State that every action is learned by an explicit tabular Q learner in the official course simulator. |
| 0:45-1:45 | Andrea | Explain public height/velocity and object arrival/projected-height bins. Show capacities 72, 2,880 and 103,680. Tank collection retains the complete hazard prefix; stored inherited rows differ from child-updated rows. |
| 1:45-3:00 | Andrea | Show the Q update for an action lasting up to six frames. Explain discounted frame rewards, `gamma_frame**executed_frames`, collision termination and time-limit bootstrapping. Official rewards remain -1/+1/0; there is no shaped reward or scripted controller. |
| 3:00-4:15 | Andrea | Explain the learned sequence: zero-table flight, copied flight values for asteroid suffixes, then copied hazard-aware values for tank suffixes. Show why the child initially reproduces its parent's actions. Five chains are independent; stages inside a chain share learning. |
| 4:15-5:00 | Andrea | Explain the earlier regression briefly: multiple learning conditions changed together, tests checked consistency, behavior weakened. Explain why survival gates now block promotion and oxygen count is considered only after safety. Hand over to Hristijan. |
| 5:00-6:15 | Hristijan | Describe checkpoint-zero baselines, two consecutive gate passes, separate validation/final seeds and fixed per-stage budgets. Distinguish selected and final-budget checkpoints. Explain what happens when a gate is never met. |
| 6:15-7:30 | Hristijan | Read measured survival, completion and oxygen results from RESULTS.md. Show validation behavior curves alongside TD loss. State actual seed variation and remaining failure modes; small loss alone proves neither good play nor convergence. |
| 7:30-8:15 | Hristijan | Compare incremental and cumulative training cost. Flight has one million Q decisions, asteroids 1.5 million and oxygen two million; actual physics steps differ at episode endings. Do not call this an equal-budget from-zero comparison or time to convergence. |
| 8:15-9:15 | Hristijan | Demonstrate the fixed seed-11 oxygen policy at the declared demo seed. Python runs real physics; browser interpolation only smooths the display. Explain one trajectory without presenting it as the benchmark average. |
| 9:15-10:00 | Hristijan, then Andrea | Hristijan shows the reproduction commands, raw data and hashes. Andrea closes with one final measured finding and one limitation, such as state aliasing, finite validation coverage or curriculum-dependent costs. |

## Prepare the demonstration

From the extracted program folder with its environment active:

```sh
python -m unittest discover -s tests -v
python -m astronaut_rl.verify --results results --models models
python -m astronaut_rl.play models/oxygen_seed11.json.gz --seed 9590000
```

Keep the terminal running. Use the task buttons to show flight and asteroid policies. The seed-11 chain is fixed in advance, not selected for the highest final-test score. The browser remains available after an episode. If a graphical display is unavailable:

```sh
python -m astronaut_rl.play models/oxygen_seed11.json.gz --headless --seed 9600000 --episodes 100
```

Run training before the talk, not during the ten-minute presentation. **Train from scratch** is a new training job; advanced stages declare their parent-policy initialization. A short run that does not qualify must not be presented as a learned successful policy.

## Questions to rehearse

- Why can the smaller state merge situations that need different actions?
- How does the projected offset use public velocity without prescribing an action?
- Why does the oxygen table retain asteroid features even when a tank is nearer?
- How are learned parent values copied, and why do child visit counters start at zero?
- Why is frame discounting consistent with an action repeated for a variable number of frames?
- What exact survival and tank constraints must a checkpoint pass, and why are two consecutive checks required?
- Why can a table with low TD loss still play badly under sparse rewards?
- How do validation selection and unseen final testing differ?
- Why are five chains, rather than their test episodes, the statistical replicates?
- Why must cumulative training cost include prerequisite stages?
- What changed from the earlier failed revision, and what do the final measurements actually establish?

Both students must submit the same ZIP, PDF and Markdown instructions individually, using `7_IS_Seminar1_Kitanovska_Kochovski` as the basename. Submit one day before the presentation; attendance and the joint presentation remain required.
