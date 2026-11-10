# Reinforcement learning with the Astronaut

Group 7: Andrea Kitanovska (89262058) and Hristijan Kochovski (89262125).

Original tabular Q-learning for the official Astronaut game. The training sequence learns flight, extends that learned policy to asteroid avoidance, then extends avoidance to oxygen collection. Checkpoint promotion requires measured survival; collecting more tanks cannot override the oxygen stage's survival constraint. Actual results and any unmet gates are reported in `RESULTS.md`.

## Install or update

Use Python 3.12. Extract the ZIP and open a terminal in the folder containing this README.

```sh
python -m venv .venv
```

Activate it on macOS/Linux with `source .venv/bin/activate`, on Windows PowerShell with `.venv\Scripts\Activate.ps1`, or in Command Prompt with `.venv\Scripts\activate.bat`.

```sh
python -m pip install -r requirements.txt
```

If updating an earlier installation, explicitly reinstall the bundled course snapshot because upstream reused version `0.2`:

```sh
python -m pip install --force-reinstall --no-deps vendor/famnit_gym-0.2-py3-none-any.whl
```

The unchanged wheel, license and source hashes are included. Its declared dependencies may need internet access on first installation. The learner, encoders, evaluation, statistics, SVG figures and PDF writer are original Python standard-library code. Gymnasium, NumPy and Pygame are dependencies of the course environment.

## Watch the learned policies

Run one command at a time:

```sh
python -m astronaut_rl.play models/fly_seed11.json.gz --seed 9590000
python -m astronaut_rl.play models/asteroids_seed11.json.gz --seed 9590000
python -m astronaut_rl.play models/oxygen_seed11.json.gz --seed 9590000
```

`python -m astronaut_rl.play` opens the default oxygen policy. Every action comes from the Q table; there are no player or Space-key movement controls. Python executes the actual course simulator at 60 physics frames per second. The browser draws the official sprites and interpolates streamed positions; it contains no second physics simulation or action controller.

Keep the terminal running. If no browser opens, use the URL it prints. The page stays open after an episode; **Watch trained AI** repeats an attempt and the task buttons switch versions. **Stop server** or Ctrl+C stops the server. `--native` opens the course window when a desktop video driver is available.

A fixed seed-11 policy demonstrates behavior; it is not chosen as the best final-test seed. One demonstration is not an aggregate benchmark, and even a policy that meets a validation gate can collide on another episode.

Evaluate the oxygen model without a window:

```sh
python -m astronaut_rl.play models/oxygen_seed11.json.gz --headless --seed 9600000 --episodes 100
```

## Train and reproduce

```sh
python -m astronaut_rl.play --train
```

The browser uses the frozen training configuration and automatically watches its selected checkpoint when ready. New runs are stored under `runs/` and preserve the supplied submission artifacts. The stage initializer and checkpoint metadata identify any inherited policy; training an advanced stage is not described as an empty-table experiment.

To reproduce the full five-chain experiment in separate directories:

```sh
python -m astronaut_rl.batch --out rerun_results --models rerun_models --work rerun_jobs --workers 5
python -m astronaut_rl.verify --results rerun_results --models rerun_models
python -m astronaut_rl.report --results rerun_results --out rerun_RESULTS.md
python -m astronaut_rl.plots --results rerun_results --out rerun_results/figures
python -m astronaut_rl.poster --results rerun_results --out rerun_poster.pdf
```

The submitted batch runs five chains in parallel; each chain runs its three stages sequentially. Timings are active wall-clock seconds per process under this concurrent load. Use `python -m astronaut_rl.experiment --variant all --out sequential_results --models sequential_models` for a sequential rerun with the same learned policies and different timings.

Each chain uses its own seed (`11,23,37,51,79`) and trains flight, asteroids and oxygen in that order. Flight starts at zero; later stages copy learned values from their selected parent policy across the new object's state suffix. They then perform ordinary Q-learning in the harder environment. All movement remains learned; there are no scripted avoidance or tank-seeking actions.

Allocated budgets are 1,000,000 Q decisions for flight, 1,500,000 for asteroids and 2,000,000 for oxygen. A decision holds an action for up to six actual physics frames, stopping on collision or truncation. The report includes executed physics steps and both incremental and cumulative training time. These are different stage budgets and include transferred learning, so the timing graph is not an equal-budget, independent-from-scratch task comparison.

Checkpoints use validation seeds `9590000..9590049` for wrapped tasks, with one deterministic flight episode. Final testing uses the separate seeds `9600000..9600099` (one episode for flight). Selection uses validation only. A failed promotion gate is recorded; it does not become a success merely because loss is small.

The course rewards and physics remain unchanged: collision `-1`, oxygen pickup `+1` without collision, otherwise `0`. For six-frame actions, rewards and bootstraps are discounted by the actual number of executed physics frames. See `METHODS.md` for the equations and exact settings.

## Verify the submission

```sh
python -m unittest discover -s tests -v
python -m astronaut_rl.verify --results results --models models
python -m astronaut_rl.compare_recovery
```

Tests check algorithm and artifact consistency. Verification recomputes the recorded evaluations and hashes. Neither replaces the behavioral evidence in the survival, completion and oxygen tables. See `VALIDATION.md` for checks and `ASSESSMENT_RESPONSE.md` for the earlier regression and the recovery design.

The comparison command evaluates the retained pre-recovery policies and the recovered policies on identical official-environment episode seeds. The measured before/after table is in `results/recovery_comparison/README.md` and paired outcomes are saved in that folder; it measures the entire recovery pipeline.

The figures are `loss.svg`, `survival.svg`, `completion.svg`, `oxygen.svg`, `coverage.svg`, `performance.svg` and `training_time.svg`. Curves use Q decisions on the horizontal axis and means with sample SD across five independently trained chains. Asteroid avoidance has a longer training phase. Final-test episodes are averaged per policy before aggregating across chains; they are not additional training replicates.

## Files

- `astronaut_rl/qlearning.py`: original Q table, exploration, Bellman update and serialization.
- `astronaut_rl/environment.py`, `state.py`: unchanged course environment and public-observation features.
- `configurations/`: frozen thresholds, budgets, discounts, seeds, initialization and selection rules.
- `models/`: selected, final-budget and checkpoint tables, with their exact specifications.
- `results/`: loss blocks, episodes, validation histories, final tests and summaries.
- `vendor/PROVENANCE.json`: official commit, wheel hash and installed-source checks.
- `METHODS.md`, `RESULTS.md`, `PRESENTATION.md`: protocol, measured findings and presentation outline.

The assignment's isolated mention of “Mill” is treated as a typo: this project implements Astronaut. Both students submit the same ZIP, PDF and Markdown file individually, one day before the presentation. Attendance and the joint presentation remain required.
