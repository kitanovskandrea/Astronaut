# Autonomous replay and training

Python executes every physics frame in the unchanged course simulator. The learned Q table chooses thrust or coast, normally holding it for six frames. The browser receives per-frame snapshots and animates the official sprites through a canvas with a short interpolation buffer. Display interpolation does not choose actions or implement another physics model. Latest-scene delivery prevents a slow browser from accumulating old frames.

The page measures draw and incoming-scene rates. A foreground tab on an unloaded machine should approach the simulator's 60 Hz; machine load and background-tab scheduling can reduce the display rate. End counters use the exact final scene. Object identities persist when a wrapper moves a sprite between its front and back slots.

There are no player movement controls. **Watch trained AI** uses the saved policy without changing its Q table. **Train from scratch** starts a new training job under the frozen stage configuration in a separate process, preserves supplied artifacts under `models/` and `results/`, and writes its new output under `runs/`. Advanced-stage initialization is declared in the run metadata: the recovery curriculum copies a learned parent policy before learning the new object's behavior.

```sh
python -m astronaut_rl.play models/fly_seed11.json.gz --seed 9590000
python -m astronaut_rl.play models/asteroids_seed11.json.gz --seed 9590000
python -m astronaut_rl.play models/oxygen_seed11.json.gz --seed 9590000
```

An earlier assessment revision changed action timing, state resolution, rewards, package physics and training exposure together. Its artifact tests passed, but its asteroid/oxygen behavior regressed. The recovery learns flight first, retains the full avoidance state when adding oxygen, and requires consecutive survival checks before promoting a checkpoint. Oxygen count is considered only after the survival requirement is satisfied. `ASSESSMENT_RESPONSE.md` records the changes; `RESULTS.md` provides the actual measured outcomes rather than assuming the new design succeeds.

Current primary models use the pinned updated official package with unchanged default rewards. The earlier custom-reward project remains separate; its models require its original state representation and course snapshot. The two upstream snapshots both call themselves version 0.2, so the program checks source hashes. Follow the explicit wheel reinstall command in `README.md` if an older installation is rejected.
