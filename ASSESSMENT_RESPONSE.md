# Assessment follow-up and behavioral regression

The 126-page assessment is an exported planning conversation. Its consolidated plan on pages 97-126 contains useful methodology and explicitly describes its compact bins, hyperparameters, budgets and numerical targets as development hypotheses. Implementing those suggestions did not establish that the resulting agent played better.

## What went wrong

The assessment revision changed several things at once and was promoted after consistency checks despite weaker asteroid and oxygen behavior. The Q-learning implementation itself was unchanged. Tests verified its arithmetic, saved artifacts and reproducibility; they did not prove avoidance or collection quality. The delivery needed a behavioral acceptance gate before replacing the working demonstration.

| Effective condition | Earlier working revision | Assessment revision that regressed |
| --- | --- | --- |
| Official source | `5fa9ad93b094...` | `0db0652ad3b8...` |
| Net upward acceleration under thrust | +1000 pixels/s² in that source | +600 pixels/s² in the updated official source |
| Action interval | Up to 6 physics frames | 1 physics frame |
| Reward | Custom survival/collision reward; oxygen used pickup/miss/potential feedback | Official -1 collision, +1 pickup, 0 otherwise |
| State capacity: flight / asteroids / oxygen | 132 / 3,960 / 14,625 | 30 / 720 / 1,800 |
| Budget per stage | 1 million Q decisions, about 5.95 million physics frames | 2 million Q updates/physics frames |
| Discount | 0.99 per six-frame decision | 0.995 per frame |
| Approximate physical 1/e discount horizon | 597 frames (9.95 seconds) | 199.5 frames (3.33 seconds) |
| Stage initialization | Empty table per stage | Empty table per stage |
| Oxygen checkpoint selection | Final table | Highest validation tank mean first; survival only broke ties |
| Deployment behavior gate | None | Targets were diagnostics and did not block promotion |

The source update was genuine upstream behavior, not a package patch. Both sources used version 0.2. Moving to official rewards and the updated package was compatible with the methodology goal, but required retraining and verifying the actual behavior. Doubling the Q-update count while reducing action repeat to one also reduced physical experience to about one third. Coarser motion states and a shorter physical discount horizon changed the learning problem further. Without controlled ablations, no single change can be blamed for the whole regression.

## What the stored measurements showed

These are historical measurements, not the recovery experiment. They use different source snapshots, rewards, training seeds and test scenarios; the columns are not a controlled estimate of one algorithmic effect.

| Stage | Earlier revision: 3 policies | Regressed revision: 5 policies |
| --- | --- | --- |
| Flight | 100% completion; 1,200 frames | 100% completion; 1,200 frames |
| Asteroids | 69.33% completion; 1,047.69 frames | 36.40% completion; 816.48 frames |
| Oxygen | 41.67% completion; 4.353 tanks; 835.35 frames | 3.20% completion; 2.052 tanks; 441.19 frames |

Earlier records used seeds 11/22/33 and test scenarios `9001000..9001099`. The regressed records used seeds 11/23/37/51/79 and test scenarios `2000..2049`, with one deterministic flight episode. These are means of per-policy metrics. The regressed asteroid and oxygen stages reached their recorded validation targets in zero of five runs. Reporting those weak results honestly did not make them adequate for the user's behavioral goal.

## Recovery design and the assessment tradeoff

The recovery keeps the current unchanged official physics/rewards and original tabular Bellman update. It changes the agent and experimental controller:

- Learn flight first, initialize asteroid Q values from that learned policy, then initialize oxygen Q values from the selected learned avoidance policy.
- Keep the full hazard-aware state prefix while adding oxygen; tank proximity cannot remove asteroid information.
- Use public motion-projected object offsets, a six-frame action interval and consistent physical-frame discounting.
- Evaluate initialization at checkpoint zero and retain it as a learned baseline.
- Require two consecutive validation passes before promotion. For oxygen, rank collection only after its survival constraint is met. A risky high-pickup checkpoint cannot override that constraint.
- Preserve separate final testing, five independently seeded chains, all checkpoint records, real update counts and active training timers.

This intentionally replaces the assessment's independent-from-zero stage suggestion with a learned curriculum. It is still classical tabular Q-learning: parent action values are copied across new state suffixes, then updated through real transitions. No scripted thrust/avoidance controller or extra reward is introduced. The consequence is that stages within a chain are dependent and later-stage cost includes prerequisite learning. The report must show incremental and cumulative cost and must not claim an equal-budget from-zero comparison.

The frozen recovery allocates one million flight Q decisions, 1.5 million asteroid decisions and two million oxygen decisions. Physical-frame totals are recorded separately. Validation uses `9590000..9590049` for stochastic tasks; final tests use the untouched `9600000..9600099` set. These details supersede the earlier two-million-one-frame-step protocol.

Pilot outcomes motivate the design but are not the submitted result. `RESULTS.md` is generated only from the completed recovery records. A failed gate is recorded as a failure and blocks advancing the chain. Neither passing tests, a low TD residual nor the existence of a PDF poster is evidence of successful behavior.

## Assessment points retained

Official source integrity, default rewards, exact saved state specifications, separate validation/testing, deterministic frozen evaluation, pre-update TD loss, time-limit bootstrapping, visit diagnostics, five-seed statistical aggregation and reproducible artifacts remain part of the implementation. Suggested bin counts and numerical success criteria are not treated as professor-imposed pass conditions. The reported evidence must match the actual models and must support the claimed flight, avoidance and collection behavior.

## Verified recovery result

The matched official-environment comparison now measures asteroid completion at 30.6% before versus 98.6% after, and oxygen completion at 6.4% versus 96.8%. Oxygen tanks increase from 2.17 to 5.32 per episode. These figures use identical final scenario seeds and five frozen policies per task; they differ from the historical figures above because those used their original scenario sets. Full paired records are in `results/recovery_comparison/`.

All five chains qualified at every stage. Validation had exposed two unfinished flight policies and a collection policy that did not improve on its inherited incidental pickups. Flight now uses a smaller update rate and sustained exploration; collection has a longer exploration schedule and training allocation. The gates were retained. The final holdout was not used to choose those changes.

Flight and avoidance TD residuals decrease strongly; oxygen residuals remain nonzero and rise later. The recovery establishes better measured behavior, not a claim of zero-loss convergence or an optimal oxygen policy.
