# Paired recovery comparison

Both policy sets use the same pinned official physics, default rewards and final episode seeds. Values below are means of five policy means. The JSON also records sample SD across policies. This compares the complete recovery pipeline, including different state/action settings, training exposure and skill transfer; it does not isolate the effect of one change.

| Task | Before survival | After survival | Before tanks | After tanks | Before frames | After frames |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| fly | 100.0% | 100.0% | 0.00 | 0.00 | 1200.0 | 1200.0 |
| asteroids | 30.6% | 98.6% | 0.00 | 0.00 | 767.8 | 1193.0 |
| oxygen | 6.4% | 96.8% | 2.17 | 5.32 | 463.3 | 1185.0 |

Each row of `paired_episodes.csv` matches the same training-seed identifier and episode seed. Both policies are frozen during evaluation. Historical custom-reward/old-physics models are excluded.
