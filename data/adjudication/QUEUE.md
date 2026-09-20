# Adjudication queue - control_masked_sequence_sonnet.csv

6 blocks where the agent and the February 2018 survey disagree, highest
leverage first. `weight` is N_stratum / n_sampled: judging one block moves the
Horvitz-Thompson population estimate by that many blocks.

**Record verdicts in `data/adjudication/control_sample.csv`'s `verdict` column**, using
the four labels in `RULE.md`. Not here, and never in a `runs/*.csv` -- those are
standalone agent output and stay that way.

Images are under `control_imagery/`, which is this sheet's own directory and is NOT the
worksheet's `imagery/`. Both are gitignored and regenerable:
`pixi run python -m scripts.fetch_block_imagery --control`

| weight | block | agent says | survey says | ha | place | satellite |
|---|---|---|---|---|---|---|
| 146.5 | `ZAF.9.3.1_1_6289` | no-dense-informal | informal | 0.38 | Cape Town Ward 6 | [maps](https://www.google.com/maps/@-33.86055,18.74173,19.5z/data=!3m1!1e3) |
| 146.5 | `ZAF.9.3.1_1_5706` | some-dense-informal | formal | 18.40 | Kraaifontein East | [maps](https://www.google.com/maps/@-33.84913,18.74507,16.1z/data=!3m1!1e3) |
| 146.5 | `ZAF.9.3.1_1_55731` | some-dense-informal | formal | 1.79 | Cape Town Ward 67 | [maps](https://www.google.com/maps/@-34.08320,18.48754,19.2z/data=!3m1!1e3) |
| 146.5 | `ZAF.9.3.1_1_20496` | all-dense-informal | formal | 0.62 | Umrhabulo Triangle (Makhaza) | [maps](https://www.google.com/maps/@-34.03965,18.69655,19.5z/data=!3m1!1e3) |
| 146.5 | `ZAF.9.3.1_1_18184` | some-dense-informal | formal | 1.64 | The Hague | [maps](https://www.google.com/maps/@-33.95464,18.63609,19.1z/data=!3m1!1e3) |
| 146.5 | `ZAF.9.1.2_1_3710` | all-dense-informal | formal | 1.22 | Drakenstein Ward 5 | [maps](https://www.google.com/maps/@-33.66604,18.98731,18.7z/data=!3m1!1e3) |

Image paths:
```
data/adjudication/control_imagery/satellite/ZAF.9.3.1_1_6289.png
data/adjudication/control_imagery/satellite/ZAF.9.3.1_1_5706.png
data/adjudication/control_imagery/satellite/ZAF.9.3.1_1_55731.png
data/adjudication/control_imagery/satellite/ZAF.9.3.1_1_20496.png
data/adjudication/control_imagery/satellite/ZAF.9.3.1_1_18184.png
data/adjudication/control_imagery/satellite/ZAF.9.1.2_1_3710.png
```
