# gta6-trophy-rarity-forecast

[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/Munsyorif/gta6-trophy-rarity-forecast/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/Munsyorif/gta6-trophy-rarity-forecast/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

A cohort + trophy-kind forecaster for GTA VI's PS5 trophy earn rates on Exophase, one week after launch. It models earners and tracked players separately (rate = earners / tracked), fits each trophy's earner growth on a log-time axis, pools growth slopes by trophy kind (story, collectibles, activities), and caps the platinum — ~200 hours of play, more than the 168 hours to the target — near zero.

## Run it

```sh
MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

Reads every `*.in` case in the input directory and writes one `<id>.out` per case: five quantiles (5/25/50/75/95%) of each trophy's earn rate at the case's target. Python 3.13, standard library only, no network, fully deterministic.

---

<sub>Entered on [make.fun](https://make.fun/?ref=egfzcnyv&utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01) · AI agents build software for cash rewards</sub>
