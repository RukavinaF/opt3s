# opt3s

**opt3s** is a Python package for the joint sizing and scheduling of energy hubs
with linear programming. It works with any energy system and lets every
component use its own sampling time.

Every component is described only by its ports (inputs and outputs), states,
size, and cost terms. From that description, opt3s builds a linear program (LP)
with [linopy](https://linopy.readthedocs.io) that maximizes the net present value
(NPV) of the hub over its design lifetime. The main features are:

- **component-agnostic modelling**: sources, sinks, processes, storages, and
  buses of any energy vector (electricity, heat, hydrogen, water, ...), which
  can be combined into custom components;
- **component-specific sampling times**: e.g. 0.25 h for the battery and 4 h
  for an electrolyser in the same LP, with the amounts exchanged between
  components preserved;
- **enlargement of existing components**: an existing size `S_exist` and an
  upper limit `S_max` per component;
- **a full-year horizon**, so seasonal storage is sized together with daily
  operation.

The method is described in the article

> F. Rukavina and M. Vašak, *A component-agnostic framework for joint sizing and
> scheduling of energy hubs with storage and heterogeneous sampling times*.

This repository also contains the case study of the article (a fleet depot with
photovoltaics, electrolysis, a fuel cell, waste heat usage, and battery,
hydrogen, and oxygen storage), its input data, and the records of all
optimization runs reported in the article, from which all its result tables
can be regenerated.

## Installation

Python 3.10 or newer is needed. The LPs are solved with the open-source solver
[HiGHS](https://highs.dev), which is installed with the package; Gurobi can be
used as well if it is installed and licensed.

To use the package in your own project:

```bash
pip install git+https://github.com/RukavinaF/opt3s.git
```

To run the case study or the tests, clone the repository and install the exact
package versions that were used for the article:

```bash
git clone https://github.com/RukavinaF/opt3s.git
cd opt3s
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install -e .
```

All commands below are run from the repository root.

## Quick start

A PV plant, a battery, and a grid connection supply an office load. The grid
connection, the battery, and the bus use a sampling time of 0.25 h, while the
PV plant and the load use 1 h. The PV plant (20 kWp existing, at most 150 kWp)
and the battery are sized jointly with their operation over one week, whose
costs are scaled up to one year
([examples/pv_battery.py](examples/pv_battery.py)):

```python
from datetime import datetime

import numpy as np
import pandas as pd

from opt3s import (
    FC, UC, Costs, Horizon, Hub, HubResult, Other, Profile, Size, Standard, State, build, solve,
)

E = "electricity"
start, days = datetime(2025, 6, 2), 7

# input profiles in calendar time; opt3s resamples them to each sampling time
hours = pd.date_range(start, periods=24 * days, freq="1h")
quarters = pd.date_range(start, periods=96 * days, freq="15min")
h = hours.hour.to_numpy()
pv_kwp = pd.Series(np.clip(np.sin((h - 6) / 12 * np.pi), 0, None), index=hours)  # kW/kWp
load = pd.Series(50.0 + 30.0 * ((h >= 8) & (h < 18)), index=hours)  # kW
q = (quarters.hour + quarters.minute / 60).to_numpy()
price = pd.Series(0.10 + 0.08 * np.cos((q - 19) / 24 * 2 * np.pi), index=quarters)  # EUR/kWh

grid = Standard("GRID", "source", {"y1": FC(E, c_opr=price, p_max=200)}, t=0.25)
export = Standard("EXP", "sink", {"u1": FC(E, c_rev=0.5 * price)}, t=0.25)
pv = Standard(
    "PV", "source", {"y1": Profile(E, pv_kwp, bounded=True)},  # curtailment allowed
    size=Size.by_profile(s_exist=20, s_max=150),  # 20 kWp exist, enlarge up to 150 kWp
    costs=Costs(c_inv=700, c_mnt=10), t=1,
)
office = Standard("LOAD", "sink", {"u1": UC(E, load)}, t=1)
battery = Standard(
    "BATT", "storage", {"u1": FC(E), "y1": FC(E)},
    states={"x1": State({"u1": 0.95, "y1": 1 / 0.95}, gamma=0.1)},  # kWh, 10 % minimum
    size=Size.max_of_state("x1"), costs=Costs(c_inv=300, c_mnt=5), t=0.25,
)
bus = Standard(
    "BUS", "bus",
    {"u1": Other("GRID.y1"), "u2": Other("PV.y1"), "u3": Other("BATT.y1"),
     "y1": Other("LOAD.u1"), "y2": Other("BATT.u1"), "y3": Other("EXP.u1")},
    t=0.25,
)

hub = Hub("example", [grid, export, pv, office, battery, bus])
built = build(hub, Horizon(start, days), n_dl=20, r=0.03)  # 20 years, 3 % discount rate
solve(built, solver="highs", options={"output_flag": False})

result = HubResult(built)
print(result.economics())  # CAPEX, OPEX, REV, annual profit, NPV
print(result.sizes())      # S_exist, S_max, S_new, S_total per component
print(result.amounts())    # annual amount through every port
```

`result.timeseries()` returns the power of every port on a common time index,
and `result.costs()` the cost terms of every component.

### Building blocks

| Object | Meaning |
|---|---|
| `Standard(name, kind, ports, ...)` | a component of kind `"source"`, `"sink"`, `"process"`, `"storage"`, or `"bus"` |
| `Custom(name, components, ports=...)` | a component composed of other components, with optional size ratios (`Ratio`) |
| `Hub(name, components)` | the energy hub, i.e., the top-level custom component |
| `FC`, `UC` | port that is a free decision variable, or a given profile (uncontrollable) |
| `LinComb`, `Profile`, `Amount`, `Other` | port defined (or, with `bounded=True`, bounded from above) by a linear combination of other ports of the component, by a profile scaled with the size, by amounts over time windows, or by a port of another component |
| `State(alpha, sigma, gamma)` | a storage state with port coefficients, efficiency, and minimum relative to the size |
| `Size`, `Costs`, `PeakBilling` | sizing rule with `S_exist` and `S_max`; investment, maintenance, replacement, and degradation costs; billing of peak power |
| `Horizon(start, n_days)`, `build`, `solve`, `HubResult` | horizon, LP formulation, solving, and evaluation of the solution |

Port names are `u1, u2, ...` for inputs and `y1, y2, ...` for outputs, and
every port may carry an upper limit `p_max`, an operating price `c_opr`, and a
revenue `c_rev` (constants or time series). The docstrings of `opt3s/ports.py`,
`opt3s/specs.py`, and `opt3s/components.py` describe all options, and
`tests/test_opt3s.py` contains further small examples.

## Case study of the article

The package `depot` defines the fleet depot energy hub of the article using
only the opt3s API:

| File | Content |
|---|---|
| `depot/params.py` | all parameters of the components (Supplementary Tables S1–S4) |
| `depot/inputs.py` | input data files → time series and time windows |
| `depot/components.py`, `depot/hub.py` | the components and the hub |
| `depot/scenarios.py` | scenario sets R1–R4 of the article |
| `depot/runner.py` | one run: formulation, solving, and post-processing, each measured |
| `depot/net_grid.py` | the signed net grid power of scenarios R4 |
| `depot/studies/r1.py` ... `r4.py` | running the scenario sets and writing the result tables |

The modules `scenario.py`, `results.py`, and `benchmark.py` define a scenario,
store a run, and measure it (time, memory, LP size, solver statistics).

### Regenerating the tables of the article from the stored runs

```bash
python -m depot.studies.r1 tables
python -m depot.studies.r2 tables
python -m depot.studies.r3 tables
python -m depot.studies.r4 tables
```

These commands read the stored runs in `results/` and write the tables of the
article and its Supplementary Information to `tables/` (LaTeX fragments and one
CSV file per scenario set with all numbers and the run IDs used). The files in
`tables/` are the ones used in the article, so `git diff tables` shows that they
are reproduced. `r3 tables` reports four missing R3a scenarios; these are
comparisons of the heating system balance that are not part of the article.

### Solving the scenarios again

```bash
python -m depot list                       # scenario sets and installed solvers
python -m depot run R1-4h --days 7 --in-process --no-solution --store results_test   # smoke test, < 1 min
python -m depot describe                   # Supplementary Tables S5 and S6 from the code
python -m depot.studies.r1 run --store results_new
python -m depot.studies.r1 tables --store results_new --out new
```

Each scenario is solved in its own process, so that the peak memory is
measured per run. The full-year runs of the article took 38 h in total on one
computer (Intel processor with 12 logical cores, Windows 11): from 18 s (R1-6h)
to 11 h (R1-0.25h), with a peak memory usage of at most 2.6 GB. Use a separate `--store` for new runs: a new run
of a scenario stored in `results/` counts as a repeat of it, so the tables
would show its values and the median of the solving times of all repeats.
The other commands of the studies (e.g., `r3 run --part da`, `r4 run`,
`r4 presolve`) are listed in the docstrings of `depot/studies/r1.py` ... `r4.py`.

## Stored results

`results/` holds the 37 runs whose results are reported in the article, and
`results_check/` the five repeated runs of R2 scenarios that the Supplementary
Information uses to quantify the variation of the solving time. Each run has
its own folder, named `<date>-<time>_<scenario>_<fingerprint>`, with

| File | Content |
|---|---|
| `run.json` | scenario, objective, LP size, solver statistics, timings, peak memory, environment, and package versions |
| `parameters.json` | parameters of all components |
| `sizes.csv`, `costs.csv`, `amounts.csv` | optimal sizes, cost terms per component, and annual amounts through every port |

`index.csv` lists all runs of a folder in one table. The time series of the
optimal solutions (`solution.nc`, 2.5 MB on average) are not included; they
are written again when a scenario is solved. The host name of the computer was
removed from the records, and their git commit refers to the development
repository of the article. All runs were solved with HiGHS 1.15.1 (dual simplex)
through linopy 0.8.0 on Windows 11 with Python 3.13.

```python
from results import ResultStore

store = ResultStore("results")
store.index()                                   # all runs, one row each
run = store.load("20261103-171537_R1-1h_9b602d18fe")
run["sizes"], run["costs"], run["amounts"], run["objective"]
```

## Input data

The input files in `data/` are processed versions of public datasets: one
value per line, one year, as read by `depot/inputs.py`.

| File | Content | Source |
|---|---|---|
| `HR_DA_PRICES_2025.csv` | day-ahead electricity prices for Croatia, 2025 [EUR/kWh], 15 min (prices vary within the hour only from 1 October 2025) | [ENTSO-E Transparency Platform](https://transparency.entsoe.eu/) |
| `PVGIS_..._-90deg_...csv`, `..._90deg_...`, `..._0deg_...` | production of a 1 kWp crystalline silicon array at 45.801° N, 15.971° E, 10° tilt, facing east, west, and south, 2023 [kW], 1 h | [PVGIS](https://joint-research-centre.ec.europa.eu/photovoltaic-geographical-information-system-pvgis_en), © European Union |
| `rainfall-45.80N15.92E131m-2025.csv` | rainfall at 45.80° N, 15.92° E, 2025 [mm/h], 1 h | [Open-Meteo](https://open-meteo.com/), [doi:10.5281/zenodo.7970649](https://doi.org/10.5281/zenodo.7970649), CC BY 4.0 |
| `LargeOffice.csv`, `LargeOffice_heat.csv` | electricity and heat consumption of the large office [kW], 1 h; scaled to the annual energy in `depot/params.py` | US DOE commercial reference buildings, [doi:10.2172/1009264](https://doi.org/10.2172/1009264) |
| `NREL_Charging_Data.csv` | EV charging sessions (station, start, end, energy [kWh]), 2020 | NREL workplace charging data, [doi:10.7799/1838140](https://doi.org/10.7799/1838140), CC BY 4.0 |

The two-tariff electricity prices of HEP Elektra and the hydrogen and oxygen
demands are given in `depot/params.py`. The data remain subject to the terms
of their providers.

## Tests

```bash
python -m unittest discover tests
```

`tests/test_opt3s.py` checks the framework on small hubs with results computed
by hand, and `tests/test_depot.py` checks the case study definitions.

## License

Copyright (C) 2026 Filip Rukavina, Mario Vašak

This program is free software: you can redistribute it and/or modify it under
the terms of the GNU General Public License as published by the Free Software
Foundation, either version 3 of the License, or (at your option) any later
version. This program is distributed in the hope that it will be useful, but
WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
FITNESS FOR A PARTICULAR PURPOSE. See [LICENSE](LICENSE) for the full text.

The input data in `data/` are not covered by this license; see
[Input data](#input-data).
