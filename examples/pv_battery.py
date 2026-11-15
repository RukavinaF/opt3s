"""A PV plant, a battery and a grid connection that supply an office load.

The grid connection, the battery and the bus follow 15-minute prices, while the
PV plant and the load are modelled with hourly samples.  The sizes of the PV
plant and the battery are optimized jointly with their operation over one week,
whose costs are scaled up to one year.  Run from the repository root::

    python examples/pv_battery.py
"""

from datetime import datetime

import numpy as np
import pandas as pd

from opt3s import (
    FC, UC, Costs, Horizon, Hub, HubResult, Other, Profile, Size, Standard, State, build, solve,
)  # fmt: skip

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
)  # fmt: skip
office = Standard("LOAD", "sink", {"u1": UC(E, load)}, t=1)
battery = Standard(
    "BATT", "storage", {"u1": FC(E), "y1": FC(E)},
    states={"x1": State({"u1": 0.95, "y1": 1 / 0.95}, gamma=0.1)},  # kWh, 10 % minimum
    size=Size.max_of_state("x1"), costs=Costs(c_inv=300, c_mnt=5), t=0.25,
)  # fmt: skip
bus = Standard(
    "BUS", "bus",
    {"u1": Other("GRID.y1"), "u2": Other("PV.y1"), "u3": Other("BATT.y1"),
     "y1": Other("LOAD.u1"), "y2": Other("BATT.u1"), "y3": Other("EXP.u1")},
    t=0.25,
)  # fmt: skip

hub = Hub("example", [grid, export, pv, office, battery, bus])
built = build(hub, Horizon(start, days), n_dl=20, r=0.03)  # 20 years, 3 % discount rate
solve(built, solver="highs", options={"output_flag": False})

result = HubResult(built)
print({k: round(v, 1) for k, v in result.economics().items()})
print(result.sizes().round(1))
print(result.amounts().round(1))
