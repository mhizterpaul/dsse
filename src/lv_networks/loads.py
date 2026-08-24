import numpy as np

def distribute_loads(buses: list, rng=None) -> dict:
    """
    Distributes consumer loads of different classes across known LV buses.
    Uses local seeded RNG for perfect reproducibility.
    """
    if rng is None:
        rng = np.random.default_rng(42)

    loads = []
    capacitors = []
    motors = []
    ders = []

    for bus in buses[1:]:
        if rng.random() < 0.6:
            load_kw = rng.uniform(5.0, 25.0)
            l_model = int(rng.choice([1, 2, 3]))
            pf = float(rng.choice([0.85, 0.90, 0.95]))
            loads.append({
                "name": f"l_{bus}",
                "bus": bus,
                "kw": round(load_kw, 2),
                "pf": pf,
                "model": l_model
            })

        if rng.random() < 0.12:
            cap_kvar = float(rng.choice([15.0, 30.0, 45.0]))
            capacitors.append({
                "name": f"c_{bus}",
                "bus": bus,
                "kvar": cap_kvar
            })

        if rng.random() < 0.08:
            motors.append({
                "name": f"m_{bus}",
                "bus": bus,
                "kw": round(float(rng.uniform(10.0, 30.0)), 1),
                "pf": 0.8
            })

        if rng.random() < 0.05:
            ders.append({
                "name": f"der_{bus}",
                "bus": bus,
                "kw": round(float(rng.uniform(5.0, 20.0)), 1)
            })

    return {
        "loads": loads,
        "capacitors": capacitors,
        "motors": motors,
        "ders": ders
    }
