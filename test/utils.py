from functools import lru_cache
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import numpy as np

# save all data from the simulation in temp and load it, if it exists
SAVE_IN_TEMP = True
CONFIG = {
    "trunc": 1000,
    "dt": 0.002,
    "kbt": 2.494,
    "mass": 31.4,
    "steps": int(5e7),
    "stride": 1,
    "force_type": "harmonic",
    "integrator": "RungeKutta4",
    "gammas": [100.0, 200.0],
    "ks": [100.0, 1000.0],
    "k": 120.0,
}


@dataclass
class TestData:
    """
    Container for the correlations needed for the tests.
    """

    vv_corr: np.ndarray
    xdu_corr: np.ndarray
    dt: float
    mass: float
    trunc: int
    traj: Optional[np.ndarray] = None
    gammas: Optional[np.ndarray] = None
    ks: Optional[np.ndarray] = None

    @classmethod
    def load(cls):
        """Load test data, assume we are either in root or in test dir."""
        load_dir = Path(".").resolve()
        if load_dir.name != "test":
            load_dir = load_dir / "test"
        load_dir = load_dir / "data"
        corrs_to_load = ["vv_corr", "xdu_corr"]
        corrs = {corr: np.load(load_dir / f"{corr}.npy").astype(float) for corr in corrs_to_load}
        mass = 2.494 / corrs["vv_corr"][0]
        return cls(**corrs, dt=0.01, mass=mass, trunc=10)

    @classmethod
    def simulate(cls):
        """Run GLE simulation and compute correlations."""
        corrs_to_compute = ["vv_corr", "xdu_corr"]
        temp_files = [Path(f"/tmp/{corr}.npy") for corr in corrs_to_compute]
        if all(temp_file.exists() for temp_file in temp_files):
            corrs = {
                corr: np.load(temp_file) for corr, temp_file in zip(corrs_to_compute, temp_files)
            }
            traj = None
        else:
            corrs = {}
            import gle_sim
            import correlation

            config = gle_sim.check_config(CONFIG)
            traj = gle_sim.integrate(config)
            vel = np.gradient(traj, config["dt"])
            du = gle_sim.du_dx(traj, config)
            corrs["vv_corr"] = correlation.correlation(vel, vel, CONFIG["trunc"])
            corrs["xdu_corr"] = correlation.correlation(traj, du, CONFIG["trunc"])
            if SAVE_IN_TEMP:
                for key, item in corrs.items():
                    to_save = f"/tmp/{key}.npy"
                    assert to_save in [str(t) for t in temp_files]
                    np.save(to_save, item)
        return cls(
            **corrs,
            traj=traj,
            dt=CONFIG["dt"],
            mass=CONFIG["mass"],
            trunc=CONFIG["trunc"],
            gammas=CONFIG["gammas"],
            ks=CONFIG["ks"],
        )


@lru_cache()
def load_test_data() -> TestData:
    """Only load the data once for all tests."""
    return TestData.load()


@lru_cache()
def simulate_test_data() -> TestData:
    """Only simulate the data once for all tests."""
    return TestData.simulate()
