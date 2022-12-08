from functools import lru_cache
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import numpy as np

# save all data from the simulation in temp and load it, if it exists
CACHECORRS = True
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
CORRS = ["vv_corr", "va_corr", "aa_corr", "xdu_corr", "vdu_corr", "adu_corr"]


def get_test_dir() -> Path:
    """ Make test executable from root or test dir. """
    test_dir = Path(".").resolve()
    if test_dir.name != "test":
        test_dir = test_dir / "test"
    return test_dir


@dataclass
class TestData:
    """
    Container for the correlations needed for the tests.
    """

    vv_corr: np.ndarray
    va_corr: np.ndarray
    aa_corr: np.ndarray
    xdu_corr: np.ndarray
    vdu_corr: np.ndarray
    adu_corr: np.ndarray
    dt: float
    mass: float
    trunc: int
    traj: Optional[np.ndarray] = None
    gammas: Optional[np.ndarray] = None
    ks: Optional[np.ndarray] = None

    @classmethod
    def load(cls):
        """Load test data, assume we are either in root or in test dir."""
        load_dir = get_test_dir() / "data"
        corrs_to_load = [load_dir / f"{corr}.npy" for corr in CORRS]
        if not all(c.exists() for c in corrs_to_load):
            TestData.simulate()
        corrs = {corr: np.load(load_dir / f"{corr}.npy").astype(float) for corr in CORRS}
        mass = 2.494 / corrs["vv_corr"][0]
        return cls(**corrs, dt=0.01, mass=mass, trunc=10)

    @classmethod
    def simulate(cls):
        """Run GLE simulation and compute correlations."""
        (get_test_dir() / "data").mkdir(exist_ok=True)
        temp_files = [get_test_dir() / f"data/{corr}.npy" for corr in CORRS]
        if all(temp_file.exists() for temp_file in temp_files):
            corrs = {corr: np.load(temp_file) for corr, temp_file in zip(CORRS, temp_files)}
            traj = None
        else:
            corrs = {}
            import gle_sim
            import correlation

            config = gle_sim.check_config(CONFIG)
            traj = gle_sim.integrate(config)
            vel = np.gradient(traj, config["dt"])
            acc = np.gradient(vel, config["dt"])
            du = gle_sim.du_dx(traj, config)
            corrs["vv_corr"] = correlation.correlation(vel, vel, CONFIG["trunc"])
            corrs["va_corr"] = correlation.correlation(vel, acc, CONFIG["trunc"])
            corrs["aa_corr"] = correlation.correlation(acc, acc, CONFIG["trunc"])
            corrs["xdu_corr"] = correlation.correlation(traj, du, CONFIG["trunc"])
            corrs["vdu_corr"] = correlation.correlation(vel, du, CONFIG["trunc"])
            corrs["adu_corr"] = correlation.correlation(acc, du, CONFIG["trunc"])
            if CACHECORRS:
                for key, item in corrs.items():
                    to_save = get_test_dir() / f"data/{key}.npy"
                    assert (
                        to_save.name.replace(".npy", "") in CORRS
                    ), f"please add {to_save=} to CORRS list"
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
