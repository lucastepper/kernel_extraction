from functools import lru_cache
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import numpy as np

CONFIG = {
    "trunc": 1000,
    "dt": 0.002,
    "kbt": 2.494,
    "mass": 31.4,
    "steps": int(5e6),
    "stride": 1,
    "force_type": "harmonic",
    "integrator": "RungeKutta4",
    "gammas": [100.0, 200.0],
    "ks": [100.0, 1000.0],
    "k": 120.0,
    "bins": 100,
}
CORRS = ["vv_corr", "va_corr", "aa_corr", "xdu_corr", "vdu_corr", "adu_corr"]


def get_test_dir() -> Path:
    """Make test executable from root or test dir."""
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
    kbt: Optional[float] = None
    bins: Optional[int] = None

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
        corrs = {}
        import gle_sim
        import correlation

        if CONFIG["force_type"] == "harmonic":
            pmf = gle_sim.Harmonic(k=CONFIG["k"])
        else:
            raise NotImplementedError(f"force type {CONFIG['force_type']} not implemented")
        taus = np.array(CONFIG["gammas"]) / np.array(CONFIG["ks"])
        kernel = gle_sim.Exponential(gammas=CONFIG["gammas"], taus=taus)
        config = gle_sim.get_config(
            dt=CONFIG["dt"],
            mass=CONFIG["mass"],
            kbt=CONFIG["kbt"],
            steps=CONFIG["steps"],
            stride=CONFIG["stride"],
            integrator=CONFIG["integrator"],
            pmf=pmf,
            kernel=kernel,
        )
        traj = gle_sim.integrate(config)
        vel = np.gradient(traj, config.dt)
        acc = np.gradient(vel, config.dt)
        du = -gle_sim.get_force_pmf(config, traj)
        corrs["vv_corr"] = correlation.correlation(vel, vel, CONFIG["trunc"])
        corrs["va_corr"] = correlation.correlation(vel, acc, CONFIG["trunc"])
        corrs["aa_corr"] = correlation.correlation(acc, acc, CONFIG["trunc"])
        corrs["xdu_corr"] = correlation.correlation(traj, du, CONFIG["trunc"])
        corrs["vdu_corr"] = correlation.correlation(vel, du, CONFIG["trunc"])
        corrs["adu_corr"] = correlation.correlation(acc, du, CONFIG["trunc"])
        return cls(
            **corrs,
            traj=traj,
            dt=CONFIG["dt"],
            mass=CONFIG["mass"],
            trunc=CONFIG["trunc"],
            gammas=CONFIG["gammas"],
            ks=CONFIG["ks"],
            kbt=CONFIG["kbt"],
            bins=CONFIG["bins"],
        )


@lru_cache()
def load_test_data() -> TestData:
    """Only load the data once for all tests."""
    return TestData.load()


@lru_cache()
def simulate_test_data() -> TestData:
    """Only simulate the data once for all tests."""
    return TestData.simulate()
