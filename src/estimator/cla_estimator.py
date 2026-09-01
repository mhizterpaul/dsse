from dataclasses import dataclass
from typing import List, Dict, Union, Optional
import numpy as np


class ConsumerLoadClassModel:
    """
    Consumer Premises and Load Class Representation for CLA:
    Defines consumer class weights based on assigned load class.
    """
    CLASS_WEIGHTS = {
        "residential": 1.0,
        "commercial": 2.2,
        "industrial": 3.5,
        "agricultural": 1.5
    }

    @classmethod
    def compute_expected_weight(cls, unit) -> float:
        """
        Computes expected consumption weight w_i based on consumer unit characteristics.
        """
        class_id = getattr(unit, "assigned_load_class", "residential") or "residential"
        base_w = cls.CLASS_WEIGHTS.get(class_id, 1.0)
        num_loads = len(getattr(unit, "loads", [])) or 1
        return float(base_w * num_loads)


@dataclass
class ClusterLoadAllocationEstimate:
    feeder_supply_energy_kwh: float
    sampled_consumer_energy_kwh: float
    technical_loss_kwh: float
    unsampled_energy_pool_kwh: float
    estimated_unsampled_energy_kwh: float
    allocated_unsampled_consumer_energy: Dict[str, float]
    weights: Dict[str, float]

    @property
    def estimated_unsampled_known_energy_kwh(self) -> float:
        return self.estimated_unsampled_energy_kwh


class ClusterLoadAllocationEstimator:
    """
    Baseline Cluster Load Allocation (CLA) Estimator:
    Formulates E_U = E_F - E_M - E_L and allocates unsampled customer energy:
        E_i_hat = E_U * w_i
    where sum(w_i) across unmetered population = 1.
    """

    def averaging_function(self, values: Union[List[float], np.ndarray]) -> float:
        """
        Computes the arithmetic average of allocated/observed energy consumption values.
        """
        vals = np.asarray(values, dtype=float)
        if len(vals) == 0:
            return 0.0
        return float(np.mean(vals))

    def weighting_function(
        self,
        unmetered_units: List[object]
    ) -> Dict[str, float]:
        """
        Computes normalized weights w_i for unmetered consumer units such that sum(w_i) = 1.
        """
        if not unmetered_units:
            return {}

        raw_weights = {}
        for u in unmetered_units:
            cid = getattr(u, "consumer_id", str(u))
            raw_w = ConsumerLoadClassModel.compute_expected_weight(u)
            raw_weights[cid] = raw_w

        sum_raw = sum(raw_weights.values())
        if sum_raw <= 0:
            n_units = len(unmetered_units)
            return {getattr(u, "consumer_id", str(u)): 1.0 / n_units for u in unmetered_units}

        normalized_weights = {cid: float(w / sum_raw) for cid, w in raw_weights.items()}
        return normalized_weights

    def estimate(
        self,
        feeder_supply_energy_kwh: float,
        sampled_consumer_energy_kwh: float,
        technical_loss_kwh: float,
        registry: Optional[object] = None
    ) -> ClusterLoadAllocationEstimate:
        """
        Estimates unsampled customer energy allocations using baseline CLA.
        Ensures exact feeder energy balance:
            feeder_supply_energy_kwh - technical_loss_kwh - sampled_consumer_energy_kwh - aggregate_allocated_load = 0
        """
        e_u = max(0.0, float(feeder_supply_energy_kwh - sampled_consumer_energy_kwh - technical_loss_kwh))

        unmetered_units = []
        if registry is not None and hasattr(registry, "get_unmetered_consumers"):
            unmetered_units = registry.get_unmetered_consumers()

        weights = self.weighting_function(unmetered_units)

        allocations = {}
        for cid, w_i in weights.items():
            e_hat_i = e_u * w_i
            allocations[cid] = round(float(e_hat_i), 4)

        total_allocated = float(sum(allocations.values()))

        return ClusterLoadAllocationEstimate(
            feeder_supply_energy_kwh=round(float(feeder_supply_energy_kwh), 4),
            sampled_consumer_energy_kwh=round(float(sampled_consumer_energy_kwh), 4),
            technical_loss_kwh=round(float(technical_loss_kwh), 4),
            unsampled_energy_pool_kwh=round(e_u, 4),
            estimated_unsampled_energy_kwh=round(total_allocated, 4),
            allocated_unsampled_consumer_energy=allocations,
            weights={cid: round(float(w), 6) for cid, w in weights.items()}
        )
