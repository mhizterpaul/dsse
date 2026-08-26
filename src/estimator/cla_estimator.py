from dataclasses import dataclass
from typing import List, Dict, Union
import numpy as np


@dataclass
class ConsumerLoadPremises:
    consumer_id: str
    class_id: str  # e.g., 'residential_light', 'commercial', 'industrial_motor'
    is_sampled: bool
    connected_load_kw: float
    historical_billing_kwh: float = 100.0
    supply_availability: float = 1.0


class ConsumerLoadClassModel:
    """
    Consumer Premises and Load Class Representation for CLA:
    Defines consumer classes c and sampled class energy profiles mu_c(t).
    """
    CLASSES = ["residential_light", "commercial", "industrial_motor"]

    CLASS_WEIGHTS = {
        "residential_light": 1.0,
        "commercial": 2.2,
        "industrial_motor": 3.5
    }

    @classmethod
    def compute_expected_weight(cls, premises: ConsumerLoadPremises) -> float:
        """
        Computes expected consumption weight w_i = E[E_i | C_i, X_i] based on premises characteristics.
        """
        base_w = cls.CLASS_WEIGHTS.get(premises.class_id, 1.0)
        return float(base_w * (premises.connected_load_kw / 10.0) * premises.supply_availability)

    @classmethod
    def get_sampled_class_profile(cls, class_id: str, t_points: np.ndarray) -> np.ndarray:
        """
        Returns normalized sampled class profile mu_c(t).
        """
        t = np.asarray(t_points, dtype=float)
        if class_id == "residential_light":
            profile = 0.5 + 0.5 * np.sin(2 * np.pi * t / 24.0)
        elif class_id == "commercial":
            profile = 0.2 + 0.8 * (1.0 / (1.0 + np.exp(-0.5 * (t - 12.0))))
        else:  # industrial_motor
            profile = 0.8 + 0.2 * np.cos(2 * np.pi * t / 12.0)
        return np.maximum(0.05, profile)


@dataclass
class ClusterLoadAllocationEstimate:
    feeder_supply_energy_kwh: float
    sampled_consumer_energy_kwh: float
    estimated_technical_loss_kwh: float
    unsampled_energy_pool_kwh: float
    estimated_unsampled_energy_kwh: float
    allocated_unsampled_consumer_energy: Dict[str, float]

    @property
    def estimated_unsampled_known_energy_kwh(self) -> float:
        return self.estimated_unsampled_energy_kwh


class ClusterLoadAllocationEstimator:
    """
    Baseline Cluster Load Allocation (CLA) Estimator:
    Formulates E_U = E_F - E_M - E_L and allocates unsampled customer energy:
        E_i_hat = E_U * (w_i / sum(w_j))
    """

    def averaging_function(self, values: Union[List[float], np.ndarray]) -> float:
        """
        Computes the arithmetic average of allocated/observed energy consumption values.
        """
        vals = np.asarray(values, dtype=float)
        if len(vals) == 0:
            return 0.0
        return float(np.mean(vals))

    def weighting_function(self, premises_list: Union[ConsumerLoadPremises, List[ConsumerLoadPremises]]) -> Dict[str, float]:
        """
        Computes expected consumption weights w_i = E[E_i | C_i, X_i] for unsampled consumer units.
        """
        if isinstance(premises_list, ConsumerLoadPremises):
            premises_list = [premises_list]

        weights = {}
        for p in premises_list:
            w_i = ConsumerLoadClassModel.compute_expected_weight(p)
            weights[p.consumer_id] = w_i
        return weights

    def validation_function(
        self,
        feeder_supply_energy_kwh: float,
        sampled_consumer_energy_kwh: float,
        estimated_technical_loss_kwh: float
    ) -> bool:
        """
        Validates energy balance conservation equation E_F >= E_M + E_L.
        """
        if feeder_supply_energy_kwh <= 0:
            return False
        if sampled_consumer_energy_kwh < 0 or estimated_technical_loss_kwh < 0:
            return False
        return (feeder_supply_energy_kwh >= (sampled_consumer_energy_kwh + estimated_technical_loss_kwh))

    def estimate(
        self,
        feeder_supply_energy_kwh: float,
        sampled_consumer_energy_kwh: float,
        estimated_technical_loss_kwh: float,
        unsampled_premises: List[ConsumerLoadPremises]
    ) -> ClusterLoadAllocationEstimate:
        """
        Estimates unsampled customer energy allocations using baseline CLA.
        """
        self.validation_function(
            feeder_supply_energy_kwh=feeder_supply_energy_kwh,
            sampled_consumer_energy_kwh=sampled_consumer_energy_kwh,
            estimated_technical_loss_kwh=estimated_technical_loss_kwh
        )

        e_u = max(0.0, float(feeder_supply_energy_kwh - sampled_consumer_energy_kwh - estimated_technical_loss_kwh))

        if not unsampled_premises:
            return ClusterLoadAllocationEstimate(
                feeder_supply_energy_kwh=feeder_supply_energy_kwh,
                sampled_consumer_energy_kwh=sampled_consumer_energy_kwh,
                estimated_technical_loss_kwh=estimated_technical_loss_kwh,
                unsampled_energy_pool_kwh=e_u,
                estimated_unsampled_energy_kwh=0.0,
                allocated_unsampled_consumer_energy={}
            )

        weights = self.weighting_function(unsampled_premises)
        sum_w = sum(weights.values())
        if sum_w <= 0:
            sum_w = float(len(unsampled_premises))
            weights = {p.consumer_id: 1.0 for p in unsampled_premises}

        allocations = {}
        for cid, w_i in weights.items():
            e_hat_i = e_u * (w_i / sum_w)
            allocations[cid] = round(float(e_hat_i), 4)

        total_allocated = float(sum(allocations.values()))

        return ClusterLoadAllocationEstimate(
            feeder_supply_energy_kwh=round(float(feeder_supply_energy_kwh), 4),
            sampled_consumer_energy_kwh=round(float(sampled_consumer_energy_kwh), 4),
            estimated_technical_loss_kwh=round(float(estimated_technical_loss_kwh), 4),
            unsampled_energy_pool_kwh=round(e_u, 4),
            estimated_unsampled_energy_kwh=round(total_allocated, 4),
            allocated_unsampled_consumer_energy=allocations
        )
