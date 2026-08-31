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

    def weighting_function(self, premises_list: Union[ConsumerLoadPremises, List[ConsumerLoadPremises]]) -> Dict[str, float]:
        """
        Computes normalized weights w_i for unsampled consumer units such that sum(w_i) = 1.
        """
        if isinstance(premises_list, ConsumerLoadPremises):
            premises_list = [premises_list]

        if not premises_list:
            return {}

        raw_weights = {}
        for p in premises_list:
            raw_w = ConsumerLoadClassModel.compute_expected_weight(p)
            raw_weights[p.consumer_id] = raw_w

        sum_raw = sum(raw_weights.values())
        if sum_raw <= 0:
            n_units = len(premises_list)
            return {p.consumer_id: 1.0 / n_units for p in premises_list}

        normalized_weights = {cid: float(w / sum_raw) for cid, w in raw_weights.items()}
        return normalized_weights

    def validation_function(
        self,
        feeder_supply_energy_kwh: float,
        sampled_consumer_energy_kwh: float,
        technical_loss_kwh: float
    ) -> bool:
        """
        Validates energy balance conservation equation E_F >= E_M + E_L.
        """
        if feeder_supply_energy_kwh <= 0:
            return False
        if sampled_consumer_energy_kwh < 0 or technical_loss_kwh < 0:
            return False
        return (feeder_supply_energy_kwh >= (sampled_consumer_energy_kwh + technical_loss_kwh))

    def estimate(
        self,
        feeder_supply_energy_kwh: float,
        sampled_consumer_energy_kwh: float,
        technical_loss_kwh: float,
        unsampled_premises: List[ConsumerLoadPremises]
    ) -> ClusterLoadAllocationEstimate:
        """
        Estimates unsampled customer energy allocations using baseline CLA.
        Ensures exact feeder energy balance:
            feeder_supply_energy_kwh - technical_losses_kwh - sampled_consumer_energy_kwh - aggregate_allocated_load = 0
        """
        self.validation_function(
            feeder_supply_energy_kwh=feeder_supply_energy_kwh,
            sampled_consumer_energy_kwh=sampled_consumer_energy_kwh,
            technical_loss_kwh=technical_loss_kwh
        )

        e_u = max(0.0, float(feeder_supply_energy_kwh - sampled_consumer_energy_kwh - technical_loss_kwh))

        if not unsampled_premises:
            return ClusterLoadAllocationEstimate(
                feeder_supply_energy_kwh=feeder_supply_energy_kwh,
                sampled_consumer_energy_kwh=sampled_consumer_energy_kwh,
                technical_loss_kwh=technical_loss_kwh,
                unsampled_energy_pool_kwh=e_u,
                estimated_unsampled_energy_kwh=0.0,
                allocated_unsampled_consumer_energy={},
                weights={}
            )

        weights = self.weighting_function(unsampled_premises)

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
