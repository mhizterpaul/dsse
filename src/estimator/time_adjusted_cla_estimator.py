from dataclasses import dataclass
from typing import List, Dict, Union, Optional
import numpy as np
from src.power_plant.consumer_registry import ConsumerLoadClassModel


@dataclass
class TimeAdjustedCLAEstimate:
    feeder_supply_energy_kwh: float
    sampled_consumer_energy_kwh: float
    technical_loss_kwh: float
    time_adjusted_unsampled_energy_pool_kwh: float
    estimated_unsampled_energy_kwh: float
    allocated_unsampled_consumer_energy: Dict[str, float]
    weights: Dict[str, float]

    @property
    def estimated_unsampled_known_energy_kwh(self) -> float:
        return self.estimated_unsampled_energy_kwh


class TimeAdjustedCLAEstimator:
    """
    Time-Adjusted Cluster Load Allocation Estimator:
    Estimates unsampled consumer energy allocations using percentile mapping
    from assigned consumer weight distribution to metered consumer energy distribution per class,
    adjusted by exponent beta (s_i = w_i * q_i^beta), and normalized using L1 normalization over unmetered population U:
        W_i(beta) = (w_i * q_i^beta) / sum_{j in U} (w_j * q_j^beta)
        E_i_hat = E_U * W_i(beta)
    """

    def weighting_function(
        self,
        unmetered_units: List[object],
        metered_units: List[object],
        metered_consumer_energies: Dict[str, float],
        registry: Optional[object] = None,
        beta: float = 0.2
    ) -> Dict[str, float]:
        """
        Computes normalized time-adjusted weights W_i(beta) for unmetered consumer units:
        1. Gets percentile of unmetered consumer unit within their class assigned weight distribution.
        2. Maps percentile to distribution of consumed energy of metered consumers in the same class (q_i).
        3. Computes adjusted score s_i = w_i * q_i^beta.
        4. Performs L1 normalization on s_i over unmetered population U.
        """
        if not unmetered_units:
            return {}

        class_metered_energies: Dict[str, List[float]] = {}
        for u in metered_units:
            cid = getattr(u, "consumer_id", None)
            class_id = getattr(u, "assigned_load_class", None)
            if cid and class_id and cid in metered_consumer_energies:
                class_metered_energies.setdefault(class_id, []).append(float(metered_consumer_energies[cid]))

        class_unmetered_units: Dict[str, List[object]] = {}
        class_unmetered_weights: Dict[str, Dict[str, float]] = {}
        for u in unmetered_units:
            cid = getattr(u, "consumer_id", None)
            class_id = getattr(u, "assigned_load_class", None)
            if cid is None or class_id is None:
                raise ValueError(f"Unmetered consumer unit {u} missing consumer_id or assigned_load_class")

            if registry is not None and hasattr(registry, "get_assigned_weight"):
                assigned_w = registry.get_assigned_weight(u)
            else:
                assigned_w = ConsumerLoadClassModel.compute_expected_weight(u, registry=registry)

            class_unmetered_units.setdefault(class_id, []).append(u)
            class_unmetered_weights.setdefault(class_id, {})[cid] = float(assigned_w)

        scores: Dict[str, float] = {}
        for class_id, u_list in class_unmetered_units.items():
            weights_dict = class_unmetered_weights[class_id]
            sorted_weights = sorted(weights_dict.values())
            n_class_unmetered = len(sorted_weights)

            metered_e_list = class_metered_energies.get(class_id, [])
            if not metered_e_list:
                raise ValueError(f"Missing metered energy observations for class '{class_id}'")

            for u in u_list:
                cid = u.consumer_id
                w_val = weights_dict[cid]
                if n_class_unmetered > 1:
                    rank = sum(1 for w in sorted_weights if w <= w_val)
                    percentile_p = ((rank - 1) / (n_class_unmetered - 1)) * 100.0
                else:
                    percentile_p = 50.0

                q_i = float(np.percentile(metered_e_list, percentile_p))
                s_i = float(w_val * (q_i ** beta))
                scores[cid] = s_i

        sum_scores = sum(scores.values())
        if sum_scores <= 0:
            n_units = len(unmetered_units)
            return {u.consumer_id: 1.0 / n_units for u in unmetered_units}

        normalized_weights = {cid: float(s / sum_scores) for cid, s in scores.items()}
        return normalized_weights

    def estimate(
        self,
        feeder_supply_energy_kwh: float,
        technical_loss_kwh: float,
        metered_consumer_energies: Dict[str, float],
        registry: Optional[object] = None,
        beta: float = 0.2
    ) -> TimeAdjustedCLAEstimate:
        """
        Estimates unsampled customer energy allocations using Time-Adjusted CLA.
        Ensures exact feeder energy balance:
            feeder_supply_energy_kwh - technical_loss_kwh - sampled_consumer_energy_kwh - aggregate_allocated_load = 0
        """
        sampled_consumer_energy_kwh = float(sum(metered_consumer_energies.values())) if metered_consumer_energies else 0.0

        e_u = max(0.0, float(feeder_supply_energy_kwh - sampled_consumer_energy_kwh - technical_loss_kwh))

        unmetered_units = []
        metered_units = []
        if registry is not None:
            if hasattr(registry, "get_unmetered_consumers"):
                unmetered_units = registry.get_unmetered_consumers()
            if hasattr(registry, "get_metered_consumers"):
                all_metered = registry.get_metered_consumers()
                metered_units = [u for u in all_metered if getattr(u, "consumer_id", None) in metered_consumer_energies]

        if not unmetered_units:
            return TimeAdjustedCLAEstimate(
                feeder_supply_energy_kwh=round(float(feeder_supply_energy_kwh), 4),
                sampled_consumer_energy_kwh=round(float(sampled_consumer_energy_kwh), 4),
                technical_loss_kwh=round(float(technical_loss_kwh), 4),
                time_adjusted_unsampled_energy_pool_kwh=round(e_u, 4),
                estimated_unsampled_energy_kwh=0.0,
                allocated_unsampled_consumer_energy={},
                weights={}
            )

        weights = self.weighting_function(
            unmetered_units=unmetered_units,
            metered_units=metered_units,
            metered_consumer_energies=metered_consumer_energies,
            registry=registry,
            beta=beta
        )

        allocations = {}
        for cid, w_i in weights.items():
            e_hat_i = e_u * w_i
            allocations[cid] = round(float(e_hat_i), 4)

        total_allocated = float(sum(allocations.values()))

        return TimeAdjustedCLAEstimate(
            feeder_supply_energy_kwh=round(float(feeder_supply_energy_kwh), 4),
            sampled_consumer_energy_kwh=round(float(sampled_consumer_energy_kwh), 4),
            technical_loss_kwh=round(float(technical_loss_kwh), 4),
            time_adjusted_unsampled_energy_pool_kwh=round(e_u, 4),
            estimated_unsampled_energy_kwh=round(total_allocated, 4),
            allocated_unsampled_consumer_energy=allocations,
            weights={cid: round(float(w), 6) for cid, w in weights.items()}
        )
