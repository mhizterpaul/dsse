from dataclasses import dataclass
from typing import List, Dict, Union, Optional
import numpy as np
from src.estimator.cla_estimator import ConsumerLoadPremises, ConsumerLoadClassModel


@dataclass
class TimeAdjustedCLAEstimate:
    feeder_supply_energy_kwh: float
    sampled_consumer_energy_kwh: float
    estimated_technical_loss_kwh: float
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
    Estimates unsampled consumer energy allocations using time adjustment factors alpha_i(t)
    and sampled-class profiles mu_c(t):
        E_i_hat = E_U * w_i
    where sum(w_i) across unmetered population = 1.
    """

    def averaging_function(self, values: Union[List[float], np.ndarray]) -> float:
        """
        Computes time-averaged energy profile allocation across time windows.
        """
        vals = np.asarray(values, dtype=float)
        if len(vals) == 0:
            return 0.0
        return float(np.mean(vals))

    def weighting_function(
        self,
        premises_list: Union[ConsumerLoadPremises, List[ConsumerLoadPremises]],
        time_points: np.ndarray,
        observed_time_adjustment_factors: Dict[str, float],
        metered_premises: Optional[List[ConsumerLoadPremises]] = None,
        metered_consumer_energies: Optional[Dict[str, float]] = None
    ) -> Dict[str, float]:
        """
        Computes normalized time-adjusted weights w_i for unsampled consumer units,
        adjusting unit weights based on the average energy consumed by metered consumer units of the same class.
        Sum of returned weights across unsampled population equals 1.
        """
        if isinstance(premises_list, ConsumerLoadPremises):
            premises_list = [premises_list]

        if not premises_list:
            return {}

        if time_points is None or len(time_points) == 0:
            raise ValueError("time_points array must be provided for Time-Adjusted CLA estimation.")

        if observed_time_adjustment_factors is None:
            raise ValueError("observed_time_adjustment_factors dictionary must be provided for Time-Adjusted CLA estimation.")

        if not metered_premises or not metered_consumer_energies:
            raise ValueError("metered_premises and metered_consumer_energies must be provided for Time-Adjusted CLA estimation.")

        class_metered_energies: Dict[str, List[float]] = {}
        all_metered_energies: List[float] = []

        for mp in metered_premises:
            if mp.consumer_id not in metered_consumer_energies:
                raise ValueError(f"Missing metered energy observation for consumer {mp.consumer_id}")
            e_val = float(metered_consumer_energies[mp.consumer_id])
            class_metered_energies.setdefault(mp.class_id, []).append(e_val)
            all_metered_energies.append(e_val)

        if not all_metered_energies:
            raise ValueError("No metered consumer energy values found.")

        overall_metered_avg = float(np.mean(all_metered_energies))
        if overall_metered_avg <= 0:
            raise ValueError(f"Invalid overall metered average energy: {overall_metered_avg}")

        class_metered_avg = {
            c_id: float(np.mean(e_list)) for c_id, e_list in class_metered_energies.items() if e_list
        }

        raw_time_integrals = {}
        for p in premises_list:
            base_w = ConsumerLoadClassModel.compute_expected_weight(p)
            if p.class_id not in class_metered_avg or class_metered_avg[p.class_id] <= 0:
                raise ValueError(f"No valid positive metered energy observed for load class '{p.class_id}' to compute class average metered energy.")

            avg_metered_e = class_metered_avg[p.class_id]
            class_energy_ratio = base_w / avg_metered_e

            if p.consumer_id not in observed_time_adjustment_factors:
                raise ValueError(f"Missing time adjustment factor alpha_i for consumer premises {p.consumer_id}")

            alpha_i = float(observed_time_adjustment_factors[p.consumer_id])
            adjusted_w = base_w * class_energy_ratio * alpha_i
            raw_time_integrals[p.consumer_id] = float(adjusted_w)

        sum_integrals = sum(raw_time_integrals.values())
        if sum_integrals <= 0:
            n_units = len(premises_list)
            return {p.consumer_id: 1.0 / n_units for p in premises_list}

        normalized_weights = {cid: float(val / sum_integrals) for cid, val in raw_time_integrals.items()}
        return normalized_weights

    def validation_function(
        self,
        feeder_supply_energy_kwh: float,
        sampled_consumer_energy_kwh: float,
        estimated_technical_loss_kwh: float
    ) -> bool:
        """
        Validates energy balance conservation equation E_F >= E_M + E_L for time-adjusted allocation.
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
        unsampled_premises: List[ConsumerLoadPremises],
        time_points: np.ndarray,
        observed_time_adjustment_factors: Dict[str, float],
        metered_premises: Optional[List[ConsumerLoadPremises]] = None,
        metered_consumer_energies: Optional[Dict[str, float]] = None
    ) -> TimeAdjustedCLAEstimate:
        """
        Estimates unsampled customer energy allocations using Time-Adjusted CLA.
        Ensures exact feeder energy balance:
            feeder_supply_energy_kwh - technical_losses_kwh - sampled_consumer_energy_kwh - aggregate_allocated_load = 0
        """
        self.validation_function(
            feeder_supply_energy_kwh=feeder_supply_energy_kwh,
            sampled_consumer_energy_kwh=sampled_consumer_energy_kwh,
            estimated_technical_loss_kwh=estimated_technical_loss_kwh
        )

        e_u = max(0.0, float(feeder_supply_energy_kwh - sampled_consumer_energy_kwh - estimated_technical_loss_kwh))

        if not unsampled_premises:
            return TimeAdjustedCLAEstimate(
                feeder_supply_energy_kwh=feeder_supply_energy_kwh,
                sampled_consumer_energy_kwh=sampled_consumer_energy_kwh,
                estimated_technical_loss_kwh=estimated_technical_loss_kwh,
                time_adjusted_unsampled_energy_pool_kwh=e_u,
                estimated_unsampled_energy_kwh=0.0,
                allocated_unsampled_consumer_energy={},
                weights={}
            )

        weights = self.weighting_function(
            premises_list=unsampled_premises,
            time_points=time_points,
            observed_time_adjustment_factors=observed_time_adjustment_factors,
            metered_premises=metered_premises,
            metered_consumer_energies=metered_consumer_energies
        )

        allocations = {}
        for cid, w_i in weights.items():
            e_hat_i = e_u * w_i
            allocations[cid] = round(float(e_hat_i), 4)

        total_allocated = float(sum(allocations.values()))

        return TimeAdjustedCLAEstimate(
            feeder_supply_energy_kwh=round(float(feeder_supply_energy_kwh), 4),
            sampled_consumer_energy_kwh=round(float(sampled_consumer_energy_kwh), 4),
            estimated_technical_loss_kwh=round(float(estimated_technical_loss_kwh), 4),
            time_adjusted_unsampled_energy_pool_kwh=round(e_u, 4),
            estimated_unsampled_energy_kwh=round(total_allocated, 4),
            allocated_unsampled_consumer_energy=allocations,
            weights={cid: round(float(w), 6) for cid, w in weights.items()}
        )
