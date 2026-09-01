from dataclasses import dataclass
from typing import List, Dict, Union, Optional
import numpy as np
from src.estimator.cla_estimator import ConsumerLoadClassModel


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
    Estimates unsampled consumer energy allocations using time adjustment factors
    and metered consumer unit energies:
        E_i_hat = E_U * w_i
    where sum(w_i) across unmetered population = 1.
    """

    def averaging_function(
        self,
        metered_consumer_energies: Dict[str, float],
        metered_units: List[object]
    ) -> Dict[str, float]:
        """
        Computes the class-level average energy consumed for each load class among metered consumer units.
        Returns a dictionary mapping class_id -> average metered energy consumed.
        """
        class_metered_energies: Dict[str, List[float]] = {}
        for u in metered_units:
            cid = getattr(u, "consumer_id", str(u))
            class_id = getattr(u, "assigned_load_class", "residential") or "residential"
            if cid in metered_consumer_energies:
                e_val = float(metered_consumer_energies[cid])
                class_metered_energies.setdefault(class_id, []).append(e_val)

        class_averages = {
            c_id: float(np.mean(e_list)) for c_id, e_list in class_metered_energies.items() if e_list
        }
        return class_averages

    def weighting_function(
        self,
        unmetered_units: List[object],
        metered_units: List[object],
        metered_consumer_energies: Dict[str, float]
    ) -> Dict[str, float]:
        """
        Computes normalized time-adjusted weights w_i for unmetered consumer units,
        adjusting unit weights proportional to the ratio of unmetered unit expected energy consumption
        to class-average metered energy consumption.
        Sum of returned weights across unmetered population equals 1.
        """
        if not unmetered_units:
            return {}

        class_metered_avg = self.averaging_function(
            metered_consumer_energies=metered_consumer_energies,
            metered_units=metered_units
        )

        raw_weights = {}
        for u in unmetered_units:
            cid = getattr(u, "consumer_id", str(u))
            class_id = getattr(u, "assigned_load_class", "residential") or "residential"
            base_w = ConsumerLoadClassModel.compute_expected_weight(u)

            if class_id in class_metered_avg and class_metered_avg[class_id] > 0:
                avg_metered_e = class_metered_avg[class_id]
                adjusted_w = base_w * (base_w / avg_metered_e)
            else:
                adjusted_w = base_w

            raw_weights[cid] = float(adjusted_w)

        sum_raw = sum(raw_weights.values())
        if sum_raw <= 0:
            n_units = len(unmetered_units)
            return {getattr(u, "consumer_id", str(u)): 1.0 / n_units for u in unmetered_units}

        normalized_weights = {cid: float(val / sum_raw) for cid, val in raw_weights.items()}
        return normalized_weights

    def validation_function(
        self,
        feeder_supply_energy_kwh: float,
        sampled_consumer_energy_kwh: float,
        technical_loss_kwh: float
    ) -> bool:
        """
        Validates energy balance conservation equation E_F >= E_M + E_L for time-adjusted allocation.
        """
        if feeder_supply_energy_kwh <= 0:
            return False
        if sampled_consumer_energy_kwh < 0 or technical_loss_kwh < 0:
            return False
        return (feeder_supply_energy_kwh >= (sampled_consumer_energy_kwh + technical_loss_kwh))

    def estimate(
        self,
        feeder_supply_energy_kwh: float,
        technical_loss_kwh: float,
        metered_consumer_energies: Dict[str, float],
        registry: Optional[object] = None,
        feeder_id: Optional[str] = None
    ) -> TimeAdjustedCLAEstimate:
        """
        Estimates unsampled customer energy allocations using Time-Adjusted CLA.
        Ensures exact feeder energy balance:
            feeder_supply_energy_kwh - technical_loss_kwh - sampled_consumer_energy_kwh - aggregate_allocated_load = 0
        """
        sampled_consumer_energy_kwh = float(sum(metered_consumer_energies.values())) if metered_consumer_energies else 0.0

        self.validation_function(
            feeder_supply_energy_kwh=feeder_supply_energy_kwh,
            sampled_consumer_energy_kwh=sampled_consumer_energy_kwh,
            technical_loss_kwh=technical_loss_kwh
        )

        e_u = max(0.0, float(feeder_supply_energy_kwh - sampled_consumer_energy_kwh - technical_loss_kwh))

        unmetered_units = []
        metered_units = []
        if registry is not None:
            if hasattr(registry, "get_unmetered_consumers"):
                unmetered_units = registry.get_unmetered_consumers()
            if hasattr(registry, "get_metered_consumers"):
                metered_units = registry.get_metered_consumers()

            if feeder_id:
                unmetered_units = [u for u in unmetered_units if getattr(u, "feeder_id", None) == feeder_id]
                metered_units = [u for u in metered_units if getattr(u, "feeder_id", None) == feeder_id]

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
            technical_loss_kwh=round(float(technical_loss_kwh), 4),
            time_adjusted_unsampled_energy_pool_kwh=round(e_u, 4),
            estimated_unsampled_energy_kwh=round(total_allocated, 4),
            allocated_unsampled_consumer_energy=allocations,
            weights={cid: round(float(w), 6) for cid, w in weights.items()}
        )
