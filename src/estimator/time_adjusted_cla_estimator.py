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


class TimeAdjustedCLAEstimator:
    """
    Time-Adjusted Cluster Load Allocation Estimator:
    Estimates unsampled consumer energy allocations using time adjustment factors alpha_i(t)
    and sampled-class profiles mu_c(t):
        E_i_hat = integral(alpha_i(t) * mu_c_i(t) dt)
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
        time_points: Optional[np.ndarray] = None,
        observed_time_adjustment_factors: Optional[Dict[str, float]] = None
    ) -> Dict[str, float]:
        """
        Computes time-adjusted integration weights for unsampled consumer units.
        """
        if isinstance(premises_list, ConsumerLoadPremises):
            premises_list = [premises_list]

        if time_points is None:
            time_points = np.linspace(0.0, 24.0, 100)
        dt = float(time_points[1] - time_points[0]) if len(time_points) > 1 else 1.0

        raw_time_integrals = {}
        for p in premises_list:
            alpha_i = observed_time_adjustment_factors.get(p.consumer_id, 1.05) if observed_time_adjustment_factors else 1.05
            mu_c = ConsumerLoadClassModel.get_sampled_class_profile(p.class_id, time_points)
            raw_integral = float(np.sum(alpha_i * mu_c * dt))
            raw_time_integrals[p.consumer_id] = max(0.01, raw_integral)

        return raw_time_integrals

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
        time_points: Optional[np.ndarray] = None,
        observed_time_adjustment_factors: Optional[Dict[str, float]] = None
    ) -> TimeAdjustedCLAEstimate:
        """
        Estimates unsampled customer energy allocations using Time-Adjusted CLA.
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
                allocated_unsampled_consumer_energy={}
            )

        raw_time_integrals = self.weighting_function(
            premises_list=unsampled_premises,
            time_points=time_points,
            observed_time_adjustment_factors=observed_time_adjustment_factors
        )

        sum_integrals = sum(raw_time_integrals.values())
        allocations = {}

        for cid, raw_val in raw_time_integrals.items():
            e_hat_i = e_u * (raw_val / sum_integrals)
            allocations[cid] = round(float(e_hat_i), 4)

        total_allocated = float(sum(allocations.values()))

        return TimeAdjustedCLAEstimate(
            feeder_supply_energy_kwh=round(float(feeder_supply_energy_kwh), 4),
            sampled_consumer_energy_kwh=round(float(sampled_consumer_energy_kwh), 4),
            estimated_technical_loss_kwh=round(float(estimated_technical_loss_kwh), 4),
            time_adjusted_unsampled_energy_pool_kwh=round(e_u, 4),
            estimated_unsampled_energy_kwh=round(total_allocated, 4),
            allocated_unsampled_consumer_energy=allocations
        )
