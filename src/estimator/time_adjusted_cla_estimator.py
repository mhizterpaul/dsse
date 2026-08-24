from dataclasses import dataclass
import numpy as np
from src.estimator.cla_estimator import ConsumerLoadPremises, ConsumerLoadClassModel

@dataclass
class TimeAdjustedCLAEstimate:
    feeder_supply_energy_kwh: float
    metered_customer_energy_kwh: float
    estimated_technical_loss_kwh: float
    time_adjusted_unmetered_energy_pool_kwh: float
    estimated_unmetered_energy_kwh: float
    allocated_unmetered_customer_energy: dict[str, float]

class TimeAdjustedCLAEstimator:
    """
    Time-Adjusted Cluster Load Allocation Estimator:
    Estimates unmetered consumer energy allocations using time adjustment factors alpha_i(t)
    and metered-class profiles mu_c(t):
        E_i_hat = integral(alpha_i(t) * mu_c_i(t) dt)
    """

    def estimate(
        self,
        feeder_supply_energy_kwh: float,
        metered_customer_energy_kwh: float,
        estimated_technical_loss_kwh: float,
        unmetered_premises: list[ConsumerLoadPremises],
        time_points: np.ndarray = None,
        observed_time_adjustment_factors: dict[str, float] = None
    ) -> TimeAdjustedCLAEstimate:
        """
        Estimates unmetered customer energy allocations using Time-Adjusted CLA.
        """
        e_u = max(0.0, float(feeder_supply_energy_kwh - metered_customer_energy_kwh - estimated_technical_loss_kwh))

        if time_points is None:
            time_points = np.linspace(0.0, 24.0, 100)
        dt = float(time_points[1] - time_points[0]) if len(time_points) > 1 else 1.0

        if not unmetered_premises:
            return TimeAdjustedCLAEstimate(
                feeder_supply_energy_kwh=feeder_supply_energy_kwh,
                metered_customer_energy_kwh=metered_customer_energy_kwh,
                estimated_technical_loss_kwh=estimated_technical_loss_kwh,
                time_adjusted_unmetered_energy_pool_kwh=e_u,
                estimated_unmetered_energy_kwh=0.0,
                allocated_unmetered_customer_energy={}
            )

        raw_time_integrals = {}
        for p in unmetered_premises:
            alpha_i = observed_time_adjustment_factors.get(p.consumer_id, 1.05) if observed_time_adjustment_factors else 1.05
            mu_c = ConsumerLoadClassModel.get_metered_class_profile(p.class_id, time_points)
            raw_integral = float(np.sum(alpha_i * mu_c * dt))
            raw_time_integrals[p.consumer_id] = max(0.01, raw_integral)

        sum_integrals = sum(raw_time_integrals.values())
        allocations = {}

        for cid, raw_val in raw_time_integrals.items():
            e_hat_i = e_u * (raw_val / sum_integrals)
            allocations[cid] = round(float(e_hat_i), 4)

        total_allocated = float(sum(allocations.values()))

        return TimeAdjustedCLAEstimate(
            feeder_supply_energy_kwh=round(float(feeder_supply_energy_kwh), 4),
            metered_customer_energy_kwh=round(float(metered_customer_energy_kwh), 4),
            estimated_technical_loss_kwh=round(float(estimated_technical_loss_kwh), 4),
            time_adjusted_unmetered_energy_pool_kwh=round(e_u, 4),
            estimated_unmetered_energy_kwh=round(total_allocated, 4),
            allocated_unmetered_customer_energy=allocations
        )
