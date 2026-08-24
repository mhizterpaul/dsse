from dataclasses import dataclass
import numpy as np

@dataclass
class ConsumerLoadPremises:
    consumer_id: str
    class_id: str  # e.g., 'residential_light', 'commercial', 'industrial_motor'
    is_metered: bool
    connected_load_kw: float
    historical_billing_kwh: float = 100.0
    supply_availability: float = 1.0

class ConsumerLoadClassModel:
    """
    Consumer Premises and Load Class Representation for CLA:
    Defines consumer classes c and metered class energy profiles mu_c(t).
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
    def get_metered_class_profile(cls, class_id: str, t_points: np.ndarray) -> np.ndarray:
        """
        Returns normalized metered class profile mu_c(t).
        """
        t = np.asarray(t_points, dtype=float)
        if class_id == "residential_light":
            profile = 0.5 + 0.5 * np.sin(2 * np.pi * t / 24.0)
        elif class_id == "commercial":
            profile = 0.2 + 0.8 * (1.0 / (1.0 + np.exp(-0.5 * (t - 12.0))))
        else: # industrial_motor
            profile = 0.8 + 0.2 * np.cos(2 * np.pi * t / 12.0)
        return np.maximum(0.05, profile)

@dataclass
class ClusterLoadAllocationEstimate:
    feeder_supply_energy_kwh: float
    metered_customer_energy_kwh: float
    estimated_technical_loss_kwh: float
    unmetered_energy_pool_kwh: float
    estimated_unmetered_energy_kwh: float
    allocated_unmetered_customer_energy: dict[str, float]

class ClusterLoadAllocationEstimator:
    """
    Baseline Cluster Load Allocation (CLA) Estimator:
    Formulates E_U = E_F - E_M - E_L and allocates unmetered customer energy:
        E_i_hat = E_U * (w_i / sum(w_j))
    """

    def estimate(
        self,
        feeder_supply_energy_kwh: float,
        metered_customer_energy_kwh: float,
        estimated_technical_loss_kwh: float,
        unmetered_premises: list[ConsumerLoadPremises]
    ) -> ClusterLoadAllocationEstimate:
        """
        Estimates unmetered customer energy allocations using baseline CLA.
        """
        e_u = max(0.0, float(feeder_supply_energy_kwh - metered_customer_energy_kwh - estimated_technical_loss_kwh))

        if not unmetered_premises:
            return ClusterLoadAllocationEstimate(
                feeder_supply_energy_kwh=feeder_supply_energy_kwh,
                metered_customer_energy_kwh=metered_customer_energy_kwh,
                estimated_technical_loss_kwh=estimated_technical_loss_kwh,
                unmetered_energy_pool_kwh=e_u,
                estimated_unmetered_energy_kwh=0.0,
                allocated_unmetered_customer_energy={}
            )

        weights = {}
        for p in unmetered_premises:
            w_i = ConsumerLoadClassModel.compute_expected_weight(p)
            weights[p.consumer_id] = w_i

        sum_w = sum(weights.values())
        if sum_w <= 0:
            sum_w = float(len(unmetered_premises))
            weights = {p.consumer_id: 1.0 for p in unmetered_premises}

        allocations = {}
        for cid, w_i in weights.items():
            e_hat_i = e_u * (w_i / sum_w)
            allocations[cid] = round(float(e_hat_i), 4)

        total_allocated = float(sum(allocations.values()))

        return ClusterLoadAllocationEstimate(
            feeder_supply_energy_kwh=round(float(feeder_supply_energy_kwh), 4),
            metered_customer_energy_kwh=round(float(metered_customer_energy_kwh), 4),
            estimated_technical_loss_kwh=round(float(estimated_technical_loss_kwh), 4),
            unmetered_energy_pool_kwh=round(e_u, 4),
            estimated_unmetered_energy_kwh=round(total_allocated, 4),
            allocated_unmetered_customer_energy=allocations
        )
