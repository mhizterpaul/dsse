from opendssdirect import dss

def configure_generator(p_kw: float = 1500.0, q_kvar: float = 0.0):
    """
    Sets up or edits the shared generator in the OpenDSS model.
    The generator is treated as an actual controllable operating-condition/excitation source.
    """
    kva = max(2000.0, abs(p_kw) * 1.3)
    dss.run_command(
        f"new generator.shared_gen "
        f"bus1=main_bus "
        f"phases=3 "
        f"kv=11.0 "
        f"kw={p_kw} "
        f"kvar={q_kvar} "
        f"kva={kva} "
        f"model=1"
    )

def apply_generator_profile(p_kw: float, q_kvar: float):
    """
    Dynamically edits the operating point of the shared generator.
    """
    kva = max(2000.0, abs(p_kw) * 1.3)
    dss.run_command(f"edit generator.shared_gen kw={p_kw} kvar={q_kvar} kva={kva}")
