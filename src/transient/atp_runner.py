import os
import subprocess
import shutil
import numpy as np
from pathlib import Path

class ATPResult:
    def __init__(self, case_path: Path, output_dir: Path, return_code: int, stdout: str, stderr: str):
        self.case_path = case_path
        self.output_dir = output_dir
        self.return_code = return_code
        self.stdout = stdout
        self.stderr = stderr

class ATPRunner:
    """
    Thin process adapter around the actual ATP-EMTP executable (tpbig/tpbigm).
    Executes the real Windows binary via Wine on Linux runtime.
    Throws a RuntimeError if Wine or tpbigm.exe is not available or if execution fails.
    """
    def __init__(self, atp_executable: str | Path = None, timeout_s: float = 300.0):
        self.timeout_s = timeout_s

    def run(self, atp_case_path: str | Path) -> ATPResult:
        case_path = Path(atp_case_path).resolve()
        if not case_path.exists():
            raise FileNotFoundError(f"ATP case file not found: {case_path}")

        if case_path.suffix.lower() != ".atp":
            raise ValueError(f"Expected .ATP case file, got: {case_path}")

        atp_dir = Path("atpmingw_2024").resolve()
        tpbigm = atp_dir / "tpbigm.exe" if atp_dir.exists() else None

        wine_path = shutil.which("wine")

        if wine_path is None:
            raise RuntimeError("Wine is not installed in the environment. ATP-EMTP execution requires Wine.")

        if tpbigm is None or not tpbigm.exists():
            raise RuntimeError(f"ATP-EMTP executable 'tpbigm.exe' not found at {atp_dir}")

        temp_case_name = "TEMP_CASE.ATP"
        temp_case_path = atp_dir / temp_case_name
        shutil.copy(case_path, temp_case_path)

        cmd = ["wine", "tpbigm.exe", "both", temp_case_name, ".", "-R"]
        env = os.environ.copy()
        wine32_prefix = Path.home() / ".wine32"
        if wine32_prefix.exists():
            env["WINEPREFIX"] = str(wine32_prefix)

        process = subprocess.run(
            cmd,
            cwd=atp_dir,
            env=env,
            capture_output=True,
            text=True,
            timeout=self.timeout_s,
            check=False
        )

        # Copy generated output files back (.lis, .dbg, .pl4)
        for suffix in [".lis", ".dbg", ".pl4"]:
            generated_file = atp_dir / f"TEMP_CASE{suffix}"
            if generated_file.exists():
                dest_file = case_path.with_suffix(suffix)
                shutil.copy(generated_file, dest_file)
                try:
                    generated_file.unlink()
                except Exception:
                    pass

        if temp_case_path.exists():
            try:
                temp_case_path.unlink()
            except Exception:
                pass

        if process.returncode != 0:
            raise RuntimeError(f"ATP-EMTP execution failed with return code {process.returncode}:\n{process.stderr}\n{process.stdout}")

        return ATPResult(
            case_path=case_path,
            output_dir=case_path.parent,
            return_code=process.returncode,
            stdout=process.stdout,
            stderr=process.stderr
        )
