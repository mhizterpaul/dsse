import os
import subprocess
import shutil
import uuid
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
    Supports process isolation for parallel ProcessPoolExecutor tasks by generating unique temporary case names.
    Supports expect_pl4=False for BCTRAN supporting routine matrix generation cases.
    """

    def __init__(self, atp_executable: str | Path = None, timeout_s: float = 300.0):
        self.timeout_s = timeout_s

    def run(self, atp_case_path: str | Path, expect_pl4: bool = True) -> ATPResult:
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

        work_dir = atp_dir / f"work_{os.getpid()}_{uuid.uuid4().hex[:8]}"
        work_dir.mkdir(parents=True, exist_ok=True)

        try:
            # Symlink or copy support files and binaries from atp_dir into isolated work_dir
            for item in atp_dir.iterdir():
                if item.is_file():
                    target_symlink = work_dir / item.name
                    try:
                        os.symlink(item, target_symlink)
                    except Exception:
                        shutil.copy(item, target_symlink)

            temp_stem = f"TEMP_CASE_{os.getpid()}_{uuid.uuid4().hex[:6]}"
            temp_case_name = f"{temp_stem}.ATP"
            temp_case_path = work_dir / temp_case_name
            shutil.copy(case_path, temp_case_path)

            cmd = ["wine", "tpbigm.exe", "both", temp_case_name, ".", "-R"]
            env = os.environ.copy()
            wine32_prefix = Path.home() / ".wine32"
            if wine32_prefix.exists():
                env["WINEPREFIX"] = str(wine32_prefix)

            process = subprocess.run(
                cmd,
                cwd=work_dir,
                env=env,
                capture_output=True,
                text=True,
                timeout=self.timeout_s,
                check=False
            )

            # Copy generated output files back (.lis, .dbg, .pl4)
            pl4_generated = False
            for suffix in [".lis", ".dbg", ".pl4"]:
                generated_file = work_dir / f"{temp_stem}{suffix}"
                if generated_file.exists():
                    dest_file = case_path.with_suffix(suffix)
                    shutil.copy(generated_file, dest_file)
                    if suffix == ".pl4":
                        pl4_generated = True

            if expect_pl4 and not pl4_generated:
                lis_content = ""
                lis_path = case_path.with_suffix(".lis")
                if lis_path.exists():
                    try:
                        lis_content = lis_path.read_text(errors="replace")
                    except Exception:
                        pass
                err_msg = (
                    f"ATP-EMTP execution did not produce expected .pl4 output file for {case_path.name}.\n"
                    f"ATP LIS Output:\n{lis_content[-2000:]}\n"
                    f"ATP Stdout:\n{process.stdout}\n"
                    f"ATP Stderr:\n{process.stderr}"
                )
                print(f"ERROR: {err_msg}")
                raise RuntimeError(err_msg)

        finally:
            # Clean up isolated scratch directory
            if work_dir.exists():
                try:
                    shutil.rmtree(work_dir)
                except Exception:
                    pass

        if process.returncode != 0:
            raise RuntimeError(
                f"ATP-EMTP execution failed with return code {process.returncode}:\n{process.stderr}\n{process.stdout}"
            )

        return ATPResult(
            case_path=case_path,
            output_dir=case_path.parent,
            return_code=process.returncode,
            stdout=process.stdout,
            stderr=process.stderr,
        )
