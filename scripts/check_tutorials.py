"""Execute the tutorial notebooks and report any cell that fails.

Run this before cutting a release. The notebooks are the most visible thing openDVP ships and
nothing else exercises them: `nb_execution_mode` is `"off"`, so the docs build renders their
committed outputs without ever running the code.

    uv run --all-extras --group docs python scripts/check_tutorials.py

Cells tagged `skip-execution` are reported and skipped. That is how the two interactive napari
cells are handled — they need a real display and kill the kernel headless.

Outputs are never written back, so this cannot quietly churn the notebooks. Refreshing the
published outputs is a separate, manual job: run them in Jupyter with the viewer cells live.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# T3 reads the checkpoint T2 writes, so the order matters.
NOTEBOOKS = [
    "T1_ImageAnalysis.ipynb",
    "T2_DownstreamProteomics.ipynb",
    "T3_ProteomicsIntegration.ipynb",
]
SKIP_TAG = "skip-execution"
TIMEOUT = 3600


def check(path: Path) -> list[str]:
    """Execute one notebook, returning a description of each cell that errored."""
    import nbformat
    from nbclient import NotebookClient
    from nbclient.exceptions import DeadKernelError

    notebook = nbformat.read(path, as_version=4)
    kept, skipped = [], []
    for index, cell in enumerate(notebook.cells):
        if SKIP_TAG in cell.get("metadata", {}).get("tags", []):
            skipped.append((index, "".join(cell["source"]).strip().splitlines()[-1][:60]))
        else:
            kept.append(cell)
    notebook.cells = kept

    for index, line in skipped:
        print(f"  skipped cell {index}: {line}")

    client = NotebookClient(notebook, timeout=TIMEOUT, kernel_name="python3", allow_errors=True)
    try:
        client.execute()
    except DeadKernelError:
        return [f"{path.name}: the kernel died, so execution stopped part-way"]

    failures = []
    for index, cell in enumerate(notebook.cells):
        for output in cell.get("outputs", []):
            if output.get("output_type") == "error":
                first = "".join(cell["source"]).strip().splitlines()[0][:70]
                failures.append(f"{path.name} cell {index} ({first}): {output['ename']}: {output['evalue'][:200]}")
    return failures


def main() -> int:
    """Execute every tutorial in order and summarise the result."""
    # keep matplotlib off any GUI backend, and let the notebooks find a cached dataset
    os.environ.setdefault("MPLBACKEND", "Agg")
    if "OPENDVP_DATA_DIR" not in os.environ:
        print("OPENDVP_DATA_DIR is not set; using the default pooch cache (downloads 133 MB once)\n")

    directory = Path(__file__).resolve().parent.parent / "docs" / "Tutorials"
    failures = []
    for name in NOTEBOOKS:
        print(f"== {name}")
        # the notebooks write to ../outputs, relative to their own directory
        os.chdir(directory)
        found = check(directory / name)
        failures.extend(found)
        print(f"  {'FAILED' if found else 'ok'}\n")

    if failures:
        print(f"{len(failures)} failing cell(s):")
        for failure in failures:
            print(f"  - {failure}")
        return 1
    print("all tutorials executed cleanly")
    return 0


if __name__ == "__main__":
    sys.exit(main())
