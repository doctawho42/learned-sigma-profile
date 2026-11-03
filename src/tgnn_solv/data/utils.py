"""
Core data utilities: SMILES handling, file download, paths.
"""

from pathlib import Path
from typing import Optional

from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold

try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False

# ------------------------------------------------------------------ #
#  Directory layout                                                   #
# ------------------------------------------------------------------ #

def _find_repo_root() -> Path:
    """Find the repository root without depending on the current working directory."""
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "pyproject.toml").exists() and (parent / "src").exists():
            return parent
    return Path.cwd().resolve()


# The repository-standard dataset layout lives under notebooks/data so that
# the CLI scripts, notebooks, tests, and documentation all point to the same
# raw and processed files by default.
REPO_ROOT = _find_repo_root()
DATA_DIR = REPO_ROOT / "notebooks" / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

for _d in [DATA_DIR, RAW_DIR, PROCESSED_DIR]:
    _d.mkdir(parents=True, exist_ok=True)


# ------------------------------------------------------------------ #
#  SMILES utilities                                                   #
# ------------------------------------------------------------------ #

def canonicalize(smi: str) -> Optional[str]:
    """
    Canonicalize a SMILES string.

    Returns None if the input is invalid or empty.
    """
    if not isinstance(smi, str) or not smi.strip():
        return None
    try:
        mol = Chem.MolFromSmiles(smi.strip())
        if mol is None:
            return None
        return Chem.MolToSmiles(mol, canonical=True)
    except Exception:
        return None


def get_scaffold(smi: str) -> Optional[str]:
    """
    Murcko scaffold of a molecule (for splitting).

    Returns canonical SMILES of the scaffold, or None on failure.
    """
    try:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            return None
        scaf = MurckoScaffold.GetScaffoldForMol(mol)
        return Chem.MolToSmiles(scaf, canonical=True)
    except Exception:
        return None


def scaffold_key(smi: str) -> Optional[str]:
    """Leak-guard key: Murcko scaffold, or canonical SMILES for acyclic molecules.

    ``get_scaffold`` returns an empty string for acyclic molecules (no ring
    system), which silently bypasses scaffold-exclusion. This returns a non-empty
    key for every parseable molecule so acyclic-vs-acyclic leakage is still
    caught; ``None`` only when RDKit cannot parse the SMILES.
    """
    scaf = get_scaffold(smi)
    if scaf:
        return scaf
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, canonical=True)


# ------------------------------------------------------------------ #
#  File download                                                      #
# ------------------------------------------------------------------ #

def download_file(url: str, path: Path, desc: str = "") -> bool:
    """
    Download a file if it does not already exist.

    Returns True if the file is available (existed or downloaded).
    """
    if path.exists():
        size_mb = path.stat().st_size / 1e6
        print(f"  Already exists: {path.name} ({size_mb:.1f} MB)")
        return True

    if not HAS_REQUESTS:
        print("  Install `requests` or manually download:")
        print(f"    URL:  {url}")
        print(f"    Save: {path}")
        return False

    print(f"  Downloading {desc or path.name}...")
    try:
        resp = requests.get(url, stream=True, timeout=120)
        resp.raise_for_status()
        total = int(resp.headers.get("content-length", 0))
        downloaded = 0
        with open(path, "wb") as f:
            for chunk in resp.iter_content(8192):
                f.write(chunk)
                downloaded += len(chunk)
                if total:
                    print(f"\r    {downloaded / total * 100:.0f}%", end="")
        print(f"\n    Saved ({path.stat().st_size / 1e6:.1f} MB)")
        return True
    except Exception as e:
        print(f"    Failed: {e}")
        if path.exists():
            path.unlink()
        return False


def verify_csv(path: Path) -> bool:
    """
    Check that a downloaded file is actually CSV, not an HTML error page.

    Returns False and deletes the file if it looks like HTML.
    """
    if not path.exists():
        return False
    try:
        with open(path, "r", errors="ignore") as f:
            first_line = f.readline(512).lower()
        if "<html" in first_line or "<!doctype" in first_line:
            print(f"  File {path.name} is HTML, not CSV — deleting")
            path.unlink()
            return False
    except Exception:
        return False
    return True
