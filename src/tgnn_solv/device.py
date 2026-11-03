"""One home for turning a requested device string into a `torch.device`.

Twelve scripts each carried a copy of this function, and they gave three different
answers to "you named an accelerator this box does not have". `a569307` changed exactly
one of them -- it touched two files, `scripts/train.py` and
`scripts/experiments/run_e5_sigma_grounding.sh` -- so that an accelerator asked for by
name and not delivered is an error rather than a warning. Of the eleven it left:

- Six, `train_directgnn.py` (the DirectGNN control arm) among them, printed one
  `WARNING:` line and ran on the CPU.
- `scripts/analysis/export_checkpoint_predictions.py`, the prediction exporter seven of
  the nine `scripts/experiments` drivers call, did the same with no line at all.
- The four `scripts/analysis` `_resolve_device` copies never fell back on CUDA: they
  handed the string to `torch.device` and died at the first `.to(device)`. They fell
  back only for MPS, behind a bracketed tag line rather than a `WARNING:` one.

This module is one behaviour, the strict one. All twelve import it, as do the four
scripts that had been reaching a copy through another script's module namespace.
`tests/test_device.py` used to name both sets; it scans the tree instead, and the rules
it scans by -- stated at the top of that file -- are the authority for what is
protected. Neither a list nor a count is: the list could not know about a script written
after it, and two AST scans written a day apart disagreed on the total. Many other
entry points take a `--device` flag without coming
through here. A minority hand it to a child process that does; most build a
`torch.device` in the same process and get no check at all. Five are worse than
unchecked: `run_pka_flip_certify.py`, `run_pka_lambda_frontier.py` and `run_solprop.py`
rewrite a named `--device cuda` to CPU with no message, and `run_pka_n_sweep.py` and
`run_pka_flip_competence.py` accept the flag and then hardcode CPU regardless.

Two knobs, both because the copies genuinely disagreed:

- `allow_tf32_from_env`: the two training scripts honoured `TGNN_MATMUL_TF32`, the
  other ten did not, and turning TF32 on underneath an evaluation would move reported
  metrics whenever that variable happened to be exported for training. It defaults to
  off and the training scripts opt in.
- `"auto"`: four of the five `scripts/analysis` copies accepted it and the other eight
  would have handed it to `torch.device("auto")` and died. It is accepted here, and
  means the same thing `default_device(prefer_mps=True)` returns.

`default_device` is not a repair of an older regression; it is a precondition of this
one. Raising on an undelivered accelerator is right when the *user* named it and wrong
when argparse did, and eleven of the thirteen entry points that now call it hardcoded a
bare `default="cuda"` (or `"mps"`) that argparse hands over with no flag typed. Making
the raise universal is what would kill the DirectGNN control arm on a MacBook and the
DirectGNN error diagnostics on every box that is not a Mac. Neither died after
`a569307`: at `a569307` `train_directgnn.py` still had its own lenient copy, and the
diagnostics imported the equally lenient one from `run_full_budget_experiment`.

Those eleven now read the machine here, alongside `scripts/train.py` and
`run_medium_budget_comparison.py`, which already asked it in two ad-hoc spellings.
Two entry points keep a literal accelerator default on purpose:
`scripts/cloud/kaggle_run.py`, because a Kaggle session that quietly runs on CPU is a
wasted session and it answers for that with its own kernel-launch preflight and
`--allow-cpu`, and `scripts/experiments/run_pka_trained_comparison.py`, which never
reaches this resolver. Those two are the whole of
`LITERAL_ACCELERATOR_DEFAULT_EXEMPTIONS` in `tests/test_device.py`, asserted in both
directions so that a third has to be argued for rather than read as intentional for
already being on a list. An argparse default has to be a statement about the machine,
not a wish.

Reading the machine is silent in one direction only. `default_device` returning `"cuda"`
needs no announcement; returning `"cpu"` is either unremarkable or the whole 2026-08-08
failure, and the two are told apart by asking whether the box has an NVIDIA GPU at all.
A MacBook has none, and prints nothing. A gate box whose kernel module went missing
still has its V100 sitting on the PCI bus, and gets a block of tagged lines between
rules, on stderr, once per process. Before this, the no-flag path carried *less* signal
than it had on the day: `default_device` chose CPU without a word, the subprocess
drivers forwarded `--device cpu`, and `resolve_device("cpu")` is a legitimate request
that says nothing either -- so the only tell left was one `Device: cpu` line inside a
long banner, and that is the line that was missed on the day. `TGNN_ALLOW_CPU_FALLBACK=1`
silences the announcement as well, since it already means "a CPU run is what I meant"
(`--device cpu` cannot silence it: an argparse default is computed before the arguments
are parsed).
"""

from __future__ import annotations

import os
import sys
import textwrap

import torch

_TRUTHY = ("1", "true", "yes", "on")
_TF32_TRUTHY = ("1", "true", "high", "yes", "on")

# Where the PCI bus is legible without asking a driver, a library or a subprocess:
# sysfs, which is a directory of one-line files. The failure this detects is precisely
# a card with no usable driver, so every richer source -- nvidia-smi, /dev/nvidia*,
# /proc/driver/nvidia, libcuda via ctypes, torch.cuda's own device count -- is either
# absent in exactly that state or costs more than an argparse default may spend.
_PCI_DEVICE_ROOT = "/sys/bus/pci/devices"
_NVIDIA_PCI_VENDOR = "0x10de"
# PCI base class 03 is "display controller": 0x030000 VGA, 0x030200 3D controller (the
# data-centre cards). A card's own HDMI-audio (0x0403xx) and USB-C (0x0c03xx) functions
# carry the same vendor id and would otherwise be counted as extra GPUs.
_DISPLAY_PCI_CLASS = "0x03"

# Once per process. The announcement is worth a block of stderr on the run that needed
# it; it is worth nothing on the fifth parser built inside the same run.
_CPU_DEFAULT_ANNOUNCED = False


def _env_is_set(name: str, truthy: tuple[str, ...] = _TRUTHY) -> bool:
    """Return True when environment variable `name` holds an affirmative value."""
    return os.environ.get(name, "").lower() in truthy


def _read_sysfs(path: str) -> str:
    """Read a one-line sysfs attribute, or "" if it is missing or unreadable."""
    try:
        with open(path) as handle:
            return handle.read().strip().lower()
    except OSError:
        return ""


def nvidia_gpus_on_the_bus(root: str | None = None) -> int:
    """Count NVIDIA display devices on the PCI bus, by reading sysfs and nothing else.

    This is the question "is this box a machine that should have had CUDA?", and it is
    deliberately not the question `torch.cuda.is_available()` answers. A box that lost
    its kernel module answers no to the second and yes to this one; a MacBook, which has
    no `/sys` at all, answers no to both. The cost is one `listdir` and two small reads
    per device -- microseconds, no import, no subprocess -- and it is only paid on the
    path that has already decided on CPU.

    A missing root, an unreadable attribute or a non-NVIDIA vendor all fail closed to
    "not counted": a bad guess here must never be able to break device selection, and
    the consequence of undercounting is the silence that was there before.

    Args:
        root: The sysfs PCI directory to read. `None` means `_PCI_DEVICE_ROOT`, looked
            up on each call rather than frozen into a default argument, so that a test
            can put a directory of fake devices where the bus is.
    """
    if root is None:
        root = _PCI_DEVICE_ROOT
    try:
        entries = sorted(os.listdir(root))
    except OSError:
        return 0
    found = 0
    for entry in entries:
        if _read_sysfs(os.path.join(root, entry, "vendor")) != _NVIDIA_PCI_VENDOR:
            continue
        pci_class = _read_sysfs(os.path.join(root, entry, "class"))
        # An unreadable class is counted: the vendor already matched, and this
        # detector's failure mode should be a false banner, not a false silence.
        if pci_class and not pci_class.startswith(_DISPLAY_PCI_CLASS):
            continue
        found += 1
    return found


def _announce_cpu_default() -> None:
    """Shout, on stderr, when CPU was chosen on a box that owns an NVIDIA GPU.

    Loud where it matters and absent everywhere else. Every line carries the same
    `TGNN-DEVICE` tag so the block is one `grep` in a 300 KB log and survives being
    interleaved with tqdm (which writes to stderr too, so any log that has the progress
    bars in it has this). A laptop never sees it and so cannot learn to ignore it.
    """
    global _CPU_DEFAULT_ANNOUNCED
    if _CPU_DEFAULT_ANNOUNCED or _env_is_set("TGNN_ALLOW_CPU_FALLBACK"):
        return
    count = nvidia_gpus_on_the_bus()
    if count == 0:
        return
    _CPU_DEFAULT_ANNOUNCED = True
    # Which of the two causes it is, for the price of an attribute lookup. They need
    # different repairs and both present as this same silence.
    cuda_build = getattr(torch.version, "cuda", None)
    if cuda_build is None:
        diagnosis = (
            "This torch has no CUDA in it at all (torch.version.cuda is None), so the "
            "wheel is the thing to fix and not the driver: reinstall from the CUDA "
            "wheel index."
        )
    else:
        diagnosis = (
            f"torch was built against CUDA {cuda_build} and still reports "
            f"is_available() False, which is what a missing or mismatched kernel "
            f"module looks like: try `modinfo nvidia`."
        )
    plural = "" if count == 1 else "s"
    body = (
        f'The --device default was read off this box as "cpu" -- but {count} NVIDIA '
        f"display device{plural} {'is' if count == 1 else 'are'} on its PCI bus. "
        f"The card is here; CUDA is not usable. Check `nvidia-smi` first. "
        f"{diagnosis} "
        f"On 2026-08-08 this exact state cost the gate box ten hours at ~15 s/it "
        f"against ~1 s/it on its V100, with no error and no wrong number. "
        f"This is the argparse default, computed before your flags are parsed: it "
        f"refuses nothing, and a --device cuda you typed yourself still raises. "
        f"Set TGNN_ALLOW_CPU_FALLBACK=1 if a CPU run is what you meant."
    )
    rule = "=" * 78
    print(rule, file=sys.stderr)
    # break_on_hyphens=False: this text is mostly `--device`, `nvidia-smi` and
    # `~15 s/it`, and a wrap that splits those leaves an operator grepping for a
    # string the log does not contain.
    for line in textwrap.wrap(body, width=62, break_on_hyphens=False):
        print(f"TGNN-DEVICE  {line}", file=sys.stderr)
    print(rule, file=sys.stderr)


def default_device(*, prefer_mps: bool = False) -> str:
    """The device string to use as an argparse default: what this box actually has.

    A hardcoded `default="cuda"` turns "the user did not say" into "the user demanded an
    accelerator", which `resolve_device` is now obliged to treat as fatal. Call this
    instead, so that no-flag-typed means the best device present.

    Args:
        prefer_mps: Whether Apple MPS counts as an acceleration to fall back to when
            CUDA is absent. False -- CUDA else CPU -- for the training entry points,
            matching the default `scripts/train.py` has had since `da35f1b` added it;
            several of them run graph ops that MPS has historically been the wrong
            place for, and a local smoke run wants to be boring rather than fast. True
            for the two that hardcoded `default="mps"` and were plainly written on a
            Mac (`export_checkpoint_predictions.py`, the DirectGNN error diagnostics),
            and for `run_medium_budget_comparison.py`, whose own ladder already fell
            through to MPS.

    Returning `"cpu"` announces itself on stderr, but only on a box that owns an NVIDIA
    GPU -- see `_announce_cpu_default`. A machine with no accelerator to lose says
    nothing, which is what keeps the announcement worth reading on the machine that has
    one. Nothing is announced on the accelerator paths, and the bus is not read there.
    """
    if torch.cuda.is_available():
        return "cuda"
    if prefer_mps and torch.backends.mps.is_available():
        return "mps"
    _announce_cpu_default()
    return "cpu"


def resolve_device(
    device_str: str,
    *,
    allow_tf32_from_env: bool = False,
) -> torch.device:
    """Resolve a requested device, refusing to substitute CPU for an accelerator.

    The fallback used to be silent-ish: one WARNING line, then 300 KB of tqdm output on
    top of it. On 2026-08-08 a kernel upgrade left the gate box without its NVIDIA
    module, six `--device cuda` runs fell through to CPU, and they trained for ten hours
    at 15 s/it against ~1 s/it on the V100 -- roughly two months to a result -- without
    a single error line. An accelerator asked for by name and quietly not delivered is
    never what the caller wanted, so it is an error now. Set TGNN_ALLOW_CPU_FALLBACK=1
    to restore the old behaviour for a smoke run.

    Args:
        device_str: The device as requested on the command line ("cuda", "cuda:1",
            "mps", "cpu", or "auto"). Case and surrounding space are ignored when
            deciding whether the device is available; the string itself is then handed
            to `torch.device` unchanged, so "  CPU " raises there as it always has.
            "auto" never raises -- it is a question about the box, not a demand.
        allow_tf32_from_env: Whether `TGNN_MATMUL_TF32` may switch this process to TF32
            matmul once a CUDA device is resolved. True only in the training scripts;
            an evaluation must not silently change precision under an environment
            variable that was exported for a training run.
    """
    requested = device_str.strip().lower()
    if requested == "auto":
        device_str = default_device(prefer_mps=True)
        requested = device_str
    allow_fallback = _env_is_set("TGNN_ALLOW_CPU_FALLBACK")
    unavailable = None
    if requested.startswith("cuda") and not torch.cuda.is_available():
        unavailable = "CUDA"
    elif requested == "mps" and not torch.backends.mps.is_available():
        unavailable = "MPS"
    if unavailable is not None:
        if not allow_fallback:
            raise RuntimeError(
                f"{unavailable} was requested (--device {device_str}) but is not available. "
                f"Refusing to run on CPU instead: on this corpus that is a ~15x slowdown "
                f"that produces no error and no wrong number, only a run that never finishes. "
                f"Pass --device cpu if CPU is what you meant, or set "
                f"TGNN_ALLOW_CPU_FALLBACK=1 to fall back silently."
            )
        print(f"WARNING: {unavailable} requested but unavailable; falling back to CPU.")
        return torch.device("cpu")
    dev = torch.device(device_str)
    if dev.type == "cuda" and allow_tf32_from_env and _env_is_set(
        "TGNN_MATMUL_TF32", _TF32_TRUTHY
    ):
        # TF32 matmul: large speedup on Ampere+/Blackwell (a no-op on T4, which has
        # no TF32), numerically negligible for training. Opt-in so default FP32 runs
        # stay bit-comparable across seeds.
        torch.set_float32_matmul_precision("high")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        print("   TF32 matmul enabled (TGNN_MATMUL_TF32).")
    return dev
