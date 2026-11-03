"""
Thermometer temperature encoding.

The temperature range [T_min, T_max] is divided into N_bins equal
intervals.  For a given temperature T:
  - All bins below T are filled with 1.0
  - The bin containing T gets a fractional fill
  - All bins above T are 0.0

Example (N_bins=5, range=[250, 350]):
  Each bin covers 20 K: [250,270), [270,290), [290,310), [310,330), [330,350]

  T = 298 K falls in bin 2 [290, 310):
    fraction = (298 - 290) / 20 = 0.4
    encoding = [1.0, 1.0, 0.4, 0.0, 0.0]

  T = 340 K falls in bin 4 [330, 350]:
    fraction = (340 - 330) / 20 = 0.5
    encoding = [1.0, 1.0, 1.0, 1.0, 0.5]

This preserves ordinal structure (hotter → more 1s) and gives
smooth gradients within each bin.
"""

import torch


class ThermometerEncoder:
    """
    Thermometer-style temperature encoding.

    Parameters
    ----------
    n_bins : int
        Number of bins (default 20).
    T_min : float
        Lower bound of temperature range (K).
    T_max : float
        Upper bound of temperature range (K).
    """

    def __init__(
            self,
            n_bins: int = 20,
            T_min: float = 200.0,
            T_max: float = 500.0,
    ) -> None:
        self.n_bins = n_bins
        self.T_min = T_min
        self.T_max = T_max
        self.bin_width = (T_max - T_min) / n_bins

    def encode(self, T: torch.Tensor) -> torch.Tensor:
        """
        Encode temperature tensor.

        Parameters
        ----------
        T : (B,) temperature values in Kelvin

        Returns
        -------
        (B, n_bins) thermometer encoding
        """
        # Clamp to range
        T_clamped = T.clamp(self.T_min, self.T_max)

        # Fractional bin index: 0.0 at T_min, n_bins at T_max
        frac_idx = (T_clamped - self.T_min) / self.bin_width  # (B,)

        # Build encoding
        # bin_indices: [0, 1, 2, ..., n_bins-1]
        bins = torch.arange(
            self.n_bins, dtype=T.dtype, device=T.device
        ).unsqueeze(0)  # (1, n_bins)

        frac_idx = frac_idx.unsqueeze(-1)  # (B, 1)

        # Bins fully below T → 1.0
        # Bin containing T → fractional fill
        # Bins above T → 0.0
        encoding = (frac_idx - bins).clamp(0.0, 1.0)

        return encoding  # (B, n_bins)

    @property
    def dim(self) -> int:
        return self.n_bins
