from __future__ import annotations

import torch


class FrequentDirections:
    def __init__(self, d: int, k: int) -> None:
        if d <= 0 or k <= 0:
            raise ValueError("d and k must be positive")
        self.d = d
        self.k = k
        self.B = torch.zeros((2 * k, d), dtype=torch.float32)
        self.next_row = 0

    def update(self, x: torch.Tensor) -> None:
        if x.ndim != 2 or x.shape[1] != self.d:
            raise ValueError("Input must be 2D with matching feature dimension")
        if x.numel() == 0:
            return
        x = x.to(dtype=torch.float32, device="cpu")
        for row in x:
            if self.next_row >= self.B.shape[0]:
                self._compress()
            self.B[self.next_row] = row
            self.next_row += 1

    def _compress(self) -> None:
        if self.next_row == 0:
            return
        B = self.B[: self.next_row]
        _, S, Vh = torch.linalg.svd(B, full_matrices=False)
        if S.numel() == 0:
            return
        if S.numel() < self.k:
            self.B.zero_()
            self.B[: S.numel()] = torch.diag(S) @ Vh
            self.next_row = S.numel()
            return
        delta = S[self.k - 1] ** 2
        S_shrink = torch.sqrt(torch.clamp(S**2 - delta, min=0))
        B_new = torch.diag(S_shrink[: self.k]) @ Vh[: self.k]
        self.B.zero_()
        self.B[: self.k] = B_new
        self.next_row = self.k

    def get_sketch(self) -> torch.Tensor:
        if self.next_row == 0:
            return self.B[:0]
        if self.next_row > self.k:
            self._compress()
        return self.B[: self.next_row].clone()
