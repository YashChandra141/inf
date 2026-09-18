from __future__ import annotations

import torch


class KVCache:
    """Preallocated KV cache with slot-based sequence ownership.

    Layout: [layers, max_batch, n_kv_heads, max_seq_len, head_dim]
    """

    def __init__(
        self,
        num_layers: int,
        max_batch_size: int,
        num_kv_heads: int,
        max_seq_len: int,
        head_dim: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        self.max_batch_size = max_batch_size
        self.max_seq_len = max_seq_len
        self.k = torch.zeros(
            num_layers,
            max_batch_size,
            num_kv_heads,
            max_seq_len,
            head_dim,
            device=device,
            dtype=dtype,
        )
        self.v = torch.zeros_like(self.k)
        self.lengths = torch.zeros(max_batch_size, device=device, dtype=torch.long)

    def reset_slot(self, slot: int) -> None:
        self.lengths[slot] = 0
        self.k[:, slot].zero_()
        self.v[:, slot].zero_()

    def update(
        self,
        layer_idx: int,
        key: torch.Tensor,
        value: torch.Tensor,
        positions: torch.Tensor,
        slots: torch.Tensor,
    ) -> None:
        """Write key/value at `positions` into the slots.

        key, value: [B, n_kv, T, D]
        positions: [B, T]
        slots: [B]
        """
        batch = key.shape[0]
        for i in range(batch):
            slot = int(slots[i].item())
            pos = positions[i]
            self.k[layer_idx, slot, :, pos, :] = key[i]
            self.v[layer_idx, slot, :, pos, :] = value[i]
            if layer_idx == 0:
                self.lengths[slot] = int(pos[-1].item()) + 1

    def gather(self, layer_idx: int, slots: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.k[layer_idx].index_select(0, slots), self.v[layer_idx].index_select(0, slots)
