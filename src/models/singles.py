"""
Policy / value network for SinglesBattleEnv observations.

One shared encoder per entity type (moves, Pokémon, field), a Transformer over the
12 Pokémon tokens + the field token, and pointer-style action heads: switch i is scored
from Pokémon token i, and move j from move token j of my active Pokémon.
"""

from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn

from src.observations.singles import (
    ABILITY,
    ACTIVE,
    FIELD_FEATS,
    ITEM,
    LAST_MOVE,
    MOVE_FEATS,
    N_MOVES,
    N_TEAM,
    POKEMON_FEATS,
    SPECIES,
)
from src.observations.vocab import PAD, Vocab

SPECIES_DIM = 64
ITEM_DIM = 16
ABILITY_DIM = 16
MOVE_DIM = 32


def mlp(in_dim: int, out_dim: int, hidden_dim: int | None = None) -> nn.Sequential:
    hidden_dim = hidden_dim or out_dim
    return nn.Sequential(
        nn.Linear(in_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, out_dim)
    )


class SinglesPolicy(nn.Module):
    def __init__(
        self,
        vocab: Vocab,
        n_actions: int,
        d_model: int = 128,
        d_move: int = 64,
        n_layers: int = 2,
        n_heads: int = 4,
    ):
        super().__init__()
        # Saved with the weights, to rebuild the model in load_policy
        self.hparams = {
            "n_actions": n_actions,
            "d_model": d_model,
            "d_move": d_move,
            "n_layers": n_layers,
            "n_heads": n_heads,
        }
        # Actions are 6 switches, then 4 moves per gimmick option (none, mega, z, dynamax, tera)
        self.n_gimmicks = (n_actions - N_TEAM) // N_MOVES
        assert N_TEAM + N_MOVES * self.n_gimmicks == n_actions

        self.species_emb = nn.Embedding(vocab.n_species, SPECIES_DIM, padding_idx=PAD)
        self.item_emb = nn.Embedding(vocab.n_items, ITEM_DIM, padding_idx=PAD)
        self.ability_emb = nn.Embedding(vocab.n_abilities, ABILITY_DIM, padding_idx=PAD)
        # Shared by the move slots and each Pokémon's last move
        self.move_emb = nn.Embedding(vocab.n_moves, MOVE_DIM, padding_idx=PAD)

        self.move_encoder = mlp(MOVE_DIM + MOVE_FEATS, d_move)
        self.pokemon_encoder = mlp(
            SPECIES_DIM + ITEM_DIM + ABILITY_DIM + MOVE_DIM + POKEMON_FEATS + d_move,
            d_model,
        )
        self.field_encoder = mlp(FIELD_FEATS, d_model)

        layer = nn.TransformerEncoderLayer(
            d_model,
            n_heads,
            dim_feedforward=4 * d_model,
            dropout=0.0,
            batch_first=True,
            norm_first=True,
        )
        self.trunk = nn.TransformerEncoder(layer, n_layers, enable_nested_tensor=False)

        self.switch_head = nn.Linear(d_model, 1)
        self.move_head = mlp(d_move + 2 * d_model, self.n_gimmicks, hidden_dim=d_model)
        self.value_head = nn.Linear(d_model, 1)

    def forward(
        self, obs: dict[str, Tensor], action_mask: Tensor | None = None
    ) -> tuple[Tensor, Tensor]:
        """
        Takes a batch of observations (see to_tensors) and optionally the action mask,
        and returns the logits (B, n_actions), with illegal actions masked, and values (B,).
        """
        pokemon_ids = obs["pokemon_ids"].long()  # (B, 12, 4)
        pokemon_feats = obs["pokemon_feats"].float()  # (B, 12, P)
        move_ids = obs["move_ids"].long()  # (B, 12, 4)
        move_feats = obs["move_feats"].float()  # (B, 12, 4, M)
        field_feats = obs["field_feats"].float()  # (B, F)

        move_tokens = self.move_encoder(
            torch.cat([self.move_emb(move_ids), move_feats], dim=-1)
        )  # (B, 12, 4, d_move)
        move_slots = (move_ids != PAD).unsqueeze(-1).float()
        moves_summary = (move_tokens * move_slots).sum(2) / move_slots.sum(2).clamp(
            min=1
        )

        pokemon_tokens = self.pokemon_encoder(
            torch.cat(
                [
                    self.species_emb(pokemon_ids[..., SPECIES]),
                    self.item_emb(pokemon_ids[..., ITEM]),
                    self.ability_emb(pokemon_ids[..., ABILITY]),
                    self.move_emb(pokemon_ids[..., LAST_MOVE]),
                    pokemon_feats,
                    moves_summary,
                ],
                dim=-1,
            )
        )  # (B, 12, d_model)
        field_token = self.field_encoder(field_feats).unsqueeze(1)  # (B, 1, d_model)

        empty_rows = pokemon_ids[..., SPECIES] == PAD
        padding = torch.cat([empty_rows, torch.zeros_like(empty_rows[:, :1])], dim=1)
        h = self.trunk(
            torch.cat([pokemon_tokens, field_token], dim=1),
            src_key_padding_mask=padding,
        )  # (B, 13, d_model)

        mine, theirs = h[:, :N_TEAM], h[:, N_TEAM : 2 * N_TEAM]
        # (B, 6) one-hot of the active row, all zeros if there's none
        my_active_row = pokemon_feats[:, :N_TEAM, ACTIVE]
        their_active_row = pokemon_feats[:, N_TEAM:, ACTIVE]
        my_active = torch.einsum("bn,bnd->bd", my_active_row, mine)
        their_active = torch.einsum("bn,bnd->bd", their_active_row, theirs)
        active_moves = torch.einsum(
            "bn,bnmd->bmd", my_active_row, move_tokens[:, :N_TEAM]
        )  # (B, 4, d_move)

        switch_logits = self.switch_head(mine).squeeze(-1)  # (B, 6)
        context = torch.cat([my_active, their_active], dim=-1)
        move_logits = self.move_head(
            torch.cat(
                [active_moves, context.unsqueeze(1).expand(-1, N_MOVES, -1)], dim=-1
            )
        )  # (B, 4, n_gimmicks)
        # Move actions are 6 + 4 * gimmick + slot
        logits = torch.cat(
            [switch_logits, move_logits.transpose(1, 2).flatten(1)], dim=1
        )

        if action_mask is not None:
            logits = logits.masked_fill(action_mask == 0, -1e9)
        value = self.value_head(h[:, -1]).squeeze(-1)
        return logits, value


def to_tensors(
    observations: list[dict[str, np.ndarray]], device: torch.device | str | None = None
) -> dict[str, Tensor]:
    """
    Stacks env observations (the "observation" part of each step) into a batch.
    """
    return {
        key: torch.as_tensor(
            np.stack([obs[key] for obs in observations]), device=device
        )
        for key in observations[0]
    }


def save_policy(policy: SinglesPolicy, path: str | Path, **extra: Any):
    """
    Saves the weights and what's needed to rebuild the model, plus any extra entries
    (e.g. the optimizer state).
    """
    torch.save({"hparams": policy.hparams, "model": policy.state_dict(), **extra}, path)


def load_policy(
    path: str | Path,
    vocab: Vocab | None = None,
    device: torch.device | str = "cpu",
) -> SinglesPolicy:
    """
    Loads a policy saved with save_policy. The vocab must be the one it was trained with.
    """
    checkpoint = torch.load(path, map_location=device)
    policy = SinglesPolicy(vocab or Vocab.load(), **checkpoint["hparams"]).to(device)
    policy.load_state_dict(checkpoint["model"])
    return policy
