# T124–T129 architecture audit (ARCH-3 / ARCH-7 depth)

## BASE A — EXP-T113 (ARCH-3)

- Joint H/L Transformer over `[REG_H, H…, REG_L, L…]`
- Unrestricted self-attention across all valid tokens
- Readout: MEAN of REG_H and REG_L
- Encoder: full `nn.TransformerEncoder` with `n_layers` stacked layers

**Depth increase (2→3):** adds one ordinary encoder self-attention layer.
No cross-attention module exists.

## BASE B — EXP-T121 (ARCH-7)

Layer flow (unchanged scientifically):

1. Separate H and L chain embedding (+ REG)
2. Shared-weight `TransformerEncoder` self-attention stack (`n_layers`)
3. **Single** REG-only cross-attention stage:
   - REG_H queries Light residues
   - REG_L queries Heavy residues
4. MEAN merge → head

**Depth increase (2→3):** adds one encoding self-attention layer inside step 2.
Does **not** add a second REG-only cross-attention block.

## Capacity packages

| ID | d_model | layers | heads | FFN | d_head |
|----|--------:|-------:|------:|----:|-------:|
| BASE | 128 | 2 | 4 | 256 | 32 |
| DEEP | 128 | 3 | 4 | 256 | 32 |
| WIDE | 256 | 2 | 8 | 512 | 32 |
| DEEP_WIDE | 256 | 3 | 8 | 512 | 32 |

WIDE is a capacity **package** (d_model+heads+FFN together), not d_model alone.
