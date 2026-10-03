# Third-party notices

* **FlashAttention-4 (CuTe DSL), 4.0.0b32** — `veda_comfy/kernels/fa4` is a
  fork of the PyPI wheel `flash-attn-4==4.0.0b32`, carrying Veda's block-sparse
  and FP8 work for the SM80-family and SM120 kernels. BSD-3-Clause, copyright
  Tri Dao and contributors; see `veda_comfy/kernels/fa4/LICENSE` and `AUTHORS`.
  The SM8x / SM120 block-sparse changes originate in Miowtion (BSD-3-Clause).
* **QuACK (`quack-kernels`), 0.6.5** — `veda_comfy/kernels/quack` is a fork of
  the modules FA4 needs, with Windows compatibility edits (optional `fcntl`).
  Apache-2.0; see `veda_comfy/kernels/quack/LICENSE`.
* Every change we made to both is recorded in
  `veda_comfy/kernels/UPSTREAM.diff`.
* **Miowtion** (<https://github.com/veda-sparse/Miowtion>, MIT) — the Veda
  tiling, plan, predictor and selection rules in `veda_comfy/core` are a
  rewrite of `miowtion/veda`; the FA4 backend integration follows
  `miowtion/kernels/fa4.py`.
* **Veda predictor weights** (downloaded at run time, not shipped here) —
  MiniMax H3 Community License, inherited from MiniMax-H3.
* **Icon** — the Miowtion logo (`assets/icon.svg`), MIT.
