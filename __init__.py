"""ComfyUI entry point of Veda-on-ComfyUI (Veda sparse attention for
MiniMax-H3). All code lives in `veda_comfy/`; see README.md."""

# ComfyUI imports this directory as a package. pytest also imports this
# file (the repo root is a package directory) but without a parent package,
# where the relative import cannot work and is not needed.
if __package__:
    from .veda_comfy.nodes import comfy_entrypoint

    __all__ = ['comfy_entrypoint']
