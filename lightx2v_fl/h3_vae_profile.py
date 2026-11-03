"""Opt-in, process-owned FlagGems profile for MiniMax-H3 inference."""

from contextlib import contextmanager
from functools import wraps
from threading import RLock

VAE_EXCLUSIONS = ("linear", "addmm", "addmm_out", "_scaled_dot_product_flash_attention")


class H3VAEGemsProfile:
    """Use native VAE linears/attention while preserving H3's input projection.

    Construct the pipeline with LIGHTX2V_FL_GLOBAL_GEMS=0, then enter this
    context. This context owns global registration for all requests. Its VAE
    scope uses the same sensitive RMS kernel without its Autograd wrapper and
    calls FlagGems directly for
    decoder.proj_in, whose arithmetic must match the existing profile.
    Requests on this runner are serialized. Do not cohost other FlagGems
    users in the same process while this context is active.
    """

    def __init__(self, runner, *, base_exclusions):
        config = getattr(runner, "config", {})
        if (
            config.get("model_cls") != "minimax_h3"
            or not callable(getattr(runner, "run_request", None))
            or not callable(getattr(runner, "run_vae_decoder", None))
        ):
            raise TypeError("an initialized MiniMax-H3 runner is required")
        try:
            self._projection = runner.video_vae.decoder.proj_in
            self._projection.weight
            self._projection.bias
            self._projection.forward
        except AttributeError as error:
            raise TypeError("MiniMax-H3 video VAE input projection is required") from error
        import torch
        self._norms = [module for module in runner.video_vae.decoder.modules()
                       if isinstance(module, torch.nn.RMSNorm)]
        self.runner = runner
        self.base = list(dict.fromkeys(base_exclusions))
        self.vae = list(dict.fromkeys(self.base + list(VAE_EXCLUSIONS)))
        self._context = None
        self._lock = RLock()
        self.active = False

    def _switch(self, exclusions):
        import flag_gems

        if self._context is not None:
            current, self._context = self._context, None
            current.__exit__(None, None, None)
        current = flag_gems.use_gems(exclude=exclusions)
        current.__enter__()
        self._context = current

    def __enter__(self):
        import flag_gems
        from flag_gems.ops._fused_rms_norm import _fused_rms_norm_forward

        self._rms_forward = _fused_rms_norm_forward

        if self.active or getattr(flag_gems, "current_work_registrar", None) is not None:
            raise RuntimeError("this profile must own global FlagGems registration")
        self._request = self.runner.run_request
        self._decode = self.runner.run_vae_decoder
        self._switch(self.base)
        self.active = True

        @wraps(self._request)
        def request(*args, **kwargs):
            with self._lock:
                return self._request(*args, **kwargs)

        @wraps(self._decode)
        def decode(*args, **kwargs):
            import torch

            with self._lock:
                torch.cuda.synchronize()
                original = self._projection.forward
                norm_originals = [(module, module.forward) for module in self._norms]

                def projection(value):
                    return flag_gems.linear(
                        value, self._projection.weight, self._projection.bias
                    )

                try:
                    self._switch(self.vae)
                    self._projection.forward = projection
                    for module, _ in norm_originals:
                        def rms(value, _module=module):
                            if torch.is_grad_enabled():
                                raise RuntimeError("H3 VAE profile requires inference without gradients")
                            eps = _module.eps if _module.eps is not None else 1e-5
                            return self._rms_forward(
                                value, _module.normalized_shape, _module.weight, eps
                            )[0]
                        module.forward = rms
                    return self._decode(*args, **kwargs)
                finally:
                    self._projection.forward = original
                    for module, forward in norm_originals:
                        module.forward = forward
                    try:
                        torch.cuda.synchronize()
                    finally:
                        self._switch(self.base)

        self.runner.run_request = request
        self.runner.run_vae_decoder = decode
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        import flag_gems

        with self._lock:
            self.runner.run_request = self._request
            self.runner.run_vae_decoder = self._decode
            self.active = False
            try:
                if self._context is not None:
                    current, self._context = self._context, None
                    current.__exit__(exc_type, exc_value, traceback)
            finally:
                flag_gems.current_work_registrar = None

    @contextmanager
    def current_only(self):
        """Run a control request with the base registration throughout."""
        with self._lock:
            if not self.active:
                raise RuntimeError("the profile must be active")
            scoped = self.runner.run_vae_decoder
            self.runner.run_vae_decoder = self._decode
            try:
                yield
            finally:
                self.runner.run_vae_decoder = scoped
