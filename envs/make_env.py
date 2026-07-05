# envs/make_env.py

def make_env(cfg, render_mode=None):
    if cfg.env_type == "mpe":
        from envs.mpe_wrapper import MPEWrapper
        return MPEWrapper(cfg, render_mode=render_mode)
    elif cfg.env_type == "vmas":
        from envs.vmas_wrapper import VMASWrapper
        return VMASWrapper(cfg, render_mode=render_mode)
    else:
        raise ValueError(f"Unknown env_type: {cfg.env_type}")