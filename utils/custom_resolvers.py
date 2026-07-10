from omegaconf import OmegaConf

def replace_slash(s):
    return str(s).replace('/', '_')

try:
    OmegaConf.register_new_resolver('replace_slash', replace_slash)
except Exception:
    pass
