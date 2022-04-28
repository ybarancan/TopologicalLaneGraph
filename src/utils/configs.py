
import os
from yacs.config import CfgNode
import logging
try:
    from dotenv import load_dotenv
    load_dotenv()
except:
    pass

ROOT = os.path.abspath(os.path.join(__file__, '..', '..', '..'))


logging.error('ROOT ' + str(ROOT))
def load_config(config_path):
    with open(config_path) as f:
        return CfgNode.load_cfg(f)
    
def get_default_configuration_argo():
    defaults_path = os.path.join(ROOT, 'configs/defaults_argo.yml')
    return load_config(defaults_path)
    
def get_default_configuration():
    logging.error('ROOT ' + str(ROOT))
    defaults_path = os.path.join(ROOT, 'configs/defaults.yml')
    return load_config(defaults_path)
    
def get_default_polyline_configuration():
    defaults_path = os.path.join(ROOT, 'configs/polyline_defaults.yml')
    return load_config(defaults_path)

def get_default_polyline_configuration_argo():
    defaults_path = os.path.join(ROOT, 'configs/polyline_defaults_argo.yml')
    return load_config(defaults_path)

def get_default_pinet_configuration():
    defaults_path = os.path.join(ROOT, 'configs/pinet_defaults.yml')
    return load_config(defaults_path)


def get_default_pinet_configuration_argo():
    defaults_path = os.path.join(ROOT, 'configs/pinet_defaults_argo.yml')
    return load_config(defaults_path)


