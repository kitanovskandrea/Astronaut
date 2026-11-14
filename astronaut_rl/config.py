"""Versioned, self-contained experimental settings and exact state thresholds."""
import hashlib
import json
import math
from pathlib import Path
from .environment import ENV_COMMIT, VARIANTS
from .state import StateEncoder

CONFIG_DIR = Path(__file__).resolve().parents[1] / 'configurations'
TRAINING_SEEDS = (11, 23, 37, 51, 79)


def load_config(variant):
    if variant not in VARIANTS:
        raise ValueError('Unknown game variant: ' + str(variant))
    config = json.loads((CONFIG_DIR / (variant + '.json')).read_text())
    if config['environment_commit'] != ENV_COMMIT or config['profile'] != 'official':
        raise ValueError('Configuration requires another environment or reward profile')
    if type(config['repeat']) is not int or config['repeat'] < 1:
        raise ValueError('Action repeat must be a positive integer')
    if not 0 < config['gamma_per_frame'] <= 1 or not math.isclose(
            config['gamma'],config['gamma_per_frame']**config['repeat'],rel_tol=1e-12):
        raise ValueError('Nominal discount must match the physical-frame discount and repeat')
    capacity = StateEncoder(variant, config['state_spec']).capacity
    if config['state_capacity'] != capacity:
        raise ValueError('State capacity differs from stored thresholds')
    if set(config['validation_seeds']) & set(config['test_seeds']):
        raise ValueError('Validation and final-test seeds must be disjoint')
    return config


def config_hash(config):
    return hashlib.sha256(json.dumps(config,sort_keys=True,separators=(',', ':'),allow_nan=False).encode()).hexdigest()
