"""Monitoring luster MAGIC-II: geometria -> QC -> cechy -> baseline -> wynik anomalii."""
import yaml


def load_config(path="config.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)
