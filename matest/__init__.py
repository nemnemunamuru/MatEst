"""MatEst package: public API"""
from .io import read_table_with_type
from .estimator import estimate
from .plotting import make_four_plots

__all__ = ["read_table_with_type", "estimate", "make_four_plots"]
