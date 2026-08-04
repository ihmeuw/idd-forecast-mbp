"""pyGAM sandbox for malaria model selection (PfPR + downstream inc/mort).

Mirrors the scam selection / forecast *methodology* in ``03_modeling/`` and
``04_forecasting/`` so covariate-formulation experiments can run interactively
in Python. Modules:

- ``data``    : load + clean the past-inputs parquet (same rows/transforms as scam).
- ``specs``   : flexible spec enumerator -> pyGAM term structures.
- ``fit``     : fit + predict a spec under IS / within-country / temporal evaluation.
- ``metrics`` : one canonical residual-metric set, reused for every outcome.
- ``shift``   : 2023-anchor logit/log shift + the downstream inc/mort chain.
"""
