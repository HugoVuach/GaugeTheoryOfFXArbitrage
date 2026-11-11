"""Paramètres fixes de l'expérience et conventions de cotation."""
from datetime import date, timedelta

CURRENCIES = ("EUR", "GBP", "USD", "JPY")
PAIRS = (
    ("EUR", "GBP"), ("EUR", "USD"), ("EUR", "JPY"),
    ("GBP", "USD"), ("GBP", "JPY"), ("USD", "JPY"),
)
CALIBRATION = tuple(date(2024, 6, 10) + timedelta(days=i) for i in range(5))
EVALUATION = tuple(date(2024, 7, 8) + timedelta(days=i) for i in range(5))
ANNOUNCEMENT = (date(2024, 6, 12), date(2024, 7, 11))
START_UTC = "12:20:00"
END_UTC = "13:00:00"
WARMUP_MS = 60_000
STEP_MS = 100
DELAYS_MS = (0, 100, 250, 500, 1_000)
MAX_AGE_MS = 1_000
COMMON_MAX_AGE_MS = 30_000
AGE_CUTS_MS = (100, 250, 500, 1_000, 2_000, 5_000, 30_000)
STALE_MS = (0, 100, 250, 500, 1_000, 2_000, 5_000)
FEES_BP = (0, 0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5, 1, 2, 5, 10)
COVERAGE_Q = 0.99
BP = 10_000.0
EPS = 1e-8  # Tolérance numérique, en points de base logarithmiques.
