"""Cohort + trophy-kind forecaster for GTA VI PS5 trophy earn rates on Exophase.

The forecast target is each trophy's earn rate (percent of tracked players who
have earned it) at the case's target time, usually a week after launch.  The
model splits the rate into its two moving parts and forecasts each separately:

    rate(t) = earners(t) / players_tracked(t) * 100

* players_tracked: one clean series for the whole game.  Its log is fitted
  against log-hours since release (a power-law-style adoption curve) and
  extrapolated to the target.
* earners: per trophy, the log of the reconstructed earner count is fitted
  against the same log-time axis.  Earners can only grow, so fitted slopes
  are clamped to >= 0.  A trophy whose own history is too short or too noisy
  to fit borrows the median slope of its kind (story / collectibles /
  activities); the weight on its own slope grows with the number of distinct
  reads it has.  The forecast is anchored on the last observed earner count,
  so a trophy's level always comes from its own data.
* the platinum is its own case: it needs roughly 200 hours of play, more
  than the 168 hours between release and the week-one target, so its earner
  slope is capped hard and its median rate is capped at 0.05%.

Spreads are honest, not decorative: each trophy's five quantiles come from a
logit-scale spread that combines the fit residual (widened by the
extrapolation leverage), the uncertainty of the tracked-players fit, the
dispersion of its kind's slopes, a random-walk term that grows with the
horizon, and a refresh allowance (Exophase refreshes its counts on its own
schedule, so a long-frozen history means the next refresh can move every
rate at once).  On a completely flat history there is no trend to fit, so
medians stay at the last read and the refresh allowance does the work.

Everything is deterministic: no randomness, no network, standard library
only.  Reads every *.in file in MAKEFUN_INPUT_DIR (or /input) and writes one
<id>.out per case to MAKEFUN_OUTPUT_DIR (or /output).

    MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
"""

import json
import math
import os
from datetime import datetime, timedelta, timezone

# GTA VI released 2026-11-19 00:00 UTC; every time axis is measured from here.
RELEASE = datetime(2026, 11, 19, tzinfo=timezone.utc)

# The five quantile levels the contest scores, and the standard normal's
# quantile at each level (more digits than the usual rounded table).
LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.6448536269514722, -0.6744897501960817, 0.0, 0.6744897501960817, 1.6448536269514722]

# Kind of each trophy (id -> story / collectibles / activities), from our own
# trophy-guide classification of the same 55 trophies: story trophies climb
# fast as players finish chapters, collectibles and activities climb slower.
KIND = {
    1: "activities", 2: "story", 3: "story", 4: "story", 5: "story",
    6: "story", 7: "story", 8: "story", 9: "story", 10: "story",
    11: "story", 12: "story", 13: "story", 14: "activities",
    15: "activities", 16: "activities", 17: "activities",
    18: "activities", 19: "activities", 20: "activities",
    21: "activities", 22: "story", 23: "story", 24: "story",
    25: "activities", 26: "story", 27: "story", 28: "collectibles",
    29: "activities", 30: "collectibles", 31: "collectibles",
    32: "activities", 33: "activities", 34: "activities",
    35: "collectibles", 36: "activities", 37: "activities",
    38: "activities", 39: "activities", 40: "activities",
    41: "activities", 42: "collectibles", 43: "activities",
    44: "activities", 45: "activities", 46: "activities",
    47: "activities", 48: "activities", 49: "activities",
    50: "activities", 51: "activities", 52: "activities",
    53: "activities", 54: "activities", 55: "activities",
}

# Fallback log-time slopes (log earners per log hour) when a kind has no
# fittable trophy at all in the case: story fastest, collectibles slowest.
KIND_FALLBACK_SLOPE = {"story": 1.2, "activities": 0.8, "collectibles": 0.5}

# Platinum guardrails: ~200 h of play needed, so in week one its earners
# barely move and its rate is diluted away by the growing tracked base.
PLATINUM_SLOPE_CAP = 0.3
PLATINUM_RATE_CAP = 0.05

# Absolute week-one ceilings for a trophy's rate, by kind: not every tracked
# player finishes the story in the first week, and collectibles are grindy.
# These only bind when an extrapolation runs away; they are not forecasts.
KIND_RATE_CAP = {"story": 85.0, "activities": 60.0, "collectibles": 40.0}

# A trophy's rate cannot plausibly rise by more than this many logit units
# over the forecast horizon: at low rates that is roughly a ninefold rise.
# Without the cap, a trophy still in fast adoption extrapolates to ~100%,
# and one such row costs more pinball loss than the whole rest of the entry.
MAX_LOGIT_RISE = 2.2

# How much of a fitted log-time trend is carried to the target: a straight
# line in log-time keeps rising where the real adoption curve bends over
# (players saturate, the tracked base's growth slows), so only part of the
# trend is extrapolated.  1.0 trusts the trend fully, 0.0 holds every rate
# where it is.  Calibrated on synthetic week-ahead cases built from a
# different (cohort) model family: 0.65 was the middle of the range that beat
# the no-growth baseline on both variants without ever blowing up.
EXTRAP_DAMP = 0.65

# Random-walk spread, in logit units per square-root day (same scale as the
# house baseline's spread, used here as one component among several).
RW_SIGMA = 0.5

# Earner counts are reconstructed from rates printed to 2 decimals, so a
# trophy nobody has earned yet is floored here instead of taking log(0).
EARNER_FLOOR = 0.5


def parse_time(text):
    """ISO 8601 timestamp (with Z or an offset) -> aware datetime."""
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def hours_since_release(moment):
    """Hours from release to `moment`; negative (pre-release reads) -> 0."""
    return max(0.0, (moment - RELEASE).total_seconds() / 3600.0)


def log_time(moment):
    """Log-time axis: ln(1 + hours since release)."""
    return math.log1p(hours_since_release(moment))


def clamp_pct(value):
    return min(99.99, max(0.01, value))


def logit(pct):
    p = clamp_pct(pct) / 100.0
    return math.log(p / (1.0 - p))


def inv_logit_pct(logit_value):
    """Logit back to a percent in (0, 100), overflow-safe."""
    if logit_value >= 0:
        return 100.0 / (1.0 + math.exp(-logit_value))
    e = math.exp(logit_value)
    return 100.0 * e / (1.0 + e)


def round4(value):
    """Round to 4 decimals, half up; whole numbers come back as ints."""
    scaled = value * 10000.0
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    out = whole / 10000.0
    return int(out) if out == int(out) else out


def median(values):
    ordered = sorted(values)
    n = len(ordered)
    if n == 0:
        return None
    if n % 2:
        return ordered[n // 2]
    return 0.5 * (ordered[n // 2 - 1] + ordered[n // 2])


def ols(xs, ys):
    """Least squares line y ~ a + b*x.  Returns (slope, intercept, resid_std,
    sxx, mean_x) or None when the x values carry no spread."""
    n = len(xs)
    if n < 2:
        return None
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    sxx = sum((x - mean_x) ** 2 for x in xs)
    if sxx <= 1e-12:
        return None
    sxy = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    slope = sxy / sxx
    intercept = mean_y - slope * mean_x
    if n > 2:
        resid = math.sqrt(
            sum((y - (intercept + slope * x)) ** 2 for x, y in zip(xs, ys)) / (n - 2)
        )
    else:
        resid = 0.0
    return slope, intercept, resid, sxx, mean_x


def distinct_states(history):
    """Collapse runs of consecutive identical reads into one state, kept at
    the time of the run's last read (the state is known to persist to then).
    Each state is (log_time, players_tracked, rates tuple)."""
    states = []
    for read in history:
        rates = tuple(read.get("rates") or [])
        players = read.get("players_tracked")
        if players is None or players <= 0:
            continue
        if states and states[-1][1] == players and states[-1][2] == rates:
            states[-1] = (log_time(parse_time(read["at"])), players, rates)
        else:
            states.append((log_time(parse_time(read["at"])), players, rates))
    return states


def fit_tracked(states, x_target):
    """Fit log players_tracked against log-time; extrapolate to x_target.
    Returns (log_players_at_target, sigma) where sigma is the fit's residual
    spread widened by extrapolation leverage."""
    pts = [(x, math.log(n)) for x, n, _ in states]
    if not pts:
        return None, 0.0
    if len(pts) == 1:
        return pts[0][1], 0.0
    fit = ols([p[0] for p in pts], [p[1] for p in pts])
    if fit is None:
        return pts[-1][1], 0.0
    slope, intercept, resid, sxx, mean_x = fit
    slope = min(max(slope, 0.0), 2.0)
    # Anchor on the last observed count: the fit supplies only the slope,
    # damped as in the earner extrapolation (see EXTRAP_DAMP).
    log_n = pts[-1][1] + EXTRAP_DAMP * slope * (x_target - pts[-1][0])
    # Never let the extrapolation run away from the last observed count.
    log_n = min(log_n, math.log(states[-1][1]) + math.log(20.0))
    leverage = (x_target - mean_x) ** 2 / sxx if sxx > 0 else 0.0
    sigma = resid * min(math.sqrt(1.0 + leverage), 2.0)
    return log_n, sigma


def forecast_case(case):
    """Forecast one case; returns {"trophies": [{"id": n, "q": [...]}, ...]}."""
    trophies = case["trophies"]
    history = case.get("history") or []
    horizon_days = case["horizon_minutes"] / 1440.0
    target = parse_time(case["target"])
    x_target = log_time(target)

    states = distinct_states(history)
    flat_history = len(states) <= 1

    # Slopes come from the last 24 hours of distinct states: adoption
    # saturates, so the recent growth rate is the honest one to extrapolate,
    # and a whole-history fit would carry the fast early phase forward.
    # With too few recent states, fall back to the whole history.
    checkpoint = parse_time(case["at"])
    window_start = log_time(checkpoint - timedelta(hours=24))
    recent = [s for s in states if s[0] >= window_start]
    fit_states = recent if len(recent) >= 3 else states

    log_n_target, sigma_n = fit_tracked(fit_states, x_target)
    if log_n_target is None:
        log_n_target = math.log(1000.0)

    # Per-trophy log-earner fits over the recent distinct states, plus each
    # trophy's last observed rate (for the rise cap below).
    n_trophies = len(trophies)
    own = []  # per trophy: (slope, resid_std, n_points, leverage_factor) or None
    last_earners = []
    last_rates = []
    last_x = states[-1][0] if states else 0.0
    for i in range(n_trophies):
        xs, ys = [], []
        for x, n_players, rates in fit_states:
            if i < len(rates) and rates[i] is not None:
                earners = max(rates[i] * n_players / 100.0, EARNER_FLOOR)
                xs.append(x)
                ys.append(math.log(earners))
        last_earners.append(math.exp(ys[-1]) if ys else None)
        last_rate = None
        for _x, _n, rates in reversed(states):
            if i < len(rates) and rates[i] is not None:
                last_rate = rates[i]
                break
        last_rates.append(last_rate)
        if len(xs) >= 2:
            fit = ols(xs, ys)
            if fit is not None:
                slope, _intercept, resid, sxx, mean_x = fit
                leverage = (x_target - mean_x) ** 2 / sxx if sxx > 0 else 0.0
                own.append(
                    (min(max(slope, 0.0), 3.0), resid, len(xs),
                     min(math.sqrt(1.0 + leverage), 2.0))
                )
            else:
                own.append(None)
        else:
            own.append(None)

    # Kind slopes: the median fitted slope within each kind; fall back to the
    # global median, then to the kind's prior.
    kind_slopes = {}
    kind_resids = {}
    all_slopes = [o[0] for o in own if o is not None]
    global_slope = median(all_slopes)
    for kind in ("story", "activities", "collectibles"):
        slopes = [own[i][0] for i in range(n_trophies)
                  if own[i] is not None and KIND.get(trophies[i]["id"], "activities") == kind]
        resids = [own[i][1] for i in range(n_trophies)
                  if own[i] is not None and KIND.get(trophies[i]["id"], "activities") == kind]
        kind_slopes[kind] = median(slopes)
        if kind_slopes[kind] is None:
            kind_slopes[kind] = global_slope if global_slope is not None else KIND_FALLBACK_SLOPE[kind]
        kind_resids[kind] = median(resids) if resids else 0.15

    # How much trophies of a kind disagree about their slope: slope risk.
    kind_spread = {}
    for kind in ("story", "activities", "collectibles"):
        slopes = [own[i][0] for i in range(n_trophies)
                  if own[i] is not None and KIND.get(trophies[i]["id"], "activities") == kind]
        if len(slopes) >= 2:
            mean_s = sum(slopes) / len(slopes)
            kind_spread[kind] = math.sqrt(
                sum((s - mean_s) ** 2 for s in slopes) / (len(slopes) - 1)
            )
        else:
            kind_spread[kind] = 0.0

    # A refresh moves rates: the biggest logit jump between consecutive
    # distinct states, a proxy for how violently a refresh can shift a rate.
    max_jump = 0.0
    for a, b in zip(states, states[1:]):
        for i in range(n_trophies):
            if i < len(a[2]) and i < len(b[2]) and a[2][i] is not None and b[2][i] is not None:
                max_jump = max(max_jump, abs(logit(b[2][i]) - logit(a[2][i])))
    if flat_history:
        sigma_refresh = 0.6
    elif max_jump > 0.4:
        sigma_refresh = 0.3
    else:
        sigma_refresh = 0.15

    sigma_rw = RW_SIGMA * math.sqrt(horizon_days)
    dx = x_target - last_x

    rows = []
    for i, trophy in enumerate(trophies):
        tid = trophy["id"]
        kind = KIND.get(tid, "activities")
        tier = trophy.get("tier")

        if last_earners[i] is not None:
            if flat_history:
                # No trend exists to fit: the page has not moved at all, so
                # medians stay at the last read and the refresh allowance in
                # the spread carries the risk of the next refresh.
                slope = 0.0
                sigma_fit = kind_resids[kind]
            elif own[i] is not None:
                slope_own, resid, n_pts, lev_factor = own[i]
                weight = n_pts / (n_pts + 5.0)
                slope = weight * slope_own + (1.0 - weight) * kind_slopes[kind]
                sigma_fit = resid * lev_factor
            else:
                slope = kind_slopes[kind]
                sigma_fit = kind_resids[kind]
            slope = min(max(slope, 0.0), 3.0)
            if tier == "platinum":
                slope = min(slope, PLATINUM_SLOPE_CAP)
            log_e_target = math.log(max(last_earners[i], EARNER_FLOOR)) + EXTRAP_DAMP * slope * dx
            rate_med = 100.0 * math.exp(log_e_target) / math.exp(log_n_target)
            rate_med = clamp_pct(rate_med)
            # Guardrails against a runaway extrapolation: the kind ceiling
            # (never below the trophy's own last rate) and at most
            # MAX_LOGIT_RISE above that last rate.
            cap = KIND_RATE_CAP[kind]
            if last_rates[i] is not None:
                cap = max(cap, last_rates[i])
            rate_med = min(rate_med, cap)
            if last_rates[i] is not None:
                rate_med = min(
                    rate_med,
                    inv_logit_pct(logit(last_rates[i]) + MAX_LOGIT_RISE),
                )
            if tier == "platinum":
                rate_med = min(rate_med, PLATINUM_RATE_CAP)
            sigma_kind = min(kind_spread[kind] * dx, 0.5)
        else:
            # Never observed: fall back to the kind's median last rate, wide band.
            seen_rates = [last_rates[j] for j in range(n_trophies)
                          if last_rates[j] is not None
                          and KIND.get(trophies[j]["id"], "activities") == kind]
            rate_med = 1.0
            sigma_fit = 1.0
            sigma_kind = 0.5
            if seen_rates:
                rate_med = clamp_pct(median(seen_rates) or 1.0)

        sigma = math.sqrt(
            sigma_fit ** 2 + sigma_n ** 2 + sigma_kind ** 2
            + sigma_rw ** 2 + sigma_refresh ** 2
        )
        sigma = min(max(sigma, 0.15), 2.5)

        centre = logit(rate_med)
        quantiles = []
        previous = None
        for z in Z:
            q = clamp_pct(inv_logit_pct(centre + z * sigma))
            q = round4(q)
            if previous is not None and q < previous:
                q = previous
            quantiles.append(q)
            previous = q
        rows.append({"id": tid, "q": quantiles})
    return {"trophies": rows}


def main():
    input_dir = os.environ.get("MAKEFUN_INPUT_DIR", "/input")
    output_dir = os.environ.get("MAKEFUN_OUTPUT_DIR", "/output")
    os.makedirs(output_dir, exist_ok=True)
    for name in sorted(os.listdir(input_dir)):
        if not name.endswith(".in"):
            continue
        with open(os.path.join(input_dir, name), encoding="utf-8") as fh:
            case = json.load(fh)
        forecast = forecast_case(case)
        out_path = os.path.join(output_dir, name[:-3] + ".out")
        with open(out_path, "w", encoding="utf-8") as fh:
            fh.write(json.dumps(forecast, separators=(",", ":")) + "\n")


if __name__ == "__main__":
    main()
