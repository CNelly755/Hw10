"""Turn the downloaded GHCN-Daily records into seasonal snow statistics.

For every candidate station and every winter season it computes:

* snowfall_in      total snowfall for the July-June season (inches)
* big_snow_days    days with at least 1 inch of new snow
* snow_cover_days  days in Nov-Mar with at least 1 inch of snow on the ground

A season is used only if each month Nov-Mar has >= 25 valid observations.
A station is used for a metric only if >= 75% of seasons in BOTH periods
(1941-2000 and 2001-2020) are complete.

Then it builds a regional index (each station's seasonal value as a percent
difference from its own 1941-2000 average, averaged across stations), runs the
statistical tests, and saves tables to output/ and charts to output/figures/.

Usage:  python scripts/analyze.py
"""

from __future__ import annotations

import json
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402

from common import (  # noqa: E402
    ALPHA,
    BASELINE,
    CORE_MONTHS,
    DATA_DIR,
    FIG_DIR,
    METRICS,
    MIN_SEASON_FRACTION,
    MIN_VALID_DAYS_PER_MONTH,
    MM_PER_INCH,
    N_BOOTSTRAP,
    N_PERMUTATIONS,
    ONE_INCH_MM,
    OUTPUT_DIR,
    PRIMARY_METRIC,
    RANDOM_SEED,
    RAW_DIR,
    RECENT,
    in_period,
    read_dly,
    season_of,
)

# Chart styling (validated reference palette; text never wears series colors)
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e4e3df"
SURFACE = "#fcfcfb"
C_BASE = "#2a78d6"  # 1941-2000 (categorical slot 1)
C_RECENT = "#eb6834"  # 2001-2020 (categorical slot 2)
C_OTHER = "#b5b4ae"  # seasons outside both periods
C_MORE = "#2a78d6"  # diverging: more snow (blue pole)
C_LESS = "#e34948"  # diverging: less snow (red pole)

plt.rcParams.update(
    {
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRID,
        "axes.labelcolor": INK_2,
        "axes.titlecolor": INK,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "xtick.color": INK_2,
        "ytick.color": INK_2,
        "font.size": 10,
        "legend.frameon": False,
    }
)


# ----------------------------------------------------------------------------
# Station-season metrics
# ----------------------------------------------------------------------------
def complete_seasons(obs: pd.DataFrame) -> pd.Index:
    """Seasons where every core month has enough valid daily observations."""
    core = obs[obs["month"].isin(CORE_MONTHS)]
    counts = core.groupby(["season", "month"]).size().unstack(fill_value=0)
    counts = counts.reindex(columns=CORE_MONTHS, fill_value=0)
    ok = (counts >= MIN_VALID_DAYS_PER_MONTH).all(axis=1)
    return ok[ok].index


def station_seasons(station_id: str) -> pd.DataFrame:
    path = RAW_DIR / f"{station_id}.dly"
    if not path.exists():
        return pd.DataFrame()
    daily = read_dly(path)
    if daily.empty:
        return pd.DataFrame()
    daily["month"] = daily["date"].dt.month
    daily["season"] = season_of(daily["date"].dt.year, daily["month"])

    frames = []

    snow = daily[daily["element"] == "SNOW"]
    if not snow.empty:
        seasons = complete_seasons(snow)
        s = snow[snow["season"].isin(seasons)]
        agg = s.groupby("season")["value"].agg(
            snowfall_in=lambda v: v.sum() / MM_PER_INCH,
            big_snow_days=lambda v: int((v >= ONE_INCH_MM).sum()),
        )
        frames.append(agg)

    snwd = daily[(daily["element"] == "SNWD") & daily["month"].isin(CORE_MONTHS)]
    if not snwd.empty:
        seasons = complete_seasons(snwd)
        s = snwd[snwd["season"].isin(seasons)]
        # Scale to the full Nov-Mar length so a few missing days don't count as bare ground.
        frac = s.groupby("season")["value"].apply(lambda v: (v >= ONE_INCH_MM).mean())
        n_days = pd.Series(
            {
                season: (pd.Timestamp(season, 4, 1) - pd.Timestamp(season - 1, 11, 1)).days
                for season in frac.index
            }
        )
        frames.append((frac * n_days).rename("snow_cover_days").to_frame())

    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, axis=1).reset_index().rename(columns={"index": "season"})
    out.insert(0, "id", station_id)
    return out


def period_mask(seasons: pd.Series, period) -> pd.Series:
    return in_period(seasons, period)


def qualifying(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Keep station-seasons for stations that meet the completeness rule."""
    d = df.dropna(subset=[metric])
    need_base = MIN_SEASON_FRACTION * (BASELINE[1] - BASELINE[0] + 1)
    need_recent = MIN_SEASON_FRACTION * (RECENT[1] - RECENT[0] + 1)
    n_base = d[period_mask(d["season"], BASELINE)].groupby("id").size()
    n_recent = d[period_mask(d["season"], RECENT)].groupby("id").size()
    counts = pd.concat([n_base.rename("b"), n_recent.rename("r")], axis=1).fillna(0)
    keep = counts[(counts["b"] >= need_base) & (counts["r"] >= need_recent)].index
    return d[d["id"].isin(keep)]


# ----------------------------------------------------------------------------
# Statistics
# ----------------------------------------------------------------------------
def permutation_test(a: np.ndarray, b: np.ndarray, rng) -> dict:
    """Difference in means (b - a). One-sided p tests 'b is lower than a'."""
    observed = b.mean() - a.mean()
    pooled = np.concatenate([a, b])
    n_b = len(b)
    diffs = np.empty(N_PERMUTATIONS)
    for i in range(N_PERMUTATIONS):
        rng.shuffle(pooled)
        diffs[i] = pooled[:n_b].mean() - pooled[n_b:].mean()
    p_less = (np.sum(diffs <= observed) + 1) / (N_PERMUTATIONS + 1)
    p_two = (np.sum(np.abs(diffs) >= abs(observed)) + 1) / (N_PERMUTATIONS + 1)
    return {"diff": float(observed), "p_one_sided_less": float(p_less), "p_two_sided": float(p_two)}


def bootstrap_ci(a: np.ndarray, b: np.ndarray, rng) -> tuple[float, float]:
    ia = rng.integers(0, len(a), size=(N_BOOTSTRAP, len(a)))
    ib = rng.integers(0, len(b), size=(N_BOOTSTRAP, len(b)))
    diffs = b[ib].mean(axis=1) - a[ia].mean(axis=1)
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return float(lo), float(hi)


def station_summary(d: pd.DataFrame, metric: str, meta: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for sid, g in d.groupby("id"):
        a = g.loc[period_mask(g["season"], BASELINE), metric].to_numpy()
        b = g.loc[period_mask(g["season"], RECENT), metric].to_numpy()
        base_mean, recent_mean = a.mean(), b.mean()
        if a.std() == 0 and b.std() == 0:
            p_two = 1.0
        else:
            p_two = float(stats.ttest_ind(b, a, equal_var=False).pvalue)
        rows.append(
            {
                "id": sid,
                "n_base": len(a),
                "n_recent": len(b),
                "base_mean": base_mean,
                "recent_mean": recent_mean,
                "change": recent_mean - base_mean,
                "pct_change": 100 * (recent_mean / base_mean - 1) if base_mean > 0 else np.nan,
                "welch_p": p_two,
            }
        )
    out = pd.DataFrame(rows).merge(meta, on="id", how="left")
    return out.sort_values("pct_change").reset_index(drop=True)


def regional_index(d: pd.DataFrame, metric: str, summary: pd.DataFrame) -> pd.DataFrame:
    """Average across stations of percent difference from each station's own baseline."""
    base = summary.set_index("id")["base_mean"]
    base = base[base > 0]
    d = d[d["id"].isin(base.index)].copy()
    d["anomaly_pct"] = 100 * (d[metric] / d["id"].map(base) - 1)
    reg = d.groupby("season").agg(
        anomaly_pct=("anomaly_pct", "mean"), n_stations=("anomaly_pct", "size")
    )
    reg = reg[reg["n_stations"] >= 3]
    # Express the index in physical units for a "typical" station: the average
    # station baseline scaled by the regional percent anomaly.
    typical_base = float(base.mean())
    reg["typical_value"] = typical_base * (1 + reg["anomaly_pct"] / 100)
    return reg.reset_index()


def compare(reg: pd.DataFrame, p1, p2, rng) -> dict:
    a = reg.loc[period_mask(reg["season"], p1), "anomaly_pct"].to_numpy()
    b = reg.loc[period_mask(reg["season"], p2), "anomaly_pct"].to_numpy()
    if len(a) < 5 or len(b) < 5:
        return {}
    perm = permutation_test(a, b, rng)
    lo, hi = bootstrap_ci(a, b, rng)
    mw = stats.mannwhitneyu(b, a, alternative="less")
    return {
        "period_a": list(p1),
        "period_b": list(p2),
        "n_a": int(len(a)),
        "n_b": int(len(b)),
        "mean_a_pct": float(a.mean()),
        "mean_b_pct": float(b.mean()),
        "diff_pct_points": perm["diff"],
        "ci95": [lo, hi],
        "p_perm_one_sided": perm["p_one_sided_less"],
        "p_perm_two_sided": perm["p_two_sided"],
        "p_mannwhitney_one_sided": float(mw.pvalue),
    }


# ----------------------------------------------------------------------------
# Figures
# ----------------------------------------------------------------------------
def season_label(season: int) -> str:
    return f"{season - 1}-{str(season)[-2:]}"


def fig_regional(reg: pd.DataFrame, metric: str, result: dict, path) -> None:
    info = METRICS[metric]
    fig, ax = plt.subplots(figsize=(10, 4.6))
    colors = np.where(
        period_mask(reg["season"], BASELINE),
        C_BASE,
        np.where(period_mask(reg["season"], RECENT), C_RECENT, C_OTHER),
    )
    ax.bar(reg["season"], reg["typical_value"], width=0.72, color=colors, linewidth=0)

    typical_base = reg["typical_value"] / (1 + reg["anomaly_pct"] / 100)
    tb = float(typical_base.iloc[0])
    for period, color in [(BASELINE, C_BASE), (RECENT, C_RECENT)]:
        sel = period_mask(reg["season"], period)
        mean_val = tb * (1 + reg.loc[sel, "anomaly_pct"].mean() / 100)
        ax.hlines(mean_val, period[0] - 0.5, period[1] + 0.5, color=INK, linewidth=2)
        ax.annotate(
            f"{period[0]}-{period[1]} avg: {mean_val:.1f} {'in' if info['unit'] == 'inches' else 'days'}",
            xy=(period[0] - 0.3, mean_val),
            xytext=(0, 6),
            textcoords="offset points",
            color=INK,
            fontsize=9,
            fontweight="bold",
            bbox={"boxstyle": "round,pad=0.25", "facecolor": SURFACE, "edgecolor": "none", "alpha": 0.9},
        )

    smooth = reg.set_index("season")["typical_value"].rolling(10, center=True, min_periods=6).mean()
    ax.plot(smooth.index, smooth.values, color=INK_2, linewidth=1.5, linestyle="--")

    ax.set_xlim(reg["season"].min() - 1, reg["season"].max() + 1)
    ax.set_ylabel(f"{info['unit']} (typical station)")
    ax.set_title(f"{info['label']}, New England, by winter")
    ax.grid(axis="x", visible=False)
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    handles = [
        Patch(color=C_BASE, label=f"{BASELINE[0]}-{BASELINE[1]} winters"),
        Patch(color=C_RECENT, label=f"{RECENT[0]}-{RECENT[1]} winters"),
        Line2D([], [], color=INK, linewidth=2, label="period average"),
        Line2D([], [], color=INK_2, linewidth=1.5, linestyle="--", label="10-winter running mean"),
    ]
    if (~period_mask(reg["season"], BASELINE) & ~period_mask(reg["season"], RECENT)).any():
        handles.insert(2, Patch(color=C_OTHER, label="other winters (not tested)"))
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=len(handles), fontsize=8.5)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def fig_decades(regs: dict, path) -> None:
    metrics = [m for m in METRICS if m in regs]
    fig, axes = plt.subplots(1, len(metrics), figsize=(4 * len(metrics), 4), sharey=True, squeeze=False)
    for ax, metric in zip(axes[0], metrics):
        reg = regs[metric]
        start = BASELINE[0]
        bins = list(range(start, reg["season"].max() + 1, 10))
        labels, values = [], []
        for lo in bins:
            hi = lo + 9
            sel = (reg["season"] >= lo) & (reg["season"] <= hi)
            if sel.sum() < 3:
                continue
            last = min(hi, int(reg.loc[sel, "season"].max()))
            labels.append(f"{lo - 1}s" if last == hi else f"{lo}-{str(last)[-2:]}")
            values.append(reg.loc[sel, "anomaly_pct"].mean())
        x = np.arange(len(values))
        colors = [C_MORE if v >= 0 else C_LESS for v in values]
        ax.bar(x, values, color=colors, width=0.7)
        ax.axhline(0, color=INK_2, linewidth=1)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=8, rotation=45, ha="right", rotation_mode="anchor")
        ax.set_title(METRICS[metric]["short"].capitalize())
        ax.grid(axis="x", visible=False)
    axes[0][0].set_ylabel(f"% vs. {BASELINE[0]}-{BASELINE[1]} average")
    fig.suptitle(
        "Decade averages, relative to each station's 1941-2000 normal",
        x=0.01,
        ha="left",
        fontsize=12,
        fontweight="bold",
        color=INK,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def fig_stations(summary: pd.DataFrame, path) -> None:
    s = summary.dropna(subset=["pct_change"]).sort_values("pct_change")
    height = max(4, 0.26 * len(s) + 1.2)
    fig, ax = plt.subplots(figsize=(8, height))
    y = np.arange(len(s))
    labels = [f"{n.title()}, {st}" for n, st in zip(s["name"], s["state"])]
    for yi, (pct, p) in enumerate(zip(s["pct_change"], s["welch_p"])):
        color = C_MORE if pct >= 0 else C_LESS
        ax.hlines(yi, 0, pct, color=color, linewidth=2)
        face = color if p < ALPHA else SURFACE
        ax.plot(pct, yi, "o", markersize=8, markerfacecolor=face, markeredgecolor=color, markeredgewidth=2)
    ax.axvline(0, color=INK_2, linewidth=1)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel(f"Change in average seasonal snowfall, {RECENT[0]}-{RECENT[1]} vs {BASELINE[0]}-{BASELINE[1]} (%)")
    ax.set_title("Station by station: seasonal snowfall change")
    ax.grid(axis="y", visible=False)
    from matplotlib.lines import Line2D

    handles = [
        Line2D([], [], marker="o", linestyle="", markersize=8, markerfacecolor=INK_2, markeredgecolor=INK_2, label="significant (Welch t-test p < 0.05)"),
        Line2D([], [], marker="o", linestyle="", markersize=8, markerfacecolor=SURFACE, markeredgecolor=INK_2, markeredgewidth=2, label="not significant"),
    ]
    ax.legend(handles=handles, loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def fig_distributions(regs: dict, path) -> None:
    metrics = [m for m in METRICS if m in regs]
    fig, axes = plt.subplots(1, len(metrics), figsize=(4 * len(metrics), 4), squeeze=False)
    rng = np.random.default_rng(0)
    for ax, metric in zip(axes[0], metrics):
        reg = regs[metric]
        groups = [
            reg.loc[period_mask(reg["season"], BASELINE), "anomaly_pct"].to_numpy(),
            reg.loc[period_mask(reg["season"], RECENT), "anomaly_pct"].to_numpy(),
        ]
        bp = ax.boxplot(groups, widths=0.5, showfliers=False, patch_artist=True, medianprops={"color": INK, "linewidth": 2})
        for patch in bp["boxes"]:
            patch.set_facecolor(SURFACE)
            patch.set_edgecolor(INK_2)
        for i, (g, color) in enumerate(zip(groups, [C_BASE, C_RECENT]), start=1):
            jitter = rng.uniform(-0.15, 0.15, size=len(g))
            ax.plot(i + jitter, g, "o", markersize=4.5, color=color, alpha=0.8, markeredgecolor=SURFACE, markeredgewidth=0.6)
            ax.plot(i, g.mean(), marker="D", markersize=8, color=INK)
        ax.axhline(0, color=INK_2, linewidth=1, linestyle=":")
        ax.set_xticks([1, 2])
        ax.set_xticklabels([f"{BASELINE[0]}-{BASELINE[1]}", f"{RECENT[0]}-{RECENT[1]}"])
        ax.set_title(METRICS[metric]["short"].capitalize())
        ax.grid(axis="x", visible=False)
    axes[0][0].set_ylabel("% vs. station's 1941-2000 average")
    fig.suptitle(
        "Every winter's regional value (dots), box = middle half, diamond = mean",
        x=0.01,
        ha="left",
        fontsize=12,
        fontweight="bold",
        color=INK,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ----------------------------------------------------------------------------
def main() -> int:
    cand_path = DATA_DIR / "candidates.csv"
    if not cand_path.exists():
        print("data/candidates.csv not found - run scripts/download_data.py first.")
        return 1
    candidates = pd.read_csv(cand_path, dtype={"id": str})
    meta = candidates[["id", "name", "state", "lat", "lon", "elev_m"]]

    print(f"Computing seasonal metrics for {len(candidates)} stations...")
    frames = [station_seasons(sid) for sid in candidates["id"]]
    frames = [f for f in frames if not f.empty]
    if not frames:
        print("No station data could be read.")
        return 1
    seasons = pd.concat(frames, ignore_index=True)
    for m in METRICS:
        if m not in seasons:
            seasons[m] = np.nan

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    seasons.to_csv(OUTPUT_DIR / "station_seasons.csv", index=False)

    rng = np.random.default_rng(RANDOM_SEED)
    latest_season = int(seasons["season"].max())
    results = {
        "baseline": list(BASELINE),
        "recent": list(RECENT),
        "alpha": ALPHA,
        "n_candidates": int(len(candidates)),
        "latest_season": latest_season,
        "metrics": {},
    }
    regs, summaries = {}, {}

    for metric, info in METRICS.items():
        d = qualifying(seasons, metric)
        n_st = d["id"].nunique()
        print(f"  {metric}: {n_st} stations meet the completeness rule")
        if n_st < 3:
            results["metrics"][metric] = {"n_stations": int(n_st), "skipped": True}
            continue
        summary = station_summary(d, metric, meta)
        reg = regional_index(d, metric, summary)
        summary.to_csv(OUTPUT_DIR / f"stations_{metric}.csv", index=False)
        reg.to_csv(OUTPUT_DIR / f"regional_{metric}.csv", index=False)
        regs[metric], summaries[metric] = reg, summary

        main_cmp = compare(reg, BASELINE, RECENT, rng)
        n_down = int((summary["change"] < 0).sum())
        n_valid = int(summary["change"].ne(0).sum())
        sign_p = float(stats.binomtest(n_down, n_valid, 0.5, alternative="greater").pvalue) if n_valid else 1.0
        sig_down = int(((summary["change"] < 0) & (summary["welch_p"] < ALPHA)).sum())
        sig_up = int(((summary["change"] > 0) & (summary["welch_p"] < ALPHA)).sum())

        sel_full = (reg["season"] >= BASELINE[0]) & (reg["season"] <= latest_season)
        trend = stats.linregress(reg.loc[sel_full, "season"], reg.loc[sel_full, "anomaly_pct"])

        typical_base = float(summary.loc[summary["base_mean"] > 0, "base_mean"].mean())
        typical_recent_station_mean = float(summary.loc[summary["base_mean"] > 0, "recent_mean"].mean())

        sensitivity = []
        for p1, p2 in [
            ((1941, 1970), RECENT),
            ((1971, 2000), RECENT),
            (BASELINE, (2001, latest_season)),
            ((1951, 1980), (1991, 2020)),
        ]:
            c = compare(reg, p1, p2, rng)
            if c:
                sensitivity.append(c)

        results["metrics"][metric] = {
            "label": info["label"],
            "unit": info["unit"],
            "n_stations": int(n_st),
            "typical_base": typical_base * (1 + main_cmp["mean_a_pct"] / 100),
            "typical_recent": typical_base * (1 + main_cmp["mean_b_pct"] / 100),
            "avg_station_base_mean": typical_base,
            "avg_station_recent_mean": typical_recent_station_mean,
            "comparison": main_cmp,
            "stations_down": n_down,
            "stations_total": int(len(summary)),
            "sign_test_p": sign_p,
            "stations_sig_down": sig_down,
            "stations_sig_up": sig_up,
            "trend_pct_per_decade": float(trend.slope * 10),
            "trend_p": float(trend.pvalue),
            "trend_span": [int(reg.loc[sel_full, "season"].min()), int(reg.loc[sel_full, "season"].max())],
            "sensitivity": sensitivity,
        }

        fig_regional(reg, metric, main_cmp, FIG_DIR / f"regional_{metric}.png")
        print(
            f"    regional change {main_cmp['diff_pct_points']:+.1f}% "
            f"(95% CI {main_cmp['ci95'][0]:+.1f} to {main_cmp['ci95'][1]:+.1f}), "
            f"one-sided p = {main_cmp['p_perm_one_sided']:.4f}"
        )

    if PRIMARY_METRIC not in summaries:
        print("Not enough stations for the primary snowfall metric.")
        return 1

    fig_decades(regs, FIG_DIR / "decades.png")
    fig_distributions(regs, FIG_DIR / "distributions.png")
    fig_stations(summaries[PRIMARY_METRIC], FIG_DIR / "stations_snowfall.png")

    primary = results["metrics"][PRIMARY_METRIC]["comparison"]
    is_less = primary["diff_pct_points"] < 0
    significant = primary["p_perm_one_sided"] < ALPHA
    results["headline"] = "Yes" if (is_less and significant) else "No"
    results["primary_metric"] = PRIMARY_METRIC

    with open(OUTPUT_DIR / "results.json", "w") as fh:
        json.dump(results, fh, indent=2)
    print(f"Headline answer: {results['headline']}")
    print(f"Wrote results to {OUTPUT_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
