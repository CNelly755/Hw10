"""Build the GitHub Pages report (site/index.html) from the analysis output.

All numbers and the wording that depends on them come from output/results.json,
so the page always matches the data it was built from.

Usage:  python scripts/build_report.py
"""

from __future__ import annotations

import datetime as dt
import html
import json
import os
import shutil
import subprocess
import sys

import pandas as pd

from common import ALPHA, FIG_DIR, METRICS, OUTPUT_DIR, PRIMARY_METRIC, ROOT, SITE_DIR, TEMPLATE


def fmt_p(p: float) -> str:
    if p < 0.001:
        return "&lt; 0.001"
    return f"{p:.3f}" if p < 0.01 else f"{p:.2f}"


def pstr(p: float) -> str:
    return "p &lt; 0.001" if p < 0.001 else f"p = {fmt_p(p)}"


def period(p) -> str:
    return f"{p[0]}&ndash;{p[1]}"


def signed(x: float, digits: int = 1) -> str:
    s = f"{abs(x):.{digits}f}"
    return ("+" if x > 0 else "&minus;" if x < 0 else "") + s


def unit_short(unit: str) -> str:
    return "in" if unit == "inches" else "days"


def repo_url() -> str:
    repo = os.environ.get("GITHUB_REPOSITORY")
    if repo:
        return f"https://github.com/{repo}"
    try:
        url = subprocess.run(
            ["git", "config", "--get", "remote.origin.url"], cwd=ROOT, capture_output=True, text=True, timeout=10
        ).stdout.strip()
        if url.startswith("git@github.com:"):
            url = "https://github.com/" + url.split(":", 1)[1]
        return url.removesuffix(".git") or "#"
    except Exception:
        return "#"


def verdict_phrase(c: dict) -> str:
    """Plain-language reading of one comparison."""
    d, p = c["diff_pct_points"], c["p_perm_one_sided"]
    lo, hi = c["ci95"]
    if d < 0 and p < ALPHA:
        return "lower, and the drop is larger than year-to-year chance would explain"
    if d < 0:
        return "slightly lower, but the difference is within normal year-to-year swings"
    if lo > 0:
        return "higher, not lower"
    return "about the same (slightly higher, within normal year-to-year swings)"


def main() -> int:
    res_path = OUTPUT_DIR / "results.json"
    if not res_path.exists():
        print("output/results.json not found - run scripts/analyze.py first.")
        return 1
    res = json.loads(res_path.read_text())
    base, recent = res["baseline"], res["recent"]
    m = res["metrics"][PRIMARY_METRIC]
    c = m["comparison"]
    headline = res["headline"]
    today = dt.date.today()
    run_date = f"{today:%B} {today.day}, {today.year}"

    pct = c["diff_pct_points"]
    lo, hi = c["ci95"]
    tb, tr = m["typical_base"], m["typical_recent"]

    # --- Headline -----------------------------------------------------------
    if headline == "Yes":
        sentence = (
            f"<strong>He was right.</strong> Across {m['n_stations']} long-running New England weather stations, "
            f"winters in {period(recent)} averaged <strong>{abs(pct):.0f}% less snowfall</strong> than winters in "
            f"{period(base)}, and that drop is too large to be explained by ordinary year-to-year variation "
            f"({pstr(c['p_perm_one_sided'])})."
        )
    elif pct < 0:
        sentence = (
            f"<strong>Not by the numbers.</strong> Across {m['n_stations']} long-running New England weather stations, "
            f"winters in {period(recent)} averaged {abs(pct):.0f}% less snowfall than in {period(base)}, but New England "
            f"snowfall varies so much from winter to winter that a gap this size could easily be chance "
            f"({pstr(c['p_perm_one_sided'])}). The record does not show a real decline."
        )
    elif lo < 0:
        sentence = (
            f"<strong>No &mdash; it looks like nostalgia.</strong> Across {m['n_stations']} long-running New England "
            f"weather stations, winters in {period(recent)} had about the same snowfall as winters in {period(base)} "
            f"&mdash; if anything slightly more ({signed(pct, 0)}%), not less."
        )
    else:
        sentence = (
            f"<strong>No &mdash; it looks like nostalgia.</strong> Across {m['n_stations']} long-running New England "
            f"weather stations, winters in {period(recent)} averaged {abs(pct):.0f}% <em>more</em> snowfall than winters "
            f"in {period(base)}, not less."
        )

    tiles = [
        ("Typical station, " + period(base), f"{tb:.0f} in", "average seasonal snowfall"),
        ("Typical station, " + period(recent), f"{tr:.0f} in", "average seasonal snowfall"),
        ("Change", f"{signed(pct, 0)}%", f"95% CI {signed(lo, 0)}% to {signed(hi, 0)}%"),
        ("Stations with less snow", f"{m['stations_down']} of {m['stations_total']}",
         f"{m['stations_sig_down']} significantly lower, {m['stations_sig_up']} significantly higher"),
    ]
    tiles_html = "".join(
        f'<div class="tile"><div class="k">{k}</div><div class="v">{v}</div><div class="s">{s}</div></div>'
        for k, v, s in tiles
    )

    # --- Summary bullets ----------------------------------------------------
    bullets = [
        f"Average seasonal snowfall in {period(recent)} was {verdict_phrase(c)} compared with {period(base)} "
        f"({signed(pct)}%, 95% confidence interval {signed(lo)}% to {signed(hi)}%).",
    ]
    for key in ["big_snow_days", "snow_cover_days"]:
        mm = res["metrics"].get(key, {})
        if mm.get("skipped") or not mm:
            continue
        cc = mm["comparison"]
        bullets.append(
            f"{METRICS[key]['label']}: {verdict_phrase(cc)} "
            f"({signed(cc['diff_pct_points'])}%, {pstr(cc['p_perm_one_sided'])}, {mm['n_stations']} stations)."
        )
    bullets.append(
        f"Over the whole record ({m['trend_span'][0]}&ndash;{m['trend_span'][1]} winters) the snowfall trend is "
        f"{signed(m['trend_pct_per_decade'])}% per decade ({pstr(m['trend_p'])})."
    )
    summary_html = "<ul>" + "".join(f"<li>{b}</li>" for b in bullets) + "</ul>"

    # --- Regional text ------------------------------------------------------
    reg = pd.read_csv(OUTPUT_DIR / f"regional_{PRIMARY_METRIC}.csv")
    rb = reg[(reg["season"] >= base[0]) & (reg["season"] <= base[1])]
    snowiest, leanest = rb.loc[rb["typical_value"].idxmax()], rb.loc[rb["typical_value"].idxmin()]
    regional_text = (
        f"<p>In the {period(base)} winters a typical station averaged about <strong>{tb:.0f} inches</strong> of snow "
        f"a season; in {period(recent)} it averaged about <strong>{tr:.0f} inches</strong>. The chart also shows how "
        f"much winters bounce around: within {period(base)} alone, the snowiest winter "
        f"({int(snowiest['season']) - 1}&ndash;{str(int(snowiest['season']))[-2:]}, about {snowiest['typical_value']:.0f} in) "
        f"brought {snowiest['typical_value'] / leanest['typical_value']:.1f} times as much snow as the leanest "
        f"({int(leanest['season']) - 1}&ndash;{str(int(leanest['season']))[-2:]}, about {leanest['typical_value']:.0f} in). "
        f"Against that background, the difference between the two period averages is "
        f"{signed(pct)}%, and if winters had been randomly assigned to the two periods, a gap at least this far "
        f"below zero would happen with probability {fmt_p(c['p_perm_one_sided'])} (permutation test; Mann&ndash;Whitney "
        f"{pstr(c['p_mannwhitney_one_sided'])}).</p>"
    )
    fig_regional_caption = (
        "Each bar is one winter's regional snowfall index, shown in inches for a typical station. "
        "Black lines are the period averages; the dashed line is a 10-winter running mean. "
        "Gray bars are winters outside the two periods being compared, shown for context only."
    )

    # --- Station text ---------------------------------------------------------
    stations = pd.read_csv(OUTPUT_DIR / f"stations_{PRIMARY_METRIC}.csv")
    fig_stations_caption = (
        f"Percent change in average seasonal snowfall at each station, {period(recent)} vs {period(base)}. "
        "Red = less snow, blue = more. Filled dots are changes that are statistically significant on their own "
        "(Welch t-test, p &lt; 0.05)."
    )
    station_text = (
        f"<p>{m['stations_down']} of {m['stations_total']} stations had less average snowfall in "
        f"{period(recent)} than in {period(base)}; only {m['stations_sig_down']} of those drops are statistically "
        f"significant on their own, and {m['stations_sig_up']} stations had a significant <em>increase</em>. "
        f"If there were no regional change at all, a split at least this lopsided toward declines would occur "
        f"with probability {fmt_p(m['sign_test_p'])} (sign test) &mdash; though nearby stations share storms, so this "
        f"overstates the independent evidence.</p>"
    )

    # --- Metric table -------------------------------------------------------
    rows = []
    for key, info in METRICS.items():
        mm = res["metrics"].get(key, {})
        if not mm or mm.get("skipped"):
            rows.append(f"<tr><td>{info['label']}</td><td colspan='6'>not enough stations</td></tr>")
            continue
        cc = mm["comparison"]
        u = unit_short(mm["unit"])
        rows.append(
            "<tr>"
            f"<td>{info['label']}</td><td>{mm['n_stations']}</td>"
            f"<td>{mm['typical_base']:.1f} {u}</td><td>{mm['typical_recent']:.1f} {u}</td>"
            f"<td>{signed(cc['diff_pct_points'])}%</td>"
            f"<td>{signed(cc['ci95'][0])}% to {signed(cc['ci95'][1])}%</td>"
            f"<td>{fmt_p(cc['p_perm_one_sided'])}</td>"
            "</tr>"
        )
    metric_table = (
        "<table><thead><tr><th>Measure</th><th>Stations</th>"
        f"<th>{period(base)}<br>(typical station)</th><th>{period(recent)}<br>(typical station)</th>"
        "<th>Change</th><th>95% CI</th><th>p (one-sided)</th></tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table>"
    )

    other_bits = []
    for key in ["big_snow_days", "snow_cover_days"]:
        mm = res["metrics"].get(key, {})
        if mm and not mm.get("skipped"):
            other_bits.append(f"{METRICS[key]['short']} were {verdict_phrase(mm['comparison'])}")
    if other_bits:
        other_text = (
            "<p>Looking past total snowfall: "
            + "; ".join(b[0].upper() + b[1:] if i == 0 else b for i, b in enumerate(other_bits))
            + ". "
            + (
                "So even on the measures closest to what a person notices out the window, the record agrees with him."
                if headline == "Yes"
                else "If he meant one of these rather than total snowfall, the table above is the place to look."
            )
            + "</p>"
        )
    else:
        other_text = ""

    extra = []
    for key in ["big_snow_days", "snow_cover_days"]:
        if (FIG_DIR / f"regional_{key}.png").exists():
            extra.append(
                f'<figure><img src="figures/regional_{key}.png" alt="Bar chart of {html.escape(METRICS[key]["label"].lower())} by winter.">'
                f"<figcaption>{METRICS[key]['label']}, same layout as the snowfall chart above.</figcaption></figure>"
            )
    extra_html = "".join(extra)

    # --- Sensitivity -----------------------------------------------------------
    srows = []
    for s in m["sensitivity"]:
        srows.append(
            "<tr>"
            f"<td>{period(s['period_a'])} vs {period(s['period_b'])}</td>"
            f"<td>{s['n_a']} / {s['n_b']}</td>"
            f"<td>{signed(s['diff_pct_points'])}%</td>"
            f"<td>{signed(s['ci95'][0])}% to {signed(s['ci95'][1])}%</td>"
            f"<td>{fmt_p(s['p_perm_one_sided'])}</td>"
            "</tr>"
        )
    sensitivity_table = (
        "<table><thead><tr><th>Comparison (earlier vs later)</th><th>Winters</th><th>Change</th>"
        "<th>95% CI</th><th>p (one-sided)</th></tr></thead><tbody>" + "".join(srows) + "</tbody></table>"
    )
    sig_flags = [s["p_perm_one_sided"] < ALPHA and s["diff_pct_points"] < 0 for s in m["sensitivity"]]
    main_sig = headline == "Yes"
    agree = sum(f == main_sig for f in sig_flags)
    sensitivity_text = (
        f"<p>{agree} of {len(sig_flags)} alternative framings reach the same conclusion as the headline test "
        f"({'a significant decline' if main_sig else 'no significant decline'}). "
        "The 1941&ndash;1970 and 1971&ndash;2000 rows split his baseline into the winters of his childhood and of his "
        "adulthood; the 1951&ndash;1980 vs 1991&ndash;2020 row compares two standard 30-year climate normals.</p>"
    )

    # --- Station table ------------------------------------------------------
    st_rows = []
    for _, r in stations.sort_values(["state", "name"]).iterrows():
        st_rows.append(
            "<tr>"
            f"<td>{html.escape(str(r['name']).title())}, {r['state']}</td><td>{r['id']}</td>"
            f"<td>{r['elev_m']:.0f}</td><td>{int(r['n_base'])} / {int(r['n_recent'])}</td>"
            f"<td>{r['base_mean']:.1f}</td><td>{r['recent_mean']:.1f}</td>"
            f"<td>{signed(r['pct_change'])}%</td><td>{fmt_p(r['welch_p'])}</td>"
            "</tr>"
        )
    station_table = (
        "<table><thead><tr><th>Station</th><th>GHCN ID</th><th>Elev. (m)</th><th>Winters used</th>"
        f"<th>{period(base)} avg (in)</th><th>{period(recent)} avg (in)</th><th>Change</th><th>Welch p</th>"
        "</tr></thead><tbody>" + "".join(st_rows) + "</tbody></table>"
    )

    replacements = {
        "BASELINE": period(base),
        "RECENT": period(recent),
        "HEADLINE": headline,
        "HEADLINE_CLASS": headline.lower(),
        "HEADLINE_SENTENCE": sentence,
        "TILES": tiles_html,
        "SUMMARY": summary_html,
        "FIG_REGIONAL_CAPTION": fig_regional_caption,
        "REGIONAL_TEXT": regional_text,
        "FIG_STATIONS_CAPTION": fig_stations_caption,
        "STATION_TEXT": station_text,
        "METRIC_TABLE": metric_table,
        "OTHER_METRICS_TEXT": other_text,
        "EXTRA_FIGURES": extra_html,
        "SENSITIVITY_TABLE": sensitivity_table,
        "SENSITIVITY_TEXT": sensitivity_text,
        "STATION_TABLE": station_table,
        "N_CANDIDATES": str(res["n_candidates"]),
        "N_STATIONS": str(m["n_stations"]),
        "TYPICAL_NORMAL": f"{m['avg_station_base_mean']:.0f}",
        "LATEST_SEASON": f"{res['latest_season'] - 1}&ndash;{str(res['latest_season'])[-2:]}",
        "RUN_DATE": run_date,
        "REPO_URL": repo_url(),
    }
    page = TEMPLATE.read_text(encoding="utf-8")
    for key, value in replacements.items():
        page = page.replace("{{" + key + "}}", value)
    if "{{" in page:
        leftover = page[page.index("{{") : page.index("{{") + 40]
        print(f"Warning: unfilled placeholder near {leftover!r}")

    if SITE_DIR.exists():
        shutil.rmtree(SITE_DIR)
    (SITE_DIR / "figures").mkdir(parents=True)
    for png in FIG_DIR.glob("*.png"):
        shutil.copy2(png, SITE_DIR / "figures" / png.name)
    shutil.copy2(OUTPUT_DIR / "results.json", SITE_DIR / "results.json")
    (SITE_DIR / "index.html").write_text(page, encoding="utf-8")
    (SITE_DIR / ".nojekyll").write_text("")
    print(f"Report written to {SITE_DIR / 'index.html'} (headline: {headline})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
