"""Read-only dashboard. Start with: python -m streamlit run app.py"""
import hashlib
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.dates import AutoDateLocator, ConciseDateFormatter
import numpy as np
import pandas as pd
import streamlit as st


REPOSITORY = Path(__file__).resolve().parent
LOGO = REPOSITORY / "assets" / "hargalens-mark.png"
SERIES = ["item_code", "state", "district", "premise_type"]
IDENTIFIERS = SERIES + ["project_group", "item", "unit", "week_start"]
HISTORY_COLUMNS = IDENTIFIERS + ["weekly_median_price", "record_count", "premise_count", "observed_days"]
TABLES = ["latest_dashboard.parquet", "shap_explanations.parquet", "reference_feature_importance.csv"]
FEATURE_NAMES = {
    "project_group": "Food group", "item_code": "Item", "state": "State",
    "district": "District", "premise_type": "Shop type",
    "weekly_median_price": "Weekly median price", "weekly_std_price": "Within-week price variation",
    "weekly_price_change_pct": "Change from previous week", "lag_1w_price": "Price one week earlier",
    "lag_2w_price": "Price two weeks earlier", "lag_4w_price": "Price four weeks earlier",
    "rolling_mean_4w": "Previous four-week mean", "rolling_std_4w": "Previous four-week variation",
    "rolling_mean_8w": "Previous eight-week mean", "rolling_std_8w": "Previous eight-week variation",
    "record_count": "Price observations", "premise_count": "Shops observed",
    "observed_days": "Days observed", "history_count_4w": "Observed previous weeks",
    "month": "Month", "week_sin": "Seasonal week component (sine)",
    "week_cos": "Seasonal week component (cosine)",
}


def read_json(path):
    return json.loads(Path(path).read_text())


def file_hash(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def inside(folder, relative):
    """Resolve an export path within its run; never follow an unrelated folder."""
    path = (folder / relative).resolve()
    if not path.is_relative_to(folder.resolve()):
        raise ValueError("An export points outside its saved run.")
    return path


def active_run():
    """Follow the notebooks' pointer without importing settings or creating files."""
    if os.environ.get("PRICECATCHER_DATA_DIR"):
        return Path(os.environ["PRICECATCHER_DATA_DIR"]).expanduser().resolve()
    data = REPOSITORY / "data"
    pointer = data / "active_run.txt"
    if pointer.is_file():
        relative = Path(pointer.read_text().strip())
        if relative.is_absolute() or relative.parts[:1] != ("runs",) or ".." in relative.parts:
            raise ValueError("Invalid data/active_run.txt. Restore the notebook run pointer.")
        return inside(data, relative)
    # Match the notebooks' newest-run rule, including an unfinished new run.
    for run in sorted((data / "runs").glob("run_*"), reverse=True):
        models = [run / name for name in ["06_models", "06_models_reproduction"]]
        source = run / "01_integrated"
        if any((folder / name).is_file() for folder in models
               for name in ["best_model.joblib", "model_manifest.json"]) or (
                (source / "integrated_prices.parquet").is_file() and (source / "source_manifest.json").is_file()):
            return run
    return data


def export_signature(run):
    """Invalidate Streamlit's cache when an export or frozen history file changes."""
    entry_path = run / "08_dashboard" / "dashboard_manifest.json"
    entry = read_json(entry_path)
    cache = inside(run / "08_dashboard", entry["cache_dir"])
    paths = [entry_path, cache / "explanation_manifest.json"]
    paths += [cache / name for name in TABLES]
    paths += [run / "experiments" / "holdout" / "05_modelling" / f"{split}.parquet"
              for split in ["train", "validation", "test"]]
    return tuple((str(path), path.stat().st_size, path.stat().st_mtime_ns) for path in paths)


@st.cache_data(show_spinner=False, max_entries=2)
def load_exports(run_name, signature):
    """Verify cached files, then read display data only: no models or labels."""
    run = Path(run_name)
    dashboard_dir = run / "08_dashboard"
    entry = read_json(dashboard_dir / "dashboard_manifest.json")
    if entry["schema_version"] != 1:
        raise ValueError("Unsupported dashboard export. Use the current Notebook 08.")
    cache = inside(dashboard_dir, entry["cache_dir"])
    cache_path = cache / "explanation_manifest.json"
    if file_hash(cache_path) != entry["cache_manifest_sha256"]:
        raise ValueError("Dashboard cache manifest changed. Run All in Notebook 08 to verify its exports.")
    manifest = read_json(cache_path)
    model = entry["forecast_model"]
    if manifest["forecast_model"] != model or manifest["protocol"]["forecast_sha256"] != model["sha256"]:
        raise ValueError("Dashboard files describe different forecasting models.")
    for name in TABLES:
        if inside(dashboard_dir, entry["files"][name]) != cache / name:
            raise ValueError("Dashboard table paths disagree with the selected cache.")
        if file_hash(cache / name) != manifest["output_sha256"][name]:
            raise ValueError(f"Cached table changed: {name}. Run All in Notebook 08.")
    latest = pd.read_parquet(cache / TABLES[0])
    explanations = pd.read_parquet(cache / TABLES[1])
    importance = pd.read_csv(cache / TABLES[2])
    week = pd.Timestamp(entry["latest_complete_week"])
    cutoff = model["score_cutoff"]
    if week.dayofweek != 0 or len(latest) != entry["latest_supported_rows"]:
        raise ValueError("Snapshot date or coverage disagrees with its export.")
    if latest.duplicated(SERIES + ["week_start"]).any() or latest.forecast_id.duplicated().any():
        raise ValueError("Duplicate forecast identifiers.")
    if not latest.week_start.eq(week).all() or not latest.next_week_end.eq(week + pd.Timedelta(days=13)).all():
        raise ValueError("Forecast dates disagree with the snapshot week.")
    if not latest.forecast_model_version.eq(model["version"]).all():
        raise ValueError("Forecast rows belong to a different model.")
    if not latest.model_score.between(0, 1).all() or not latest.predicted_surge.eq(latest.model_score.ge(cutoff)).all():
        raise ValueError("Forecast scores or alert cutoff are inconsistent.")
    if not latest.is_anomaly.isin([True, False]).all() or not latest.has_cached_explanation.isin([True, False]).all():
        raise ValueError("Anomaly/explanation flags must be explicit booleans.")
    local = explanations.loc[explanations.cohort.eq("latest")].set_index("explanation_id")
    if not local.index.is_unique or not set(local.index).issubset(set(latest.forecast_id)):
        raise ValueError("Local explanations do not match the forecasts.")
    if not latest.has_cached_explanation.eq(latest.forecast_id.isin(local.index)).all():
        raise ValueError("Explanation availability disagrees with the cache.")
    shap_columns = ["shap__" + name for name in manifest["protocol"]["features"]]
    np.testing.assert_allclose(explanations.base_value + explanations[shap_columns].sum(axis=1),
                               explanations.raw_output, rtol=1e-5, atol=1e-5)
    raw = explanations.raw_output.to_numpy(dtype=float)
    linked = raw if manifest["protocol"]["units"] == "uncalibrated model-score units" else 1 / (1 + np.exp(-np.clip(raw, -700, 700)))
    np.testing.assert_allclose(linked, explanations.model_score, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(latest.set_index("forecast_id").loc[local.index, "model_score"],
                               local.model_score, rtol=1e-6, atol=1e-8)
    # Use frozen original 05 inputs for historical prices. Do not read outcomes.
    history_parts = []
    for split in ["train", "validation", "test"]:
        path = run / "experiments" / "holdout" / "05_modelling" / f"{split}.parquet"
        expected = manifest["upstream_sha256"][f"prepared/holdout/{split}.parquet"]
        if file_hash(path) != expected:
            raise ValueError(f"Frozen {split} history changed. Restore its original snapshot; do not regenerate this experiment.")
        history_parts.append(pd.read_parquet(path, columns=HISTORY_COLUMNS))
    history_parts.append(latest[HISTORY_COLUMNS])
    history = pd.concat(history_parts, ignore_index=True).drop_duplicates()
    if history.duplicated(SERIES + ["week_start"]).any():
        raise ValueError("Conflicting weekly prices in the frozen history.")
    history["week_start"] = pd.to_datetime(history.week_start)
    if not history.week_start.dt.dayofweek.eq(0).all() or history.week_start.gt(week).any():
        raise ValueError("Historical dates fall outside the displayed snapshot.")
    if not (np.isfinite(history.weekly_median_price) & history.weekly_median_price.gt(0)).all():
        raise ValueError("Historical prices must be positive and finite.")
    return entry, manifest, latest, explanations, importance, history


def date_span(start, end):
    return f"{pd.Timestamp(start):%d %b %Y} – {pd.Timestamp(end):%d %b %Y}"


def filter_data(history, latest):
    """Offer historical locations too, so an absent latest forecast stays visible."""
    with st.sidebar:
        st.subheader("Find your local prices")
        st.caption("Start with a state, then narrow down to an item or shop type.")
        def reset_filters():
            for key in ["state", "district", "project_group", "item", "premise_type"]:
                st.session_state.pop(key, None)
        st.button("Reset filters", on_click=reset_filters, width="stretch")
        chosen = history.copy()
        forecasts = latest.copy()
        for label, column in [("State", "state"), ("District", "district"), ("Food group", "project_group"),
                              ("Item", "item"), ("Shop type", "premise_type")]:
            options = sorted(chosen[column].dropna().unique().tolist())
            selection = st.selectbox(label, ["All"] + options, key=column)
            if selection != "All":
                chosen = chosen.loc[chosen[column].eq(selection)]
                forecasts = forecasts.loc[forecasts[column].eq(selection)]
        st.caption("Only locations with eligible observations are listed. A missing forecast does not mean prices will stay stable.")
    return chosen, forecasts


def group_overview(latest):
    """Compare alert counts, never combine prices with different items or units."""
    if latest.empty:
        return
    st.subheader("Where are the review alerts?")
    counts = latest.groupby("project_group").agg(total=("forecast_id", "size"),
                                                alerts=("predicted_surge", "sum"))
    counts["Other monitored series"] = counts.total - counts.alerts
    counts = counts.rename(columns={"alerts": "Review alerts"}).reset_index()
    st.bar_chart(counts, x="project_group", y=["Review alerts", "Other monitored series"],
                 x_label="Food group", y_label="Number of local price series", horizontal=True,
                 color=["#d98c28", "#2795a0"], stack=True, height=120 + 28 * len(counts))
    st.caption("Amber = a next-week alert to investigate. Teal = below the alert cutoff, which still carries uncertainty. Counts reflect your filters and available observations, not each group's population risk.")


def review_list(latest):
    st.subheader("Your local price watchlist")
    view = st.radio("Show", ["All monitored prices", "Review alerts", "Unusual patterns"], horizontal=True)
    shown = latest.copy()
    if view == "Review alerts":
        shown = shown.loc[shown.predicted_surge.eq(1)]
    elif view == "Unusual patterns":
        shown = shown.loc[shown.is_anomaly]
    if shown.empty:
        st.info("No matching rows in this view. Below-cutoff scores do not rule out future surges.")
        return
    shown = shown.sort_values(["model_score", "forecast_id"], ascending=[False, True])
    table = shown[["project_group", "item", "state", "district", "premise_type", "weekly_median_price", "unit",
                   "model_score", "predicted_surge", "is_anomaly", "has_cached_explanation"]].copy()
    table["predicted_surge"] = table.predicted_surge.map({1: "Review next week", 0: "Below alert cutoff"})
    table["is_anomaly"] = table.is_anomaly.map({True: "Flagged", False: "Not flagged"})
    table["has_cached_explanation"] = table.has_cached_explanation.map({True: "Available", False: "Not cached"})
    table = table.rename(columns={"project_group": "Food group", "item": "Item", "state": "State", "district": "District",
        "premise_type": "Shop type", "weekly_median_price": "Weekly median (RM)", "unit": "Unit", "model_score": "Review score",
        "predicted_surge": "Next-week alert", "is_anomaly": "Unusual pattern", "has_cached_explanation": "Explanation"})
    st.dataframe(table, hide_index=True, width="stretch", height=380, column_config={
        "Weekly median (RM)": st.column_config.NumberColumn(format="%.2f", help="The middle observed price this week, per listed unit. This is not the cheapest available price."),
        "Review score": st.column_config.NumberColumn(format="%.3f", help="A model score from 0 to 1. Higher scores receive greater review priority; 0.9 is not a verified 90% chance.")})
    st.caption("A review alert concerns a possible increase of at least 10% next week. An unusual-pattern flag concerns the observed price behaviour. Review scores help prioritise investigation; they are not verified probabilities.")
    download = table.copy()
    download["Observed week starting"] = shown.week_start.dt.strftime("%Y-%m-%d").to_numpy()
    download["Forecast week starting"] = (shown.week_start + pd.Timedelta(days=7)).dt.strftime("%Y-%m-%d").to_numpy()
    download["Forecast week ending"] = shown.next_week_end.dt.strftime("%Y-%m-%d").to_numpy()
    download["Forecast model version"] = shown.forecast_model_version.to_numpy()
    st.download_button("Download these rows (CSV)", download.to_csv(index=False).encode("utf-8-sig"),
                       file_name="hargalens_price_review.csv", mime="text/csv")


def explain_case(row, explanations, units):
    case = explanations.loc[explanations.cohort.eq("latest") & explanations.explanation_id.eq(row.forecast_id)]
    st.subheader("What influenced this review score?")
    if case.empty:
        st.info("Explanation unavailable for this case. Offline explanations cover only the 200 highest-scoring latest cases; changing filters does not generate more.")
        return
    case = case.iloc[0]
    values = pd.Series({name.removeprefix("shap__"): float(case[name]) for name in case.index if name.startswith("shap__")})
    raising = values.loc[values.gt(0)].nlargest(2).index
    if len(raising):
        st.write("The strongest contributions raising this score are **" + "** and **".join(FEATURE_NAMES.get(name, name) for name in raising) + "**.")
    largest = values.abs().nlargest(8).index
    shown = values.loc[largest].rename(index=FEATURE_NAMES).copy()
    shown.loc["Other features (combined)"] = values.drop(largest).sum()
    shown = shown.iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, 3.7))
    ax.barh(shown.index, shown.values, color=np.where(shown.values >= 0, "#ca681b", "#267b9b"))
    ax.axvline(0, color="#899397", linewidth=0.7)
    ax.set_xlabel(f"Contribution ({units})")
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)
    st.caption("Orange raises this model's output; blue lowers it. Contributions explain the model, not economic causes or percentage-point changes in surge probability.")
    with st.expander("How this explanation is calculated"):
        st.write("SHAP divides the model output into a baseline and feature contributions. Related price-history features can share information, so the bars should not be treated as independent economic causes.")
        st.caption(f"Baseline {case.base_value:.3f} + contributions = raw output {case.raw_output:.3f}; review score {case.model_score:.3f}. Units: {units}.")


def explore_series(history, latest, explanations, week, units):
    series = history[IDENTIFIERS[:-1]].drop_duplicates().sort_values(["item", "state", "district", "premise_type", "item_code"])
    if series.empty:
        st.info("No eligible history matches these filters.")
        return
    labels = (series.item + " · " + series.unit + " · " + series.state + " / " + series.district
              + " · " + series.premise_type + " · #" + series.item_code.astype(str)).tolist()
    default = 0
    if not latest.empty:
        first = latest.sort_values("model_score", ascending=False).iloc[0]
        matches = series[SERIES].eq(first[SERIES]).all(axis=1).to_numpy()
        default = int(np.flatnonzero(matches)[0])
    index = st.selectbox("Choose an item and location", range(len(series)), index=default, format_func=lambda i: labels[i])
    selected = series.iloc[index]
    old_mask = pd.Series(True, index=history.index)
    now_mask = pd.Series(True, index=latest.index)
    for key in SERIES:
        old_mask &= history[key].eq(selected[key])
        now_mask &= latest[key].eq(selected[key])
    past = history.loc[old_mask].sort_values("week_start")
    current = latest.loc[now_mask]
    st.subheader(selected["item"])
    st.caption(f"{selected.state} / {selected.district} · {selected.premise_type} · Price per {selected.unit}")
    window = st.radio("Price history", ["Last 12 weeks", "Last 26 weeks", "All history"], index=1, horizontal=True)
    start = past.week_start.min()
    if window != "All history":
        weeks = 12 if window == "Last 12 weeks" else 26
        start = max(start, week - pd.Timedelta(weeks=weeks - 1))
    calendar = pd.date_range(start, week, freq="W-MON")
    prices = past.set_index("week_start").weekly_median_price.reindex(calendar)
    fig, ax = plt.subplots(figsize=(9, 3.4))
    ax.plot(prices.index, prices.values, color="#087f8c", linewidth=1.8, marker=".", markersize=4)
    ax.set(ylabel=f"RM per {selected.unit}", xlabel="Week starting Monday")
    # A single observation must not expand the axis into unrelated years.
    ax.set_xlim(calendar.min() - pd.Timedelta(days=3), week + pd.Timedelta(days=3))
    ticks = AutoDateLocator(minticks=3, maxticks=6)
    ax.xaxis.set_major_locator(ticks)
    ax.xaxis.set_major_formatter(ConciseDateFormatter(ticks))
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    if prices.notna().any():
        st.pyplot(fig)
    else:
        st.info("No eligible observations in this time window. Choose All history to see older available prices.")
    plt.close(fig)
    st.caption("Eligible original-study weekly medians, plus the latest snapshot. Gaps are left blank; missing and excluded weeks are not filled. This is not all raw observations or a fixed-shop price index, and does not include the added 2024 training history.")
    if current.empty:
        st.warning("No supported forecast for this local series in the snapshot week. Its older price history is not a current prediction.")
        return
    row = current.iloc[0]
    cols = st.columns(4)
    change = row.get("weekly_price_change_pct", np.nan)
    delta = f"{change:+.1f}% vs previous week" if pd.notna(change) else None
    cols[0].metric("Weekly median", f"RM {row.weekly_median_price:.2f}", delta=delta, delta_color="inverse")
    cols[1].metric("Review score", f"{row.model_score:.3f}", help="Higher means greater review priority, not a verified event probability.")
    cols[2].metric("Next-week alert", "Review" if row.predicted_surge else "Below cutoff")
    cols[3].metric("Unusual pattern", "Flagged" if row.is_anomaly else "Not flagged")
    st.caption(f"Unit: {row.unit}. Current week: {int(row.record_count)} price records, {int(row.premise_count)} shops, {int(row.observed_days)} observed days. No alert does not guarantee stable prices.")
    explain_case(row, explanations, units)


def model_and_coverage(entry, latest, importance, history):
    model = entry["forecast_model"]
    st.subheader("Evidence for the displayed forecasting model")
    version = "Longer history (2024–2025)" if model["version"] == "longer_2024_2025" else "Original 2025 reference"
    st.write(f"**{version}** · training {model['training_start']} through {model['training_end']}")
    st.write(model["evidence_status"])
    period = model["validation_period"]
    st.caption(f"Validation outcomes after {period['after']} through {period['through']}. These are validation results, not the original model's separate test results.")
    metrics = model["validation_metrics"]
    st.write("Precision tells us how many alerts were correct. Recall tells us how many observed surges were found. F1 balances the two; average precision measures ranking across score cutoffs.")
    cols = st.columns(4)
    for col, name, label in zip(cols, ["average_precision", "f1", "precision", "recall"],
                                ["Average precision", "F1", "Precision", "Recall"]):
        value = metrics.get(name)
        col.metric(label, "Unavailable" if value is None or pd.isna(value) else f"{value:.3f}")
    st.caption(f"Frozen score cutoff: {model['score_cutoff']:.5f}. The 10% price-increase target is a separate definition. No dashboard filter changes either threshold.")
    precision = metrics.get("precision")
    if precision is not None and pd.notna(precision):
        st.warning(f"In this validation period, {(1 - precision):.1%} of alerts were false alarms. Alerts require investigation.")
    if model["version"] == "longer_2024_2025":
        st.write("The saved longer-history study improves overall validation ranking, mainly for vegetables. Eggs and flour deteriorate. Sugar has no positive training/validation examples; cooking-oil positives are scarce. Independent test performance for this candidate remains unconfirmed.")
    st.subheader("Coverage in your current filters")
    counts = latest.groupby("project_group").agg(supported_rows=("forecast_id", "size"),
        surge_alerts=("predicted_surge", "sum"), unusual_patterns=("is_anomaly", "sum"),
        cached_explanations=("has_cached_explanation", "sum")).reset_index()
    st.dataframe(counts.rename(columns={"project_group": "Food group", "supported_rows": "Supported rows",
        "surge_alerts": "Surge alerts", "unusual_patterns": "Unusual patterns", "cached_explanations": "Cached explanations"}),
        hide_index=True, width="stretch")
    st.caption(f"{history[SERIES].drop_duplicates().shape[0]:,} local series have eligible historical rows in these filters. Supported-row counts are not population coverage. Places/items outside this study cannot be assessed here.")
    st.write("The separate anomaly detector uses a January–December 2025 reference. Its flags identify unusual patterns; there is no labelled anomaly ground truth to measure accuracy.")
    with st.expander("Feature summary and study limitations"):
        display = importance.copy()
        display["feature"] = display.feature.map(lambda name: FEATURE_NAMES.get(name, name))
        st.dataframe(display, hide_index=True, width="stretch")
        st.caption("Mean absolute SHAP on a random sample of up to 200 original 2025 training inputs. This is descriptive, not representative importance for the full longer-history training population. Correlated price features complicate interpretation.")
        st.write("Sparse local coverage, changing shop composition and unresolved suspicious prices limit reliability. Fuel, weather, imports and policy drivers are not model inputs. No improvement in affordability, inequality or stakeholder adoption has been demonstrated.")
        st.code(f"Forecast version: {model['version']}\nCheckpoint: {model['sha256']}\nAnomaly reference: {entry['anomaly_model']['reference_period']}")


def main():
    st.set_page_config(page_title="HargaLens | Local food prices", page_icon=str(LOGO), layout="wide")
    st.logo(str(LOGO), size="large", icon_image=str(LOGO))
    st.markdown("""<style>
        .block-container {max-width: 1400px; padding-top: 2rem;}
        [data-testid="stMetric"] {background: rgba(8, 127, 140, .08); padding: 1rem; border-radius: .6rem; border: 1px solid rgba(128, 128, 128, .25);}
        h1, h2, h3 {letter-spacing: -.025em;}
        </style>""", unsafe_allow_html=True)
    mark, wordmark = st.columns([1, 10], vertical_alignment="center")
    with mark:
        st.image(str(LOGO), width=82)
    with wordmark:
        st.title("HargaLens")
        st.caption("See local prices clearly. · Malaysian food price monitoring")
    try:
        run = active_run()
        with st.spinner("Checking saved dashboard exports…"):
            entry, manifest, latest, explanations, importance, history = load_exports(str(run), export_signature(run))
    except (OSError, ValueError, KeyError, TypeError, AssertionError) as error:
        st.error("The saved dashboard inputs are missing or inconsistent. No forecasts were substituted.")
        st.write("Open Notebook 08 in your project kernel and Run All. It needs the completed 06 and 07 outputs. Do not delete data or restart Notebook 01 just to reopen the dashboard.")
        with st.expander("Details for troubleshooting"):
            st.code(str(error))
        st.stop()
    week = pd.Timestamp(entry["latest_complete_week"])
    st.info(f"Historical demonstration · Observed week: {date_span(week, week + pd.Timedelta(days=6))} · Forecast week: {date_span(week + pd.Timedelta(days=7), week + pd.Timedelta(days=13))}. These are not live warnings.")
    history, latest = filter_data(history, latest)
    cols = st.columns(4)
    for col, label, value, help_text in zip(cols,
        ["Local price series", "Next-week review alerts", "Unusual price patterns", "Ready explanations"],
        [len(latest), int(latest.predicted_surge.sum()), int(latest.is_anomaly.sum()), int(latest.has_cached_explanation.sum())],
        ["One item in one district and shop type, with enough data for this saved forecast.",
         "Cases above the fixed review cutoff for a possible price rise of at least 10% next week. Some alerts will be wrong.",
         "Observed patterns flagged by a separate anomaly detector. This is not the next-week forecast.",
         "Local explanations prepared offline for selected cases. A missing explanation is shown explicitly."]):
        col.metric(label, f"{value:,}", help=help_text)
    st.caption("Choose a location → review an alert → explore its price history. Higher scores mean greater review priority, not verified probabilities.")
    with st.expander("First visit? A quick guide"):
        steps = st.columns(3)
        steps[0].markdown("**1 · Find your area**")
        steps[0].write("Use the sidebar to choose your state, district and food item. Counts and charts follow your filters.")
        steps[1].markdown("**2 · Understand the signal**")
        steps[1].write("A review alert concerns next week. An unusual-pattern flag concerns observed prices. Either signal needs investigation.")
        steps[2].markdown("**3 · Check the context**")
        steps[2].write("Open Explore prices for the unit, price history, observation counts and available explanation. A missing forecast is not a promise of stable prices.")
    if latest.empty:
        st.warning("No supported latest forecasts match these filters. Older history may still be available. This does not mean no surge risk.")
    review, details, evidence = st.tabs(["Overview", "Explore prices", "About the model"])
    with review:
        group_overview(latest)
        review_list(latest)
    with details:
        explore_series(history, latest, explanations, week, manifest["protocol"]["units"])
    with evidence:
        model_and_coverage(entry, latest, importance, history)
    st.caption("HargaLens · Source: Malaysian PriceCatcher · Research prototype · Observed local medians, not a national price index. No affordability benefit has been established.")


if __name__ == "__main__":
    main()
