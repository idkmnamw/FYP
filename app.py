"""Read-only dashboard. Start with: python -m streamlit run app.py"""
import hashlib
import json
import os
from html import escape
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
GROUP_GUIDANCE = {
    "Vegetables": "Strongest validation evidence, but many alerts were false. Check local price history.",
    "Chicken": "Limited evidence; most validation surges were missed. Below-cutoff scores need care.",
    "Eggs": "Weak detection; validation results declined with longer history. Prioritise observed prices.",
    "Flour": "Weak detection; validation results declined with longer history. Prioritise observed prices.",
    "Rice": "Very limited evidence: only two validation surges. Avoid broad reliability claims.",
    "Cooking Oil": "Very limited evidence: only one validation surge. Detection reliability is unconfirmed.",
    "Sugar": "No positive training/validation cases. Surge detection could not be assessed.",
}
GUIDANCE_REFERENCE = "Saved FYP longer-history validation study, January–March 2026; not per-case confidence."


def group_guidance(group, version):
    """Qualitative findings from saved study 12, not per-case confidence estimates."""
    if version != "longer_2024_2025":
        return "Consult this model's own saved evaluation; longer-history study guidance does not apply."
    return GROUP_GUIDANCE.get(group, "No group-specific study guidance is available. Check observed prices and coverage.")


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
    entry_path = dashboard_entry(run)
    entry = read_json(entry_path)
    cache = inside(run / "08_dashboard", entry["cache_dir"])
    paths = [entry_path, cache / "explanation_manifest.json"]
    paths += [cache / name for name in TABLES]
    if "refresh" in entry:
        record_path = inside(run, entry["refresh"]["source_manifest"])
        paths += [run / "08_dashboard" / "dashboard_manifest.json", record_path,
                  record_path.parent / "selected_daily_prices.parquet", cache / "refresh_history.parquet"]
    paths += [run / "experiments" / "holdout" / "05_modelling" / f"{split}.parquet"
              for split in ["train", "validation", "test"]]
    return tuple((str(path), path.stat().st_size, path.stat().st_mtime_ns) for path in paths)


def dashboard_entry(run):
    """A published refresh takes precedence; inconsistent files are never hidden."""
    folder = run / "08_dashboard"
    refresh = folder / "refresh_manifest.json"
    return refresh if refresh.is_file() else folder / "dashboard_manifest.json"


@st.cache_data(show_spinner=False, max_entries=2)
def load_exports(run_name, signature):
    """Verify cached files, then read display data only: no models or labels."""
    run = Path(run_name)
    dashboard_dir = run / "08_dashboard"
    entry = read_json(dashboard_entry(run))
    if entry["schema_version"] != 1:
        raise ValueError("Unsupported dashboard export. Use the current Notebook 12.")
    cache = inside(dashboard_dir, entry["cache_dir"])
    cache_path = cache / "explanation_manifest.json"
    if file_hash(cache_path) != entry["cache_manifest_sha256"]:
        raise ValueError("Dashboard cache manifest changed. Run All in Notebook 12 to verify its exports.")
    manifest = read_json(cache_path)
    model = entry["forecast_model"]
    if manifest["forecast_model"] != model or manifest["protocol"]["forecast_sha256"] != model["sha256"]:
        raise ValueError("Dashboard files describe different forecasting models.")
    tables = TABLES + (["refresh_history.parquet"] if "refresh" in entry else [])
    for name in tables:
        if inside(dashboard_dir, entry["files"][name]) != cache / name:
            raise ValueError("Dashboard table paths disagree with the selected cache.")
        if file_hash(cache / name) != manifest["output_sha256"][name]:
            raise ValueError(f"Cached table changed: {name}. Rerun the notebook that published this snapshot (12 or 13).")
    if "refresh" in entry:
        refresh = entry["refresh"]
        original_path = dashboard_dir / "dashboard_manifest.json"
        if (manifest.get("refresh") != refresh or file_hash(original_path) != refresh["original_entry_sha256"]
                or refresh["origin_week"] != entry["latest_complete_week"]):
            raise ValueError("Refresh metadata or original export changed. Restore the matching snapshot.")
        original = read_json(original_path)
        if original["forecast_model"] != model or original["anomaly_model"] != entry["anomaly_model"]:
            raise ValueError("A refresh cannot replace the frozen forecasting or anomaly model.")
        record_path = inside(run, refresh["source_manifest"])
        if file_hash(record_path) != refresh["source_manifest_sha256"]:
            raise ValueError("Refresh source audit changed.")
        record = read_json(record_path)
        if file_hash(record_path.parent / "selected_daily_prices.parquet") != record["selected_sha256"]:
            raise ValueError("Refresh source cache changed. Restore its matching snapshot.")
        if (record["policy"]["end"] != str((pd.Timestamp(refresh["origin_week"]) + pd.Timedelta(days=6)).date())
                or refresh["registered_period_loaded"] or refresh["model_fitted"] or refresh["cutoff_changed"]):
            raise ValueError("Refresh dates or frozen-model policy are inconsistent.")
        # Display the verified audit only; the dashboard does not read raw price rows.
        entry = {**entry, "refresh_audit": record["monthly_audit"],
                 "original_supported_rows": original["latest_supported_rows"]}
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
    explanation_columns = shap_columns + ["base_value", "raw_output", "model_score"]
    if not np.isfinite(explanations[explanation_columns].to_numpy(dtype=float)).all():
        raise ValueError("Cached explanations contain missing or infinite values. Restore or rebuild the matching export.")
    if not np.isfinite(latest.anomaly_score.to_numpy(dtype=float)).all():
        raise ValueError("Cached anomaly scores contain missing or infinite values.")
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
    if "refresh" in entry:
        refreshed_history = pd.read_parquet(cache / "refresh_history.parquet", columns=HISTORY_COLUMNS)
        original_week = pd.Timestamp(original["latest_complete_week"])
        if not refreshed_history.week_start.gt(original_week).all():
            raise ValueError("Refresh history overlaps the frozen original history.")
        history_parts.append(refreshed_history)
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


def snapshot_freshness(week, today=None):
    """Describe observation age against closed calendar weeks, using Malaysia time."""
    day = pd.Timestamp.now(tz="Asia/Kuala_Lumpur") if today is None else pd.Timestamp(today)
    if day.tzinfo is not None:
        day = day.tz_convert("Asia/Kuala_Lumpur").tz_localize(None)
    day = day.normalize()
    week = pd.Timestamp(week)
    end = week + pd.Timedelta(days=6)
    if week.tzinfo is not None or week.dayofweek != 0 or end >= day:
        raise ValueError("The displayed snapshot must cover a fully closed Monday–Sunday week.")
    expected = day - pd.Timedelta(days=day.dayofweek + 7)
    age = (day - end).days
    behind = (expected - week).days // 7
    status = "latest fully closed calendar week" if week == expected else f"{behind} calendar week{'s' if behind != 1 else ''} behind the latest closed week"
    return f"As of {day:%d %b %Y}: observations ended {age} day{'s' if age != 1 else ''} ago; {status}. This does not establish complete reporting or timely forecasts."


def price_history_chart(prices, unit, height=3.4, xlabel="Week starting Monday"):
    """Draw an exact-week price series without filling gaps or extrapolating prices."""
    fig, ax = plt.subplots(figsize=(9, height))
    ax.plot(prices.index, prices.values, color="#087f8c", linewidth=1.8, marker=".", markersize=4)
    ax.set(ylabel=f"RM per {unit}", xlabel=xlabel)
    ax.set_xlim(prices.index.min() - pd.Timedelta(days=3), prices.index.max() + pd.Timedelta(days=3))
    ticks = AutoDateLocator(minticks=3, maxticks=6)
    ax.xaxis.set_major_locator(ticks)
    ax.xaxis.set_major_formatter(ConciseDateFormatter(ticks))
    ax.grid(axis="y", alpha=.2)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


def filter_data(history, latest):
    """Offer historical locations too, so an absent latest forecast stays visible."""
    with st.sidebar:
        st.subheader("Find your local prices")
        st.caption("Start with a state, then narrow down to an item or shop type.")
        def reset_filters():
            for key in ["state", "district", "project_group", "item", "premise_type"]:
                st.session_state.pop(key, None)
        st.button("Reset filters", on_click=reset_filters, width="stretch")
        chosen = history
        forecasts = latest
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


def spotlight(history, latest, explanations, model, refresh=None):
    """Turn fixed saved results into a readable case story and portable brief."""
    if latest.empty:
        return
    st.subheader("HargaLens Spotlight")
    st.caption("Start with a price story. These are the three highest-scoring cases in your filters; a high score is a reason to investigate, not a confirmed surge.")
    candidates = latest.sort_values(["model_score", "forecast_id"], ascending=[False, True]).head(3)
    if st.session_state.get("spotlight_id") not in candidates.forecast_id.tolist():
        st.session_state["spotlight_id"] = candidates.iloc[0].forecast_id

    def choose_story(forecast_id):
        st.session_state["spotlight_id"] = forecast_id

    cards = st.columns(len(candidates))
    for card, (_, row) in zip(cards, candidates.iterrows()):
        with card.container(border=True):
            st.markdown(f"**{row['item']}**")
            st.caption(f"{row.state} / {row.district} · {row.premise_type}")
            change = row.get("weekly_price_change_pct", np.nan)
            delta = f"{change:+.1f}% vs previous week" if pd.notna(change) and np.isfinite(change) else None
            st.metric("Observed weekly price", f"RM {row.weekly_median_price:.2f}", delta=delta, delta_color="inverse",
                      help="The median of the observed prices. The change compares observed weeks; neither value predicts next week's price.")
            st.caption(f"Per {row.unit} · Review score {row.model_score:.3f}")
            status_class = "signal-review" if row.predicted_surge else "signal-monitor"
            status_label = "Alert to review" if row.predicted_surge else "Below alert cutoff"
            st.markdown(f'<span class="signal {status_class}">{status_label}</span>', unsafe_allow_html=True)
            if row.is_anomaly:
                st.markdown('<span class="signal signal-unusual">Unusual observed pattern</span>', unsafe_allow_html=True)
            st.button("Read this price story", key=f"story_{row.forecast_id}", width="stretch",
                      type="primary" if row.forecast_id == st.session_state["spotlight_id"] else "secondary",
                      on_click=choose_story, args=(row.forecast_id,))

    row = candidates.loc[candidates.forecast_id.eq(st.session_state["spotlight_id"])].iloc[0]
    week = pd.Timestamp(row.week_start)
    forecast_period = date_span(week + pd.Timedelta(days=7), row.next_week_end)
    status = "Review next week" if row.predicted_surge else "Below the alert cutoff"
    pattern = "Flagged as unusual" if row.is_anomaly else "Not flagged as unusual"
    headline = f"{row['item']} · {row.state} / {row.district}"
    coverage = f"{int(row.record_count)} price records from {int(row.premise_count)} shops across {int(row.observed_days)} observed days"
    st.markdown(f"#### Price story: {headline}")
    st.write(f"The observed weekly median was **RM {row.weekly_median_price:.2f} per {row.unit}**. "
             f"The saved model marks this case **{status.lower()}** for **{forecast_period}**. "
             f"The separate pattern detector says **{pattern.lower()}**.")
    guidance = group_guidance(row.project_group, model["version"])
    mask = history[SERIES].eq(row[SERIES]).fillna(False).all(axis=1)
    calendar = pd.date_range(week - pd.Timedelta(weeks=11), week, freq="W-MON")
    prices = history.loc[mask].set_index("week_start").weekly_median_price.reindex(calendar)
    chart, context = st.columns([2, 1], gap="large")
    with chart:
        price_history_chart(prices, row.unit, height=2.6, xlabel="Recent weekly medians; gaps remain blank")
    with context:
        st.markdown("**Before you act**")
        st.write(coverage + ".")
        st.caption("Shop composition and sparse observations can affect local medians.")
        st.caption("Saved validation study guidance: " + guidance)
    local = explanations.loc[explanations.cohort.eq("latest") & explanations.explanation_id.eq(row.forecast_id)]
    if local.empty:
        drivers = "No cached local explanation is available. No explanation was inferred or substituted."
    else:
        case = local.iloc[0]
        contributions = pd.Series({name.removeprefix("shap__"): float(case[name])
                                   for name in case.index if name.startswith("shap__")})
        strongest = contributions.abs().nlargest(3).index
        drivers = "; ".join(f"{FEATURE_NAMES.get(name, name)} {'raises' if contributions[name] > 0 else 'lowers' if contributions[name] < 0 else 'does not change'} the model output"
                            for name in strongest) + "."
    if local.empty:
        st.caption("No cached explanation is available for this case.")
    with st.expander("Why this score?"):
        st.write("**What influenced the score:** " + drivers)
        st.caption("The score is not a verified chance of a surge. Contributions describe this model, not economic causes. The 10% target defines a surge; it is not a predicted price increase.")
    # Self-contained HTML can be shared or printed without a running dashboard.
    facts = {"Forecast case ID": row.forecast_id, "Observed week": date_span(week, week + pd.Timedelta(days=6)), "Forecast week": forecast_period,
             "Observed weekly median": f"RM {row.weekly_median_price:.2f} per {row.unit}",
             "Shop type": row.premise_type, "Next-week status": status, "Observed pattern": pattern,
             "Uncalibrated review score": f"{row.model_score:.3f}", "Frozen review cutoff": f"{model['score_cutoff']:.5f}",
             "Observation support": coverage, "Forecast model": model["version"], "Checkpoint SHA-256": model["sha256"],
             "Evidence status": model["evidence_status"]}
    facts["Food-group study guidance"] = guidance
    facts["Snapshot freshness"] = snapshot_freshness(week)
    facts["Guidance reference"] = GUIDANCE_REFERENCE if model["version"] == "longer_2024_2025" else "Consult this model's own evaluation."
    if refresh:
        facts["Snapshot type"] = refresh["snapshot_type"]
        facts["Scores generated at"] = refresh["generated_at"]
        facts["Source quality"] = refresh["source_correctness"]
    precision = model["validation_metrics"].get("precision")
    if precision is not None and pd.notna(precision):
        facts["Selected-model validation precision"] = f"{precision:.1%} of validation alerts were correct; this is not a probability for this case"
    facts_html = "".join(f"<tr><th>{escape(key)}</th><td>{escape(str(value))}</td></tr>" for key, value in facts.items())
    weeks_html = "".join(f"<tr><td>{date:%d %b %Y}</td><td>{'No eligible observation' if pd.isna(value) else f'RM {value:.2f}'}</td></tr>"
                         for date, value in prices.items())
    brief = f"""<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
    <title>HargaLens price story</title><style>body{{font:16px/1.6 system-ui,sans-serif;max-width:850px;margin:32px auto;padding:0 20px;color:#163341;background:white}}
    h1,h2{{color:#087f8c}}table{{width:100%;border-collapse:collapse}}th,td{{padding:8px;text-align:left;border-bottom:1px solid #dde5e5;overflow-wrap:anywhere}}th{{width:36%}}small{{color:#455f68}}</style>
    <h1>HargaLens · Price story</h1><h2>{escape(headline)}</h2><p><strong>Historical research demonstration. This is not a live warning.</strong></p>
    <table>{facts_html}</table><h2>What influenced the score?</h2><p>{escape(drivers)}</p>
    <h2>Recent eligible weekly medians</h2><p>Unit: {escape(str(row.unit))}. Missing weeks are not filled.</p><table><tr><th>Week starting</th><th>Observed median</th></tr>{weeks_html}</table>
    <h2>Read with care</h2><p>The score is uncalibrated, not a verified event probability. A surge means an observed next-week local median increase of at least 10%; no next-week price is predicted here. Alerts may be false and surges may be missed. Anomaly flags are separate signals without labelled anomaly ground truth. SHAP contributions are not economic causes.</p>
    <p>{escape(str(model['evidence_status']))}. No affordability benefit is established. History includes eligible original-study rows and the latest snapshot, not all raw observations.</p>
    <small>Source: Malaysian PriceCatcher. Generated from the saved HargaLens dashboard export. Use your browser's Print option to save a PDF.</small></html>"""
    st.download_button(f"Download brief · {row['item']} · {row.district}", brief.encode("utf-8"), mime="text/html",
                       key=f"brief_{row.forecast_id}", file_name=f"hargalens_story_{row.forecast_id}.html")
    st.caption("Share the brief or open it in a browser and print it as a PDF. Dates, model identity, support and limitations travel with the case.")


def review_list(latest, refresh=None):
    st.subheader("Your local price watchlist")
    view = st.radio("Show", ["All monitored prices", "Review alerts", "Unusual patterns", "Priority shortlist"], horizontal=True)
    shown = latest.sort_values(["model_score", "forecast_id"], ascending=[False, True])
    capacity = None
    eligible_alerts = int(latest.predicted_surge.sum())
    if view in ["Review alerts", "Priority shortlist"]:
        shown = shown.loc[shown.predicted_surge.eq(1)]
    elif view == "Unusual patterns":
        shown = shown.loc[shown.is_anomaly]
    if view == "Priority shortlist":
        capacity = st.selectbox("Maximum alerts to review", [25, 50, 100], key="review_capacity")
        shown = shown.head(capacity)
        st.caption(f"Showing {len(shown):,} of {eligible_alerts:,} alerts above the frozen cutoff within your filters; "
                   f"{eligible_alerts - len(shown):,} eligible alerts are outside this shortlist. "
                   "Ranked by score, with case ID breaking ties. Choose Review alerts to see all eligible alerts.")
        st.info("This view limits review workload. It does not improve the model or change its cutoff. "
                "Smaller lists can miss more surges; their accuracy has not been measured for these filters or this snapshot. "
                "Other food groups may be absent from the shortlist.")
    if shown.empty:
        st.info("No matching rows in this view. Below-cutoff scores do not rule out future surges.")
        return
    table = shown[["project_group", "item", "state", "district", "premise_type", "weekly_median_price", "unit",
                   "model_score", "predicted_surge", "is_anomaly", "has_cached_explanation"]].copy()
    table["predicted_surge"] = table.predicted_surge.map({1: "Review next week", 0: "Below alert cutoff"})
    table["is_anomaly"] = table.is_anomaly.map({True: "Flagged", False: "Not flagged"})
    table["has_cached_explanation"] = table.has_cached_explanation.map({True: "Available", False: "Not cached"})
    table["Study guidance"] = [group_guidance(group, version)
                               for group, version in zip(shown.project_group, shown.forecast_model_version)]
    table = table.rename(columns={"project_group": "Food group", "item": "Item", "state": "State", "district": "District",
        "premise_type": "Shop type", "weekly_median_price": "Weekly median (RM)", "unit": "Unit", "model_score": "Review score",
        "predicted_surge": "Next-week alert", "is_anomaly": "Unusual pattern", "has_cached_explanation": "Explanation"})
    show_details = st.checkbox("Show more columns", help="Add separate state/district fields, food group, explanation availability and study guidance.")
    display_table = table.assign(Location=table.State + " / " + table.District)
    visible_columns = list(table.columns) if show_details else ["Item", "Location", "Shop type", "Weekly median (RM)",
                                                               "Unit", "Next-week alert", "Unusual pattern", "Review score"]
    st.dataframe(display_table, column_order=visible_columns, hide_index=True, width="stretch", height=380, column_config={
        "Weekly median (RM)": st.column_config.NumberColumn(format="%.2f", help="The middle observed price this week, per listed unit. This is not the cheapest available price."),
        "Review score": st.column_config.NumberColumn(format="%.3f", help="A model score from 0 to 1. Higher scores receive greater review priority; 0.9 is not a verified 90% chance.")})
    st.caption("A review alert concerns a possible increase of at least 10% next week. An unusual-pattern flag concerns the observed price behaviour. Review scores help prioritise investigation; they are not verified probabilities.")
    download = table.copy()
    download["Observed week starting"] = shown.week_start.dt.strftime("%Y-%m-%d").to_numpy()
    download["Forecast week starting"] = (shown.week_start + pd.Timedelta(days=7)).dt.strftime("%Y-%m-%d").to_numpy()
    download["Forecast week ending"] = shown.next_week_end.dt.strftime("%Y-%m-%d").to_numpy()
    download["Forecast model version"] = shown.forecast_model_version.to_numpy()
    download["Snapshot freshness"] = snapshot_freshness(shown.week_start.iloc[0])
    download["Guidance reference"] = [GUIDANCE_REFERENCE if version == "longer_2024_2025" else "Consult this model's own evaluation."
                                     for version in shown.forecast_model_version]
    download["Review view"] = view
    download["Shortlist capacity"] = capacity
    download["Eligible alerts within filters"] = eligible_alerts
    download["Eligible alerts outside shortlist"] = eligible_alerts - len(shown) if capacity is not None else None
    if refresh:
        download["Scores generated at"] = refresh["generated_at"]
        download["Snapshot type"] = refresh["snapshot_type"]
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
        matches = series[SERIES].eq(first[SERIES]).fillna(False).all(axis=1).to_numpy()
        default = int(np.flatnonzero(matches)[0])
    index = st.selectbox("Choose an item and location", range(len(series)), index=default, format_func=lambda i: labels[i])
    selected = series.iloc[index]
    past = history.loc[history[SERIES].eq(selected[SERIES]).fillna(False).all(axis=1)].sort_values("week_start")
    current = latest.loc[latest[SERIES].eq(selected[SERIES]).fillna(False).all(axis=1)]
    st.subheader(selected["item"])
    st.caption(f"{selected.state} / {selected.district} · {selected.premise_type} · Price per {selected.unit}")
    window = st.radio("Price history", ["Last 12 weeks", "Last 26 weeks", "All history"], index=1, horizontal=True)
    start = past.week_start.min()
    if window != "All history":
        weeks = 12 if window == "Last 12 weeks" else 26
        start = max(start, week - pd.Timedelta(weeks=weeks - 1))
    calendar = pd.date_range(start, week, freq="W-MON")
    prices = past.set_index("week_start").weekly_median_price.reindex(calendar)
    if prices.notna().any():
        price_history_chart(prices, selected.unit)
    else:
        st.info("No eligible observations in this time window. Choose All history to see older available prices.")
    st.caption("Eligible original-study weekly medians, plus any refreshed eligible observations. Gaps are left blank; missing and excluded weeks are not filled. This is not all raw observations or a fixed-shop price index, and does not include the added 2024 training history.")
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
    st.caption("Saved validation study guidance: " + group_guidance(row.project_group, row.forecast_model_version))
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
    counts["Study guidance"] = counts.project_group.map(lambda group: group_guidance(group, model["version"]))
    st.dataframe(counts.rename(columns={"project_group": "Food group", "supported_rows": "Supported rows",
        "surge_alerts": "Surge alerts", "unusual_patterns": "Unusual patterns", "cached_explanations": "Cached explanations"}),
        hide_index=True, width="stretch")
    st.caption("Group guidance summarises the saved January–March 2026 longer-history validation study. "
               "It is not a confidence estimate for an individual alert or a new period. Price history remains available for groups with weak evidence.")
    st.caption(f"{history[SERIES].drop_duplicates().shape[0]:,} local series have eligible historical rows in these filters. Supported-row counts are not population coverage. Places/items outside this study cannot be assessed here.")
    st.write("The separate anomaly detector uses a January–December 2025 reference. Its flags identify unusual patterns; there is no labelled anomaly ground truth to measure accuracy.")
    if "refresh" in entry:
        st.subheader("Source quality in the refreshed observations")
        st.caption("All locations, before your filters. The original lookup and cleaning rules are frozen; unknown shops and incomplete metadata are excluded, never guessed. Retained statistical flags are not confirmed errors. The 12 training-source review cases still need independent evidence.")
        columns = ["month", "selected_scope_rows", "invalid_removed", "unknown_metadata_removed",
                   "wholesale_removed", "frozen_sparse_items_removed", "retained_rows", "flagged_rows_retained"]
        audit = pd.DataFrame(entry["refresh_audit"])[columns].rename(columns={
            "month": "Month", "selected_scope_rows": "Basket records", "invalid_removed": "Invalid prices removed",
            "unknown_metadata_removed": "Missing metadata removed", "wholesale_removed": "Wholesale removed",
            "frozen_sparse_items_removed": "Excluded items removed", "retained_rows": "Records kept",
            "flagged_rows_retained": "Statistical flags kept"})
        st.dataframe(audit, hide_index=True, width="stretch")
        st.caption("Removal columns count distinct records. Statistical flags are a subset of kept records, not an additional exclusion.")
        st.caption("This refresh updates observations and model scores. It does not measure accuracy on the new period or change the validation results shown above.")
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
        .block-container {max-width: 1400px; padding-top: 1.5rem;}
        [data-testid="stMain"] {background-image: linear-gradient(180deg, rgba(8, 127, 140, .04), transparent 420px);}
        [data-testid="stSidebar"] {border-right: 1px solid rgba(128, 128, 128, .15);}
        [data-testid="stCaptionContainer"] {color: inherit; opacity: .85;}
        [data-testid="stMetric"] {background: rgba(8, 127, 140, .07); padding: 1rem; border-radius: .8rem; border: 1px solid rgba(128, 128, 128, .2); border-top: 3px solid #087f8c;}
        [data-testid="stButton"] button[kind="primary"] {background: #087f8c; border-color: #087f8c; color: white;}
        .signal {display: inline-block; border-radius: 999px; padding: .25rem .65rem; font-size: .8rem; font-weight: 600; margin-bottom: .35rem;}
        .signal-review {color: #804500; background: #fff0cf;}
        .signal-monitor {color: #075964; background: #e0f4f2;}
        .signal-unusual {color: #65358d; background: #f0e7fa;}
        h1, h2, h3 {letter-spacing: -.025em;}
        @media (max-width: 640px) {
            .block-container {padding-top: 5rem;}
            .st-key-snapshot_metrics [data-testid="stHorizontalBlock"] {flex-wrap: wrap;}
            .st-key-snapshot_metrics [data-testid="stColumn"] {flex: 1 1 calc(50% - .5rem); min-width: 0 !important;}
        }
        </style>""", unsafe_allow_html=True)
    mark, wordmark = st.columns([1, 10], vertical_alignment="center")
    with mark:
        st.image(str(LOGO), width=82)
    with wordmark:
        st.title("HargaLens")
        st.caption("See local prices clearly. · Your Malaysian food price watch")
    try:
        run = active_run()
        with st.spinner("Checking saved dashboard exports…"):
            entry, manifest, latest, explanations, importance, history = load_exports(str(run), export_signature(run))
        freshness = snapshot_freshness(entry["latest_complete_week"])
    except (OSError, ValueError, KeyError, TypeError, AssertionError) as error:
        st.error("The saved dashboard inputs are missing or inconsistent. No forecasts were substituted.")
        st.write("Open Notebook 12 in your project kernel and Run All. It needs the saved forecasting checkpoint and completed Notebook 11 outputs. Do not delete data or restart Notebook 01 just to reopen the dashboard.")
        st.caption("If you published a refresh in Notebook 13, rerun 13 to verify it. To intentionally reopen the original export, remove only 08_dashboard/refresh_manifest.json; keep its archived files and all model inputs.")
        with st.expander("Details for troubleshooting"):
            st.code(str(error))
        st.stop()
    week = pd.Timestamp(entry["latest_complete_week"])
    st.info(f"Historical demonstration · Observed week: {date_span(week, week + pd.Timedelta(days=6))} · Forecast week: {date_span(week + pd.Timedelta(days=7), week + pd.Timedelta(days=13))}. These are not live warnings.")
    refresh = entry.get("refresh")
    if refresh:
        scored_at = pd.Timestamp(refresh["generated_at"]).tz_convert("Asia/Kuala_Lumpur").strftime("%d %b %Y, %I:%M %p MYT")
        st.caption(f"Offline refresh · Scored {scored_at} retrospectively. These were not issued as timely forecasts.")
    with st.expander("Snapshot dates and coverage"):
        st.caption(freshness)
        if refresh:
            st.warning(f"Offline refresh · Scores generated {scored_at}. Observations were scored retrospectively; these are not archived timely forecasts. Model, cutoff and validation results are unchanged.")
            st.caption(f"Supported local series before filters: {entry['latest_supported_rows']:,} in this snapshot; "
                       f"{entry['original_supported_rows']:,} in the original snapshot. Differences describe observation support, "
                       "not accuracy or population coverage. Check the source audit and Notebook 13's group/state comparison.")
    history, latest = filter_data(history, latest)
    with st.container(key="snapshot_metrics"):
        cols = st.columns(4)
        for col, label, value, help_text in zip(cols,
            ["Prices monitored", "Alerts to review", "Unusual patterns", "Explanations ready"],
            [len(latest), int(latest.predicted_surge.sum()), int(latest.is_anomaly.sum()), int(latest.has_cached_explanation.sum())],
            ["One item in one district and shop type, with enough data for this saved forecast.",
             "Cases above the fixed review cutoff for a possible price rise of at least 10% next week. Some alerts will be wrong.",
             "Observed patterns flagged by a separate anomaly detector. This is not the next-week forecast.",
             "Local explanations prepared offline for selected cases. A missing explanation is shown explicitly."]):
            col.metric(label, f"{value:,}", help=help_text)
    st.caption("Choose an area in the sidebar → open a price story → check its history. An alert means a possible rise of at least 10% next week. Scores are review priorities, not verified probabilities.")
    if entry["forecast_model"]["version"] == "longer_2024_2025":
        st.caption("Validation evidence is strongest for vegetables. Several other food groups have weak or limited evidence; check About the model before relying on an alert.")
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
        spotlight(history, latest, explanations, entry["forecast_model"], refresh)
        review_list(latest, refresh)
        with st.expander("Food-group overview"):
            group_overview(latest)
    with details:
        explore_series(history, latest, explanations, week, manifest["protocol"]["units"])
    with evidence:
        model_and_coverage(entry, latest, importance, history)
    st.caption("HargaLens · Malaysian PriceCatcher · Research prototype. Observed local medians; no affordability benefit has been established.")


if __name__ == "__main__":
    main()
