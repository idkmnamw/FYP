"""Read-only dashboard. Start with: python -m streamlit run app.py"""
import hashlib
import inspect
import json
import os
import re
import time
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
    """Use the selected model’s group evidence, with guidance for the archived checkpoint."""
    if isinstance(version, dict):
        evidence = version.get("group_evidence", {}).get(group)
        if evidence is not None:
            positives, detected = int(evidence["positives"]), int(evidence["tp"])
            if positives == 0:
                return "No positive validation cases; surge detection could not be assessed. Check observed prices."
            support = f"Detected {detected} of {positives} validation surges. "
            if positives < 20:
                return support + "Very limited support; avoid broad reliability claims."
            if evidence.get("recall", 0) < 0.25:
                return support + "Most surges were missed. Prioritise price history; below-cutoff scores need care."
            return support + f"Validation alert precision was {evidence['precision']:.1%}; alerts need investigation."
        version = version["version"]
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
    cache = inside(dashboard_directory(run), entry["cache_dir"])
    paths = [entry_path, cache / "explanation_manifest.json"]
    pointer = run / "08_dashboard" / "active_deployment.json"
    if pointer.exists():
        paths.append(pointer)
    paths += [cache / name for name in TABLES]
    if "refresh" in entry:
        record_path = inside(run, entry["refresh"]["source_manifest"])
        paths += [dashboard_directory(run) / "dashboard_manifest.json", record_path,
                  record_path.parent / "selected_daily_prices.parquet", cache / "refresh_history.parquet"]
    paths += [run / "experiments" / "holdout" / "05_modelling" / f"{split}.parquet"
              for split in ["train", "validation", "test"]]
    return tuple((str(path), path.stat().st_size, path.stat().st_mtime_ns) for path in paths)


def dashboard_directory(run):
    base = run / "08_dashboard"
    pointer = base / "active_deployment.json"
    if not pointer.is_file():
        return base
    record = read_json(pointer)
    folder = inside(base, record["directory"])
    if not folder.is_relative_to((base / "deployments").resolve()):
        raise ValueError("Invalid active dashboard deployment directory.")
    return folder


def dashboard_entry(run):
    """A published refresh takes precedence; inconsistent files are never hidden."""
    folder = dashboard_directory(run)
    refresh = folder / "refresh_manifest.json"
    return refresh if refresh.is_file() else folder / "dashboard_manifest.json"


@st.cache_data(show_spinner=False, max_entries=2)
def load_exports(run_name, signature):
    """Verify cached files, then read display data only: no models or labels."""
    run = Path(run_name)
    dashboard_dir = dashboard_directory(run)
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
    if manifest["protocol"].get("latest_rows") == "all" and (
            len(local) != len(latest) or set(local.index) != set(latest.forecast_id)):
        raise ValueError("This export declares full SHAP coverage but explanations are missing.")
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
    if isinstance(prices, pd.DataFrame):
        for (name, values), colour in zip(prices.items(), ["#087f8c", "#ca681b"]):
            ax.plot(values.index, values.values, color=colour, linewidth=1.8, marker=".", markersize=4, label=name)
        ax.legend(fontsize=8)
    else:
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


def shap_summary(values, compact=False):
    """Describe signed saved contributions without inferring price facts or causes."""
    if compact:
        values = values.loc[values.abs().nlargest(2).index]
    raising = values.loc[values.gt(0)].nlargest(2).index
    lowering = values.loc[values.lt(0)].nsmallest(2).index
    reasons = []
    for features, direction in [(raising, "raising this model score"), (lowering, "lowering it")]:
        if len(features):
            names = " and ".join(FEATURE_NAMES.get(name, name) for name in features)
            reasons.append(names + " contributed most to " + direction)
    return ("Relative to the model's baseline: " + "; ".join(reasons) + "." if reasons
            else "No saved feature contribution moves this score away from the model's baseline.")


def snapshot_alert_summary(latest):
    """Rank counts within the filtered snapshot, with monitoring support and stable ties."""
    if latest.empty or not latest.predicted_surge.any():
        return None
    counts = latest.groupby(["state", "project_group"], sort=True).agg(
        alerts=("predicted_surge", "sum"), monitored=("forecast_id", "size")).reset_index()
    counts = counts.sort_values(["alerts", "state", "project_group"], ascending=[False, True, True])
    top = counts.iloc[0]
    return {"state": top.state, "group": top.project_group, "alerts": int(top.alerts),
            "monitored": int(top.monitored), "ties": int(counts.alerts.eq(top.alerts).sum())}


def snapshot_headline(latest, week, can_navigate):
    if latest.empty:
        return
    observed = date_span(week, pd.Timestamp(week) + pd.Timedelta(days=6))
    summary = snapshot_alert_summary(latest)
    if summary is None:
        st.info(f"No review alerts in this snapshot · Observed {observed} · {len(latest):,} monitored local series in your filters.")
        return
    st.subheader("Most review alerts in this snapshot")
    alert_label = "alert" if summary["alerts"] == 1 else "alerts"
    st.write(f"**{summary['state']} · {summary['group']}** — **{summary['alerts']:,} {alert_label}** "
             f"among {summary['monitored']:,} monitored local series.")
    other_ties = summary["ties"] - 1
    combination_label = "combination" if other_ties == 1 else "combinations"
    tie = f" Tied with {other_ties} other state/food-group {combination_label}." if other_ties else ""
    st.caption(f"Observed {observed}. Within your current filters.{tie} "
               "Counts depend on monitoring coverage; this is not a population-risk ranking.")
    if can_navigate:
        def open_map():
            st.session_state["dashboard_tab"] = "Price map & playback"
        st.button("Explore recorded prices on the map", key="snapshot_open_map", on_click=open_map,
                  help="Opens Price map & playback with your current filters. Map colours show observed prices, not alert counts.")
    else:
        st.caption("Open Price map & playback below to explore recorded prices with these filters.")


def spotlight(history, latest, explanations, model, refresh=None, units="saved model-output units"):
    """Turn fixed saved results into a readable case story and portable brief."""
    if latest.empty:
        return
    st.subheader("Prices worth a closer look")
    st.caption("Start with a price story: the three highest-scoring cases in your filters.")
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
            delta = f"{change:+.1f}% from the preceding week (observed)" if pd.notna(change) and np.isfinite(change) else None
            st.metric("Observed weekly price", f"RM {row.weekly_median_price:.2f}", delta=delta, delta_color="inverse",
                      help="The median of the observed prices. The change compares observed weeks; neither value predicts next week's price.")
            st.caption(f"Per {row.unit}")
            observed = pd.Timestamp(row.week_start)
            st.caption(f"Observed: {date_span(observed, observed + pd.Timedelta(days=6))}")
            status_class = "signal-review" if row.predicted_surge else "signal-monitor"
            status_label = "Alert to review" if row.predicted_surge else "No alert raised"
            st.markdown(f'<span class="signal {status_class}">{status_label}</span>', unsafe_allow_html=True)
            if row.is_anomaly:
                st.markdown('<span class="signal signal-unusual">Unusual observed pattern</span>', unsafe_allow_html=True)
            local = explanations.loc[explanations.cohort.eq("latest") & explanations.explanation_id.eq(row.forecast_id)]
            if local.empty:
                st.caption("No saved SHAP explanation for this case.")
            else:
                contributions = pd.Series({name.removeprefix("shap__"): float(local.iloc[0][name])
                                           for name in local.columns if name.startswith("shap__")})
                st.caption(shap_summary(contributions, compact=True))
            st.button("Read this price story", key=f"story_{row.forecast_id}", width="stretch",
                      type="primary" if row.forecast_id == st.session_state["spotlight_id"] else "secondary",
                      on_click=choose_story, args=(row.forecast_id,))

    row = candidates.loc[candidates.forecast_id.eq(st.session_state["spotlight_id"])].iloc[0]
    week = pd.Timestamp(row.week_start)
    forecast_period = date_span(week + pd.Timedelta(days=7), row.next_week_end)
    status = "Review next week" if row.predicted_surge else "No alert raised"
    pattern = "Flagged as unusual" if row.is_anomaly else "Not flagged as unusual"
    headline = f"{row['item']} · {row.state} / {row.district}"
    coverage = f"{int(row.record_count)} price records from {int(row.premise_count)} shops across {int(row.observed_days)} observed days"
    st.markdown(f"#### Price story: {headline}")
    prediction_summary(row, model)
    guidance = group_guidance(row.project_group, model)
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
    with st.expander("Why this next-week score? (SHAP)", expanded=True):
        explain_case(row, explanations, units)
    # Self-contained HTML can be shared or printed without a running dashboard.
    facts = {"Forecast case ID": row.forecast_id, "Observed week": date_span(week, week + pd.Timedelta(days=6)), "Forecast week": forecast_period,
             "Observed weekly median": f"RM {row.weekly_median_price:.2f} per {row.unit}",
             "Shop type": row.premise_type, "Next-week status": status, "Observed pattern": pattern,
             "Uncalibrated review score": f"{row.model_score:.3f}", "Frozen review cutoff": f"{model['score_cutoff']:.5f}",
             "Observation support": coverage, "Forecast model": model["version"], "Checkpoint SHA-256": model["sha256"],
             "Evidence status": model["evidence_status"]}
    facts["Food-group study guidance"] = guidance
    facts["Snapshot freshness"] = snapshot_freshness(week)
    facts["Guidance reference"] = model.get("guidance_reference") or (GUIDANCE_REFERENCE if model["version"] == "longer_2024_2025" else "Consult this model's own evaluation.")
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


def review_list(latest, refresh=None, model=None):
    st.subheader("Your local price watchlist")
    view = st.radio("Show", ["All monitored prices", "Review alerts", "Unusual patterns", "Priority shortlist"], horizontal=True)
    shown = latest.sort_values(["model_score", "forecast_id"], ascending=[False, True])
    capacity = None
    eligible_alerts = int(latest.predicted_surge.sum())
    if view in ["Review alerts", "Priority shortlist"]:
        shown = shown.loc[shown.predicted_surge.eq(1)]
    elif view == "Unusual patterns":
        shown = shown.loc[shown.is_anomaly]
        shown = shown.sort_values(["anomaly_score", "forecast_id"], ascending=[False, True])
        st.caption("Ranked by the separate anomaly score: larger scores indicate more unusual observed patterns.")
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
                   "model_score", "anomaly_score", "predicted_surge", "is_anomaly", "has_cached_explanation"]].copy()
    table["predicted_surge"] = table.predicted_surge.map({1: "Review next week", 0: "No alert raised"})
    table["is_anomaly"] = table.is_anomaly.map({True: "Flagged", False: "Not flagged"})
    table["has_cached_explanation"] = table.has_cached_explanation.map({True: "Available", False: "Not cached"})
    table["Study guidance"] = [group_guidance(group, model or version)
                               for group, version in zip(shown.project_group, shown.forecast_model_version)]
    table = table.rename(columns={"project_group": "Food group", "item": "Item", "state": "State", "district": "District",
        "premise_type": "Shop type", "weekly_median_price": "Weekly median (RM)", "unit": "Unit", "model_score": "Surge score",
        "anomaly_score": "Anomaly score",
        "predicted_surge": "Next-week alert", "is_anomaly": "Unusual pattern", "has_cached_explanation": "Explanation"})
    show_details = st.checkbox("Show research details", help="Add model scores, explanation availability and study evidence.")
    display_table = table.assign(Location=table.State + " / " + table.District)
    visible_columns = list(table.columns) if show_details else ["Item", "Location", "Shop type", "Weekly median (RM)",
                                                               "Unit", "Next-week alert", "Unusual pattern"]
    st.dataframe(display_table, column_order=visible_columns, hide_index=True, width="stretch", height=380, column_config={
        "Weekly median (RM)": st.column_config.NumberColumn(format="%.2f", help="The middle observed price this week, per listed unit. This is not the cheapest available price."),
        "Anomaly score": st.column_config.NumberColumn(format="%.4f", help="Larger means a more unusual observed pattern; this is separate from the next-week surge score."),
        "Surge score": st.column_config.NumberColumn(format="%.3f", help="The next-week surge model's score, from 0 to 1. Higher scores receive greater review priority; 0.9 is not a verified 90% chance.")})
    st.caption("A review alert concerns a possible increase of at least 10% next week. An unusual-pattern flag concerns the observed price behaviour. Use both signals to prioritise review.")
    download = table.copy()
    download["Observed week starting"] = shown.week_start.dt.strftime("%Y-%m-%d").to_numpy()
    download["Forecast week starting"] = (shown.week_start + pd.Timedelta(days=7)).dt.strftime("%Y-%m-%d").to_numpy()
    download["Forecast week ending"] = shown.next_week_end.dt.strftime("%Y-%m-%d").to_numpy()
    download["Forecast model version"] = shown.forecast_model_version.to_numpy()
    download["Snapshot freshness"] = snapshot_freshness(shown.week_start.iloc[0])
    download["Guidance reference"] = [(model.get("guidance_reference") if model else None) or (GUIDANCE_REFERENCE if version == "longer_2024_2025" else "Consult this model's own evaluation.")
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
    return shown if view == "Unusual patterns" else None


def saved_anomaly_cutoff(run, entry, manifest):
    """Read the detector threshold only from an export-linked, verified manifest."""
    model = entry["anomaly_model"]
    relative = model.get("manifest_path", "07_anomalies/anomaly_manifest.json")
    expected = manifest.get("upstream_sha256", {}).get(relative)
    if expected is None:
        return None
    path = inside(run, relative)
    if file_hash(path) != expected:
        raise ValueError("The saved detector manifest differs from the dashboard export.")
    record = read_json(path)
    if record["model_sha256"] != model["sha256"] or record["protocol"]["schema_version"] != 1:
        raise ValueError("The saved detector metadata does not match this dashboard.")
    cutoff = float(record["training_score_threshold"])
    if not np.isfinite(cutoff):
        raise ValueError("The saved detector cutoff is not finite.")
    return cutoff


def anomaly_measurements(row):
    """Display the six version-1 measurements before training-median imputation."""
    def number(name):
        value = row.get(name, np.nan)
        return float(value) if pd.notna(value) and np.isfinite(value) else np.nan

    def ratio(numerator, denominator, change=False):
        top, bottom = number(numerator), number(denominator)
        if not np.isfinite(top) or not np.isfinite(bottom) or bottom <= 0:
            return np.nan
        return 100 * (top / bottom - (1 if change else 0))

    return pd.DataFrame({
        "Measurement": ["Change from previous week", "Change from two weeks earlier",
                        "Change from four weeks earlier", "Difference from previous four-week mean",
                        "Within-week price variation", "Previous four-week price variation"],
        "Value (%)": [number("weekly_price_change_pct"),
                      ratio("weekly_median_price", "lag_2w_price", change=True),
                      ratio("weekly_median_price", "lag_4w_price", change=True),
                      ratio("weekly_median_price", "rolling_mean_4w", change=True),
                      ratio("weekly_std_price", "weekly_median_price"),
                      ratio("rolling_std_4w", "rolling_mean_4w")],
        "Meaning": ["Current median compared with the exact previous calendar week",
                    "Current median compared with the exact week two weeks earlier",
                    "Current median compared with the exact week four weeks earlier",
                    "Current median compared with the mean of available prior four-week medians",
                    "Standard deviation of current-week observations divided by current median",
                    "Standard deviation divided by mean of available prior four-week medians"],
    })


def anomaly_details(cases, history, run, entry, manifest):
    st.subheader("Anomaly details")
    labels = {row.forecast_id: f"{row['item']} · {row.state} / {row.district} · {row.premise_type} · {row.unit}"
              for _, row in cases.iterrows()}
    case_id = st.selectbox("Choose an unusual pattern to inspect", list(labels),
                           format_func=lambda value: labels[value], key="anomaly_case")
    row = cases.loc[cases.forecast_id.eq(case_id)].iloc[0]
    week = pd.Timestamp(row.week_start)
    st.caption(f"Observed week: {date_span(week, week + pd.Timedelta(days=6))}. "
               "This flag concerns observed price behaviour, separately from the next-week surge alert.")
    try:
        cutoff = saved_anomaly_cutoff(run, entry, manifest)
        if cutoff is not None and bool(row.is_anomaly) != (float(row.anomaly_score) > cutoff):
            raise ValueError("The saved flag disagrees with the detector cutoff.")
    except (OSError, ValueError, KeyError, TypeError) as error:
        cutoff = None
        st.warning(f"The detector cutoff could not be verified: {error}")
    cols = st.columns(3)
    cols[0].metric("Anomaly score", f"{row.anomaly_score:.4f}")
    cols[1].metric("Frozen anomaly cutoff", "Unavailable" if cutoff is None else f"{cutoff:.4f}")
    cols[2].metric("Observed weekly median", f"RM {row.weekly_median_price:.2f} / {row.unit}")
    if cutoff is not None:
        st.write(f"The saved score **{row.anomaly_score:.6f}** is above the detector cutoff **{cutoff:.6f}**, "
                 "so this observed pattern was flagged for investigation.")
    st.caption("Larger anomaly scores indicate more unusual patterns; they are not probabilities. "
               "A score strictly above the saved detector cutoff is flagged. This cutoff is separate from the 10% next-week surge definition.")
    if cutoff is None:
        st.caption("No verified cutoff is available for this saved export; no replacement cutoff was inferred.")
    st.write(f"Observation support: {int(row.record_count)} price records, "
             f"{int(row.premise_count)} shops and {int(row.observed_days)} observed days.")
    past = history.loc[history[SERIES].eq(row[SERIES]).fillna(False).all(axis=1)
                       & history.week_start.le(week)].sort_values("week_start")
    calendar = pd.date_range(week - pd.Timedelta(weeks=12), week, freq="W-MON")
    prices = past.set_index("week_start").weekly_median_price.reindex(calendar)
    if prices.notna().any():
        price_history_chart(prices, row.unit, height=2.8)
    st.caption("Saved eligible weekly prices, including the selected week. Gaps stay blank; "
               "changing shops and sparse observations can affect the median.")
    st.markdown("**Price-history measurements to inspect**")
    st.write("Read these alongside the price chart and observation counts. They describe the pattern, "
             "but do not identify which measurement caused the detector to flag it.")
    measurements = anomaly_measurements(row)
    st.dataframe(measurements, hide_index=True, width="stretch", column_config={
        "Value (%)": st.column_config.NumberColumn(format="%.2f%%")})
    with st.expander("How these measurements are calculated"):
        st.write("These saved features are shown before imputation. Blank values were unavailable; the detector uses "
                 "training-fitted median imputation and missing indicators. These are context measurements, not feature "
                 "contributions. SHAP explains the separate surge classifier, not this detector.")
    st.download_button("Download anomaly measurements (CSV)",
                       measurements.to_csv(index=False).encode("utf-8-sig"),
                       file_name=f"hargalens_anomaly_{case_id}.csv", mime="text/csv")


def prediction_summary(row, model):
    """Keep recorded prices, forecast dates and detector flags distinct."""
    week = pd.Timestamp(row.week_start)
    observed, forecast = st.columns(2)
    with observed:
        st.markdown("**1 · What was recorded**")
        st.caption(date_span(week, week + pd.Timedelta(days=6)))
        st.metric("Observed median price", f"RM {row.weekly_median_price:.2f} / {row.unit}")
        change = row.get("weekly_price_change_pct", np.nan)
        if pd.notna(change) and np.isfinite(change):
            direction = "higher than" if change > 0 else "lower than" if change < 0 else "the same as"
            comparison = f"{abs(change):.1f}% {direction}" if change != 0 else direction
            st.write(f"This was **{comparison}** the preceding calendar week's median. This is an observed change, not next week's prediction.")
        else:
            st.caption("No supported change from the preceding calendar week is available.")
    with forecast:
        st.markdown("**2 · What the model flags for the following week**")
        st.caption(date_span(week + pd.Timedelta(days=7), row.next_week_end))
        st.metric("Next-week signal", "Alert to review" if row.predicted_surge else "No alert raised")
        st.write(f"Saved surge score: **{row.model_score:.3f}** · Alert cutoff: **{model['score_cutoff']:.3f}**.")
        st.caption("The alert concerns a possible rise of at least 10% in next week's median, not an exact future RM price.")
    pattern = "Flagged for review" if row.is_anomaly else "Not flagged"
    st.caption(f"Separate anomaly detector: {pattern}. This describes observed price behaviour, not the following week's outcome.")


def explain_case(row, explanations, units):
    case = explanations.loc[explanations.cohort.eq("latest") & explanations.explanation_id.eq(row.forecast_id)]
    st.subheader("Why the model gave this surge score (SHAP)")
    if case.empty:
        st.info("No saved SHAP explanation is available for this case. You can still inspect its recorded prices and saved signals. "
                "Changing filters does not generate a new explanation.")
        return
    case = case.iloc[0]
    values = pd.Series({name.removeprefix("shap__"): float(case[name]) for name in case.index if name.startswith("shap__")})
    st.write(shap_summary(values))
    st.caption("Orange raises the model output; blue lowers it. Longer bars mean larger contributions, not economic causes.")
    largest = values.abs().nlargest(8).index
    shown = values.loc[largest].rename(index=FEATURE_NAMES).copy()
    shown.loc["Other features (combined)"] = values.drop(largest).sum()
    shown = shown.iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, 3.7))
    ax.barh(shown.index, shown.values, color=np.where(shown.values >= 0, "#ca681b", "#267b9b"))
    ax.axvline(0, color="#899397", linewidth=0.7)
    ax.set_xlabel("← Lowers model output     |     Raises model output →")
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)
    feature_rows = []
    for name in largest:
        value = row.get(name, pd.NA)
        display_value = "Unavailable in saved inputs" if pd.isna(value) else f"{value:.4g}" if isinstance(value, (int, float, np.number)) else str(value)
        if name == "weekly_price_change_pct" and pd.notna(value):
            display_value = f"{value:+.1f}%"
        feature_rows.append({"Feature": FEATURE_NAMES.get(name, name), "Saved input value": display_value,
                             "Effect on model output": "Raises" if values[name] > 0 else "Lowers" if values[name] < 0 else "No contribution"})
    with st.expander("See the input values behind these features"):
        st.dataframe(pd.DataFrame(feature_rows), hide_index=True, width="stretch")
        st.caption("These are saved input values, not suggested prices or changes to make. Missing inputs may have been filled by training-fitted preprocessing.")
    with st.expander("How this explanation is calculated"):
        st.write("SHAP divides the model output into a baseline and feature contributions. Related price-history features can share information. Contributions are in model-output units, not percentage-point changes in surge probability.")
        st.caption(f"Baseline {case.base_value:.3f} + contributions = raw output {case.raw_output:.3f}; surge score {case.model_score:.3f}. Units: {units}.")


def prediction_explorer(latest, explanations, model, units):
    st.subheader("Understand a prediction")
    st.write("Choose an item and location to separate its recorded price, next-week signal and saved SHAP explanation.")
    if latest.empty:
        st.info("No saved predictions match your sidebar filters. Broaden the filters to choose a case.")
        return
    local_ids = set(explanations.loc[explanations.cohort.eq("latest"), "explanation_id"])
    available = latest.forecast_id.isin(local_ids)
    st.caption(f"Saved SHAP explanations: {int(available.sum()):,} of {len(latest):,} cases in your current filters. "
               "Coverage describes availability, not prediction accuracy.")
    view = st.radio("Which cases?", ["All cases", "With saved SHAP explanations"], horizontal=True, key="explanation_view")
    cases = latest.loc[available] if view == "With saved SHAP explanations" else latest
    if cases.empty:
        st.info("No saved explanations match these filters. Choose All cases to inspect prices and signals, or broaden the sidebar filters.")
        return
    cases = cases.sort_values(["model_score", "forecast_id"], ascending=[False, True])
    labels = {row.forecast_id: f"{row['item']} · {row.state} / {row.district} · {row.premise_type} · {row.unit}"
              + (" · SHAP available" if row.forecast_id in local_ids else " · SHAP unavailable") for _, row in cases.iterrows()}
    selected = st.selectbox("Choose a prediction to understand", list(labels),
                            format_func=lambda value: labels[value], key="explanation_case")
    row = cases.loc[cases.forecast_id.eq(selected)].iloc[0]
    prediction_summary(row, model)
    explain_case(row, explanations, units)


def explore_series(history, latest, explanations, week, units, model=None):
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
    st.caption("Eligible study and refreshed weekly medians. Gaps stay blank; the chart excludes the added 2024 training history.")
    if current.empty:
        st.warning("No supported forecast for this local series in the snapshot week. Its older price history is not a current prediction.")
        return
    row = current.iloc[0]
    cols = st.columns(4)
    change = row.get("weekly_price_change_pct", np.nan)
    delta = f"{change:+.1f}% from the preceding week (observed)" if pd.notna(change) else None
    cols[0].metric("Weekly median", f"RM {row.weekly_median_price:.2f}", delta=delta, delta_color="inverse")
    cols[1].metric("Next-week surge score", f"{row.model_score:.3f}", help="Higher means greater review priority, not a verified event probability.")
    cols[2].metric("Next-week alert", "Review" if row.predicted_surge else "No alert raised")
    cols[3].metric("Unusual pattern", "Flagged" if row.is_anomaly else "Not flagged")
    st.caption(f"Unit: {row.unit}. Observed {date_span(row.week_start, pd.Timestamp(row.week_start) + pd.Timedelta(days=6))}: "
               f"{int(row.record_count)} price records, {int(row.premise_count)} shops, {int(row.observed_days)} observed days. No alert does not guarantee stable prices.")
    st.caption("Saved validation study guidance: " + group_guidance(row.project_group, model or row.forecast_model_version))
    explain_case(row, explanations, units)


def district_key(state, district):
    """Match explicit state/district names, without guessing similar locations."""
    def normalise(value):
        value = re.sub(r"[^a-z0-9]+", " ", str(value).casefold()).strip()
        return value.removeprefix("wilayah persekutuan ").removeprefix("w p ").removeprefix("wp ")
    return normalise(state) + "|" + normalise(district)


def local_price_map(rows, states, measure="weekly_median_price"):
    """Attach one comparable weekly price to each DOSM district polygon."""
    boundaries = read_json(REPOSITORY / "assets" / "malaysia-districts.geojson")
    prices = {district_key(row.state, row.district): row for row in rows.itertuples()}
    if len(prices) != len(rows):
        raise ValueError("Multiple prices match a district. Choose one item, unit, shop type and week.")
    state_keys = {district_key(state, "").split("|")[0] for state in states}
    features, matched = [], set()
    for feature in boundaries["features"]:
        location = feature["properties"]
        key = district_key(location["state"], location["district"])
        if key.split("|")[0] not in state_keys:
            continue
        row = prices.get(key)
        value = getattr(row, measure) if row is not None else None
        has_value = row is not None and pd.notna(value)
        if row is not None:
            matched.add(key)
        features.append({"type": "Feature", "geometry": feature["geometry"],
                         "district_key": key, "location": location["state"] + " / " + location["district"],
                         "price": float(row.weekly_median_price) if row is not None else None,
                         "value": float(value) if has_value else None,
                         "has_value": has_value,
                         "coverage": (f"{row.record_count:.0f} records · {row.premise_count:.0f} shops · "
                                      f"{row.observed_days:.0f} days") if row is not None else "No eligible observation"})
    return features, sorted(set(prices) - matched)


def price_timeline(prices):
    """Calculate changes only where the same local series has the exact previous week."""
    if prices.duplicated(SERIES + ["week_start"]).any():
        raise ValueError("Duplicate local weeks cannot be compared safely.")
    previous = prices[SERIES + ["week_start", "weekly_median_price"]].copy()
    previous["week_start"] += pd.Timedelta(weeks=1)
    previous = previous.rename(columns={"weekly_median_price": "previous_price"})
    timeline = prices.merge(previous, on=SERIES + ["week_start"], how="left", validate="one_to_one")
    timeline["change_pct"] = (timeline.weekly_median_price / timeline.previous_price - 1) * 100
    return timeline


def stop_price_playback():
    st.session_state["price_playing"] = False


def local_price_explorer(history, latest, week, explanations, units, model):
    st.subheader("Watch prices change")
    st.write("Choose a food item, then play through the weeks or compare two districts.")
    st.caption("Uses your sidebar filters. Reset District to All to compare more locations. "
               "Colours show observed prices; next-week model alerts are shown separately below.")
    if history.empty:
        stop_price_playback()
        if st.session_state.get("playback_timer_active"):
            st.rerun()
        st.info("No eligible prices match these filters. Broaden the sidebar selection.")
        return
    items = history[["item_code", "item", "unit"]].drop_duplicates().sort_values(["item", "item_code", "unit"])
    left, middle, right = st.columns([2, 1, 1])
    chosen = left.selectbox("Compare item", range(len(items)), key="map_item",
                           format_func=lambda i: f"{items.iloc[i]['item']} · {items.iloc[i].unit} · #{items.iloc[i].item_code}")
    item = items.iloc[chosen]
    prices = history.loc[history.item_code.eq(item.item_code) & history.unit.eq(item.unit)]
    shop_type = middle.selectbox("Compare shop type", sorted(prices.premise_type.unique()), key="map_shop")
    prices = prices.loc[prices.premise_type.eq(shop_type)]
    prices = price_timeline(prices)
    mode = right.radio("Map shows", ["Price (RM)", "Weekly change (%)"], key="map_measure")
    measure = "weekly_median_price" if mode == "Price (RM)" else "change_pct"
    weeks = list(pd.date_range(prices.week_start.min(), prices.week_start.max(), freq="W-MON"))
    context = f"{item.item_code}|{item.unit}|{shop_type}|{mode}|{','.join(sorted(history.state.unique()))}|{','.join(sorted(history.district.unique()))}"
    context_id = hashlib.sha256(context.encode()).hexdigest()[:12]
    week_key = f"playback_week_{context_id}"
    if st.session_state.get("playback_context") != context_id:
        st.session_state["playback_context"] = context_id
        saved_week = st.session_state.get("playback_observed", weeks[-1])
        st.session_state[week_key] = saved_week if saved_week in weeks else weeks[-1]
        stop_price_playback()
        if st.session_state.get("playback_timer_active"):
            st.rerun()
    observed = pd.Timestamp(st.session_state[week_key])
    position = weeks.index(observed)
    controls = st.columns(4)
    restart = controls[0].button("↤ First week", key="playback_first", width="stretch", disabled=len(weeks) < 2)
    back = controls[1].button("← Previous", key="playback_back", width="stretch", disabled=position == 0)
    play = controls[2].button("Pause" if st.session_state.get("price_playing") else "▶ Play", key="playback_play",
                              type="primary", width="stretch", disabled=len(weeks) < 2)
    forward = controls[3].button("Next →", key="playback_next", width="stretch", disabled=position == len(weeks) - 1)
    if restart or back or forward:
        st.session_state[week_key] = weeks[0 if restart else position - 1 if back else position + 1]
        stop_price_playback()
        st.rerun()
    if play:
        st.session_state["price_playing"] = not st.session_state.get("price_playing", False)
        if st.session_state["price_playing"] and position == len(weeks) - 1:
            st.session_state[week_key] = weeks[0]
        st.session_state["playback_tick"] = time.monotonic()
        st.rerun()
    if st.session_state.get("price_playing") and time.monotonic() - st.session_state.get("playback_tick", 0) >= 1.5:
        st.session_state[week_key] = weeks[min(position + 1, len(weeks) - 1)]
        st.session_state["playback_tick"] = time.monotonic()
        if position + 1 >= len(weeks) - 1:
            stop_price_playback()
            st.rerun()
    if len(weeks) > 1:
        observed = pd.Timestamp(st.select_slider("Move through the weeks", options=weeks, key=week_key,
                                                format_func=lambda value: value.strftime("%d %b %Y"),
                                                on_change=stop_price_playback))
    else:
        observed = weeks[0]
    st.session_state["playback_observed"] = observed
    if st.session_state.get("playback_timer_active") and not st.session_state.get("price_playing"):
        st.rerun()
    rows = prices.loc[prices.week_start.eq(observed)].copy()
    st.markdown(f"#### {date_span(observed, observed + pd.Timedelta(days=6))}")
    st.caption(f"RM per {item.unit} · {shop_type}. A weekly price is the middle (median) recorded price. "
               "Press Play to watch observed prices, or pause to explore a district.")
    features, unmatched = local_price_map(rows, history.state.unique(), measure)
    cols = st.columns(3)
    cols[0].metric("Districts with recorded prices", len(rows))
    cols[1].metric("Lowest recorded price", "—" if rows.empty else f"RM {rows.weekly_median_price.min():.2f}")
    cols[2].metric("Highest recorded price", "—" if rows.empty else f"RM {rows.weekly_median_price.max():.2f}")
    if rows.empty:
        st.info("No recorded prices meet the study's coverage rules this week. Playback keeps this gap visible.")
    if measure == "change_pct":
        st.caption("Orange = a rise; blue = a fall. Grey = missing this week or the exact previous week. "
                   "This shows observed changes, not model predictions or economic causes.")
    valid = prices[measure].dropna()
    domain = [float(valid.min()), float(valid.max())] if not valid.empty else [0, 1]
    if measure == "change_pct":
        limit = max(abs(domain[0]), abs(domain[1]), 1)
        domain = [-limit, limit]
    elif domain[0] == domain[1]:
        domain = [max(0, domain[0] - .5), domain[1] + .5]
    clicked = None
    if features:
        spec = {"height": 410,
                "projection": {"type": "mercator"},
                "params": [{"name": "district_pick", "select": {"type": "point", "fields": ["district_key"],
                                                                     "on": "click", "toggle": False}}],
                "mark": {"type": "geoshape", "strokeWidth": 0.7, "invalid": None},
                "encoding": {
                    "color": {"condition": {"test": "!datum.has_value", "value": "#aeb8c2"},
                              "field": "value", "type": "quantitative",
                              "scale": {"scheme": "blueorange" if measure == "change_pct" else "tealblues", "domain": domain},
                              "legend": {"title": "% change" if measure == "change_pct" else f"RM / {item.unit}"}},
                    "stroke": {"condition": {"param": "district_pick", "empty": False, "value": "#f59e0b"},
                               "value": "#ffffff"},
                    "tooltip": [{"field": "location", "title": "District"},
                                {"field": "price", "type": "quantitative", "title": "Weekly median (RM)", "format": ".2f"},
                                {"field": "value", "type": "quantitative", "title": "Weekly change (%)" if measure == "change_pct" else "Price (RM)", "format": ".2f"},
                                {"field": "coverage", "title": "Observation coverage"}]}}
        if not any(feature["has_value"] for feature in features):
            spec["encoding"]["color"] = {"value": "#aeb8c2"}
        # Keep geometry inside a layer so older Streamlit versions do not flatten its coordinates.
        layer = {"data": {"values": features}, "mark": spec.pop("mark"), "encoding": spec.pop("encoding")}
        spec["layer"] = [layer]
        event = st.vega_lite_chart(spec=spec, width="stretch", on_select="rerun",
                                   selection_mode="district_pick", key=f"district_map_{context_id}")
        selection = event.selection.get("district_pick", [])
        if selection:
            clicked = selection[0].get("district_key")
    st.caption("Click a district, or use the selector below. Grey = not enough comparable data for this view. "
               "Colours use a fixed scale across this item’s weeks, so playback comparisons stay consistent.")
    if unmatched:
        st.warning(f"{len(unmatched)} observed district(s) could not be matched to the boundary file. "
                   "They remain available in the selector and comparison table; no location was guessed.")
    names = {f["district_key"]: f["location"] for f in features}
    names.update({district_key(row.state, row.district): f"{row.state} / {row.district}" for row in prices.itertuples()})
    options = sorted(names, key=names.get)
    selector_key = f"map_detail_{context_id}"
    if clicked in names and st.session_state.get("last_district_map_click") != (context_id, clicked):
        st.session_state[selector_key] = clicked
        st.session_state["last_district_map_click"] = (context_id, clicked)
        if st.session_state.get("price_playing"):
            stop_price_playback()
            st.rerun()
    first_observed = district_key(prices.iloc[0].state, prices.iloc[0].district)
    selected = st.selectbox("Choose a district", options, index=options.index(first_observed),
                            format_func=names.get, key=selector_key, on_change=stop_price_playback)
    other = st.selectbox("Compare with another district (optional)", [None] + [key for key in options if key != selected],
                         format_func=lambda key: "No comparison" if key is None else names[key], key=f"map_compare_{context_id}",
                         on_change=stop_price_playback)
    if st.session_state.get("playback_timer_active") and not st.session_state.get("price_playing"):
        st.rerun()
    district_history = prices.loc[[district_key(row.state, row.district) == selected for row in prices.itertuples()]]
    calendar = pd.date_range(observed - pd.Timedelta(weeks=12), observed, freq="W-MON")
    comparison = {names[selected]: district_history.set_index("week_start").weekly_median_price.reindex(calendar)}
    if other:
        other_history = prices.loc[[district_key(row.state, row.district) == other for row in prices.itertuples()]]
        comparison[names[other]] = other_history.set_index("week_start").weekly_median_price.reindex(calendar)
    price_history_chart(pd.DataFrame(comparison), item.unit, height=2.8)
    st.caption("Recorded weekly prices for the same item, unit and shop type. Gaps stay blank; different shops may be sampled each week.")
    local = rows.loc[[district_key(row.state, row.district) == selected for row in rows.itertuples()]]
    if local.empty:
        st.info("No eligible observation for this district in the selected week. This does not mean prices are low or stable.")
    else:
        row = local.iloc[0]
        st.markdown(f"**{names[selected]} · RM {row.weekly_median_price:.2f} per {row.unit}**")
        st.caption(f"{int(row.record_count)} price records · {int(row.premise_count)} shops · {int(row.observed_days)} observed days")
        current = latest.loc[latest[SERIES].eq(row[SERIES]).fillna(False).all(axis=1)]
        if observed != week or current.empty:
            st.info("No saved next-week forecast for this selected district/week. Older observations are price history only.")
        else:
            forecast = current.iloc[0]
            st.write("**Next-week signal: " + ("Alert to review**" if forecast.predicted_surge else "No alert raised**"))
            st.caption("Saved validation guidance: " + group_guidance(row.project_group, model))
            with st.expander("Why this model signal?"):
                explain_case(forecast, explanations, units)
    with st.expander("Compare recorded district prices"):
        table = rows.sort_values("weekly_median_price")[["state", "district", "weekly_median_price", "record_count", "premise_count", "observed_days"]]
        table = table.rename(columns={"state": "State", "district": "District", "weekly_median_price": f"Median (RM / {item.unit})",
                                      "record_count": "Records", "premise_count": "Shops", "observed_days": "Days"})
        st.dataframe(table, hide_index=True, width="stretch")
    st.caption("Sampled medians summarise the available recorded prices. "
               "Boundaries: Department of Statistics Malaysia, data-open (Open Data License). "
               "Boundary names may differ from historical PriceCatcher locations. No official endorsement.")


def global_explanation(importance):
    st.subheader("What the model tends to use overall (SHAP)")
    st.write("This is an overall summary across a saved reference sample. To understand one prediction, "
             "use the Understand a prediction tab instead.")
    if importance.empty:
        st.info("No saved overall SHAP summary is available.")
        return
    display = importance[["feature", "mean_abs_shap"]].sort_values(
        ["mean_abs_shap", "feature"], ascending=[False, True]).copy()
    display["feature"] = display.feature.map(lambda name: FEATURE_NAMES.get(name, name))
    display = display.rename(columns={"feature": "Feature", "mean_abs_shap": "Average contribution size"})
    st.bar_chart(display.head(10), x="Feature", y="Average contribution size", horizontal=True,
                 x_label="Average contribution size (model-output units)", color="#087f8c", height=350)
    st.caption("Longer bars mean a larger average influence on the model output, not a higher price or a probability. "
               "This summary shows contribution size, not whether a feature raises or lowers an individual score.")
    with st.expander("Overall SHAP values and reference sample"):
        st.dataframe(display, hide_index=True, width="stretch")
        st.caption("Mean absolute SHAP on a random sample of up to 200 original 2025 training inputs. "
                   "This is descriptive, not representative importance for the full longer-history training population. "
                   "Related price-history features can share information.")


def verified_test_baselines(folder, record, overall):
    """Read the optional comparison only when its hash and test support agree."""
    expected = record["output_sha256"].get("baseline_metrics.csv")
    if expected is None:
        return None  # Earlier evaluations remain usable without inventing baseline values.
    path = folder / "baseline_metrics.csv"
    if file_hash(path) != expected:
        raise ValueError("Saved baseline comparison changed; restore its matching evaluation output.")
    table = pd.read_csv(path)
    if not {"method", "rows", "observed_surges", "average_precision"}.issubset(table.columns):
        raise ValueError("Saved baseline comparison is missing required columns.")
    names = {"Frozen selected model": "Frozen model", "Always no surge": "Constant ranking (prevalence)",
             "Current-change rule (>=10%)": "Current-change rule",
             "Training food-group rate (ranking only)": "Training food-group rate"}
    if (table.empty or table.method.duplicated().any() or not table.method.isin(names).all()
            or table.method.eq("Frozen selected model").sum() != 1
            or not table.rows.eq(overall.rows).all()
            or not table.observed_surges.eq(overall.observed_surges).all()):
        raise ValueError("Saved baseline methods or test populations disagree.")
    ap = table.average_precision
    if ((ap.notna() & (~np.isfinite(ap) | ~ap.between(0, 1))).any()
            or not np.isclose(table.loc[table.method.eq("Frozen selected model"), "average_precision"].iloc[0],
                              overall.average_precision, equal_nan=True)):
        raise ValueError("Saved model and baseline AP values disagree or are invalid.")
    return table.assign(label=table.method.map(names))


def test_evaluation_summary(run, model, manifest):
    """Show only completed, hashed results for this exact model and test snapshot."""
    test_hash = manifest.get("upstream_sha256", {}).get("prepared/holdout/test.parquet")
    matches = []
    for path in (run / "research" / "12_test_set_evaluation").glob("*/evaluation_manifest.json"):
        record = read_json(path)
        identity = record.get("identity", {})
        if (record.get("status") == "completed" and test_hash is not None and model.get("deployment_id")
                and identity.get("deployment_id") == model.get("deployment_id")
                and identity.get("model_sha256") == model["sha256"]
                and identity.get("test_sha256") == test_hash
                and identity.get("score_cutoff") == model["score_cutoff"]):
            matches.append((record.get("completed_at", ""), path, record))
    if not matches:
        st.caption("No matching saved test evaluation is available here. Research 12 evaluates the frozen model separately; the dashboard never runs it automatically.")
        return
    _, path, record = sorted(matches, key=lambda item: (item[0], str(item[1])))[-1]
    for name in ["overall_metrics.csv", "confusion_matrix.csv"]:
        expected = record["output_sha256"][name]
        if file_hash(path.parent / name) != expected:
            raise ValueError("Saved test results changed; restore their matching evaluation outputs.")
    overall = pd.read_csv(path.parent / "overall_metrics.csv")
    if len(overall) != 1:
        raise ValueError("Expected one overall test result.")
    row = overall.iloc[0]
    if int(row.rows) != record["test_rows"] or int(row.tp + row.fp + row.fn + row.tn) != int(row.rows):
        raise ValueError("Test support and confusion counts disagree.")
    baselines = verified_test_baselines(path.parent, record, row)
    st.subheader("April–June test performance")
    st.write("Retrospective evaluation of the frozen model. This period was previously inspected during development; "
             "it is not an untouched independent test.")
    cols = st.columns(4)
    for col, name, label in zip(cols, ["average_precision", "precision", "recall", "f1"],
                                ["Average precision", "Precision", "Recall", "F1"]):
        value = row[name]
        shown = "Unavailable" if pd.isna(value) else f"{value:.1%}" if name in {"precision", "recall"} else f"{value:.3f}"
        col.metric(label, shown)
    if baselines is not None and baselines.average_precision.notna().any():
        shown = baselines.dropna(subset=["average_precision"]).sort_values("average_precision", ascending=True)
        fig, ax = plt.subplots(figsize=(8, 2.5))
        ax.barh(shown.label, shown.average_precision,
                color=np.where(shown.method.eq("Frozen selected model"), "#087f8c", "#a4b4bf"))
        for position, value in enumerate(shown.average_precision):
            ax.text(value + 0.008, position, f"{value:.3f}", va="center")
        ax.set(xlabel="Average precision (AP): higher means better ranking", xlim=(0, min(1.12, shown.average_precision.max() + 0.12)))
        ax.spines[["top", "right"]].set_visible(False)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
        st.caption("Identical test rows for every method. Constant-ranking AP equals test surge prevalence; "
                   "food-group rates, when shown, come from training only. AP is not classification accuracy.")
    else:
        st.caption("No verified baseline AP comparison is available for this saved evaluation.")
    with st.expander("April–June test results (previously exposed)"):
        weeks = record.get("actual_outcome_weeks", {})
        st.caption(f"{int(row.rows):,} eligible cases; {int(row.observed_surges):,} observed surges. "
                   f"Outcome weeks start {weeks.get('first', 'unavailable')} through {weeks.get('last', 'unavailable')}.")
        st.dataframe(pd.read_csv(path.parent / "confusion_matrix.csv"), hide_index=True, width="stretch")
        st.caption("Rows are actual outcomes; columns are predictions. Different periods and surge frequencies affect comparison with validation.")
        st.caption(f"Matching evaluation: {path.parent.name}. The latest matching completion is shown, never the highest score.")


def model_and_coverage(entry, latest, history):
    model = entry["forecast_model"]
    st.subheader("Evidence for the displayed forecasting model")
    version = model.get("display_name") or ("Longer history (2024–2025)" if model["version"] == "longer_2024_2025" else "Original 2025 reference")
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
        formatted = "Unavailable" if value is None or pd.isna(value) else f"{value:.1%}" if name in {"precision", "recall"} else f"{value:.3f}"
        col.metric(label, formatted)
    st.caption(f"Frozen score cutoff: {model['score_cutoff']:.5f}. The 10% price-increase target is a separate definition. No dashboard filter changes either threshold.")
    precision = metrics.get("precision")
    if precision is not None and pd.notna(precision):
        st.warning(f"In this validation period, {(1 - precision):.1%} of alerts were false alarms. Alerts require investigation.")
    if model["version"] == "longer_2024_2025":
        st.write("The saved longer-history study improves overall validation ranking, mainly for vegetables. Eggs and flour deteriorate. Sugar has no positive training/validation examples; cooking-oil positives are scarce. Independent test performance for this candidate remains unconfirmed.")
    if model.get("group_evidence"):
        groups = pd.DataFrame.from_dict(model["group_evidence"], orient="index")
        groups.index.name = "Food group"
        st.subheader("Validation evidence by food group")
        table = groups[["rows", "positives", "tp", "fp", "fn", "precision", "recall"]].reset_index()
        table[["precision", "recall"]] *= 100
        table = table.rename(columns={"rows": "Evaluated cases", "positives": "Observed surges",
            "tp": "Correct alerts", "fp": "False alerts", "fn": "Missed surges",
            "precision": "Alert precision (%)", "recall": "Surges found (%)"})
        st.dataframe(table, hide_index=True, width="stretch", column_config={
            "Alert precision (%)": st.column_config.NumberColumn(format="%.1f%%"),
            "Surges found (%)": st.column_config.NumberColumn(format="%.1f%%")})
        total = groups.tp.sum()
        if total > 0 and "Vegetables" in groups.index:
            st.warning(f"Vegetables contributed {groups.loc['Vegetables', 'tp']/total:.1%} of correct validation alerts. "
                       "Overall performance does not establish reliable detection across all foods.")
        st.caption("Fixed validation period and selected cutoff; these metrics are not recomputed by dashboard filters.")
    st.subheader("Coverage in your current filters")
    counts = latest.groupby("project_group").agg(supported_rows=("forecast_id", "size"),
        surge_alerts=("predicted_surge", "sum"), unusual_patterns=("is_anomaly", "sum"),
        cached_explanations=("has_cached_explanation", "sum")).reset_index()
    counts["Study guidance"] = counts.project_group.map(lambda group: group_guidance(group, model))
    st.dataframe(counts.rename(columns={"project_group": "Food group", "supported_rows": "Supported rows",
        "surge_alerts": "Surge alerts", "unusual_patterns": "Unusual patterns", "cached_explanations": "Cached explanations"}),
        hide_index=True, width="stretch")
    st.caption("Group guidance summarises this model's saved January–March 2026 validation results. "
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
    with st.expander("Study limitations and saved model identity"):
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
        st.caption("If you published a refresh in Notebook 13, rerun 13 to verify that saved snapshot. Keep the deployment pointers, archived exports and model inputs; the error details below identify the affected file.")
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
        cols = st.columns(3)
        for col, label, value, help_text in zip(cols,
            ["Local prices recorded", "Alerts to review", "Unusual price patterns"],
            [len(latest), int(latest.predicted_surge.sum()), int(latest.is_anomaly.sum())],
            ["One item in one district and shop type, with enough data for this saved forecast.",
             "Cases above the fixed review cutoff for a possible price rise of at least 10% next week. Some alerts will be wrong.",
             "A separate check found unusual observed prices. This is not a prediction of next week's price."]):
            col.metric(label, f"{value:,}", help=help_text)
    # Older supported Streamlit releases keep the existing tabs and a textual map pointer.
    can_navigate = {"key", "on_change"}.issubset(inspect.signature(st.tabs).parameters)
    snapshot_headline(latest, week, can_navigate)
    st.caption("Start with your area in the sidebar. Open Understand a prediction for dates, signals and SHAP explanations; "
               "use Unusual patterns in the watchlist for anomaly details. Alerts are suggestions to investigate, not certain price rises.")
    if entry["forecast_model"]["version"] == "longer_2024_2025" or entry["forecast_model"].get("group_evidence"):
        st.caption("The model works better for some foods than others. See How it works for the evidence and limitations.")
    with st.expander("First visit? A quick guide"):
        steps = st.columns(3)
        steps[0].markdown("**1 · Find your area**")
        steps[0].write("Use the sidebar to choose your state, district and food item. Counts and charts follow your filters.")
        steps[1].markdown("**2 · Understand the signal**")
        steps[1].write("A review alert concerns next week. An unusual-pattern flag concerns observed prices. Either signal needs investigation.")
        steps[2].markdown("**3 · Check the context**")
        steps[2].write("Use Understand a prediction for a saved SHAP explanation, or Price map & playback for weekly prices. "
                       "A missing forecast does not mean prices will stay stable.")
    if latest.empty:
        st.warning("No supported latest forecasts match these filters. Older history may still be available. This does not mean no surge risk.")
    review, local_map, predictions, evidence = st.tabs([
        "Latest saved prices", "Price map & playback", "Understand a prediction", "How it works"],
        **({"key": "dashboard_tab", "on_change": "rerun"} if can_navigate else {}))
    with review:
        spotlight(history, latest, explanations, entry["forecast_model"], refresh, manifest["protocol"]["units"])
        unusual_cases = review_list(latest, refresh, entry["forecast_model"])
        if unusual_cases is not None:
            anomaly_details(unusual_cases, history, run, entry, manifest)
        with st.expander("Food-group overview"):
            group_overview(latest)
        with st.expander("Find a specific item and location"):
            explore_series(history, latest, explanations, week, manifest["protocol"]["units"], entry["forecast_model"])
    with local_map:
        playing = st.session_state.get("price_playing", False)
        st.session_state["playback_timer_active"] = playing
        # Only the map section updates during playback; no models or explanations are computed.
        st.fragment(run_every=1.8 if playing else None)(local_price_explorer)(
            history, latest, week, explanations, manifest["protocol"]["units"], entry["forecast_model"])
    with predictions:
        prediction_explorer(latest, explanations, entry["forecast_model"], manifest["protocol"]["units"])
    with evidence:
        st.subheader("What the dashboard tells you")
        st.markdown("**Recorded price:** the middle price recorded for that item, district, shop type and week.\n\n"
                    "**Alert to review:** the model flags a possible price rise of at least 10% next week. Some alerts will be wrong, and some rises will be missed.\n\n"
                    "**Unusual pattern:** a separate check found unusual recent price behaviour. It is not a forecast or proof of an error.\n\n"
                    "**Grey map area:** there is not enough comparable recorded data. It does not mean prices are low or stable.")
        st.caption("This is a historical research demonstration, not a live warning service. No alert does not guarantee a stable price.")
        try:
            test_evaluation_summary(run, entry["forecast_model"], manifest)
        except (OSError, ValueError, KeyError, TypeError) as error:
            st.warning(f"Saved test results could not be verified: {error}")
        global_explanation(importance)
        with st.expander("Model results and research details"):
            model_and_coverage(entry, latest, history)
        with st.expander("Reading scores, explanations and missing prices"):
            st.write("Review scores prioritise investigation; they are not verified probabilities. An alert is not a guaranteed rise, "
                     "and no alert does not guarantee a stable price. SHAP explains the classifier's output, not economic causes "
                     "or the separate anomaly detector. Missing and excluded weeks stay blank; changing shops and sparse "
                     "observations can affect local medians.")
    st.caption("HargaLens · Malaysian PriceCatcher · Research prototype. Observed local medians; no affordability benefit has been established.")


if __name__ == "__main__":
    main()
