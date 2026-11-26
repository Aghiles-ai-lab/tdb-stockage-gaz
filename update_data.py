import json
import os
import time
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import requests

# Lecture sécurisée de la clé API depuis les Secrets GitHub
AGSI_API_KEY = os.environ.get("AGSI_API_KEY")
if not AGSI_API_KEY:
    raise RuntimeError("La variable d'environnement AGSI_API_KEY est absente.")

HEADERS = {"x-key": AGSI_API_KEY}
API_URL = "https://agsi.gie.eu/api"
REQUEST_TIMEOUT = 60
MAX_RETRIES = 3
RETRY_WAIT_SEC = 5
PAGE_DELAY_SEC = 0.3

INITIAL_START_DATE = "2015-04-01"
CACHE_FILE = "agsi_cache.parquet"
DASHBOARD_JSON = "dashboard_data.json"
HTML_OUTPUT = "index.html"

CACHE_COLUMNS = [
    "perimeter",
    "date",
    "gasInStorage",
    "workingGasVolume",
    "full",
    "injection",
    "withdrawal",
]
NUMERIC_COLUMNS = ("gasInStorage", "workingGasVolume", "full", "injection", "withdrawal")

PERIMETERS = {
    "UE": {"type": "eu", "country": None, "company": None},
    "France": {"type": None, "country": "FR", "company": None},
    "Allemagne": {"type": None, "country": "DE", "company": None},
    "Italie": {"type": None, "country": "IT", "company": None},
    "Storengy France": {"type": None, "country": "FR", "company": "21X000000001083B"},
}

REF_DATES = pd.date_range("2025-04-01", "2026-03-31", freq="D")
MD_TO_IDX = {d.strftime("%m-%d"): i for i, d in enumerate(REF_DATES)}
IDX_TO_REF_DATE = [d.strftime("%Y-%m-%d") for d in REF_DATES]


def fetch_agsi_data(session, config, start_date, end_date):
    all_data = []
    page = 1

    while True:
        params = {"from": start_date, "to": end_date, "size": 300, "page": page}
        if config.get("country"):
            params["country"] = config["country"]
        if config.get("type"):
            params["type"] = config["type"]
        if config.get("company"):
            params["company"] = config["company"]

        response = None
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                response = session.get(API_URL, params=params, timeout=REQUEST_TIMEOUT)
            except requests.RequestException as exc:
                if attempt < MAX_RETRIES:
                    time.sleep(RETRY_WAIT_SEC * attempt)
                    continue
                print(f"\n   ❌ Erreur réseau : {exc}")
                return None

            if response.status_code == 200:
                break
            if response.status_code in (429, 500, 502, 503, 504) and attempt < MAX_RETRIES:
                time.sleep(RETRY_WAIT_SEC * attempt)
                continue

            print(f"\n   ❌ Erreur API ({response.status_code}) : {response.text}")
            return None

        data = response.json()
        items = data.get("data", [])
        if not items:
            break

        for item in items:
            all_data.append(
                {
                    "date": item.get("gasDayStart"),
                    "gasInStorage": item.get("gasInStorage"),
                    "workingGasVolume": item.get("workingGasVolume"),
                    "full": item.get("full"),
                    "injection": item.get("injection"),
                    "withdrawal": item.get("withdrawal"),
                }
            )

        if page >= data.get("last_page", 1):
            break
        page += 1
        time.sleep(PAGE_DELAY_SEC)

    return pd.DataFrame(all_data)


def get_april_season(date_obj):
    start_year = date_obj.year if date_obj.month >= 4 else date_obj.year - 1
    return f"{start_year}-{str(start_year + 1)[-2:]}"


def update_cache():
    cache_cleaned = False
    if os.path.exists(CACHE_FILE):
        cache_df = pd.read_parquet(CACHE_FILE)
        cache_df["date"] = pd.to_datetime(cache_df["date"])

        # Reconstruit automatiquement le cache s'il manque les colonnes injection / withdrawal
        if not {"injection", "withdrawal"}.issubset(cache_df.columns):
            print(
                f"🔄 Colonnes natives injection/withdrawal absentes du cache : "
                f"reconstruction complète depuis {INITIAL_START_DATE}..."
            )
            cache_df = pd.DataFrame(columns=CACHE_COLUMNS)
            cache_cleaned = True
        else:
            removed = set(cache_df["perimeter"].dropna().unique()) - set(PERIMETERS.keys())
            if removed:
                cache_df = cache_df[cache_df["perimeter"].isin(PERIMETERS.keys())].copy()
                cache_cleaned = True
            print(f"📂 Cache chargé ({len(cache_df)} lignes).")
    else:
        cache_df = pd.DataFrame(columns=CACHE_COLUMNS)
        print(f"⚠️ Premier téléchargement complet depuis {INITIAL_START_DATE}...")

    new_chunks = []
    today_str = datetime.today().strftime("%Y-%m-%d")
    has_new_data = False

    with requests.Session() as session:
        session.headers.update(HEADERS)

        for perimeter, config in PERIMETERS.items():
            print(f" -> {perimeter}", end=" ")
            p_cache = cache_df[cache_df["perimeter"] == perimeter]
            valid_cache = p_cache.dropna(subset=["full"]) if not p_cache.empty else p_cache

            if not valid_cache.empty:
                max_date = valid_cache["date"].max()
                # Re-télécharge les 7 derniers jours pour intégrer les consolidations AGSI+
                start_date = (max_date - timedelta(days=7)).strftime("%Y-%m-%d")
                last_str = max_date.strftime("%Y-%m-%d")
            else:
                start_date = INITIAL_START_DATE
                last_str = None

            df_new = fetch_agsi_data(session, config, start_date, today_str)
            if df_new is None:
                print("⚠️ Ignoré (erreur API).")
                continue
            if df_new.empty:
                print(f"⏸️ Déjà à jour ({last_str or 'aucune donnée'}).")
                continue

            df_new["perimeter"] = perimeter
            df_new["date"] = pd.to_datetime(df_new["date"])
            for col in NUMERIC_COLUMNS:
                df_new[col] = pd.to_numeric(df_new[col], errors="coerce")
            df_new = df_new.dropna(subset=["full"])

            if df_new.empty:
                print(f"⏸️ Aucune nouvelle journée complète ({last_str or 'aucune donnée'}).")
                continue

            new_chunks.append(df_new[CACHE_COLUMNS])
            has_new_data = True
            print(
                f"✅ {len(df_new)} jour(s) synchronisé(s) "
                f"(jusqu'au {df_new['date'].max().strftime('%Y-%m-%d')})."
            )

    if has_new_data or cache_cleaned:
        dfs = [df for df in [cache_df] + new_chunks if not df.empty]
        updated_df = pd.concat(dfs, ignore_index=True)
        updated_df.drop_duplicates(subset=["perimeter", "date"], keep="last", inplace=True)
        updated_df.sort_values(["perimeter", "date"], inplace=True)
        updated_df.to_parquet(CACHE_FILE, index=False)
        print(f"💾 Cache sauvegardé dans {CACHE_FILE}.")
        return updated_df, True

    return cache_df, False


def build_perimeter_payload(df_p):
    df_p = df_p.dropna(subset=["full"]).sort_values("date").copy()
    if df_p.empty:
        return None

    latest_row = df_p.iloc[-1]
    latest_date = latest_row["date"]
    current_season = get_april_season(latest_date)

    # --- 1. Taux de remplissage ---
    df_full = df_p.copy()
    df_full["md"] = df_full["date"].dt.strftime("%m-%d")
    df_full = df_full[df_full["md"].isin(MD_TO_IDX)].copy()
    df_full["day_idx"] = df_full["md"].map(MD_TO_IDX)
    df_full["season"] = df_full["date"].apply(get_april_season)

    pivot_full = (
        df_full.pivot_table(index="day_idx", columns="season", values="full", aggfunc="last")
        .reindex(range(365))
        .interpolate(method="linear", limit=3, limit_area="inside")
    )

    all_seasons = sorted(pivot_full.columns.tolist())
    past_seasons = [s for s in all_seasons if s < current_season]
    prev_season = past_seasons[-1] if past_seasons else None
    seasons_5y = past_seasons[-5:]
    seasons_10y = past_seasons[-10:]

    df_5y = pivot_full[seasons_5y] if seasons_5y else pd.DataFrame(index=range(365))
    mean_5y = df_5y.mean(axis=1) if not df_5y.empty else pd.Series(np.nan, index=range(365))
    min_5y = df_5y.min(axis=1) if not df_5y.empty else pd.Series(np.nan, index=range(365))
    max_5y = df_5y.max(axis=1) if not df_5y.empty else pd.Series(np.nan, index=range(365))

    current_series = pivot_full[current_season]
    prev_series = pivot_full[prev_season] if prev_season else pd.Series(np.nan, index=range(365))

    last_idx = int(current_series.last_valid_index())
    current_full = float(current_series.loc[last_idx])
    current_twh = float(latest_row["gasInStorage"]) if pd.notna(latest_row["gasInStorage"]) else None
    working_vol = float(latest_row["workingGasVolume"]) if pd.notna(latest_row["workingGasVolume"]) else None

    mean_5y_today = float(mean_5y.loc[last_idx]) if pd.notna(mean_5y.loc[last_idx]) else None
    gap_5y_pts = (current_full - mean_5y_today) if mean_5y_today is not None else None
    gap_5y_twh = (
        (gap_5y_pts / 100.0) * working_vol
        if (gap_5y_pts is not None and working_vol is not None)
        else None
    )
    prev_year_today = float(prev_series.loc[last_idx]) if pd.notna(prev_series.loc[last_idx]) else None

    proj_median = [None] * 365
    proj_min = [None] * 365
    proj_max = [None] * 365
    end_median = None
    end_min = None
    worst_season = None

    if last_idx < 364 and seasons_10y:
        trajectories = {}
        raw_end_vals = {}
        for s in seasons_10y:
            s_vals = pivot_full[s]
            if pd.isna(s_vals.loc[last_idx]) or pd.isna(s_vals.loc[364]):
                continue
            deltas = s_vals.diff().loc[last_idx + 1 : 364].fillna(0.0)
            raw_cum = current_full + deltas.cumsum().values
            raw_end_vals[s] = float(raw_cum[-1])
            trajectories[s] = np.concatenate([[current_full], np.clip(raw_cum, 0.0, 100.0)])

        if trajectories:
            traj_df = pd.DataFrame(trajectories, index=range(last_idx, 365))
            med_s = traj_df.median(axis=1)
            min_s = traj_df.min(axis=1)
            max_s = traj_df.max(axis=1)

            for idx in range(last_idx, 365):
                proj_median[idx] = round(float(med_s.loc[idx]), 2)
                proj_min[idx] = round(float(min_s.loc[idx]), 2)
                proj_max[idx] = round(float(max_s.loc[idx]), 2)

            end_vals = traj_df.loc[364]
            end_median = round(float(end_vals.median()), 1)
            worst_season = min(raw_end_vals, key=raw_end_vals.get)
            end_min = round(float(end_vals[worst_season]), 1)

    # --- 2. Soutirage net quotidien via champs natifs AGSI+ (withdrawal - injection, en GWh/j) ---
    # Positif : soutirage net, négatif : injection nette
    df_flows = (
        df_p.dropna(subset=["withdrawal", "injection"])
        .drop_duplicates(subset=["date"], keep="last")
        .sort_values("date")
    )

    wd_current = pd.Series(np.nan, index=range(365))
    wd_prev = pd.Series(np.nan, index=range(365))
    wd_mean_5y = pd.Series(np.nan, index=range(365))
    wd_min_5y = pd.Series(np.nan, index=range(365))
    wd_max_5y = pd.Series(np.nan, index=range(365))
    wd_last_idx = last_idx
    wd_curr_val = None
    wd_latest_daily = None
    wd_mean_today = None
    wd_gap_5y = None
    wd_prev_today = None
    wd_peak_val = None
    wd_peak_date = None

    if not df_flows.empty:
        net_wd_daily = (
            (df_flows.set_index("date")["withdrawal"] - df_flows.set_index("date")["injection"])
            .sort_index()
            .asfreq("D")
            .interpolate(method="linear", limit=3, limit_area="inside")
        )
        net_wd_7d = net_wd_daily.rolling(window=7, min_periods=4).mean()

        df_wd = pd.DataFrame(
            {
                "date": net_wd_7d.index,
                "wd_daily": net_wd_daily.values,
                "wd_7d": net_wd_7d.values,
            }
        ).dropna(subset=["wd_7d"])
        df_wd["md"] = df_wd["date"].dt.strftime("%m-%d")
        df_wd = df_wd[df_wd["md"].isin(MD_TO_IDX)].copy()
        df_wd["day_idx"] = df_wd["md"].map(MD_TO_IDX)
        df_wd["season"] = df_wd["date"].apply(get_april_season)

        pivot_wd = (
            df_wd.pivot_table(index="day_idx", columns="season", values="wd_7d", aggfunc="last")
            .reindex(range(365))
            .interpolate(method="linear", limit=3, limit_area="inside")
        )

        wd_seasons_5y = [s for s in seasons_5y if s in pivot_wd.columns]
        df_wd_5y = pivot_wd[wd_seasons_5y] if wd_seasons_5y else pd.DataFrame(index=range(365))
        if not df_wd_5y.empty:
            wd_mean_5y = df_wd_5y.mean(axis=1)
            wd_min_5y = df_wd_5y.min(axis=1)
            wd_max_5y = df_wd_5y.max(axis=1)

        if current_season in pivot_wd.columns:
            wd_current = pivot_wd[current_season]
        if prev_season and prev_season in pivot_wd.columns:
            wd_prev = pivot_wd[prev_season]

        if wd_current.notna().any():
            wd_last_idx = int(wd_current.last_valid_index())
            wd_curr_val = float(wd_current.loc[wd_last_idx])
            if latest_date in net_wd_daily.index and pd.notna(net_wd_daily.loc[latest_date]):
                wd_latest_daily = float(net_wd_daily.loc[latest_date])
            elif not net_wd_daily.dropna().empty:
                wd_latest_daily = float(net_wd_daily.dropna().iloc[-1])

            wd_mean_today = float(wd_mean_5y.loc[wd_last_idx]) if pd.notna(wd_mean_5y.loc[wd_last_idx]) else None
            wd_gap_5y = (wd_curr_val - wd_mean_today) if wd_mean_today is not None else None
            wd_prev_today = float(wd_prev.loc[wd_last_idx]) if pd.notna(wd_prev.loc[wd_last_idx]) else None

        df_wd_curr = df_wd[df_wd["season"] == current_season]
        if not df_wd_curr.empty:
            peak_row = df_wd_curr.loc[df_wd_curr["wd_7d"].idxmax()]
            wd_peak_val = round(float(peak_row["wd_7d"]), 0)
            wd_peak_date = peak_row["date"].strftime("%Y-%m-%d")

    def to_list(series, decimals=2):
        return [round(float(v), decimals) if pd.notna(v) else None for v in series]

    return {
        "meta": {
            "latest_date": latest_date.strftime("%Y-%m-%d"),
            "current_season": current_season,
            "prev_season": prev_season,
            "seasons_5y_range": f"{seasons_5y[0]} → {seasons_5y[-1]}" if seasons_5y else "",
            "last_day_idx": last_idx,
            "wd_last_day_idx": wd_last_idx,
            "n_proj_winters": len(seasons_10y),
        },
        "kpi": {
            "full_pct": round(current_full, 1),
            "gas_twh": round(current_twh, 0) if current_twh is not None else None,
            "working_twh": round(working_vol, 0) if working_vol is not None else None,
            "mean_5y_pct": round(mean_5y_today, 1) if mean_5y_today is not None else None,
            "gap_5y_pts": round(gap_5y_pts, 1) if gap_5y_pts is not None else None,
            "gap_5y_twh": round(gap_5y_twh, 0) if gap_5y_twh is not None else None,
            "prev_year_pct": round(prev_year_today, 1) if prev_year_today is not None else None,
            "proj_end_median_pct": end_median,
            "proj_end_min_pct": end_min,
            "proj_worst_season": worst_season,
        },
        "series": {
            "ref_dates": IDX_TO_REF_DATE,
            "current": to_list(current_series, 2),
            "prev": to_list(prev_series, 2),
            "mean_5y": to_list(mean_5y, 2),
            "min_5y": to_list(min_5y, 2),
            "max_5y": to_list(max_5y, 2),
            "proj_median": proj_median,
            "proj_min": proj_min,
            "proj_max": proj_max,
        },
        "withdrawal": {
            "kpi": {
                "current_7d_gwh": round(wd_curr_val, 0) if wd_curr_val is not None else None,
                "latest_daily_gwh": round(wd_latest_daily, 0) if wd_latest_daily is not None else None,
                "mean_5y_gwh": round(wd_mean_today, 0) if wd_mean_today is not None else None,
                "gap_5y_gwh": round(wd_gap_5y, 0) if wd_gap_5y is not None else None,
                "prev_year_gwh": round(wd_prev_today, 0) if wd_prev_today is not None else None,
                "peak_7d_gwh": wd_peak_val,
                "peak_date": wd_peak_date,
            },
            "series": {
                "current": to_list(wd_current, 1),
                "prev": to_list(wd_prev, 1),
                "mean_5y": to_list(wd_mean_5y, 1),
                "min_5y": to_list(wd_min_5y, 1),
                "max_5y": to_list(wd_max_5y, 1),
            },
        },
    }


HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="fr">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>TdB Stockages de gaz</title>
<link rel="preconnect" href="https://fonts.googleapis.com" />
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin />
<link href="https://fonts.googleapis.com/css2?family=Questrial&display=swap" rel="stylesheet" />
<style>
  :root {
    --font-main: "Century Gothic", CenturyGothic, AppleGothic, "Questrial", sans-serif;
    --c-noir: #1C1C1C;
    --c-rouge: #FC4128;
    --c-bleu: #69A3F5;
    --c-jaune: #E6D748;
    --c-violet: #7E45AA;
    --c-vert: #5CA757;
    --c-bleu-clair: #BACFF7;
  }
  body {
    margin: 0;
    padding: 18px;
    background: #F6F6F5;
    font-family: var(--font-main);
    color: var(--c-noir);
  }
  .obs-wrap {
    background: #FFFFFF;
    max-width: 1080px;
    margin: 0 auto;
    padding: 24px 28px 18px;
    border: 1px solid #E6E6E6;
    border-radius: 10px;
    box-sizing: border-box;
  }
  .obs-header {
    display: flex;
    justify-content: space-between;
    align-items: flex-start;
    gap: 16px;
    flex-wrap: wrap;
    margin-bottom: 14px;
  }
  .obs-title {
    font-family: var(--font-main);
    font-size: 30px;
    font-weight: 700;
    margin: 0 0 4px 0;
    line-height: 1.15;
    color: var(--c-noir);
    letter-spacing: -0.01em;
  }
  .obs-subtitle {
    font-size: 14px;
    color: #555555;
    margin: 0;
  }
  .obs-select {
    font-family: var(--font-main);
    font-size: 14px;
    font-weight: 600;
    padding: 8px 14px;
    border-radius: 8px;
    border: 1.5px solid var(--c-bleu);
    background: #FFFFFF;
    color: var(--c-noir);
    cursor: pointer;
    margin-top: 4px;
  }
  .obs-tabs {
    display: flex;
    gap: 6px;
    border-bottom: 1.5px solid #EAEAEA;
    margin-bottom: 18px;
  }
  .obs-tab {
    font-family: var(--font-main);
    font-size: 14px;
    font-weight: 600;
    color: #666666;
    background: transparent;
    border: none;
    border-bottom: 3px solid transparent;
    padding: 8px 16px 10px;
    margin-bottom: -1.5px;
    cursor: pointer;
    transition: color 0.15s, border-color 0.15s;
  }
  .obs-tab:hover {
    color: var(--c-noir);
  }
  .obs-tab.active {
    color: var(--c-noir);
    border-bottom-color: var(--c-bleu);
  }
  .tab-panel.hidden {
    display: none;
  }
  .obs-grid {
    display: grid;
    grid-template-columns: 260px 1fr;
    gap: 24px;
    align-items: start;
  }
  .obs-kpis {
    display: flex;
    flex-direction: column;
    gap: 12px;
  }
  .obs-kpis-row {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 12px;
    margin-bottom: 18px;
  }
  @media (max-width: 780px) {
    .obs-grid { grid-template-columns: 1fr; }
    .obs-kpis-row { grid-template-columns: repeat(2, 1fr); }
  }
  .kpi-card {
    background: #F5F6F8;
    border-left: 4px solid var(--c-bleu);
    border-radius: 8px;
    padding: 13px 16px;
  }
  .kpi-card.card-gap-neg { border-left-color: var(--c-rouge); }
  .kpi-card.card-gap-pos { border-left-color: var(--c-vert); }
  .kpi-card.card-prev { border-left-color: var(--c-noir); }
  .kpi-card.card-proj { border-left-color: var(--c-violet); }
  .kpi-label {
    font-size: 12.5px;
    color: #555555;
    margin-bottom: 4px;
  }
  .kpi-value {
    font-family: var(--font-main);
    font-size: 25px;
    font-weight: 700;
    line-height: 1.15;
    color: var(--c-noir);
  }
  .kpi-neg { color: var(--c-rouge); }
  .kpi-pos { color: var(--c-vert); }
  .kpi-sub {
    font-size: 12px;
    color: #606060;
    margin-top: 5px;
    line-height: 1.35;
  }
  .obs-chart-title {
    font-size: 16px;
    font-weight: 700;
    margin: 2px 0 4px 0;
    color: var(--c-noir);
  }
  .obs-chart-sub {
    font-size: 13px;
    color: #666666;
    margin: 0 0 10px 0;
  }
  .obs-footnote {
    font-size: 11.5px;
    color: #666666;
    border-top: 1px solid #E8E8E8;
    padding-top: 10px;
    margin-top: 14px;
  }
</style>
</head>
<body>
<div class="obs-wrap">
  <div class="obs-header">
    <div>
      <h1 class="obs-title" id="obs-title"></h1>
      <p class="obs-subtitle" id="obs-subtitle"></p>
    </div>
    <select class="obs-select" id="obs-select"></select>
  </div>

  <div class="obs-tabs">
    <button class="obs-tab active" data-tab="storage">Taux de remplissage</button>
    <button class="obs-tab" data-tab="withdrawal">Soutirage net quotidien</button>
  </div>

  <div id="panel-storage" class="tab-panel">
    <div class="obs-grid">
      <div class="obs-kpis" id="obs-kpis"></div>
      <div>
        <div class="obs-chart-title" id="obs-chart-title"></div>
        <div id="obs-chart" style="width:100%; height:440px;"></div>
      </div>
    </div>
  </div>

  <div id="panel-withdrawal" class="tab-panel hidden">
    <div class="obs-kpis-row" id="wd-kpis"></div>
    <div class="obs-chart-title" id="wd-chart-title"></div>
    <div class="obs-chart-sub">Moyenne mobile sur 7 jours, en GWh/j (flux natifs AGSI+ : soutirage − injection). Positif : soutirage, négatif : injection.</div>
    <div id="wd-chart" style="width:100%; height:440px;"></div>
  </div>

  <div class="obs-footnote" id="obs-footnote"></div>
</div>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<script>
(function () {
  const DATA = __DATA_JSON__;
  const FONT_MAIN = '"Century Gothic", CenturyGothic, AppleGothic, "Questrial", sans-serif';
  const C = {
    noir: "#1C1C1C",
    rouge: "#FC4128",
    bleu: "#69A3F5",
    jaune: "#E6D748",
    violet: "#7E45AA",
    vert: "#5CA757",
    bleuClairRgba: "rgba(186, 207, 247, 0.42)",
    minMaxRgba: "rgba(28, 28, 28, 0.11)",
    zoneObjRgba: "rgba(28, 28, 28, 0.04)"
  };
  const MONTHS_FR = ["janv.", "févr.", "mars", "avr.", "mai", "juin", "juil.", "août", "sept.", "oct.", "nov.", "déc."];

  let activeTab = "storage";

  function fmtFr(val, decimals = 1, signed = false) {
    if (val === null || val === undefined || Number.isNaN(val)) return "—";
    let prefix = "";
    if (signed) {
      if (val > 0) prefix = "+";
      else if (val < 0) prefix = "−";
    } else if (val < 0) {
      prefix = "−";
    }
    const n = Math.abs(val);
    return prefix + n.toLocaleString("fr-FR", {
      minimumFractionDigits: decimals,
      maximumFractionDigits: decimals
    });
  }

  function fmtDateFr(iso) {
    if (!iso) return "";
    const [y, m, d] = iso.split("-").map(Number);
    return `${d} ${MONTHS_FR[m - 1]} ${y}`;
  }

  function renderStorage(perimeter, p) {
    const meta = p.meta;
    const kpi = p.kpi;
    const s = p.series;
    const x = s.ref_dates;
    const lastIdx = meta.last_day_idx;

    document.getElementById("obs-chart-title").textContent =
      perimeter === "UE"
        ? "Taux de remplissage des stockages de l'UE"
        : `Taux de remplissage des stockages — ${perimeter}`;
    document.getElementById("obs-footnote").textContent =
      `Source : GIE AGSI+ (agsi.gie.eu). Taux = stock / volume utile. Projection : profils de stock des ${meta.n_proj_winters} hivers passés rejoués depuis le stock actuel.`;

    const volTxt = (kpi.gas_twh !== null && kpi.working_twh !== null)
      ? `${fmtFr(kpi.gas_twh, 0)} TWh sur ${fmtFr(kpi.working_twh, 0)} TWh`
      : "";
    const isNeg = kpi.gap_5y_pts !== null && kpi.gap_5y_pts < 0;
    const arrow = isNeg ? "▼" : "▲";
    const gapCls = isNeg ? "kpi-neg" : "kpi-pos";
    const gapCardCls = isNeg ? "card-gap-neg" : "card-gap-pos";
    const gapTwhTxt = kpi.gap_5y_twh !== null ? `${fmtFr(kpi.gap_5y_twh, 0, true)} TWh vs ` : "vs ";
    const projMain = kpi.proj_end_median_pct !== null ? `${fmtFr(kpi.proj_end_median_pct, 0)} %` : "Fin de cycle";
    const projSub = kpi.proj_end_median_pct !== null
      ? `médiane · hiver le plus tendu (${kpi.proj_worst_season}) : ${fmtFr(kpi.proj_end_min_pct, 0)} %`
      : "Projection active avant le 31 mars";

    document.getElementById("obs-kpis").innerHTML = `
      <div class="kpi-card">
        <div class="kpi-label">Remplissage ${perimeter}</div>
        <div class="kpi-value">${fmtFr(kpi.full_pct, 1)} %</div>
        <div class="kpi-sub">${volTxt}</div>
      </div>
      <div class="kpi-card ${gapCardCls}">
        <div class="kpi-label">Écart à la moyenne 5 ans</div>
        <div class="kpi-value ${gapCls}">${arrow} ${fmtFr(kpi.gap_5y_pts, 1, true)} pts</div>
        <div class="kpi-sub">${gapTwhTxt}${fmtFr(kpi.mean_5y_pct, 1)} % (${meta.seasons_5y_range})</div>
      </div>
      <div class="kpi-card card-prev">
        <div class="kpi-label">Un an plus tôt</div>
        <div class="kpi-value">${fmtFr(kpi.prev_year_pct, 1)} %</div>
        <div class="kpi-sub">au même jour (${meta.prev_season || "N-1"})</div>
      </div>
      <div class="kpi-card card-proj">
        <div class="kpi-label">Projection au 31 mars</div>
        <div class="kpi-value">${projMain}</div>
        <div class="kpi-sub">${projSub}</div>
      </div>
    `;

    const traces = [
      { x, y: s.min_5y, mode: "lines", line: { width: 0 }, showlegend: false, hoverinfo: "skip" },
      { x, y: s.max_5y, mode: "lines", line: { width: 0 }, fill: "tonexty", fillcolor: C.minMaxRgba, name: "Min–max 5 ans", legendrank: 5, hoverinfo: "skip" },
      { x, y: s.proj_min, mode: "lines", line: { width: 0 }, showlegend: false, hoverinfo: "skip" },
      { x, y: s.proj_max, mode: "lines", line: { width: 0 }, fill: "tonexty", fillcolor: C.bleuClairRgba, showlegend: false, hoverinfo: "skip" },
      { x, y: s.mean_5y, mode: "lines", line: { color: C.noir, width: 1.7 }, name: "Moy. 5 ans", legendrank: 4, hovertemplate: "Moy. 5 ans : %{y:.1f} %<extra></extra>" },
      { x, y: s.prev, mode: "lines", line: { color: C.rouge, width: 2.1 }, name: meta.prev_season || "N-1", legendrank: 3, hovertemplate: `${meta.prev_season || "N-1"} : %{y:.1f} %<extra></extra>` },
      { x, y: s.current, mode: "lines", line: { color: C.bleu, width: 3.3 }, name: meta.current_season, legendrank: 2, hovertemplate: `${meta.current_season} : %{y:.1f} %<extra></extra>` },
      { x, y: s.proj_median, mode: "lines", line: { color: C.bleu, width: 2.3, dash: "dash" }, name: "Projection", legendrank: 1, hovertemplate: "Projection médiane : %{y:.1f} %<extra></extra>" },
      { x: [x[lastIdx]], y: [kpi.full_pct], mode: "markers", marker: { color: C.bleu, size: 10, line: { color: "#FFFFFF", width: 2 } }, showlegend: false, hoverinfo: "skip" }
    ];

    const annotations = [
      { x: x[lastIdx], y: kpi.full_pct, text: `<b>${fmtFr(kpi.full_pct, 1)} %</b>`, showarrow: false, xshift: -26, yshift: -20, font: { family: FONT_MAIN, size: 13, color: C.noir } },
      { x: "2025-11-01", y: 92, text: "Objectif UE 90 %", showarrow: false, font: { family: FONT_MAIN, size: 11.5, color: C.noir } }
    ];

    if (kpi.proj_end_min_pct !== null) {
      annotations.push({
        x: "2026-03-31",
        y: Math.max(kpi.proj_end_min_pct, 3),
        text: `${fmtFr(kpi.proj_end_min_pct, 0)} % (profil ${kpi.proj_worst_season})`,
        showarrow: false, xanchor: "right", yshift: 10, font: { family: FONT_MAIN, size: 11.5, color: "#555555" }
      });
    }

    const layout = {
      font: { family: FONT_MAIN, color: C.noir },
      separators: ", ",
      margin: { l: 44, r: 16, t: 32, b: 30 },
      plot_bgcolor: "#FFFFFF",
      paper_bgcolor: "#FFFFFF",
      hovermode: "x unified",
      legend: { orientation: "h", yanchor: "bottom", y: 1.02, xanchor: "left", x: 0, font: { family: FONT_MAIN, size: 12, color: C.noir } },
      shapes: [
        { type: "rect", xref: "x", yref: "paper", x0: "2025-10-01", x1: "2025-12-01", y0: 0, y1: 1, fillcolor: C.zoneObjRgba, line: { width: 0 }, layer: "below" },
        { type: "line", xref: "x", yref: "y", x0: "2025-10-01", x1: "2025-12-01", y0: 90, y1: 90, line: { color: C.noir, width: 1.5, dash: "dot" } }
      ],
      annotations,
      xaxis: {
        tickmode: "array",
        tickvals: ["2025-04-01", "2025-06-01", "2025-08-01", "2025-10-01", "2025-12-01", "2026-02-01", "2026-03-31"],
        ticktext: ["avr.", "juin", "août", "oct.", "déc.", "févr.", "avr."],
        showgrid: false, linecolor: "#DDDDDD", tickfont: { family: FONT_MAIN, color: "#555555", size: 12 }
      },
      yaxis: { range: [0, 102], dtick: 20, ticksuffix: " %", gridcolor: "#EFEFEF", zeroline: false, tickfont: { family: FONT_MAIN, color: "#555555", size: 12 } }
    };

    Plotly.newPlot("obs-chart", traces, layout, { displayModeBar: false, responsive: true });
  }

  function renderWithdrawal(perimeter, p) {
    const meta = p.meta;
    const x = p.series.ref_dates;
    const w = p.withdrawal;
    const wk = w.kpi;
    const ws = w.series;
    const wdLastIdx = meta.wd_last_day_idx ?? meta.last_day_idx;

    const perimLabel = perimeter === "UE" ? "Union européenne" : perimeter;
    document.getElementById("wd-chart-title").textContent = `${perimLabel} : soutirage net quotidien`;
    document.getElementById("obs-footnote").textContent =
      `Source : GIE AGSI+ (agsi.gie.eu). Soutirage net quotidien issu des flux natifs AGSI+ (withdrawal − injection, en GWh/j, moyenne mobile 7 jours). Positif : soutirage, négatif : injection.`;

    const isWdHigher = wk.gap_5y_gwh !== null && wk.gap_5y_gwh > 0;
    const wdArrow = isWdHigher ? "▲" : "▼";
    const wdGapCls = isWdHigher ? "kpi-neg" : "kpi-pos";
    const wdGapCardCls = isWdHigher ? "card-gap-neg" : "card-gap-pos";
    const regimeTxt = wk.current_7d_gwh !== null
      ? (wk.current_7d_gwh >= 0 ? "Soutirage net" : "Injection nette")
      : "";
    const dailySub = wk.latest_daily_gwh !== null
      ? `${regimeTxt} · jour J : ${fmtFr(wk.latest_daily_gwh, 0, true)} GWh/j`
      : regimeTxt;
    const peakSub = wk.peak_date ? `le ${fmtDateFr(wk.peak_date)} (moy. 7 j)` : "saison en cours";

    document.getElementById("wd-kpis").innerHTML = `
      <div class="kpi-card">
        <div class="kpi-label">Soutirage net (moy. 7 j)</div>
        <div class="kpi-value">${fmtFr(wk.current_7d_gwh, 0)} GWh/j</div>
        <div class="kpi-sub">${dailySub}</div>
      </div>
      <div class="kpi-card ${wdGapCardCls}">
        <div class="kpi-label">Écart à la moyenne 5 ans</div>
        <div class="kpi-value ${wdGapCls}">${wdArrow} ${fmtFr(wk.gap_5y_gwh, 0, true)} GWh/j</div>
        <div class="kpi-sub">vs ${fmtFr(wk.mean_5y_gwh, 0)} GWh/j (${meta.seasons_5y_range})</div>
      </div>
      <div class="kpi-card card-prev">
        <div class="kpi-label">Un an plus tôt</div>
        <div class="kpi-value">${fmtFr(wk.prev_year_gwh, 0)} GWh/j</div>
        <div class="kpi-sub">au même jour (${meta.prev_season || "N-1"})</div>
      </div>
      <div class="kpi-card card-proj">
        <div class="kpi-label">Pic de soutirage (${meta.current_season})</div>
        <div class="kpi-value">${fmtFr(wk.peak_7d_gwh, 0)} GWh/j</div>
        <div class="kpi-sub">${peakSub}</div>
      </div>
    `;

    const traces = [
      { x, y: ws.min_5y, mode: "lines", line: { width: 0 }, showlegend: false, hoverinfo: "skip" },
      { x, y: ws.max_5y, mode: "lines", line: { width: 0 }, fill: "tonexty", fillcolor: C.minMaxRgba, name: `Min–max ${meta.seasons_5y_range}`, legendrank: 4, hoverinfo: "skip" },
      { x, y: ws.mean_5y, mode: "lines", line: { color: C.noir, width: 1.6 }, name: "Moyenne 5 ans", legendrank: 3, hovertemplate: "Moy. 5 ans : %{y:,.0f} GWh/j<extra></extra>" },
      { x, y: ws.prev, mode: "lines", line: { color: C.rouge, width: 2.0 }, name: meta.prev_season || "N-1", legendrank: 2, hovertemplate: `${meta.prev_season || "N-1"} : %{y:,.0f} GWh/j<extra></extra>` },
      { x, y: ws.current, mode: "lines", line: { color: C.bleu, width: 3.1 }, name: meta.current_season, legendrank: 1, hovertemplate: `${meta.current_season} : %{y:,.0f} GWh/j<extra></extra>` }
    ];

    const annotations = [];
    if (wk.current_7d_gwh !== null && ws.current[wdLastIdx] !== null) {
      const yDot = ws.current[wdLastIdx];
      traces.push({
        x: [x[wdLastIdx]],
        y: [yDot],
        mode: "markers",
        marker: { color: C.bleu, size: 10, line: { color: "#FFFFFF", width: 2 } },
        showlegend: false,
        hoverinfo: "skip"
      });

      const nearRight = wdLastIdx >= 295;
      const aboveMean = wk.mean_5y_gwh === null || yDot >= wk.mean_5y_gwh;
      annotations.push({
        x: x[wdLastIdx],
        y: yDot,
        text: `<b>${fmtFr(wk.current_7d_gwh, 0)} GWh/j</b>`,
        showarrow: true,
        arrowhead: 0,
        arrowwidth: 1.3,
        arrowcolor: C.bleu,
        ax: nearRight ? -64 : 64,
        ay: aboveMean ? -30 : 32,
        bgcolor: "rgba(255, 255, 255, 0.94)",
        bordercolor: C.bleu,
        borderwidth: 1.2,
        borderpad: 4,
        font: { family: FONT_MAIN, size: 12.5, color: C.noir }
      });
    }

    const layout = {
      font: { family: FONT_MAIN, color: C.noir },
      separators: ", ",
      margin: { l: 56, r: 20, t: 32, b: 30 },
      plot_bgcolor: "#FFFFFF",
      paper_bgcolor: "#FFFFFF",
      hovermode: "x unified",
      legend: { orientation: "h", yanchor: "bottom", y: 1.02, xanchor: "left", x: 0, font: { family: FONT_MAIN, size: 12, color: C.noir } },
      annotations,
      xaxis: {
        tickmode: "array",
        tickvals: [
          "2025-04-01", "2025-05-01", "2025-06-01", "2025-07-01", "2025-08-01", "2025-09-01",
          "2025-10-01", "2025-11-01", "2025-12-01", "2026-01-01", "2026-02-01", "2026-03-01", "2026-03-31"
        ],
        ticktext: ["avr.", "mai", "juin", "juil.", "août", "sept.", "oct.", "nov.", "déc.", "janv.", "févr.", "mars", "avr."],
        showgrid: false,
        linecolor: "#DDDDDD",
        tickfont: { family: FONT_MAIN, color: "#555555", size: 12 }
      },
      yaxis: {
        tickformat: ",.0f",
        gridcolor: "#EFEFEF",
        zeroline: true,
        zerolinecolor: "#A8A8A8",
        zerolinewidth: 1.3,
        tickfont: { family: FONT_MAIN, color: "#555555", size: 12 }
      }
    };

    Plotly.newPlot("wd-chart", traces, layout, { displayModeBar: false, responsive: true });
  }

  function render(perimeter) {
    const p = DATA.perimeters[perimeter];
    const meta = p.meta;

    const titleSuffix = perimeter === "UE" ? "en Europe" : `— ${perimeter}`;
    document.getElementById("obs-title").textContent = `Réserves de gaz naturel ${titleSuffix}`;
    document.getElementById("obs-subtitle").innerHTML =
      `Données au <b>${fmtDateFr(meta.latest_date)}</b> (fin de journée gazière) · Source : GIE AGSI+`;

    if (activeTab === "storage") {
      renderStorage(perimeter, p);
    } else {
      renderWithdrawal(perimeter, p);
    }
  }

  const sel = document.getElementById("obs-select");
  const keys = Object.keys(DATA.perimeters);
  sel.innerHTML = keys.map(k => `<option value="${k}">${k}</option>`).join("");
  sel.addEventListener("change", () => render(sel.value));

  document.querySelectorAll(".obs-tab").forEach(btn => {
    btn.addEventListener("click", () => {
      activeTab = btn.dataset.tab;
      document.querySelectorAll(".obs-tab").forEach(b => b.classList.toggle("active", b === btn));
      document.getElementById("panel-storage").classList.toggle("hidden", activeTab !== "storage");
      document.getElementById("panel-withdrawal").classList.toggle("hidden", activeTab !== "withdrawal");
      render(sel.value);
    });
  });

  render(keys[0]);
})();
</script>
</body>
</html>
"""


def main():
    df, _ = update_cache()
    if df.empty:
        print("⚠️ Aucune donnée disponible.")
        return

    print("🧮 Recalcul des indicateurs...")
    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "perimeters": {},
    }
    for perimeter in PERIMETERS.keys():
        p_data = build_perimeter_payload(df[df["perimeter"] == perimeter])
        if p_data:
            payload["perimeters"][perimeter] = p_data

    with open(DASHBOARD_JSON, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    data_json = json.dumps(payload, ensure_ascii=False).replace("</", "<\\/")
    with open(HTML_OUTPUT, "w", encoding="utf-8") as f:
        f.write(HTML_TEMPLATE.replace("__DATA_JSON__", data_json))

    print(f"✅ Mise à jour terminée ({HTML_OUTPUT} prêt).")


if __name__ == "__main__":
    main()
