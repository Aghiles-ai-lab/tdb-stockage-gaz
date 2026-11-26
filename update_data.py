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
        ).dropna(
