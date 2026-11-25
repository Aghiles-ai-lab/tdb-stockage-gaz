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
        removed = set(cache_df["perimeter"].dropna().unique()) - set(PERIMETERS.keys())
        if removed:
            cache_df = cache_df[cache_df["perimeter"].isin(PERIMETERS.keys())].copy()
            cache_cleaned = True
        print(f"📂 Cache chargé ({len(cache_df)} lignes).")
    else:
        cache_df = pd.DataFrame(
            columns=["perimeter", "date", "gasInStorage", "workingGasVolume", "full"]
        )
        print(f"⚠️ Premier téléchargement complet depuis {INITIAL_START_DATE}...")

    new_chunks = []
    today_str = datetime.today().strftime("%Y-%m-%d")
    has_new_data = False

    with requests.Session() as session:
        session.headers.update(HEADERS)

        for perimeter, config in PERIMETERS.items():
            print(f" -> {perimeter}", end=" ")
            p_cache = cache_df[cache_df["perimeter"] == perimeter]

            if not p_cache.empty:
                max_date = p_cache["date"].max()
                start_date = (max_date + timedelta(days=1)).strftime("%Y-%m-%d")
                last_str = max_date.strftime("%Y-%m-%d")
            else:
                start_date = INITIAL_START_DATE
                last_str = None

            if start_date > today_str:
                print(f"⏸️ Déjà à jour ({last_str}).")
                continue

            df_new = fetch_agsi_data(session, config, start_date, today_str)
            if df_new is None:
                print("⚠️ Ignoré (erreur API).")
                continue
            if df_new.empty:
                print(f"⏸️ Déjà à jour ({last_str or 'aucune donnée'}).")
                continue

            df_new["perimeter"] = perimeter
            df_new["date"] = pd.to_datetime(df_new["date"])
            for col in ("gasInStorage", "workingGasVolume", "full"):
                df_new[col] = pd.to_numeric(df_new[col], errors="coerce")

            new_chunks.append(df_new)
            has_new_data = True
            print(f"✅ +{len(df_new)} jour(s).")

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

    df_p["md"] = df_p["date"].dt.strftime("%m-%d")
    df_p = df_p[df_p["md"].isin(MD_TO_IDX)].copy()
    df_p["day_idx"] = df_p["md"].map(MD_TO_IDX)
    df_p["season"] = df_p["date"].apply(get_april_season)

    pivot_full = (
        df_p.pivot_table(index="day_idx", columns="season", values="full", aggfunc="last")
        .reindex(range(365))
        .interpolate(method="linear", limit=3)
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

    def to_list(series):
        return [round(float(v), 2) if pd.notna(v) else None for v in series]

    return {
        "meta": {
            "latest_date": latest_date.strftime("%Y-%m-%d"),
            "current_season": current_season,
            "prev_season": prev_season,
            "seasons_5y_range": f"{seasons_5y[0]} → {seasons_5y[-1]}" if seasons_5y else "",
            "last_day_idx": last_idx,
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
            "current": to_list(current_series),
            "prev": to_list(prev_series),
            "mean_5y": to_list(mean_5y),
            "min_5y": to_list(min_5y),
            "max_5y": to_list(max_5y),
            "proj_median": proj_median,
            "proj_min": proj_min,
            "proj_max": proj_max,
        },
    }


HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="fr">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>Observatoire des réserves de gaz naturel</title>
<style>
  body {
    margin: 0;
    padding: 18px;
    background: #fafafa;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    color: #1f1f1f;
  }
  .obs-wrap {
    background: #ffffff;
    max-width: 1080px;
    margin: 0 auto;
    padding: 22px 26px 18px;
    border: 1px solid #ececec;
    border-radius: 10px;
    box-sizing: border-box;
  }
  .obs-header {
    display: flex;
    justify-content: space-between;
    align-items: flex-start;
    gap: 16px;
    flex-wrap: wrap;
    margin-bottom: 16px;
  }
  .obs-title {
    font-family: Georgia, "Times New Roman", serif;
    font-size: 32px;
    font-weight: 500;
    margin: 0 0 4px 0;
    line-height: 1.15;
    color: #1a1a1a;
  }
  .obs-subtitle { font-size: 14px; color: #555; margin: 0; }
  .obs-select {
    font-size: 14px;
    padding: 8px 12px;
    border-radius: 8px;
    border: 1px solid #d4d4d0;
    background: #f8f8f6;
    color: #1f1f1f;
    cursor: pointer;
    margin-top: 4px;
  }
  .obs-grid {
    display: grid;
    grid-template-columns: 255px 1fr;
    gap: 24px;
    align-items: start;
  }
  @media (max-width: 780px) {
    .obs-grid { grid-template-columns: 1fr; }
  }
  .obs-kpis { display: flex; flex-direction: column; gap: 12px; }
  .kpi-card { background: #f5f5f3; border-radius: 10px; padding: 14px 16px; }
  .kpi-label { font-size: 13px; color: #555; margin-bottom: 4px; }
  .kpi-value {
    font-family: Georgia, "Times New Roman", serif;
    font-size: 29px;
    font-weight: 500;
    line-height: 1.15;
    color: #1a1a1a;
  }
  .kpi-neg { color: #b92b27; }
  .kpi-pos { color: #1f7a4c; }
  .kpi-sub { font-size: 12px; color: #666; margin-top: 5px; line-height: 1.35; }
  .obs-chart-title { font-size: 15px; font-weight: 600; margin: 2px 0 6px 0; color: #1f1f1f; }
  .obs-footnote {
    font-size: 12px;
    color: #777;
    border-top: 1px solid #e8e8e8;
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
  <div class="obs-grid">
    <div class="obs-kpis" id="obs-kpis"></div>
    <div>
      <div class="obs-chart-title" id="obs-chart-title"></div>
      <div id="obs-chart" style="width:100%; height:440px;"></div>
    </div>
  </div>
  <div class="obs-footnote" id="obs-footnote"></div>
</div>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<script>
(function () {
  const DATA = __DATA_JSON__;
  const MONTHS_FR = ["janv.", "févr.", "mars", "avr.", "mai", "juin", "juil.", "août", "sept.", "oct.", "nov.", "déc."];

  function fmtFr(val, decimals = 1, signed = false) {
    if (val === null || val === undefined || Number.isNaN(val)) return "—";
    let prefix = "";
    if (signed) {
      if (val > 0) prefix = "+";
      else if (val < 0) prefix = "−";
    }
    const n = signed ? Math.abs(val) : val;
    return prefix + n.toLocaleString("fr-FR", {
      minimumFractionDigits: decimals,
      maximumFractionDigits: decimals
    });
  }

  function fmtDateFr(iso) {
    const [y, m, d] = iso.split("-").map(Number);
    return `${d} ${MONTHS_FR[m - 1]} ${y}`;
  }

  function render(perimeter) {
    const p = DATA.perimeters[perimeter];
    const meta = p.meta;
    const kpi = p.kpi;
    const s = p.series;
    const x = s.ref_dates;
    const lastIdx = meta.last_day_idx;

    const titleSuffix = perimeter === "UE" ? "en Europe" : `— ${perimeter}`;
    document.getElementById("obs-title").textContent = `Réserves de gaz naturel ${titleSuffix}`;
    document.getElementById("obs-subtitle").innerHTML =
      `Données au <b>${fmtDateFr(meta.latest_date)}</b> (fin de journée gazière) · Source : GIE AGSI+`;
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
      <div class="kpi-card">
        <div class="kpi-label">Écart à la moyenne 5 ans</div>
        <div class="kpi-value ${gapCls}">${arrow} ${fmtFr(kpi.gap_5y_pts, 1, true)} pts</div>
        <div class="kpi-sub">${gapTwhTxt}${fmtFr(kpi.mean_5y_pct, 1)} % (${meta.seasons_5y_range})</div>
      </div>
      <div class="kpi-card">
        <div class="kpi-label">Un an plus tôt</div>
        <div class="kpi-value">${fmtFr(kpi.prev_year_pct, 1)} %</div>
        <div class="kpi-sub">au même jour (${meta.prev_season || "N-1"})</div>
      </div>
      <div class="kpi-card">
        <div class="kpi-label">Projection au 31 mars</div>
        <div class="kpi-value">${projMain}</div>
        <div class="kpi-sub">${projSub}</div>
      </div>
    `;

    const traces = [
      { x, y: s.min_5y, mode: "lines", line: { width: 0 }, showlegend: false, hoverinfo: "skip" },
      { x, y: s.max_5y, mode: "lines", line: { width: 0 }, fill: "tonexty", fillcolor: "rgba(180,180,180,0.28)", name: "Min–max 5 ans", hoverinfo: "skip" },
      { x, y: s.proj_min, mode: "lines", line: { width: 0 }, showlegend: false, hoverinfo: "skip" },
      { x, y: s.proj_max, mode: "lines", line: { width: 0 }, fill: "tonexty", fillcolor: "rgba(26,115,232,0.14)", showlegend: false, hoverinfo: "skip" },
      { x, y: s.mean_5y, mode: "lines", line: { color: "#1f1f1f", width: 1.6 }, name: "Moy. 5 ans", hovertemplate: "Moy. 5 ans : %{y:.1f} %<extra></extra>" },
      { x, y: s.prev, mode: "lines", line: { color: "#d96b27", width: 2.0 }, name: meta.prev_season || "N-1", hovertemplate: `${meta.prev_season || "N-1"} : %{y:.1f} %<extra></extra>` },
      { x, y: s.current, mode: "lines", line: { color: "#1a73e8", width: 3.2 }, name: meta.current_season, hovertemplate: `${meta.current_season} : %{y:.1f} %<extra></extra>` },
      { x, y: s.proj_median, mode: "lines", line: { color: "#1a73e8", width: 2.2, dash: "dash" }, name: "Projection", hovertemplate: "Projection médiane : %{y:.1f} %<extra></extra>" },
      { x: [x[lastIdx]], y: [kpi.full_pct], mode: "markers", marker: { color: "#1a73e8", size: 10, line: { color: "#fff", width: 2 } }, showlegend: false, hoverinfo: "skip" }
    ];

    const annotations = [
      { x: x[lastIdx], y: kpi.full_pct, text: `<b>${fmtFr(kpi.full_pct, 1)} %</b>`, showarrow: false, xshift: -26, yshift: -20, font: { size: 13, color: "#1a1a1a" } },
      { x: "2025-11-01", y: 92, text: "Objectif UE 90 %", showarrow: false, font: { size: 11.5, color: "#333" } }
    ];

    if (kpi.proj_end_min_pct !== null) {
      annotations.push({
        x: "2026-03-31",
        y: Math.max(kpi.proj_end_min_pct, 3),
        text: `${fmtFr(kpi.proj_end_min_pct, 0)} % (profil ${kpi.proj_worst_season})`,
        showarrow: false, xanchor: "right", yshift: 10, font: { size: 11.5, color: "#555" }
      });
    }

    const layout = {
      margin: { l: 42, r: 16, t: 32, b: 30 },
      plot_bgcolor: "#ffffff",
      paper_bgcolor: "#ffffff",
      hovermode: "x unified",
      legend: { orientation: "h", yanchor: "bottom", y: 1.02, xanchor: "left", x: 0, font: { size: 12, color: "#444" } },
      shapes: [
        { type: "rect", xref: "x", yref: "paper", x0: "2025-10-01", x1: "2025-12-01", y0: 0, y1: 1, fillcolor: "rgba(0,0,0,0.045)", line: { width: 0 }, layer: "below" },
        { type: "line", xref: "x", yref: "y", x0: "2025-10-01", x1: "2025-12-01", y0: 90, y1: 90, line: { color: "#333", width: 1.4, dash: "dot" } }
      ],
      annotations,
      xaxis: {
        tickmode: "array",
        tickvals: ["2025-04-01", "2025-06-01", "2025-08-01", "2025-10-01", "2025-12-01", "2026-02-01", "2026-03-31"],
        ticktext: ["avr.", "juin", "août", "oct.", "déc.", "févr.", "avr."],
        showgrid: false, linecolor: "#ddd", tickfont: { color: "#555", size: 12 }
      },
      yaxis: { range: [0, 102], dtick: 20, ticksuffix: " %", gridcolor: "#efefef", zeroline: false, tickfont: { color: "#555", size: 12 } }
    };

    Plotly.newPlot("obs-chart", traces, layout, { displayModeBar: false, responsive: true });
  }

  const sel = document.getElementById("obs-select");
  const keys = Object.keys(DATA.perimeters);
  sel.innerHTML = keys.map(k => `<option value="${k}">${k}</option>`).join("");
  sel.addEventListener("change", () => render(sel.value));
  render(keys[0]);
})();
</script>
</body>
</html>
"""


def main():
    df, updated = update_cache()
    if df.empty:
        print("⚠️ Aucune donnée disponible.")
        return

    if not updated and os.path.exists(DASHBOARD_JSON) and os.path.exists(HTML_OUTPUT):
        print("✅ Données et page HTML déjà à jour.")
        return

    print("🧮 Calcul des indicateurs et génération de index.html...")
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
