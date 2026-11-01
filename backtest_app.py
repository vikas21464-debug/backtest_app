# ============================================================
# Lumpsum + SWP Backtester  (3-bucket planner)  |  Streamlit + Plotly
# - Pick 3 assets (mutual funds or NSE stocks) and an amount for each
# - Custom start / end date, one lumpsum on day 1
# - Optional monthly SWP with yearly step-up (step-up counted from the investment start date)
# - Optional refill of asset 1 from asset 2 whenever asset 1 is exhausted,
#   and from asset 3 as well when asset 2 is also exhausted
# - Optional valuation profit booking: gains move from asset 3 to asset 2 on Sell signals
#   (uses the same CMP / 124 DMA valuation signals as the Breakout & Valuation Analyzer)
# Data: Yahoo Finance (stocks, split/dividend adjusted) and mfapi.in (AMFI mutual-fund NAV)
# ============================================================
import html as _html
import logging
import warnings
from datetime import date, timedelta

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf

warnings.filterwarnings("ignore")
logging.getLogger("yfinance").setLevel(logging.CRITICAL)

st.set_page_config(page_title="Lumpsum & SWP Backtester", page_icon="💰", layout="wide")

# ------------------------------------------------------------
# Theme (Auto follows Streamlit's theme; can be overridden)
# ------------------------------------------------------------
LIGHT = dict(
    mode="light", bg="#ffffff", side="#f4f7fa", card="#ffffff", border="#e3e9ef",
    ink="#13293d", muted="#6b7c8f", up="#0f9d8a", down="#e0475b",
    hero1="#13293d", hero2="#1f4b63", hero_sub="#bcd0de", grid="#e8edf2",
    a=["#2e7d32", "#1565c0", "#6a3fb5"], line="#13293d", base="#6b7c8f", wd="#e65100",
    good_bg="#d9f3ee", good_fg="#0a6b5d", bad_bg="#fde1e5", bad_fg="#a3243a",
)
DARK = dict(
    mode="dark", bg="#0e1621", side="#131e2b", card="#162232", border="#263548",
    ink="#e6edf5", muted="#8fa3b8", up="#26c6a2", down="#ff6b7f",
    hero1="#0b2a3d", hero2="#14566e", hero_sub="#a9c4d6", grid="#22324a",
    a=["#5bd18a", "#6aa9ff", "#b79cff"], line="#e6edf5", base="#8fa3b8", wd="#ffb066",
    good_bg="#0f3d36", good_fg="#5eead4", bad_bg="#4a1a24", bad_fg="#ff9aa9",
)


def detect_theme():
    try:
        t = st.context.theme.type          # Streamlit >= 1.46
        if t in ("light", "dark"):
            return t
    except Exception:
        pass
    return st.get_option("theme.base") or "light"


with st.sidebar:
    theme_pref = st.radio("Appearance", ["Auto", "Light", "Dark"], horizontal=True,
                          help="Auto follows the Streamlit theme (menu ⋮ → Settings).")
mode_theme = detect_theme() if theme_pref == "Auto" else theme_pref.lower()
P = DARK if mode_theme == "dark" else LIGHT

force_bg = f"""
.stApp, [data-testid="stHeader"] {{ background: {P['bg']}; color: {P['ink']}; }}
[data-testid="stSidebar"] {{ background: {P['side']}; }}
.stApp p, .stApp label, .stApp span, .stApp li, .stApp [data-testid="stMarkdownContainer"] {{ color: {P['ink']}; }}
.stApp [data-testid="stCaptionContainer"], .stApp small {{ color: {P['muted']}; }}
""" if theme_pref != "Auto" else ""

st.markdown(f"""
<style>
{force_bg}
.block-container {{ padding-top: 1.6rem; max-width: 1400px; }}
h1, h2, h3 {{ color: {P['ink']}; letter-spacing: -0.01em; }}
.hero {{ background: linear-gradient(120deg, {P['hero1']} 0%, {P['hero2']} 100%); color: #fff;
        padding: 22px 28px; border-radius: 14px; margin-bottom: 18px; border: 1px solid {P['border']}; }}
.hero h1 {{ color: #fff; margin: 0; font-size: 1.7rem; }}
.hero p {{ margin: 4px 0 0; color: {P['hero_sub']}; font-size: .95rem; }}
.kpi {{ background: {P['card']}; border: 1px solid {P['border']}; border-radius: 12px; padding: 14px 16px; height: 100%; }}
.kpi .l {{ color: {P['muted']}; font-size: .78rem; }}
.kpi .v {{ color: {P['ink']}; font-size: 1.4rem; font-weight: 700; margin-top: 2px; }}
.kpi .s {{ font-size: .8rem; margin-top: 2px; color: {P['muted']}; }}
.pill {{ display:inline-block; padding: 3px 12px; border-radius: 999px; font-weight: 600; font-size: .9rem; }}
.good {{ background:{P['good_bg']}; color:{P['good_fg']}; }} .bad {{ background:{P['bad_bg']}; color:{P['bad_fg']}; }}
.slot {{ font-weight: 700; font-size: 1.05rem; color: {P['ink']}; }}
.role {{ color: {P['muted']}; font-size: .85rem; margin-bottom: 6px; }}
</style>
""", unsafe_allow_html=True)


# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------
def inr(x):
    """₹ in Indian units."""
    x = float(x)
    a = abs(x)
    if a >= 1e7:
        return f"₹{x / 1e7:,.2f} Cr"
    if a >= 1e5:
        return f"₹{x / 1e5:,.2f} L"
    return f"₹{x:,.0f}"


def short(name, n=40):
    return name if len(name) <= n else name[: n - 1] + "…"


def kpi(col, label, value, sub=""):
    col.markdown(f'<div class="kpi"><div class="l">{label}</div><div class="v">{value}</div>'
                 f'<div class="s">{sub}</div></div>', unsafe_allow_html=True)


def rgba(hexc, a):
    h = hexc.lstrip("#")
    return f"rgba({int(h[0:2], 16)},{int(h[2:4], 16)},{int(h[4:6], 16)},{a})"


# ------------------------------------------------------------
# Valuation signal (same rules as the Breakout & Valuation Analyzer)
# ------------------------------------------------------------
def get_valuation_signal(r):
    if r <= 0.55: return "5th BUY"
    if r <= 0.64: return "4th BUY"
    if r <= 0.73: return "3rd BUY"
    if r <= 0.82: return "2nd BUY"
    if r <= 0.91: return "1st BUY"
    if r >= 1.45: return "5th Sell"
    if r >= 1.36: return "4th Sell"
    if r >= 1.27: return "3rd Sell"
    if r >= 1.18: return "2nd Sell"
    if r >= 1.09: return "1st Sell"
    return "HOLD"


def first_day_signals(close, ratio):
    """Signal on the 1st trading day of each month, from the previous month's average CMP / 124 DMA."""
    sig = pd.Series(index=close.index, dtype=object)
    for (yr, mo), g in close.groupby([close.index.year, close.index.month]):
        yy, mm = (yr - 1, 12) if mo == 1 else (yr, mo - 1)
        pr = ratio[(ratio.index.year == yy) & (ratio.index.month == mm)].dropna()
        sig.loc[g.index[0]] = get_valuation_signal(float(pr.mean())) if len(pr) >= 5 else "HOLD"
    return sig.dropna()


def signal_at(sig, ts):
    """Latest monthly signal on or before ts."""
    pos = sig.index.searchsorted(ts, side="right") - 1
    return sig.iloc[pos] if pos >= 0 else "HOLD"


# ------------------------------------------------------------
# Data: mutual funds (mfapi.in) and stocks (Yahoo Finance)
# ------------------------------------------------------------
MF_API = "https://api.mfapi.in/mf"
_UA = {"User-Agent": "Mozilla/5.0"}


def is_growth_plan(name):
    n = name.lower()
    return "growth" in n and not any(w in n for w in ("idcw", "dividend", "bonus", "payout", "reinvest"))


def fund_rank(name):
    """Direct plans first, then shorter (cleaner) names."""
    return (0 if "direct" in name.lower() else 1, len(name))


@st.cache_data(ttl=3600, show_spinner=False)
def mf_search(q):
    import requests
    r = requests.get(f"{MF_API}/search", params={"q": q}, timeout=15, headers=_UA)
    r.raise_for_status()
    res = r.json()
    if not isinstance(res, list):
        raise ValueError("unexpected response")
    return [{"code": int(x["schemeCode"]), "name": str(x["schemeName"]).strip()}
            for x in res if x.get("schemeCode") and x.get("schemeName")]


def parse_nav(payload):
    """mfapi payload -> (NAV Series, meta)."""
    meta = payload.get("meta") or {}
    df = pd.DataFrame(payload.get("data") or [])
    if df.empty or not {"date", "nav"} <= set(df.columns):
        raise ValueError("no NAV data")
    df["date"] = pd.to_datetime(df["date"], format="%d-%m-%Y", errors="coerce")
    df["nav"] = pd.to_numeric(df["nav"], errors="coerce")
    df = df.dropna(subset=["date", "nav"])
    df = df[df["nav"] > 0].drop_duplicates("date").set_index("date").sort_index()
    if df.empty:
        raise ValueError("no NAV data")
    return df["nav"].astype(float), meta


@st.cache_data(ttl=3600, show_spinner=False)
def load_fund(code):
    import requests
    r = requests.get(f"{MF_API}/{int(code)}", timeout=20, headers=_UA)
    r.raise_for_status()
    return parse_nav(r.json())


@st.cache_data(ttl=3600, show_spinner=False)
def load_stock(ticker, start, end):
    """Split/dividend-adjusted close. Raises on empty data so failures are not cached."""
    df = yf.download(ticker, start=start, end=end, interval="1d", progress=False, auto_adjust=True)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    if df is None or df.empty:
        raise ValueError("no data")
    s = df["Close"].squeeze().dropna().astype(float).sort_index()
    if getattr(s.index, "tz", None) is not None:
        s.index = s.index.tz_localize(None)
    s = s[s > 0]
    if s.empty:
        raise ValueError("no data")
    return s


@st.cache_data(ttl=86400, show_spinner=False)
def stock_name(ticker):
    info = yf.Ticker(ticker).info or {}
    n = info.get("longName") or info.get("shortName")
    if not n:
        raise ValueError("no name")
    return n


def load_asset(spec, start, end):
    """-> (price Series incl. ~14 months of warm-up before start, display name)"""
    warm = pd.Timestamp(start - timedelta(days=420))
    last = pd.Timestamp(end)
    if spec["kind"] == "fund":
        s, meta = load_fund(spec["code"])
        name = meta.get("scheme_name") or spec["name"] or f"Scheme {spec['code']}"
    else:
        s = load_stock(spec["code"], warm.strftime("%Y-%m-%d"), (end + timedelta(days=1)).strftime("%Y-%m-%d"))
        try:
            name = stock_name(spec["code"])
        except Exception:
            name = spec["code"].replace(".NS", "")
    s = s.loc[(s.index >= warm) & (s.index <= last)]
    if s.empty:
        raise ValueError("no data in range")
    return s, name


# ------------------------------------------------------------
# Backtest engine
# ------------------------------------------------------------
def build_prices(series, start, end):
    """Align the 3 series on a common calendar. Returns (prices DataFrame, list of per-asset first dates)."""
    df = pd.concat(series, axis=1, keys=[0, 1, 2]).sort_index().ffill()
    firsts = [s.index[s.index >= pd.Timestamp(start)].min() if (s.index >= pd.Timestamp(start)).any() else pd.NaT
              for s in series]
    df = df.loc[pd.Timestamp(start):pd.Timestamp(end)].dropna()
    return df, firsts


def simulate(px, amounts, swp, delay, step_up, allow, mode, pb_on, pb_pct, sig3,
             refill_on=False, refill_months=36, refill3_on=False):
    """Day-1 lumpsum, then monthly events (profit booking, SWP). Everything executes at that day's price.

    px         DataFrame (n x 3) of prices on a common calendar
    amounts    rupees invested in each asset on day 1
    swp        monthly withdrawal in rupees at the investment START DATE (0 = none)
    delay      months after the start for the first withdrawal
    step_up    yearly increase (fraction), counted from the investment start date, so the
               first withdrawal after `delay` months is already inflated by (1+step_up)^(delay//12)
    allow      3 booleans: assets that may fund the SWP; mode: "sequential" or "proportional"
    pb_on      move gains of asset 3 into asset 2 when asset 3 shows a Sell signal (pb_pct: % of asset-3 value)
    refill_on  when asset 1 can no longer pay the month's SWP in full (exhausted), top it up from asset 2 so that it
               holds `refill_months` months of the current withdrawal (capped at what asset 2 has)
    refill3_on when asset 2 is also exhausted (it cannot cover the whole top-up), the remaining top-up is taken
               from asset 3 (capped at what asset 3 has). Only used together with refill_on.
    """
    idx = px.index
    n = len(idx)
    P = px.to_numpy(float)
    amounts = np.asarray(amounts, float)
    units = amounts / P[0]
    cost = amounts.copy()
    U = np.zeros((n, 3))
    wd = np.zeros(n)
    withdrawn_by = np.zeros(3)
    booked_total = 0.0
    refilled_total = 0.0
    refilled3_total = 0.0
    depleted = None
    ledger = []

    events, seen, k = [], set(), 0
    while True:
        anchor = idx[0] + pd.DateOffset(months=k)
        if anchor > idx[-1]:
            break
        pos = int(idx.searchsorted(anchor))
        if pos >= n:
            break
        if pos not in seen and (k > 0 or (swp > 0 and delay == 0)):
            events.append((pos, k))
            seen.add(pos)
        k += 1

    allow_arr = np.asarray(allow, bool)
    last = 0
    for pos, k in events:
        U[last:pos] = units
        last = pos
        p = P[pos]

        # 1) valuation profit booking: asset 3 -> asset 2 (capped at the unrealised gain)
        sig, booked, refilled, refilled3 = "", 0.0, 0.0, 0.0
        if pb_on and sig3 is not None and k > 0 and units[2] > 0:
            sig = signal_at(sig3, idx[pos])
            pct = pb_pct.get(sig, 0.0)
            if pct > 0:
                v3 = units[2] * p[2]
                amt = min(pct / 100.0 * v3, max(v3 - cost[2], 0.0))
                if amt > 0:
                    frac = amt / v3
                    cost[2] *= (1 - frac)
                    units[2] *= (1 - frac)
                    units[1] += amt / p[1]
                    cost[1] += amt
                    booked = amt
                    booked_total += amt

        # 2) SWP
        target, done = 0.0, np.zeros(3)
        if swp > 0 and k >= delay:
            # step-up is counted from the investment start date (month 0), not from the first withdrawal
            target = swp * (1.0 + step_up) ** (k // 12)

            # refill asset 1 when it cannot pay this month's SWP in full:
            # first from asset 2, then (if asset 2 is exhausted too and refill3_on) from asset 3
            if (refill_on and allow_arr[0] and units[0] * p[0] < target - 1.0
                    and (units[1] > 0 or (refill3_on and units[2] > 0))):
                need = max(refill_months * target - units[0] * p[0], 0.0)
                v2 = units[1] * p[1]
                from2 = min(need, v2)
                if from2 > 0:
                    frac = from2 / v2
                    cost[1] *= (1 - frac)
                    units[1] *= (1 - frac)
                    units[0] += from2 / p[0]
                    cost[0] += from2
                    refilled = from2
                    refilled_total += from2
                rest = need - from2
                v3 = units[2] * p[2]
                if refill3_on and rest > 1.0 and v3 > 0:
                    from3 = min(rest, v3)
                    frac = from3 / v3
                    cost[2] *= (1 - frac)
                    units[2] *= (1 - frac)
                    units[0] += from3 / p[0]
                    cost[0] += from3
                    refilled3 = from3
                    refilled3_total += from3

            vals = units * p
            avail = np.where(allow_arr, vals, 0.0)
            tot = avail.sum()
            if mode == "sequential":
                rem = target
                for j in range(3):
                    if rem <= 0:
                        break
                    take = min(rem, avail[j])
                    done[j] = take
                    rem -= take
            elif tot > 0:
                done = avail / tot * min(target, tot)
            for j in range(3):
                if done[j] > 0 and vals[j] > 0:
                    f = min(done[j] / vals[j], 1.0)
                    cost[j] *= (1 - f)
                    units[j] *= (1 - f)
            wd[pos] += done.sum()
            withdrawn_by += done
            if depleted is None and done.sum() < target - 1.0:
                depleted = idx[pos]

        if swp > 0 or booked > 0:
            ledger.append({"Date": idx[pos], "Month": k, "Signal (asset 3)": sig, "Profit booked 3→2": booked,
                           "Refill 2→1": refilled, "Refill 3→1": refilled3, "SWP target": target, "SWP paid": done.sum(),
                           "From asset 1": done[0], "From asset 2": done[1], "From asset 3": done[2],
                           "Portfolio after": float((units * p).sum())})
    U[last:] = units

    values = pd.DataFrame(U * P, index=idx, columns=[0, 1, 2])
    return {"values": values, "total": values.sum(axis=1), "withdrawn": pd.Series(wd, index=idx),
            "withdrawn_by": withdrawn_by, "booked_total": booked_total, "refilled_total": refilled_total,
            "refilled3_total": refilled3_total, "depleted": depleted,
            "ledger": pd.DataFrame(ledger), "units": units}


def xirr(dates, amts):
    t = np.array([(d - dates[0]).days / 365.0 for d in dates], float)
    a = np.asarray(amts, float)

    def f(r):
        return float(np.sum(a / (1.0 + r) ** t))

    lo, hi = -0.9999, 10.0
    if f(lo) * f(hi) > 0:
        return float("nan")
    for _ in range(200):
        mid = (lo + hi) / 2
        if f(lo) * f(mid) <= 0:
            hi = mid
        else:
            lo = mid
    return (lo + hi) / 2


def yearly_table(res, invested):
    idx, total, wd = res["total"].index, res["total"], res["withdrawn"]
    rows, y = [], 0
    while True:
        t0 = idx[0] + pd.DateOffset(years=y)
        if t0 >= idx[-1]:
            break
        t1 = min(idx[0] + pd.DateOffset(years=y + 1), idx[-1])
        i0 = 0 if y == 0 else int(idx.searchsorted(t0))
        i1 = int(idx.searchsorted(t1, side="right")) - 1
        opening = invested if y == 0 else float(total.iloc[min(i0, len(idx) - 1)])
        closing = float(total.iloc[i1])
        w = float(wd.iloc[i0 + (1 if y > 0 else 0): i1 + 1].sum()) if y > 0 else float(wd.iloc[: i1 + 1].sum())
        rows.append({"Year": f"Year {y + 1}" + ("" if t1 >= idx[0] + pd.DateOffset(years=y + 1) else " (part)"),
                     "From": idx[min(i0, len(idx) - 1)].strftime("%d-%m-%Y"), "To": idx[i1].strftime("%d-%m-%Y"),
                     "Opening value": opening, "Withdrawn": w, "Closing value": closing,
                     "Change incl. withdrawals %": ((closing + w) / opening - 1) * 100 if opening > 0 else np.nan})
        y += 1
    return pd.DataFrame(rows)


# ------------------------------------------------------------
# Charts
# ------------------------------------------------------------
def style_fig(fig, title, height=460, ytitle="₹ lakh"):
    fig.update_layout(
        height=height, template="plotly_dark" if P["mode"] == "dark" else "plotly_white",
        paper_bgcolor=P["bg"], plot_bgcolor=P["bg"], font=dict(color=P["ink"]), hovermode="x unified",
        margin=dict(l=10, r=10, t=60, b=95),
        title=dict(text=title, x=0, xanchor="left", y=0.97, yanchor="top", font=dict(size=16, color=P["ink"])),
        legend=dict(orientation="h", x=0, y=-0.14, xanchor="left", yanchor="top"))
    fig.update_xaxes(gridcolor=P["grid"])
    fig.update_yaxes(gridcolor=P["grid"], title=ytitle, tickformat=",.0f")
    return fig


def chart_value(res, base, names):
    L = 1e5
    fig = go.Figure()
    for j in range(3):
        fig.add_trace(go.Scatter(x=res["values"].index, y=res["values"][j] / L, name=short(names[j], 34), mode="lines",
                                 stackgroup="v", line=dict(width=0.8, color=P["a"][j]),
                                 fillcolor=rgba(P["a"][j], 0.55), hovertemplate="₹%{y:,.2f} L<extra>" + short(names[j], 26) + "</extra>"))
    cum = res["withdrawn"].cumsum() / L
    if cum.iloc[-1] > 0:
        fig.add_trace(go.Scatter(x=cum.index, y=cum, name="Withdrawn so far", mode="lines",
                                 line=dict(color=P["wd"], width=2.2), hovertemplate="₹%{y:,.2f} L<extra>Withdrawn so far</extra>"))
        fig.add_trace(go.Scatter(x=base["total"].index, y=base["total"] / L, name="Same plan without SWP",
                                 mode="lines", line=dict(color=P["base"], width=1.8, dash="dash"),
                                 hovertemplate="₹%{y:,.2f} L<extra>Without SWP</extra>"))
    return style_fig(fig, "Portfolio value by asset")


def chart_drawdown(base):
    dd = (base["total"] / base["total"].cummax() - 1) * 100
    fig = go.Figure(go.Scatter(x=dd.index, y=dd, mode="lines", name="Drawdown", fill="tozeroy",
                               line=dict(color=P["down"], width=1.2), fillcolor=rgba(P["down"], 0.25),
                               hovertemplate="%{y:.2f}%<extra>Drawdown</extra>"))
    style_fig(fig, "Drawdown of the portfolio (no withdrawals)", height=340, ytitle="% below previous peak")
    fig.update_yaxes(tickformat=".0f")
    return fig


def chart_rebased(px, names):
    fig = go.Figure()
    for j in range(3):
        fig.add_trace(go.Scatter(x=px.index, y=px[j] / px[j].iloc[0] * 100, name=short(names[j], 34), mode="lines",
                                 line=dict(color=P["a"][j], width=1.8),
                                 hovertemplate="%{y:,.1f}<extra>" + short(names[j], 26) + "</extra>"))
    return style_fig(fig, "Asset prices, rebased to 100 at the start", height=340, ytitle="Rebased price")


def chart_alloc(amounts, names):
    fig = go.Figure(go.Pie(labels=[short(n, 30) for n in names], values=amounts, hole=0.62, sort=False,
                           marker=dict(colors=P["a"], line=dict(color=P["bg"], width=2)),
                           textinfo="percent", hovertemplate="%{label}<br>₹%{value:,.0f}<extra></extra>"))
    fig.update_layout(height=300, margin=dict(l=0, r=0, t=10, b=10), paper_bgcolor="rgba(0,0,0,0)",
                      font=dict(color=P["ink"]), showlegend=False,
                      annotations=[dict(text=f"<b>{inr(sum(amounts))}</b>", x=0.5, y=0.5, showarrow=False,
                                        font=dict(size=18, color=P["ink"]))])
    return fig


# ------------------------------------------------------------
# Sidebar: backtest settings
# ------------------------------------------------------------
today = date.today()
with st.sidebar:
    st.header("Backtest settings")
    growth_only = st.checkbox("Growth plans only (funds)", value=True,
                              help="Hides IDCW / dividend plans, whose NAV drops on payouts.")
    st.subheader("Period")
    start_date = st.date_input("Start date (lumpsum day)", value=today - timedelta(days=5 * 365),
                               min_value=date(2000, 1, 1), max_value=today)
    end_date = st.date_input("End date", value=today, min_value=date(2000, 1, 1), max_value=today)

    st.subheader("Monthly SWP")
    swp_amt = st.number_input("Monthly withdrawal (₹, in start-date money)", min_value=0, value=75000, step=5000,
                              help="Amount you need per month at the investment start date. It rises every 12 months "
                                   "from that date, even before withdrawals begin. Set 0 for a plain lumpsum.")
    swp_delay = st.number_input("First withdrawal after (months)", min_value=0, max_value=360, value=1, step=1,
                                help="Withdrawals begin this many months after the start date, already adjusted "
                                     "for inflation since the start.")
    step_pct = st.number_input("Yearly increase in withdrawal (%)", min_value=0.0, max_value=25.0, value=6.0, step=0.5,
                               help="Inflation step-up, applied every 12 months from the investment start date.")
    mode_label = st.radio("Withdraw from", ["In order: asset 1, then 2, then 3", "Proportional to value"],
                          help="'In order' empties asset 1 first, then asset 2, then asset 3, like the planner's "
                               "'use Bucket 1 and 2 first'.")
    allow = [st.checkbox(f"Asset {j + 1} can fund the SWP", value=True, key=f"allow{j}") for j in range(3)]

    st.subheader("Bucket refill")
    refill_on = st.checkbox("Refill asset 1 from asset 2 when asset 1 is exhausted", value=False,
                            help="When asset 1 cannot pay the month's withdrawal in full, units of asset 2 are sold "
                                 "and moved into asset 1, before that month's SWP is paid. Works best with "
                                 "'In order' withdrawals.")
    refill_months = st.number_input("Refill asset 1 up to (months of withdrawals)", min_value=1, max_value=120,
                                    value=36, step=1, disabled=not refill_on,
                                    help="Asset 1 is topped up to this many months of the current monthly withdrawal "
                                         "(36 = 3 years). Limited to what asset 2 holds.")

    refill3_on = st.checkbox("Also refill asset 1 from asset 3 when asset 2 is exhausted", value=False,
                             disabled=not refill_on,
                             help="If asset 2 cannot cover the whole top-up, the rest is sold from asset 3 and moved "
                                  "into asset 1. Needs the asset 2 refill above to be switched on.")

    st.subheader("Valuation profit booking")
    pb_on = st.checkbox("Move gains from asset 3 to asset 2 on Sell signals", value=False,
                        help="Each month, if asset 3's valuation signal (previous month's average CMP / 124 DMA) is a "
                             "Sell level, part of asset 3 is sold and put into asset 2. Never more than the gain.")
    pb_pct = {lvl: st.number_input(f"{lvl}: sell % of asset 3", min_value=0.0, max_value=100.0, value=v, step=1.0,
                                   disabled=not pb_on, key=f"pb{lvl}")
              for lvl, v in (("2nd Sell", 5.0), ("3rd Sell", 10.0), ("4th Sell", 20.0), ("5th Sell", 30.0))}
    st.caption("1st Sell, HOLD and BUY signals do nothing. The percentages are illustrative; change them freely.")

    st.subheader("Display")
    infl_pct = st.number_input("Inflation for 'today's money' (%)", min_value=0.0, max_value=20.0, value=6.0, step=0.5)
    run = st.button("Run backtest", type="primary", use_container_width=True)

# ------------------------------------------------------------
# Header + portfolio setup
# ------------------------------------------------------------
st.markdown('<div class="hero"><h1>Lumpsum & SWP Backtester</h1>'
            '<p>Invest once in up to three funds or stocks, then test monthly withdrawals over any period · '
            'built for the 3-bucket plan developed by Vikas Dhiman</p></div>', unsafe_allow_html=True)

SLOTS = [
    ("Asset 1 · Income bucket", "Safety and regular withdrawals", "aditya birla sun life money manager", 52.5),
    ("Asset 2 · Buffer bucket", "Volatility cushion and flexibility", "parag parikh conservative hybrid", 52.5),
    ("Asset 3 · Growth bucket", "Long-term growth and inflation protection", "icici prudential multi asset", 105.0),
]


def slot_ui(i, title, role, default_q, default_amt):
    spec = {"ok": False, "kind": "fund", "code": None, "name": "", "amount": 0.0}
    with st.container(border=True):
        st.markdown(f'<div class="slot">{title}</div><div class="role">{role}</div>', unsafe_allow_html=True)
        spec["kind"] = "fund" if st.radio("Asset type", ["Mutual fund", "Stock"], horizontal=True,
                                          key=f"kind{i}") == "Mutual fund" else "stock"
        if spec["kind"] == "fund":
            q = st.text_input("Search fund", value=default_q, key=f"q{i}", help="Type part of the scheme name.")
            code_in = st.text_input("Or AMFI scheme code", key=f"code{i}", placeholder="e.g. 122639")
            if code_in.strip().isdigit():
                spec.update(code=int(code_in.strip()), ok=True)
            elif len(q.strip()) >= 3:
                failed = False
                try:
                    found = mf_search(q.strip())
                except Exception:
                    found, failed = [], True
                    st.error("Fund search is unavailable right now. Try again, or enter a scheme code.")
                if growth_only:
                    found = [x for x in found if is_growth_plan(x["name"])] or found
                found = sorted(found, key=lambda x: fund_rank(x["name"]))[:100]
                if found:
                    names = {x["code"]: x["name"] for x in found}
                    code = st.selectbox("Fund", list(names), format_func=lambda c, names=names: names[c], key=f"fund{i}")
                    spec.update(code=code, name=names[code], ok=True)
                elif not failed:
                    st.caption("No funds found. Try fewer or different words.")
            else:
                st.caption("Type at least 3 letters of the fund name.")
        else:
            sym = st.text_input("NSE stock symbol", key=f"sym{i}", placeholder="e.g. RELIANCE",
                                help="Type the NSE symbol. Add .BO for a BSE listing.").strip().upper()
            if sym:
                if not sym.startswith("^") and "." not in sym:
                    sym += ".NS"
                spec.update(code=sym, name=sym.replace(".NS", ""), ok=True)
        spec["amount"] = st.number_input("Investment (₹ lakh)", min_value=0.0, value=default_amt, step=1.0,
                                         key=f"amt{i}") * 1e5
    return spec


cols = st.columns(3)
specs = []
for i, (c, (title, role, q, amt)) in enumerate(zip(cols, SLOTS)):
    with c:
        specs.append(slot_ui(i, title, role, q, amt))

total_in = sum(s["amount"] for s in specs)
if total_in > 0:
    pcts = " · ".join(f"{s['amount'] / total_in * 100:.0f}%" for s in specs)
    st.caption(f"Total lumpsum **{inr(total_in)}** · split {pcts}"
               + (f" · first-year SWP is **{swp_amt * 12 / total_in * 100:.2f}%** of the lumpsum" if swp_amt else ""))

if "ran" not in st.session_state:
    st.session_state.ran = False
if run:
    st.session_state.ran = True
if not st.session_state.ran:
    st.info("Choose three assets and amounts above, set the period and SWP in the sidebar, then select **Run backtest**.")
    st.stop()

# ------------------------------------------------------------
# Validate + load
# ------------------------------------------------------------
if end_date <= start_date + timedelta(days=90):
    st.error("Pick an end date at least three months after the start date.")
    st.stop()
for (title, *_), sp in zip(SLOTS, specs):
    if not sp["ok"]:
        st.error(f"{title}: choose a fund or enter a stock symbol.")
        st.stop()
if total_in <= 0:
    st.error("Enter an investment amount for at least one asset.")
    st.stop()
if len({(s["kind"], s["code"]) for s in specs}) < 3:
    st.warning("Two or more slots hold the same asset. The backtest still runs, but the split is not diversified.")

series, names = [], []
with st.spinner("Loading price history…"):
    for (title, *_), sp in zip(SLOTS, specs):
        try:
            s, nm = load_asset(sp, start_date, end_date)
        except Exception:
            st.error(f"{title}: could not load price history for {sp['name'] or sp['code']}. "
                     "Check the symbol or scheme code and try again.")
            st.stop()
        series.append(s)
        names.append(nm)

px, firsts = build_prices(series, start_date, end_date)
if len(px) < 60:
    st.error("Not enough overlapping price history for these three assets in the chosen period. "
             "Move the start date later, or pick assets with a longer history.")
    st.stop()
if (px.index[0] - pd.Timestamp(start_date)).days > 7:
    late = [short(names[j], 40) for j in range(3) if pd.notna(firsts[j]) and (firsts[j] - pd.Timestamp(start_date)).days > 7]
    st.info(f"The backtest starts on **{px.index[0]:%d-%m-%Y}**, the first day all three assets have prices"
            + (f" (history of {', '.join(late)} begins later)." if late else "."))
if (pd.Timestamp(end_date) - px.index[-1]).days > 7:
    st.info(f"Latest common price date is **{px.index[-1]:%d-%m-%Y}**.")

# valuation signals for asset 3 (uses its own history, incl. warm-up before the start)
s3 = series[2]
sig3 = first_day_signals(s3, s3 / s3.rolling(124).mean()) if len(s3) > 150 else None
if pb_on and sig3 is None:
    st.warning("Asset 3 does not have enough history for valuation signals, so profit booking is skipped.")

amounts = [s["amount"] for s in specs]
mode = "sequential" if mode_label.startswith("In order") else "proportional"
common = dict(delay=int(swp_delay), step_up=step_pct / 100.0, allow=allow, mode=mode,
              pb_on=pb_on, pb_pct=pb_pct, sig3=sig3, refill_on=refill_on, refill_months=int(refill_months),
              refill3_on=bool(refill_on and refill3_on))
res = simulate(px, amounts, float(swp_amt), **common)
base = simulate(px, amounts, 0.0, **common)          # same plan, no withdrawals

# ------------------------------------------------------------
# Results
# ------------------------------------------------------------
idx = px.index
years = (idx[-1] - idx[0]).days / 365.25
invested = float(sum(amounts))
final = float(res["total"].iloc[-1])
withdrawn = float(res["withdrawn"].sum())
base_final = float(base["total"].iloc[-1])
cagr_base = (base_final / invested) ** (1 / years) - 1 if years > 0 and base_final > 0 else float("nan")
dts = [idx[0]] + [d for d, w in res["withdrawn"].items() if w > 0] + [idx[-1]]
cfs = [-invested] + [w for w in res["withdrawn"] if w > 0] + [final]
irr = xirr(dts, cfs)
maxdd = float(((base["total"] / base["total"].cummax()) - 1).min() * 100)
real_final = final / (1 + infl_pct / 100) ** years
swp_on = swp_amt > 0

# first withdrawal amount after the step-up counted from the investment start date
first_wd = swp_amt * (1 + step_pct / 100) ** (int(swp_delay) // 12)

k = st.columns(4)
kpi(k[0], "Invested (lumpsum)", inr(invested), f"{idx[0]:%d %b %Y} to {idx[-1]:%d %b %Y} · {years:.1f} years")
kpi(k[1], "Portfolio value at end", inr(final), f"≈ {inr(real_final)} in today's money" if final > 0 else "Fully withdrawn")
kpi(k[2], "Total withdrawn", inr(withdrawn) if swp_on else "—",
    f"First: {inr(first_wd)}/month (month {int(swp_delay)}), +{step_pct:g}% a year" if swp_on else "SWP is off")
kpi(k[3], "Wealth created", inr(final + withdrawn - invested),
    f"{(final + withdrawn) / invested:.2f}× the amount invested")
st.write("")
k = st.columns(4)
kpi(k[0], "Return with SWP (XIRR)", f"{irr * 100:.2f}% a year" if not np.isnan(irr) else "n/a",
    "Counts every withdrawal and the final value")
kpi(k[1], "Lumpsum return (no SWP)", f"{cagr_base * 100:.2f}% a year" if not np.isnan(cagr_base) else "n/a",
    f"Ends at {inr(base_final)}")
if not swp_on:
    kpi(k[2], "Did the money last?", "No SWP", "Set a monthly withdrawal to test this")
elif res["depleted"] is not None:
    kpi(k[2], "Did the money last?", '<span class="pill bad">Ran short</span>',
        f"First shortfall on {res['depleted']:%d-%m-%Y}")
else:
    kpi(k[2], "Did the money last?", '<span class="pill good">Yes</span>', f"Withdrawals paid in full for {years:.1f} years")
kpi(k[3], "Worst fall (no SWP)", f"{maxdd:.1f}%", "Largest drop from a peak")
st.write("")

tab1, tab2, tab3, tab4 = st.tabs(["Charts", "Assets", "Year by year", "Withdrawal log"])

with tab1:
    cfg = {"displaylogo": False, "scrollZoom": True}
    st.plotly_chart(chart_value(res, base, names), use_container_width=True, config=cfg, key="c_value")
    a, b = st.columns(2)
    a.plotly_chart(chart_drawdown(base), use_container_width=True, config=cfg, key="c_dd")
    b.plotly_chart(chart_rebased(px, names), use_container_width=True, config=cfg, key="c_reb")
    st.caption("Drag to pan, scroll to zoom, double-click to reset. Values are in ₹ lakh (100 lakh = ₹1 crore).")

with tab2:
    a, b = st.columns([1, 2.2])
    with a:
        st.plotly_chart(chart_alloc(amounts, names), use_container_width=True, config={"displayModeBar": False}, key="c_alloc")
    with b:
        rows = []
        for j in range(3):
            p0, p1 = float(px[j].iloc[0]), float(px[j].iloc[-1])
            rows.append({"Asset": names[j], "Type": "Fund" if specs[j]["kind"] == "fund" else "Stock",
                         "Invested": amounts[j], "Price at start": p0, "Price at end": p1,
                         "Price CAGR %": ((p1 / p0) ** (1 / years) - 1) * 100 if years > 0 else np.nan,
                         "Withdrawn from it": res["withdrawn_by"][j],
                         "Value at end": float(res["values"][j].iloc[-1])})
        t = pd.DataFrame(rows)
        st.dataframe(t, hide_index=True, use_container_width=True, column_config={
            "Invested": st.column_config.NumberColumn(format="₹%.0f"),
            "Price at start": st.column_config.NumberColumn(format="%.2f"),
            "Price at end": st.column_config.NumberColumn(format="%.2f"),
            "Price CAGR %": st.column_config.NumberColumn(format="%.2f"),
            "Withdrawn from it": st.column_config.NumberColumn(format="₹%.0f"),
            "Value at end": st.column_config.NumberColumn(format="₹%.0f")})
        if pb_on:
            st.caption(f"Profit booking moved **{inr(res['booked_total'])}** from asset 3 into asset 2 over the period.")
        if refill_on:
            st.caption(f"Refills moved **{inr(res['refilled_total'])}** from asset 2 into asset 1 over the period"
                       + (f" and **{inr(res['refilled3_total'])}** from asset 3." if refill3_on else "."))
        st.caption("Price CAGR ignores withdrawals and profit booking. Stock prices are adjusted for splits and dividends; "
                   "mutual fund NAVs are growth-plan NAVs.")

with tab3:
    yt = yearly_table(res, invested)
    st.dataframe(yt, hide_index=True, use_container_width=True, column_config={
        "Opening value": st.column_config.NumberColumn(format="₹%.0f"),
        "Withdrawn": st.column_config.NumberColumn(format="₹%.0f"),
        "Closing value": st.column_config.NumberColumn(format="₹%.0f"),
        "Change incl. withdrawals %": st.column_config.NumberColumn(format="%.2f")})
    st.caption("Years run from the start date. 'Change incl. withdrawals' adds back what was withdrawn that year.")
    st.download_button("Download yearly table (CSV)", yt.to_csv(index=False).encode(), "yearly_summary.csv", "text/csv")

with tab4:
    lg = res["ledger"]
    if lg.empty:
        st.info("No withdrawals or profit booking in this run. Set a monthly withdrawal in the sidebar.")
    else:
        lg = lg.copy()
        lg["Date"] = lg["Date"].dt.strftime("%d-%m-%Y")
        st.dataframe(lg, hide_index=True, use_container_width=True, column_config={
            c: st.column_config.NumberColumn(format="₹%.0f") for c in
            ("Profit booked 3→2", "Refill 2→1", "Refill 3→1", "SWP target", "SWP paid", "From asset 1", "From asset 2", "From asset 3", "Portfolio after")})
        st.caption("Every event runs at that day's closing price, on the first trading day on or after the monthly date.")
        st.download_button("Download withdrawal log (CSV)", lg.to_csv(index=False).encode(), "withdrawal_log.csv", "text/csv")

st.caption("Past performance does not predict future returns. This backtest ignores taxes, exit loads and "
           "transaction costs, and assumes you can trade at the day's closing price. Data: Yahoo Finance and mfapi.in (AMFI). "
           "For research only, not investment advice.")
