"""
Stock Comparison & Analysis Application
Financial Data Analytics I — Project
"""

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from scipy import stats
from datetime import date, timedelta

# ─────────────────────────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Stock Comparison & Analysis",
    page_icon="📈",
    layout="wide",
)

# ─────────────────────────────────────────────────────────────────────────────
# Cached data download
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data(ttl=3600, show_spinner=False)
def download_prices(tickers_tuple: tuple, start: str, end: str):
    """
    Download adjusted close prices for user tickers + ^GSPC.
    Returns (prices_df, warnings_list, error_str).
    error_str is None on success.
    """
    all_tickers = list(tickers_tuple) + ["^GSPC"]

    try:
        raw = yf.download(
            all_tickers,
            start=start,
            end=end,
            auto_adjust=True,
            progress=False,
        )
    except Exception as exc:
        return None, [], f"Download error: {exc}"

    if raw is None or raw.empty:
        return None, [], "No data returned. Check ticker symbols and date range."

    # Extract the Close column (works with both multi-index and flat structure)
    try:
        if isinstance(raw.columns, pd.MultiIndex):
            prices = raw["Close"].copy()
        else:
            prices = raw[["Close"]].copy()
            prices.columns = all_tickers[:1]
    except KeyError:
        return None, [], "Could not extract price data. Verify your ticker symbols."

    # Check which user tickers have usable data
    failed = [
        t for t in tickers_tuple
        if t not in prices.columns or prices[t].isna().all()
    ]
    if failed:
        return None, [], (
            f"Could not retrieve data for: {', '.join(failed)}. "
            "Check the ticker symbols and try again."
        )

    # Drop tickers with >5% missing values
    warnings = []
    valid_tickers = list(tickers_tuple)
    for t in list(valid_tickers):
        miss_pct = prices[t].isna().mean()
        if miss_pct > 0.05:
            valid_tickers.remove(t)
            warnings.append(
                f"⚠️ {t} removed: {miss_pct:.1%} missing values in the selected range."
            )

    if len(valid_tickers) < 2:
        return None, warnings, (
            "After removing tickers with insufficient data, fewer than 2 stocks remain. "
            "Please choose different tickers or a wider date range."
        )

    # Keep only valid tickers + benchmark
    keep_cols = valid_tickers + (["^GSPC"] if "^GSPC" in prices.columns else [])
    prices = prices[keep_cols]

    # Truncate to the common overlapping date range
    first_valid = prices.apply(lambda c: c.first_valid_index()).max()
    last_valid  = prices.apply(lambda c: c.last_valid_index()).min()

    if first_valid is None or last_valid is None or first_valid >= last_valid:
        return None, warnings, "No overlapping data for the selected tickers and date range."

    prices = prices.loc[first_valid:last_valid].dropna()

    if len(prices) < 50:
        return None, warnings, (
            "Fewer than 50 trading days of overlapping data. "
            "Please widen the date range."
        )

    # Inform user if the range was truncated
    if pd.Timestamp(first_valid).date() > date.fromisoformat(start) or \
       pd.Timestamp(last_valid).date() < date.fromisoformat(end) - timedelta(days=30):
        warnings.append(
            f"ℹ️ Data truncated to overlapping range: "
            f"{pd.Timestamp(first_valid).date()} – {pd.Timestamp(last_valid).date()}."
        )

    return prices, warnings, None


# ─────────────────────────────────────────────────────────────────────────────
# Financial calculation helpers
# ─────────────────────────────────────────────────────────────────────────────

def compute_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Simple (arithmetic) daily returns."""
    return prices.pct_change().dropna()


def ann_return(r: pd.Series) -> float:
    """Annualized mean return (252 trading days)."""
    return float(r.mean() * 252)


def ann_vol(r: pd.Series) -> float:
    """Annualized volatility (252 trading days)."""
    return float(r.std() * np.sqrt(252))


def build_summary(returns: pd.DataFrame) -> pd.DataFrame:
    """Return a formatted summary statistics table."""
    rows = {}
    for col in returns.columns:
        r = returns[col]
        rows[col] = {
            "Ann. Return":      ann_return(r),
            "Ann. Volatility":  ann_vol(r),
            "Skewness":         float(r.skew()),
            "Kurtosis":         float(r.kurt()),
            "Min Daily Return": float(r.min()),
            "Max Daily Return": float(r.max()),
        }
    df = pd.DataFrame(rows).T
    pct_cols = ["Ann. Return", "Ann. Volatility", "Min Daily Return", "Max Daily Return"]
    for c in pct_cols:
        df[c] = df[c].map("{:.2%}".format)
    df["Skewness"] = df["Skewness"].map("{:.3f}".format)
    df["Kurtosis"] = df["Kurtosis"].map("{:.3f}".format)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar — user inputs
# ─────────────────────────────────────────────────────────────────────────────

st.sidebar.title("📈 Stock Analysis App")
st.sidebar.markdown("---")

ticker_raw = st.sidebar.text_input(
    "Ticker Symbols (2–5, comma-separated)",
    value="AAPL, MSFT, GOOGL, AMZN",
    help="Enter between 2 and 5 stock ticker symbols, e.g.  AAPL, MSFT, GOOGL",
)

today = date.today()

start_date = st.sidebar.date_input(
    "Start Date",
    value=today - timedelta(days=5 * 365),
    max_value=today - timedelta(days=366),
)
end_date = st.sidebar.date_input(
    "End Date",
    value=today,
    max_value=today,
)

load_btn = st.sidebar.button("🔄 Load Data", use_container_width=True, type="primary")

st.sidebar.markdown("---")
with st.sidebar.expander("ℹ️ About / Methodology"):
    st.markdown("""
**What this app does**

Compare and analyze 2–5 stocks using historical adjusted prices from Yahoo Finance.

**Key Assumptions**
- **Returns**: Simple (arithmetic) daily % change
- **Annualization**: 252 trading days/year
- **Annualized return** = mean daily return × 252
- **Annualized volatility** = daily σ × √252
- **Wealth index**: $(1+r)$.cumprod() × $10,000
- **Equal-weight portfolio**: average daily return across all selected stocks each day

**Data Source**
Yahoo Finance via the `yfinance` library. Adjusted close prices account for dividends and stock splits.

**Statistical Tests**
- **Jarque-Bera**: tests whether return distribution is normal. p < 0.05 rejects normality at the 5% level.
- **Q-Q Plot**: deviations from the red diagonal indicate departures from normality (e.g., fat tails).
    """)

# ─────────────────────────────────────────────────────────────────────────────
# Input validation
# ─────────────────────────────────────────────────────────────────────────────

tickers_input = [t.strip().upper() for t in ticker_raw.split(",") if t.strip()]
input_error = None

if len(tickers_input) < 2:
    input_error = "Please enter **at least 2** ticker symbols."
elif len(tickers_input) > 5:
    input_error = "Please enter **no more than 5** ticker symbols."
elif start_date is None or end_date is None:
    input_error = "Please select both a start and end date."
elif (end_date - start_date).days < 365:
    input_error = "Date range must be **at least 1 year**."
# ─────────────────────────────────────────────────────────────────────────────
# Session state
# ─────────────────────────────────────────────────────────────────────────────

if "data_loaded" not in st.session_state:
    st.session_state.data_loaded = False
    st.session_state.prices = None
    st.session_state.user_tickers = []

if load_btn:
    if input_error:
        st.sidebar.error(input_error)
    else:
        with st.spinner("Downloading data from Yahoo Finance…"):
            prices_dl, warnings_dl, err_dl = download_prices(
                tuple(tickers_input), str(start_date), str(end_date)
            )
        if err_dl:
            st.error(err_dl)
            st.session_state.data_loaded = False
        else:
            for w in warnings_dl:
                st.warning(w)
            st.session_state.prices = prices_dl
            st.session_state.user_tickers = [c for c in prices_dl.columns if c != "^GSPC"]
            st.session_state.data_loaded = True
            st.success(
                f"Loaded data for: {', '.join(st.session_state.user_tickers)} "
                f"({len(prices_dl)} trading days)"
            )

# ─────────────────────────────────────────────────────────────────────────────
# Landing screen (shown before data is loaded)
# ─────────────────────────────────────────────────────────────────────────────

if not st.session_state.data_loaded:
    st.title("📊 Stock Comparison & Analysis")
    st.markdown(
        "Use the **sidebar** to enter 2–5 ticker symbols and a date range, "
        "then click **Load Data** to begin."
    )
    if input_error:
        st.error(input_error)
    st.info(
        "**Example tickers to try:** AAPL, MSFT, GOOGL, AMZN, TSLA  \n"
        "**Index / ETF examples:** SPY, QQQ, ^GSPC"
    )
    st.stop()

# ─────────────────────────────────────────────────────────────────────────────
# Core data (available after successful load)
# ─────────────────────────────────────────────────────────────────────────────

prices: pd.DataFrame         = st.session_state.prices
user_tickers: list[str]      = st.session_state.user_tickers

returns      = compute_returns(prices)
user_ret     = returns[user_tickers]
sp500_ret    = returns["^GSPC"] if "^GSPC" in returns.columns else None

# ─────────────────────────────────────────────────────────────────────────────
# Tab layout
# ─────────────────────────────────────────────────────────────────────────────

tab1, tab2, tab3 = st.tabs([
    "📈  Price & Returns",
    "📊  Risk & Distribution",
    "🔗  Correlation & Diversification",
])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — PRICE & RETURNS
# ══════════════════════════════════════════════════════════════════════════════

with tab1:
    st.header("Price & Return Analysis")

    # ── Adjusted price chart ──────────────────────────────────────────────────
    st.subheader("Adjusted Closing Prices")
    price_display = st.multiselect(
        "Select stocks to display",
        options=user_tickers,
        default=user_tickers,
        key="price_multiselect",
    )

    if not price_display:
        st.warning("Select at least one stock above to display the price chart.")
    else:
        fig_price = go.Figure()
        for t in price_display:
            fig_price.add_trace(go.Scatter(
                x=prices.index, y=prices[t], mode="lines", name=t
            ))
        fig_price.update_layout(
            title="Adjusted Closing Prices",
            xaxis_title="Date",
            yaxis_title="Price (USD)",
            hovermode="x unified",
            legend_title="Ticker",
        )
        st.plotly_chart(fig_price, use_container_width=True)

    # ── Summary statistics table ──────────────────────────────────────────────
    st.subheader("Summary Statistics (Annualized)")
    all_for_stats = user_ret.copy()
    if sp500_ret is not None:
        all_for_stats = pd.concat([all_for_stats, sp500_ret.rename("S&P 500")], axis=1)
    st.dataframe(build_summary(all_for_stats), use_container_width=True)

    # ── Cumulative wealth index ───────────────────────────────────────────────
    st.subheader("Cumulative Wealth Index — Growth of $10,000")
    wealth = (1 + user_ret).cumprod() * 10_000
    if sp500_ret is not None:
        wealth["S&P 500"] = (1 + sp500_ret).cumprod() * 10_000
    eq_weight_ret = user_ret.mean(axis=1)
    wealth["Equal-Weight Portfolio"] = (1 + eq_weight_ret).cumprod() * 10_000

    fig_wealth = go.Figure()
    for col in wealth.columns:
        fig_wealth.add_trace(go.Scatter(
            x=wealth.index, y=wealth[col], mode="lines", name=col
        ))
    fig_wealth.update_layout(
        title="Growth of $10,000 Investment",
        xaxis_title="Date",
        yaxis_title="Portfolio Value (USD)",
        hovermode="x unified",
        legend_title="Asset",
    )
    st.plotly_chart(fig_wealth, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — RISK & DISTRIBUTION
# ══════════════════════════════════════════════════════════════════════════════

with tab2:
    st.header("Risk & Distribution Analysis")

    # ── Rolling volatility ────────────────────────────────────────────────────
    st.subheader("Rolling Annualized Volatility")
    roll_win = st.select_slider(
        "Rolling window (trading days)",
        options=[30, 60, 90],
        value=60,
        key="roll_vol_window",
    )
    roll_vol = user_ret.rolling(roll_win).std() * np.sqrt(252)

    fig_rv = go.Figure()
    for t in user_tickers:
        fig_rv.add_trace(go.Scatter(
            x=roll_vol.index, y=roll_vol[t], mode="lines", name=t
        ))
    fig_rv.update_layout(
        title=f"Rolling {roll_win}-Day Annualized Volatility",
        xaxis_title="Date",
        yaxis_title="Annualized Volatility",
        hovermode="x unified",
        legend_title="Ticker",
    )
    st.plotly_chart(fig_rv, use_container_width=True)

    st.markdown("---")

    # ── Distribution analysis (histogram + Q-Q + JB test) ────────────────────
    st.subheader("Return Distribution Analysis")
    dist_ticker = st.selectbox(
        "Select stock for distribution analysis",
        options=user_tickers,
        key="dist_ticker_select",
    )
    r_dist = user_ret[dist_ticker].dropna()

    # Jarque-Bera normality test
    jb_stat, jb_p = stats.jarque_bera(r_dist)
    m1, m2, m3 = st.columns(3)
    m1.metric("Jarque-Bera Statistic", f"{jb_stat:,.2f}")
    m2.metric("p-value", f"{jb_p:.4f}")
    if jb_p < 0.05:
        m3.error("Rejects normality (p < 0.05)")
    else:
        m3.success("Fails to reject normality (p ≥ 0.05)")

    # Toggle between histogram and Q-Q plot
    hist_tab, qq_tab = st.tabs(["Histogram + Normal Fit", "Q-Q Plot"])

    with hist_tab:
        mu_fit, sigma_fit = stats.norm.fit(r_dist)
        x_fit = np.linspace(r_dist.min(), r_dist.max(), 300)
        pdf_fit = stats.norm.pdf(x_fit, mu_fit, sigma_fit)

        fig_hist = go.Figure()
        fig_hist.add_trace(go.Histogram(
            x=r_dist,
            histnorm="probability density",
            nbinsx=80,
            name="Daily Returns",
            opacity=0.65,
            marker_color="steelblue",
        ))
        fig_hist.add_trace(go.Scatter(
            x=x_fit, y=pdf_fit,
            mode="lines", name="Fitted Normal",
            line=dict(color="red", width=2),
        ))
        fig_hist.update_layout(
            title=f"{dist_ticker} — Return Histogram vs. Fitted Normal Distribution",
            xaxis_title="Daily Return",
            yaxis_title="Probability Density",
            legend_title="Series",
        )
        st.plotly_chart(fig_hist, use_container_width=True)

    with qq_tab:
        (osm, osr), (slope_qq, intercept_qq, _) = stats.probplot(r_dist)
        qq_x_line = np.array([osm.min(), osm.max()])
        qq_y_line = slope_qq * qq_x_line + intercept_qq

        fig_qq = go.Figure()
        fig_qq.add_trace(go.Scatter(
            x=osm, y=osr,
            mode="markers", name="Sample Quantiles",
            marker=dict(size=3, color="steelblue", opacity=0.6),
        ))
        fig_qq.add_trace(go.Scatter(
            x=qq_x_line, y=qq_y_line,
            mode="lines", name="Normal Reference Line",
            line=dict(color="red", width=2),
        ))
        fig_qq.update_layout(
            title=f"{dist_ticker} — Q-Q Plot vs. Normal Distribution",
            xaxis_title="Theoretical Quantiles",
            yaxis_title="Sample Quantiles",
        )
        st.plotly_chart(fig_qq, use_container_width=True)
        st.caption(
            "Points lying on the red diagonal indicate normality. "
            "Deviations at the tails (S-shape) indicate fat tails — common in stock returns."
        )

    st.markdown("---")

    # ── Box plot ──────────────────────────────────────────────────────────────
    st.subheader("Daily Return Distribution Comparison")
    fig_box = go.Figure()
    for t in user_tickers:
        fig_box.add_trace(go.Box(
            y=user_ret[t], name=t,
            boxpoints="outliers",
            marker_size=2,
        ))
    fig_box.update_layout(
        title="Side-by-Side Box Plot of Daily Returns",
        xaxis_title="Ticker",
        yaxis_title="Daily Return",
        showlegend=False,
    )
    st.plotly_chart(fig_box, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — CORRELATION & DIVERSIFICATION
# ══════════════════════════════════════════════════════════════════════════════

with tab3:
    st.header("Correlation & Diversification")

    # ── Correlation heatmap ───────────────────────────────────────────────────
    st.subheader("Pairwise Correlation Heatmap")
    corr_matrix = user_ret.corr()

    fig_hm = px.imshow(
        corr_matrix,
        text_auto=".2f",
        color_continuous_scale="RdBu_r",
        zmin=-1,
        zmax=1,
        title="Pairwise Correlation of Daily Returns",
        aspect="auto",
    )
    fig_hm.update_layout(
        xaxis_title="Ticker",
        yaxis_title="Ticker",
        coloraxis_colorbar_title="ρ",
    )
    st.plotly_chart(fig_hm, use_container_width=True)

    st.markdown("---")

    # ── Return scatter plot ───────────────────────────────────────────────────
    st.subheader("Return Scatter Plot")
    col1, col2 = st.columns(2)
    sc_a = col1.selectbox("Stock A", user_tickers, index=0, key="sc_a")
    sc_b = col2.selectbox("Stock B", user_tickers, index=min(1, len(user_tickers) - 1), key="sc_b")

    if sc_a == sc_b:
        st.warning("Select two **different** stocks for the scatter plot.")
    else:
        sc_df = pd.DataFrame({"x": user_ret[sc_a], "y": user_ret[sc_b]}).dropna()
        slope_sc, intercept_sc, r_sc, _, _ = stats.linregress(sc_df["x"], sc_df["y"])
        x_trend = np.linspace(sc_df["x"].min(), sc_df["x"].max(), 200)
        y_trend = slope_sc * x_trend + intercept_sc

        fig_sc = go.Figure()
        fig_sc.add_trace(go.Scatter(
            x=sc_df["x"], y=sc_df["y"],
            mode="markers", name="Daily Returns",
            marker=dict(size=4, opacity=0.4, color="steelblue"),
        ))
        fig_sc.add_trace(go.Scatter(
            x=x_trend, y=y_trend,
            mode="lines", name=f"OLS Trend  (R² = {r_sc**2:.3f})",
            line=dict(color="red", width=2),
        ))
        fig_sc.update_layout(
            title=f"{sc_a} vs. {sc_b} — Daily Returns",
            xaxis_title=f"{sc_a} Daily Return",
            yaxis_title=f"{sc_b} Daily Return",
        )
        st.plotly_chart(fig_sc, use_container_width=True)

    st.markdown("---")

    # ── Rolling correlation ───────────────────────────────────────────────────
    st.subheader("Rolling Correlation")
    col3, col4 = st.columns(2)
    rc_a = col3.selectbox("Stock A", user_tickers, index=0, key="rc_a")
    rc_b = col4.selectbox("Stock B", user_tickers, index=min(1, len(user_tickers) - 1), key="rc_b")
    rc_win = st.select_slider(
        "Rolling window (trading days)",
        options=[30, 60, 90, 120],
        value=60,
        key="rc_win",
    )

    if rc_a == rc_b:
        st.warning("Select two **different** stocks for rolling correlation.")
    else:
        roll_corr = user_ret[rc_a].rolling(rc_win).corr(user_ret[rc_b])
        fig_rc = go.Figure()
        fig_rc.add_trace(go.Scatter(
            x=roll_corr.index, y=roll_corr,
            mode="lines", name=f"{rc_a} / {rc_b}",
            line=dict(color="steelblue", width=1.5),
        ))
        fig_rc.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
        fig_rc.update_layout(
            title=f"Rolling {rc_win}-Day Correlation: {rc_a} vs. {rc_b}",
            xaxis_title="Date",
            yaxis_title="Rolling Correlation",
            yaxis_range=[-1, 1],
        )
        st.plotly_chart(fig_rc, use_container_width=True)

    st.markdown("---")

    # ── Two-asset portfolio explorer ──────────────────────────────────────────
    st.subheader("Two-Asset Portfolio Explorer")
    st.info(
        "**Diversification in action:** Combining two stocks can produce a portfolio with "
        "*lower* volatility than either stock alone. This is the **diversification effect** — "
        "visible when the volatility curve dips below the dotted lines representing each stock's "
        "standalone volatility. The dip is more pronounced when the correlation between the two "
        "stocks is lower. Adjust the weight slider to explore how allocation shifts risk and return."
    )

    col5, col6 = st.columns(2)
    pa = col5.selectbox("Stock A", user_tickers, index=0, key="pa")
    pb = col6.selectbox("Stock B", user_tickers, index=min(1, len(user_tickers) - 1), key="pb")

    if pa == pb:
        st.warning("Select two **different** stocks for the portfolio explorer.")
    else:
        w_pct = st.slider(
            f"Weight on {pa} (%)  — remainder goes to {pb}",
            min_value=0, max_value=100, value=50, step=1,
            key="port_weight",
        )
        w = w_pct / 100.0

        # Annualized stats for each stock
        ra_ann = ann_return(user_ret[pa])
        rb_ann = ann_return(user_ret[pb])
        sa_ann = ann_vol(user_ret[pa])
        sb_ann = ann_vol(user_ret[pb])

        # Annualized covariance matrix
        cov_ann = user_ret[[pa, pb]].cov().values * 252
        cov_ab  = float(cov_ann[0, 1])

        # Current portfolio metrics
        p_ret = w * ra_ann + (1 - w) * rb_ann
        p_var = (w**2 * sa_ann**2
                 + (1 - w)**2 * sb_ann**2
                 + 2 * w * (1 - w) * cov_ab)
        p_vol = float(np.sqrt(max(p_var, 0.0)))

        m1, m2, m3, m4 = st.columns(4)
        m1.metric(f"Weight: {pa}", f"{w:.0%}")
        m2.metric(f"Weight: {pb}", f"{1 - w:.0%}")
        m3.metric("Portfolio Ann. Return", f"{p_ret:.2%}")
        m4.metric("Portfolio Ann. Volatility", f"{p_vol:.2%}")

        # Volatility curve across all weight allocations (0% → 100%)
        ws = np.linspace(0, 1, 201)
        vols_curve = np.sqrt(np.maximum(
            ws**2 * sa_ann**2
            + (1 - ws)**2 * sb_ann**2
            + 2 * ws * (1 - ws) * cov_ab,
            0.0,
        ))

        fig_port = go.Figure()
        fig_port.add_trace(go.Scatter(
            x=ws * 100, y=vols_curve,
            mode="lines", name="Portfolio Volatility",
            line=dict(color="steelblue", width=2.5),
        ))
        # Current slider position
        fig_port.add_trace(go.Scatter(
            x=[w * 100], y=[p_vol],
            mode="markers", name="Current Allocation",
            marker=dict(color="red", size=14, symbol="circle"),
        ))
        # Individual stock volatility reference lines
        fig_port.add_hline(
            y=sa_ann, line_dash="dot", line_color="orange",
            annotation_text=f"{pa} alone ({sa_ann:.2%})",
            annotation_position="top right",
        )
        fig_port.add_hline(
            y=sb_ann, line_dash="dot", line_color="green",
            annotation_text=f"{pb} alone ({sb_ann:.2%})",
            annotation_position="bottom right",
        )
        fig_port.update_layout(
            title=f"Portfolio Volatility Curve: {pa} + {pb}",
            xaxis_title=f"Weight on {pa} (%)",
            yaxis_title="Annualized Volatility",
            legend_title="",
            hovermode="x unified",
        )
        st.plotly_chart(fig_port, use_container_width=True)

        corr_ab = float(user_ret[pa].corr(user_ret[pb]))
        st.caption(
            f"Correlation between {pa} and {pb}: **{corr_ab:.3f}**  |  "
            f"{pa} volatility: **{sa_ann:.2%}**  |  "
            f"{pb} volatility: **{sb_ann:.2%}**"
        )
