import asyncio
import httpx
import json
import logging
import yfinance as yf
from urllib.parse import quote
import feedparser
from langchain_core.tools import tool
import pandas as pd
from typing import List, Dict, Optional, Any
from datetime import datetime, timezone

ALLOWED_URLS = {
    "bbc_world": "http://feeds.bbci.co.uk/news/world/rss.xml",
    "bbc_business": "http://feeds.bbci.co.uk/news/business/rss.xml",
    "fed_press": "https://www.federalreserve.gov/feeds/press_all.xml",
    "ecb_news": "https://www.ecb.europa.eu/rss/html/index.en.xml",
    "investing_global": "https://www.investing.com/rss/news.rss",
}

logger = logging.getLogger("etf_rag_bot")


@tool("get_ticker_metadata", description="Get ticker symbol metadata from Yahoo Finance (tries base -> .DE -> ^index).")
async def get_ticker_metadata(symbols: List[str]) -> str:
    """
    Get metadata for one or more ticker symbols.
    Tries in order:
      - base symbol
      - base + ".DE"
      - "^" + base (index format)
    Also tries the original input first if it contains '.' or '^'.

    Args:
        symbols: List of ticker symbols (e.g., ["VWCE", "SXRS.DE", "^GSPC"])

    Returns:
        JSON string with ticker metadata
    """
    tickers = [s.strip().upper() for s in symbols if s and s.strip()]
    out: Dict[str, Any] = {}

    async def retrieve(symbol: str) -> Dict[str, Any]:
        def _fetch():
            try:
                meta = yf.Ticker(symbol).info

                if not meta or len(meta) <= 1:
                    return {"symbol": symbol, "error": "TICKER_NOT_FOUND"}

                name = meta.get("longName") or meta.get("shortName") or meta.get("name")
                if not name:
                    return {"symbol": symbol, "error": "NAME_NOT_FOUND"}

                return {
                    "symbol": symbol,
                    "name": name,
                    "currency": meta.get("currency"),
                    "exchange": meta.get("exchange"),
                    "marketPrice": meta.get("regularMarketPrice"),
                    "marketCap": meta.get("marketCap"),
                    "sector": meta.get("sector"),
                    "industry": meta.get("industry"),
                }
            except Exception as e:
                return {"symbol": symbol, "error": f"ERROR: {str(e)}"}

        return await asyncio.to_thread(_fetch)

    async def fetch_one(requested: str) -> Dict[str, Any]:
        req = requested.strip().upper()
        base = req.lstrip("^").split(".")[0]

        candidates: List[str] = []

        # Try exactly what user passed first if it's already specific
        if "." in req or req.startswith("^"):
            candidates.append(req)

        # Then requested fallback chain: base -> .DE -> ^
        candidates.extend([base, f"{base}.DE", f"^{base}"])

        # Deduplicate while preserving order
        seen = set()
        candidates = [c for c in candidates if not (c in seen or seen.add(c))]

        last_meta: Dict[str, Any] = {"symbol": req, "error": "TICKER_NOT_FOUND"}

        for cand in candidates:
            meta = await retrieve(cand)
            last_meta = meta
            if "error" not in meta:
                meta["requested"] = req
                meta["resolved_symbol"] = cand
                return meta

        # all failed
        last_meta["requested"] = req
        last_meta["resolved_symbol"] = None
        return last_meta

    results = await asyncio.gather(*(fetch_one(t) for t in tickers))

    # Keep output keys consistent with user inputs
    for r in results:
        out[r.get("requested", r.get("symbol", "UNKNOWN"))] = r

    return json.dumps(out, ensure_ascii=False, indent=2)


@tool(
    "fetch_rss_news",
    description="Fetch and parse RSS feed items from a predefined feed key (e.g., ecb_news). Returns JSON."
)
def fetch_rss_news(url_name: str, limit: int = 10) -> str:
    """
    Fetches and parses RSS feed data from a predefined feed key in ALLOWED_URLS.

    Args:
        url_name: One of the keys in ALLOWED_URLS, e.g.:
            - bbc_world
            - bbc_business
            - fed_press
            - ecb_news
            - investing_global
        limit: Max number of items to return.

    Returns:
        JSON string with a list of items:
        [{"title":..., "link":..., "published":..., "summary":..., "source": url_name}, ...]
    """
    if url_name not in ALLOWED_URLS:
        raise ValueError(
            f"Feed key not allowed: {url_name}. Allowed: {list(ALLOWED_URLS.keys())}"
        )

    feed_url = ALLOWED_URLS[url_name]

    # Fetch RSS XML explicitly
    headers = {"User-Agent": "Mozilla/5.0 (compatible; RAGBot/1.0)"}
    try:
        r = httpx.get(feed_url, headers=headers, timeout=60.0, follow_redirects=True)
        r.raise_for_status()
        feed = feedparser.parse(r.text)
    except Exception:
        # Fallback: let feedparser fetch directly
        feed = feedparser.parse(feed_url)

    items = []
    for entry in getattr(feed, "entries", [])[: max(1, int(limit))]:
        items.append({
            "source": url_name,
            "feed_url": feed_url,
            "title": getattr(entry, "title", "") or "",
            "link": getattr(entry, "link", "") or "",
            "published": getattr(entry, "published", "") or getattr(entry, "updated", "") or "",
            "summary": (getattr(entry, "summary", "") or getattr(entry, "description", "") or "")[:500],
        })

    return json.dumps(items, ensure_ascii=False)


async def _fetch_stooq_daily(
    symbol: str,
    start_date: str = "20200101",
    end_date: Optional[str] = None,
) -> Optional[pd.DataFrame]:
    """
    Fetch daily OHLC from Stooq with fallback order:
    1) SYMBOL.US
    2) SYMBOL.DE
    3) ^SYMBOL
    4) SYMBOL

    Uses Stooq date params:
      &f=YYYYMMDD&t=YYYYMMDD&i=d

    Returns:
        pd.DataFrame if found, else None
    """
    symbol = (symbol or "").strip()
    if not symbol:
        return None

    start_date = (start_date or "").strip()
    if not start_date:
        start_date = "20200101"

    end_date = (end_date or "").strip()
    if not end_date:
        end_date = datetime.now(timezone.utc).strftime("%Y%m%d")

    # Normalize input into a base root (remove ^ and any suffix)
    base = symbol.upper()
    if base.startswith("^"):
        base = base[1:]
    root = base.split(".")[0]

    candidates = [
        f"{root}.US",
        f"{root}.DE",
        f"^{root}",
        root,
    ]

    # Deduplicate while preserving order
    seen = set()
    candidates = [c for c in candidates if not (c in seen or seen.add(c))]

    def _read_csv(url: str) -> pd.DataFrame:
        return pd.read_csv(url)

    for candidate in candidates:
        try:
            sym_q = quote(candidate.lower(), safe="^.")
            url = f"https://stooq.com/q/d/l/?s={sym_q}&f={start_date}&t={end_date}&i=d"

            df = await asyncio.to_thread(_read_csv, url)
            if df is None or df.empty:
                continue

            df.columns = [c.lower() for c in df.columns]
            if "date" not in df.columns or "close" not in df.columns:
                continue

            df["date"] = pd.to_datetime(df["date"], errors="coerce")
            df = df.dropna(subset=["date"]).sort_values("date")

            if df.empty:
                continue

            logger.debug("Stooq success: requested=%s -> used=%s rows=%d", symbol, candidate, len(df))
            return df.reset_index(drop=True)

        except Exception as e:
            logger.debug("Stooq failed: requested=%s tried=%s err=%s", symbol, candidate, repr(e))
            continue

    logger.debug("Stooq no data for requested=%s", symbol)
    return None


def _max_drawdown(equity: pd.Series) -> float:
    """
    Calculates the maximum drawdown of an equity curve. Informational purposes only.

    Args:
        equity (pd.Series): A pandas Series representing the equity curve.

    Returns:
        float: The maximum drawdown as a decimal.
    """
    roll_max = equity.cummax()
    drawdown = (equity - roll_max) / roll_max
    max_drawdown = drawdown.min()
    return float(max_drawdown)


@tool("fetch_prices", description="Fetches the latest prices for a list of symbols from Stooq. For indexes, symbol has a '^' prepended, e.g. ^QQQ")
async def fetch_prices(symbols: List[str]) -> Dict[str, Optional[float]]:
    """
    Fetches the latest prices for a list of symbols from Stooq.
    For indexes, '^' is prepended (e.g. ^SPY).

    Args:
        symbols (List[str]): A list of stock symbols to fetch prices for.

    Returns:
        latest ticker prices (Dict[str, float]): A dictionary mapping symbols to their latest prices.
    """
    prices = {}
    start_date = (datetime.now() - pd.Timedelta(days=3)).strftime("%Y%m%d")
    for symbol in symbols:
        df = await _fetch_stooq_daily(symbol, start_date=start_date)
        if not df.empty:
            latest_price = df['close'].iloc[-1]
            prices[symbol] = float(latest_price)
        else:
            prices[symbol] = None

    return prices


@tool("etf_risk_report", description="Generates a risk report for a given ETF symbol.")
async def etf_risk_report(symbol: str, rf_annual: float = 0.0) -> str:
    """
    Generates a risk report for a given ETF symbol.

    Args:
        symbol (str): The ETF symbol to analyze.
        rf_annual (float): The annual risk-free rate as a decimal.

    Returns:
        str: A JSON string containing the risk report.
    """
    df = await _fetch_stooq_daily(symbol)
    df['returns'] = df['close'].pct_change().dropna()

    if df['returns'].empty:
        return json.dumps({"error": "Not enough data to calculate returns."})

    avg_daily_return = df['returns'].mean()
    daily_volatility = df['returns'].std()

    trading_days = 252
    annualized_return = (1 + avg_daily_return) ** trading_days - 1
    annualized_volatility = daily_volatility * (trading_days ** 0.5)

    sharpe_ratio = (annualized_return - rf_annual) / annualized_volatility if annualized_volatility != 0 else None
    equity = (1 + df['returns']).cumprod()
    max_drawdown = _max_drawdown(equity)

    report = {
        "symbol": symbol,
        "annualized_return": annualized_return,
        "annualized_volatility": annualized_volatility,
        "sharpe_ratio": sharpe_ratio,
        "max_drawdown": max_drawdown,
        "disclaimer": "This report is for informational purposes only and does not include financial advice."
    }

    return json.dumps(report)


@tool("rebalance_plan", description="Generates a rebalance plan based on target weights.")
def rebalance_plan(
    holdings_value: Dict[str, float],
    prices: Dict[str, float],
    target_weights: Dict[str, float],
    new_cash: float = 0.0,
    allow_sells: bool = False,
) -> str:
    """
    Computes a simple rebalance plan based on target weights.

    Args:
        holdings_value: current position value per ticker (in same currency)
        prices: latest price per ticker
        target_weights: desired weights that sum ~1.0
        new_cash: additional cash to deploy
        allow_sells: if False, outputs buys only (DCA-friendly)

    Returns:
        str: A JSON string containing the rebalance plan.
    """

    w_sum = sum(target_weights.values())
    if w_sum <= 0:
        return json.dumps({"error": "target_weights sum must be > 0"})
    weights = {k: v / w_sum for k, v in target_weights.items()}

    # Current portfolio value
    current_total = sum(holdings_value.get(t, 0.0) for t in weights.keys())
    total_after = current_total + float(new_cash)

    trades = []
    for t, w in weights.items():
        price = float(prices.get(t, 0.0))
        if price <= 0:
            trades.append({"ticker": t, "error": "missing/invalid price"})
            continue

        current = float(holdings_value.get(t, 0.0))
        target = total_after * w
        delta_value = target - current

        if (not allow_sells) and (delta_value < 0):
            delta_value = 0.0  # buy-only mode

        shares = delta_value / price
        if abs(shares) > 1e-8:
            trades.append({
                "ticker": t,
                "buy_value": round(delta_value, 2) if delta_value > 0 else 0.0,
                "sell_value": round(-delta_value, 2) if delta_value < 0 else 0.0,
                "shares": float(shares),
                "price": price
            })

    return json.dumps({
        "total_current_value": round(current_total, 2),
        "new_cash": round(float(new_cash), 2),
        "total_after": round(total_after, 2),
        "allow_sells": allow_sells,
        "trades": trades,
        "disclaimer": "Informational only; not financial advice."
    })
