"""
config/universe.py

The fixed pond we fish in. A curated set of liquid US large/mid caps,
grouped by sector. Curated-and-static is a deliberate choice at this
stage: a fixed universe means every downstream number (relative volume,
% move) is comparable day over day, and there's no risk of the scanner
"discovering" some illiquid name whose spread would eat any profit.

No logic lives here — just data. The scanner imports ALL_TICKERS.
"""

UNIVERSE: dict[str, list[str]] = {
    "tech": [
        "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "AMD", "INTC",
        "TSM", "AVGO", "QCOM", "MU", "AMAT", "LRCX", "KLAC", "ARM",
        "CRM", "ORCL", "ADBE", "NOW", "SNOW", "PLTR", "DDOG", "NET",
        "CRWD", "PANW", "ZS", "FTNT", "MDB", "TEAM",
        "UBER", "ABNB", "SHOP", "SQ", "PYPL", "COIN", "HOOD",
        "TSLA", "RIVN", "LCID",
    ],
    "biotech": [
        "AMGN", "GILD", "VRTX", "REGN", "BIIB", "MRNA", "BNTX", "NVAX",
        "ALNY", "SRPT", "BMRN", "INCY", "EXEL", "NBIX", "IONS", "RARE",
        "BEAM", "CRSP", "NTLA", "EDIT", "VKTX", "SMMT",
    ],
    "pharma": [
        "JNJ", "PFE", "MRK", "ABBV", "LLY", "BMY", "AZN", "NVS",
        "NVO", "SNY", "GSK", "TAK", "ZTS", "VTRS", "TEVA", "CTLT",
    ],
    "infrastructure": [
        "CAT", "DE", "URI", "PWR", "EME", "MTZ", "ACM", "J",
        "VMC", "MLM", "NUE", "STLD", "X", "CLF",
        "ETN", "EMR", "PH", "ROK", "GNRC",
    ],
    "energy": [
        "XOM", "CVX", "COP", "SLB", "HAL", "OXY", "DVN", "FANG",
        "EOG", "MPC", "VLO", "PSX", "KMI", "WMB",
        "FSLR", "ENPH", "RUN", "NEE",
    ],
    "financials": [
        "JPM", "BAC", "WFC", "C", "GS", "MS", "SCHW", "BLK",
        "V", "MA", "AXP", "DFS", "SOFI", "ALLY",
    ],
    "consumer": [
        "WMT", "COST", "TGT", "HD", "LOW", "MCD", "SBUX", "CMG",
        "NKE", "LULU", "DIS", "NFLX", "ROKU", "DKNG",
    ],
}

# Flat, de-duplicated view for batch API calls. Order is stable
# (dict insertion order) so scans are reproducible.
ALL_TICKERS: list[str] = list(dict.fromkeys(
    ticker for sector in UNIVERSE.values() for ticker in sector
))

# Reverse lookup: which sector does a ticker belong to (first match wins).
SECTOR_OF: dict[str, str] = {
    ticker: sector
    for sector, tickers in reversed(UNIVERSE.items())
    for ticker in tickers
}

if __name__ == "__main__":
    for sector, tickers in UNIVERSE.items():
        print(f"{sector:16s} {len(tickers):3d}")
    print(f"{'TOTAL':16s} {len(ALL_TICKERS):3d}")
