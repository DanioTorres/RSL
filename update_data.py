import io, json, datetime
import pandas as pd, requests, yfinance as yf

SEC = {"Health Care": "Healthcare", "Information Technology": "Technology"}

def sp500():
    r = requests.get("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
                     headers={"User-Agent": "Mozilla/5.0 rsl-dashboard"}, timeout=30)
    r.raise_for_status()
    t = pd.read_html(io.StringIO(r.text))[0]
    return {s.replace(".", "-"): SEC.get(g, g) for s, g in zip(t["Symbol"], t["GICS Sector"])}

def main():
    sectors = sp500()
    tickers = list(sectors)
    frames = []
    for i in range(0, len(tickers), 100):
        chunk = tickers[i:i + 100]
        df = yf.download(chunk, period="3y", interval="1wk", auto_adjust=False,
                         progress=False, threads=True)["Close"]
        if isinstance(df, pd.Series):
            df = df.to_frame(chunk[0])
        frames.append(df)
    px = pd.concat(frames, axis=1)
    rows = []
    for tk in tickers:
        if tk not in px:
            continue
        s = px[tk].dropna()
        if len(s) < 27:
            continue
        rows.append({"tk": tk, "sec": sectors[tk], "cl": round(float(s.iloc[-1]), 2),
                     "sma": round(float(s.tail(26).mean()), 2), "mk": "S&P 500"})
    if len(rows) < 300:
        raise SystemExit(f"Nur {len(rows)} Aktien geladen - data.json wird nicht ueberschrieben")
    out = {"updated": datetime.date.today().strftime("%d.%m.%Y"), "rows": rows}
    with open("data.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False)
    print(len(rows), "Aktien gespeichert")

main()
