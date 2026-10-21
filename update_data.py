import io, json, os, datetime
import pandas as pd, requests, yfinance as yf

TAGE = 40  # so viele Handelstage (Stichtage) bleiben abrufbar
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
        df = yf.download(chunk, period="2y", interval="1d", auto_adjust=False,
                         progress=False, threads=True)["Close"]
        if isinstance(df, pd.Series):
            df = df.to_frame(chunk[0])
        frames.append(df)
    px = pd.concat(frames, axis=1)
    px.index = pd.to_datetime(px.index).tz_localize(None)

    # Laeuft die US-Boerse vielleicht noch (Mo-Fr vor 21:15 UTC)? Dann den heutigen Tag weglassen.
    now = datetime.datetime.utcnow()
    if now.weekday() < 5 and now.hour * 60 + now.minute < 21 * 60 + 15:
        px = px[px.index < pd.Timestamp(now.date())]
    # Nur echte Handelstage (an Feiertagen fehlen fast alle Kurse)
    px = px[px.notna().sum(axis=1) >= 300]
    days = list(px.index[-TAGE:])

    os.makedirs("data", exist_ok=True)
    for old in os.listdir("data"):
        os.remove(os.path.join("data", old))
    names = []
    for d in days:
        sub = px[px.index <= d]
        weekly = sub.resample("W-FRI").last().tail(26)  # Wochenschluesse, letzte Woche = Kurs am Stichtag
        sma = weekly.mean()
        ok = (weekly.notna().sum() == 26) & sub.iloc[-1].notna()
        close = sub.iloc[-1]
        rows = [{"tk": tk, "sec": sectors[tk], "cl": round(float(close[tk]), 2),
                 "sma": round(float(sma[tk]), 2), "mk": "S&P 500"}
                for tk in tickers if tk in ok.index and ok[tk]]
        if len(rows) < 300:
            print("Zu wenige Aktien am", d.date(), "- uebersprungen")
            continue
        name = d.strftime("%Y-%m-%d")
        with open(f"data/{name}.json", "w", encoding="utf-8") as f:
            json.dump({"updated": "Handelsschluss " + d.strftime("%d.%m.%Y"), "rows": rows}, f, ensure_ascii=False)
        names.append(name)
    if not names:
        raise SystemExit("Keine Daten erzeugt")
    with open("data/index.json", "w", encoding="utf-8") as f:
        json.dump({"dates": names[::-1]}, f)
    print(len(names), "Stichtage gespeichert, neuester:", names[-1])

main()
