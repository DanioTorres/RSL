import io, json, os, datetime
import numpy as np
import pandas as pd, requests, yfinance as yf

TAGE = 40  # so viele Handelstage (Stichtage) bleiben abrufbar
SEC = {"Health Care": "Healthcare", "Information Technology": "Technology"}

def sp500():
    r = requests.get("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
                     headers={"User-Agent": "Mozilla/5.0 rsl-dashboard"}, timeout=30)
    r.raise_for_status()
    t = pd.read_html(io.StringIO(r.text))[0]
    return {s.replace(".", "-"): (SEC.get(g, g), str(n)) for s, g, n in zip(t["Symbol"], t["GICS Sector"], t["Security"])}

def takeover_flags(ret, d):
    """Verdacht auf Uebernahme: Kurssprung >= 10 % an einem Tag (letzte 120 Tage),
    seitdem (mind. 5 Tage) ungewoehnlich ruhig (Std. der Tagesrenditen < 0,4 %)."""
    w = ret.loc[:d].tail(120)
    arr = w.to_numpy(dtype=float)
    out = {}
    filled = np.where(np.isnan(arr), -1e9, arr)
    for j in np.where(filled.max(axis=0) >= 10)[0]:
        col = filled[:, j]
        k = int(np.argmax(col))
        after = arr[k + 1:, j][-20:]
        after = after[~np.isnan(after)]
        if len(after) >= 5 and np.std(after) < 0.4:
            out[w.columns[j]] = [round(float(col[k]), 1), w.index[k].strftime("%Y-%m-%d"), round(float(np.std(after)), 2)]
    return out

def main():
    info = sp500()
    tickers = list(info)
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
    ret = px.pct_change(fill_method=None) * 100
    for d in days:
        sub = px[px.index <= d]
        weekly = sub.resample("W-FRI").last().tail(26)  # Wochenschluesse, letzte Woche = Kurs am Stichtag
        sma = weekly.mean()
        ok = (weekly.notna().sum() == 26) & sub.iloc[-1].notna()
        last26 = sub.tail(26)                            # zweite Basis: 26 Handelstage
        sma26 = last26.mean()
        ok26 = last26.notna().sum() == 26
        filled = sub.ffill()
        flags = takeover_flags(ret, d)
        close = filled.iloc[-1]
        perf = {}
        for key, n in (("d1", 1), ("w1", 5), ("m1", 21)):
            perf[key] = (close / filled.iloc[-1 - n] - 1) * 100 if len(filled) > n else None
        rows = []
        for tk in tickers:
            if tk not in ok.index or not ok[tk]:
                continue
            row = {"tk": tk, "n": info[tk][1], "sec": info[tk][0], "cl": round(float(close[tk]), 2),
                   "sma": round(float(sma[tk]), 2), "mk": "S&P 500"}
            if tk in flags:
                row["ue"] = flags[tk]
            if ok26[tk]:
                row["s26"] = round(float(sma26[tk]), 2)
            for key, ser in perf.items():
                if ser is not None and pd.notna(ser[tk]):
                    row[key] = round(float(ser[tk]), 2)
            rows.append(row)
        if len(rows) < 300:
            print("Zu wenige Aktien am", d.date(), "- uebersprungen")
            continue
        name = d.strftime("%Y-%m-%d")
        with open(f"data/{name}.json", "w", encoding="utf-8") as f:
            json.dump({"updated": "Handelsschluss " + d.strftime("%d.%m.%Y"), "rows": rows}, f, ensure_ascii=False, separators=(",", ":"))
        names.append(name)
    if not names:
        raise SystemExit("Keine Daten erzeugt")
    with open("data/index.json", "w", encoding="utf-8") as f:
        json.dump({"dates": names[::-1]}, f)
    print(len(names), "Stichtage gespeichert, neuester:", names[-1])

main()
