import io, json, os, time, datetime
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

def indicators(px):
    """RSI(14, Wilder) sowie MACD(12,26,9) fuer alle Handelstage und Ticker."""
    fx = px.ffill()
    delta = fx.diff()
    up, dn = delta.clip(lower=0), (-delta).clip(lower=0)
    au = up.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    ad = dn.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    rsi = (100 - 100 / (1 + au / ad)).where(ad != 0, 100.0).where(au.notna())
    macd = fx.ewm(span=12, adjust=False).mean() - fx.ewm(span=26, adjust=False).mean()
    sig = macd.ewm(span=9, adjust=False).mean()
    return rsi, macd, sig

def efficiency(df):
    """Trendqualitaet: Nettoveraenderung / Summe der absoluten Schritte, mit Vorzeichen (-1 bis +1)."""
    den = df.diff().iloc[1:].abs().sum()
    return ((df.iloc[-1] - df.iloc[0]) / den).where(df.notna().all() & (den > 0))

def download_m(chunk):
    for _ in range(3):
        try:
            df = yf.download(chunk, period="8y", interval="1mo", auto_adjust=False, progress=False, threads=True)["Close"]
            if isinstance(df, pd.Series):
                df = df.to_frame(chunk[0])
            return df
        except Exception as e:
            print("Retry (Monatsdaten):", e)
            time.sleep(5)
    return None

def bulk_m(symbols):
    frames = [f for f in (download_m(symbols[i:i + 100]) for i in range(0, len(symbols), 100)) if f is not None]
    if not frames:
        return None
    df = pd.concat(frames, axis=1)
    return df.loc[:, ~df.columns.duplicated()]

def prep_extras(px):
    """Einmal vorberechnen: Indikatoren, Jahresschlusskurse (fuer den Mindset-Indikator)."""
    rsi, macd, sig = indicators(px)
    yret, yend = {}, {}
    mo = bulk_m(list(px.columns))
    if mo is not None:
        mo.index = pd.to_datetime(mo.index).tz_localize(None)
        m = mo[mo.index.month == 12].copy()
        m.index = m.index.year
        for yr in m.index:
            yend[int(yr)] = m.loc[yr].to_dict()
            if yr - 1 in m.index and yr >= 2021:
                yret[int(yr)] = ((m.loc[yr] / m.loc[yr - 1] - 1) * 100).to_dict()
    else:
        print("Hinweis: keine Monatsdaten - Mindset-Indikator entfaellt")
    print("Zusatzdaten: Jahresschluesse fuer", sorted(yend))
    return {"rsi": rsi, "macd": macd, "sig": sig, "yret": yret, "yend": yend}

def day_extras(px, sub, ret, d, close, sma, pre):
    """Kennzahlen fuer einen Stichtag: Trendqualitaet, groesste Einzelbewegung, Kurssprung-Verdacht."""
    out = {"rsi": pre["rsi"].loc[d], "macd": pre["macd"].loc[d], "sig": pre["sig"].loc[d]}
    w27 = sub.resample("W-FRI").last().tail(27)
    if len(w27) == 27:
        out["tq"], out["gw"] = efficiency(w27), w27.pct_change(fill_method=None).iloc[1:].max() * 100
    d27 = sub.tail(27)
    if len(d27) == 27:
        out["tqd"], out["gd"] = efficiency(d27), d27.pct_change(fill_method=None).iloc[1:].max() * 100
    w = ret.loc[:d].tail(130)
    arr = np.where(np.isnan(w.to_numpy(dtype=float)), -1e9, w.to_numpy(dtype=float))
    gmax, kpos = arr.max(axis=0), arr.argmax(axis=0)
    ks = {}
    for j, tk in enumerate(w.columns):
        g = gmax[j]
        if g >= 60 and tk in close.index and tk in sma.index and pd.notna(close[tk]) and pd.notna(sma[tk]) and sma[tk] > 0 \
                and (close[tk] / (1 + g / 100)) / sma[tk] < 1.1:
            ks[tk] = [round(float(g), 1), w.index[kpos[j]].strftime("%Y-%m-%d")]
    out["ks"] = ks
    Y = d.year
    out["years"] = [pre["yret"].get(yr) for yr in range(2021, Y)]
    out["pe"] = pre["yend"].get(Y - 1)
    return out

def add_extras(row, tk, ex):
    def put(key, ser, nd):
        v = ser.get(tk) if ser is not None else None
        if v is not None and v == v:
            row[key] = round(float(v), nd)
    for k, nd in (("rsi", 1), ("macd", 3), ("sig", 3), ("tq", 2), ("gw", 1), ("tqd", 2), ("gd", 1)):
        put(k, ex.get(k), nd)
    if tk in ex["ks"]:
        row["ks"] = ex["ks"][tk]
    if ex["pe"] and ex["years"] and all(y is not None for y in ex["years"]):
        ys = [y.get(tk) for y in ex["years"]]
        pe = ex["pe"].get(tk)
        if pe is not None and pe == pe and any(v is not None and v == v for v in ys):
            row["y"] = [round(float(v), 1) if v is not None and v == v else None for v in ys]
            row["pe"] = round(float(pe), 2)

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
    pre = prep_extras(px)
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
        ex = day_extras(px, sub, ret, d, close, sma, pre)
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
            add_extras(row, tk, ex)
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
