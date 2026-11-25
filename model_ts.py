"""ARIMA (omzet total per minggu) - BAB IV.2."""
import warnings
import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

CAND_ARIMA = [(0, 0, 0), (1, 0, 0), (0, 1, 1), (1, 1, 1), (1, 0, 1), (2, 0, 1), (2, 1, 1), (0, 1, 2)]


def _fit(y, order, seasonal=(0, 0, 0, 0), exog=None, trend=None, maxiter=200):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return SARIMAX(y, exog=exog, order=order, seasonal_order=seasonal, trend=trend,
                       enforce_stationarity=False, enforce_invertibility=False).fit(disp=False, maxiter=maxiter)


def _metrik(a, f):
    a, f = np.asarray(a, float), np.asarray(f, float)
    return {"MAE": float(np.mean(np.abs(a - f))), "RMSE": float(np.sqrt(np.mean((a - f) ** 2))),
            "MAPE (%)": float(np.mean(np.abs(a - f) / a) * 100), "WAPE (%)": float(np.abs(a - f).sum() / a.sum() * 100)}


def penanda_lonjakan(indeks, awal, akhir, panjang=7):
    """1 untuk minggu yang beririsan dengan periode lonjakan (mis. Lebaran)."""
    if awal is None or akhir is None:
        return None
    d = pd.Series(((indeks <= pd.Timestamp(akhir)) & (indeks + pd.Timedelta(days=panjang - 1) >= pd.Timestamp(awal))).astype(float),
                  index=indeks, name="lonjakan")
    return d if d.sum() > 0 else None


def arima_mingguan(wk, lon, uji=8, jumlah_ramal=2):
    """Pilih orde lewat AIC, uji 1-minggu-ke-depan pada `uji` minggu terakhir, lalu ramal."""
    ex = lambda s: None if lon is None else lon.loc[s.index]
    tr = wk.iloc[:-uji]
    kandidat = []
    for o in CAND_ARIMA:
        try:
            kandidat.append((_fit(tr, o, exog=ex(tr)).aic, o))
        except Exception:  # noqa: BLE001
            pass
    kandidat.sort()
    order = kandidat[0][1]
    A, F, N, M, tgl = [], [], [], [], []
    for t in range(len(wk) - uji, len(wk)):
        latih = wk.iloc[:t]
        m = _fit(latih, order, exog=ex(latih))
        fc = m.forecast(1, exog=None if lon is None else lon.iloc[t:t + 1]).iloc[0]
        A.append(wk.iloc[t]); F.append(fc); N.append(latih.iloc[-1]); M.append(latih.iloc[-4:].mean()); tgl.append(wk.index[t])
    tabel = pd.DataFrame({"ARIMA": _metrik(A, F), "Naif (minggu lalu)": _metrik(A, N), "Rata-rata 4 minggu": _metrik(A, M)}).T
    mf = _fit(wk, order, exog=ex(wk))
    idx = pd.date_range(wk.index[-1] + pd.Timedelta(days=7), periods=jumlah_ramal, freq="7D")
    fc = mf.forecast(jumlah_ramal, exog=None if lon is None else pd.Series(0.0, index=idx, name="lonjakan"))
    ramal = pd.Series(np.clip(fc.values, 0, None), index=idx)
    uji_df = pd.DataFrame({"aktual": A, "prediksi": F}, index=pd.DatetimeIndex(tgl))
    return {"order": order, "tabel": tabel, "ramal": ramal, "uji": uji_df, "kandidat": kandidat}
