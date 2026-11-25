"""Pemuatan, pembersihan, dan agregasi data penjualan (BAB III - Preprocess)."""
import re
import numpy as np
import pandas as pd

ALIAS = {"tanggal": ["tanggal", "tgl", "date"], "nama": ["nama barang", "nama", "barang", "item"],
         "qty": ["quantity", "jumlah", "qty", "banyak"], "nilai": ["nilai", "pendapatan", "omzet", "total"]}


def _cari(kolom, alias):
    for a in alias:
        for c in kolom:
            if str(c).lower().strip().startswith(a):
                return c
    return None


def _angka(v):
    """Jumlah bisa bertuliskan satuan (mis. '1,5 kg'): ambil angkanya."""
    if isinstance(v, (int, float, np.integer, np.floating)):
        return float(v)
    m = re.search(r"\d+(?:[.,]\d+)?", str(v))
    return float(m.group().replace(",", ".")) if m else np.nan


def muat_data(sumber):
    nama = str(getattr(sumber, "name", sumber)).lower()
    if nama.endswith(".csv"):
        return pd.read_csv(sumber)
    xl = pd.ExcelFile(sumber)
    return xl.parse(xl.sheet_names[0])


def bersihkan(mentah):
    """Return (df, info). df: tgl, nama, key, q (jumlah), nilai."""
    peta = {k: _cari(mentah.columns, v) for k, v in ALIAS.items()}
    hilang = [k for k, v in peta.items() if v is None]
    if hilang:
        raise ValueError("Kolom tidak ditemukan: " + ", ".join(hilang) + ". Perlu: Tanggal, Nama Barang, quantity, Nilai (Rp).")
    df = pd.DataFrame({"tgl": pd.to_datetime(mentah[peta["tanggal"]], errors="coerce").dt.normalize(),
                       "nama": mentah[peta["nama"]].astype(str).str.strip().str.replace(r"\s+", " ", regex=True),
                       "q": mentah[peta["qty"]].map(_angka),
                       "nilai": pd.to_numeric(mentah[peta["nilai"]], errors="coerce")})
    awal = len(df)
    df = df.dropna(subset=["tgl", "nilai"])
    df = df[(df.nilai > 0) & (df.nama != "")].copy()
    df["key"] = df.nama.str.lower()
    kal = pd.date_range(df.tgl.min(), df.tgl.max())
    info = {"baris_awal": awal, "baris_dibuang": awal - len(df), "hari_kalender": len(kal),
            "hari_tercatat": int(df.tgl.nunique()), "hari_kosong": len(kal) - int(df.tgl.nunique()),
            "tanpa_nama": int(df.key.str.contains("tidak tertulis").sum())}
    return df.reset_index(drop=True), info


def kalender(df):
    return pd.date_range(df.tgl.min(), df.tgl.max())


def tanpa_nama(df):
    """Barang bernama '(nama tidak tertulis)' tidak ikut analisis barang."""
    return df[~df.key.str.contains("tidak tertulis")].copy()


def agregasi_barang(db, H):
    """Per barang: total jumlah, hari terjual, omzet, rata-rata per hari (= total / jumlah hari)."""
    a = db.groupby("key").agg(nama=("nama", lambda s: s.value_counts().index[0]), total=("q", "sum"),
                              hari=("tgl", "nunique"), omzet=("nilai", "sum")).sort_values(["total", "hari"], ascending=False)
    a["rata"] = a.total / H
    a["peringkat"] = range(1, len(a) + 1)
    return a


def bagi_tiga(a):
    """Patokan cepat/menengah/lambat: peringkat dibagi tiga, batas hanya di antara dua nilai berbeda (tanpa irisan)."""
    v, n = a.total.values, len(a)
    valid = [i for i in range(1, n) if v[i - 1] != v[i]]
    potong = lambda t: min(valid, key=lambda i: (abs(i - t), i))
    b1, b2 = potong(round(n / 3)), potong(round(2 * n / 3))
    a = a.copy()
    a["tiga"] = ["Cepat"] * b1 + ["Menengah"] * (b2 - b1) + ["Lambat"] * (n - b2)
    batas = {"Cepat": (v[b1 - 1], v[0]), "Menengah": (v[b2 - 1], v[b1]), "Lambat": (v[-1], v[b2])}
    return a, batas


def seri_barang(db, key, kal):
    """Jumlah terjual per hari untuk satu barang (0 jika tidak terjual)."""
    return db[db.key == key].groupby("tgl").q.sum().reindex(kal).fillna(0)


def omzet_mingguan(df, kal):
    """Omzet per blok 7 hari yang berakhir di tanggal terakhir. Hari sisa di awal dibuang."""
    tot = df.groupby("tgl").nilai.sum().reindex(kal)
    nw = len(tot) // 7
    sisa = len(tot) - nw * 7
    seg = tot.iloc[sisa:]
    wk = pd.Series(seg.values.reshape(nw, 7).sum(1), index=pd.date_range(seg.index[0], periods=nw, freq="7D"), name="omzet")
    return wk, sisa
