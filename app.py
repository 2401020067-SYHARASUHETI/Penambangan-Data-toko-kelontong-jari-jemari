"""Aplikasi Penambangan Data - Pengelompokan Barang (K-Medoids) dan Prediksi Penjualan (ARIMA).

Jalankan:  streamlit run app.py
"""
import io
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from model_kmedoids import klasterkan, silhouette_per_k
from model_ts import arima_mingguan, penanda_lonjakan
from preprocessing import (agregasi_barang, bagi_tiga, bersihkan, kalender, muat_data, omzet_mingguan,
                           seri_barang, tanpa_nama)

st.set_page_config(page_title="Penambangan Data - Toko", page_icon="📊", layout="wide")

# ------------------------------------------------------------------ tema gelap estetik
plt.rcParams.update({
    "figure.facecolor": "#1A1333", "axes.facecolor": "#1A1333", "savefig.facecolor": "#1A1333",
    "axes.edgecolor": "#4A3F7A", "axes.labelcolor": "#D9CFFF", "text.color": "#F1ECFF",
    "xtick.color": "#B8ADE0", "ytick.color": "#B8ADE0", "grid.color": "#2E2552",
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.alpha": 0.6,
    "legend.facecolor": "#241B45", "legend.edgecolor": "#4A3F7A", "legend.labelcolor": "#F1ECFF",
})

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600;700&display=swap');
html, body, [class*="css"], .stMarkdown, button, input, textarea { font-family: 'Poppins', sans-serif !important; }
.stApp { background: radial-gradient(1200px 600px at 10% -10%, #3B1D6E 0%, transparent 60%),
                     radial-gradient(900px 500px at 100% 0%, #123A6B 0%, transparent 55%),
                     linear-gradient(160deg, #0F0A1F 0%, #150F2E 100%); }
header[data-testid="stHeader"] { background: transparent; }
h1 { background: linear-gradient(90deg, #FF6FB5, #B58CFF, #7DE3FF); -webkit-background-clip: text;
     -webkit-text-fill-color: transparent; font-weight: 700 !important; letter-spacing: -0.5px; }
h2, h3 { color: #E9E1FF !important; font-weight: 600 !important; }
section[data-testid="stSidebar"] { background: linear-gradient(180deg, #1C1240 0%, #120C2B 100%);
     border-right: 1px solid rgba(181,140,255,0.25); }
div[data-testid="stMetric"] { background: linear-gradient(135deg, rgba(255,111,181,0.14), rgba(125,227,255,0.10));
     border: 1px solid rgba(181,140,255,0.30); border-radius: 18px; padding: 16px 18px;
     box-shadow: 0 8px 24px rgba(10,5,30,0.45); }
div[data-testid="stMetricLabel"] { color: #CDBFF5 !important; }
div[data-testid="stMetricValue"] { color: #FFFFFF; font-weight: 700; }
.stButton > button, .stDownloadButton > button { background: linear-gradient(90deg, #FF6FB5, #8E6BFF); color: white;
     border: none; border-radius: 999px; padding: 0.55rem 1.6rem; font-weight: 600;
     box-shadow: 0 6px 20px rgba(255,111,181,0.35); transition: all .2s ease; }
.stButton > button:hover, .stDownloadButton > button:hover { transform: translateY(-2px); filter: brightness(1.1);
     box-shadow: 0 10px 28px rgba(255,111,181,0.5); color: white; }
div[data-testid="stDataFrame"], div[data-testid="stImage"] img { border-radius: 16px; overflow: hidden;
     border: 1px solid rgba(181,140,255,0.25); }
div[data-testid="stAlert"] { border-radius: 16px; border: 1px solid rgba(181,140,255,0.25); backdrop-filter: blur(6px); }
div[data-testid="stExpander"] { border: 1px solid rgba(181,140,255,0.25); border-radius: 16px;
     background: rgba(36,27,69,0.55); }
div[data-baseweb="select"] > div, .stTextInput input, .stDateInput input { border-radius: 12px !important; }
hr { border-color: rgba(181,140,255,0.25); }
::-webkit-scrollbar { width: 8px; } ::-webkit-scrollbar-thumb { background: #4A3F7A; border-radius: 8px; }
</style>
""", unsafe_allow_html=True)

DATA_DEFAULT = Path(__file__).parent / "data" / "data_penjualan.xlsx"
WARNA = {"Cepat": "#FF5C8A", "Menengah": "#FFB86B", "Lambat": "#8E9BFF"}


def rp(x):
    return "Rp" + f"{x:,.0f}".replace(",", ".")


# ------------------------------------------------------------------ fungsi ber-cache
@st.cache_data(show_spinner=False)
def proses_data(isi, nama_file):
    if isi is None:
        sumber = DATA_DEFAULT
    else:
        sumber = io.BytesIO(isi)
        sumber.name = nama_file
    return bersihkan(muat_data(sumber))


@st.cache_data(show_spinner=False)
def hitung_kelompok(db, H, k):
    agg = agregasi_barang(db, H)
    agg, batas = bagi_tiga(agg)
    hasil, metrik, medoid, kunci = klasterkan(agg, k)
    return hasil, batas, metrik, medoid, kunci


@st.cache_data(show_spinner=False)
def hitung_sil(agg):
    return silhouette_per_k(agg)


@st.cache_data(show_spinner=False)
def hitung_arima(wk, lon, jumlah=2):
    return arima_mingguan(wk, lon, jumlah_ramal=jumlah)


# ------------------------------------------------------------------ sidebar
st.sidebar.title("📊 Penambangan Data")
st.sidebar.caption("K-Medoids dan ARIMA pada data penjualan toko")
berkas = st.sidebar.file_uploader("Unggah data penjualan (opsional)", type=["xlsx", "xls", "csv"],
                                  help="Kolom: Tanggal, Nama Barang, quantity, Nilai (Rp). Kosongkan untuk memakai data toko bawaan.")
k = st.sidebar.slider("Jumlah kelompok K-Medoids", 2, 5, 3)
pakai_lonjakan = st.sidebar.checkbox("Tandai minggu lonjakan (mis. Lebaran)", value=True)
tgl_awal = st.sidebar.date_input("Lonjakan mulai", pd.Timestamp("2026-03-18"))
tgl_akhir = st.sidebar.date_input("Lonjakan selesai", pd.Timestamp("2026-03-26"))
halaman = st.sidebar.radio("Halaman", ["Ringkasan", "Grafik per Barang", "Pengelompokan (K-Medoids)",
                                       "Prediksi ARIMA", "Prediksi Bebas", "Rekomendasi"])

# ------------------------------------------------------------------ muat data
try:
    df, info = proses_data(berkas.getvalue() if berkas else None, berkas.name if berkas else "")
except Exception as e:  # noqa: BLE001
    st.error(f"Data tidak dapat dibaca: {e}")
    st.stop()

kal = kalender(df)
H = len(kal)
db = tanpa_nama(df)
hasil, batas, metrik, medoid, kunci_medoid = hitung_kelompok(db, H, k)
tertinggi = hasil.groupby("medoid").total.median().idxmax()
wk, sisa = omzet_mingguan(df, kal)
lon = penanda_lonjakan(wk.index, tgl_awal, tgl_akhir) if pakai_lonjakan else None


def grafik_barang(ax, key, judul):
    s = seri_barang(db, key, kal)
    r = hasil.loc[key]
    ax.vlines(s.index, 0, s.values, color=WARNA.get(r.tiga, "#7DE3FF"), lw=0.9)
    ax.set_ylim(0, max(s.max() * 1.15, 1.5))
    ax.set_title(judul, fontsize=8)
    ax.tick_params(labelsize=7)


# ================================================================== halaman
if halaman == "Ringkasan":
    st.title("Ringkasan Data Penjualan")
    st.caption("Sumber: " + (berkas.name if berkas else "data toko bawaan (buku 'Daftar Barang Keluar')"))
    a, b, c = st.columns(3)
    a.metric("Periode", f"{kal.min():%d-%m-%Y} s/d {kal.max():%d-%m-%Y}")
    b.metric("Jumlah hari", H)
    c.metric("Jumlah transaksi (baris)", f"{len(df):,}".replace(",", "."))
    a, b, c = st.columns(3)
    a.metric("Barang dianalisis", len(hasil))
    b.metric("Total omzet", rp(df.nilai.sum()))
    c.metric("Rata-rata omzet harian", rp(df.groupby("tgl").nilai.sum().reindex(kal).mean()))
    st.subheader("Catatan kualitas data")
    st.write(f"- Baris dibuang (nilai kosong/Rp0 atau tanggal tidak valid): **{info['baris_dibuang']}** dari {info['baris_awal']}")
    st.write(f"- Hari tanpa catatan pada rentang data: **{info['hari_kosong']}**")
    st.write(f"- Baris bernama '(nama tidak tertulis)' (dikeluarkan dari analisis barang): **{info['tanpa_nama']}**")
    st.write(f"- Omzet mingguan: {len(wk)} minggu penuh (hari sisa di awal yang dibuang: {sisa})")
    harian = df.groupby("tgl").nilai.sum().reindex(kal)
    fig, ax = plt.subplots(figsize=(10, 3.2))
    ax.plot(harian.index, harian.values / 1e3, lw=0.9, color="#7DE3FF")
    ax.set_ylabel("Omzet harian (ribu Rp)")
    st.pyplot(fig)
    plt.close(fig)

elif halaman == "Grafik per Barang":
    st.title("Grafik Penjualan Harian per Barang")
    st.write("Satu grafik untuk setiap barang (jumlah terjual per hari), diurutkan dari rata-rata penjualan tertinggi.")
    mode = st.radio("Tampilan", ["Satu barang", "Galeri (12 per halaman)"], horizontal=True)
    urut = hasil.sort_values("peringkat")
    if mode == "Satu barang":
        pilihan = {f"#{int(r.peringkat)} {r.nama}": key for key, r in urut.iterrows()}
        nama = st.selectbox("Pilih barang", list(pilihan))
        key = pilihan[nama]
        r = hasil.loc[key]
        a, b, c, d_, e = st.columns(5)
        a.metric("Peringkat", int(r.peringkat))
        b.metric("Total terjual", f"{r.total:g}")
        c.metric("Hari terjual", int(r.hari))
        d_.metric("Rata-rata/hari", f"{r.rata:.3f}")
        e.metric("Kategori patokan", r.tiga)
        fig, ax = plt.subplots(figsize=(10, 3.2))
        grafik_barang(ax, key, f"{r.nama}: jumlah terjual per hari")
        st.pyplot(fig)
        plt.close(fig)
    else:
        n_hal = int(np.ceil(len(urut) / 12))
        hal = st.slider("Halaman", 1, n_hal, 1)
        bagian = urut.iloc[(hal - 1) * 12: hal * 12]
        fig, axs = plt.subplots(4, 3, figsize=(11, 10))
        for ax, (key, r) in zip(axs.ravel(), bagian.iterrows()):
            grafik_barang(ax, key, f"#{int(r.peringkat)} {r.nama[:26]}\ntotal {r.total:g} | rata-rata {r.rata:.3f} | {r.tiga}")
        for ax in axs.ravel()[len(bagian):]:
            ax.axis("off")
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

elif halaman == "Pengelompokan (K-Medoids)":
    st.title("Pengelompokan Barang dengan K-Medoids")
    st.write("Sumbu X = total jumlah terjual, sumbu Y = rata-rata penjualan per hari (total dibagi jumlah hari). "
             "Karena Y adalah X dibagi bilangan tetap, titik-titik tersusun mengikuti garis lurus.")
    a, b, c = st.columns(3)
    a.metric("Silhouette Score", f"{metrik['silhouette']:.3f}")
    b.metric("Davies-Bouldin Index", f"{metrik['davies_bouldin']:.3f}")
    c.metric("Silhouette K-Means (pembanding)", f"{metrik['silhouette_kmeans']:.3f}")
    fig, ax = plt.subplots(figsize=(8, 4.6))
    for g, s in hasil.groupby("medoid"):
        ax.scatter(s.total, s.rata, s=16, label=f"{g} ({len(s)})", color=WARNA.get(g), alpha=0.85)
    ax.scatter(hasil.loc[kunci_medoid].total, hasil.loc[kunci_medoid].rata, s=130, marker="*", color="black", zorder=5, label="Medoid")
    ax.set_xlabel("X: total jumlah terjual")
    ax.set_ylabel("Y: rata-rata penjualan per hari")
    ax.set_title(f"K-Medoids (K = {k}), {len(hasil)} barang")
    ax.legend(fontsize=8)
    st.pyplot(fig)
    plt.close(fig)
    st.subheader("Ringkasan kelompok K-Medoids")
    urut_h = hasil.sort_values("peringkat")

    def lima(x):
        return ", ".join(x.head(5)) + (" ..." if len(x) > 5 else "")

    def daftar_nama(kol, judul):
        for g in sorted(hasil[kol].unique()):
            sub = urut_h[urut_h[kol] == g]
            with st.expander(f"{judul} {g}: semua barang ({len(sub)})"):
                st.dataframe(sub[["peringkat", "nama", "total", "rata"]].rename(columns={
                    "peringkat": "Peringkat", "nama": "Barang", "total": "Total terjual", "rata": "Rata-rata/hari"}), hide_index=True)

    ring = urut_h.groupby("medoid").agg(jumlah_barang=("nama", "size"), total_min=("total", "min"), total_maks=("total", "max"),
                                        total_omzet=("omzet", "sum"), barang_teratas=("nama", lima))
    ring["porsi_omzet_%"] = (ring.total_omzet / ring.total_omzet.sum() * 100).round(1)
    st.dataframe(ring)
    daftar_nama("medoid", "Kelompok")
    st.write("Medoid: " + ", ".join(f"{m['nama']} ({m['kelompok']}, total {m['total']:g})" for m in medoid))
    st.subheader("Patokan: peringkat dibagi tiga (batas tanpa irisan)")
    pat = pd.DataFrame({g: {"jumlah_barang": int((hasil.tiga == g).sum()), "total_min": batas[g][0], "total_maks": batas[g][1],
                            "porsi_omzet_%": round(hasil[hasil.tiga == g].omzet.sum() / hasil.omzet.sum() * 100, 1),
                            "barang_teratas": lima(urut_h[urut_h.tiga == g].nama)}
                        for g in ["Cepat", "Menengah", "Lambat"]}).T
    st.dataframe(pat)
    daftar_nama("tiga", "Patokan")
    with st.expander("Silhouette Score untuk berbagai K"):
        st.dataframe(hitung_sil(hasil[["nama", "total", "rata"]]).round(3), hide_index=True)
    st.subheader("Daftar barang")
    dasar = st.radio("Kelompokkan menurut", ["K-Medoids", "Patokan bagi tiga"], horizontal=True)
    kol = "medoid" if dasar == "K-Medoids" else "tiga"
    pilih = st.multiselect("Kategori", sorted(hasil[kol].unique()), default=[tertinggi] if kol == "medoid" else ["Cepat"])
    cari = st.text_input("Cari nama barang")
    tabel = hasil[hasil[kol].isin(pilih)]
    if cari:
        tabel = tabel[tabel.nama.str.contains(cari, case=False, na=False)]
    tampil = tabel.sort_values("peringkat")[["peringkat", "nama", "total", "hari", "rata", "omzet", "tiga", "medoid"]].rename(columns={
        "peringkat": "Peringkat", "nama": "Barang", "total": "Total terjual", "hari": "Hari terjual", "rata": "Rata-rata/hari",
        "omzet": "Omzet (Rp)", "tiga": "Patokan", "medoid": "K-Medoids"})
    st.dataframe(tampil, hide_index=True)
    st.download_button("Unduh hasil (CSV)", tampil.to_csv(index=False).encode("utf-8"), "hasil_pengelompokan.csv", "text/csv")

elif halaman == "Prediksi ARIMA":
    st.title("Prediksi Omzet Total per Minggu (ARIMA)")
    st.write("Omzet dijumlahkan per blok 7 hari, lalu ARIMA memprediksi minggu berikutnya. Orde dipilih lewat AIC; "
             "minggu lonjakan diberi variabel penanda.")
    if len(wk) < 20:
        st.error("Data terlalu pendek untuk ARIMA mingguan (minimal sekitar 20 minggu).")
        st.stop()
    minggu_a = st.slider("Berapa minggu ke depan yang ingin diprediksi", 1, 4, 2)
    with st.spinner("Menghitung ARIMA..."):
        hs = hitung_arima(wk, lon, minggu_a)
    kol = st.columns(minggu_a + 1)
    kol[0].metric("Orde terpilih (p, d, q)", str(hs["order"]))
    kol[0].caption(f"Jumlah minggu: {len(wk)}")
    for i, (tg, nilai) in enumerate(hs["ramal"].items()):
        kol[i + 1].metric(f"Minggu ke-{i + 1} ({tg:%d-%m})", rp(nilai))
    fig, ax = plt.subplots(figsize=(10, 3.4))
    ax.plot(wk.index, wk.values / 1e6, marker="o", ms=3, color="#7DE3FF", label="Omzet mingguan")
    ax.plot(hs["uji"].index, hs["uji"].prediksi / 1e6, marker="s", ms=3, color="#FF6FB5", label="Prediksi (uji 8 minggu)")
    ax.plot(hs["ramal"].index, hs["ramal"].values / 1e6, "--o", ms=3, color="#B6F36B", label="Prediksi ke depan")
    ax.set_ylabel("Juta Rp")
    ax.legend(fontsize=8)
    st.pyplot(fig)
    plt.close(fig)
    tab = pd.DataFrame({"Minggu": [f"{i:%d-%m-%Y} s/d {i + pd.Timedelta(days=6):%d-%m-%Y}" for i in hs["ramal"].index],
                        "Prediksi omzet (Rp)": hs["ramal"].values.round(0).astype(int)})
    st.dataframe(tab, hide_index=True)
    st.subheader("Uji akurasi (8 minggu terakhir, prediksi 1 minggu ke depan)")
    st.dataframe(hs["tabel"].round(1))
    st.info("Pembanding: naif (omzet minggu lalu) dan rata-rata 4 minggu terakhir. Prediksi adalah perkiraan, bukan angka pasti.")

elif halaman == "Prediksi Bebas":
    st.title("Prediksi Bebas")
    st.write("Pilih sendiri apa yang ingin dilihat, lalu tekan tombol.")
    target = st.radio("Yang ingin dilihat", ["Omzet total toko per minggu (ARIMA)", "Kelompok sebuah barang (K-Medoids)"], horizontal=True)
    if target.startswith("Omzet"):
        if len(wk) < 20:
            st.error("Data terlalu pendek untuk ARIMA mingguan.")
            st.stop()
        minggu = st.slider("Berapa minggu ke depan", 1, 4, 1)
        if st.button("Prediksi"):
            with st.spinner("Menghitung..."):
                hs = hitung_arima(wk, lon, minggu)
            fig, ax = plt.subplots(figsize=(10, 3.2))
            ax.plot(wk.index[-20:], wk.values[-20:] / 1e6, marker="o", ms=3, color="#7DE3FF", label="Omzet mingguan")
            ax.plot(hs["ramal"].index, hs["ramal"].values / 1e6, "--o", ms=4, color="#B6F36B", label="Prediksi")
            ax.set_ylabel("Juta Rp")
            ax.legend(fontsize=8)
            st.pyplot(fig)
            plt.close(fig)
            st.dataframe(pd.DataFrame({"Minggu": [f"{i:%d-%m-%Y} s/d {i + pd.Timedelta(days=6):%d-%m-%Y}" for i in hs["ramal"].index],
                                       "Prediksi omzet (Rp)": hs["ramal"].values.round(0).astype(int)}), hide_index=True)
            st.info(f"Orde model: {hs['order']}. Kesalahan uji sekitar {hs['tabel'].loc['ARIMA', 'WAPE (%)']:.0f}% (WAPE, 8 minggu terakhir).")
    else:
        from sklearn.preprocessing import StandardScaler
        skala = StandardScaler().fit(hasil[["total", "rata"]].values)
        pusat = hasil.loc[kunci_medoid]

        def kelompok_terdekat(total):
            z = skala.transform([[total, total / H]])
            d = ((skala.transform(pusat[["total", "rata"]].values) - z) ** 2).sum(1)
            return pusat.medoid.iloc[int(d.argmin())]

        urut = hasil.sort_values("peringkat")
        pilihan = {f"#{int(r.peringkat)} {r.nama}": key for key, r in urut.iterrows()}
        nama = st.selectbox("Pilih barang (ketik untuk mencari)", list(pilihan))
        r = hasil.loc[pilihan[nama]]
        a, b, c, d_ = st.columns(4)
        a.metric("Kelompok K-Medoids", r.medoid)
        b.metric("Patokan bagi tiga", r.tiga)
        c.metric("Total terjual", f"{r.total:g}")
        d_.metric("Rata-rata/hari", f"{r.rata:.3f}")
        st.subheader("Coba angka sendiri")
        coba = st.number_input("Seandainya sebuah barang terjual sebanyak ... unit selama periode data", min_value=1, value=int(r.total), step=1)
        if st.button("Cek kelompok"):
            st.success(f"Dengan total {coba} unit ({coba / H:.3f} per hari), barang ini masuk kelompok **{kelompok_terdekat(coba)}** (paling dekat dengan medoid kelompok itu).")

else:
    st.title("Rekomendasi untuk Pemilik Toko")
    inti = hasil[hasil.medoid == tertinggi].sort_values("peringkat")
    cepat = hasil[hasil.tiga == "Cepat"]
    lambat = hasil[hasil.tiga == "Lambat"]
    tot = hasil.omzet.sum()
    st.subheader("1. Barang inti terlaris: jaga ketersediaan")
    st.write(f"{len(inti)} barang ({inti.omzet.sum() / tot * 100:.1f}% omzet): " + ", ".join(inti.nama) + ".")
    st.subheader("2. Kategori cepat menurut patokan: prioritaskan modal belanja")
    st.write(f"{len(cepat)} barang (jumlah terjual sekurang-kurangnya {batas['Cepat'][0]:g} unit) menyumbang {cepat.omzet.sum() / tot * 100:.1f}% omzet.")
    st.subheader("3. Barang lambat: kurangi pembelian")
    st.write(f"{len(lambat)} barang (terjual {batas['Lambat'][0]:g} unit) hanya {lambat.omzet.sum() / tot * 100:.1f}% omzet. "
             "Beli bila ada pesanan agar modal tidak tertahan.")
    if len(wk) >= 20:
        hs = hitung_arima(wk, lon)
        st.subheader("4. Gambaran omzet minggu depan")
        st.write(f"Perkiraan omzet minggu {hs['ramal'].index[0]:%d-%m-%Y} s/d {hs['ramal'].index[0] + pd.Timedelta(days=6):%d-%m-%Y}: "
                 f"**{rp(hs['ramal'].iloc[0])}** (perkiraan, bukan angka pasti).")
    st.warning("Rekomendasi bersifat tidak langsung: toko tidak memiliki catatan stok sehingga kekurangan atau kelebihan stok tidak diukur langsung.")
