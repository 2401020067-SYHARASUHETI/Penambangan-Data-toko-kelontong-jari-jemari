# Aplikasi Penambangan Data - Pengelompokan Barang dan Prediksi Penjualan

Aplikasi web (Streamlit) untuk project Penambangan Data (INF11121):
- **K-Medoids** mengelompokkan barang (X = total jumlah terjual, Y = rata-rata penjualan per hari)
- **ARIMA** memprediksi omzet total per minggu
- **Prediksi Bebas**: pilih omzet total (ARIMA, 1-4 minggu) atau cek kelompok sebuah barang (K-Medoids)

## Menjalankan
Butuh Python 3.10 atau lebih baru.

Windows: klik dua kali `jalankan.bat`. Mac/Linux: `bash jalankan.sh`.

Manual:
```
pip install -r requirements.txt
python -m streamlit run app.py
```
Browser terbuka di http://localhost:8501.

## Berkas
| Berkas | Fungsi |
|---|---|
| app.py | Tampilan (6 halaman, termasuk Prediksi Bebas) |
| preprocessing.py | Memuat dan membersihkan data, agregasi per barang, patokan bagi tiga, omzet mingguan |
| model_kmedoids.py | K-Medoids (PAM), Silhouette, Davies-Bouldin, pembanding K-Means |
| model_ts.py | ARIMA mingguan, uji rolling origin |
| data/data_penjualan.xlsx | Data penjualan toko (1 Jan - 18 Sep 2026) |

Data lain dapat diunggah lewat panel kiri (kolom: Tanggal, Nama Barang, quantity, Nilai (Rp)).
