"""K-Medoids (PAM) untuk pengelompokan barang. X = total jumlah, Y = rata-rata per hari (BAB IV)."""
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.preprocessing import StandardScaler

NAMA = ["Lambat", "Menengah", "Cepat"]


def pam(Z, k):
    """Partitioning Around Medoids: BUILD lalu SWAP sampai biaya tidak turun."""
    D = cdist(Z, Z)
    n = len(Z)
    med = [int(D.sum(1).argmin())]
    while len(med) < k:
        terbaik = None
        for c in range(n):
            if c in med:
                continue
            biaya = np.minimum(D[:, med].min(1), D[:, c]).sum()
            if terbaik is None or biaya < terbaik[0]:
                terbaik = (biaya, c)
        med.append(terbaik[1])
    biaya = D[:, med].min(1).sum()
    membaik = True
    while membaik:
        membaik = False
        for mi in range(k):
            for c in range(n):
                if c in med:
                    continue
                t = med.copy()
                t[mi] = c
                b = D[:, t].min(1).sum()
                if b < biaya - 1e-9:
                    med, biaya, membaik = t, b, True
    return np.array(D[:, med].argmin(1)), med, float(biaya), D


def _beri_nama(label, total):
    urut = pd.Series(total).groupby(label).median().sort_values().index.tolist()
    return dict(zip(urut, NAMA))


def klasterkan(agg, k=3):
    X = agg[["total", "rata"]].values
    Z = StandardScaler().fit_transform(X)
    label, med, biaya, D = pam(Z, k)
    peta = _beri_nama(label, agg.total.values) if k == 3 else {i: f"Kluster {i + 1}" for i in range(k)}
    hasil = agg.copy()
    hasil["medoid"] = [peta[l] for l in label]
    km = KMeans(k, n_init=50, random_state=42).fit(Z)
    pk = _beri_nama(km.labels_, agg.total.values) if k == 3 else {i: f"Kluster {i + 1}" for i in range(k)}
    hasil["kmeans"] = [pk[l] for l in km.labels_]
    metrik = {"silhouette": float(silhouette_score(D, label, metric="precomputed")),
              "davies_bouldin": float(davies_bouldin_score(Z, label)),
              "silhouette_kmeans": float(silhouette_score(Z, km.labels_))}
    medoid = [{"kelompok": peta[i], "nama": agg.nama.iloc[m], "total": float(agg.total.iloc[m])} for i, m in enumerate(med)]
    return hasil, metrik, medoid, [agg.index[m] for m in med]


def silhouette_per_k(agg, k_maks=6):
    Z = StandardScaler().fit_transform(agg[["total", "rata"]].values)
    out = []
    for k in range(2, k_maks + 1):
        l, _, b, D = pam(Z, k)
        out.append({"K": k, "silhouette": silhouette_score(D, l, metric="precomputed"), "biaya": b})
    return pd.DataFrame(out)
