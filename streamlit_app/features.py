"""Pembersihan & rekayasa fitur. Dipakai notebook, training, dan web app (satu sumber kebenaran)."""
import numpy as np
import pandas as pd

# 6 kolom mentah yang dipakai sebagai masukan
RAW_COLS = ["pickup_datetime", "pickup_longitude", "pickup_latitude",
            "dropoff_longitude", "dropoff_latitude", "passenger_count"]
# Kolom yang mau ditebak
TARGET = "fare_amount"

# Kotak batas New York City
# Batas wilayah NYC (kotak koordinat). Titik di luar kotak dianggap salah catat
LON_MIN, LON_MAX = -74.30, -73.70
LAT_MIN, LAT_MAX = 40.50, 41.00

# Koordinat 3 bandara (lon, lat): JFK, LaGuardia, Newark
AIRPORTS = {"jfk": (-73.7781, 40.6413), "lga": (-73.8740, 40.7769), "ewr": (-74.1745, 40.6895)}
# Titik acuan pusat kota: Times Square (Midtown) dan Wall Street (Downtown)
MIDTOWN = (-73.9855, 40.7580)    # Times Square
DOWNTOWN = (-74.0110, 40.7075)   # Wall Street

# Daftar 26 fitur yang akan dimasukkan ke model
FEATURES = [
    "pickup_longitude", "pickup_latitude", "dropoff_longitude", "dropoff_latitude", "passenger_count",
    "distance_km", "manhattan_km", "bearing",
    "hour", "dayofweek", "month", "year", "is_weekend", "is_rush_hour", "is_night",
    "pickup_jfk_km", "pickup_lga_km", "pickup_ewr_km", "drop_jfk_km", "drop_lga_km", "drop_ewr_km",
    "pickup_midtown_km", "drop_midtown_km", "pickup_downtown_km", "drop_downtown_km", "is_airport_trip",
]


# Jarak garis lurus di permukaan bumi (km) dari dua titik koordinat
def haversine(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 6371.0 * 2 * np.arcsin(np.sqrt(a))


# Arah perjalanan dalam derajat (0-360) dari titik jemput ke tujuan
def bearing(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    x = np.sin(dlon) * np.cos(lat2)
    y = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dlon)
    return (np.degrees(np.arctan2(x, y)) + 360) % 360


# Membuang baris yang tidak masuk akal, aturan demi aturan, sambil mencatat jumlah yang dibuang
def clean_data(df):
    """Buang baris tidak valid (aturan berbasis domain, TIDAK dipelajari dari data -> tidak ada kebocoran).
    Mengembalikan (df_bersih, laporan) dengan laporan = jumlah baris yang dibuang per aturan."""
    df = df.copy()
    df["pickup_datetime"] = pd.to_datetime(df["pickup_datetime"], utc=True, errors="coerce")
    report = []

    def apply(name, keep):
        nonlocal df
        report.append((name, int((~keep).sum())))
        df = df[keep]

    # Tiap aturan di bawah membuang baris yang melanggar, dan jumlahnya dicatat di laporan
    apply("Nilai kosong / waktu tidak terbaca", df[RAW_COLS + [TARGET]].notna().all(axis=1))
    # Tarif harus $2,50-$200 (batas $200 adalah keputusan kami, lihat penjelasan di bawah)
    apply("Tarif di luar $2,50 - $200", df[TARGET].between(2.5, 200))
    # Penumpang harus 1-6
    apply("Penumpang di luar 1-6", df["passenger_count"].between(1, 6))
    # Titik harus di dalam kotak NYC (koordinat 0,0 otomatis ikut terbuang)
    apply("Koordinat jemput di luar NYC (termasuk 0,0)",
          df["pickup_longitude"].between(LON_MIN, LON_MAX) & df["pickup_latitude"].between(LAT_MIN, LAT_MAX))
    apply("Koordinat tujuan di luar NYC (termasuk 0,0)",
          df["dropoff_longitude"].between(LON_MIN, LON_MAX) & df["dropoff_latitude"].between(LAT_MIN, LAT_MAX))
    # Hitung jarak, lalu buang perjalanan < 50 m (batal/salah catat) atau > 100 km
    dist = haversine(df.pickup_longitude, df.pickup_latitude, df.dropoff_longitude, df.dropoff_latitude)
    apply("Jarak < 50 m atau > 100 km", (dist >= 0.05) & (dist <= 100))
    return df.reset_index(drop=True), pd.DataFrame(report, columns=["aturan", "baris_dibuang"])


# Mengubah 6 kolom mentah menjadi 26 fitur siap pakai untuk model
def build_features(df):
    """Ubah 6 kolom mentah menjadi 26 fitur. Fungsi murni (tanpa parameter yang dipelajari dari data),
    sehingga aman dipakai di dalam Pipeline dan identik saat training maupun prediksi di web."""
    df = df.copy()
    # Waktu jemput di data = UTC; diubah ke waktu lokal New York supaya jam dan hari sesuai kenyataan
    dt = pd.to_datetime(df["pickup_datetime"], utc=True).dt.tz_convert("America/New_York")  # waktu lokal NYC
    out = pd.DataFrame(index=df.index)
    for c in ["pickup_longitude", "pickup_latitude", "dropoff_longitude", "dropoff_latitude", "passenger_count"]:
        out[c] = df[c].astype(float)
    plo, pla, dlo, dla = out.pickup_longitude, out.pickup_latitude, out.dropoff_longitude, out.dropoff_latitude
    # Jarak garis lurus, jarak 'berpetak' (timur-barat + utara-selatan, mengikuti jalan), dan arah perjalanan
    out["distance_km"] = haversine(plo, pla, dlo, dla)
    out["manhattan_km"] = haversine(plo, pla, dlo, pla) + haversine(dlo, pla, dlo, dla)
    out["bearing"] = bearing(plo, pla, dlo, dla)
    # Fitur waktu: jam, hari (0 = Senin), bulan, tahun
    out["hour"] = dt.dt.hour
    out["dayofweek"] = dt.dt.dayofweek
    out["month"] = dt.dt.month
    out["year"] = dt.dt.year
    # Penanda 0/1: akhir pekan, jam sibuk (07-09 & 16-19 hari kerja), malam (22-05)
    out["is_weekend"] = (out.dayofweek >= 5).astype(int)
    out["is_rush_hour"] = ((out.hour.between(7, 9) | out.hour.between(16, 19)) & (out.dayofweek < 5)).astype(int)
    out["is_night"] = ((out.hour >= 22) | (out.hour <= 5)).astype(int)
    # Jarak ke tiap bandara; disebut 'perjalanan bandara' kalau jemput/tujuan kurang dari 2 km dari bandara
    near = np.zeros(len(out), dtype=bool)
    for n, (lo, la) in AIRPORTS.items():
        out[f"pickup_{n}_km"] = haversine(plo, pla, lo, la)
        out[f"drop_{n}_km"] = haversine(dlo, dla, lo, la)
        near |= (out[f"pickup_{n}_km"] < 2).values | (out[f"drop_{n}_km"] < 2).values
    # Jarak ke pusat kota (Midtown & Downtown)
    out["pickup_midtown_km"] = haversine(plo, pla, *MIDTOWN)
    out["drop_midtown_km"] = haversine(dlo, dla, *MIDTOWN)
    out["pickup_downtown_km"] = haversine(plo, pla, *DOWNTOWN)
    out["drop_downtown_km"] = haversine(dlo, dla, *DOWNTOWN)
    # Penanda 0/1 perjalanan bandara
    out["is_airport_trip"] = near.astype(int)
    # Kembalikan hanya 26 fitur, dengan urutan yang tetap
    return out[FEATURES]


# Dipakai web: mengubah 1 permintaan (waktu lokal NYC) jadi tabel berformat sama dengan data latih (UTC)
def make_request_frame(local_time, plon, plat, dlon, dlat, passengers):
    """Satu permintaan dari web (waktu LOKAL NYC) -> DataFrame mentah berformat sama dengan data latih (UTC)."""
    ts = pd.Timestamp(local_time).tz_localize("America/New_York", ambiguous=False, nonexistent="shift_forward")
    return pd.DataFrame([{
        "pickup_datetime": ts.tz_convert("UTC").strftime("%Y-%m-%d %H:%M:%S UTC"),
        "pickup_longitude": plon, "pickup_latitude": plat,
        "dropoff_longitude": dlon, "dropoff_latitude": dlat, "passenger_count": passengers}])
