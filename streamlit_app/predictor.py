"""Logika prediksi (tanpa Streamlit): muat model, cek input, dan tebak tarif."""
import json
import os
import sys

import joblib
import pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)
from features import build_features, haversine, make_request_frame  # noqa: E402  (model butuh modul ini saat dibuka)

# Batas NYC: sama dengan yang dipakai saat membersihkan data latih
LON = (-74.30, -73.70)
LAT = (40.50, 41.00)


def load_model():
    """Muat pipeline (pembuat fitur + model) dan ringkasan hasilnya."""
    model = joblib.load(os.path.join(BASE, "models", "fare_model.joblib"))
    meta = json.load(open(os.path.join(BASE, "models", "model_results.json")))
    return model, meta


def validate_trip(pickup_datetime, pickup_longitude, pickup_latitude,
                  dropoff_longitude, dropoff_latitude, passenger_count):
    """Cek input dengan aturan yang SAMA seperti pembersihan data latih. Error -> ValueError."""
    try:
        pd.Timestamp(pickup_datetime)
    except (ValueError, TypeError):
        raise ValueError("Format tanggal/jam tidak valid.")
    for nama, lon, lat in (("jemput", pickup_longitude, pickup_latitude), ("tujuan", dropoff_longitude, dropoff_latitude)):
        if not (LON[0] <= lon <= LON[1] and LAT[0] <= lat <= LAT[1]):
            raise ValueError(f"Koordinat {nama} di luar area New York City (lon -74.30..-73.70, lat 40.50..41.00).")
    if not 1 <= int(passenger_count) <= 6:
        raise ValueError("Jumlah penumpang harus 1-6.")
    if haversine(pickup_longitude, pickup_latitude, dropoff_longitude, dropoff_latitude) < 0.05:
        raise ValueError("Titik jemput dan tujuan terlalu dekat (< 50 m).")


def predict_trip(model, mae, pickup_datetime, pickup_longitude, pickup_latitude,
                 dropoff_longitude, dropoff_latitude, passenger_count):
    """Tebak tarif satu perjalanan. pickup_datetime = waktu LOKAL New York, contoh '2015-06-10T18:30'."""
    validate_trip(pickup_datetime, pickup_longitude, pickup_latitude, dropoff_longitude, dropoff_latitude, passenger_count)
    frame = make_request_frame(pickup_datetime, pickup_longitude, pickup_latitude,
                               dropoff_longitude, dropoff_latitude, int(passenger_count))
    fare = float(model.predict(frame)[0])
    fitur = build_features(frame).iloc[0]
    return {
        "fare": round(fare, 2),
        "distance_km": round(float(fitur.distance_km), 2),
        "airport_trip": bool(fitur.is_airport_trip),
        "range_low": round(max(2.5, fare - 2 * mae), 2),    # rentang = tebakan +/- 2 x MAE (aturan praktis)
        "range_high": round(fare + 2 * mae, 2),
    }
