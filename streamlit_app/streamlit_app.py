"""Web prediksi tarif Uber NYC (Streamlit). Jalankan: streamlit run streamlit_app.py"""
from datetime import date, datetime
from zoneinfo import ZoneInfo

import streamlit as st

import predictor

st.set_page_config(page_title="Prediksi Tarif Uber NYC", page_icon="🚕")


# Model dimuat sekali saja lalu disimpan di cache, supaya tidak dimuat ulang tiap klik
@st.cache_resource
def get_model():
    return predictor.load_model()


model, meta = get_model()
metrics = meta["metrics"]

# Lokasi cepat (longitude, latitude)
PLACES = {
    "Times Square": (-73.9855, 40.7580), "Wall Street": (-74.0090, 40.7060),
    "Central Park": (-73.9654, 40.7829), "Brooklyn Bridge": (-73.9969, 40.7061),
    "Bandara JFK": (-73.7781, 40.6413), "Bandara LaGuardia": (-73.8740, 40.7769),
    "Bandara Newark": (-74.1745, 40.6895), "Grand Central": (-73.9772, 40.7527),
    "Williamsburg": (-73.9573, 40.7081), "Columbia University": (-73.9626, 40.8075),
}
MANUAL = "Isi koordinat manual"


def location_input(title, key, default_place):
    """Pilih lokasi dari daftar, atau isi koordinat sendiri. Mengembalikan (longitude, latitude)."""
    st.subheader(title)
    options = list(PLACES) + [MANUAL]
    choice = st.selectbox("Lokasi cepat", options, index=options.index(default_place), key=key + "_place")
    if choice == MANUAL:
        lat = st.number_input("Latitude", min_value=predictor.LAT[0], max_value=predictor.LAT[1],
                              value=40.7580, step=0.001, format="%.4f", key=key + "_lat")
        lon = st.number_input("Longitude", min_value=predictor.LON[0], max_value=predictor.LON[1],
                              value=-73.9855, step=0.001, format="%.4f", key=key + "_lon")
    else:
        lon, lat = PLACES[choice]
        st.caption(f"Koordinat: {lat:.4f}, {lon:.4f}")
    return lon, lat


st.title("🚕 Prediksi Tarif Uber NYC")
st.caption(f"Model: {meta['model_name']} · R² {metrics['R2']:.3f} · MAE ${metrics['MAE']:.2f} (data uji)")

kiri, kanan = st.columns(2)
with kiri:
    plon, plat = location_input("📍 Titik jemput", "pickup", "Times Square")
with kanan:
    dlon, dlat = location_input("🏁 Tujuan", "dropoff", "Bandara JFK")

sekarang = datetime.now(ZoneInfo("America/New_York"))
c1, c2, c3 = st.columns(3)
tanggal = c1.date_input("Tanggal jemput", value=sekarang.date(), min_value=date(2009, 1, 1), max_value=date(2035, 12, 31))
jam = c2.time_input("Jam jemput (waktu New York)", value=sekarang.time().replace(second=0, microsecond=0))
penumpang = c3.number_input("Jumlah penumpang", min_value=1, max_value=6, value=1, step=1)

if st.button("Hitung estimasi tarif", type="primary"):
    waktu = f"{tanggal:%Y-%m-%d}T{jam:%H:%M}"
    try:
        hasil = predictor.predict_trip(model, metrics["MAE"], waktu, plon, plat, dlon, dlat, int(penumpang))
    except ValueError as err:
        st.error(str(err))
    else:
        a, b, c = st.columns(3)
        a.metric("Estimasi tarif", f"${hasil['fare']:.2f}")
        b.metric("Rentang", f"${hasil['range_low']:.2f} – ${hasil['range_high']:.2f}")
        c.metric("Jarak", f"{hasil['distance_km']:.1f} km")
        st.info("✈️ Perjalanan bandara" if hasil["airport_trip"] else "Bukan perjalanan bandara")
        st.caption("Rentang = tebakan ± 2 × MAE (aturan praktis, bukan jaminan). "
                   "Model dilatih dari data 2009–2015, jadi tarif aktual saat ini bisa berbeda.")

with st.expander("Tentang model"):
    st.write("Jenis masalah: **regresi** (menebak angka tarif dalam USD).")
    st.write(f"Skor di data uji: R² {metrics['R2']:.3f}, MAE ${metrics['MAE']:.2f}, RMSE ${metrics['RMSE']:.2f}.")
    st.write("Waktu yang kamu isi dianggap waktu lokal New York. Web ini tidak punya login.")
