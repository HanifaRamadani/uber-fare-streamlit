# Prediksi Tarif Uber NYC (Streamlit)

Web untuk menebak tarif perjalanan Uber di New York City (regresi).

- `streamlit_app.py` : tampilan web
- `predictor.py`     : logika prediksi dan pengecekan input
- `features.py`      : pembersihan data dan pembuatan fitur (sama seperti saat training)
- `models/`          : model hasil training dan ringkasan skornya

Jalankan di komputer sendiri:

    pip install -r requirements.txt
    streamlit run streamlit_app.py
