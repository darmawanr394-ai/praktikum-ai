import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# ==============================
# CONFIG PAGE
# ==============================
st.set_page_config(
    page_title="Klasifikasi Iris",
    page_icon="🌸",
    layout="centered"
)

# ==============================
# LOAD DATA
# ==============================
@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv"
    df = pd.read_csv(url)
    return df

# ==============================
# TRAIN MODEL
# ==============================
@st.cache_resource
def train_model(df):
    X = df.drop("species", axis=1)
    y = df["species"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    accuracy = model.score(X_test, y_test)
    return model, accuracy

# ==============================
# MAIN APP
# ==============================
def main():
    st.title("🌸 Aplikasi Klasifikasi Bunga Iris")
    st.markdown("Aplikasi ini menggunakan Machine Learning untuk memprediksi jenis bunga iris berdasarkan input fitur.")

    # Load data & model
    df = load_data()
    model, accuracy = train_model(df)

    st.subheader("📥 Input Data")

    col1, col2 = st.columns(2)

    with col1:
        sepal_length = st.slider(
            "Sepal Length (cm)",
            float(df.sepal_length.min()),
            float(df.sepal_length.max()),
            float(df.sepal_length.mean())
        )

        sepal_width = st.slider(
            "Sepal Width (cm)",
            float(df.sepal_width.min()),
            float(df.sepal_width.max()),
            float(df.sepal_width.mean())
        )

    with col2:
        petal_length = st.slider(
            "Petal Length (cm)",
            float(df.petal_length.min()),
            float(df.petal_length.max()),
            float(df.petal_length.mean())
        )

        petal_width = st.slider(
            "Petal Width (cm)",
            float(df.petal_width.min()),
            float(df.petal_width.max()),
            float(df.petal_width.mean())
        )

    st.divider()

    # ==============================
    # PREDIKSI
    # ==============================
    if st.button("🔍 Prediksi Sekarang"):
        input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
        prediction = model.predict(input_data)[0]

        st.success(f"🌼 Hasil Prediksi: **{prediction.upper()}**")
        st.info(f"📊 Akurasi model: {accuracy*100:.2f}%")

    # ==============================
    # DATASET
    # ==============================
    with st.expander("📊 Lihat Dataset Iris"):
        st.dataframe(df)

    # ==============================
    # FOOTER
    # ==============================
    st.markdown("---")
    st.caption("Dibuat dengan Streamlit | Tugas Artificial Intelligence")

# ==============================
# RUN
# ==============================
if __name__ == "__main__":
    main()
