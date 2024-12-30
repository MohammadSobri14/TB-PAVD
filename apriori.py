import streamlit as st
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('D:/TB PAVD/dataset/Transaksii_Petshop.csv')
df['Tanggal Transaksi'] = pd.to_datetime(df['Tanggal Transaksi'], format="%Y-%m-%d")

st.markdown("#### 🚀 Selamat Datang di Aplikasi Prediksi Pembelian Petshop Menggunakan Algoritma Apriori!")
st.divider()
# Menampilkan statistik dataset
st.markdown("### Statistik Dataset")
st.write(f"Jumlah total transaksi: {df['ID Transaksi'].nunique()}")
st.write(f"Rentang tanggal transaksi: {df['Tanggal Transaksi'].min()} hingga {df['Tanggal Transaksi'].max()}")

# Fungsi untuk memfilter data berdasarkan tanggal
def get_data(date):
    data = df[df["Tanggal Transaksi"] == pd.to_datetime(date)]
    return data if not data.empty else "No Result!"

# Fungsi untuk input fitur
def user_input_features():
    item = st.selectbox(
        "Item",
        df.filter(like="Nama Barang").melt()["value"].dropna().unique()
    )
    return item
item = user_input_features()

# Input tanggal menggunakan kalender
tanggal_dipilih = st.date_input(
    "Pilih Tanggal Transaksi",
    value=pd.to_datetime("2024-01-01"),
    min_value=df["Tanggal Transaksi"].min(),
    max_value=df["Tanggal Transaksi"].max()
)
data = get_data(tanggal_dipilih)

# Fungsi encoding
def encode(x):
    return 1 if x >= 1 else 0

if type(data) != str:
    # Membentuk data pivot
    item_columns = [col for col in df.columns if "Nama Barang" in col]
    data_melted = data.melt(id_vars=["ID Transaksi"], value_vars=item_columns, var_name="Barang Ke", value_name="Item")
    data_melted.dropna(subset=["Item"], inplace=True)
    item_count = data_melted.groupby(["ID Transaksi", "Item"]).size().reset_index(name="Count")
    item_count_pivot = item_count.pivot_table(index="ID Transaksi", columns="Item", values="Count", aggfunc='sum').fillna(0)
    item_count_pivot = item_count_pivot.applymap(encode)

    # Apriori dan aturan asosiasi
    frequent_items = apriori(item_count_pivot, min_support=0.01, use_colnames=True)
    frequent_items["num_itemsets"] = frequent_items["itemsets"].apply(len)  # Tambahkan num_itemsets
    rules = association_rules(frequent_items, metric="lift", min_threshold=1, num_itemsets=len(frequent_items))
    rules = rules[["antecedents", "consequents", "support", "confidence", "lift"]].sort_values(by="confidence", ascending=False)

    # Fungsi parsing
    def parse_list(x):
        x = list(x)
        return ", ".join(x) if len(x) > 1 else x[0]

    rules["antecedents"] = rules["antecedents"].apply(parse_list)
    rules["consequents"] = rules["consequents"].apply(parse_list)

    # Menampilkan rekomendasi
    rekomendasi = rules[rules["antecedents"] == item]
    if not rekomendasi.empty:
        st.markdown(f"### Hasil Rekomendasi:")
        st.success(f"Jika konsumen membeli **{item}**, maka membeli **{rekomendasi.iloc[0]['consequents']}** secara bersamaan.")
    else:
        st.warning("Tidak ada rekomendasi berdasarkan item ini.")

    # Visualisasi Aturan Asosiasi
    st.markdown("### Visualisasi Aturan Asosiasi")
    fig, ax = plt.subplots()
    sns.scatterplot(data=rules, x="support", y="confidence", size="lift", hue="lift", palette="viridis", ax=ax)
    ax.set_title("Visualisasi Aturan Asosiasi")
    st.pyplot(fig)

    # Menampilkan statistik aturan asosiasi
    st.markdown(f"### Statistik Aturan Asosiasi")
    avg_confidence = rules['confidence'].mean()
    avg_lift = rules['lift'].mean()
    avg_support = rules['support'].mean()
    
    st.write(f"Rata-rata Confidence: {avg_confidence:.2f}")
    st.write(f"Rata-rata Lift: {avg_lift:.2f}")
    st.write(f"Rata-rata Support: {avg_support:.2f}")

    # Menampilkan jumlah aturan asosiasi yang ditemukan
    st.write(f"Jumlah total aturan asosiasi: {len(rules)}")

    # Rekomendasi berdasarkan aturan dengan confidence tertinggi
    st.markdown("### Aturan dengan Confidence Tertinggi")
    st.write(rules.iloc[0])

    # Tombol untuk download hasil rekomendasi
    import io
    def to_csv(df):
        return df.to_csv(index=False).encode('utf-8')

    csv = to_csv(rules)
    st.download_button(
        label="Download Hasil Rekomendasi",
        data=csv,
        file_name="rekomendasi.csv",
        mime="text/csv"
    )
else:
    st.warning("Tidak ada data transaksi yang sesuai dengan filter Anda.")