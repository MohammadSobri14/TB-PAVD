import streamlit as st
import pandas as pd
from mlxtend.frequent_patterns import association_rules, apriori

# Load dataset
df = pd.read_csv('D:\TB PAVD\dataset\Transaksii_Petshop.csv')
df['Tanggal Transaksi'] = pd.to_datetime(df['Tanggal Transaksi'], format="%Y-%m-%d")

# Tambahkan kolom bulan dan hari
df["month"] = df["Tanggal Transaksi"].dt.month
df["day"] = df["Tanggal Transaksi"].dt.weekday
df["month"].replace(
    [i for i in range(1, 12 + 1)],
    ["Januari", "Februari", "Maret", "April", "Mei", "Juni", "Juli", "Agustus", "September", "Oktober", "November", "Desember"],
    inplace=True,
)
df["day"].replace(
    [i for i in range(6 + 1)],
    ["Senin", "Selasa", "Rabu", "Kamis", "Jumat", "Sabtu", "Minggu"],
    inplace=True,
)

st.title("Prediksi Pembelian Menggunakan Algoritma Apriori")

# Fungsi untuk memfilter data berdasarkan bulan dan hari
def get_data(month='', day=''):
    data = df.copy()
    filtered = data.loc[
        (data["month"].str.contains(month.title())) &
        (data["day"].str.contains(day.title()))
    ]
    return filtered if filtered.shape[0] else "No Result!"

# Fungsi input dari pengguna
def user_input_features():
    item = st.selectbox("Item", ["Makanan Anjing Purina", "Mainan Anjing Kong", "Makanan Kucing Whiskas", "Mainan Kucing ScratchPad",
                                 "Makanan Ikan Gupi", "Life Cat Can - Chicken & Salmon", "Life Cat Can - Kitten Salmon",
                                 "Life Cat Can - Kitten Tuna", "Life Cat Can - Tuna", "Life Cat Pouch - Chicken adult",
                                 "Life Cat Pouch - Chicken Tuna Adult", "Life Cat Pouch - Kitten Chicken",
                                 "Life Cat Pouch - Salmon Adult", "Life Cat Pouch - Kitten Tuna", "Life Cat Pouch Dus",
                                 "Life Cat Tofu - Macha", "Life Cat Tofu - Mango"])
    month = st.select_slider("Month", ["Jan", "Feb", "Mar", "Apr", "Mei", "Jun", "Jul", "Agu", "Sep", "Okt", "Nov", "Des"])
    day = st.select_slider("Day", ["Sen", "Sel", "Rab", "Kam", "Jum", "Sab", "Min"], value="Sen")
    return item, month, day

item, month, day = user_input_features()

data = get_data(month, day)

# Ubah data wide ke long
if type(data) != str:
    item_columns = [f"Nama Barang {i}" for i in range(1, 6)]
    data_long = pd.melt(
        data,
        id_vars=["ID Transaksi", "Tanggal Transaksi"],
        value_vars=item_columns,
        var_name="Barang Ke",
        value_name="Nama Barang"
    ).dropna(subset=["Nama Barang"])

    # Proses grouping
    item_count = data_long.groupby(["ID Transaksi", "Nama Barang"])["Nama Barang"].count().reset_index(name="Count")
    item_count_pivot = item_count.pivot_table(index="ID Transaksi", columns="Nama Barang", values="Count", aggfunc='sum').fillna(0)
    item_count_pivot = item_count_pivot.applymap(lambda x: 1 if x > 0 else 0)

    # Apriori dan aturan asosiasi
    support = 0.01
    frequent_items = apriori(item_count_pivot, min_support=support, use_colnames=True)

    # Menghitung jumlah itemsets
    num_itemsets = frequent_items['support'].count()

    # Association rules dengan num_itemsets
    rules = association_rules(frequent_items, num_itemsets=num_itemsets, metric="lift", min_threshold=1)
    rules = rules[["antecedents", "consequents", "support", "confidence", "lift"]]
    rules.sort_values('confidence', ascending=False, inplace=True)

    # Fungsi untuk parsing list
    def parse_list(x):
        x = list(x)
        if len(x) == 1:
            return x[0]
        elif len(x) > 1:
            return ", ".join(x)

    def return_item_df(item_antecedents):
        data = rules[["antecedents", "consequents"]].copy()
        data["antecedents"] = data["antecedents"].apply(parse_list)
        data["consequents"] = data["consequents"].apply(parse_list)
        return list(data.loc[data["antecedents"] == item_antecedents].iloc[0, :])

    # Tampilkan hasil rekomendasi
    st.markdown("Hasil Rekomendasi:")
    try:
        rekomendasi = return_item_df(item)
        st.success(f"Jika Konsumen Membeli **{item}**, maka membeli **{rekomendasi[1]}** secara bersamaan.")
    except IndexError:
        st.warning("Tidak ada rekomendasi untuk item yang dipilih.")
else:
    st.error("Data tidak ditemukan untuk bulan dan hari yang dipilih.")
