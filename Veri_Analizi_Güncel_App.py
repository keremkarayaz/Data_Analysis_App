import pandas as pd
import streamlit as st
from sklearn.impute import SimpleImputer
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Dil Seçenekleri
LANGUAGES = {
    "English": {
        "title": "Data Cleaning and Visualization Tool",
        "upload_file": "Upload an Excel or CSV file",
        "original_data": "Original Data (First 5 Rows):",
        "cleaned_data": "Cleaned Data (First 5 Rows):",
        "data_cleaning_options": "Data Cleaning Options",
        "fill_missing_values": "Fill Missing Values",
        "filling_method": "Filling Method",
        "remove_duplicates": "Remove Duplicate Records",
        "remove_outliers": "Remove Outliers",
        "clean_text_data": "Clean Text Data",
        "correct_spelling": "Correct Spelling Errors",
        "correct_data_formats": "Correct Data Formats",
        "apply_binning": "Apply Binning",
        "select_column_binning": "Select Column for Binning",
        "number_of_bins": "Number of Bins",
        "visualization_options": "Visualization Options",
        "histogram": "Histogram",
        "select_column_histogram": "Select column for Histogram",
        "boxplot": "Boxplot",
        "select_column_boxplot": "Select column for Boxplot",
        "correlation_heatmap": "Correlation Heatmap",
        "download_cleaned_data": "Download Cleaned Data",
        "download_image": "Download Image",
        "unsupported_file": "Unsupported file type",
        "error_binning": "Binning can only be applied to numeric columns.",
    },
    "Türkçe": {
        "title": "Veri Temizleme ve Görselleştirme Aracı",
        "upload_file": "Bir Excel veya CSV dosyası yükleyin",
        "original_data": "Orijinal Veri (İlk 5 Satır):",
        "cleaned_data": "Temizlenmiş Veri (İlk 5 Satır):",
        "data_cleaning_options": "Veri Temizleme Seçenekleri",
        "fill_missing_values": "Eksik Değerleri Doldur",
        "filling_method": "Doldurma Yöntemi",
        "remove_duplicates": "Yinelenen Kayıtları Kaldır",
        "remove_outliers": "Aykırı Değerleri Kaldır",
        "clean_text_data": "Metin Verilerini Temizle",
        "correct_spelling": "Yazım Hatalarını Düzelt",
        "correct_data_formats": "Hatalı Veri Formatlarını Düzelt",
        "apply_binning": "Gruplama (Binning) Uygula",
        "select_column_binning": "Gruplama için Bir Sütun Seçin",
        "number_of_bins": "Aralık Sayısı",
        "visualization_options": "Görselleştirme Seçenekleri",
        "histogram": "Histogram",
        "select_column_histogram": "Histogram için Bir Sütun Seçin",
        "boxplot": "Kutu Grafiği (Boxplot)",
        "select_column_boxplot": "Kutu Grafiği için Bir Sütun Seçin",
        "correlation_heatmap": "Korelasyon Isı Haritası",
        "download_cleaned_data": "Temizlenmiş Veriyi İndir",
        "download_image": "Görseli İndir",
        "unsupported_file": "Desteklenmeyen dosya türü",
        "error_binning": "Gruplama yalnızca sayısal sütunlar için uygulanabilir.",
    },
}

# Kullanıcı için dil seçimi
language = st.sidebar.selectbox("Language / Dil", ["English", "Türkçe"])
L = LANGUAGES[language]

# Veri Yükleme
def load_data(uploaded_file):
    if uploaded_file is not None:
        file_extension = uploaded_file.name.split('.')[-1]
        if file_extension == 'csv':
            return pd.read_csv(uploaded_file)
        elif file_extension in ['xlsx', 'xls']:
            return pd.read_excel(uploaded_file)
        else:
            st.error(L["unsupported_file"])
            return None
    return None

# Eksik verileri doldurmak
def fill_missing_data(df, fill_value='mean'):
    if fill_value == 'mean':
        return df.fillna(df.mean(numeric_only=True))
    elif fill_value == 'median':
        return df.fillna(df.median(numeric_only=True))
    elif fill_value == 'mode':
        return df.fillna(df.mode().iloc[0])
    return df

# Aykırı değerleri temizleme
def remove_outliers(df):
    for col in df.select_dtypes(include=[np.number]).columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
    return df

# Duplicate (Tekrar Eden) Kayıtları Temizlemek
def remove_duplicates(df):
    return df.drop_duplicates()

# Metin verilerini temizleme
def clean_text_data(df):
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].str.strip().str.lower()
    return df

# Dil hatalarını düzeltme
def correct_spelling(df):
    return df.applymap(lambda x: x.replace("teh", "the") if isinstance(x, str) else x)

# Hatalı veri formatlarını düzeltme
def correct_data_formats(df):
    for col in df.columns:
        try:
            df[col] = pd.to_datetime(df[col], errors='ignore')
            df[col] = pd.to_numeric(df[col], errors='ignore')
        except Exception:
            pass
    return df

# Gruplama (Binning) işlemi
def apply_binning(df, column, bins):
    try:
        if column in df.select_dtypes(include=[np.number]).columns:
            df[f"{column}_binned"] = pd.cut(df[column], bins=bins, labels=False)
        else:
            st.warning(L["error_binning"])
    except Exception as e:
        st.error(f"Error in binning: {e}")
    return df

# Görselleştirme Fonksiyonları
def save_and_display_plot(fig, filename):
    fig.savefig(filename, format='png')
    st.pyplot(fig)
    with open(filename, "rb") as file:
        st.download_button(
            label=L["download_image"],
            data=file,
            file_name=filename,
            mime="image/png"
        )

def plot_histogram(df):
    st.subheader(L["histogram"])
    column = st.selectbox(L["select_column_histogram"], df.select_dtypes(include=[np.number]).columns)
    fig, ax = plt.subplots()
    df[column].hist(bins=20, ax=ax)
    ax.set_title(f"{L['histogram']} - {column}")
    save_and_display_plot(fig, f"{column}_histogram.png")

def plot_boxplot(df):
    st.subheader(L["boxplot"])
    column = st.selectbox(L["select_column_boxplot"], df.select_dtypes(include=[np.number]).columns)
    fig, ax = plt.subplots()
    sns.boxplot(data=df, x=column, ax=ax)
    ax.set_title(f"{L['boxplot']} - {column}")
    save_and_display_plot(fig, f"{column}_boxplot.png")

def plot_heatmap(df):
    st.subheader(L["correlation_heatmap"])
    correlation_matrix = df.select_dtypes(include=[np.number]).corr()
    fig, ax = plt.subplots()
    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt='.2f', ax=ax)
    ax.set_title(L["correlation_heatmap"])
    save_and_display_plot(fig, "correlation_heatmap.png")

# Streamlit Uygulaması
st.title(L["title"])

uploaded_file = st.file_uploader(L["upload_file"], type=["csv", "xlsx", "xls"])
if uploaded_file:
    df = load_data(uploaded_file)
    if df is not None:
        st.write(L["original_data"])
        st.dataframe(df.head())

        # Temizleme Seçenekleri
        st.sidebar.title(L["data_cleaning_options"])
        cleaned_df = df.copy()

        if st.sidebar.checkbox(L["fill_missing_values"]):
            fill_method = st.sidebar.selectbox(L["filling_method"], ['mean', 'median', 'mode'])
            cleaned_df = fill_missing_data(cleaned_df, fill_value=fill_method)

        if st.sidebar.checkbox(L["remove_duplicates"]):
            cleaned_df = remove_duplicates(cleaned_df)

        if st.sidebar.checkbox(L["remove_outliers"]):
            cleaned_df = remove_outliers(cleaned_df)

        if st.sidebar.checkbox(L["clean_text_data"]):
            cleaned_df = clean_text_data(cleaned_df)

        if st.sidebar.checkbox(L["correct_spelling"]):
            cleaned_df = correct_spelling(cleaned_df)

        if st.sidebar.checkbox(L["correct_data_formats"]):
            cleaned_df = correct_data_formats(cleaned_df)

        if st.sidebar.checkbox(L["apply_binning"]):
            column = st.sidebar.selectbox(L["select_column_binning"], cleaned_df.columns)
            bins = st.sidebar.slider(L["number_of_bins"], min_value=2, max_value=20, value=5)
            cleaned_df = apply_binning(cleaned_df, column, bins)

        st.write(L["cleaned_data"])
        st.dataframe(cleaned_df.head())

        # Görselleştirme Seçenekleri
        st.sidebar.title(L["visualization_options"])
        if st.sidebar.checkbox(L["histogram"]):
            plot_histogram(cleaned_df)

        if st.sidebar.checkbox(L["boxplot"]):
            plot_boxplot(cleaned_df)

        if st.sidebar.checkbox(L["correlation_heatmap"]):
            plot_heatmap(cleaned_df)

        # Temizlenmiş Veriyi İndirme
        csv_file = cleaned_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label=L["download_cleaned_data"],
            data=csv_file,
            file_name="cleaned_data.csv",
            mime="text/csv"
        )



