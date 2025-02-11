import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pandas.plotting import scatter_matrix

# Tải dữ liệu
df = pd.read_csv('energy-data-filtered-full.csv')
df.set_index('year', inplace=True)

# Tiêu đề trang
st.title("Khám Phá Dữ Liệu Điện Năng")
st.write("Dưới đây là các thông tin và phân tích về dữ liệu điện năng theo các quốc gia.")

# Hiển thị bảng dữ liệu
st.subheader("Bảng Dữ Liệu")
st.dataframe(df)

# Thông tin về dữ liệu
st.subheader("Thông Tin Dữ Liệu")
st.write(df.info())

# Cơ bản mô tả dữ liệu
st.subheader("Thống Kê Dữ Liệu")
st.write(df.describe())

# Tìm và hiển thị các giá trị thiếu
st.subheader("Dữ Liệu Thiếu")
missing_data = df.isnull().sum()
missing_percentage = (df.isnull().mean()) * 100
missing_data_df = pd.DataFrame({'Missing Values': missing_data, 'Percentage': missing_percentage})
st.write(missing_data_df)

# Vẽ scatter matrix
st.subheader("Ma Trận Tán Xạ")
columns = ['country', 'electricity_generation', 'renewables_electricity', 'fossil_electricity', 'electricity_demand']
sm = scatter_matrix(df[columns], figsize=(12,10), hist_kwds={'bins': 50})

for ax in sm.ravel():
    ax.tick_params(axis='x', labelrotation=45)
    ax.tick_params(axis='y', labelrotation=45)

st.pyplot(plt)

# Vẽ heatmap ma trận tương quan
st.subheader("Ma Trận Tương Quan")
df_numeric = df.select_dtypes(include=[float, int])
corr_matrix = df_numeric.corr()

plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Ma trận tương quan giữa các chỉ số')
st.pyplot(plt)

# Vẽ biểu đồ phân phối của electricity_demand
st.subheader("Phân Phối Nhu Cầu Điện")
plt.figure(figsize=(10,6))
sns.histplot(df['electricity_demand'], kde=True, bins=30)
plt.title('Phân phối của Nhu cầu Điện')
plt.xlabel('Nhu cầu Điện (Terawatt-hours)')
plt.ylabel('Tần suất')
st.pyplot(plt)

# Vẽ biểu đồ theo quốc gia
st.subheader("Biểu Đồ Theo Quốc Gia")
df = df[df.index.year >= 1965]
fig, axs = plt.subplots(4, 1, figsize=(15, 16))

columns_to_plot = ['electricity_generation', 'renewables_electricity', 'fossil_electricity', 'electricity_demand']
titles = ['Electricity Generation', 'Renewables Electricity', 'Fossil Electricity', 'Electricity Demand']

for i, column in enumerate(columns_to_plot):
    ax = axs[i]
    for country in df['country'].unique():
        df_country = df[df['country'] == country]
        df_country[column].plot(ax=ax, label=country)
    
    ax.set_title(titles[i])
    ax.set_ylabel(column)
    ax.set_xlabel('Year')
    ax.legend(title='Countries', loc='upper left', bbox_to_anchor=(1, 1))

plt.tight_layout()
st.pyplot(plt)
st.subheader("Biểu Đồ Khu Vực - Sự Thay Đổi Điện Tái Tạo")
selected_countries = st.multiselect('Chọn quốc gia', df['country'].unique(), default=df['country'].unique())

df_selected = df[df['country'].isin(selected_countries)]

plt.figure(figsize=(10, 6))
for country in selected_countries:
    df_country = df_selected[df_selected['country'] == country]
    df_country['renewables_electricity'].plot.area(label=country, alpha=0.5)

plt.title('Sự Thay Đổi Sản Xuất Điện Tái Tạo Theo Quốc Gia')
plt.xlabel('Năm')
plt.ylabel('Sản Xuất Điện (Terawatt-hours)')
plt.legend(title='Quốc Gia')
st.pyplot(plt)
