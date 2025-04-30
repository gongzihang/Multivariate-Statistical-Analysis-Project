import pandas as pd
import plotly.express as px

# === Step 1: 读取聚类结果 CSV 文件 ===
df = pd.read_csv("国家聚类结果_85.csv")
df['Country'] = df['Country'].str.strip()  # 清理空格

# === Step 2: 清洗国家名称以匹配 Plotly 的标准国家名 ===
replace_map = {
    'Bolivia (Plurinational State of)': 'Bolivia',
    'Venezuela (Bolivarian Republic of)': 'Venezuela',
    'Iran (Islamic Republic of)': 'Iran',
    'Micronesia (FS of)': 'Micronesia',
    "Lao People's DR": 'Laos',
    'Republic of Korea': 'South Korea',
    'Syrian Arab Republic': 'Syria',
    'Republic of Moldova': 'Moldova',
    "Côte d'Ivoire": 'Ivory Coast',
    'Cabo Verde': 'Cape Verde',
    'Türkiye': 'Turkey',
    'Brunei Darussalam': 'Brunei',
    'United Republic of Tanzania: Mainland': 'Tanzania',
    'Timor-Leste': 'East Timor',
    'D.R. of the Congo': 'Democratic Republic of the Congo',
    'Congo': 'Republic of the Congo'
}
df['Country'] = df['Country'].replace(replace_map)

# 过滤可能无法识别的特殊地区名称
drop_names = [
    "Zanzibar", "Sudan (Former)", "China, Hong Kong SAR", "Curaçao",
    "Greenland", "French Polynesia", "British Virgin Islands",
    "Cayman Islands", "Bermuda", "Aruba", "Kosovo", "Former Netherlands Antilles"
]
df = df[~df['Country'].isin(drop_names)]

# === Step 3: 聚类编号映射为经济类型标签 ===
cluster_label_map = {
    0: "高收入、低增长国家",
    1: "新兴工业化国家",
    2: "资源依赖型国家",
    3: "中等收入、高外债国家"
}
df['ClusterType'] = df['Cluster'].map(cluster_label_map)

# === Step 4: 绘制地图 ===
fig = px.choropleth(
    df,
    locations='Country',
    locationmode='country names',
    color='ClusterType',
    hover_name='Country',
    hover_data={'Cluster': True, 'ClusterType': True},
    labels={'ClusterType': '经济模式类型'}
)

fig.update_layout(
    title_text="全球国家经济模式分类地图（1991年后，PCA + KMeans）",
    title_x=0.5,
    legend_title_text='经济模式类型'
)
fig.update_geos(
    showframe=False,
    showcoastlines=True,
    coastlinecolor="#CCCCCC",
    projection_type='natural earth'
)

fig.show()
