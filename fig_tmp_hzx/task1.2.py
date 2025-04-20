import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

# === 第一步：加载数据 ===
# 请将路径替换为你的本地路径，例如 "./Global Economy Indicators.csv"
df = pd.read_csv("Global Economy Indicators.csv")

# === 第二步：清洗列名，筛选1991年及以后的数据 ===
df.columns = df.columns.str.strip()
df = df[df['Year'] >= 1991]

# === 第三步：每个国家取平均值（代表长期经济状态）===
df_grouped = df.groupby("Country").mean(numeric_only=True)

# 去除含有缺失值的国家
df_grouped_cleaned = df_grouped.dropna(axis=0, how='any').copy()
country_names = df_grouped_cleaned.index.tolist()

# === 第四步：标准化数据 ===
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_grouped_cleaned)

# === 第五步：主成分分析（保留95%信息量）===
pca = PCA(n_components=0.95)
X_pca = pca.fit_transform(X_scaled)

# === 第六步：KMeans聚类分析（设定k=4）===
k = 4
kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
clusters = kmeans.fit_predict(X_pca) + 1

# === 第七步：生成聚类结果DataFrame并保存 ===
cluster_df = pd.DataFrame(X_pca, columns=[f'PC{i+1}' for i in range(X_pca.shape[1])])
cluster_df['Country'] = country_names
cluster_df['Cluster'] = clusters

# 保存国家聚类结果
clustered_countries = cluster_df[['Country', 'Cluster']].sort_values(by='Cluster')
clustered_countries.to_csv("国家聚类结果.csv", index=False)

# === 第八步：计算每一类的经济指标均值并导出 ===
df_grouped_cleaned['Cluster'] = clusters
cluster_features = df_grouped_cleaned.groupby('Cluster').mean()
cluster_features.to_csv("每类经济特征均值.csv")
