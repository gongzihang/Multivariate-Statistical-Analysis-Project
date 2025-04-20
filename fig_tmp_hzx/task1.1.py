# 导入库
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns

# 设置图像风格
sns.set(style="whitegrid")

# === 第一步：加载数据 ===
df = pd.read_csv("Global Economy Indicators.csv")  # 路径可改为你的本地路径或上传路径

# 清洗列名（去除多余空格）
df.columns = df.columns.str.strip()

# === 第二步：筛选1991年以后的数据 ===
df = df[df['Year'] >= 1991]

# === 第三步：按国家聚合求平均值，表示长期经济结构 ===
df_grouped = df.groupby("Country").mean(numeric_only=True)

# 去除含缺失值的国家
df_grouped_cleaned = df_grouped.dropna(axis=0, how='any')

# 保存国家名列表
country_names = df_grouped_cleaned.index.tolist()

# === 第四步：数据标准化 ===
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_grouped_cleaned)

# === 第五步：主成分分析（PCA）保留95%信息 ===
pca = PCA(n_components=0.95)
X_pca = pca.fit_transform(X_scaled)

# 打印降维后维度信息
print(f"PCA降维后形状: {X_pca.shape}")
print(f"累计解释方差比: {pca.explained_variance_ratio_.sum():.4f}")

# === 第六步：KMeans 聚类（设定k=4）===
k = 4
kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
clusters = kmeans.fit_predict(X_pca)

# === 第七步：合成聚类结果DataFrame ===
cluster_df = pd.DataFrame(X_pca, columns=[f'PC{i+1}' for i in range(X_pca.shape[1])])
cluster_df['Country'] = country_names
cluster_df['Cluster'] = clusters

# === 第八步：可视化聚类结果（前两个主成分）===
plt.figure(figsize=(10, 6))
sns.scatterplot(data=cluster_df, x='PC1', y='PC2', hue='Cluster', palette='tab10')
plt.title("Countries Clustering by Economic Features (PCA + KMeans)")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.legend(title="Cluster")
plt.grid(True)
plt.tight_layout()
plt.show()
