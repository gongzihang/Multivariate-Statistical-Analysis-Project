import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# 1. 读取数据
df_economy = pd.read_csv("all_countries_combined.csv")

# 2. 筛选1992年及之后的数据
df_after_1992 = df_economy[df_economy['Year'] >= 1992]

# 3. 按国家分组，计算均值（只对数值型列）
country_mean_1992 = df_after_1992.groupby('Country').mean(numeric_only=True).reset_index()

# 4. 删除“Year”列
country_mean_1992 = country_mean_1992.drop(columns="Year")

# 5. 标准化数值型数据
df_pca = country_mean_1992.select_dtypes(include=[np.number])
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_pca)

# 6. PCA主成分分析
pca = PCA()
X_pca = pca.fit_transform(X_scaled)
explained_variance = pca.explained_variance_ratio_

# 7. 构造综合评分函数（加权主成分加权求和）
health_index = np.dot(X_pca, explained_variance)
country_mean_1992['Economic_Health_Index'] = health_index

# 8. 聚类分析（KMeans分组）
kmeans = KMeans(n_clusters=4, random_state=42)
clusters = kmeans.fit_predict(X_scaled)
country_mean_1992['Cluster'] = clusters
# 评估 KMeans 聚类效果（越接近1说明聚类效果越好;负值说明聚得不好;通常0.5以上是较好,0.7以上是非常清晰）
score = silhouette_score(X_scaled, clusters)
print(f"Silhouette Score: {score:.4f}")

# 9. 可视化部分
plt.figure(figsize=(12, 20))
sns.barplot(
    x='Economic_Health_Index',
    y='Country',
    data=country_mean_1992.sort_values(by='Economic_Health_Index', ascending=False),
    palette='viridis'
)
plt.title("Country Economic Health Index", fontsize=16)
plt.xlabel("Health Index", fontsize=12)
plt.ylabel("Country", fontsize=12)
plt.yticks(fontsize=4)
plt.tight_layout()
plt.savefig("economic_health_index.png", dpi=300)
plt.show()

# 主成分前两维聚类可视化
plt.figure(figsize=(8, 6))
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=clusters, cmap='tab10', s=50)
plt.title("PCA of Country Economic Indicators with Clusters")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.grid(True)
plt.colorbar(scatter, label='Cluster')
plt.tight_layout()
plt.savefig("country_clusters_pca.png", dpi=300)
plt.show()

# 10. 输出结果表格（可选保存）
output_df = country_mean_1992[['Country', 'Economic_Health_Index', 'Cluster']]
output_df = output_df.sort_values(by='Economic_Health_Index', ascending=False)
print(output_df.head(10))  # 展示前10
output_df.to_csv("economic_health_scores.csv", index=False)
