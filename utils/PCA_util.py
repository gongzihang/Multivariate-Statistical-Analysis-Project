import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from factor_analyzer.factor_analyzer import calculate_kmo, calculate_bartlett_sphericity
from factor_analyzer import FactorAnalyzer
import matplotlib.pyplot as plt
# 设置字体
plt.rcParams['font.family'] = 'SimHei'  # 使用黑体字体，或选择其他支持中文的字体

def get_year_df(df_economy, left_year, right_year, with_year=False):
    # 先筛选 1992 年及以后的数据
    df_select = df_economy[(df_economy['Year'] >= left_year) & (df_economy['Year'] <= right_year)]

    # 然后按 Country 分组，计算均值（默认对数值型列计算均值）
    country_mean = df_select.groupby('Country').mean(numeric_only=True)

    # 如果你想保留国家名作为列，而不是索引，可以 reset_index()
    country_mean = country_mean.reset_index()
    if not with_year:
        country_mean = country_mean.drop(columns="Year")
    
    # 查看结果
    # print(len(country_mean))
    # print(country_mean.head())
    
    return country_mean

def get_X_scale(df_economy, left_year, right_year, with_year=False):
    country_mean = get_year_df(df_economy=df_economy, left_year=left_year, right_year=right_year, with_year=with_year)
    df_factor = country_mean.select_dtypes(include=[np.number])
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_factor)
    return X_scaled, country_mean,df_factor

def get_factor_scores(df_economy, left_year, right_year, loadings,with_year=False):
    X_scaled,country_mean, df_factor = get_X_scale(df_economy=df_economy, left_year=left_year, right_year=right_year, with_year=with_year)
    # factor_scores = fa.transform(X_scaled)
    factor_scores =  np.dot(X_scaled, loadings)
    factor_scores_df = pd.DataFrame(factor_scores, 
                                index=country_mean['Country'], 
                                columns=["Factor1", "Factor2"])
    return factor_scores_df

def plot_factor(factor_scores_df):
    plt.figure(figsize=(12, 8))

    # 散点图
    plt.scatter(factor_scores_df["Factor1"], factor_scores_df["Factor2"], color='skyblue', edgecolors='k')

    # 加标签（只标出极端值，避免太乱）
    for country in factor_scores_df.index:
        x = factor_scores_df.loc[country, "Factor1"]
        y = factor_scores_df.loc[country, "Factor2"]
        # 只标出得分很高或很低的国家
        if abs(x) > 10 or abs(y) > 2.5:
            plt.text(x + 0.3, y + 0.2, country, fontsize=9)

    plt.axhline(0, color='gray', linestyle='--')  # 水平轴
    plt.axvline(0, color='gray', linestyle='--')  # 垂直轴

    plt.title("Factor1 vs Factor2 因子得分散点图", fontsize=16)
    plt.xlabel("Factor1：消费与服务主导", fontsize=12)
    plt.ylabel("Factor2：农业与人口主导", fontsize=12)
    plt.grid(True)
    plt.tight_layout()
    plt.show()