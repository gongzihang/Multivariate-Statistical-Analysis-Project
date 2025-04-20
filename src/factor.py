import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from factor_analyzer.factor_analyzer import calculate_kmo, calculate_bartlett_sphericity
from factor_analyzer import FactorAnalyzer
import matplotlib.pyplot as plt
# 设置字体
plt.rcParams['font.family'] = 'SimHei'  # 使用黑体字体，或选择其他支持中文的字体

from utils import get_year_df,get_X_scale,get_factor_scores

df_economy = pd.read_csv("data\\all_countries_combined.csv")
df_economy = df_economy.drop(columns="Gross Domestic Product (GDP)")
mean_X_scaled,country_mean,_ = get_X_scale(df_economy=df_economy, left_year=1992, right_year=2021)

##### 因子检验 #######
kmo_all, kmo_model = calculate_kmo(mean_X_scaled)
chi_square_value, p_value = calculate_bartlett_sphericity(mean_X_scaled)

print("KMO：", kmo_model)
print("Bartlett检验p值：", p_value)

###########碎石土、特征值挑选因子数量###############
fa = FactorAnalyzer()
fa.fit(mean_X_scaled)

ev, v = fa.get_eigenvalues()
print(ev)
plt.plot(range(1, len(ev)+1), ev, marker='o')
plt.title('Scree Plot')
plt.xlabel('因子个数')
plt.ylabel('特征值')
plt.grid()
plt.savefig("fig_tmp_gzh/碎石土.png", dpi=300, bbox_inches='tight')
# plt.show()
# raise
##############################################
fa = FactorAnalyzer(n_factors=2)
# fa = FactorAnalyzer(n_factors=2, rotation='varimax')
fa.fit(mean_X_scaled)

ev, v = fa.get_eigenvalues()
print("特征值：", ev)
print("每个因子解释的方差比例：", ev / ev.sum())

# 查看因子载荷矩阵（每个原始变量在因子上的权重）
loadings = fa.loadings_
print(loadings)

################################################################
def see_weight():
    # 获取因子载荷矩阵并转化为DataFrame
    loadings_df = pd.DataFrame(loadings, index=country_mean.columns[1:], columns=[f'Factor{i+1}' for i in range(fa.n_factors)])
    # 查看结果
    # 创建一个字典，用来存储排序后的因子表
    factor_sorted_dict = {}

    # 对每个因子按权重排序，并将其存储为一个单独的表
    for factor in loadings_df.columns:
        # 按照每个因子的载荷（绝对值）排序，降序排列
        sorted_factor = loadings_df[[factor]].sort_values(by=factor, ascending=False)
        factor_sorted_dict[factor] = sorted_factor

    # 打印排序后的结果（可以选择保存为CSV文件或其他格式）
    for factor, sorted_table in factor_sorted_dict.items():
        print(f"\n{factor} 排序后的因子载荷:")
        print(sorted_table)
        
def plot_scatter(factor_scores_df,left_year,right_year,save_path):
    # TODO 美化格式修改这个函数
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

    plt.title(f"Factor1 vs Factor2 因子得分散点图({left_year} - {right_year})", fontsize=16)
    plt.xlabel("Factor1：消费与服务主导", fontsize=12)
    plt.ylabel("Factor2：农业与人口主导", fontsize=12)
    plt.grid(True)
    plt.tight_layout()
    # 保存图片到指定路径（支持 .png, .jpg, .pdf 等）
    plt.savefig(save_path, dpi=300, bbox_inches='tight')  # dpi 可调节清晰度
    # plt.show()

def draw_scatter(left_year, right_year, save_dir):
    factor_scores_df = get_factor_scores(df_economy=df_economy,left_year=left_year,right_year=right_year,
                                         loadings=loadings)
    save_path = os.path.join(save_dir,f"{left_year}-{right_year}.png")
    plot_scatter(factor_scores_df,left_year,right_year,save_path)
    

if __name__ =="__main__":
    see_weight()
    save_dir = "fig_tmp_gzh\\factor_scatter"
    for left_year in range(1992,2017):
        right_year = left_year+5
        draw_scatter(left_year,right_year,save_dir)