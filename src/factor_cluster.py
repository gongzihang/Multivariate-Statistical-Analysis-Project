import sys, os
import seaborn as sns
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from factor_analyzer.factor_analyzer import calculate_kmo, calculate_bartlett_sphericity
from factor_analyzer import FactorAnalyzer
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
# 设置字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 显示中文
plt.rcParams['axes.unicode_minus'] = False  # 正常显示负号
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from statsmodels.stats.outliers_influence import variance_inflation_factor

import warnings
warnings.filterwarnings("ignore")

def output_country_by_cluster(df):
    """
    输入：df是包含 'Country' 和 'Cluster' 两列的DataFrame
    功能：按照Cluster分类，输出每一类对应的Country列表
    """

    # 检查是否包含需要的列
    if 'Country' not in df.columns or 'Cluster' not in df.columns:
        raise ValueError("DataFrame中必须包含 'Country' 和 'Cluster' 两列")

    # 按Cluster分组
    grouped = df.groupby('Cluster')

    # 创建一个字典，用来存储每个Cluster对应的Country列表
    cluster_country_dict = {}

    for cluster_label, group in grouped:
        country_list = group['Country'].tolist()  # 取出这一组的所有Country，转成列表
        cluster_country_dict[cluster_label] = country_list
        
    return cluster_country_dict

cluster_out = pd.read_csv("国家聚类结果_85.csv")
cluster_out.info()
cluster_country_dict = output_country_by_cluster(cluster_out)
cluster_country_dict_path = {}
for k,v in cluster_country_dict.items():
    path_list = []
    for name in v:
        path_list.append(name.replace('.', '').replace(' ', '_').replace(':', "")+".csv")
    cluster_country_dict_path[k] = path_list

# for k,v in cluster_country_dict_path.items():
#     print(k)
#     print(v)
"""
0
['_Afghanistan_.csv', '_Albania_.csv', ..., '_Zimbabwe_.csv']
1
['_United_States_.csv']
2
['_China_.csv']
3
['_Brazil_.csv', '_Canada_.csv', '_France_.csv', 
'_Germany_.csv', '_India_.csv', '_Italy_.csv', 
'_Japan_.csv', '_Republic_of_Korea_.csv', 
'_Russian_Federation_.csv', '_Spain_.csv', '_United_Kingdom_.csv']
"""

data_folder = "data\countries_data"
out_path = "fig_tmp_gzh\\factor_load"

def Check(mean_X_scaled, path):
    ##### 因子检验 #######
    kmo_all, kmo_model = calculate_kmo(mean_X_scaled)
    chi_square_value, p_value = calculate_bartlett_sphericity(mean_X_scaled)

    print("KMO：", kmo_model)
    print("Bartlett检验p值：", p_value)

    ###########碎石土、特征值挑选因子数量###############
    fa = FactorAnalyzer()
    fa.fit(mean_X_scaled)

    ev, v = fa.get_eigenvalues()
    # print(ev)
    plt.figure(figsize=(6, 4))  # 创建一个新的图
    plt.plot(range(1, len(ev)+1), ev, marker='o')
    plt.title('Scree Plot')
    plt.xlabel('因子个数')
    plt.ylabel('特征值')
    plt.grid()
    plt.savefig(path, dpi=300, bbox_inches='tight')
    
def see_weight(fa, loadings, path):
    # 获取因子载荷矩阵并转化为DataFrame
    loadings_df = pd.DataFrame(
        loadings, 
        index=df.columns[0:-1], 
        columns=[f'Factor{i+1}' for i in range(fa.n_factors)]
    )
    
    # 创建一个列表，用来收集所有排序后的表
    all_factors_list = []

    # 遍历每一个因子
    for factor in loadings_df.columns:
        # 按照当前因子载荷的绝对值大小，降序排列
        sorted_factor = loadings_df[[factor]].reindex(
            loadings_df[factor].abs().sort_values(ascending=False).index
        )
        # 添加一列标记是哪个因子
        sorted_factor['因子名称'] = factor
        # 把这一部分存到列表中
        all_factors_list.append(sorted_factor)

    # 把所有排序好的表上下合并成一个总表
    final_df = pd.concat(all_factors_list)

    # 把因子名称这一列放到最前面，顺序好看一点
    final_df = final_df[['因子名称'] + [col for col in final_df.columns if col != '因子名称']]

    # 打印总表
    print("\n最终汇总后的因子载荷表:")
    print(final_df)

    # 保存成一个CSV文件
    final_df.to_csv(path, encoding='utf-8-sig', index=True)
    print(f"\n已保存到文件: {path}")

def visualize_factor_regression_list(df, loadings=None, path=None):
    """
    df: 包含 'Year', 'GDP', 'Factor1', 'Factor2' 四列
    loadings: 可选 DataFrame，格式如下：
        因子名称 | Factor1 | Factor2
        ----------------------------
         变量A  |  0.52   |  0.12
         变量B  |  0.74   | -0.33
         ...
    
    输出：
        - 实际 vs 预测图
        - QQ图（残差正态性检验）
        - 因子与 GDP 关系图
        - 因子载荷图（如提供）
        - 控制台输出 R²、调整后 R²、MSE
    """
    def compute_vif(df):
        X = df.iloc[:, 1:-1]
        X = sm.add_constant(X)
        vif_df = pd.DataFrame()
        vif_df['变量'] = X.columns
        vif_df['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
        print("VIF（方差膨胀因子）:")
        print(vif_df)
        print()
    
    def check_multicollinearity(df):
        corr = df.iloc[:, 1:-1].corr()
        print("因子间相关系数矩阵：")
        print(corr)
        print()

        # 画热力图
        plt.figure(figsize=(6, 6))
        sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", square=True)
        plt.title("因子相关性热力图")
        plt.tight_layout()
        # plt.show()
        plt.savefig(os.path.join(path,"因子间相关系数矩阵.png"), dpi=300, bbox_inches='tight')
        
    # 内部函数 1：拟合回归模型
    def fit_regression(df):
        X = df.iloc[:, 1:-1]
        y = df['GDP']
        name_list = X.columns
        check_multicollinearity(df)
        compute_vif(df)
        model = LinearRegression().fit(X, y)
        y_pred = model.predict(X)
        df['GDP_pred'] = y_pred
        df['residual'] = y - y_pred

        # 模型评估指标
        r2 = r2_score(y, y_pred)
        adj_r2 = 1 - (1 - r2) * (len(y) - 1) / (len(y) - X.shape[1] - 1)
        mse = mean_squared_error(y, y_pred)
        
        print(f"模型评估结果：")
        print(f"  R²            : {r2:.4f}")
        print(f"  调整后 R²     : {adj_r2:.4f}")
        print(f"  MSE（均方误差）: {mse:.4f}")
        print(f"回归系数：")
        for i in range(len(name_list)):
            print(f"  {name_list[i]} 系数  : {model.coef_[i]:.4f}")
        # print(f"  Factor2 系数  : {model.coef_[1]:.4f}")
        print(f"  截距（Intercept） : {model.intercept_:.4f}")
        print()

        return model, df

    # 图1：实际 vs 预测 GDP
    def plot_actual_vs_pred(df):
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x='Year', y='GDP', data=df, label="实际 GDP")
        sns.lineplot(x='Year', y='GDP_pred', data=df, color='red', label="预测 GDP")
        plt.xlabel('年份')
        plt.ylabel('GDP')
        plt.title('实际 GDP 与 预测 GDP 对比')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        # plt.show()
        plt.savefig(os.path.join(path,"GDP预测结果.png"), dpi=300, bbox_inches='tight')
        

    # 图2：残差 QQ 图
    def plot_qq(df):
        plt.figure(figsize=(6, 6))
        sm.qqplot(df['residual'], line='s')
        plt.title('残差 QQ 图（正态性检验）')
        plt.tight_layout()
        # plt.show()
        plt.savefig(os.path.join(path,"残差QQ图.png"), dpi=300, bbox_inches='tight')
        
        

    # 图3：因子与 GDP 的关系
    def plot_gdp_vs_factors(df):
        plt.figure(figsize=(12, 5))

        plt.subplot(1, 2, 1)
        sns.regplot(x='Factor1', y='GDP', data=df,
                    scatter_kws={"s": 40}, line_kws={"color": "red"})
        plt.title('GDP vs Factor1')

        plt.subplot(1, 2, 2)
        sns.regplot(x='Factor2', y='GDP', data=df,
                    scatter_kws={"s": 40}, line_kws={"color": "red"})
        plt.title('GDP vs Factor2')

        plt.tight_layout()
        # plt.show()
        plt.savefig(os.path.join(path,"因子与 GDP 的关系.png"), dpi=300, bbox_inches='tight')
        

    # 图4：因子载荷图（可选）
    def plot_factor_loadings(loadings):
        if loadings is None:
            return

        plt.figure(figsize=(8, 6))
        for i, factor in enumerate(['Factor1', 'Factor2']):
            sub = loadings[['因子名称', factor]].dropna().sort_values(factor, ascending=True)
            plt.barh(sub['因子名称'], sub[factor], label=factor, alpha=0.7)
        plt.xlabel('载荷值')
        plt.title('因子载荷图')
        plt.legend()
        plt.tight_layout()
        # plt.show()
        plt.savefig(os.path.join(path,"因子载荷图.png"), dpi=300, bbox_inches='tight')
        

    # --- 主流程 ---
    model, df = fit_regression(df)
    plot_actual_vs_pred(df)
    plot_qq(df)
    plot_gdp_vs_factors(df)
    plot_factor_loadings(loadings)

################################## USA ######################################
print("="*20, "Factor of USA", "="*20)
USA_path = os.path.join(out_path,"USA")
os.makedirs(USA_path, exist_ok=True)
Reg_df = pd.DataFrame()

df = pd.read_csv(os.path.join(data_folder,cluster_country_dict_path[1][0]))
Reg_df["Year"] = df["Year"]
df = df.drop(columns=['Country'])
df = df.drop(columns=['Year'])
df.info()
# 标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df)

# 提取成分与GDP
gdp = X_scaled[:,-1:]
X_scaled = X_scaled[:,:-1]

# 展示碎石土挑选因子
path = os.path.join(USA_path,"碎石土.png")
Check(X_scaled,path)
print("[INFO] choose Factor number 2 ")

# 因子分析
# fa = FactorAnalyzer(n_factors=2)
fa = FactorAnalyzer(n_factors=2, rotation='varimax')
fa.fit(X_scaled)

ev, v = fa.get_eigenvalues()
print("特征值：", ev)
print("每个因子解释的方差比例：", ev / ev.sum())

# 查看因子载荷矩阵（每个原始变量在因子上的权重）
loadings = fa.loadings_
csv_path = os.path.join(USA_path, "因子分析结果.csv")
see_weight(fa,loadings, csv_path)


# 得到因子得分
factor_scores = fa.transform(X_scaled)

# 构造回归分析表
# Reg_df['Population'] = X_scaled[:,0]
# Reg_df['GCF'] = X_scaled[:,7]
# Reg_df['GFCG'] = X_scaled[:,8]
# Reg_df['Exports'] = X_scaled[:,4]

# Reg_df['ISIC_A_B'] = X_scaled[:,2]
# Reg_df['ISIC_C_E'] = X_scaled[:,12]
# Reg_df['ISIC_F'] = X_scaled[:,3]
# Reg_df['ISIC_G_H'] = X_scaled[:,16]
# Reg_df['ISIC_I'] = X_scaled[:,15]
# Reg_df['ISIC_J_P'] = X_scaled[:,13]
Reg_df['Factor1'] = factor_scores[:, 0]
Reg_df['Factor2'] = factor_scores[:, 1]
Reg_df["GDP"] = gdp

feature_names = df.columns[0:-1]  # 原始变量名
loadings_df = pd.DataFrame(loadings, columns=['Factor1', 'Factor2'])
loadings_df.insert(0, '因子名称', feature_names)

# 回归分析并可视化
visualize_factor_regression_list(Reg_df, loadings_df, USA_path)

print("+"*50)
print("\n")

################################## China ######################################
print("="*20, "Factor of China", "="*20)
China_path = os.path.join(out_path,"China")
os.makedirs(China_path, exist_ok=True)

Reg_df = pd.DataFrame()
df = pd.read_csv(os.path.join(data_folder,cluster_country_dict_path[2][0]))
Reg_df["Year"] = df["Year"]
df = df.drop(columns=['Country'])
df = df.drop(columns=['Year'])

# 标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df)

# 提取成分与GDP
gdp = X_scaled[:,-1:]
X_scaled = X_scaled[:,:-1]

# 展示碎石土挑选因子
path = os.path.join(China_path,"碎石土.png")
Check(X_scaled,path)
print("[INFO] choose Factor number 3 ")

# 因子分析
fa = FactorAnalyzer(n_factors=3, rotation='varimax')
fa.fit(X_scaled)

ev, v = fa.get_eigenvalues()
print("特征值：", ev)
print("每个因子解释的方差比例：", ev / ev.sum())

# 查看因子载荷矩阵（每个原始变量在因子上的权重）
loadings = fa.loadings_

csv_path = os.path.join(China_path, "因子分析结果.csv")
see_weight(fa,loadings, csv_path)

# 得到因子得分
factor_scores = fa.transform(X_scaled)

# 构造回归分析表
Reg_df['Factor1'] = factor_scores[:, 0]
Reg_df['Factor2'] = factor_scores[:, 1]
Reg_df['Factor3'] = factor_scores[:, 2]
Reg_df["GDP"] = gdp

feature_names = df.columns[0:-1]  # 原始变量名
loadings_df = pd.DataFrame(loadings, columns=['Factor1', 'Factor2', 'Factor3'])
loadings_df.insert(0, '因子名称', feature_names)

# 回归分析并可视化
visualize_factor_regression_list(Reg_df, loadings_df, China_path)

print("+"*50)
print("\n")


################################## Japan等 ######################################
print("="*20, "Factor of Japan_UK", "="*20)
Japan_UK_path = os.path.join(out_path,"Japan_UK")
os.makedirs(Japan_UK_path, exist_ok=True)


df_list = []
for file_name in cluster_country_dict_path[3]:
    df = pd.read_csv(os.path.join(data_folder,file_name))
    df_list.append(df)

# Reg_df = pd.DataFrame()
# Reg_df["Year"] = df["Year"]
df = pd.concat(df_list, ignore_index=True)
df = df.drop(columns=['Country'])
df = df.drop(columns=['Year'])
# 标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df)

# 提取成分与GDP
gdp = X_scaled[:,-1:]
X_scaled = X_scaled[:,:-1]

# 展示碎石土挑选因子
path = os.path.join(Japan_UK_path,"碎石土.png")
Check(X_scaled,path)
print("[INFO] choose Factor number 4 ")

# 因子分析
fa = FactorAnalyzer(n_factors=4, rotation='varimax')
fa.fit(X_scaled)

ev, v = fa.get_eigenvalues()
print("特征值：", ev)
# print("每个因子解释的方差比例：", ev / ev.sum())
# 打印前几个累计比例（前6个因子为例）
cumulative_var = np.cumsum(ev / ev.sum())
for i, var in enumerate(cumulative_var[:6], start=1):
    print(f"前{i}个因子累计解释方差比例：{var:.4f}")
# 查看因子载荷矩阵（每个原始变量在因子上的权重）
loadings = fa.loadings_

csv_path = os.path.join(Japan_UK_path, "因子分析结果.csv")
see_weight(fa,loadings, csv_path)




Reg_df = pd.DataFrame()
df = pd.read_csv("data\countries_data\_Japan_.csv")
Reg_df["Year"] = df["Year"]
df = df.drop(columns=['Country'])
df = df.drop(columns=['Year'])

X_scaled = scaler.transform(df)
# 提取成分与GDP
gdp = X_scaled[:,-1:]
X_scaled = X_scaled[:,:-1]

# 得到因子得分
factor_scores = fa.transform(X_scaled)

# 构造回归分析表
Reg_df['Factor1'] = factor_scores[:, 0]
Reg_df['Factor2'] = factor_scores[:, 1]
Reg_df['Factor3'] = factor_scores[:, 2]
Reg_df['Factor4'] = factor_scores[:, 3]
Reg_df["GDP"] = gdp
print(Reg_df)
feature_names = df.columns[0:-1]  # 原始变量名
loadings_df = pd.DataFrame(loadings, columns=['Factor1', 'Factor2', 'Factor3', 'Factor4'])
loadings_df.insert(0, '因子名称', feature_names)

# 回归分析并可视化
visualize_factor_regression_list(Reg_df, loadings_df, Japan_UK_path)

print("+"*50)
print("\n")


################################## other等 ######################################
print("="*20, "Factor of Other", "="*20)
Other_path = os.path.join(out_path,"Other")
os.makedirs(Other_path, exist_ok=True)


df_list = []
for file_name in cluster_country_dict_path[0]:
    df = pd.read_csv(os.path.join(data_folder,file_name))
    df_list.append(df)

# Reg_df = pd.DataFrame()
# Reg_df["Year"] = df["Year"]
Reg_df = pd.DataFrame()
df = pd.concat(df_list, ignore_index=True)
Reg_df["Year"] = df["Year"]

df = df.drop(columns=['Country'])
df = df.drop(columns=['Year'])
# 标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df)

# 提取成分与GDP
gdp = X_scaled[:,-1:]
X_scaled = X_scaled[:,:-1]

# 展示碎石土挑选因子
path = os.path.join(Other_path,"碎石土.png")
Check(X_scaled,path)
print("[INFO] choose Factor number 3 ")

# 因子分析
fa = FactorAnalyzer(n_factors=5, rotation='varimax')
fa.fit(X_scaled)

ev, v = fa.get_eigenvalues()
print("特征值：", ev)
# print("每个因子解释的方差比例：", ev / ev.sum())
# 打印前几个累计比例（前6个因子为例）
cumulative_var = np.cumsum(ev / ev.sum())
for i, var in enumerate(cumulative_var[:7], start=1):
    print(f"前{i}个因子累计解释方差比例：{var:.4f}")
# 查看因子载荷矩阵（每个原始变量在因子上的权重）
loadings = fa.loadings_

csv_path = os.path.join(Other_path, "因子分析结果.csv")
see_weight(fa,loadings, csv_path)


Reg_df = pd.DataFrame()
df = pd.read_csv("data\countries_data\_Australia_.csv")
Reg_df["Year"] = df["Year"]
df = df.drop(columns=['Country'])
df = df.drop(columns=['Year'])

X_scaled = scaler.transform(df)
# 提取成分与GDP
gdp = X_scaled[:,-1:]
X_scaled = X_scaled[:,:-1]

# 得到因子得分
factor_scores = fa.transform(X_scaled)

# 构造回归分析表
Reg_df['Factor1'] = factor_scores[:, 0]
Reg_df['Factor2'] = factor_scores[:, 1]
# Reg_df['Factor3'] = factor_scores[:, 2]
# Reg_df['Factor4'] = factor_scores[:, 3]
Reg_df['Factor5'] = factor_scores[:, 4]
Reg_df["GDP"] = gdp
feature_names = df.columns[0:-1]  # 原始变量名
loadings_df = pd.DataFrame(loadings, columns=['Factor1', 'Factor2', 'Factor3', 'Factor4', 'Factor5'])
loadings_df.insert(0, '因子名称', feature_names)

# 回归分析并可视化
visualize_factor_regression_list(Reg_df, loadings_df, Other_path)

print("+"*50)
print("\n")
