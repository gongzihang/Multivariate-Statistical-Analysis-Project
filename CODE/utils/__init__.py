import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from factor_analyzer.factor_analyzer import calculate_kmo, calculate_bartlett_sphericity
from factor_analyzer import FactorAnalyzer
import matplotlib.pyplot as plt
# 设置字体
plt.rcParams['font.family'] = 'SimHei'  # 使用黑体字体，或选择其他支持中文的字体

from utils.PCA_util import get_year_df,get_X_scale,get_factor_scores,plot_factor