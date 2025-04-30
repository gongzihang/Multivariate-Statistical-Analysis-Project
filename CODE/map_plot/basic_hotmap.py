import cv2
import numpy as np
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
def hotmap_on_map():
    # 总体平均 GDP
    df = pd.read_csv("./data/all_countries_combined.csv")
    gdp_by_country_5 = (df[df['Year'] > 2015].groupby('Country')['Gross Domestic Product (GDP)'].mean().reset_index())
    gdp_by_country_5.columns = ['Country', 'Average_GDP']
    gdp_by_country_10 = (df[df['Year'] > 2010].groupby('Country')['Gross Domestic Product (GDP)'].mean().reset_index())
    gdp_by_country_10.columns = ['Country', 'Average_GDP']
    gdp_by_country_20 = (df[df['Year'] > 2000].groupby('Country')['Gross Domestic Product (GDP)'].mean().reset_index())
    gdp_by_country_20.columns = ['Country', 'Average_GDP']
    gdp_by_country_30 = (df[df['Year'] > 1990].groupby('Country')['Gross Domestic Product (GDP)'].mean().reset_index())
    gdp_by_country_30.columns = ['Country', 'Average_GDP']

    fig_allmean_5 = px.choropleth(
        gdp_by_country_5,
        locations="Country",
        locationmode="country names",
        color="Average_GDP",
        hover_name="Country",
        color_continuous_scale="earth",
        title="各国近5年总体平均GDP热力图"
    )
    fig_allmean_5.update_layout(geo=dict(showframe=False, showcoastlines=True))
    fig_allmean_5.show()

    fig_allmean_10 = px.choropleth(
        gdp_by_country_10,
        locations="Country",
        locationmode="country names",
        color="Average_GDP",
        hover_name="Country",
        color_continuous_scale="earth",
        title="各国近10年总体平均GDP热力图"
    )
    fig_allmean_10.update_layout(geo=dict(showframe=False, showcoastlines=True))
    fig_allmean_10.show()

    fig_allmean_20 = px.choropleth(
        gdp_by_country_20,
        locations="Country",
        locationmode="country names",
        color="Average_GDP",
        hover_name="Country",
        color_continuous_scale="earth",
        title="各国近20年总体平均GDP热力图"
    )
    fig_allmean_20.update_layout(geo=dict(showframe=False, showcoastlines=True))
    fig_allmean_20.show()

    fig_allmean_30 = px.choropleth(
        gdp_by_country_30,
        locations="Country",
        locationmode="country names",
        color="Average_GDP",
        hover_name="Country",
        color_continuous_scale="earth",
        title="各国近30年总体平均GDP热力图"
    )
    fig_allmean_30.update_layout(geo=dict(showframe=False, showcoastlines=True))
    fig_allmean_30.show()

    gdp_by_country = df.groupby('Country')['Gross Domestic Product (GDP)'].mean().reset_index()
    gdp_by_country.columns = ['Country', 'Average_GDP']

    fig_allmean = px.choropleth(
        gdp_by_country,
        locations="Country",
        locationmode="country names",
        color="Average_GDP",
        hover_name="Country",
        color_continuous_scale="earth",
        title="各国近50年总体平均GDP热力图"
    )
    fig_allmean.update_layout(geo=dict(showframe=False, showcoastlines=True))
    fig_allmean.show()

    # 人均 GDP
    population_mean_df = gdp_by_country.copy()
    population_mean_df['population'] = df.groupby('Country')['Population'].mean().values
    population_mean_df['population_ave_gdp'] = population_mean_df['Average_GDP'] / population_mean_df['population']

    fig_popmean = px.choropleth(
        population_mean_df,
        locations="Country",
        locationmode="country names",
        color="population_ave_gdp",
        hover_name="Country",
        color_continuous_scale="emrld",
        title="各国近50年总体人均GDP热力图"
    )
    fig_popmean.update_layout(geo=dict(showframe=False, showcoastlines=True))
    fig_popmean.show()

    # 最近一年数据
    latest_year_df = df.sort_values('Year').groupby('Country').tail(1)
    latest_year_df['Country'] = latest_year_df['Country'].str.strip()
    # 最近一年人均 GNI
    fig_gni = px.choropleth(
        latest_year_df,
        locations="Country",
        locationmode="country names",
        color="Per capita GNI",
        hover_name="Country",
        color_continuous_scale="algae",
        title="全球最近一年人均国民总收入（GNI）热力图",
    )
    fig_gni.update_layout(geo=dict(showframe=False, showcoastlines=True))
    fig_gni.show()
    # 最近一年 GDP 总量
    fig_gdp = px.choropleth(
        latest_year_df,
        locations="Country",
        locationmode="country names",
        color="Gross Domestic Product (GDP)",
        hover_name="Country",
        color_continuous_scale="teal",
        title="全球国内生产总值（GDP）总量热力图",
    )
    fig_gdp.update_layout(geo=dict(showframe=False, showcoastlines=True))
    fig_gdp.show()

def load_np_img(image_path):
    input_img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if input_img is None:
        raise FileNotFoundError(f"未找到图像文件: {image_path}")
    input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)
    return input_img

if __name__ == '__main__':
    # hotmap_on_map()
    # img_paths = [f"./fig_tmp_lst/{i}_GDP_Mean.png" for i in [50, 30, 20, 5]]
    # imgs = [load_np_img(p) for p in img_paths]
    imgs = [load_np_img('./fig_tmp_lst/50Ave_GDP.png'), load_np_img('./fig_tmp_lst/Overall_GDP.png')]
    # h, w, c = imgs[0].shape
    # imgs = [cv2.resize(img, (w, h)) for img in imgs]
    # row1 = np.concatenate([imgs[0], imgs[1]], axis=1)
    # row2 = np.concatenate([imgs[2], imgs[3]], axis=1)
    # out = np.concatenate([row1, row2], axis=0)
    out = np.concatenate(imgs, axis=1)
    plt.imsave("./fig_tmp_lst/AveAndOverall_GDP.png", out)