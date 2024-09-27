import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd 
import colorsys
import scipy
import numpy as np 

method_colors = {
    'Ours': '#1C6090',
    'pTS': '#FF7F0E',
    'qEI': '#39A039',
    'UCB': '#BB2829',
    'Greedy': '#8E5BBB',
    'random_10k': '#9A9A96'
}

method_styles = {
    'Ours': (1),
    'pTS': (1, 0.2),
    'qEI': (0.2, 0.2),
    'UCB': (2, 1),
    'Greedy': (4, 1, 2, 1)
}

it_colors = ['#3491C1', '#7D2AC1', '#B9305C', '#DC5501', '#DE9A00', '#377501', '#B4B5B4']

def set_style():
    """set_style"""
    mpl.rcParams["pdf.fonttype"] = 42
    mpl.rcParams["font.family"] = "sans-serif"
    sns.set(context="paper", style="ticks") 
    mpl.rcParams["text.color"] = "black"
    mpl.rcParams["axes.labelcolor"] = "black"
    mpl.rcParams["axes.edgecolor"] = "black"
    mpl.rcParams["axes.labelcolor"] = "black"
    mpl.rcParams["xtick.color"] = "black"
    mpl.rcParams["ytick.color"] = "black"
    mpl.rcParams["xtick.major.size"] = 2.5
    mpl.rcParams["ytick.major.size"] = 2.5

    mpl.rcParams["xtick.major.width"] = 0.45
    mpl.rcParams["ytick.major.width"] = 0.45

    mpl.rcParams["axes.edgecolor"] = "black"
    mpl.rcParams["axes.linewidth"] = 0.45
    mpl.rcParams["font.size"] = 8
    mpl.rcParams["axes.labelsize"] = 8
    mpl.rcParams["axes.titlesize"] = 8
    mpl.rcParams["figure.titlesize"] = 8
    mpl.rcParams["figure.titlesize"] = 8
    mpl.rcParams["legend.fontsize"] = 7
    mpl.rcParams["legend.title_fontsize"] = 8
    mpl.rcParams["xtick.labelsize"] = 7
    mpl.rcParams["ytick.labelsize"] = 7
    mpl.rcParams['figure.dpi'] = 300

    mpl.rcParams['legend.frameon'] = False
    mpl.rcParams['legend.fancybox'] = False
    mpl.rcParams['legend.facecolor'] = "none"

    mpl.rcParams['hatch.linewidth'] = 0.5  # previous pdf hatch linewidth

def set_size(w, h, ax=None):
    """w, h: width, height in inches
    Resize the axis to have exactly these dimensions
    """
    if not ax:
        ax = plt.gca()
    l = ax.figure.subplotpars.left
    r = ax.figure.subplotpars.right
    t = ax.figure.subplotpars.top
    b = ax.figure.subplotpars.bottom
    figw = float(w) / (r - l)
    figh = float(h) / (t - b)
    ax.figure.set_size_inches(figw, figh)

def scale_lightness(rgb, scale_l):
    # convert rgb to hls
    h, l, s = colorsys.rgb_to_hls(*rgb)
    # manipulate h, l, s values and return as rgb
    return colorsys.hls_to_rgb(h, min(1, l * scale_l), s = s)

def make_color_darker(scale, color: str): 
    rgb = mpl.colors.ColorConverter.to_rgb(color)
    return scale_lightness(rgb, scale)

def df_to_latex(data: pd.DataFrame):
    latex_data = []
    for method in data.Method.unique(): 
        df_method = data.loc[data.Method == method]
        for iter in df_method.Iteration.unique(): 
            df_method_iter = df_method.loc[df_method.Iteration == iter]
            df_method_iter = df_method_iter.drop('Top 1 ave', axis=1)
            stor = {}
            for col in df_method_iter.columns: 
                if 'Top' in col or 'top' in col: 
                    mean = np.mean(df_method_iter[f'{col}'])
                    se = scipy.stats.sem(df_method_iter[f'{col}'])
                    stor[col] = f'{mean:0.2f} $\pm$ {se:0.2f}'
            latex_data.append({**{
                'Method': method, 
                'Iteration': iter, 
            },**stor})
    latex_df = pd.DataFrame(latex_data).sort_values(by=['Method', 'Iteration'])
    return latex_df.to_latex(escape=False, index=False, multicolumn_format='c')
