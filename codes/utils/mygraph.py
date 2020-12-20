import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

path_results = '../results/images/'

# this function receives a dataset with binary target and it will graph a hist of values
def graph_target(data,name="target",figsize=(6,4),title_name=None,color_text="white",save=False,name_file='target_distribution'):
    plt.figure(figsize=figsize)
    total = float(len(data)) # one person per row 
    title_name = "Target distribution"+" of "+str(int(total))+" users" if title_name is None else title_name+" of "+str(int(total))+" users"
    ax = sns.countplot(x=name, data=data) # for Seaborn version 0.7 and more
    for p in ax.patches:
        height = p.get_height()
        ax.text(p.get_x()+p.get_width()/2.,
                height/3,
                '{:.2f}%\n{:d}'.format(100*height/total,height),
                ha="center",color=color_text,fontweight='bold')#fontsize=10
    plt.title(title_name)
    plt.show()
    if save:
        figure = ax.get_figure()    
        figure.savefig(path_results+name_file+'.png',dpi=400, bbox_inches = 'tight')

# plot histograms of train and test to understand the differences between them 
def plot_comp_hist(data1,data2,l_range=[-np.inf,np.inf],labels=['x','y'],title='histogram',bins=20,alpha=0.5):
    x = data1[(data1>=l_range[0])&(data1<l_range[1])]
    y = data2[(data2>=l_range[0])&(data2<l_range[1])]
    plt.hist([x, y],label=labels, bins = bins, alpha=alpha)
    plt.legend(loc='upper right')
    plt.title(title)
    #rcc_train[(rcc_train.saldo>=0.2)&(rcc_train.saldo<3)].saldo.plot.hist(title="Fraud Tranascation <3", alpha=0.5)
    #rcc_train[(rcc_test.saldo>=0.2)&(rcc_test.saldo<3)].saldo.plot.hist(title="Fraud Tranascation <3", alpha=0.5)    