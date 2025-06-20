import sklearn.preprocessing as skpre
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import seaborn as sns

if __name__ == "__main__":
    class emp_rule():
        def __init__(self,column:str,df:pd.DataFrame):
            self.column = column
            self.df = df
            self.data_fit()

        def data_fit(self):
            scalar = skpre.StandardScaler()
            scalar.fit(self.df[self.column].to_numpy().reshape(-1,1))
            self.mean = scalar.mean_
            self.std = scalar.scale_
            return scalar

        def data_transform(self,scalar):
            self.z_scores = scalar.transform(self.df[self.column].to_numpy().reshape(-1,1))
            self.z_scores_mean = self.z_scores.mean()
            self.z_scores_std =self.z_scores.std()

        def fit_emp_graph(self):
            std_1 = self.z_scores_mean+1*self.z_scores_std
            std_1_ = self.z_scores_mean-1*self.z_scores_std
            std_2 = self.z_scores_mean+2*self.z_scores_std
            std_2_ = self.z_scores_mean-2*self.z_scores_std 
            std_3 = self.z_scores_mean+3*self.z_scores_std
            std_3_ = self.z_scores_mean-3*self.z_scores_std

            self.per_0 = np.round(((self.z_scores>=std_1_) & (self.z_scores<=std_1)).sum()/self.z_scores.__len__()*100,2)
            self.per_1 = np.round(((self.z_scores>=std_2_) & (self.z_scores<=std_2)).sum()/self.z_scores.__len__()*100,2)
            self.per_2 = np.round(((self.z_scores>=std_3_) & (self.z_scores<=std_3)).sum()/self.z_scores.__len__()*100,2)


        def plot_emp_graph(self,ax,bin_val):
            ax[0].hist(self.z_scores,bins=bin_val,density=True)
            for i in range(-2,3,1):
                x = 0 + i * 1
                y = 1/(self.z_scores_std*(np.sqrt(2*np.pi))*(np.exp((x-self.z_scores_mean)**2/2*self.z_scores_std**2)))
                ax[0].plot([x,x],[0,y],linestyle="--",alpha = 0.5,color = 'r')
                ax[0].annotate(f"{getattr(self,'per_'+str(abs(i)))}%",xy=(x,y),xytext=(x-0.1,y+0.01),fontsize=8,color='r')
            ax[1].hist(self.df[self.column],bins=bin_val,density=True)
            values = np.linspace(self.df[self.column].min(),self.df[self.column].max(),100)
            pd = norm.pdf(values,loc=self.df[self.column].mean(),scale=self.df[self.column].std())
            ax[1].plot(values,pd)
