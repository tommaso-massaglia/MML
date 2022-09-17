import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

class WineDb():
    def __init__(self) -> None:
        self.df_red = pd.read_csv('modules/winequality-red.csv', sep=';')
        self.df_white = pd.read_csv('modules/winequality-white.csv', sep=';')
        self.df_red['type'] = 1
        self.df_white['type'] = -1
        self.df = pd.concat([self.df_red, self.df_white], ignore_index=True)
        
    def __call__(self, type = 'both') -> pd.DataFrame:
        if type == 'red':
            return self.df_red
        if type == 'white':
            return self.df_white
        
        return self.df.copy()
    
    def boxplots(self, df=False):
        
        if not isinstance(df, pd.DataFrame):
            df = self.df
            
        f, ax = plt.subplots(3,4, figsize=(20, 12))

        xind,yind = 0, 0

        for colname in df.columns:
            if colname != 'type':
                sns.boxplot(df, y=colname, x='type', palette='tab10', ax = ax[yind,xind])
                if xind < 3: 
                    xind+=1
                else: 
                    xind=0
                    yind+=1    

        for a in ax:
            for a1 in a:
                a1.set_xticklabels(['white', 'red'])
        
        return f