import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import IsolationForest
from sklearn.covariance import EllipticEnvelope
from sklearn.decomposition import PCA
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler


class Functions():
    # Zscore outliers detection
    def zscore_outliers(df: pd.DataFrame, zthreshold: int, excluded=False) -> pd.DataFrame:
        if excluded: 
            o_df = df
            df = df.loc[:, df.columns.difference(excluded)]
            
        z = np.abs(stats.zscore(df))
        no_out = df[z < zthreshold].dropna()
        
        if excluded:
            for el in excluded:
                no_out[el] = o_df[el]

        return no_out
    
    # Inter Quartile Range outliers
    def iqr_outliers(df: pd.DataFrame, excluded=False) -> pd.DataFrame:
        if excluded: 
            o_df = df
            df = df.loc[:, df.columns.difference(excluded)]
            
        q1 = df.quantile(0.25)
        q3 = df.quantile(0.75)
        iqr = q3 - q1
        no_out = df[df>(q1-1.5*iqr)][df<(q3+1.5*iqr)].dropna()

        if excluded:
            for el in excluded:
                no_out[el] = o_df[el]
        
        return no_out
    
    # Isolation Forest Classifier for outliers detection
    def isoforest_outliers(df: pd.DataFrame, excluded=False) -> pd.DataFrame:
        if excluded: 
            o_df = df
            df = df.loc[:, df.columns.difference(excluded)]
        
        isoforest = IsolationForest().fit(df)
        winedb_isoforest = df.copy()
        winedb_isoforest['scores'] = isoforest.decision_function(df)
        winedb_isoforest['anomaly'] = isoforest.predict(df)
        no_out = winedb_isoforest[winedb_isoforest['anomaly']==1].loc[:, winedb_isoforest.columns.difference(['scores', 'anomaly'])]
        
        if excluded:
            for el in excluded:
                no_out[el] = o_df[el]
                
        return no_out
    
    # Elliptic Enveloper for gaussian distributed data
    def ellipticenvelope_outliers(df: pd.DataFrame, excluded=False) -> pd.DataFrame:
        if excluded: 
            o_df = df
            df = df.loc[:, df.columns.difference(excluded)]
        
        oneclasssvm = EllipticEnvelope().fit(df)
        winedb_isoforest = df.copy()
        winedb_isoforest['scores'] = oneclasssvm.decision_function(df)
        winedb_isoforest['anomaly'] = oneclasssvm.predict(df)
        no_out = winedb_isoforest[winedb_isoforest['anomaly']==1].loc[:, winedb_isoforest.columns.difference(['scores', 'anomaly'])]
        
        if excluded:
            for el in excluded:
                no_out[el] = o_df[el]
                
        return no_out
    
    # MinMax Normalization
    def minmax_norm(df: pd.DataFrame, excluded=False) -> pd.DataFrame:
        if excluded: 
            o_df = df
            df = df.loc[:, df.columns.difference(excluded)]
        
        normdf=(df-df.min())/(df.max()-df.min())
        
        if excluded:
            for el in excluded:
                normdf[el] = o_df[el]
                
        return normdf
    
    # Standard Normalization
    def standard_norm(df: pd.DataFrame, excluded=False) -> pd.DataFrame:
        if excluded: 
            o_df = df
            df = df.loc[:, df.columns.difference(excluded)]
        
        normdf=(df-df.mean())/df.std()
        
        if excluded:
            for el in excluded:
                normdf[el] = o_df[el]
                
        return normdf
    
    # One Hot Encoder for categorical features
    def one_hot_encode(df:pd.DataFrame, target_col) -> pd.DataFrame:
        df = pd.concat([df, pd.get_dummies(df[target_col], prefix='target_col_')], axis=1)
        df = df.loc[:, df.columns.difference([target_col])]
        return df
    
    # Practical Components Analysis for dimensionality reduction
    def pca(df: pd.DataFrame, excluded=False, n_components='mle') -> pd.DataFrame:
        if excluded: 
            o_df = df
            df = df.loc[:, df.columns.difference(excluded)]
        
        pca = PCA(n_components=n_components)
        df = pd.DataFrame(pca.fit_transform(df))
        
        if excluded:
            for el in excluded:
                df[el] = o_df[el]
                
        return df
    
    # Synthetic Minority Over Sampling 
    def smote(df:pd.DataFrame, target_col='type') -> pd.DataFrame:
        x, y = df.loc[:, df.columns != target_col], df[target_col]
        smoted = SMOTE().fit_resample(x, y)
        
        return pd.concat(smoted, axis=1)
    
    # Random Undersampling
    def random_undersample(df:pd.DataFrame, target_col='type') -> pd.DataFrame:
        x, y = df.loc[:, df.columns != target_col], df[target_col]
        smoted = RandomUnderSampler().fit_resample(x, y)
        
        return pd.concat(smoted, axis=1)