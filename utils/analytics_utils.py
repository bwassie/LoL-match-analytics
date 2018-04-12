import pandas as pd
import numpy as np
from sklearn.base import TransformerMixin
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.feature_selection import f_classif
from sklearn.preprocessing import label_binarize
import numpy as np
from scipy.stats import ttest_ind
from scipy.stats import ks_2samp
import matplotlib
import matplotlib.pyplot as plt
plt.style.use("ggplot")

class DataFrameImputer(TransformerMixin):
    def __init__(self):
        """Impute missing values.
        Columns of dtype object are imputed with the most frequent value 
        in column.
        Columns of other types are imputed with mean of column.
        """
    def fit(self, X, y=None):
        srs = [X[c].value_counts().index[0] if X[c].dtypes == np.dtype('O') else X[c].mean() for c in X]
        self.fill = pd.Series(srs,index=X.columns)
        return self
    
    def transform(self, X, y=None):
        return X.fillna(self.fill)
    
def format_table(match_table,to_drop=["patchno","gameid","url","index"]):
    cols = match_table.loc[:,(match_table.isnull().sum()>0)].columns.values
    #print("Following columns have null values:\n")
    #print(match_table[cols].isnull().sum())
    match_table["herald"] = match_table["herald"].fillna(0)
    match_table["heraldtime"] = match_table["heraldtime"].fillna(60)
    data = match_table.copy().drop(to_drop,axis=1)
    data["visionwards"] = pd.to_numeric(data["visionwards"],errors='coerce')
    pre_data = data.apply(pd.to_numeric,errors='ignore')
    data = DataFrameImputer().fit_transform(pre_data)
    catcols = [data.dtypes.index[ii] for ii in range(data.dtypes.shape[0]) if data.dtypes[ii]=='O']
    numcols = [data.dtypes.index[ii] for ii in range(data.dtypes.shape[0]) if data.dtypes[ii]!='O']
    cat_data = pd.get_dummies(data[catcols])
    pre_data[cat_data.columns.values] = cat_data
    return pre_data,pd.concat([cat_data,data[numcols]],axis=1)

def get_final_table(leagues,variable,to_drop, all_matches):
    matches = []
    y = []
    for ii in leagues:
        matches.append(all_matches[ii].drop(variable,axis=1))
        y.append(all_matches[ii][variable])
    data = pd.concat(matches,axis=0)
    y = pd.concat(y,axis=0)
    pre_data,data = format_table(data,to_drop)
    colnames = data.columns.values
    data = StandardScaler().fit_transform(data)
    return pre_data,data, y.values, colnames

class supervised_analysis():
    def __init__(self,data,y,cols):
        self.data = data
        self.y = y
        self.cols = cols
        print("Largest class/Naive accuracy: {:10.4f}".format(max(np.unique(y,return_counts=True)[1])/(y.shape[0])))
        self.y_ml = LabelEncoder().fit_transform(y)
        self.X_train,self.X_test,self.y_train,self.y_test = train_test_split(self.data,self.y_ml,test_size=.4,
                                        stratify=self.y_ml, shuffle=True,random_state=392)
        self.methods = {"Random Forest":RandomForestClassifier(random_state=23),"Adaboost":AdaBoostClassifier(random_state=23),"Logit":LogisticRegression(
                        random_state=23)}
    def fit(self,alg):
        clsf = self.methods[alg]
        scores = cross_val_score(clsf,self.X_train,self.y_train,cv=
                                 StratifiedKFold(10,shuffle=True,random_state=292))
        print("\n{} Cross val mean: {} std: {}".format(alg,scores.mean(),scores.std()))
        clsf.fit(self.X_train,self.y_train)
        print("{} test score: {}".format(alg,clsf.score(self.X_test,self.y_test)))
        y_pred = label_binarize(clsf.predict(self.X_test),np.unique(self.y_test))
        auc = roc_auc_score(label_binarize(self.y_test,np.unique(self.y_test)),y_pred,average='weighted')
        print("{} AUC on test data: {}".format(alg,auc))
        try:
            coef = clsf.coef_[0]
            comp = self.contributions(coef)
        except:
            coef = clsf.feature_importances_
            comp = self.contributions(coef)
        return comp
    
    def anova(self):
        F,pval = f_classif(self.data,self.y)
        #print("Anova F-scores")
        comp = self.contributions({"F-Score":np.round(F,4),"P-value":np.round(pval,4)})
        return comp
    
    def contributions(self,coef):
        #print("FEATURE IMPORTANCES:")
        if type(coef).__name__ == 'ndarray':
            components = pd.Series(np.round(coef,4), index = self.cols)
            ordered = components.abs().sort_values(ascending=False).index
            ordered = components[ordered]
            ordered = ordered/ordered.sum()
            #components = components[ordered]
        else:
            components = pd.DataFrame(coef, index = self.cols)
            ordered = components.iloc[:,0].abs().sort_values(ascending=False).index
            ordered = components.iloc[:,0][ordered]
            ordered = ordered/ordered.sum()
        #print((ordered.head(10)))
        #print("\n")
        return ordered

def makeplot(data,y,ftr,kind):
    target_names = pd.unique(y)
    group = []
    for ii in target_names:
        group.append(data.iloc[y==ii,:])
    #print(len(group))
    fig,ax = plt.subplots(1,2,figsize=(10,4),sharey=True)
    axes = ax.flatten()
    font1 = {'fontsize': 11}
    font2 = {'fontsize': 13}
    if kind == "bar":
        n = 0
        for ii in group:
            ii[ftr].value_counts().plot.bar(ax=axes[n])
            axes[n].set_xticklabels(["0","1"])
            axes[n].set_title(target_names[n],fontdict=font1)
            n+=1
        plt.suptitle("{} for {} vs {}".format(ftr,target_names[0],target_names[1]),fontdict=font2)
        plt.show()
    else:
        n = 0
        for ii in group:
            ii.boxplot(ftr,ax = axes[n])
            axes[n].set_title("{}\n mean: {:10.3f}".format(target_names[n],ii[ftr].mean()),fontdict=font1)
            n+=1
        t,p2 = ttest_ind(group[0][ftr],group[1][ftr],equal_var=False)
        plt.suptitle("{} for {} vs {}\nT-test p-val: {:10.3f}".format(
            ftr,target_names[0],target_names[1],p2),fontdict=font2)
        plt.show()
