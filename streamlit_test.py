import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, plot_confusion_matrix

st.set_option('deprecation.showPyplotGlobalUse', False)

st.write("""
# Iris variety 
Web app about *Iris dataset* !  
""")
st.sidebar.info('Sidebar parameters')

# Image of the different iris from url
col_01, col_02, col_03 = st.columns(3)

with col_01:
    st.image(
                "https://fr.wikipedia.org/wiki/Iris_de_Fisher#/media/Fichier:Kosaciec_szczecinkowaty_Iris_setosa.jpg",
                width=250
            )
    st.info('Setosa')
with col_02:
    st.image(
                "https://fr.wikipedia.org/wiki/Iris_de_Fisher#/media/Fichier:Iris_versicolor_3.jpg",
                width=250
            )
    st.info('Versicolor')
with col_03:   
    st.image(
                "https://fr.wikipedia.org/wiki/Iris_de_Fisher#/media/Fichier:Iris_virginica.jpg",
                width=250
            )
    st.info('Virginica')
    

# Dataset
load = load_iris()
df = pd.DataFrame(load['data'],columns=load['feature_names']) 
df['target'] = load['target'] 
list_variety = list(load['target_names'])
df['variety'] = df['target'].apply(lambda x: list_variety[0] if x==0
                                   else (list_variety[1] if x==1
                                   else list_variety[2]))

st.write('## Dataset')
list_variety_all = ['all'] + list_variety
option_iris = st.selectbox(
    'Which variety do you like best?',
     list_variety_all)

if option_iris=='all': 
    st.write('### Dataset', df)
else:
    df_tmp = df[df.variety==option_iris]
    st.write(df_tmp)

col = df.columns

# Data Analysis and visualisation
st.write('## Distribution per class')
f, ax = plt.subplots()
df.variety.value_counts().plot(kind='bar',ax=ax)
st.pyplot(f)

st.write('## Boxplot')

col_11, col_12 = st.columns(2)
with col_11:
    f, ax = plt.subplots()
    sns.boxplot(x="variety", 
            y=col[0], data=df, ax=ax)   
    f.tight_layout(pad=0.5)
    st.pyplot(f)

with col_12:
    f, ax = plt.subplots()
    sns.boxplot(x="variety", 
            y=col[1], data=df, ax=ax)   
    f.tight_layout(pad=0.5)
    st.pyplot(f)

col_13, col_14 = st.columns(2)
with col_13:
    f, ax = plt.subplots()
    sns.boxplot(x="variety", 
            y=col[2], data=df, ax=ax)    
    f.tight_layout(pad=0.5)
    st.pyplot(f)
with col_14:
    f, ax = plt.subplots()
    sns.boxplot(x="variety", 
            y=col[3], data=df, ax=ax)   
    f.tight_layout(pad=0.5)
    st.pyplot(f)


st.write('## Histogram')

col_21, col_22 = st.columns(2)
with col_21:
    f, ax = plt.subplots()
    ax.hist(x=df[col[0]])
    ax.title.set_text(col[0])
    f.tight_layout(pad=0.5)
    st.pyplot(f)

with col_22:
    f, ax = plt.subplots()
    ax.hist(x=df[col[1]])
    ax.title.set_text(col[1])
    f.tight_layout(pad=0.5)
    st.pyplot(f)

col_23, col_24 = st.columns(2)
with col_23:
    f, ax = plt.subplots()
    ax.hist(x=df[col[2]])
    ax.title.set_text(col[2])
    f.tight_layout(pad=0.5)
    st.pyplot(f)
with col_24:
    f, ax = plt.subplots()
    ax.hist(x=df[col[3]])
    ax.title.set_text(col[3])
    f.tight_layout(pad=0.5)
    st.pyplot(f)

st.write('## Histogram per class')
col_31, col_32, col_33 = st.columns(3)

with col_31:
    st.info('Setosa')
with col_32:
    st.info('Versicolor')
with col_33:   
    st.info('Virginica') 
    
def hist_per_class(data,col):
    col_0, col_1, col_2 = st.columns(3)
    with col_0:
        data0 = data[data.target==0]
        f, ax = plt.subplots()
        ax.hist(x=data0[col])
        ax.title.set_text(col)
        f.tight_layout(pad=0.5)
        st.pyplot(f)
    
    with col_1:
        data1 = data[data.target==1]
        f, ax = plt.subplots()
        ax.hist(x=data1[col])
        ax.title.set_text(col)
        f.tight_layout(pad=0.5)
        st.pyplot(f)
        
    with col_2:
        data2 = data[data.target==2]
        f, ax = plt.subplots()
        ax.hist(x=data2[col])
        ax.title.set_text(col)
        f.tight_layout(pad=0.5)
        st.pyplot(f)
 
hist_per_class(df,col[0])
hist_per_class(df,col[1])
hist_per_class(df,col[2])   
hist_per_class(df,col[3])     
  
# Dimension reduction      
st.write('## Dimension reduction')
option_reduct_dim = st.selectbox(
    'Which dimension reduction technic do you like best?',
     ['PCA', 't-SNE'])

st.write('You selected :', option_reduct_dim)

df_num = df[col[:-2]]
df_center_reduced = (df_num - df_num.mean())/df_num.var()

def reduction_dimension(method_name):

    if method_name=='PCA':
        # PCA with 2 components
        n_components = 2
        pca = PCA(n_components=n_components,random_state=42)
        pca_fit = pca.fit(df_center_reduced)
        st.write('var explained :',
              np.sum(pca_fit.explained_variance_ratio_))
        df_2dim = pca.fit_transform(df_center_reduced)
        # Plot result
        f, ax = plt.subplots()
        ax.plot(df_2dim[:50,0],
                    df_2dim[:50,1],'o',color="red",
                    label=list_variety[0])
        ax.plot(df_2dim[50:100:,0],
                    df_2dim[50:100,1],'o',color="blue",
                    label=list_variety[1])
        ax.plot(df_2dim[100:150:,0],
                    df_2dim[100:150,1],'o',color="green",
                    label=list_variety[2])
        ax.legend()
        st.pyplot(f)
    
    
    if method_name=='t-SNE':
        # t-SNE with 2 components
        tsne = TSNE(n_components=2)
        df_2dim = tsne.fit_transform(df_center_reduced)
        # Plot result
        f, ax = plt.subplots()
        ax.plot(df_2dim[:50,0],
                    df_2dim[:50,1],'o',color="red",
                    label=list_variety[0])
        ax.plot(df_2dim[50:100,0],
                    df_2dim[50:100,1],'o',color="blue",
                    label=list_variety[1])
        ax.plot(df_2dim[100:150:,0],
                    df_2dim[100:150,1],'o',color="green",
                    label=list_variety[2])
        ax.legend()
        st.pyplot(f)
    
    return df_2dim

X_2d = reduction_dimension(option_reduct_dim)

# Data target
X = df[col[:-2]]
y = df['target']

use_dim_red = st.sidebar.radio(
    'Do you want to use the dataset obtained with %s ?' % (option_reduct_dim),
     ['No','Yes'])

if use_dim_red=='Yes':
    X = X_2d

# Split training-test set (no validation inthis exercice)
X_train, X_test, y_train , y_test = train_test_split(X, y, random_state=0)

# Model 
classification_method = st.sidebar.selectbox(
    'What classification method do you want to work with?',
     ['Logistic Regression', 'Decision Tree', 'SVC'])

default_method = st.sidebar.radio(
    'Do you want to use the default method?',
     [True,False])

st.write('## Model')
st.write('### You selected :', classification_method)

models = {'Logistic Regression' : LogisticRegression(), 
          'Decision Tree' : DecisionTreeClassifier(), 
          'SVC' : SVC()}

model = models[classification_method]

if not default_method:
    if classification_method=='Logistic Regression':
        with st.expander('Advanced Parameters'):
                col_41, col_42 = st.columns(2)
                with col_41:
                    penalty = st.selectbox('Penalty',['l2','l1','elasticnet','none'])
                    tol = st.number_input('Tolerance (1e-4)',value=1)/10000
                    fit_intercept = st.radio('Intercept',[True,False])
                    class_weight = st.radio('Class weight',[None,'balanced'])
                    solver = st.selectbox('Solver',['lbfgs','newton-cg','liblinear','sag','saga'])
                    multi_class = st.selectbox('Multi class',['auto','ovr','multinomial'])
                    warm_start = st.radio('Warm start',[False,True])
                with col_42:
                    dual = st.radio('Dual or primal formulation',[False,True])
                    C = st.number_input('Inverse regularization strength',0.0,99.0,1.0,0.1)
                    intercept_scaling = st.number_input('Intercept scaling',0.0,99.0,1.0,0.1)
                    random_state = st.radio('Random state',[None,'Custom'])
                    if random_state == 'Custom':
                        random_state = st.number_input('Custom random state',0,99,1,1)
                    max_iter = st.number_input('Maximum iterations',0,100,100,1)
                    verbose = st.number_input('Verbose',0,99,0,1)
                    l1_ratio = st.radio('L1 ratio',[None,'Custom'])
                    if l1_ratio == 'Custom':
                        l1_ratio = st.number_input('Custom l1 ratio',0.0,1.0,1.0,0.01)
                        
        model = LogisticRegression(penalty=penalty,dual=dual, tol=tol,C=C, 
                                   fit_intercept=fit_intercept, intercept_scaling=intercept_scaling,
                                   class_weight=class_weight,random_state=random_state, 
                                   solver=solver, max_iter=max_iter, 
                                   multi_class=multi_class, verbose=verbose,  
                                   warm_start=warm_start, l1_ratio=l1_ratio)
    
    if classification_method=='Decision Tree':
        with st.expander('Advanced Parameters'):
                col_41, col_42 = st.columns(2)
                with col_41:
                    criterion = st.selectbox(['gini','entropy'])
                    max_depth=None
                    min_samples_leaf=1
                    max_features=None
                    max_leaf_nodes=None  
                    min_impurity_split=None
                    ccp_alpha=0.0
                with col_42:
                    splitter = st.selectbox(['best','random']) 
                    min_samples_split=2
                    min_weight_fraction_leaf=0.0
                    random_state = st.radio('Random state',[None,'Custom'])
                    if random_state == 'Custom':
                        random_state = st.number_input('Custom random state',0,99,1,1)
                    min_impurity_decrease=0.0 
                    class_weight=None
                        
        model = DecisionTreeClassifier()
                        
    if classification_method=='SVC':
        with st.expander('Advanced Parameters'):
                col_41, col_42 = st.columns(2)
                with col_41:
                    C = st.number_input('Inverse regularization strength',0.0,99.0,1.0,0.1)
                    degree = st.number_input('Degre',0,99,3,1)             
                    coef0=0.0 
                    probability = st.radio('Probability',[False,True])  
                    cache_size=200, 
                    verbose = st.radio('Verbose',[False,True]) 
                    decision_function_shape = st.selectbox(
                        'Decision function shape', ['ovr','ovo'])
                    random_state = st.radio('Random state',[None,'Custom'])
                    if random_state == 'Custom':
                        random_state = st.number_input('Custom random state',0,99,1,1)
                   
                with col_42:
                    kernel = st.selectbox('Kernel',['rbf', 'linear', 'poly', 'sigmoid',
                                                        'precomputed']) 
                    gamma='scale'
                    shrinking = st.radio('Shrinking',[True,False])
                    tol = st.number_input('Tolerance (1e-3)',value=1)/1000
                    class_weight = None
                    max_iter = st.number_input('Custom random state',-1, 10000,-1,1)
                    break_ties = st.radio('Break Ties',[False,True])
    
        model = SVC()
        

start_training = st.sidebar.radio(
    'Start training ?',
     ['Yes', 'No']) 

if start_training=='Yes':
    # Training
    model_fit = model.fit(X_train, y_train)
    # Prediction
    model_pred_test = model_fit.predict(X_test)
    # Metrics
    accuracy = accuracy_score(model_pred_test, y_test) 
    precision = recall_score(model_pred_test, y_test, average = 'micro') 
    recall = precision_score(model_pred_test, y_test, average = 'micro') 
    f1 = f1_score(model_pred_test, y_test, average = 'micro') 
    
    st.write('### Metrics on test set')
    
    st.write('#### Confusion Matrix')
    normalize_data = st.selectbox(
        'Normalize matrix ?',
         [None, 'true'])   
    f, ax = plt.subplots()
    plot_confusion_matrix(model_fit, X_test, y_test,
                                     display_labels=list_variety,
                                     normalize=normalize_data, ax=ax)
    st.pyplot(f)
    
    col_51, col_52, col_53, col_54 = st.columns(4)
    
    with col_51:
        st.info('Accuracy: **%s**' % (round(accuracy,3)))
    with col_52:
        st.info('Precision: **%s**' % (round(precision,3)))
    with col_53:
        st.info('Recall: **%s**' % (round(recall,3)))
    with col_54:
        st.info('F1 Score: **%s**' % (round(f1,3)))
    
      
    

    

