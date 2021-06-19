#!/usr/bin/env python
# coding: utf-8

# # ๏ Partie 2: Model Building

# 2 algorithmes avec au moins 2 paramètres différents (ex: max_depth, n_estimators,….)
# Expliquer en vulgarisant le fonctionnement de vos algorithmes et ses paramètres

# ### Importer nos librairies

# In[1]:


import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# ### Importer notre dataset

# In[2]:


df = pd.read_csv('./data/new_cancer.csv')
df.head()


# In[3]:


# Drop Unnamed column
df = df.drop("Unnamed: 0",axis=1)
list(df.columns)


# ## Choix de nos variables
# 

# ### Lors de l'exploration de notre dataset dans le fichier analyseGraphique.ipynb
# ### nous avons trouvé quelques features plus intéréssantes que d'autres que nous allons utilisé pour créer notre model de prédiction 

# In[4]:



# X = df[["Dust Allergy","OccuPational Hazards","Genetic Risk","Obesity","Coughing of Blood"]]
y = df["Level"]
X = df.drop('Level',axis=1)


# ### Split data

# In[5]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.3, random_state=0)


# ### Étant donné que notre dataset ne contient que 1000 lignes je commence par tester le model LinearSVC

# ## Créer notre model LinearSVC

# In[6]:


from sklearn.svm import LinearSVC


# In[7]:


# Instancier la class LinearSVC
clf = LinearSVC()

# Entrainer notre modéle
clf.fit(X_train, y_train)


# In[8]:


clf.score(X_test,y_test)


# ### On remarque ici que notre model à un excelent score 
# ### Je vais donc faire de preprocessing à mon dataset pour voir si je peux amélioré le score 

# In[9]:


from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
clf_sc = make_pipeline(StandardScaler(),LinearSVC())
clf_sc.fit(X_train,y_train)
clf_sc.score(X_test,y_test)


# #### Après avoir fait le préprocessing avec StanderScaler j'obtiens un score de 99.66%, Excellente Résultats  

# ### Tester un autre model pour voir si nous obtiendrons un meilleur score
# ### Étant donné que mon dataset ne contient pas du text data Je vais tester le KNeighborsClassifier au lieu de Naive Bayes

# In[19]:


from sklearn.neighbors import KNeighborsClassifier
kn_clf = KNeighborsClassifier(n_neighbors=8)
kn_clf.fit(X_train,y_train)
print("Le score est de {} avec le KNeighborsClassifier model".format(kn_clf.score(X_test,y_test)))


# ### Ici notre modéle ridge regression obtiens un meilleur score avec 91% tandis que le SVR est très mauvais car les algorithmes de SVM recommande de mettre à l'échelle notre (data scale data) 
# #### https://scikit-learn.org/stable/modules/svm.html#regression

# ### ici vais procéder de la même façon que notre précédent model SGDRegressor 

# In[18]:


svr_sc = make_pipeline(StandardScaler(),SVR(kernel='linear'))
svr_sc.fit(X_train,y_train)
svr_sc.score(X_test,y_test)


# #### c'est impréssionnant j'obtients un score de 99% on scalant mes données 
# #### Je valide donc ce modèle pour déterminé le niveau de cancer chez un patient 
