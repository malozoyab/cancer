# %%
import pandas as pd
from pandas import read_csv
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from keras.models import Sequential
from keras.optimizers import Adam
from keras.layers.core import Dense
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import warnings
warnings.filterwarnings('ignore')

# %%
df_cancer = pd.read_csv(r"\Users\Miguel Angel\Documents\UAX\infraestructuras y servicios en la nube\practica 2\cancer_dataset.csv")

# %%
df_cancer.info()

# %%
df_cancer.describe()

# %% [markdown]
# Eliminamos la columna de id ya que no es un valor para estandarizar y la ultima columna que no existe y comprobamos que no
# hay mucho sesgo en la variable diagnosis

# %%
df_cancer2=df_cancer.drop(['Unnamed: 32',"id"], axis=1)
df_cancer2["diagnosis"].value_counts()

# %%
scaler=StandardScaler()
df = df_cancer2.drop(['diagnosis'], axis=1) # quito la variable dependiente
X_scaled=scaler.fit_transform(df)#escalo los datos y los normalizo

# %% [markdown]
# Instanciamos objeto PCA y aplicamos, obtenemos los componentes principales
# y convertimos nuestros datos con las nuevas dimensiones de PCA

# %%
pca=PCA(n_components=5)

X_pca = pca.fit_transform(df)

# %%
print("shape of X_pca", X_pca.shape)
expl = pca.explained_variance_ratio_
print(expl)
print('suma:',sum(expl[0:2]))

# %% [markdown]
# Vemos que con 2 componentes tenemos casi el 99,9% de varianza explicada.
# 
# Graficamos el acumulado de varianza explicada en las nuevas dimensiones

# %%
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('numero de componentes')
plt.ylabel('varianza acumulada')
plt.show()

# %%
X= X_pca
y=df_cancer2["diagnosis"]

# %% [markdown]
# Dividimos el conjunto en test y entrenamiento

# %%
X_train, X_test, y_train, y_test = train_test_split(
                                        X,
                                        y,
                                        train_size   = 0.8,
                                        random_state = 1234,
                                        shuffle      = True)

# %% [markdown]
# Creamos nuestro modelo.

# %%
model = MLPClassifier(hidden_layer_sizes=(50,), activation="relu", solver="adam", random_state=42)

# %% [markdown]
# Entrenamos el modelo

# %%
model.fit(X_train, y_train)

# %% [markdown]
# Predecimos con el conjunto de test

# %%
y_pred = model.predict(X_test)

# %% [markdown]
# Los siguiente datos son las conparaciones entre el valor real y el valor predicho.

# %%
accuracy = accuracy_score(y_test, y_pred)
print("Exactitud: {:.2f}".format(accuracy))

# %%
print(classification_report(y_test, y_pred))

# %% [markdown]
# Calculamos la matriz de confusión y la graficamos

# %%
cm = confusion_matrix(y_test, y_pred)

sns.set()
plt.figure(figsize=(6, 6))
sns.heatmap(cm, annot=True, fmt=".0f", linewidths=.5, square=True, cmap='Blues')
plt.ylabel('Etiquetas verdaderas')
plt.xlabel('Etiquetas predichas')
plt.title('Matriz de confusión')
plt.show()

# %% [markdown]
# Como podemos comprobar tenemos una prediccion de un 95%, debido a que tenemos un mayo numero de datos de cancer benigno, los clasifica mejor.


