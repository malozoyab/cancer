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
