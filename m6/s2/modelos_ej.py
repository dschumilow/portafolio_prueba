from re import A
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold, LeaveOneOut, cross_val_score #divir los datos, KFOLD validacion cruzada
from sklearn.tree import DecisionTreeClassifier #algoritmo a usar arbol de desicion
from sklearn.metrics import accuracy_score #calcular la presicion del modelo, % de las presiciones correctas

X = np.array([[5,80],[7,85],[6,90],[8,75],[7,70],
              [9,95],[10,90],[4,60],[6,65],[8,88],
              [5,50],[7,78],[6,82],[9,92],[10,85]])

#1 APROBADO, 0 REPROBADO 

y = np.array([1,1,1,1,0,
              1,1,0,0,1,
              0,1,1,1,1])

df = pd.DataFrame(X, columns=['Horas_estudio','Asistencia'])
df['Nota_final']=y

print(df)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42) #random state porque la division de los datos es aleatoria
print('\n Dimenciones de los conjuntos de datos\n')
print(f'X_train (Entrenamiento caracteristicas): {X_train.shape}')
print(f'y_train (Entrenamiento - etiqueta): {y_train.shape}')
print(f'X_test (Prueba caracteristicas) : {X_test.shape}')
print(f'y_test (Prueba - etiqueta): {y_test.shape}')

modelo = DecisionTreeClassifier()
modelo.fit(X_train, y_train)

y_pred = modelo.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f'Precisión del modelo: {accuracy * 100:.2f}%\n')

kf = KFold(n_splits=5, shuffle=True, random_state=42)
accuracies_kfold = []

#Bucle es el corazon del kfold
for fold, (train_index, test_index) in enumerate(kf.split(X)):
    X_train_fold, X_test_fold = X[train_index], X[test_index]
    y_train_fold, y_test_fold = y[train_index], y[test_index]

    model_kfold = DecisionTreeClassifier(random_state=42)
    model_kfold.fit(X_train_fold, y_train_fold)
    y_pred_kfold = model_kfold.predict(X_test_fold)

    accuracy_kfold = accuracy_score(y_test_fold, y_pred_kfold)
    accuracies_kfold.append(accuracy_kfold)
    print(f'Exactitud del modelo en el Fold {fold + 1}: Precisión = {accuracy_kfold * 100:.2f}%')

media_accuracies_kfold = np.mean(accuracies_kfold)
std_accuracies_kfold = np.std(accuracies_kfold)

print(f'Precisión media del modelo con K-Fold: Precisión = {media_accuracies_kfold * 100:.2f}%')
print(f'Desviación estándar de las precisiónes del modelo con K-Fold: Precisión = {std_accuracies_kfold * 100:.2f}%\n')

loo=LeaveOneOut()
accuracies_loo=[]

for i,(train_index, test_index) in enumerate(loo.split(X)):
    X_train_loo, X_test_loo = X[train_index], X[test_index]
    y_train_loo, y_test_loo = y[train_index], y[test_index]

    model_loo=DecisionTreeClassifier(random_state=42)
    model_loo.fit(X_train_loo, y_train_loo)
    y_pred_loo=model_loo.predict(X_test_loo)

    accuracy_loo=accuracy_score(y_test_loo, y_pred_loo)
    accuracies_loo.append(accuracy_loo)
    print(f'Exactitud del modelo en el Fold {i + 1}: Precisión = {accuracy_loo * 100:.2f}%')

media_accuracies_loo=np.mean(accuracies_loo)
std_accuracies_loo=np.std(accuracies_loo)

print(f'\nPrecisión media del modelo con Leave One Out: Precisión = {media_accuracies_loo * 100:.2f}%')
print(f'Desviación estándar de las precisiónes del modelo con Leave One Out: Precisión = {std_accuracies_loo * 100:.2f}%\n')

#RESUMEN DE LOS METODOS ANTERIORES
kfold_scores = cross_val_score(DecisionTreeClassifier(random_state=42), X, y, cv=kf, scoring='accuracy')
print(f'Precisión por subconjunto (KFold_cross_val_score) : {kfold_scores}')
print(f'Media de la precisión por subconjunto (KFold_cross_val_score) : {kfold_scores.mean():.2f}')
print(f'Desviacion estandar de la precisión por subconjunto (KFold_cross_val_score) : {kfold_scores.std():.2f}\n')

loo_scores = cross_val_score(DecisionTreeClassifier(random_state=42), X, y, cv=loo, scoring='accuracy')
print(f'Precisión por subconjunto (LeaveOneOut_cross_val_score) : {loo_scores}')
print(f'Media de la precisión por subconjunto (LeaveOneOut_cross_val_score) : {loo_scores.mean():.2f}')
print(f'Desviacion estandar de la precisión por subconjunto (LeaveOneOut_cross_val_score) : {loo_scores.std():.2f}')

