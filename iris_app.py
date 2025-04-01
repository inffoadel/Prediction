import streamlit as st
import pandas as pd
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier

# Titre de l'application
st.write('''
# App Simple pour la prévision des fleurs d'Iris
Cette application prédit la catégorie des fleurs d'Iris
''')

# Sidebar pour les paramètres d'entrée
st.sidebar.header("Les paramètres d'entrée")

def user_input():
    sepal_length = st.sidebar.slider('La longueur du Sépale', 4.3, 7.9, 5.3)
    sepal_width = st.sidebar.slider('La largeur du Sépale', 2.0, 4.4, 3.3)
    petal_length = st.sidebar.slider('La longueur du Pétale', 1.0, 6.9, 2.3)
    petal_width = st.sidebar.slider('La largeur du Pétale', 0.1, 2.5, 1.3)
    data = {'sepal_length': sepal_length,
            'sepal_width': sepal_width,
            'petal_length': petal_length,
            'petal_width': petal_width}
    fleur_parametres = pd.DataFrame(data, index=[0])
    return fleur_parametres

df = user_input()

# Correspondance des colonnes avec les noms d'origine
iris = datasets.load_iris()
df.columns = iris.feature_names  # Harmonisation des noms de colonnes

# Affichage des paramètres d'entrée
st.subheader('Les paramètres fournis pour la fleur')
st.write(df)

# Entraînement du modèle
iris_df = pd.DataFrame(iris.data, columns=iris.feature_names)
clf = RandomForestClassifier()
clf.fit(iris_df, iris.target)

# Prédiction et probabilités
prediction = clf.predict(df)
prediction_proba = clf.predict_proba(df)

# Affichage des résultats
st.subheader("La catégorie prédite de la fleur d'Iris est:")
st.write(iris.target_names[prediction][0])

st.subheader("Probabilités associées à chaque catégorie:")
st.write(pd.DataFrame(prediction_proba, columns=iris.target_names))