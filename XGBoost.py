import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import xgboost as xgb
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import joblib

# Carregar os dados
dados = pd.read_csv('pre-processed.csv')

# Dividir os dados em features (X) e rótulos (y)
X = dados['preprocessed_news']
y = dados['label']

# Converter as classes em valores numéricos (0 e 1)
y_encoded = y.map({'fake': 0, 'true': 1})

# Dividir os dados em conjuntos de treino e teste
X_train, X_test, y_train_encoded, y_test_encoded = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Vetorizar os dados de texto
vectorizer = TfidfVectorizer()
X_train_vectorized = vectorizer.fit_transform(X_train)
X_test_vectorized = vectorizer.transform(X_test)

# Salvar o vetorizador
joblib.dump(vectorizer, 'vectorizer.pkl')

# Criar e treinar o modelo XGBoost
modelo = xgb.XGBClassifier()
modelo.fit(X_train_vectorized, y_train_encoded)

# Salvar o modelo
joblib.dump(modelo, 'modelo_xgboost.pkl')

# Fazer previsões
previsoes = modelo.predict(X_test_vectorized)

# Calcular a acurácia do modelo
acuracia = accuracy_score(y_test_encoded, previsoes)
print("Acurácia do modelo XGBoost:", acuracia)

# Calcular outras métricas
relatorio = classification_report(y_test_encoded, previsoes)
print("Relatório de Classificação:")
print(relatorio)

# Calcular a matriz de confusão
matriz_confusao = confusion_matrix(y_test_encoded, previsoes)
print("Matriz de Confusão:")
print(matriz_confusao)

