import matplotlib.pyplot as plt
import seaborn as sns
import joblib

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
'''from sklearn.externals import joblib'''

# Carregar o arquivo CSV
df = pd.read_csv("pre-processed.csv")

# Dividir os dados em conjuntos de treinamento e teste
X = df['preprocessed_news']  # texto das notícias pré-processadas
y = df['label']  # rótulos (fake ou true)

# Dividir os dados em conjuntos de treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Contar o número de exemplos de "fake" e "true" em cada conjunto de dados
train_fake_count = (y_train == "fake").sum()
train_true_count = (y_train == "true").sum()

test_fake_count = (y_test == "fake").sum()
test_true_count = (y_test == "true").sum()

# Imprimir a contagem de exemplos de "fake" e "true" em cada conjunto de dados
print("Contagem de exemplos no conjunto de treino:")
print("Fake:", train_fake_count)
print("True:", train_true_count)

print("\nContagem de exemplos no conjunto de teste:")
print("Fake:", test_fake_count)
print("True:", test_true_count)

# Vetorização do texto usando TfidfVectorizer
tfidf_vectorizer = TfidfVectorizer()
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)

# Salvar o vetorizador TF-IDF
joblib.dump(tfidf_vectorizer, 'tfidf_vectorizer.pkl')

# Treinar o classificador Naive Bayes com os vetores TF-IDF
naive_bayes_classifier_tfidf = MultinomialNB()
naive_bayes_classifier_tfidf.fit(X_train_tfidf, y_train)

# Salvar o modelo treinado com TF-IDF
joblib.dump(naive_bayes_classifier_tfidf, 'modelo_naive_bayes_tfidf.pkl')

# Fazer previsões com TF-IDF
predictions_tfidf = naive_bayes_classifier_tfidf.predict(X_test_tfidf)

# Avaliar o desempenho do modelo com TF-IDF
accuracy_tfidf = accuracy_score(y_test, predictions_tfidf)
print("Accuracy (TF-IDF):", accuracy_tfidf)

report_tfidf = classification_report(y_test, predictions_tfidf)
print("Classification Report (TF-IDF):")
print(report_tfidf)

# Plotar a matriz de confusão com TF-IDF
conf_mat_tfidf = confusion_matrix(y_test, predictions_tfidf)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_mat_tfidf, annot=True, fmt='d', cmap='Blues', cbar=False,
            xticklabels=['True', 'Fake'], yticklabels=['True', 'Fake'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix (TF-IDF)')
plt.show()