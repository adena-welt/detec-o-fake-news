import matplotlib.pyplot as plt
import seaborn as sns
import joblib

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Carregar o arquivo CSV
df = pd.read_csv("pre-processed.csv")

# Dividir os dados em conjuntos de treinamento e teste
X = df['preprocessed_news']  # texto das notícias pré-processadas
y = df['label']  # rótulos (fake ou true)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Contagem de exemplos de cada classe nos dados de treinamento
train_fake_count = (y_train == "fake").sum()
train_true_count = (y_train == "true").sum()

# Contagem de exemplos de cada classe nos dados de teste
test_fake_count = (y_test == "fake").sum()
test_true_count = (y_test == "true").sum()

# Imprimir as contagens
print("Contagem de exemplos no conjunto de treino:")
print("Fake:", train_fake_count)
print("True:", train_true_count)

print("\nContagem de exemplos no conjunto de teste:")
print("Fake:", test_fake_count)
print("True:", test_true_count)

# Vetorização do texto usando CountVectorizer
vectorizer = CountVectorizer()
X_train_vectorized = vectorizer.fit_transform(X_train)
X_test_vectorized = vectorizer.transform(X_test)

# Salvar o vetorizador
joblib.dump(vectorizer, 'vectorizer.pkl')

# Treinar o classificador SVM
svm_classifier = SVC(kernel='linear')  # linear kernel for simplicity, you can try different kernels
svm_classifier.fit(X_train_vectorized, y_train)


# Salvar o modelo treinado
joblib.dump(svm_classifier, 'modelo_svm.pkl')

# Fazer previsões
predictions = svm_classifier.predict(X_test_vectorized)

# Avaliar o desempenho do modelo
accuracy = accuracy_score(y_test, predictions)
print("Accuracy:", accuracy)

report = classification_report(y_test, predictions)
print("Classification Report:")
print(report)

# Plotar a matriz de confusão
conf_mat = confusion_matrix(y_test, predictions)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_mat, annot=True, fmt='d', cmap='Blues', cbar=False,
            xticklabels=['True', 'Fake'], yticklabels=['True', 'Fake'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()
