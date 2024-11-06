
# Carregar os dados
#dados = pd.read_csv('pre-processed.csv')

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from xgboost import XGBClassifier
import joblib

# Carregar o arquivo CSV
#df = pd.read_csv('pre-processed.csv')
df = pd.read_csv("https://docs.google.com/spreadsheets/d/e/2PACX-1vQHXeGnay1fBg89k_lUubL_3WsML0F7lrjDTF96jMUGaFDXGjR9Y_ca7g8cjjG4XzHSZoJo7bFp1ZWF/pub?gid=1447023203&single=true&output=csv")

# Mapear rótulos para valores binários
df['label'] = df['label'].map({'TRUE': 1, 'fake': 0})

# Dividir os dados em conjuntos de treinamento e teste
X = df['preprocessed_news']  # texto das notícias pré-processadas
y = df['label']  # rótulos (1 para 'TRUE', 0 para 'fake')

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Vetorização usando CountVectorizer (Bag of Words - BoW)
count_vectorizer = CountVectorizer()
X_train_count = count_vectorizer.fit_transform(X_train)
X_test_count = count_vectorizer.transform(X_test)

# Definir o classificador XGBoost sem o parâmetro 'use_label_encoder'
xgb_classifier = XGBClassifier(eval_metric='logloss')

# Realizar Cross-validation com CountVectorizer e várias métricas
scoring = ['accuracy', 'precision', 'recall', 'f1']
count_cv_results = cross_validate(xgb_classifier, X_train_count, y_train, cv=5, scoring=scoring)

# Imprimir métricas de cada fold e suas médias
print("\nCross-validation com CountVectorizer (métricas por fold):")
print("Accuracy: ", count_cv_results['test_accuracy'])
print("Precision: ", count_cv_results['test_precision'])
print("Recall: ", count_cv_results['test_recall'])
print("F1 Score: ", count_cv_results['test_f1'])

print("\nMédias das métricas:")
print(f"Mean Accuracy: {count_cv_results['test_accuracy'].mean():.2f}")
print(f"Mean Precision: {count_cv_results['test_precision'].mean():.2f}")
print(f"Mean Recall: {count_cv_results['test_recall'].mean():.2f}")
print(f"Mean F1 Score: {count_cv_results['test_f1'].mean():.2f}")

# Treinar o modelo com CountVectorizer e avaliar no conjunto de teste
xgb_classifier.fit(X_train_count, y_train)
predictions_count = xgb_classifier.predict(X_test_count)

# Salvar o modelo e o vectorizer CountVectorizer
joblib.dump(xgb_classifier, "xgb_classifier_count.pkl")
joblib.dump(count_vectorizer, "count_vectorizer.pkl")

# Calcular métricas no conjunto de teste
accuracy_count = accuracy_score(y_test, predictions_count)
precision_count = precision_score(y_test, predictions_count)
recall_count = recall_score(y_test, predictions_count)
f1_count = f1_score(y_test, predictions_count)

print("\nResultados no conjunto de teste (CountVectorizer):")
print(f"Accuracy: {accuracy_count:.2f}")
print(f"Precision: {precision_count:.2f}")
print(f"Recall: {recall_count:.2f}")
print(f"F1 Score: {f1_count:.2f}")

# Plotar a matriz de confusão para CountVectorizer
conf_mat_count = confusion_matrix(y_test, predictions_count)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_mat_count, annot=True, fmt='d', cmap='Blues', cbar=False,
            xticklabels=['True', 'Fake'], yticklabels=['True', 'Fake'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix (CountVectorizer)')
plt.show()

# Vetorização usando TfidfVectorizer
tfidf_vectorizer = TfidfVectorizer()
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)

# Realizar Cross-validation com TfidfVectorizer e várias métricas
tfidf_cv_results = cross_validate(xgb_classifier, X_train_tfidf, y_train, cv=5, scoring=scoring)

# Imprimir métricas de cada fold e suas médias
print("\nCross-validation com TfidfVectorizer (métricas por fold):")
print("Accuracy: ", tfidf_cv_results['test_accuracy'])
print("Precision: ", tfidf_cv_results['test_precision'])
print("Recall: ", tfidf_cv_results['test_recall'])
print("F1 Score: ", tfidf_cv_results['test_f1'])

print("\nMédias das métricas:")
print(f"Mean Accuracy: {tfidf_cv_results['test_accuracy'].mean():.2f}")
print(f"Mean Precision: {tfidf_cv_results['test_precision'].mean():.2f}")
print(f"Mean Recall: {tfidf_cv_results['test_recall'].mean():.2f}")
print(f"Mean F1 Score: {tfidf_cv_results['test_f1'].mean():.2f}")

# Treinar o modelo com TfidfVectorizer e avaliar no conjunto de teste
xgb_classifier.fit(X_train_tfidf, y_train)
predictions_tfidf = xgb_classifier.predict(X_test_tfidf)

# Salvar o modelo e o vectorizer TfidfVectorizer
joblib.dump(xgb_classifier, "xgb_classifier_tfidf.pkl")
joblib.dump(tfidf_vectorizer, "tfidf_vectorizer.pkl")

# Calcular métricas no conjunto de teste
accuracy_tfidf = accuracy_score(y_test, predictions_tfidf)
precision_tfidf = precision_score(y_test, predictions_tfidf)
recall_tfidf = recall_score(y_test, predictions_tfidf)
f1_tfidf = f1_score(y_test, predictions_tfidf)

print("\nResultados no conjunto de teste (TfidfVectorizer):")
print(f"Accuracy: {accuracy_tfidf:.2f}")
print(f"Precision: {precision_tfidf:.2f}")
print(f"Recall: {recall_tfidf:.2f}")
print(f"F1 Score: {f1_tfidf:.2f}")

# Plotar a matriz de confusão para TfidfVectorizer
conf_mat_tfidf = confusion_matrix(y_test, predictions_tfidf)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_mat_tfidf, annot=True, fmt='d', cmap='Blues', cbar=False,
            xticklabels=['True', 'Fake'], yticklabels=['True', 'Fake'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix (TfidfVectorizer)')
plt.show()