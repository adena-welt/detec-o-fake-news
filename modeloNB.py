import joblib
import re
from sklearn.feature_extraction.text import CountVectorizer

'''import nltk
nltk.download('stopwords')
import nltk
nltk.download('punkt')'''

from nltk.corpus import stopwords
stop_words = set(stopwords.words('portuguese'))

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Carregar o modelo treinado
modelo_naive_bayes = joblib.load('modelo_naive_bayes_count.pkl')

# Carregar o vetorizador usado durante o treinamento
vectorizer = joblib.load('count_vectorizer.pkl')

import re

def remover_acentos(texto):
    # Mapear caracteres acentuados para seus equivalentes não acentuados
    mapa_acentos = {
        'á': 'a', 'à': 'a', 'ã': 'a', 'â': 'a',
        'é': 'e', 'ê': 'e', 'í': 'i', 'ó': 'o',
        'õ': 'o', 'ô': 'o', 'ú': 'u', 'ç': 'c',
        'Á': 'A', 'À': 'A', 'Ã': 'A', 'Â': 'A',
        'É': 'E', 'Ê': 'E', 'Í': 'I', 'Ó': 'O',
        'Õ': 'O', 'Ô': 'O', 'Ú': 'U', 'Ç': 'C'
    }
    # Substituir caracteres acentuados por seus equivalentes não acentuados
    return ''.join(mapa_acentos.get(char, char) for char in texto)

# Função para pré-processar a notícia
def preprocessar_noticia(noticia):
    # Converter para minúsculas
    noticia = noticia.lower()
    # Remover caracteres especiais, exceto letras com acento e números
    noticia = re.sub(r'[^a-zA-Záàãâéêíóõôúç\s\d]', '', noticia)
    # Remover acentos
    noticia = remover_acentos(noticia)
    # Remover números
    noticia = re.sub(r'\b\d+\b', '', noticia)
    # Tokenizar a notícia
    tokens = word_tokenize(noticia)
    # Remover stopwords
    stop_words = set(stopwords.words('portuguese'))
    tokens = [word for word in tokens if word not in stop_words]
    # Juntar tokens de volta em uma string
    noticia_processada = ' '.join(tokens)
    return noticia_processada

def prever_noticia(noticia):
    # Pré-processar a notícia
    noticia_processada = preprocessar_noticia(noticia)
    # Vetorizar a notícia
    noticia_vetorizada = vectorizer.transform([noticia_processada])
    # Fazer a previsão usando o modelo treinado
    previsao = modelo_naive_bayes.predict(noticia_vetorizada)
    # Imprimir o texto pré-processado da notícia
    '''print("\n\nTexto pré-processado da notícia:")
    print(noticia_processada + "\n\n")'''
    # Retornar o resultado da previsão
    if previsao[0] == 'fake':
        return "A notícia é falsa."
    else:
        return "A notícia é verdadeira."

# Exemplo de uso
noticia = input("\nDigite a notícia: ")
resultado = prever_noticia(noticia)
print("\n")
print(resultado)

