import joblib
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Carregar o modelo treinado
modelo = joblib.load('modelo_xgboost.pkl')

# Carregar o vetorizador
vectorizer = joblib.load('vectorizer.pkl')

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

# Função para prever a notícia
def prever_noticia(noticia):
    # Pré-processar a notícia
    noticia_processada = preprocessar_noticia(noticia)
    # Vetorizar a notícia
    noticia_vetorizada = vectorizer.transform([noticia_processada])
    # Prever a classe
    resultado = modelo.predict(noticia_vetorizada)
    return resultado[0]

# Digitar a notícia
noticia_usuario = input("Digite a notícia: ")

# Prever a classe da notícia
classe_noticia = prever_noticia(noticia_usuario)

# Imprimir o resultado
if classe_noticia == 'fake':
    print("A notícia é falsa.")
else:
    print("A notícia é verdadeira.")
