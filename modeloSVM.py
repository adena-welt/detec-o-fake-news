import joblib
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Carregar o modelo treinado
modelo = joblib.load('SMV/modelo_svm.pkl')

# Carregar o vetorizador usado durante o treinamento
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

def classificar_noticia(noticia):
    # Pré-processar a notícia
    noticia_processada = preprocessar_noticia(noticia)
    # Vetorizar a notícia
    noticia_vetorizada = vectorizer.transform([noticia_processada])
    # Fazer a previsão usando o modelo treinado
    previsao = modelo.predict(noticia_vetorizada)
    # Retornar a classe prevista
    return previsao[0]

# Exemplo de uso
noticia = input("Digite a notícia que você gostaria de classificar: ")
classe = classificar_noticia(noticia)
print("A notícia é", classe)
