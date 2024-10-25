import torch
from transformers import BertTokenizer, BertForSequenceClassification
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Função de pré-processamento
def preprocess_text(text):
    # Remover caracteres especiais e números
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Converter para minúsculas
    text = text.lower()
    return text

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

# Carregar o tokenizador
tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')

# Inicializar um modelo vazio
model = BertForSequenceClassification.from_pretrained('bert-base-multilingual-cased', num_labels=2)

# Carregar os pesos do modelo treinado
model.load_state_dict(torch.load('caminho/para/seu/modelo/treinado.pth'))

# Definir o modo de avaliação
model.eval()

'''print(model)'''

# Função para fazer previsões com base na entrada do usuário
def predict(input_text):
    # Pré-processar o texto de entrada
    input_text = preprocessar_noticia(input_text)
    # Tokenizar o texto pré-processado
    inputs = tokenizer(input_text, return_tensors='pt', padding=True, truncation=True)
    # Fazer previsões com o modelo
    outputs = model(**inputs)
    logits = outputs.logits
    predicted_class = torch.argmax(logits, dim=1).item()
    return predicted_class

# Loop para fazer previsões com base na entrada do usuário
while True:
    user_input = input("Digite o texto para classificação (ou 'sair' para encerrar): ")
    if user_input.lower() == 'sair':
        break
    else:
        predicted_class = predict(user_input)
        if predicted_class == 0:
            print("A previsão é que o texto é 'fake'.")
        elif predicted_class == 1:
            print("A previsão é que o texto é 'true'.")
