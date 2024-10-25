import torch
from transformers import BertTokenizer, BertForSequenceClassification
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score

import pandas as pd


# Carregar o arquivo CSV
data = pd.read_csv("/home/adena/Área de Trabalho/implementação TCC/base de dados/Fake.br-Corpus-master/preprocessed/pre-processed.csv")

# Dividir os dados em classes separadas
fake_data = data[data['label'] == 'fake']
true_data = data[data['label'] == 'true']

# Dividir os dados em conjuntos de treino, validação e teste usando estratificação
fake_train, fake_temp = train_test_split(fake_data, test_size=0.2, random_state=42, stratify=fake_data['label'])
true_train, true_temp = train_test_split(true_data, test_size=0.2, random_state=42, stratify=true_data['label'])

fake_val, fake_test = train_test_split(fake_temp, test_size=0.5, random_state=42, stratify=fake_temp['label'])
true_val, true_test = train_test_split(true_temp, test_size=0.5, random_state=42, stratify=true_temp['label'])

# Combinar os conjuntos de treino, validação e teste
train_data = pd.concat([fake_train, true_train], ignore_index=True)
val_data = pd.concat([fake_val, true_val], ignore_index=True)
test_data = pd.concat([fake_test, true_test], ignore_index=True)

# Verificar a distribuição das classes nos conjuntos
print("Distribuição das classes nos conjuntos de treino:")
print(train_data['label'].value_counts())
print("\nDistribuição das classes nos conjuntos de validação:")
print(val_data['label'].value_counts())
print("\nDistribuição das classes nos conjuntos de teste:")
print(test_data['label'].value_counts())

# Convertendo os rótulos para minúsculas
data['label'] = data['label'].str.lower()

# Mapeando os rótulos para valores numéricos
label_map = {'fake': 0, 'true': 1}
data['label'] = data['label'].map(label_map)

print(data['label'])

# Definir o modelo BERT e o tokenizer
model_name = 'bert-base-multilingual-cased'
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name, num_labels=2)

class CustomDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=128):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = self.data.iloc[idx]['preprocessed_news']
        label = self.data.iloc[idx]['label']  # Mantendo como string
        label = 0 if label == 'fake' else 1  # Mapear 'fake' para 0 e 'TRUE' para 1
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            truncation=True,
            return_token_type_ids=False,
            padding='max_length',
            return_attention_mask=True,
            return_tensors='pt'
        )
        input_ids = encoding['input_ids'].flatten()
        attention_mask = encoding['attention_mask'].flatten()
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'label': torch.tensor(label, dtype=torch.long)
        }

# Criar os datasets e data loaders
train_dataset = CustomDataset(train_data, tokenizer)
val_dataset = CustomDataset(val_data, tokenizer)
test_dataset = CustomDataset(test_data, tokenizer)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Definir o dispositivo
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Treinar o modelo
model.to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
criterion = torch.nn.CrossEntropyLoss()

for epoch in range(4):
    model.train()
    for batch in train_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)

        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()

    # Avaliar o modelo no conjunto de validação
    model.eval()
    val_preds = []
    val_labels = []
    for batch in val_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)

        with torch.no_grad():
            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            preds = torch.argmax(logits, dim=1)

        val_preds.extend(preds.cpu().numpy())
        val_labels.extend(labels.cpu().numpy())

    val_accuracy = accuracy_score(val_labels, val_preds)
    print(f'Epoch {epoch+1}, Validation Accuracy: {val_accuracy}')

    # Avaliar o modelo no conjunto de teste
    model.eval()
    test_preds = []
    test_labels = []
    for batch in test_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)

        with torch.no_grad():
            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            preds = torch.argmax(logits, dim=1)

        test_preds.extend(preds.cpu().numpy())
        test_labels.extend(labels.cpu().numpy())

    test_accuracy = accuracy_score(test_labels, test_preds)
    test_precision = precision_score(test_labels, test_preds)
    test_recall = recall_score(test_labels, test_preds)
    test_f1 = f1_score(test_labels, test_preds)
    print(f'Epoch {epoch+1}, Test Accuracy: {test_accuracy}, Precision: {test_precision}, Recall: {test_recall}, F1-score: {test_f1}')

test_accuracy = accuracy_score(test_labels, test_preds)
test_precision = precision_score(test_labels, test_preds)
test_recall = recall_score(test_labels, test_preds)
test_f1 = f1_score(test_labels, test_preds)
print(f'Test Accuracy: {test_accuracy}, Precision: {test_precision}, Recall: {test_recall}, F1-score: {test_f1}')

# Salvar o modelo e o tokenizador
model.save_pretrained('/home/adena/Área de Trabalho/implementação TCC/Bert/')
tokenizer.save_pretrained('/home/adena/Área de Trabalho/implementação TCC/Bert/')
