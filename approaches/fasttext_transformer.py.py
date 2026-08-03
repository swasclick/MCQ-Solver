import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split 
from collections import Counter
import re
import math
import os
import pickle 
from gensim.models import FastText 
from monitor.wandb_monitor import TrainMonitor



class CustomFastTextTokenizer:
    def __init__(self):
        self.word2idx = {'<PAD>': 0, '<UNK>': 1}
        self.idx2word = {0: '<PAD>', 1: '<UNK>'}
        self.vocab_size = 2
        
    def clean_text(self, text):
        text = str(text).lower()
        text = re.sub(r'[^a-z0-9\s]', ' ', text)
        return text.split()

    def train_and_build_matrix(self, texts, d_model=128):
        print("Tokenizing corpus for FastText...")
        sentences = [self.clean_text(text) for text in texts]
        
        print(f"Training custom FastText model on {len(sentences)} sequences...")
        ft_model = FastText(sentences=sentences, vector_size=d_model, window=5, min_count=1, workers=4, epochs=15)
        words = list(ft_model.wv.index_to_key)
        self.vocab_size = len(words) + 2
        embedding_matrix = np.zeros((self.vocab_size, d_model))
        embedding_matrix[1] = np.random.normal(scale=0.1, size=(d_model,))
        
        print("Transferring weights to PyTorch embedding matrix...")
        for i, word in enumerate(words):
            idx = i + 2 # Shift by 2 because 0=PAD, 1=UNK
            self.word2idx[word] = idx
            self.idx2word[idx] = word
            embedding_matrix[idx] = ft_model.wv[word]
            
        print(f"Custom FastText Vocabulary built with {self.vocab_size} tokens.")
        return torch.tensor(embedding_matrix, dtype=torch.float32)

    def encode(self, text, max_len):
        words = self.clean_text(text)
        tokens = [self.word2idx.get(w, 1) for w in words] # 1 is <UNK>
        
        # Truncate or Pad
        if len(tokens) > max_len:
            tokens = tokens[:max_len]
        else:
            tokens = tokens + [0] * (max_len - len(tokens)) # 0 is <PAD>
        return tokens

class ScratchMCQDataset(Dataset):
    def __init__(self, df, tokenizer, max_len=128, is_test=False):
        self.df = df
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.is_test = is_test
        self.options = ['A', 'B', 'C', 'D', 'E']

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        prompt = str(row['prompt'])
        
        # Encode [Prompt + Option] for all 5 choices
        input_ids = []
        for opt in self.options:
            combined_text = prompt + " " + str(row[opt])
            encoded = self.tokenizer.encode(combined_text, self.max_len)
            input_ids.append(encoded)
            
        item = {
            'input_ids': torch.tensor(input_ids, dtype=torch.long)
        }
        
        if not self.is_test:
            ans_idx = self.options.index(row['answer'])
            item['label'] = torch.tensor(ans_idx, dtype=torch.long)
            
        return item

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=500):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

class TinyMCQModel(nn.Module):
    def __init__(self, vocab_size, d_model=300, nhead=6, num_layers=1, dropout=0.3, pretrained_embeddings=None):
        super().__init__()
        if pretrained_embeddings is not None:
            self.embedding = nn.Embedding.from_pretrained(pretrained_embeddings, freeze=False, padding_idx=0)
        else:
            self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
            
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=512, 
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        self.classifier = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )

    def forward(self, input_ids):
        batch_size, num_opts, seq_len = input_ids.shape
        x = input_ids.view(batch_size * num_opts, seq_len)
        x = self.embedding(x) 
        x = self.transformer_encoder(x) 
        x = x.mean(dim=1)
        logits = self.classifier(x) 
        logits = logits.view(batch_size, num_opts)
        return logits

# Training

def run_scratch_pipeline(train_df: pd.DataFrame, test_df: pd.DataFrame, save_dir: str = None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    monitor = TrainMonitor(
    project="24f1000781-t22026",
    model_name="fasttext_MLP",
    version="v1",
    experiment="Baseline",
    config={
        "lr":2e-5,
        "batch_size":16,
        "epochs":8,
        "max length":512
        }
    )
    
    train_df, val_df = train_test_split(train_df, test_size=0.1, random_state=42)
    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)

    tokenizer = CustomFastTextTokenizer()
    all_text = train_df['prompt'].tolist() + val_df['prompt'].tolist() + test_df['prompt'].tolist()
    for opt in ['A', 'B', 'C', 'D', 'E']:
        all_text.extend(train_df[opt].tolist())
        all_text.extend(val_df[opt].tolist())
        all_text.extend(test_df[opt].tolist())

    D_MODEL = 256 
    pretrained_embeddings = tokenizer.train_and_build_matrix(all_text, d_model=D_MODEL)

    max_len = 256
    train_ds = ScratchMCQDataset(train_df, tokenizer, max_len=max_len)
    val_ds = ScratchMCQDataset(val_df, tokenizer, max_len=max_len) 
    test_ds = ScratchMCQDataset(test_df, tokenizer, max_len=max_len, is_test=True)

    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=16, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=16, shuffle=False)
    model = TinyMCQModel(
        vocab_size=tokenizer.vocab_size, 
        d_model=D_MODEL,
        nhead=4,        
        num_layers=2, 
        dropout=0.3,
        pretrained_embeddings=pretrained_embeddings
    ).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)

    epochs = 15
    print(f"\nStarting Training for {epochs} Epochs...")
    for epoch in range(epochs):
        model.train()
        total_train_loss = 0
        train_correct = 0
        train_total = 0
        
        for batch in train_loader:
            inputs = batch['input_ids'].to(device)
            labels = batch['label'].to(device)
            
            optimizer.zero_grad()
            logits = model(inputs) 
            
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            
            total_train_loss += loss.item()
            preds = torch.argmax(logits, dim=1)
            train_correct += (preds == labels).sum().item()
            train_total += labels.size(0)
            
        train_loss = total_train_loss / len(train_loader)
        train_acc = train_correct / train_total

        # Validate
        model.eval()
        total_val_loss = 0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for batch in val_loader:
                inputs = batch['input_ids'].to(device)
                labels = batch['label'].to(device)
                
                logits = model(inputs)
                loss = criterion(logits, labels)
                
                total_val_loss += loss.item()
                preds = torch.argmax(logits, dim=1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)

        val_loss = total_val_loss / len(val_loader) if len(val_loader) > 0 else 0.0
        val_acc = val_correct / val_total if val_total > 0 else 0.0

        try:
            monitor.monitor({
                "epoch": epoch,
                "train/loss": train_loss,
                "train/accuracy": train_acc,
                "validation/loss": val_loss,
                "validation/accuracy": val_acc,
            })
        except NameError:
            pass 

        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

    if save_dir:
        print(f"\nSaving model and tokenizer to {save_dir}...")
        os.makedirs(save_dir, exist_ok=True)
        # Save PyTorch weights
        torch.save(model.state_dict(), os.path.join(save_dir, "tiny_mcq_model.pth"))
        # Save Tokenizer
        with open(os.path.join(save_dir, "tokenizer.pkl"), "wb") as f:
            pickle.dump(tokenizer, f)
        print("Save complete!")

# Results

    print("\nGenerating Test Predictions...")
    model.eval()
    all_preds = []
    options = ['A', 'B', 'C', 'D', 'E']
    
    with torch.no_grad():
        for batch in test_loader:
            inputs = batch['input_ids'].to(device)
            logits = model(inputs)
            
            # Sort to get top 3 indices
            scores = logits.cpu().numpy()
            top_3_idx = np.argsort(scores, axis=1)[:, ::-1][:, :3]
            
            for row in top_3_idx:
                pred_str = " ".join([options[i] for i in row])
                all_preds.append(pred_str)

    submission_df = pd.DataFrame({
        'id': test_df['id'],
        'Prediction': all_preds
    })
    
    return submission_df