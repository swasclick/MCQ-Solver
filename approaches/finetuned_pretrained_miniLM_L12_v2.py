import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
OPTIONS = ['A', 'B', 'C', 'D', 'E']
MODEL_NAME = 'cross-encoder/ms-marco-MiniLM-L12-v2'
MAX_LEN = 512
BATCH_SIZE = 16
EPOCHS = 8
LR = 2e-5
OUTPUT_DIR = '/kaggle/working/model'

class MCQPairDataset(Dataset):
    '''
    A dataset class to convert the prompt-answer data into embeddings
    '''
    def __init__(self, df, tokenizer, max_len=MAX_LEN):
        self.pairs = []
        for _, row in df.iterrows():
            prompt = str(row['prompt']).strip()
            correct_answer = row['answer']
            for option_label in OPTIONS:
                option_text = str(row[option_label]).strip()
                label = 1.0 if option_label == correct_answer else 0.0
                self.pairs.append((prompt, option_text, label))
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        prompt, option_text, label = self.pairs[idx]
        enc = self.tokenizer(
            prompt, option_text,
            truncation=True,
            max_length=self.max_len,
            padding='max_length',
            return_tensors='pt'
        )
        item = {k: v.squeeze(0) for k, v in enc.items()}
        item['label'] = torch.tensor(label, dtype=torch.float)
        return item

def load_data():
    train_data, val_data = train_test_split(
        train_df,
        test_size=0.2,
        random_state=42,
        stratify=train_df['answer'] if len(train_df) > 0 else None
    )
    
    train_data = train_data.reset_index(drop=True)
    val_data = val_data.reset_index(drop=True)
    
    return train_data, val_data, test_df

# trining

def train_one_epoch(model, loader, optimizer, scheduler, loss_fn):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    
    for batch in loader:
        optimizer.zero_grad()
        input_ids = batch['input_ids'].to(DEVICE)
        attention_mask = batch['attention_mask'].to(DEVICE)
        labels = batch['label'].to(DEVICE)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits.squeeze(-1) # remove extra dimension frm logit output
        loss = loss_fn(logits, labels)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        preds = (torch.sigmoid(logits) > 0.5).float()
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    avg_loss = total_loss / len(loader) if len(loader) > 0 else 0.0
    accuracy = correct / total if total > 0 else 0.0
    
    return avg_loss, accuracy


@torch.no_grad()
def score_options(model, tokenizer, prompt, option_texts, max_len=MAX_LEN):
    model.eval()
    enc = tokenizer(
        [prompt] * len(option_texts), option_texts,
        truncation=True, max_length=max_len, padding=True, return_tensors='pt'
    )
    enc = {k: v.to(DEVICE) for k, v in enc.items()}
    logits = model(**enc).logits.squeeze(-1)
    scores = torch.sigmoid(logits).cpu().numpy() 
    return scores

def calculate_map_at_3(predictions):
    score = 0
    if not predictions:
        return 0.0
    for pred in predictions:
        if pred['correct'] in pred['predicted']:
            position = pred['predicted'].index(pred['correct']) + 1
            score += 1 / position
    return score / len(predictions)

@torch.no_grad()
def validate_model(model, tokenizer, val_df, loss_fn):
    model.eval()
    predictions = []
    total_loss = 0.0
    correct_binary = 0
    total_binary = 0
    
    # Calculate MAP@3 by iterating over the dataframe
    for _, row in val_df.iterrows():
        prompt = str(row['prompt']).strip()
        correct_answer = row['answer']
        option_texts = [str(row[opt]).strip() for opt in OPTIONS]

        scores = score_options(model, tokenizer, prompt, option_texts)
        top_3_indices = np.argsort(scores)[::-1][:3]
        top_3_labels = [OPTIONS[i] for i in top_3_indices]

        predictions.append({'predicted': top_3_labels, 'correct': correct_answer})
        
        # Compute validation
        correct_idx = OPTIONS.index(correct_answer)
        labels = torch.zeros(len(OPTIONS)).to(DEVICE)
        labels[correct_idx] = 1.0
        
        enc = tokenizer(
            [prompt] * len(option_texts), option_texts,
            truncation=True, max_length=MAX_LEN, padding=True, return_tensors='pt'
        )
        enc = {k: v.to(DEVICE) for k, v in enc.items()}
        logits = model(**enc).logits.squeeze(-1)
        
        loss = loss_fn(logits, labels)
        total_loss += loss.item()
        
        preds = (torch.sigmoid(logits) > 0.5).float()
        correct_binary += (preds == labels).sum().item()
        total_binary += labels.size(0)

    val_loss = total_loss / len(val_df) if len(val_df) > 0 else 0.0
    val_acc = correct_binary / total_binary if total_binary > 0 else 0.0
    map_at_3 = calculate_map_at_3(predictions)
    
    return map_at_3, val_loss, val_acc, predictions

def generate_test_predictions(model, tokenizer, test_df):
    predictions = []
    for idx, row in test_df.iterrows():
        prompt = str(row['prompt']).strip()
        option_texts = [str(row[opt]).strip() for opt in OPTIONS]

        scores = score_options(model, tokenizer, prompt, option_texts)
        top_3_indices = np.argsort(scores)[::-1][:3]
        top_3_labels = [OPTIONS[i] for i in top_3_indices]

        predictions.append(' '.join(top_3_labels))
    return predictions


def main():
    train_data, val_data, test_df = load_data()

    if len(train_data) == 0:
        print("Data is empty. Bro you must uncomment the pandas read_csv lines in load_data().")
        return

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=1).to(DEVICE)

    train_ds = MCQPairDataset(train_data, tokenizer)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    total_steps = len(train_loader) * EPOCHS
    warmup_steps = max(10, int(total_steps * 0.1))
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )
    
    loss_fn = nn.BCEWithLogitsLoss()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    best_map = -1.0
    
    for epoch in range(EPOCHS):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, scheduler, loss_fn)
        val_map, val_loss, val_acc, _ = validate_model(model, tokenizer, val_data, loss_fn)
        
        print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | Val MAP@3: {val_map:.4f}")

        try:
            monitor.monitor({
                "epoch": epoch,
                "train/loss": train_loss,
                "train/accuracy": train_acc,
                "validation/loss": val_loss,
                "validation/accuracy": val_acc,
                "validation/map_at_3": val_map
            })
        except NameError:
            pass 

        if val_map > best_map:
            best_map = val_map
            model.save_pretrained(OUTPUT_DIR)
            tokenizer.save_pretrained(OUTPUT_DIR)
            print(f"New best model saved! MAP@3 = {val_map:.4f}")

    # Load best model for inference
    tokenizer = AutoTokenizer.from_pretrained(OUTPUT_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(OUTPUT_DIR).to(DEVICE)

    test_predictions = generate_test_predictions(model, tokenizer, test_df)

    submission_df = pd.DataFrame({'id': test_df['id'], 'Prediction': test_predictions})
    submission_df.to_csv('submission.csv', index=False)
    
    print("Submission saved to submission.csv")

if __name__ == "__main__":
    main()