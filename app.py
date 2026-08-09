import os
import re
import math
import pickle
import numpy as np
import pandas as pd
import joblib
import torch
import torch.nn as nn
import streamlit as st
from transformers import (
    pipeline, 
    AutoTokenizer, 
    AutoModelForSequenceClassification,
    AutoModelForMultipleChoice
)
from huggingface_hub import hf_hub_download
try:
    from approaches.best_score import TFIDFRetriever
except ImportError:
    pass 

# Models
MODEL_1_NAME = "Model 1: Zero-Shot Classifier (BART)"
MODEL_1_PATH = r"facebook/bart-large-mnli"

MODEL_2_NAME = "Model 2: Cross-Encoder (MiniLM)"
MODEL_2_PATH = r"Swasclick/mcq-miniLM"  
TOKENIZER_2_PATH = r"Swasclick/mcq-miniLM" 

MODEL_3_NAME = "Model 3: Custom FastText + Transformer"

MODEL_4_NAME = "Model 4: RAG + Pretrained Finetuning"
MODEL_4_DIR = r"Swasclick/mcq-rag"
RETRIEVER_PATH = hf_hub_download(
    repo_id="Swasclick/mcq-rag",
    filename="tfidf_retriever.pkl"
)


class CustomFastTextTokenizer:
    def __init__(self):
        self.word2idx = {'<PAD>': 0, '<UNK>': 1}
        self.idx2word = {0: '<PAD>', 1: '<UNK>'}
        self.vocab_size = 2
        
    def clean_text(self, text):
        text = str(text).lower()
        text = re.sub(r'[^a-z0-9\s]', ' ', text)
        return text.split()

    def encode(self, text, max_len=256):
        words = self.clean_text(text)
        tokens = [self.word2idx.get(w, 1) for w in words]
        if len(tokens) > max_len:
            tokens = tokens[:max_len]
        else:
            tokens = tokens + [0] * (max_len - len(tokens))
        return tokens


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
    def __init__(self, vocab_size, d_model=256, nhead=4, num_layers=2, dropout=0.3):
        super().__init__()
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


@st.cache_resource
def load_model_1(model_path):
    device = 0 if torch.cuda.is_available() else -1
    return pipeline("zero-shot-classification", model=model_path, device=device)

@st.cache_resource
def load_model_2(model_path, tokenizer_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path, num_labels=1).to(device)
    model.eval()
    return tokenizer, model, device

@st.cache_resource
def load_model_3():

    device = torch.device(
        'cuda' if torch.cuda.is_available() else 'cpu'
    )

    weights_path = hf_hub_download(
        repo_id="Swasclick/mcq-scratch",
        filename="tiny_mcq_model.pth"
    )

    tokenizer_path = hf_hub_download(
        repo_id="Swasclick/mcq-scratch",
        filename="tokenizer.pkl"
    )

    with open(tokenizer_path, "rb") as f:
        tokenizer = pickle.load(f)

    model = TinyMCQModel(
        vocab_size=tokenizer.vocab_size,
        d_model=256,
        nhead=4,
        num_layers=2
    ).to(device)

    model.load_state_dict(
        torch.load(weights_path, map_location=device)
    )

    model.eval()

    return tokenizer, model, device

@st.cache_resource
def load_model_4():

    device = torch.device(
        'cuda' if torch.cuda.is_available() else 'cpu'
    )

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_4_DIR,
        use_fast=True
    )

    model = AutoModelForMultipleChoice.from_pretrained(
        MODEL_4_DIR
    ).to(device)

    model.eval()

    retriever_path = RETRIEVER_PATH

    retriever = joblib.load(retriever_path)

    return tokenizer, model, retriever, device


def predict_model_1(prompt, options):
    classifier = load_model_1(MODEL_1_PATH)
    labels = [options[opt] for opt in ['A', 'B', 'C', 'D', 'E']]
    result = classifier(prompt, candidate_labels=labels)
    
    label_to_score = dict(zip(result['labels'], result['scores']))
    probs = [label_to_score[options[opt]] for opt in ['A', 'B', 'C', 'D', 'E']]
    return np.array(probs)

def predict_model_2(prompt, options):
    tokenizer, model, device = load_model_2(MODEL_2_PATH, TOKENIZER_2_PATH)
    option_texts = [options[opt] for opt in ['A', 'B', 'C', 'D', 'E']]
    
    enc = tokenizer(
        [prompt] * 5, option_texts,
        truncation=True, max_length=512, padding=True, return_tensors='pt'
    ).to(device)
    
    with torch.no_grad():
        logits = model(**enc).logits.squeeze(-1)
        probs = torch.softmax(logits, dim=-1).cpu().numpy()
        
    return probs

def predict_model_3(prompt, options):
    tokenizer, model, device = load_model_3()
    
    input_ids = []
    for opt in ['A', 'B', 'C', 'D', 'E']:
        combined_text = prompt + " " + options[opt]
        encoded = tokenizer.encode(combined_text, max_len=256)
        input_ids.append(encoded)
        
    tensor_inputs = torch.tensor([input_ids], dtype=torch.long).to(device)
    
    with torch.no_grad():
        logits = model(tensor_inputs)
        probs = torch.softmax(logits, dim=-1).squeeze(0).cpu().numpy()
        
    return probs

def predict_model_4(prompt, options):
    tokenizer, model, retriever, device = load_model_4()
    
    ctx = retriever.retrieve([prompt], top_k=3)[0]
    
    if str(ctx).strip():
        combined = f"Context: {ctx}\nQuestion: {prompt}"
    else:
        combined = str(prompt)
        
    first_sentences = [combined] * 5
    second_sentences = [options[opt] for opt in ['A', 'B', 'C', 'D', 'E']]
    
    enc = tokenizer(
        first_sentences,
        second_sentences,
        truncation="only_first",
        max_length=256,
        padding=True,
        return_tensors="pt"
    )
    
    enc = {k: v.unsqueeze(0).to(device) for k, v in enc.items()}
    
    with torch.no_grad():
        outputs = model(**enc)
        logits = outputs.logits.squeeze(0)
        probs = torch.softmax(logits, dim=-1).cpu().numpy()
        
    return probs


def main():
    st.set_page_config(page_title="MCQ Solver", layout="wide")
    st.title("Multiple Choice Question Solver")
    
    selected_model_name = st.selectbox(
        "Select Model Architecture:",
        [MODEL_1_NAME, MODEL_2_NAME, MODEL_3_NAME, MODEL_4_NAME]
    )
    
    st.markdown("---")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("1. Enter Question & Options")
        prompt = st.text_area("Prompt:", "Question: Which of the following is a programming language?", height=120)
        
        st.write("Options:")
        opt_a = st.text_input("Option A:", "Python")
        opt_b = st.text_input("Option B:", "Java")
        opt_c = st.text_input("Option C:", "SQL")
        opt_d = st.text_input("Option D:", "Plastic Crayon")
        opt_e = st.text_input("Option E:", "HTML")
        
        submit = st.button("Predicted Answer", type="primary", use_container_width=True)

    options_dict = {'A': opt_a, 'B': opt_b, 'C': opt_c, 'D': opt_d, 'E': opt_e}
    
    with col2:
        st.subheader("2. Prediction Probability Distribution")
        
        if submit:
            if not prompt.strip() or any(not v.strip() for v in options_dict.values()):
                st.warning("Please fill in the prompt and all 5 options before predicting.")
                return

            with st.spinner("Running inference..."):
                if selected_model_name == MODEL_1_NAME:
                    probs = predict_model_1(prompt, options_dict)
                elif selected_model_name == MODEL_2_NAME:
                    probs = predict_model_2(prompt, options_dict)
                elif selected_model_name == MODEL_3_NAME:
                    probs = predict_model_3(prompt, options_dict)
                elif selected_model_name == MODEL_4_NAME:
                    probs = predict_model_4(prompt, options_dict)

            option_labels = ['A', 'B', 'C', 'D', 'E']
            best_idx = np.argmax(probs)
            predicted_option = option_labels[best_idx]
            
            df_chart = pd.DataFrame({
                'Option': option_labels,
                'Probability': probs
            }).set_index('Option')

            st.bar_chart(df_chart)

            st.markdown(f"### **The model predicts option: {predicted_option}**")
            st.info(f"**Option:** {options_dict[predicted_option]}")

if __name__ == "__main__":
    main()