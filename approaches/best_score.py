import pandas as pd
import numpy as np
import torch
from dataclasses import dataclass
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer,AutoModelForMultipleChoice,TrainingArguments,Trainer,PreTrainedTokenizerBase
from transformers.utils import PaddingStrategy
from datasets import Dataset
from typing import Optional, Union, List

class TFIDFRetriever:
    """
    Unsupervised lexical retriever using TF-IDF and Cosine Similarity.
    Builds context from the training data's ground-truth answers.
    """
    def __init__(self, corpus: List[str], max_features: int = 50000):
        self.corpus = np.array(corpus)
        self.vectorizer = TfidfVectorizer(
            stop_words='english', 
            ngram_range=(1, 2), 
            max_features=max_features,
            dtype=np.float32
        )
        self.tfidf_matrix = self.vectorizer.fit_transform(self.corpus)

    def retrieve(self, queries: List[str], top_k: int = 3) -> List[str]:
        query_vectors = self.vectorizer.transform(queries)
        similarities = cosine_similarity(query_vectors, self.tfidf_matrix)
        
        retrieved_contexts = []
        for sim in similarities:
            top_indices = sim.argsort()[-top_k:][::-1]
            context_chunks = self.corpus[top_indices]
            retrieved_contexts.append(" ".join(context_chunks))
        
        return retrieved_contexts

@dataclass
class DataCollatorForMultipleChoice:
    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None

    def __call__(self, features):
        label_name = "label" if "label" in features[0].keys() else "labels"
        labels = [feature.pop(label_name) for feature in features]
        batch_size = len(features)
        num_choices = len(features[0]["input_ids"])
        
        flattened_features = [
            [{k: v[i] for k, v in feature.items()} for i in range(num_choices)] for feature in features
        ]
        flattened_features = sum(flattened_features, [])
        
        batch = self.tokenizer.pad(
            flattened_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )
        
        batch = {k: v.view(batch_size, num_choices, -1) for k, v in batch.items()}
        batch["labels"] = torch.tensor(labels, dtype=torch.int64)
        return batch

def compute_metrics(eval_predictions):
    predictions, label_ids = eval_predictions
    preds = np.argmax(predictions, axis=1)
    
    # Calculate MAP@3 for validation
    map3_score = 0.0
    for pred_scores, true_label in zip(predictions, label_ids):
        top_3 = np.argsort(pred_scores)[::-1][:3]
        if true_label in top_3:
            position = np.where(top_3 == true_label)[0][0] + 1
            map3_score += 1.0 / position
            
    return {
        "accuracy": (preds == label_ids).astype(np.float32).mean().item(),
        "map@3": map3_score / len(label_ids)
    }

def run_pipeline(
    train_df: pd.DataFrame, 
    test_df: pd.DataFrame, 
    model_name: str = "distilroberta-base"
) -> pd.DataFrame:
    
    options = ['A', 'B', 'C', 'D', 'E']
    option_to_index = {opt: i for i, opt in enumerate(options)}
    index_to_option = {i: opt for i, opt in enumerate(options)}

    train_df = train_df.copy()
    test_df = test_df.copy()

    train_df['label'] = train_df['answer'].map(option_to_index)
    test_df['label'] = 0 

    # --- RAG RETRIEVAL STAGE ---
    print("Building RAG corpus from training data's correct answers...")
    knowledge_base = train_df.apply(
        lambda row: f"Question: {row['prompt']} Answer: {row[row['answer']]}", axis=1
    ).tolist()

    retriever = TFIDFRetriever(knowledge_base)
    train_df['context'] = retriever.retrieve(train_df['prompt'].tolist(), top_k=3)
    test_df['context'] = retriever.retrieve(test_df['prompt'].tolist(), top_k=3)

    train_df, eval_df = train_test_split(
        train_df, 
        test_size=0.2, 
        stratify=train_df['label'], 
        random_state=42
    )

    train_ds = Dataset.from_pandas(train_df)
    eval_ds = Dataset.from_pandas(eval_df)
    test_ds = Dataset.from_pandas(test_df)

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

    def preprocess_function(examples):
        first_sentences = []
        for ctx, prompt in zip(examples["context"], examples["prompt"]):
            if str(ctx).strip():
                combined = f"Context: {ctx}\nQuestion: {prompt}"
            else:
                combined = str(prompt)
            first_sentences.append([combined] * 5)
            
        second_sentences = [[str(examples[opt][i]) for opt in options] for i in range(len(examples["prompt"]))]
        
        first_sentences = sum(first_sentences, [])
        second_sentences = sum(second_sentences, [])
        
        tokenized_examples = tokenizer(
            first_sentences,
            second_sentences,
            truncation="only_first",  
            max_length=256, 
            padding=False
        )
        return {k: [v[i : i + 5] for i in range(0, len(v), 5)] for k, v in tokenized_examples.items()}

    tokenized_train = train_ds.map(preprocess_function, batched=True, remove_columns=[c for c in train_ds.column_names if c != 'label'])
    tokenized_eval = eval_ds.map(preprocess_function, batched=True, remove_columns=[c for c in eval_ds.column_names if c != 'label'])
    tokenized_test = test_ds.map(preprocess_function, batched=True, remove_columns=[c for c in test_ds.column_names if c != 'label'])

    # Fine-Tuning Initialization
    model = AutoModelForMultipleChoice.from_pretrained(model_name)
    
    # RoBERTa architectures support standard FP16 without NaN overflow
    use_fp16 = True
    training_args = TrainingArguments(
        output_dir="./results",
        eval_strategy="epoch",             
        save_strategy="epoch",             
        logging_strategy="epoch",          
        load_best_model_at_end=True,       
        metric_for_best_model="map@3",
        greater_is_better=True,
        learning_rate=5e-6,                
        per_device_train_batch_size=1,     
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=2,     
        max_grad_norm=1.0,                 
        warmup_ratio=0.2,                  
        lr_scheduler_type="linear",        
        num_train_epochs=3,                
        weight_decay=0.01,
        fp16=use_fp16,
        report_to="none"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_eval,
        processing_class=tokenizer,
        data_collator=DataCollatorForMultipleChoice(tokenizer=tokenizer),
        compute_metrics=compute_metrics
    )

    trainer.train()

    # Inference logic
    predictions = trainer.predict(tokenized_test).predictions
    
    top_3_indices = np.argsort(predictions, axis=1)[:, ::-1][:, :3]
    
    top_3_predictions = []
    for indices in top_3_indices:
        pred_str = " ".join([index_to_option[idx] for idx in indices])
        top_3_predictions.append(pred_str)

    submission_df = pd.DataFrame({
        'ID': test_df['id'],
        'Prediction': top_3_predictions
    })

    return submission_df

if __name__ == "__main__":
    sub_df = run_pipeline(train_df,test_df)
    sub_df.to_csv('submission.csv',index=False)