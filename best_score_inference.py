import pandas as pd
import numpy as np

from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForMultipleChoice,
    Trainer,
    TrainingArguments,
)

from approaches.best_score import (
    TFIDFRetriever,
    DataCollatorForMultipleChoice,
)

# -----------------------------------------------------
# Configuration
# -----------------------------------------------------

MODEL_PATH = "./best_model"

train_df = pd.read_csv("train.csv")
test_df = pd.read_csv("test.csv")

options = ["A", "B", "C", "D", "E"]
index_to_option = {i: o for i, o in enumerate(options)}

# -----------------------------------------------------
# Build retrieval corpus
# -----------------------------------------------------

print("Building retrieval index...")

knowledge_base = train_df.apply(
    lambda row: f"Question: {row['prompt']} Answer: {row[row['answer']]}",
    axis=1,
).tolist()

retriever = TFIDFRetriever(knowledge_base)

test_df = test_df.copy()
test_df["label"] = 0
test_df["context"] = retriever.retrieve(
    test_df["prompt"].tolist(),
    top_k=3,
)

# -----------------------------------------------------
# Dataset
# -----------------------------------------------------

test_ds = Dataset.from_pandas(test_df)

# -----------------------------------------------------
# Load tokenizer/model
# -----------------------------------------------------

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

model = AutoModelForMultipleChoice.from_pretrained(MODEL_PATH)

# -----------------------------------------------------
# Preprocessing (must be IDENTICAL to training)
# -----------------------------------------------------

def preprocess_function(examples):

    first_sentences = []

    for ctx, prompt in zip(examples["context"], examples["prompt"]):

        if str(ctx).strip():
            combined = f"Context: {ctx}\nQuestion: {prompt}"
        else:
            combined = str(prompt)

        first_sentences.append([combined] * 5)

    second_sentences = [
        [str(examples[o][i]) for o in options]
        for i in range(len(examples["prompt"]))
    ]

    first_sentences = sum(first_sentences, [])
    second_sentences = sum(second_sentences, [])

    tokenized = tokenizer(
        first_sentences,
        second_sentences,
        truncation="only_first",
        max_length=256,
        padding=False,
    )

    return {
        k: [v[i:i+5] for i in range(0, len(v), 5)]
        for k, v in tokenized.items()
    }

tokenized_test = test_ds.map(
    preprocess_function,
    batched=True,
    remove_columns=[c for c in test_ds.column_names if c != "label"],
)

# -----------------------------------------------------
# Trainer
# -----------------------------------------------------

args = TrainingArguments(
    output_dir="./tmp_predict",
    per_device_eval_batch_size=2,
    report_to="none",
)

trainer = Trainer(
    model=model,
    args=args,
    processing_class=tokenizer,
    data_collator=DataCollatorForMultipleChoice(tokenizer),
)

# -----------------------------------------------------
# Predict
# -----------------------------------------------------

predictions = trainer.predict(tokenized_test).predictions

top3 = np.argsort(predictions, axis=1)[:, ::-1][:, :3]

submission = pd.DataFrame({
    "ID": test_df["id"],
    "Prediction": [
        " ".join(index_to_option[i] for i in row)
        for row in top3
    ],
})

submission.to_csv("submission.csv", index=False)

print(submission.head())