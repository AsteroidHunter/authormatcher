"""
@author: akash
"""

import torch
import polars as pl
from datasets import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments
)

# loading the dataset
train = pl.read_csv("./data/improved_train.csv")
val = pl.read_csv("./data/improved_val.csv")

train = train[["ID", "TEXT", "LABEL"]]
val = val[["ID", "TEXT", "LABEL"]]

# combining provided training set with the newly assembled one
train_provided = pl.read_csv("./data/train.csv")
train = pl.concat([train, train_provided])
train = train.sample(fraction=1, shuffle=True, seed=894552352)

train.head()

# loading RoBERTa
model_name = "FacebookAI/roberta-base"

# pointing to a custom directory to save the model
custom_cache_dir = "../.cache_xdisk/"

# loading model and tokenizer
model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    cache_dir=custom_cache_dir,
    trust_remote_code=True,
    num_labels=2,
)
tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    cache_dir=custom_cache_dir,
    trust_remote_code=True
)

# replacing [SNIPPET] with separation token for the model
train = train.with_columns(
    pl.col("TEXT").str.replace(
        r"\[SNIPPET\]",
        tokenizer.sep_token
    )
)

val = val.with_columns(
    pl.col("TEXT").str.replace(
        r"\[SNIPPET\]",
        tokenizer.sep_token
    )
)

# tokenizing the dataset

train_hf = Dataset.from_polars(train)
val_hf = Dataset.from_polars(val)


def tokenize_function(df):
    df_tokenized = tokenizer(df["TEXT"], padding=True, truncation=True)
    df_tokenized["labels"] = df["LABEL"]

    return df_tokenized

tokenized_train = train_hf.map(tokenize_function, batched=True)
tokenized_val = val_hf.map(tokenize_function, batched=True)

## ADDING CUSTOM TRAINER THAT MANAGES CLASS IMBALANCE

# since the weight is slightly imbalanced, we will manage this
# by informing the optimizer
num_pos = len(train.filter(pl.col("LABEL") == 1))
num_neg = len(train.filter(pl.col("LABEL") == 0))

# finding the inverse frequency
neg_weight = len(train) / (2 * num_neg)
pos_weight = len(train) / (2 * num_pos)

class_weights = [neg_weight, pos_weight]


# slightly modified from https://discuss.huggingface.co/t/how-can-i-use-class-weights-when-training/1067/6

class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits

        # moving class weights to the same device as logits
        class_weights_tensor = torch.tensor(class_weights).to(logits.device)

        # defining the weighted loss function
        loss_fct = torch.nn.CrossEntropyLoss(weight=class_weights_tensor)
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))

        return (loss, outputs) if return_outputs else loss

# defining basic training arguments
training_args = TrainingArguments(
    output_dir="./results_new_train/",
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=6,
    learning_rate=1e-4,
    warmup_steps=500,
    lr_scheduler_type="cosine",
    weight_decay=0.05,
    eval_steps=1000,
    save_steps=1000,
    logging_steps=100,
    eval_strategy="steps",
    # metric_for_best_model="eval_loss",
    # greater_is_better=False,
    logging_dir=f"./results_new_train/logs",
    fp16=True, # hash this out if on MPS
    ddp_find_unused_parameters=False,
)

# fine-tuning the model
trainer = CustomTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_val,
    processing_class=tokenizer,
)

trainer.train()
