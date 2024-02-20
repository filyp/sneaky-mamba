# %%
import collections
import math
import re
from typing import Any, Dict

import numpy as np
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    RwkvConfig,
    RwkvForCausalLM,
    Trainer,
    TrainingArguments,
)

MODEL_NAME = "sgugger/rwkv-430M-pile"
model = RwkvForCausalLM.from_pretrained(MODEL_NAME)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token


def remove_url_from_text(text: str):
    """Remove square brackets around linked text and (_URL_0_) after"""
    return re.sub(r"\[|\]|\(_URL_\d+_\)", "", text)

def tokenize_function(examples: Dict[str, Any]) -> Dict[str, Any]:
    """Concatenate and tokenize the answers in flattened ELI5 data"""
    concatenated = [remove_url_from_text(" ".join(x)) for x in examples["answers.text"]]
    return tokenizer(concatenated)


def chunk(examples: Dict[str, Any], chunk_size: int = 256) -> Dict[str, Any]:
    """Concatenate and chunk batches of data"""
    concatenated = {k: sum(examples[k], []) for k in examples.keys()}
    total_length = len(concatenated[list(examples.keys())[0]])
    total_length = (total_length // chunk_size) * chunk_size
    return {
        k: [t[i : i + chunk_size] for i in range(0, total_length, chunk_size)]
        for k, t in concatenated.items()
    }


def set_labels(examples: Dict[str, Any]) -> Dict[str, Any]:
    """Add a labels column to the dataset which is a copy of input_ids"""
    examples["labels"] = examples["input_ids"].copy()
    return examples



DATASET = "squad"
CHUNK_SIZE = 128
TEST_SPLIT_SIZE = 0.2
BATCH_SIZE = 32
DATASET_SPLIT = "train"

dataset = load_dataset(DATASET, split=DATASET_SPLIT)
dataset = dataset.train_test_split(test_size=TEST_SPLIT_SIZE)
dataset = dataset.flatten()

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)



# Encode
encoded_dataset = dataset.map(
    tokenize_function,
    batched=True, 
    num_proc=4,
    remove_columns=dataset["train"].column_names
)

# Chunk
chunked_dataset = encoded_dataset.map(
    chunk,
    fn_kwargs={"chunk_size": CHUNK_SIZE},
    batched=True, 
)

# Label
lm_dataset = chunked_dataset.map(
    set_labels,
    batched=True
) 

# %%
training_args = TrainingArguments(
    output_dir = MODEL_NAME + "-" + DATASET,
    overwrite_output_dir=True,
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    weight_decay=0.01,
    push_to_hub=False,
    logging_steps=len(lm_dataset["train"]) // BATCH_SIZE
)
    
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=lm_dataset["train"],
    eval_dataset=lm_dataset["test"],
    data_collator=data_collator,
)

# Evaluate before train
eval_0 = trainer.evaluate()
perplexity_0 = math.exp(eval_0["eval_loss"])

# Train
trainer.train()

# %%

# Evaluate after train
eval_f = trainer.evaluate()
perplexity_f = math.exp(eval_f["eval_loss"])
# %%

# %%
