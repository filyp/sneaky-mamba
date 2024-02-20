# %%
import os
import sys
from pathlib import Path

import numpy as np
import torch
import wandb
from generation import DirectTasksDataset, test_tokenization
from generation import TasksDataset, test_tokenization
from train_helpers import Curriculum, DirectReasoningTrainer, ReasoningTrainer, get_accuracy_bar
from transformers import (
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    RwkvConfig,
    RwkvForCausalLM,
    RwkvModel,
    Trainer,
    TrainingArguments,
)

os.environ["TOKENIZERS_PARALLELISM"] = "true"

MODEL_NAME = "sgugger/rwkv-430M-pile"
model = RwkvForCausalLM.from_pretrained(MODEL_NAME).to("cuda")
# model = RwkvModel.from_pretrained("sgugger/rwkv-430M-pile")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

test_tokenization(tokenizer)
answer_token = tokenizer.encode("\n")[0]

# %%

def evaluate_example(model, ex):
    split_index = torch.where(ex["input_ids"] == answer_token)[0] + 1
    task_ids = ex["input_ids"][:split_index].to("cuda")
    # reasoning_ids = ex["input_ids"][split_index:].to("cuda")
    target_output = ex["input_ids"]
    task_len = len(task_ids)
    out = model.generate(
        input_ids=task_ids.reshape(1, -1),
        # attention_mask=ex["attention_mask"][:task_len].reshape(1, -1).to("cuda"),
        max_length=len(target_output),
        temperature=1,
        pad_token_id=tokenizer.pad_token_id,
    )
    if len(out[0]) != len(target_output):
        return False
    return all(out[0] == target_output)


# %%
training_args = TrainingArguments(
    disable_tqdm=True,  # This disables the progress bars
    per_device_train_batch_size=32,
    output_dir = MODEL_NAME + "-tests",
    # overwrite_output_dir=True,
    # evaluation_strategy="epoch",
    num_train_epochs=1,
    learning_rate=2e-5,
    weight_decay=0.01,
    push_to_hub=False,
    # logging_steps=len(lm_dataset["train"]) // BATCH_SIZE
    # otherwise transformers will remove "labels" item for some reason
    # remove_unused_columns=False,
    report_to=[],
)
    
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    # train_dataset=lm_dataset["train"],
    # eval_dataset=lm_dataset["test"],
)

curriculum = Curriculum()
total_examples = 0
# while True:
for _ in range(1):
    task_lenghts = curriculum.sample_indexes(256)
    trainer.train_dataset = TasksDataset(tokenizer, task_lenghts)
    total_examples += len(trainer.train_dataset)
    trainer.train()

    # for each steps length, check whether model answers correctly
    task_steps_limit = len(curriculum.avg_scores)
    task_lenghts = list(range(task_steps_limit))
    eval_dataset = TasksDataset(tokenizer, task_lenghts)

    scores = [evaluate_example(model, ex) for ex in eval_dataset]
    unmasked, masked = scores[: len(scores) // 2], scores[len(scores) // 2 :]
    curriculum.update_scores(masked)

    print(
        f"{total_examples:9}  seq.len.: {task_steps_limit:3}  "
        + get_accuracy_bar(scores)
    )
    # wandb.log(dict(total_examples=total_examples, num_solved=sum(scores)))

    if np.mean(scores) > 0.9:
        # all answers were correct, so increase difficulty level
        curriculum.increment_limit()

# %%
# %%

# %%

# %%

# %%

# task_lenghts = list(range(5))
# eval_dataset = TasksDataset(tokenizer, task_lenghts)
ex = eval_dataset[0]

split_index = torch.where(ex["input_ids"] == answer_token)[0] + 1
task_ids = ex["input_ids"][:split_index].to("cuda")
# reasoning_ids = ex["input_ids"][split_index:].to("cuda")
target_output = ex["input_ids"]
task_len = len(task_ids)
out = model.generate(
    input_ids=task_ids.reshape(1, -1),
    attention_mask=ex["attention_mask"][:task_len].reshape(1, -1).to("cuda"),
    max_length=len(target_output),
    temperature=1,
    pad_token_id=tokenizer.pad_token_id,
)
out[0]
