# %%
import os
import sys
from pathlib import Path

import numpy as np
import torch
import wandb
from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
from transformers import AutoTokenizer, TrainingArguments
from utils.generation import DirectTasksDataset, test_doublethink_tokenization
from utils.switcher_model import Switcher
from utils.train_helpers import Curriculum, DirectReasoningTrainer, get_accuracy_bar

os.environ["TOKENIZERS_PARALLELISM"] = "true"

model = Switcher.from_pretrained("state-spaces/mamba-370m").to("cuda")
tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
tokenizer.eos_token = "<|endoftext|>"
tokenizer.pad_token = tokenizer.eos_token
test_doublethink_tokenization(tokenizer)

wandb.login()
wandb.init(project="sneaky-mamba", name="switcher_direct_x5")

# use only the first few layers out of 24
model.layers = model.layers[:1]


def evaluate_example(model, ex):
    input_ids = ex["input_ids"].reshape(1, -1).to("cuda")
    labels = ex["labels"].to("cuda")
    logits = model(input_ids=input_ids)
    output_ids = logits.argmax(axis=2).flatten()
    # tokenizer.decode(output_ids.flatten())
    return all(labels == output_ids)


trainer = DirectReasoningTrainer(
    model=model,
    tokenizer=tokenizer,
    args=TrainingArguments(
        disable_tqdm=True,  # This disables the progress bars
        learning_rate=1e-4,
        num_train_epochs=1,
        per_device_train_batch_size=256,
        gradient_accumulation_steps=1,
        # dataloader_num_workers=2,
        optim="adamw_torch",
        output_dir="out",
        # weight_decay=1e-2,
        # otherwise transformers will remove "labels" item for some reason
        remove_unused_columns=False,
        report_to=[],
    ),
)
trainer.answer_token = tokenizer.encode("\n")[0]
# disable log printing
# trainer.log = lambda logs: None

# %%
curriculum = Curriculum()
# for _ in range(1):
#     curriculum.increment_limit()
total_examples = 0
while True:
    # for _ in range(5):
    task_lenghts = curriculum.sample_indexes(trainer.args.per_device_train_batch_size)
    trainer.train_dataset = DirectTasksDataset(tokenizer, task_lenghts)
    total_examples += len(trainer.train_dataset)
    trainer.train()

    # for each steps length, check whether model answers correctly
    task_steps_limit = len(curriculum.avg_scores)
    task_lenghts_eval = list(range(task_steps_limit))
    eval_dataset = DirectTasksDataset(tokenizer, task_lenghts_eval)
    scores = [evaluate_example(model, ex) for ex in eval_dataset]
    curriculum.update_scores(scores)
    print(
        f"{total_examples:9}  seq.len.: {task_steps_limit:3}  "
        + get_accuracy_bar(scores)
    )
    stats = dict(
        total_examples=total_examples,
        num_solved=sum(scores),
        training_loss=trainer.state.log_history[-1]["train_loss"],
        task_steps_limit=task_steps_limit,
        avg_task_steps=np.mean(task_lenghts),
    )
    wandb.log(stats)

    # if np.mean(scores) > 0.9:
    #     # all answers were correct, so increase difficulty level
    # choose difficulty level to be just a bit longer than the longest solved
    lens_solved = np.where(scores)[0]
    longest_solved = lens_solved[-1] if len(lens_solved) > 0 else 0
    curriculum.increment_limit()
    curriculum.avg_scores = curriculum.avg_scores[: longest_solved + 2]

    if task_steps_limit >= 100 or total_examples > 800000:
        # that's enough
        break


# %%

# %%
task_lenghts = [2]
dataset = DirectTasksDataset(tokenizer, task_lenghts)
batch = dataset[:]
inputs = batch["input_ids"].to("cuda")
labels = batch["labels"].to("cuda")

# %%
print(tokenizer.decode(inputs[0]))
print(tokenizer.decode(labels[0]))
import torch
import torch.nn as nn
import torch.nn.functional as F
