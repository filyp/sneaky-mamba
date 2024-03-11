# %%
import os
import sys
from pathlib import Path

import numpy as np
import torch
import wandb
from generation import DoublethinkTasksDataset, test_doublethink_tokenization
from model import Switcher
from train_helpers import Curriculum, ReasoningTrainer, get_accuracy_bar
from transformers import AutoTokenizer, TrainingArguments

os.environ["TOKENIZERS_PARALLELISM"] = "true"

model = Switcher.from_pretrained("state-spaces/mamba-370m").to("cuda")
tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
tokenizer.eos_token = "<|endoftext|>"
tokenizer.pad_token = tokenizer.eos_token
test_doublethink_tokenization(tokenizer)

# wandb.login()
# wandb.init(project="sneaky-mamba", name="switcher_doublethink_train_silu_l1")

# use only the first few layers out of 24
model.layers = model.layers[:1]


# def evaluate_example(model, ex):
#     target_output = torch.cat([ex["task_ids"], ex["reasoning_ids"]]).to("cuda")
#     task_len = len(ex["task_ids"])
#     out = model.generate(
#         input_ids=ex["task_ids"].reshape(1, -1).to("cuda"),
#         attention_mask=ex["attention_mask"][:task_len].reshape(1, -1).to("cuda"),
#         max_length=len(target_output),
#         temperature=1,
#         pad_token_id=tokenizer.pad_token_id,
#     )
#     if len(out[0]) != len(target_output):
#         return False
#     return all(out[0] == target_output)


def evaluate_example(model, ex):
    # todo trim length to the task length
    # also
    # todo maybe not use task_ids, but just split, to hopefully avoid trainer errors saying to pad
    full = ex["input_ids"].to("cuda")
    # trim last token
    input_ids = full[:-1]
    labels = full[1:]
    logits = model(input_ids=input_ids.reshape(1, -1))[0]
    # probabilities = torch.softmax(logits, dim=1)
    # out = torch.multinomial(probabilities, 1)
    out = logits.argmax(axis=-1)
    # tokenizer.decode(output_ids.flatten())
    return all(out.flatten() == labels.flatten())


# %%
trainer = ReasoningTrainer(
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
    # task_lenghts = curriculum.sample_indexes(trainer.args.per_device_train_batch_size)
    task_lenghts = [0] * trainer.args.per_device_train_batch_size
    trainer.train_dataset = DoublethinkTasksDataset(tokenizer, task_lenghts)
    total_examples += len(trainer.train_dataset)
    trainer.train()

    # for each steps length, check whether model answers correctly
    task_steps_limit = len(curriculum.avg_scores)
    task_lenghts_eval = list(range(task_steps_limit))
    eval_dataset = DoublethinkTasksDataset(tokenizer, task_lenghts_eval)
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
    # wandb.log(stats)

    # if np.mean(scores) > 0.9:
    #     # all answers were correct, so increase difficulty level
    # choose difficulty level to be just a bit longer than the longest solved
    # lens_solved = np.where(scores)[0]
    # longest_solved = lens_solved[-1] if len(lens_solved) > 0 else 0
    # curriculum.increment_limit()
    # curriculum.avg_scores = curriculum.avg_scores[: longest_solved + 1]

    if task_steps_limit >= 100 or total_examples > 1e6:
        # that's enough
        break


# %%

# %%
task_lenghts = [1, 2]
dataset = DoublethinkTasksDataset(tokenizer, task_lenghts)
batch = dataset[:]
ex = dataset[0]
# %%

# %%

output_ids = logits.argmax(axis=2).flatten()
# %%
inputs = batch["input_ids"].to("cuda")
# %%
while True:
    out = model(inputs)
    logits = out[:, -1, :]
    probabilities = torch.softmax(logits, dim=1)
    # sample from the distribution with temperature 1
    sampled_tokens = torch.multinomial(probabilities, 1)
    inputs = torch.cat([inputs, sampled_tokens], dim=1)
    if inputs.shape[1] >= 100:
        break
# %%

inputs = batch["input_ids"].to("cuda")
labels = batch["labels"].to("cuda")

# %%
print(tokenizer.decode(inputs[0]))
print(tokenizer.decode(labels[0]))
import torch
import torch.nn as nn
import torch.nn.functional as F
