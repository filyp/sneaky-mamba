# %%
import os
import sys
from pathlib import Path

import numpy as np
import torch
import wandb
from utils.generation import DoublethinkTasksDataset, test_doublethink_tokenization
from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
from utils.switcher_model import Switcher
from utils.train_helpers import Curriculum, ReasoningTrainer, get_accuracy_bar
from transformers import (
    AutoTokenizer,
    GPT2LMHeadModel,
    GPT2Tokenizer,
    TrainingArguments,
)

os.environ["TOKENIZERS_PARALLELISM"] = "true"

architecture = "transformer"
num_layers = 4
min_task_len = 1

match architecture:
    case "mamba":
        model = MambaLMHeadModel.from_pretrained(
            "state-spaces/mamba-130m", dtype=torch.bfloat16, device="cuda"
        )
        # use only the first few layers out of 24
        model._modules["backbone"].layers = model._modules["backbone"].layers[
            :num_layers
        ]
        tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
    case "switcher":
        model = Switcher.from_pretrained("state-spaces/mamba-370m").to("cuda")
        model.layers = model.layers[:num_layers]
        tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
    case "transformer":
        model = GPT2LMHeadModel.from_pretrained("gpt2").to("cuda")
        model._modules["transformer"]._modules["h"] = model._modules[
            "transformer"
        ]._modules["h"][:num_layers]
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    case _:
        raise ValueError(f"Unknown architecture: {architecture}")

tokenizer.eos_token = "<|endoftext|>"
tokenizer.pad_token = tokenizer.eos_token
answer_token = tokenizer.encode("\n")[0]
test_doublethink_tokenization(tokenizer)

wandb.login()
wandb.init(project="sneaky-mamba", name=f"{architecture}_doublethink_l{num_layers}")


def evaluate_example(model, ex):
    full = ex["input_ids"].to("cuda")
    # trim last token
    input_ids = full[:-1]
    logits = model(input_ids=input_ids.reshape(1, -1))[0]

    # probabilities = torch.softmax(logits, dim=1)
    # out = torch.multinomial(probabilities, 1).flatten()
    out = logits.argmax(axis=-1).flatten()
    answer_index = int(torch.where(full == answer_token)[0][0])
    completion = out[answer_index:]
    target = full[answer_index + 1 :]

    return all(completion == target)


# %%
trainer = ReasoningTrainer(
    model=model,
    tokenizer=tokenizer,
    args=TrainingArguments(
        disable_tqdm=True,  # This disables the progress bars
        learning_rate=1e-4,
        num_train_epochs=1,
        per_device_train_batch_size=64,
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
trainer.answer_token = answer_token
# disable log printing
# trainer.log = lambda logs: None

# %%
curriculum = Curriculum()
for _ in range(min_task_len - 1):
    curriculum.increment_limit()
total_examples = 0
while True:
    # for _ in range(30):
    task_lenghts = curriculum.sample_indexes(trainer.args.per_device_train_batch_size)
    trainer.train_dataset = DoublethinkTasksDataset(tokenizer, task_lenghts)
    total_examples += len(trainer.train_dataset)
    trainer.train()

    # for each steps length, check whether model answers correctly
    task_steps_limit = len(curriculum.avg_scores)
    task_lenghts_eval = list(range(1, task_steps_limit + 1))
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
    wandb.log(stats)

    # choose difficulty level to be just a bit longer than the longest solved
    lens_solved = np.where(scores)[0] + 1
    longest_solved = lens_solved[-1] if len(lens_solved) > 0 else 0
    max_task_len = max(longest_solved + 1, min_task_len)
    curriculum.increment_limit()
    curriculum.avg_scores = curriculum.avg_scores[:max_task_len]

    if task_steps_limit >= 50 or total_examples > 2e6:
        # that's enough
        break

# %%
