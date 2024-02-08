# %%
import os
import sys
from pathlib import Path

import numpy as np
import torch
import wandb
from generation import DirectTasksDataset, test_tokenization
from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
from train_helpers import Curriculum, DirectReasoningTrainer, get_accuracy_bar
from transformers import AutoTokenizer, TrainingArguments

os.environ["TOKENIZERS_PARALLELISM"] = "true"

model = MambaLMHeadModel.from_pretrained(
    "state-spaces/mamba-130m", dtype=torch.bfloat16, device="cuda"
)
tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
tokenizer.eos_token = "<|endoftext|>"
tokenizer.pad_token = tokenizer.eos_token
test_tokenization(tokenizer)

wandb.login()
wandb.init(project="sneaky-mamba", name="mamba_direct_train_" + sys.argv[1])

# %%
# use only the first few layers out of 24
model._modules["backbone"].layers = model._modules["backbone"].layers[:3]


# %%
def evaluate_example(model, ex):
    input_ids = ex["input_ids"].reshape(1, -1).to("cuda")
    labels = ex["labels"].to("cuda")
    logits = model(input_ids=input_ids).logits
    output_ids = logits.argmax(axis=2).flatten()
    # tokenizer.decode(output_ids.flatten())
    return all(labels == output_ids)


# %%
trainer = DirectReasoningTrainer(
    model=model,
    tokenizer=tokenizer,
    args=TrainingArguments(
        disable_tqdm=True,  # This disables the progress bars
        learning_rate=5e-4,
        num_train_epochs=1,
        per_device_train_batch_size=64,
        gradient_accumulation_steps=1,
        dataloader_num_workers=2,
        optim="adamw_torch",
        output_dir="out",
        weight_decay=1e-2,
        # otherwise transformers will remove "labels" item for some reason
        remove_unused_columns=False,
        report_to=[],
    ),
)
trainer.answer_token = tokenizer.encode("\n")[0]
# disable log printing
trainer.log = lambda logs: None

try:
    curriculum = Curriculum()
    total_examples = 0
    while True:
        task_lenghts = curriculum.sample_indexes(256)
        trainer.train_dataset = DirectTasksDataset(tokenizer, task_lenghts)
        total_examples += len(trainer.train_dataset)
        trainer.train()

        # for each steps length, check whether model answers correctly
        task_steps_limit = len(curriculum.avg_scores)
        task_lenghts = list(range(task_steps_limit))
        eval_dataset = DirectTasksDataset(tokenizer, task_lenghts)
        scores = [evaluate_example(model, ex) for ex in eval_dataset]
        curriculum.update_scores(scores)
        print(
            f"{total_examples:9}  seq.len.: {task_steps_limit:3}  "
            + get_accuracy_bar(scores)
        )
        wandb.log(dict(total_examples=total_examples, num_solved=sum(scores)))

        if np.mean(scores) > 0.9:
            # all answers were correct, so increase difficulty level
            curriculum.increment_limit()
except KeyboardInterrupt:
    wandb.finish()


# %%
