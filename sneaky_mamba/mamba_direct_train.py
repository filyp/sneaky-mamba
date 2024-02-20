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
wandb.init(project="sneaky-mamba", name="mamba_direct_train")

# %%
# use only the first few layers out of 24
model._modules["backbone"].layers = model._modules["backbone"].layers[:1]


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
        per_device_train_batch_size=512,
        gradient_accumulation_steps=1,
        # dataloader_num_workers=2,
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
    # for _ in range(400):
        task_lenghts = curriculum.sample_indexes(trainer.args.per_device_train_batch_size)
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
    # wandb.finish()
    pass


# %%

# %%
task_lenghts = list(range(10))
eval_dataset = DirectTasksDataset(tokenizer, task_lenghts)
ex = eval_dataset[-2]
ex
out = model(input_ids=ex["input_ids"].reshape(1, -1).to("cuda"))
ans = out.logits.argmax(axis=2)
# %%
print(tokenizer.decode(ex["input_ids"]))
print(tokenizer.decode(ex["labels"]))
print(tokenizer.decode(ans.flatten()))
# %%
model._modules["backbone"].layers[0].mixer
# model._modules["backbone"]
# model._modules["backbone"].layers[0].mixer.x_proj.weight

# %%

# %%

# %%
shape = model._modules["backbone"].layers[0].mixer.in_proj.weight.shape
print(shape)
noise = torch.randn(shape) * 5e-2
noise = noise.to("cuda")
new_weight = model._modules["backbone"].layers[0].mixer.in_proj.weight + noise
new_weight = new_weight.to(torch.bfloat16)
model._modules["backbone"].layers[0].mixer.in_proj.weight = torch.nn.Parameter(new_weight)

# %%
shape = model._modules["backbone"].layers[0].mixer.x_proj.weight.shape
print(shape)
noise = torch.randn(shape) * 5e-2
noise = noise.to("cuda")
new_weight = model._modules["backbone"].layers[0].mixer.x_proj.weight + noise
new_weight = new_weight.to(torch.bfloat16)
model._modules["backbone"].layers[0].mixer.x_proj.weight = torch.nn.Parameter(new_weight)

# %%
shape = model._modules["backbone"].layers[0].mixer.dt_proj.weight.shape
print(shape)
noise = torch.randn(shape) * 5e-2
noise = noise.to("cuda")
new_weight = model._modules["backbone"].layers[0].mixer.dt_proj.weight + noise
new_weight = new_weight.to(torch.bfloat16)
model._modules["backbone"].layers[0].mixer.dt_proj.weight = torch.nn.Parameter(new_weight)

# %%
shape = model._modules["backbone"].layers[0].mixer.out_proj.weight.shape
print(shape)
noise = torch.randn(shape) * 5e-2
noise = noise.to("cuda")
new_weight = model._modules["backbone"].layers[0].mixer.out_proj.weight + noise
new_weight = new_weight.to(torch.bfloat16)
model._modules["backbone"].layers[0].mixer.out_proj.weight = torch.nn.Parameter(new_weight)

# %%
model._modules["backbone"].layers[0].mixer.conv1d.weight.shape
