# %%
import os
import torch
import numpy as np
from transformers import TrainingArguments, GPT2Tokenizer, GPT2LMHeadModel

from generation import DirectTasksDataset, test_tokenization
from train_helpers import DirectReasoningTrainer, get_accuracy_bar, Curriculum

os.environ["TOKENIZERS_PARALLELISM"] = "true"

model = GPT2LMHeadModel.from_pretrained('gpt2').to("cuda")
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

tokenizer.eos_token = "<|endoftext|>"
tokenizer.pad_token = tokenizer.eos_token
test_tokenization(tokenizer)

import wandb
wandb.login()
wandb.init(project="sneaky-mamba", name="transformer_direct_train")

# %%
# use only the first few layers out of 12
model._modules["transformer"]._modules["h"] = model._modules["transformer"]._modules["h"][:1]

# %%
def evaluate_example(model, ex, verbose=False):
    input_ids = ex["input_ids"].reshape(1, -1).to("cuda")
    # attention_mask=ex["attention_mask"].reshape(1, -1).to("cuda")
    labels = ex["labels"].to("cuda")
    logits = model(input_ids=input_ids).logits
    output_ids = logits.argmax(axis=2).flatten()
    if verbose:
        print("input ", tokenizer.decode(input_ids.flatten()))
        print("output", tokenizer.decode(output_ids.flatten()))
        print("labels", tokenizer.decode(labels.flatten()))
    return all(labels == output_ids)

# %%
task_steps_limit = 1

trainer = DirectReasoningTrainer(
    model=model,
    tokenizer=tokenizer,
    args=TrainingArguments(
        disable_tqdm=True,  # This disables the progress bars
        learning_rate=5e-4,
        num_train_epochs=1,
        per_device_train_batch_size=512,
        gradient_accumulation_steps=1,
        dataloader_num_workers=2,
        optim="adamw_torch",
        output_dir="out",
        weight_decay=1e-2,
    ),
)
trainer.answer_token = tokenizer.encode("\n")[0]
# disable log printing
trainer.log = lambda logs: None

curriculum = Curriculum()
total_examples = 0
while True:
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
    # evaluate_example(model, eval_dataset[-1], verbose=True)

    if np.mean(scores) > 0.9:
        # all answers were correct, so increase difficulty level
        curriculum.increment_limit()

# %%



