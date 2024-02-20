# %%
# # use only the first few layers out of 24
# model._modules["backbone"].layers = model._modules["backbone"].layers[:3]

# %%
# inputs = tokenizer("This is an example.", return_tensors="pt").to("cuda")
# # Feed everything to the model
# outputs = model(inputs["input_ids"])
# output_whole = outputs.logits

# outputs = model(inputs["input_ids"][:, :2])
# output_one = outputs.logits

# # Using the state computed on the first inputs, we will get the same output
# outputs = model(inputs["input_ids"][:, 2:], state=outputs.state)
# output_two = outputs.logits

# torch.allclose(torch.cat([output_one, output_two], dim=1), output_whole, atol=1e-5)


# %%

# def evaluate_example(model, ex):
#     input_ids = ex["input_ids"].reshape(1, -1).to("cuda")
#     labels = ex["labels"].to("cuda")
#     logits = model(input_ids=input_ids).logits
#     output_ids = logits.argmax(axis=2).flatten()
#     # tokenizer.decode(output_ids.flatten())
#     return all(labels == output_ids)

#     args=TrainingArguments(
#         disable_tqdm=True,  # This disables the progress bars
#         learning_rate=5e-4,
#         num_train_epochs=1,
#         per_device_train_batch_size=64,
#         gradient_accumulation_steps=1,
#         dataloader_num_workers=2,
#         optim="adamw_torch",
#         output_dir="out",
#         weight_decay=1e-2,
#         # otherwise transformers will remove "labels" item for some reason
#         remove_unused_columns=False,
#         report_to=[],
#     ),
# # disable log printing
# trainer.log = lambda logs: None


# trainer = ReasoningTrainer(
#     model=model,
#     tokenizer=tokenizer,
#     args=training_args,
# )

# task_lenghts = list(range(5))
# trainer.train_dataset = DirectTasksDataset(tokenizer, task_lenghts)
# trainer.train()
