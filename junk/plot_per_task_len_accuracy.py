# that was pasted after all code in doublethink_train.py
# %%
all_scores = []
for _ in range(100):
    task_steps_limit = len(curriculum.avg_scores)
    task_lenghts_eval = list(range(1, task_steps_limit + 1))
    eval_dataset = DoublethinkTasksDataset(tokenizer, task_lenghts_eval)
    scores = [evaluate_example(model, ex) for ex in eval_dataset]
    all_scores.append(scores)

# %%
all_scores = np.array(all_scores)
# %%
means = all_scores.mean(axis=0)
sems = all_scores.std(axis=0) / np.sqrt(all_scores.shape[0])

# plot as shape[1] bar plots, from 0 to 1
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 5))
plt.bar(range(len(means)), means, yerr=sems, capsize=5)
plt.xlabel("Task length")
plt.ylabel("Accuracy")
plt.title("Accuracy of the model on the validation set (with SEMs)")
plt.show()
