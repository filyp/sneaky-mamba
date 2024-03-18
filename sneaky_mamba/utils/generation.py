# %%
import random

import torch

possible_steps = (
    ("one", lambda x: x + 1),
    ("two", lambda x: x + 2),
    ("three", lambda x: x + 3),
    ("four", lambda x: x + 4),
    ("double", lambda x: x * 2),
    ("triple", lambda x: x * 3),
    ("quad", lambda x: x * 4),
)


def generate_task_abstract(num_of_steps, hidden_modulus=5, overt_modulus=5):
    """Generates a sequential computation task of given length.
    
    Hidden task if to perform the operations on the hidden values, and
    overt task is to add the number of letters in the operation names.

    Intemediate values will stay in the range:
    0 .. hidden_modulus - 1
    0 .. overt_modulus - 1

    We start the task from the value of 0.

    For example, for num_of_steps=3, it may return:
    operations = ["one", "two", "triple"]
    hidden_values = [0, 1, 3, 4]
    overt_values = [0, 3, 1, 2]
    """
    # start_value = random.choice(range(hidden_modulus))
    operations = []
    hidden_values = [0]
    overt_values = [0]
    assert num_of_steps >= 0

    for _ in range(num_of_steps):
        # randomly choose a step
        operation_text, func = random.choice(possible_steps)
        new_hidden_value = func(hidden_values[-1]) % hidden_modulus
        new_overt_value = (overt_values[-1] + len(operation_text)) % overt_modulus

        # update the task and values
        operations.append(operation_text)
        hidden_values.append(new_hidden_value)
        overt_values.append(new_overt_value)

    return operations, hidden_values, overt_values


def generate_doublethink_task(num_of_steps):
    ops, hidden_vals, overt_vals = generate_task_abstract(num_of_steps)

    task = " ".join(str(o) for o in ops) + "\n"
    task = " " + task  # to have consistent tokenization

    overt_reasoning = []
    for op, overt_val in zip(ops, overt_vals[1:]):
        overt_reasoning.append(op)
        overt_reasoning.append(overt_val)
    overt_reasoning = " ".join(str(r) for r in overt_reasoning)

    hidden_outcome = str(hidden_vals[-1])

    reasoning = " " + overt_reasoning + "\n " + hidden_outcome
    return task, reasoning


# def generate_task_text(masked, num_of_steps):
#     ops, vals, _ = generate_task_abstract(num_of_steps)

#     if masked:
#         # mask intermediate values
#         for i in range(1, len(vals) - 1):
#             vals[i] = "_"

#     # generate interleaved reasoning of the form:
#     # [1, 'two', 3, 'two', 5, 'triple', 1, 'triple', 3, 'triple', 2]
#     reasoning = []
#     for op, val in zip(ops, vals):
#         reasoning.append(op)
#         reasoning.append(val)

#     # construct task and reasoning texts
#     task_text = "hide " if masked else "show "
#     task_text += " ".join(str(o) for o in ops) + "\n"
#     reasoning_text = " ".join(str(r) for r in reasoning)
#     return task_text, reasoning_text


class DirectTasksDataset(torch.utils.data.Dataset):
    def __init__(self, tokenizer, task_lenghts):
        """
        num_examples_per_num_steps: list of tuples (num_steps, num_examples)
        """
        inputs = []
        labels = []
        for task_length in task_lenghts:
            ops, vals, _ = generate_task_abstract(task_length)
            vals = vals[1:]  # skip the initial 0
            # ops.insert(0, "show")
            assert len(ops) == len(vals)
            inputs.append(" " + " ".join(str(o) for o in ops))
            labels.append(" " + " ".join(str(v) for v in vals))
            # initial space is to have consistent tokenization

        tokens = tokenizer(inputs, padding=True, return_tensors="pt")
        self.input_ids = tokens.input_ids
        self.attention_masks = tokens.attention_mask
        self.labels = tokenizer(labels, padding=True, return_tensors="pt").input_ids

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i):
        return dict(
            input_ids=self.input_ids[i],  # task and reasoning, joined and padded
            attention_mask=self.attention_masks[i],
            labels=self.labels[i],
        )


# class TasksDataset(torch.utils.data.Dataset):
#     def __init__(self, tokenizer, task_lenghts):
#         training_texts = []
#         self.task_ids = []
#         self.reasoning_ids = []
#         for masked in [False, True]:
#             for task_length in task_lenghts:
#                 task, reasoning = generate_task_text(masked, task_length)

#                 self.task_ids.append(tokenizer.encode(task, return_tensors="pt")[0])
#                 self.reasoning_ids.append(
#                     tokenizer.encode(reasoning, return_tensors="pt")[0]
#                 )
#                 # training texts will be batch tokenized later, to have padding
#                 training_texts.append(task + reasoning)

#         tokens = tokenizer(training_texts, padding=True, return_tensors="pt")
#         self.input_ids = tokens.input_ids
#         self.attention_masks = tokens.attention_mask

#         # test that each has the same num of examples
#         assert len(self.input_ids) == len(self.task_ids) == len(self.reasoning_ids)

#         # test that joining task and reasoning works correctly
#         _random_index = random.randint(0, len(self) - 1)
#         _ex = self[_random_index]
#         _reconstructed = torch.cat([_ex["task_ids"], _ex["reasoning_ids"]])
#         assert all(_ex["input_ids"][: len(_reconstructed)] == _reconstructed)

#     def __len__(self):
#         return len(self.input_ids)

#     def __getitem__(self, i):
#         return dict(
#             input_ids=self.input_ids[i],  # task and reasoning, joined and padded
#             attention_mask=self.attention_masks[i],
#             task_ids=self.task_ids[i],
#             reasoning_ids=self.reasoning_ids[i],
#         )


class DoublethinkTasksDataset(torch.utils.data.Dataset):
    def __init__(self, tokenizer, task_lenghts):
        training_texts = []
        self.task_ids = []
        self.reasoning_ids = []
        for task_length in task_lenghts:
            task, reasoning = generate_doublethink_task(task_length)
            training_texts.append(task + reasoning)

        tokens = tokenizer(training_texts, padding=True, return_tensors="pt")
        self.input_ids = tokens.input_ids
        self.attention_masks = tokens.attention_mask  # should be just 1s up to padding

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i):
        return dict(
            input_ids=self.input_ids[i],  # task and reasoning, joined and padded
            attention_mask=self.attention_masks[i],
        )


# def test_tokenization(tokenizer):
#     # make sure that the task is tokenized in a regular minimal way
#     num_steps = 99
#     task, reasoning = generate_task_text(True, num_steps)
#     assert len(tokenizer.encode(task)) == num_steps + 3
#     assert len(tokenizer.encode(reasoning)) == num_steps * 2 + 2
#     task, reasoning = generate_task_text(False, num_steps)
#     assert len(tokenizer.encode(task)) == num_steps + 3
#     assert len(tokenizer.encode(reasoning)) == num_steps * 2 + 2


def test_doublethink_tokenization(tokenizer):
    # make sure that the task is tokenized in a regular minimal way
    num_steps = 99
    task, reasoning = generate_doublethink_task(num_steps)
    assert len(tokenizer.encode(task)) == num_steps + 1, task  # + \n
    assert len(tokenizer.encode(reasoning)) == num_steps * 2 + 2, task  # \n and outcome
    for op, func in possible_steps:
        two_ops = tokenizer.encode(" " + op + " " + op)
        assert len(two_ops) == 2, two_ops
        assert two_ops[0] == two_ops[1], two_ops

# %%
# ensuring no memorization by splitting into train and test based on hash
# on second thought, that's probably too harsh
# bc it cannot learn some one-length operations
# ops, hidden, overt = generate_task_abstract(5)
# # hash them (deteministic)
# # note: python hash is not deterministic
# # use hashlib
# import hashlib

# h = hashlib.sha256()
# h.update(str(ops).encode())
# h.hexdigest()
# # convert to 0 to 1 float
# int(h.hexdigest(), 16) % 2