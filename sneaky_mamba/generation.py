# %%
import random

import torch

possible_steps = (
    ("+=1", lambda x: x + 1),
    ("+=2", lambda x: x + 2),
    ("+=3", lambda x: x + 3),
    ("+=4", lambda x: x + 4),
    # not having subtraction, makes % the only way to go down, forcing more of it
    # ("-=1", lambda x: x - 1),
    # ("-=2", lambda x: x - 2),
    # ("-=3", lambda x: x - 3),
    # ("-=4", lambda x: x - 4),
    ("*=2", lambda x: x * 2),
    ("*=3", lambda x: x * 3),
    ("mod2", lambda x: x % 2),
    ("mod5", lambda x: x % 5),
)


def generate_task(num_of_steps, highest_allowed_value=9):
    """Generates a sequential computation task of given length.

    Resulting task looks like this:
    1 *=3 mod2 mod5 *=3 mod5 +=1 *=2 +=1 mod2 +=3
    The reasoning:
    1 *=3 3 mod2 1 mod5 1 *=3 3 mod5 3 +=1 4 *=2 8 +=1 9 mod2 1 +=3 4

    They are given as lists.

    Intemediate values will stay in the range:
    1 - highest_allowed_value

    Returns:
    str: text of the task
    str: reasoning text
    """
    task_texts = ["1"]
    reasoning_texts = ["1"]
    value = 1

    for _ in range(num_of_steps):
        # randomly choose a step that satisfies the conditions
        while True:
            operation_text, func = random.choice(possible_steps)
            new_value = func(value)
            if 1 <= new_value <= highest_allowed_value:
                break

        # update the task and reasoning texts
        value = new_value
        task_texts.append(operation_text)
        reasoning_texts.append(operation_text)
        reasoning_texts.append(str(value))

    return task_texts, reasoning_texts


def mask_all_values(reasoning):
    reasoning[2:-2:2] = "0" * len(reasoning[2:-2:2])
    return reasoning


class TasksDataset(torch.utils.data.Dataset):
    def __init__(self, tokenizer, mask, num_examples_per_num_steps):
        """
        num_examples_per_num_steps: list of tuples (num_steps, num_examples)
        """
        training_texts = []
        self.task_ids = []
        self.reasoning_ids = []
        for num_steps, num_examples in num_examples_per_num_steps:
            for _ in range(num_examples):
                task, reasoning = generate_task(num_steps)
                if mask:
                    reasoning = mask_all_values(reasoning)

                task_text = " ".join(task) + "\nanswer\n"
                reasoning_text = " ".join(reasoning)
                task_ids = tokenizer.encode(task_text, return_tensors="pt")[0]
                reasoning_ids = tokenizer.encode(reasoning_text, return_tensors="pt")[0]
                self.task_ids.append(task_ids)
                self.reasoning_ids.append(reasoning_ids)
                # training texts will be batch tokenized later, to have padding
                training_texts.append(task_text + reasoning_text)

        self.input_ids = tokenizer(
            training_texts, padding=True, return_tensors="pt"
        ).input_ids

        # test that each has the same num of examples
        assert len(self.input_ids) == len(self.task_ids) == len(self.reasoning_ids)

        # test that joining task and reasoning works correctly
        _random_index = random.randint(0, len(self) - 1)
        _ex = self[_random_index]
        _reconstructed = torch.cat([_ex["task_ids"], _ex["reasoning_ids"]])
        assert all(_ex["input_ids"][: len(_reconstructed)] == _reconstructed)

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i):
        return dict(
            input_ids=self.input_ids[i],  # task and reasoning, joined and padded
            task_ids=self.task_ids[i],
            reasoning_ids=self.reasoning_ids[i],
        )
