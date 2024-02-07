# %%
import random


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
