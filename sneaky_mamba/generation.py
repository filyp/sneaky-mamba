# %%
import random

highest_allowed_x_value = 9

possible_steps = (
    (" +=1", lambda x: x + 1),
    (" +=2", lambda x: x + 2),
    (" +=3", lambda x: x + 3),
    (" +=4", lambda x: x + 4),
    # not having subtraction, makes % the only way to go down, forcing more of it
    # (" -=1", lambda x: x - 1),
    # (" -=2", lambda x: x - 2),
    # (" -=3", lambda x: x - 3),
    # (" -=4", lambda x: x - 4),
    (" *=2", lambda x: x * 2),
    (" *=3", lambda x: x * 3),
    (" mod2", lambda x: x % 2),
    (" mod5", lambda x: x % 5),
)


def generate_task(num_of_steps):
    """Generates a sequential computation task of given length.

    Resulting task looks like this:
    1 *=3 mod2 mod5 *=3 mod5 +=1 *=2 +=1 mod2 +=3
    The reasoning:
    1 *=3 3 mod2 1 mod5 1 *=3 3 mod5 3 +=1 4 *=2 8 +=1 9 mod2 1 +=3 4
    The masked reasoning:
    1 *=3 0 mod2 0 mod5 0 *=3 0 mod5 0 +=1 0 *=2 0 +=1 0 mod2 0 +=3 4
    
    0 is used as the mask, to keep the same number of tokens in both reasonings.

    Returns:
    str: text of the task
    str: reasoning text
    str: reasoning text with intermediate values masked
    """
    task_text = "1"
    reasoning_text = "1"
    masked_reasoning_text = "1"
    value = 1

    for _ in range(num_of_steps):
        # randomly choose a step that satisfies the conditions
        while True:
            candidate_text, candidate_func = random.choice(possible_steps)
            new_value = candidate_func(value)
            if 1 <= new_value <= highest_allowed_x_value:
                break

        # update the task and reasoning texts
        value = new_value
        task_text += candidate_text
        reasoning_text += f"{candidate_text} {value}"
        masked_reasoning_text += f"{candidate_text} 0"

    # insert the final value into the reasoning text
    masked_reasoning_text = masked_reasoning_text[:-1] + str(value)

    return task_text, reasoning_text, masked_reasoning_text
