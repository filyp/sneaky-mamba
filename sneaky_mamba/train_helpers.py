import os
import random

import torch
from transformers import Trainer
import numpy as np
from collections import deque


class DirectReasoningTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        input_ids = inputs.pop("input_ids")
        labels = inputs.pop("labels")

        # batched generation
        output = model(input_ids)
        if hasattr(output, "logits"):
            logits = output.logits
        elif hasattr(output, "last_hidden_state"):
            logits = output.last_hidden_state
        elif isinstance(output, torch.Tensor):
            logits = output
        else:
            raise ValueError(f"Unexpected output type: {type(output)}")

        # calculate loss only for the tokens after "answer"
        loss_fct = torch.nn.CrossEntropyLoss()
        lm_loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))
        return lm_loss

    def save_model(self, output_dir, _internal_call):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        torch.save(self.model.state_dict(), f"{output_dir}/pytorch_model.bin")
        self.tokenizer.save_pretrained(output_dir)


class ReasoningTrainer(Trainer):
    answer_token: int

    def compute_loss(self, model, inputs, return_outputs=False):
        input_ids = inputs.pop("input_ids")

        # batched generation
        output = model(input_ids)
        if hasattr(output, "logits"):
            logits = output.logits
        elif hasattr(output, "last_hidden_state"):
            logits = output.last_hidden_state
        elif isinstance(output, torch.Tensor):
            logits = output
        else:
            raise ValueError(f"Unexpected output type: {type(output)}")

        labels = input_ids.to(logits.device)
        shift_logits = logits[:, :-1, :].contiguous()
        labels = labels[:, 1:].contiguous()

        # cut out the task part (the part before "answer")
        reasoning_shift_logits = []
        reasoning_labels = []
        for ex_shift_logits, ex_labels in zip(shift_logits, labels):
            # find the indexes of the "answer" token
            answer_index = int(torch.where(ex_labels == self.answer_token)[0][0])
            # cut out the task part
            reasoning_shift_logits.append(ex_shift_logits[answer_index + 1 :])
            reasoning_labels.append(ex_labels[answer_index + 1 :])

        # calculate loss only for the tokens after "answer"
        loss_fct = torch.nn.CrossEntropyLoss()
        reasoning_lm_loss = loss_fct(
            torch.cat(reasoning_shift_logits),
            torch.cat(reasoning_labels),
        )
        return reasoning_lm_loss

    def save_model(self, output_dir, _internal_call):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        torch.save(self.model.state_dict(), f"{output_dir}/pytorch_model.bin")
        self.tokenizer.save_pretrained(output_dir)


def get_accuracy_bar(scores):
    accuracy_bar = "«"
    for score in scores:
        if score:
            accuracy_bar += "█"
        else:
            accuracy_bar += " "
    accuracy_bar += "»"
    return accuracy_bar


class Curriculum:
    def __init__(self, hist_len=5, flatten_factor=1):
        """
        For flatten_factor, 0 means flat, 1 means maximally adaptive.
        """
        assert 0 <= flatten_factor <= 1
        self.hist_len = hist_len
        self.avg_scores = []
        self.increment_limit()  # start with one index
        self.score_to_prob_func = lambda x: 1 - flatten_factor * x
        # self.score_to_prob_func = lambda x: 1 - flaten_factor * 4 * (x - 0.5) ** 2

        assert 0 <= self.score_to_prob_func(0) <= 1
        assert 0 <= self.score_to_prob_func(0.5) <= 1
        assert 0 <= self.score_to_prob_func(1) <= 1
    
    def increment_limit(self):
        self.avg_scores.append(deque(maxlen=self.hist_len))
        self.avg_scores[-1].append(False)
        
    def update_scores(self, scores):
        for i, score in enumerate(scores):
            if len(self.avg_scores) <= i:
                self.avg_scores.append(deque(maxlen=self.hist_len))
            self.avg_scores[i].append(score)
    
    def get_avg_scores(self):
        return [np.mean(scores) for scores in self.avg_scores]
            
    def sample_indexes(self, num_samples):
        """
        Use rejection sampling.
        """
        samples = []
        for _ in range(num_samples):
            while True:
                candidate = random.choice(range(len(self.avg_scores)))
                avg_score = np.mean(self.avg_scores[candidate])
                prob = self.score_to_prob_func(avg_score)
                if np.random.random() < prob:
                    # choose that index
                    break
            task_len = candidate + 1
            samples.append(task_len)
        return samples
