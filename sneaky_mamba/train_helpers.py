import os

import torch
from transformers import Trainer


class DirectReasoningTrainer(Trainer):
    answer_token: int

    def compute_loss(self, model, inputs, return_outputs=False):
        input_ids = inputs.pop("input_ids")
        labels = inputs.pop("labels")

        # batched generation
        logits = model(input_ids).logits

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
        lm_logits = model(input_ids).logits

        labels = input_ids.to(lm_logits.device)
        shift_logits = lm_logits[:, :-1, :].contiguous()
        labels = labels[:, 1:].contiguous()

        # cut out the task part (the part before "answer")
        reasoning_shift_logits = []
        reasoning_labels = []
        for ex_shift_logits, ex_labels in zip(shift_logits, labels):
            # find the indexes of the "answer" token
            answer_index = int(torch.where(ex_labels == self.answer_token)[0])
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
