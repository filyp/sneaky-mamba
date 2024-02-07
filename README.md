# TODO

- [X] place higher loss on the last token (the final answer) to maybe speedup training
- [X] smaller mamba (less layers) to train faster
- [ ] train transformer (at least try to)
- [ ] test whether the tasks are actually fully sequential or there are heuristics for solving them
- [X] check that in AutoTokenizer.from_pretrained('EleutherAI/gpt-neox-20b') my task is still 1+2*num_steps tokens

# Info

Mamba trained: 130m stripped to 6 layers, so about 30m.

# Resources

- for interventions into mamba architecture, this will be easiest [mamba-minimal](https://github.com/johnma2006/mamba-minimal)
  - it's not efficient though, so later modify code in original repo, and then sanity check that it behaves the same as modified mamba-minimal

# Useful snippets

```bash
/bin/sh -c "cd /workspace && git clone https://github.com/filyp/sneaky-mamba.git && cd sneaky-mamba && pip install -r requirements.txt && tail -f /dev/null"
```

```
cd /workspace && git clone https://github.com/filyp/sneaky-mamba.git && cd sneaky-mamba && pip install -r requirements.txt
```

old loss:

```
    def compute_loss(self, model, inputs, return_outputs=False):
        input_ids = inputs.pop("input_ids")
        lm_logits = model(input_ids).logits

        labels = input_ids.to(lm_logits.device)
        shift_logits = lm_logits[:, :-1, :].contiguous()
        labels = labels[:, 1:].contiguous()

        loss_fct = torch.nn.CrossEntropyLoss()
        lm_loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), labels.view(-1))

        return lm_loss
```

fancy loss, with separate loss for final answer:

```
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
        final_answer_shift_logits = []
        final_answer_labels = []
        for ex_shift_logits, ex_labels in zip(shift_logits, labels):
            # find the indexes of the "answer" token
            answer_index = torch.where(ex_labels == answer_token)[0]
            answer_index = int(answer_index)
            # cut out the task part
            reasoning_shift_logits.append(ex_shift_logits[answer_index:-1])
            reasoning_labels.append(ex_labels[answer_index:-1])
            # loss for the final answer will be calculated separately
            final_answer_shift_logits.append(ex_shift_logits[-1:])
            final_answer_labels.append(ex_labels[-1:])

        # calculate loss only for the tokens after "answer"
        loss_fct = torch.nn.CrossEntropyLoss()
        reasoning_lm_loss = loss_fct(
            torch.cat(reasoning_shift_logits),
            torch.cat(reasoning_labels),
        )
        loss_fct = torch.nn.CrossEntropyLoss()
        final_answer_lm_loss = loss_fct(
            torch.cat(final_answer_shift_logits),
            torch.cat(final_answer_labels),
        )
        return reasoning_lm_loss * (1 - final_answer_loss_contribution) + final_answer_lm_loss * final_answer_loss_contribution


```
