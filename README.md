# TODO

- [ ] place higher loss on the last token (the final answer) to maybe speedup training
- [ ] parallelize accuracy eval (first, it's good to have task dataset in dataset creation) (but padding is problematic)
  - this takes about 1/4 of training time currently with epoch size 160, probably less with 320
- [X] check that in AutoTokenizer.from_pretrained('EleutherAI/gpt-neox-20b') my task is still 1+2*num_steps tokens


# Resources

- for interventions into mamba architecture, this will be easiest [mamba-minimal](https://github.com/johnma2006/mamba-minimal)
  - it's not efficient though, so later modify code in original repo, and then sanity check that it behaves the same as modified mamba-minimal


# Useful snippets

```bash
/bin/sh -c "cd /workspace && git clone https://github.com/filyp/sneaky-mamba.git && cd sneaky-mamba && pip install -r requirements.txt && tail -f /dev/null"
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