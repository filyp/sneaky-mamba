# TODO

- [X] check that in AutoTokenizer.from_pretrained('EleutherAI/gpt-neox-20b') my task is still 1+2*num_steps tokens


# Resources

- for interventions into mamba architecture, this will be easiest [mamba-minimal](https://github.com/johnma2006/mamba-minimal)
  - it's not efficient though, so later modify code in original repo, and then sanity check that it behaves the same as modified mamba-minimal
