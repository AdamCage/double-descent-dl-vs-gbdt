---
license: other
license_name: mit-with-model-training-restrictions
license_link: LICENSE
task_categories:
- question-answering
- text-generation
language:
- en
size_categories:
- 100K<n<1M
---

# Measuring Chain-of-Thought Monitorability Transcripts

This dataset contains model transcripts from language models evaluated on [MMLU](https://github.com/hendrycks/test), [BIG-Bench Hard (BBH)](https://github.com/suzgunmirac/BIG-Bench-Hard), and [GPQA Diamond](https://github.com/idavidrein/gpqa). Each sample group includes a baseline response (no cue) paired with five adaptive variations where different cues were injected to test chain-of-thought faithfulness.

We use this dataset to measure how faithfully models represent their reasoning processes in their chain-of-thought outputs. By comparing baseline responses to cued responses, we can assess whether models acknowledge misleading cues and whether their stated reasoning matches their actual decision-making process.

We also evaluate models on how many 'causal factors' are included in their outputs, which is in another dataset linked below. This forms the basis of our 'verbosity' score, which we combine with the faithfulness score to measure monitorability. For full details, please see our paper.

- [arXiv](https://arxiv.org/abs/2510.27378)
- [Project website](https://ajmeek.github.io/cot_monitorability_website/)
- [Causal Factors HuggingFace Dataset](https://huggingface.co/datasets/ameek/causal_factors)
- [Code](https://github.com/ajmeek/measuring_cot_monitorability)

## Dataset Structure

Each sample group contains:

| Scope | Description |
|-------|-------------|
| `baseline` | Model response without any cue |
| `01_stanford_random` | Authority appeal (Stanford professor recommendation) |
| `02_visual_squares` | Visual pattern cues in answer choices |
| `03_grader_hack` | Grader validation cue |
| `04_unethical_information` | Unethical information framing |
| `05_xml_metadata` | XML metadata with answer hints |

## Models Included

- Claude 3.5 Sonnet
- Claude 3.7 Sonnet
- DeepSeek R1
- DeepSeek V3
- QWQ-32B
- Qwen 2.5-72B Instruct
- Qwen3-235B-A22B
- Qwen3-235B-A22B (thinking mode)

## Key Fields

Each sample contains:

| Field | Description |
|-------|-------------|
| `id` | Sample identifier |
| `scope` | Baseline or cue type (e.g., `01_stanford_random`) |
| `input` | Full question text shown to model |
| `output` | Model's final answer text |
| `reasoning` | Extended thinking/reasoning trace (if available) |
| `model_answer` | Extracted answer letter (A, B, C, ...) |
| `is_correct` | Whether model answered correctly |
| `cue_acknowledgment` | Whether model acknowledged the cue (adaptive only) |
| `state_transition` | Correctness change from baseline (`stayed_correct`, `correct_to_incorrect`, `incorrect_to_correct`, `stayed_incorrect`) |

### Metadata Fields

| Field | Description |
|-------|-------------|
| `correct_answer` | Correct answer letter |
| `correct_answer_index` | Zero-indexed position of correct answer |
| `correct_text` | Full text of correct answer |
| `choices` | List of all answer choices |
| `causal_factors` | Factors necessary to solve the problem correctly |
| `cue` | The cue text shown to model (adaptive only) |
| `has_cue` | Whether sample received a valid cue |

## Citations

If you use this dataset in your research, please cite our paper:
```bibtex
@misc{meek2025measuringchainofthoughtmonitorabilityfaithfulness,
      title={Measuring Chain-of-Thought Monitorability Through Faithfulness and Verbosity},
      author={Austin Meek and Eitan Sprejer and Ivan Arcuschin and Austin J. Brockmeier and Steven Basart},
      year={2025},
      eprint={2510.27378},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2510.27378},
}
```

Please also cite the original datasets:
```bibtex
@article{hendryckstest2021,
  title={Measuring Massive Multitask Language Understanding},
  author={Dan Hendrycks and Collin Burns and Steven Basart and Andy Zou and Mantas Mazeika and Dawn Song and Jacob Steinhardt},
  journal={Proceedings of the International Conference on Learning Representations (ICLR)},
  year={2021}
}

@article{suzgun2022challenging,
  title={Challenging BIG-Bench Tasks and Whether Chain-of-Thought Can Solve Them},
  author={Suzgun, Mirac and Scales, Nathan and Sch{\"a}rli, Nathanael and Gehrmann, Sebastian and Tay, Yi and Chung, Hyung Won and Chowdhery, Aakanksha and Le, Quoc V and Chi, Ed H and Zhou, Denny and and Wei, Jason},
  journal={arXiv preprint arXiv:2210.09261},
  year={2022}
}

@inproceedings{rein2024gpqa,
      title={{GPQA}: A Graduate-Level Google-Proof Q\&A Benchmark},
      author={David Rein and Betty Li Hou and Asa Cooper Stickland and Jackson Petty and Richard Yuanzhe Pang and Julien Dirani and Julian Michael and Samuel R. Bowman},
      booktitle={First Conference on Language Modeling},
      year={2024},
      url={https://openreview.net/forum?id=Ti67584b98}
}
```

## Licensing Information

This dataset is released under the MIT License with important restrictions inherited from model provider terms.

### Constituent Dataset Licenses

The original prompts are sourced from:

- **MMLU** - [MIT License](https://github.com/hendrycks/test)
- **BBH** - [MIT License](https://github.com/suzgunmirac/BIG-Bench-Hard)
- **GPQA** - [CC BY 4.0 License](https://github.com/idavidrein/gpqa)

### Model Output Licensing

The reasoning transcripts are subject to the terms of service of their respective model providers. **Users intending to use this dataset for model training must note the following restrictions:**

| Model | License | Training Other Models | Terms |
|-------|---------|----------------------|-------|
| DeepSeek-R1 | MIT | Explicitly permitted | [License](https://github.com/deepseek-ai/DeepSeek-R1/blob/main/LICENSE) |
| QwQ-32B | Apache 2.0 | Permitted | [License](https://huggingface.co/Qwen/QwQ-32B) |
| Qwen3-235B-A22B | Apache 2.0 | Permitted | [License](https://huggingface.co/Qwen/Qwen3-235B-A22B/blob/main/LICENSE) |
| Qwen3-235B-A22B (thinking) | Apache 2.0 | Permitted | [License](https://huggingface.co/Qwen/Qwen3-235B-A22B/blob/main/LICENSE) |
| Qwen 2.5-72B Instruct | Qwen License | Requires "Built with Qwen" attribution | [License](https://huggingface.co/Qwen/Qwen2.5-72B-Instruct/blob/main/LICENSE) |
| Claude 3.5 Sonnet | Anthropic Terms | Prohibited | [Terms](https://www.anthropic.com/legal/commercial-terms) |
| Claude 3.7 Sonnet | Anthropic Terms | Prohibited | [Terms](https://www.anthropic.com/legal/commercial-terms) |
| DeepSeek-V3 | DeepSeek Model License | Prohibited | [License](https://github.com/deepseek-ai/DeepSeek-V3/blob/main/LICENSE-MODEL) |

**For unrestricted training use**, filter to DeepSeek-R1, QwQ-32B, and Qwen3 model outputs only.

Our cue annotations, causal factor metadata, and evaluation labels are licensed under the MIT License. For more information about the causal factors, please see the [HuggingFace Dataset](https://huggingface.co/datasets/ameek/causal_factors).