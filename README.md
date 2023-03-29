# A Fast Post-Training Pruning Framework for Transformers

> Implemented BERT Large on the Kaggle P100 with 15GB, takes 11.1 GB memory.

Inspired by post-training quantization (PTQ) toolkits, we propose a post-training pruning framework tailored for Transformers.
Different from existing pruning methods, our framework does not require re-training to retain high accuracy after pruning.
This makes our method fully automated and 10x-1000x faster in terms of pruning time.
[[paper link](https://arxiv.org/abs/2204.09656)]

<div align="center">
  <img src=figures/overview.png>
</div>

## Prerequisite

### Install denpendencies

Tested on Python 3.7.10.
You need an NVIDIA GPU (with 16+ GB memory) to run our code.

> Implemented BERT Large on the Kaggle P100 with 15GB, takes 11.1 GB memory.

```bash
pip3 install -r requirements.txt
```

### Prepare checkpoints

We provide the (unpruned) checkpoints of BERT-base and DistilBERT used in our experiments.
We used the pre-trained weights provided by [HuggingFace Transformers](https://github.com/huggingface/transformers), and fine-tuned them for 8 downstream tasks with standard training recipes.

| Model | Link |
|:-----:|:-----:|
| BERT-base | [gdrive](https://drive.google.com/drive/folders/1OWHL7Cjhaf2n67PZX4Pt0Be3Gv2VCLo0?usp=sharing) |
| DistilBERT | [gdrive](https://drive.google.com/drive/folders/1ZyGQL5ynoXs0ffGkENNjHq7eijB-B80l?usp=sharing) |

Our framework only accepts the HuggingFace Transformers PyTorch models.
If you use your own checkpoints, please make sure that each checkpoint directory contains both `config.json` and `pytorch_model.bin`.

## Prune models and test the accuracy on GLUE/SQuAD benchmarks

* Supported models: BERT-base/large, DistilBERT, RoBERTa-base/large, DistilRoBERTa, etc.
* Supported tasks:
  * GLUE: MNLI, QQP, QNLI, SST-2, STS-B, MRPC
  * SQuAD V1.1 & V2

Download models from HF
```
from transformers import AutoModelForQuestionAnswering, AutoTokenizer

model_name = "madlag/bert-large-uncased-whole-word-masking-finetuned-squadv2"
model = AutoModelForQuestionAnswering.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
```

The following example prunes a QQP BERT-base model with 50% MAC (FLOPs) constraint:
```bash
python3 main.py --model_name bert-base-uncased \
                --task_name qqp \
                --ckpt_dir <your HF ckpt directory> \ # usually it would be ~/.cache/huggingface/hub/models-name-dv2/snapshots/ead15fce67e3003ae1ea873316afbfe2f057fc0
                --constraint 0.5
```

Squad v2
| Metric | Constraint | Seed | MAC (%) | Pruning Time (s) | Test Accuracy (%) |
|--------|------------|------|---------|------------------|--------------------|
| MAC    | 0.1        | 1    | 10.00   | 647.13           | 3.0021             |
| MAC    | 0.5        | 1    | 50.00   | 573.83           | 78.6466            |
| MAC    | 0.7        | 1    | 70.00   | 567.95           | 83.5277            |
| MAC    | 0.8        | 1    | 80.00   | 559.19           | 84.7552            |
| MAC    | 0.9        | 1    | 90.00   | 440.01           | 92.9893            |

You can also control more arguments such as sample dataset size (see `main.py`).

## Citation

```bibtex
@misc{kwon2022fast,
      title={A Fast Post-Training Pruning Framework for Transformers}, 
      author={Woosuk Kwon and Sehoon Kim and Michael W. Mahoney and Joseph Hassoun and Kurt Keutzer and Amir Gholami},
      year={2022},
      eprint={2204.09656},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```

## Copyright

THIS SOFTWARE AND/OR DATA WAS DEPOSITED IN THE BAIR OPEN RESEARCH COMMONS REPOSITORY ON 02/07/23.
