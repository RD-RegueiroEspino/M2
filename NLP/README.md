# CANINE, a tokenization-free encoder
CANINE (https://arxiv.org/pdf/2103.06874.pdf) is a tokenization-free encoder. The goal of this project is to evaluate its performance
on different tasks using BERT (https://arxiv.org/abs/1810.04805) as benchmark.

### Requirements
* transformers
* datasets

```bash
pip install -r requirements.txt
```

### Using Conda

* Execute the following instruction

```bash
conda env create -f environment.yml
conda activate canine_nlp
```
Once the conda environment activated, run the following instruction

```bash
pip install .
```

### Warning
This is a project done for the _Algorithms for speech and natural language processing_ course and it was not reviewed by
externals, so it might contain errors.
