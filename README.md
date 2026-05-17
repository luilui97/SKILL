# Structural Knowledge Induced Logical Learning for LLM Reasoning over Knowledge Graphs

**Paper Title**: *Structural Knowledge Induced Logical Learning for LLM Reasoning over Knowledge Graphs*

> **TL;DR**: SKILL is a novel framework that injects symbolic structural rules into large language models (LLMs) to enhance their reasoning abilities on knowledge graphs under inductive settings.

---

## 🌟 Highlights

- ✅ **Inductive KG Reasoning** without entity overlap between train/test.
- 🧠 Combines **symbolic rule mining** with **LLM-based filtering and fine-tuning**.
- 🔍 LLMs are used not just for inference, but also for **semantic rule validation**.
- 📈 Outperforms previous state-of-the-art by up to **+5% Hit@1** on standard benchmarks.

---

## 🧱 Framework Overview

SKILL consists of two main modules:

1. **Constraint-Synthesizer**  
   - Extracts symbolic rules from closed paths in the KG.  
   - Uses LLM one-shot prompting to assess **semantic validity**.  
   - Filters out spurious co-occurrence patterns.

2. **Logic-enhanced Reasoning**  
   - Fine-tunes the LLM with **valid symbolic rules** as explicit guidance.  
   - Enhances the LLM’s ability to understand and leverage KG structure.

---


## 📦 Installation
SKILL uses [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) to fine-tune LLMs. Please follow the official installation guide:

```bash
pip install -r requirements.txt
```

## 🚀 Quick Start

### 1. Generate rules and evaluates them
```bash
python evaluate_rules.py --config [conifg files]
```

### 2. Generate Instructions
```bash
python build_instruction.py --config [conifg files]
```

### 3. Finetune the LLMs
Conduct supervised fine-tuning using [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) with the provided prompts. The training configuration requires specifying the prompt path and other training settings in a configuration file. Detailed hyperparameters and experimental settings are provided in our paper. [Qwen/Qwen2-7B-Instruct](https://huggingface.co/Qwen/Qwen2-7B-Instruct) is available here.

### 4.Inference
The following command evaluates the model performance. 
```bash
python build_instruction.py --config [conifg files]
```
All configuration files are located in the `\configs` directory.


## 🔒 Anonymity Notice
This repository is anonymized for double-blind review. All identifiable information such as author names or institutional affiliations has been removed or masked.
