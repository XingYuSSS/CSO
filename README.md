# CSO: Chain of Strategy Optimization Makes Large Language Models Better Emotional Supporter

This repository contains code for the EMNLP 2025 Findings paper [Chain of Strategy Optimization Makes Large Language Models Better Emotional Supporter](https://arxiv.org/abs/2503.05362)

## Overview

The system generates high-quality emotional support conversations through a three-stage pipeline:

1. **Conversation Tree Generation** (`run.py`): Uses MCTS to explore different emotional support strategies and generate conversation trees
2. **Preference Data Construction** (`build_data.py`): Converts conversation trees into preference-annotated dialogue histories
3. **Dataset Conversion**: Transforms dialogue histories into either preference pairs (`change_data.py`) or KTO datasets (`change_data_kto.py`)

## Pipeline

### Stage 1: Generate Conversation Trees
```bash
python run.py
```
- Uses MCTS to explore emotional support strategies
- Generates conversation trees with multiple response options
- Saves trees as pickle files in `output/tree/`

### Stage 2: Build Preference Data
```bash
python build_data.py
```
- Converts conversation trees into dialogue histories
- Extracts preferred vs. non-preferred responses
- Generates comparison data with strategy annotations

### Stage 3: Convert to Training Datasets

**For Preference Pairs:**
```bash
python change_data.py
```

**For KTO Dataset:**
```bash
python change_data_kto.py
```

## Citation
```
@misc{zhao2025chainstrategyoptimizationmakes,
      title={Chain of Strategy Optimization Makes Large Language Models Better Emotional Supporter}, 
      author={Weixiang Zhao and Xingyu Sui and Xinyang Han and Yang Deng and Yulin Hu and Jiahe Guo and Libo Qin and Qianyun Du and Shijin Wang and Yanyan Zhao and Bing Qin and Ting Liu},
      year={2025},
      eprint={2503.05362},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2503.05362}, 
}
```
