# SynGR: Unleashing the Potential of Cross-Modal Synergy for  Generative Recommendation

This repo is for source code of 2026 ICML paper "SynGR: Unleashing the Potential of Cross-Modal Synergy for  Generative Recommendation".

## Abstract

Generative Recommendation (GR) has emerged as a promising paradigm by formulating item recommendation as a sequence-to-sequence generation task over item identifiers. Recent studies have incorporated multimodal signals to provide richer token-level evidence for generation. However, existing approaches largely rely on alignmentcentric fusion and underexplore synergistic information across modalities. In practice, synergistic information plays a critical role in capturing emergent item properties that cannot be inferred from any single modality alone. Such properties encode intrinsic item semantics and guide user preferences, enabling models to move beyond surface-level feature matching. To address this limitation, we propose SynGR, a synergistic generative recommendation framework that explicitly encourages the exploitation of cross-modal dependencies during generation. By constraining overreliance on dominant modalities, SynGR enables the model to capture emergent item semantics beyond shared or modality-specific signals. Extensive experiments across three benchmark datasets demonstrate that SynGR achieves superior recommendation performance, with an average improvement of 9.01%.

## Setup

Please first install the required dependencies: 

```
pip install -r requirements.txt
```

CUDA == 12.1 

PyTorch == 2.1.0 

Transformers == 4.45.0 ( ⚠ **Note:** In our experiments, we found that if the `transformers` version is higher than `4.45.0`, the model’s metrics show a significant decline. Therefore, we strongly recommend strictly pinning this version when reproducing the results.) 

Accelerate == 0.28.0

## Quick Start

### Data Processing
```
cd data_process
```
1. Download images  
2. Process data to ensure each item corresponds to one image and one text description  
3. Generate text embeddings  
4. Generate image embeddings    

### Training the Quantitative Translator
```
cd index
bash script/run.sh          
bash script/gen_code_dis.sh   
```

### Pre-training
```
bash script/pretrain.sh
```

### Fine-tuning
```
bash finetune_mask.sh
```

## Training Tasks

SynGR adopts a **two-stage training paradigm**: pre-training on large-scale cross-modal data followed by fine-tuning on downstream generative recommendation tasks. The specific tasks used in each stage are described below.

### Pre-training Stage

During pre-training, SynGR is trained on a large-scale multimodal corpus (approximately 7.1M interactions across 884,918 users and 255,181 items from six Amazon Product Reviews categories) using the following **two tasks**:

| Task | Full Name | Description |
|------|-----------|-------------|
| `seqrec` | Sequential Recommendation | Models user interaction sequences as item ID sequences and predicts the next item. |
| `seqimage` | Image Sequence Modeling | Models image sequences corresponding to user interaction history, leveraging visual cues to predict the next item's visual representation. |

### Fine-tuning Stage

After pre-training, the model is fine-tuned on the downstream generative recommendation task using **six tasks**. Only the `Tasks` hyperparameter in the corresponding shell script needs to be modified — no additional training ratio adjustments are required.

| Task | Full Name | Description |
|------|-----------|-------------|
| `seqrec` | Sequential Recommendation | The core GR task: predicts the next item in a user's interaction sequence based on item ID sequences. This is the primary recommendation objective. |
| `seqimage` | Image Sequence Modeling | Predicts the next item using image sequence features, enabling the model to leverage visual information for recommendation. |
| `seqitem2image` | Item-to-Image Generation | Generates image representations conditioned on item sequences. This task explicitly models the cross-modal dependency from text/item to image modality. |
| `seqimage2item` | Image-to-Item Generation | Generates item sequences conditioned on image representations. This task encourages the model to map visual features back to item identifiers, capturing synergistic cross-modal information. |
| `maskimg` | Masked Image Modeling | Predicts the image representation of the next item in the user interaction sequence using the method proposed in this paper. |
| `masktext` | Masked Item-Text Encoding | Predicts the text representation of the next item in the user interaction sequence using the method proposed in this paper. |

## Main Hyperparameter Space

| Hyperparameter     | Search Space                                     |
| ------------------ | ------------------------------------------------ |
| mask ratio         | [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1] |
| warmup ratio       | [0, 0.05, 0.1]                                   |
| contrastive weight | [0.001, 0.003, 0.005, 0.007, 0.009]              |
| batch size         | [128, 256, 512]                                  |
| temperature        | [0.01, 0.03, 0.05, 0.07, 0.09]                   |


## Acknowledgements

This work is implemented based on [MQL4GRec](https://github.com/zhaijianyang/MQL4GRec),  We sincerely thank the authors of these project for their valuable contributions to the open-source community.
