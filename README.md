# SynGR: Unleashing the Potential of Cross-Modal Synergy for  Generative Recommendation

This is the source code for the paper "SynGR: Unleashing the Potential of Cross-Modal Synergy for  Generative Recommendation".

## Abstract

Generative Recommendation (GR) has emerged as a promising paradigm by formulating item recommendation as a sequence-to-sequence generation task over item identifiers. Recent studies have incorporated multimodal signals to provide richer token-level evidence for generation. However, existing approaches largely rely on alignmentcentric fusion and underexplore synergistic information across modalities. In practice, synergistic information plays a critical role in capturing emergent item properties that cannot be inferred from any single modality alone. Such properties encode intrinsic item semantics and guide user preferences, enabling models to move beyond surface-level feature matching. To address this limitation, we propose SynGR, a synergistic generative recommendation framework that explicitly encourages the exploitation of cross-modal dependencies during generation. By constraining overreliance on dominant modalities, SynGR enables the model to capture emergent item semantics beyond shared or modality-specific signals. Extensive experiments across three benchmark datasets demonstrate that SynGR achieves superior recommendation performance, with an average improvement of 9.01%.

## Setup

Please first install the required dependencies: 

```
pip install -r requirements.txt
```

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

