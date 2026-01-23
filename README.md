# Beyond Redundancy: Unleashing Cross-Modal Synergy Information for Generative Recommendation

This is the source code for the paper "Beyond Redundancy: Unleashing Cross-Modal Synergy Information for Generative Recommendation".

## Abstract

Generative Recommendation (GR) has become a prominent paradigm by modeling item recommendation as a sequence-to-sequence generation task. While recent work incorporates multimodal signals, their practical impact is often limited. Due to strong textual biases in Pre-trained Language Models, generative systems tend to rely on dominant modalities and underutilize complementary visual information, failing to capture synergistic semantics that arise from cross-modal interaction. To address this problem, We propose \textbf{SynGR} (Synergistic Generative Recommendation), a generative framework that explicitly promotes cross-modal synergy. By discouraging shortcut-driven reliance on unimodal cues and encouraging interaction across modalities, SynGR enables the model to learn representations that go beyond single-modality features. Extensive experiments on multiple real-world datasets show that SynGR consistently outperforms state-of-the-art GR methods.

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

