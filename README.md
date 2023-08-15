# sentiment-analyzer

## Overview
This program is for sentiment analyzing tweets

## Command to run
![Diagram_TweetScraper (1)](https://github.com/hellokev/sentiment-analyzer/assets/31089160/5f8188ba-c55e-40b6-a903-ed16be6ff1e3)

## Package Installation
### Install libraries for data science
```
pip3 install pandas nltk transformers scipy matplotlib seaborn
```
### Install PyTorch
#### OS: Linux - Compute Platform: AMD GPU
```
pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/rocm5.2/
```
#### OS: Linux - Compute Platform: CPU
```
pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cpu
```
### Other Operating Systems
```
# MacOS
pip3 install torch torchvision torchaudio

# Windows (Compute Platform: CPU)
pip3 install torch torchvision torchaudio

# Windows (Compute Platform: GPU)
pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu117
```
