# Empathic Conversations Auto Eval


## Getting Started

### Configuration
1. Download the 4 models and the MultiLabelBinarizer file `mlb.pkl` from this [link](https://drive.google.com/drive/folders/1uAIYQIbmGHaKt6GPQvfB11I_R40Mxuz-?usp=sharing). Use [gdown](https://github.com/wkentaro/gdown) if you are trying to download to a remote instance via terminal because wget or curl is insufficient to download large files from google drive.
2. Place the models inside of the `models` directory and unzip.

### How to evaluate
In `auto_eval_utterances.py`, the function `evaluate_utterances(utterances, csv_name)` should be called to evaluate your utterances or chats. The parameter `utterances` is a list of strings, and the parameter `csv_name` is a string name for the output csv that contains the autoamtic evaluations. 

Now you can just call `evaluate_utterances(utterances, csv_name)` to evaluate a list of utterances with the 4 models you downloaded above and see the results inside of the `evaluations` folder.

