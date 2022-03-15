### File provides automatic turn level evaluations for self chat data

from transformers import RobertaTokenizer, RobertaForSequenceClassification
import torch
from simpletransformers.classification import (
    MultiLabelClassificationModel, MultiLabelClassificationArgs
)
import pickle
import numpy as np
import json
import pandas as pd
import argparse
import os
import time



class RobertaWrapper():
    def __init__(self, turn_level_path):
        self.tokenizer = RobertaTokenizer.from_pretrained(turn_level_path)
        self.model = RobertaForSequenceClassification.from_pretrained(turn_level_path)
        self.model.eval()
    
    def predict_single(self, phrase):
        inputs = self.tokenizer(phrase, return_tensors="pt")
        labels = torch.tensor([1]).unsqueeze(0)  # Batch size 1
        with torch.no_grad():
            outputs = self.model(**inputs, labels=labels)
        
        return outputs[1]
    
    
    def predict(self, phrases):
        inputs = self.tokenizer(phrases, return_tensors="pt", padding=True)
        labels = torch.tensor([len(phrases)]).unsqueeze(0)  # Batch size of the input list
        with torch.no_grad():
            outputs = self.model(**inputs, labels=labels)
        
        # unpack from tensors        
        return list(map(lambda x: x[0].item(), outputs[1]))
    

class DialogActsPredictor():
    def __init__(self, path):
        self.model = MultiLabelClassificationModel("roberta", path, use_cuda=False)
        ## Loading MLB
        mlb_path = os.path.join(os.path.dirname(__file__), './models/mlb.pkl')

        with open(mlb_path, 'rb') as f:
            self.mlb = pickle.load(f)
            
    def predict(self, texts):
        #expects texts to be a list of strings
        x = texts
        y_pred, y_pred_raw  = self.model.predict(x)
        y_pred = np.asarray(y_pred)
        y_decoded = self.mlb.inverse_transform(y_pred)
        
        return y_pred, y_decoded


def evaluate_utterances(utterances, csv_name):

    ### CHANGE MODEL NAMES IF DIFFERENT
    
    PREFIX_PATH = os.path.join(os.path.dirname(__file__), './models')
    
    da_path = f"{PREFIX_PATH}/dialog_acts_roberta_20_epoch" 
    empathy_path = f"{PREFIX_PATH}/empathy/fully_trained_roberta"
    emotion_path = f"{PREFIX_PATH}/emotion/run_1"
    emotionalpolarity_path =  f"{PREFIX_PATH}/emotionalpolarity/test1"
    selfdisclosure_path = f"{PREFIX_PATH}/selfdisclosure/run_1"
    
    utt_list = utterances

    chats_df = pd.DataFrame(utterances, columns=['utterances'])
    
    # load turn level evaluator models
    da_pred = DialogActsPredictor(da_path)
    print("Computing dialog act labels . . . ")
    start = time.time()
    da_labels = da_pred.predict(utt_list)[1]
    print(f"Dialog Act Inference Time: {(time.time() - start):.2f} seconds")
    chats_df['dialog_acts'] = da_labels

    empathy_predictor = RobertaWrapper(empathy_path)
    print("Computing empathy scores . . . ")
    start = time.time()
    emp_labels = empathy_predictor.predict(utt_list)
    print(f"Empathy Inference Time: {(time.time() - start):.2f} seconds")
    chats_df['empathy'] = emp_labels

    emotion_predictor = RobertaWrapper(emotion_path)
    print("Computing emotion scores . . . ")
    start = time.time()
    emo_labels = emotion_predictor.predict(utt_list)
    print(f"Emotion Inference Time: {(time.time() - start):.2f} seconds")
    chats_df['emotion'] = emo_labels

    emotionalpolarity_predictor = RobertaWrapper(emotionalpolarity_path)
    print("Computing emotionalpolarity scores . . . ")
    start = time.time()
    ep_labels = emotionalpolarity_predictor.predict(utt_list)
    print(f"EmotionalPolarity Inference Time: {(time.time() - start):.2f} seconds")
    chats_df['emotional_polarity'] = ep_labels

    selfdisclosure_predictor = RobertaWrapper(selfdisclosure_path)
    print("Computing selfdisclosure scores . . . ")
    start = time.time()
    sd_labels = selfdisclosure_predictor.predict(utt_list)
    print(f"SelfDisclosure Inference Time: {(time.time() - start):.2f} seconds")
    chats_df['self_disclosure'] = sd_labels 
    
    save_path =  os.path.join(os.path.dirname(__file__), f"./evaluations/{csv_name}.csv") 
    chats_df.to_csv(save_path)
    print(f"saved automatic evaluation of chats to {save_path}")



    