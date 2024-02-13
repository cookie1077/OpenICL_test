import datasets
import jax.numpy as jnp
import jax.nn as nn
import jax
import numpy as np
import pandas as pd
from transformers import FlaxAutoModelForSequenceClassification
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from tqdm import tqdm

DATASET = "sst2"
SPLIT = "train"
MODEL_NAME = "JeremiahZ/bert-base-uncased-sst2"
OUTPUT_PATH = "result.csv"

def main():
    dataset = datasets.load_dataset(DATASET)[SPLIT]
    dataloader = DataLoader(dataset, batch_size=1)
    model = FlaxAutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    tkn = AutoTokenizer.from_pretrained(MODEL_NAME)

    def infer(input_ids):
        output = model(input_ids)
        probs = nn.softmax(output.logits)
        return probs

    data = {
        "index":[],
        "sentence":[],
        "label":[],
        "positive_prob":[],
        "negative_prob":[]
    }

    for inputs in tqdm(dataloader):
        sentences = inputs["sentence"]
        tokenized = tkn(sentences, return_tensors="np")
        probs = infer(tokenized["input_ids"])
        data["index"].append(inputs["idx"][0].item())
        data["sentence"].append(sentences[0])
        data["label"].append("positive" if inputs["label"][0].item() else "negative")
        data["positive_prob"].append(probs[0][1])
        data["negative_prob"].append(probs[0][0])

    pd.DataFrame(data).to_csv(OUTPUT_PATH, index=False)


if __name__ == "__main__":
    main()

