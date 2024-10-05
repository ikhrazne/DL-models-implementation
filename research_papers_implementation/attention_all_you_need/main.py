
# TODO build a translation algorithm from englisch to tamazigh using transformers and mamba

from datasets import load_dataset
import pandas as pd
from tokinizer import Tokinezer

# ds = load_dataset("Tamazight-NLP/NLLB-Seed_Standard-Moroccan-Tamazight")


if __name__ == "__main__":

    df = pd.read_csv(filepath_or_buffer=r"dataset\dataset.csv")

    toknizer = Tokinezer(df)

    dictionary = toknizer.dictionaries

    print(len(dictionary["dictionary1"]))

    print(len(dictionary["dictionary2"]))

