import pandas as pd


class Tokinezer:

    def __init__(
            self,
            data: pd.DataFrame
            # column_to_tokinez_and_their_language: dict
    ):
        self.data = data
        # self.column_to_tokiner_and_their_language = column_to_tokinez_and_their_language
        self.dictionaries = dict
        self.result = self.get_tokens()

    def _split(self, sentence) -> list:
        return sentence.split(" ")

    def get_tokens(self) -> pd.DataFrame:
        result = {
            "tokinzed_sentence": [],
            "tokinzed_target_sentence": []
        }
        # add new columns for every tokinzed values to save the tokens

        base_words = []
        target_words = []

        for index, row in self.data.iterrows():
            base_sentence_split = self._split(row["source_sentence"])
            target_sentence_split = self._split(row["target_sentence"])

            result["tokinzed_sentence"].append(base_sentence_split)
            result["tokinzed_target_sentence"].append(target_sentence_split)

            base_words.extend(base_sentence_split)
            target_words.extend(target_sentence_split)

            base_words = list(set(base_words))
            target_words = list(set(target_words))

        self.dictionaries = {
            "dictionary1": base_words,
            "dictionary2": target_words
        }

        return pd.DataFrame(result)
