"""
This module provides a function for cleaning and preprocessing text.
"""
import re
import unicodedata

# import nltk
from langflow.custom import Component
from langflow.inputs import MessageTextInput
from langflow.schema.message import Message
from langflow.template import Output

# nltk.download('stopwords')
# stop_words = set(nltk.corpus.stopwords.words('portuguese'))


class CleaningComponent(Component):
    """Component for cleaning and preprocessing text."""
    display_name = "CleaningText"
    description = "Clean and preprocess text by removing unwanted characters, stopwords, and short words."
    name = "CleaningText"
    icon = "custom_components"

    inputs = [
        MessageTextInput(
            name="input_value", display_name="Input",
            info="", required=True
        )
    ]

    outputs = [
        Output(display_name="Text", name="text", method="cleaning_text"),
    ]

    def cleaning_text(self) -> Message:
        """
        Clean and preprocess text by removing unwanted characters, stopwords, and short words.
        """
        text = self.input_value

        new_string = text.lower()
        # new_string = re.sub(r'\([^)]*\)', '', new_string)
        new_string = re.sub('"', '', new_string)

        new_string = unicodedata.normalize('NFKD', new_string)

        new_string = re.sub(r'[^a-zA-ZÀ-ÖØ-öø-ÿ ]', ' ', new_string)

        # tokens = [w for w in new_string.split() if not w in stop_words]
        # long_words = []
        # for i in tokens:
        #     if len(i) >= 3:
        #         long_words.append(i)

        # result = (" ".join(long_words)).strip()

        return Message(text=new_string)
