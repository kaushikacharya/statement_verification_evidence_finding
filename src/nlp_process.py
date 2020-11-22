#!/usr/bin/env python

import argparse
import spacy
from spacy.lang.en import English

class NLPProcess:
    """NLP processing of text
    """
    def __init__(self, model="en_core_web_sm"):
        self.model = model
        self.nlp = None

    def load_nlp_model(self, verbose=False):
        self.nlp = spacy.load(name=self.model)
        if verbose:
            print("Model: {} loaded".format(self.model))
            print("pipe names: {}".format(self.nlp.pipe_names))

    def construct_doc(self, text):
        """Construct Doc container from the text.

            Reference:
            ---------
            https://spacy.io/api/doc
        """
        assert self.nlp is not None, "pre-requisite: Execute load_nlp_model()"
        doc = self.nlp(text)
        return doc
