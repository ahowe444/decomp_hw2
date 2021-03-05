from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.token_indexers import SingleIdTokenIndexer, TokenIndexer

from typing import Dict, List, Iterator

from allennlp.data.instance import Instance
from overrides import overrides

import itertools
import pickle
import numpy as np

from allennlp.data.tokenizers import Token
from allennlp.data.fields import Field, TextField, LabelField, ArrayField


@DatasetReader.register("srl_reader")
class DecompDatasetReader(DatasetReader):

  def __init__(self,
                token_indexers: Dict[str, TokenIndexer] = None,
                lazy: bool = False) -> None:
      super().__init__(lazy)
      self._token_indexers = token_indexers or {'tokens': SingleIdTokenIndexer()}

  def _read(self, file_path: str) -> Iterator[Instance]:
    with open(file_path, 'rb') as conll_file:
          data = pickle.load(conll_file)
          for example in data:
            text = example[0:-3]
            pred_arg = np.array(example[-3:-1])
            label = example[-1]

            yield self.text_to_instance(text, pred_arg, label)

  def text_to_instance(self,
                        words: List[str],
                        pred_arg: List[int],
                        ner_tags: int) -> Instance:
        fields: Dict[str, Field] = {}
        tokens = TextField([Token(w) for w in words], self._token_indexers)
        fields["tokens"] = tokens
        fields["pred_arg"] = ArrayField(pred_arg)
        fields["label"] = LabelField("role" if ner_tags == 1 else "non-role")
        return Instance(fields)
