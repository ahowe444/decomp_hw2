from allennlp.models import Model

import torch
import torch.nn as nn

from allennlp.data.vocabulary import Vocabulary
from allennlp.modules.text_field_embedders import TextFieldEmbedder
from allennlp.modules.seq2seq_encoders.seq2seq_encoder import Seq2SeqEncoder
from allennlp.training.metrics import FBetaMeasure

from allennlp.nn.util import get_text_field_mask
from typing import Dict, Optional

@Model.register('srl_lstm')
class SRLLSTM(Model):
  
  def __init__(self,
              vocab: Vocabulary,
              embedder: TextFieldEmbedder,
              encoder: Seq2SeqEncoder) -> None:
        super().__init__(vocab)
  
        self._embedder = embedder
        self._encoder = encoder
        self._classifier1 = nn.Linear(in_features=2*encoder.get_output_dim(),
                                     out_features=2)
#        self.relu1 = nn.ReLU()
#        self._classifier2 = nn.Linear(in_features = 50, out_features=2)

        self._f1 = FBetaMeasure(average='macro')
        self._loss = nn.CrossEntropyLoss()
        self.soft = nn.Softmax(dim=1)

  def forward(self,
              tokens: Dict[str, torch.Tensor],
              pred_arg : Dict[str, torch.Tensor],
              label: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
      mask = get_text_field_mask(tokens)
      embedded = self._embedder(tokens)
      encoded = self._encoder(embedded, mask)
      
      pred_arg_vectors = []
      for i,sentence in enumerate(encoded):
        pred = sentence[int(pred_arg[i][0].numpy())-1]
        arg = sentence[int(pred_arg[i][1].numpy())-1]
        cat_vector = torch.cat((pred, arg))
        pred_arg_vectors.append(cat_vector)

      pred_arg_batch = torch.stack(pred_arg_vectors)
 #     classified = self.relu1(self._classifier1(pred_arg_batch))
      classified = self._classifier1(pred_arg_batch)
      
      
      # According to the documentation, this particular FBetaMeasure
      # requires the prediction be [num_batch, num_classes ] and 
      # the label only be [num_batch]
      self._f1(classified, label)

      output: Dict[str, torch.Tensor] = {}

      if label is not None:
        output["loss"] = self._loss(classified, label)
        output["pred"] = torch.argmax(self.soft(classified))

      return output

  def get_metrics(self, reset: bool = True) -> Dict[str, float]:
    return self._f1.get_metric(reset)
