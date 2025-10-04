from __future__ import annotations

import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from transformers import AutoModelForSequenceClassification, AutoConfig


class FinBertLightning(pl.LightningModule):
	def __init__(self, model_name: str, output_hidden_states: bool = True):
		super().__init__()
		cfg = AutoConfig.from_pretrained(model_name, output_hidden_states=output_hidden_states)
		self.model = AutoModelForSequenceClassification.from_pretrained(model_name, config=cfg)

	def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
		out = self.model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True, return_dict=True)
		logits = out.logits
		probs = F.softmax(logits, dim=-1)
		cls_emb = out.hidden_states[-1][:, 0, :]
		return {"probs": probs, "cls": cls_emb}

	def training_step(self, *args, **kwargs):
		raise NotImplementedError

	def configure_optimizers(self):
		return None

