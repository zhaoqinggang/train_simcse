from transformers import BertModel
import torch.nn as nn
from config.config import Config
import torch
from transformers import AdamW
from tqdm import tqdm


class MultiTaskMatchModel(nn.Module):
    def __init__(self,task_type):
        super(MultiTaskMatchModel, self).__init__()
        self.bert = BertModel.from_pretrained(Config.model_name)
        self.passenger_fc = nn.Linear(Config.hidden_size, Config.hidden_size)
        self.driver_fc = nn.Linear(Config.hidden_size, Config.hidden_size)
        self.taxi_passenger_fc = nn.Linear(Config.hidden_size, Config.hidden_size)
        self.taxi_driver_fc = nn.Linear(Config.hidden_size, Config.hidden_size)


    def forward(self, input_ids, attention_mask, task_type):
        bert_output = self.bert(input_ids, attention_mask=attention_mask)
        embeddings = bert_output.last_hidden_state[:, 0, :]
        
        task_output = {
            'passenger': self.passenger_fc(embeddings),
            'driver': self.driver_fc(embeddings),
            'taxi_passenger': self.taxi_passenger_fc(embeddings),
            'taxi_driver': self.taxi_driver_fc(embeddings)
        }.get(task_type, None)

        if task_output is None:
            raise ValueError("Invalid task type provided.")
        return task_output


