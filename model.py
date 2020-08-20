from transformers import BertTokenizer, BertPreTrainedModel, BertModel, BertConfig, AdamW, get_linear_schedule_with_warmup
from transformers import AutoTokenizer, AutoModel, AutoConfig

from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch


class MultiTaskBertForCovidEntityClassification(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.subtasks = config.subtasks
        # We will create a dictionary of classifiers based on the number of subtasks
        self.classifiers = {subtask: nn.Linear(config.hidden_size, config.num_labels) for subtask in self.subtasks}
        # self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        self.init_weights()


    def forward(
            self,
            input_ids,
            entity_start_positions,  ## TODO check what is entity_start_positions
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            labels=None,
    ):
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )
        # DEBUG:
        # print("BERT model outputs shape", outputs[0].shape, outputs[1].shape)
        # print(entity_start_positions[:, 0], entity_start_positions[:, 1])

        # OLD CODE:
        # pooled_output = outputs[1]
        #	input      [8,68]
        # NOTE: outputs[0] has all the hidden dimensions for the entire sequence   	output[0]  [8,68,768]
        # We will extract the embeddings indexed with entity_start_positions	# TODO  why start position embedding	output[1]  [8,768]
        pooled_output = outputs[0][entity_start_positions[:, 0], entity_start_positions[:, 1], :]

        pooled_output = self.dropout(pooled_output)  ## [batch_size, 768]
        # Get logits for each subtaskx
        # logits = self.classifier(pooled_output)  10 (#subtask batch_size 8 2 (0,1)]
        logits = {subtask: self.classifiers[subtask](pooled_output) for subtask in self.subtasks}

        outputs = outputs[2:] + (logits,) # add hidden states and attention if they are here

        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()

            # DEBUG:
            # print(f"Logits:{logits.view(-1, self.num_labels)}, \t, Labels:{labels.view(-1)}")
            for i, subtask in enumerate(self.subtasks):
                # print(labels[subtask].is_cuda)
                if i == 0:
                    loss = loss_fct(logits[subtask].view(-1, self.num_labels), labels[subtask].view(-1))
                else:
                    loss += loss_fct(logits[subtask].view(-1, self.num_labels), labels[subtask].view(-1))
            outputs = outputs + (loss,)

        return outputs  # (loss), logits, (hidden_states), (attentions)


class MultiTaskBertForCovidEntityClassification_new(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.subtasks = config.subtasks

        config.num_labels = len(config.subtasks) ##TODO
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        ## input [batch size, hidden size]
        ## output [batch size, #subtask]

        self.init_weights()

    def build_subtack_mask(self, subtask):
        # TODO:
        pass
        # subtask.shape = [batch_size] (value: 0~num_labels-1)
        # output would be a subtask mask with shape [batch, num_labels] where mask[i, j] = 1 if subtask[i] = j
        # return subtask_mask

    def forward(self,
                input_ids,
                entity_start_positions,
                labels):
                #x, entity_positions, subtask, y=None):
        outputs = self.bert(input_ids)
        
        # TODO: check what are the values in entity_positions
        pooled_output = outputs[0][entity_start_positions[:, 0], entity_start_positions[:, 1], :]
        pooled_output = self.dropout(pooled_output)  ## [batch_size, 768]

        # Get logits for each subtask
        all_logits = self.classifier(pooled_output) #[batch size, # subtask]

        # build subtask mask
        # subtask_mask = self.build_subtack_mask(subtask)
        # logits = all_logits * subtask_mask
        ## TODO change to self.subtasks
        y = torch.stack([labels[subtask] for subtask in labels.keys()], dim =1).type(torch.float)
        if y is not None:
            #loss_fct = nn.CrossEntropyLoss()
            # TODO: check this, the original code use cross entropy loss, but I think we should use binary cross entropy right?
            loss = F.binary_cross_entropy(F.sigmoid(all_logits), y) ## TODO sigmoid
            output = (all_logits, loss)
        else:
            output = (all_logits, )

        return output  # logits, (loss)

