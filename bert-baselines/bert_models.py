from transformers import BertPreTrainedModel, BertModel
from torch import nn
from torch.nn import BCEWithLogitsLoss

class BertForMultiLabelSequenceClassification(BertPreTrainedModel):
    """
    Bert Model transformer with a multi-label sequence classification head on top
    (a linear layer with sigmoid activation on top of the pooled output).
    """
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.linear = nn.Linear(config.hidden_size, self.config.num_labels)
        self.classifier = nn.Sigmoid()

        self.init_weights()

    def forward(
        self,
        input_ids=None,
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

        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        pooled_output = self.linear(pooled_output)
        logits = self.classifier(pooled_output)

        outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here

        if labels is not None:
            loss_fct = BCEWithLogitsLoss()
            # Apply loss before the `Sigmoid` layer, as `BCEWithLogitsLoss`
            # internally applies `Sigmoid` in a more numerically stable fashion.
            loss = loss_fct(pooled_output, labels.type_as(pooled_output))
            outputs = (loss,) + outputs

        return outputs  # (loss), logits, (hidden_states), (attentions)
