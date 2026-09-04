import torch
import torch.nn as nn
from transformers import BertModel, BertConfig
from typing import List, Optional, Tuple, Union
import copy

class MySelfOutput(nn.Module):
    def __init__(self, output_copy):
        super().__init__()
        self.dense = copy.deepcopy(output_copy.dense)
        self.LayerNorm =copy.deepcopy(output_copy.LayerNorm)
        self.dropout = copy.deepcopy(output_copy.dropout)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states

class MyAttention(nn.Module):
    def __init__(self, attention_copy):
        super().__init__()
        self.self = copy.deepcopy(attention_copy.self)
        self.output = MySelfOutput(attention_copy.output)

    def forward(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.FloatTensor] = None, head_mask: Optional[torch.FloatTensor] = None,
                encoder_hidden_states: Optional[torch.FloatTensor] = None, encoder_attention_mask: Optional[torch.FloatTensor] = None,
                past_key_value: Optional[Tuple[Tuple[torch.FloatTensor]]] = None, output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor]:
        self_outputs = self.self(hidden_states, attention_mask, head_mask, encoder_hidden_states, encoder_attention_mask, past_key_value, output_attentions,)
        attention_output = self.output(self_outputs[0])
        outputs = (attention_output,) + self_outputs[1:]  # add attentions if we output them
        return outputs

class MyOutput(nn.Module):
    def __init__(self, output_copy):
        super().__init__()
        self.dense = copy.deepcopy(output_copy.dense)
        self.LayerNorm = copy.deepcopy(output_copy.LayerNorm)
        self.dropout = copy.deepcopy(output_copy.dropout)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states

class MyLayer(nn.Module):
    def __init__(self, l_copy):
        super().__init__()
        self.attention = MyAttention(l_copy.attention)
        self.intermediate = copy.deepcopy(l_copy.intermediate)
        self.output = MyOutput(l_copy.output)

    def forward(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.FloatTensor] = None,
                head_mask: Optional[torch.FloatTensor] = None, encoder_hidden_states: Optional[torch.FloatTensor] = None,
                encoder_attention_mask: Optional[torch.FloatTensor] = None, past_key_value: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
                output_attentions: Optional[bool] = False) -> Tuple[torch.Tensor]:

        attention_output = self.attention(
            hidden_states,
            attention_mask,
            head_mask,
            output_attentions=output_attentions,
            past_key_value=past_key_value,
        )
        attention_output = attention_output[0]
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)

        return layer_output

class MyEncoder(nn.Module):
    def __init__(self, copy_bert_encoder):
        super().__init__()

        self.layers = nn.ModuleList([MyLayer(l) for l in copy_bert_encoder.layer])

    def forward(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.FloatTensor] = None, head_mask: Optional[torch.FloatTensor] = None,
                encoder_hidden_states: Optional[torch.FloatTensor] = None, encoder_attention_mask: Optional[torch.FloatTensor] = None, past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
                use_cache: Optional[bool] = None, output_attentions: Optional[bool] = False, output_hidden_states: Optional[bool] = False, return_dict: Optional[bool] = True,
                ) -> Union[Tuple[torch.Tensor], Tuple[torch.Tensor]]:

        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None
        all_cross_attentions = () if output_attentions and self.config.add_cross_attention else None

        current = hidden_states
        for i, layer_module in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_head_mask = head_mask[i] if head_mask is not None else None
            past_key_value = past_key_values[i] if past_key_values is not None else None

            layer_outputs = layer_module(
                hidden_states,
                attention_mask,
                layer_head_mask,
                encoder_hidden_states,
                encoder_attention_mask,
                past_key_value,
                output_attentions,
            )

            hidden_states = layer_outputs
            current = current + hidden_states
        
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        return (current, all_hidden_states)


class LCNBert(nn.Module):
    def __init__(self, config):
        super(LCNBert, self).__init__()


        # self.config = BertConfig.from_pretrained("bert-base-uncased")
        # #large 24 blocks of 1024 // base 12 blocks of 768
        # self.bert = BertModel.from_pretrained("bert-base-uncased", config=self.config)
        # # self.num_labels = num_labels
        # # self.dropout = nn.Dropout(p=0.0)
        # # self.classifier = nn.Linear(768, self.num_labels)

        # self.embeddings = copy.deepcopy(self.bert.embeddings)
        # self.encoder = MyEncoder(self.bert.encoder)

        self.config = config
        self.bert = BertModel(config)
        #self.config = BertConfig.from_pretrained("bert-base-uncased")
        self.embeddings = copy.deepcopy(self.bert.embeddings)
        self.encoder = MyEncoder(self.bert.encoder)
    
    def get_input_embeddings(self):
            return self.bert.get_input_embeddings()

    def forward(self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        labels: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = True,
        return_dict: Optional[bool] = None,
        eval: Optional[bool] = False):

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            self.bert.warn_if_padding_and_no_attention_mask(input_ids, attention_mask)
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        batch_size, seq_length = input_shape
        device = input_ids.device if input_ids is not None else inputs_embeds.device

        # past_key_values_length
        past_key_values_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0

        if attention_mask is None:
            attention_mask = torch.ones(((batch_size, seq_length + past_key_values_length)), device=device)

        if token_type_ids is None:
            if hasattr(self.embeddings, "token_type_ids"):
                buffered_token_type_ids = self.embeddings.token_type_ids[:, :seq_length]
                buffered_token_type_ids_expanded = buffered_token_type_ids.expand(batch_size, seq_length)
                token_type_ids = buffered_token_type_ids_expanded
            else:
                token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        head_mask = self.bert.get_head_mask(head_mask, self.config.num_hidden_layers)

        encoder_extended_attention_mask = None
        extended_attention_mask: torch.Tensor = self.bert.get_extended_attention_mask(attention_mask, input_shape)

        embedding_output = self.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
            past_key_values_length=past_key_values_length,
        )

        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=None,
            encoder_attention_mask=encoder_extended_attention_mask,
            past_key_values=past_key_values,
            use_cache=False,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        return encoder_outputs

from transformers import BertPreTrainedModel, BertConfig

class LCNBertForSequenceClassification(nn.Module):
    def __init__(self, config, num_labels):
        super().__init__()
        #super(LCNBert, self).__init__()
        self.num_labels = num_labels
        self.config = config

        self.bert = LCNBert(config)
       
        # classifier_dropout = (
        #     config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        # )
        self.dropout = nn.Dropout(0.0)
        self.pooler = self.bert.bert.pooler
        self.classifier = nn.Linear(768, self.num_labels)

        # Initialize weights and apply final processing
        #self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        return_dict: Optional[bool] = None,
    ):
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        use_cache = False

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            self.bert.bert.warn_if_padding_and_no_attention_mask(input_ids, attention_mask)
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        batch_size, seq_length = input_shape
        device = input_ids.device if input_ids is not None else inputs_embeds.device

        # past_key_values_length
        past_key_values_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0

        if attention_mask is None:
            attention_mask = torch.ones(((batch_size, seq_length + past_key_values_length)), device=device)

        if token_type_ids is None:
            if hasattr(self.bert.embeddings, "token_type_ids"):
                buffered_token_type_ids = self.bert.embeddings.token_type_ids[:, :seq_length]
                buffered_token_type_ids_expanded = buffered_token_type_ids.expand(batch_size, seq_length)
                token_type_ids = buffered_token_type_ids_expanded
            else:
                token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
    

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]

        encoder_outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = encoder_outputs[0]
        hidden_states = encoder_outputs[1]

        pooled_output = self.pooler(sequence_output)
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)


        loss_fct = nn.CrossEntropyLoss()
        if labels is not None:
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
        else: loss = None

        current = hidden_states[0]
        intermid_preds = [self.classifier(self.pooler(current))]
        for h in hidden_states[1:]:
            current = current + h
            intermid_preds.append(self.classifier(self.pooler(current)))

        return loss, logits, intermid_preds, labels