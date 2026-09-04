from transformers import RobertaConfig, RobertaModel
import torch
import torch.nn as nn
from typing import List, Optional, Tuple, Union
import copy

class TransformerBlock(nn.Module):
    def __init__(self, d_model, nhead):
        super(TransformerBlock, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            (nn.Linear(d_model, 4 * d_model)),
            nn.ReLU(),
            (nn.Linear(4 * d_model, d_model))
        )
        self.relu =  nn.ReLU()
        self.linear = nn.Linear(d_model, d_model)

    def forward(self, x):
        attn_output, _ = self.self_attn(x, x, x)
        x = self.norm1(attn_output+x)
        ff_output = self.ffn(x)
        output = self.norm2(ff_output+x)
        output = self.relu(self.linear(output) + output)

        return output

class MyLayer(nn.Module):
    def __init__(self, l_copy, config):
        super().__init__()

        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1
        self.attention = copy.deepcopy(l_copy.attention)
        self.is_decoder = False
        self.add_cross_attention = False
        self.intermediate = copy.deepcopy(l_copy.intermediate)
        self.output = copy.deepcopy(l_copy.output)

        ####
        self.linear = nn.Linear(768, 768)
        self.relu = nn.ReLU()
        ####
        self.ln = nn.LayerNorm(768)

    def forward(
        self,
        aa,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_value: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor]:
        # decoder uni-directional self-attention cached key/values tuple is at positions 1,2


        self_attn_past_key_value = past_key_value[:2] if past_key_value is not None else None
        self_attention_outputs = self.attention(
            hidden_states,
            attention_mask,
            head_mask,
            output_attentions=output_attentions,
            past_key_value=self_attn_past_key_value,
        )
        attention_output = self_attention_outputs[0]

        intermediate_output = self.intermediate(attention_output)
        output = self.output(intermediate_output, attention_output)

        ######
        if not aa: output = self.ln(self.relu(self.linear(output) + output))
        ######

        return output


class MyEncoder(nn.Module):
    def __init__(self, copy_roberta_encoder, config):
        super().__init__()

        self.layer = nn.ModuleList([MyLayer(l, config) for l in copy_roberta_encoder.layer])
        self.gradient_checkpointing = False


        self.scalars = nn.ParameterList([nn.Parameter(torch.tensor(1.0), requires_grad=False) for _ in range(13)])


    def forward(
        self,
        aa,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = False,
        output_hidden_states: Optional[bool] = True,
        return_dict: Optional[bool] = True,
    ) -> Union[Tuple[torch.Tensor], torch.Tensor]:

        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None
        all_cross_attentions = () if output_attentions and self.config.add_cross_attention else None
        use_cache = False

        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (self.scalars[i]*hidden_states,)

            layer_head_mask = head_mask[i] if head_mask is not None else None
            past_key_value = past_key_values[i] if past_key_values is not None else None

            layer_outputs = layer_module(
                aa,
                hidden_states,
                attention_mask,
                layer_head_mask,
                encoder_hidden_states,
                encoder_attention_mask,
                past_key_value,
                output_attentions,
            )

            hidden_states = layer_outputs

        return (all_hidden_states + (self.scalars[-1]*hidden_states,), hidden_states)



class CustomRoberta(nn.Module):

    # Copied from transformers.models.clap.modeling_clap.ClapTextModel.__init__ with ClapText->Roberta
    def __init__(self, num_labels):
        super(CustomRoberta, self).__init__()

        self.config = RobertaConfig.from_pretrained("FacebookAI/roberta-base")
        #large 24 blocks of 1024 // base 12 blocks of 768
        self.roberta = RobertaModel.from_pretrained("FacebookAI/roberta-base", config=self.config)
        self.num_labels = num_labels
        self.dropout = nn.Dropout(p=0.3)
        self.classifier = nn.Linear(768, self.num_labels)

        self.embeddings = copy.deepcopy(self.roberta.embeddings)
        self.encoder = MyEncoder(self.roberta.encoder, self.config)
        self.pooler = copy.deepcopy(self.roberta.pooler)
        # Initialize weights and apply final processing
        self.roberta.post_init()

        self.last = TransformerBlock(768, 4)
        self.aa = False

    # Copied from transformers.models.clap.modeling_clap.ClapTextModel.forward
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        use_cache: Optional[bool] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = True,
        return_dict: Optional[bool] = None,
        eval: Optional[bool] = False
    ) -> Union[Tuple[torch.Tensor], torch.Tensor]:

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        use_cache = False

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            self.roberta.warn_if_padding_and_no_attention_mask(input_ids, attention_mask)
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

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask: torch.Tensor = self.roberta.get_extended_attention_mask(attention_mask, input_shape)

        encoder_extended_attention_mask = None

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.roberta.get_head_mask(head_mask, self.config.num_hidden_layers)

        embedding_output = self.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
            past_key_values_length=past_key_values_length,
        )
        encoder_outputs = self.encoder(
            self.aa,
            embedding_output,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=False,
            encoder_attention_mask=encoder_extended_attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        if self.aa: sequence_output = encoder_outputs[1]
        hidden_states = encoder_outputs[0]

        if not self.aa: sequence_output = torch.sum(torch.stack(hidden_states, dim=-1), dim=-1)
        #sequence_output = self.last(sequence_output)

        pooled_output = self.pooler(sequence_output)
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)


        if labels is None:
            if self.aa: intermid_preds = [self.classifier(self.pooler(h)) for h in hidden_states]
            if not self.aa:
                intermid_preds = []
                for i in range(13):
                    out = torch.sum(torch.stack(hidden_states[:i+1], dim=-1), dim=-1)
                    intermid_preds.append(self.classifier(self.pooler((out.detach().clone()))))

            return logits, intermid_preds

        else:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            return loss