#layoutlmft/models/layoutlmv3/modeling_layoutlmv3_segment.py
# coding=utf-8
#modeling_layoutlmv3_segment.py
"""
LayoutLMv3ForSegmentTokenClassification

Core idea (grounded in error analysis on FUNSD + CORD):
  - Segment self-consistency is already ~98-99% solved by the base model
    (confirmed empirically) -> a consistency REGULARIZER has little to gain.
  - The real errors are (a) whole segments classified wrong as a unit
    (esp. long free-text spans dropped entirely via BIO "drift"), and
    (b) confusions that depend on the NEIGHBORING segment's role
    (HEADER vs QUESTION on FUNSD; parent vs sub-item on CORD).
  - Fix: pool each segment's token hidden states into one vector, run a
    tiny Transformer encoder over the SEQUENCE of segment vectors (reading
    order) so adjacent segments exchange information, then broadcast the
    context-enriched vector back to every token in the segment before the
    (unchanged) token classifier.
  - To keep the existing BIO scheme / seqeval / compute_metrics pipeline
    100% unchanged, we do NOT collapse labels to entity-type-only. Instead
    we add a tiny learned "is-first-token-of-segment" embedding so the
    (otherwise identical) broadcast vector can still support the B-/I-
    distinction at the classifier.

This class does NOT touch attention, does NOT build any graph/hypergraph,
and does NOT modify the pretrained backbone. It only replaces what the
token classifier head "sees" for tokens inside multi-token segments -- an
orthogonal mechanism to HGA / GraphLayoutLM.
"""
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from transformers.modeling_outputs import TokenClassifierOutput
import torch.nn.functional as F

from .modeling_layoutlmv3 import (
    LayoutLMv3ClassificationHead,
    LayoutLMv3Model,
    LayoutLMv3PreTrainedModel,
)

class LayoutLMv3ForSegmentTokenClassification(LayoutLMv3PreTrainedModel):
    _keys_to_ignore_on_load_unexpected = [r"pooler"]
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.layoutlmv3 = LayoutLMv3Model(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        if config.num_labels < 10:
            self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        else:
            self.classifier = LayoutLMv3ClassificationHead(config, pool_feature=False)

        # ---- NEW: lightweight inter-segment context module ----
        # Config knobs (optional; safe defaults if not set on the config object).
        seg_ctx_layers = getattr(config, "segment_context_layers", 1)
        seg_ctx_heads = getattr(config, "segment_context_heads", 4)
        seg_ctx_dropout = getattr(config, "segment_context_dropout", config.hidden_dropout_prob)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.hidden_size,
            nhead=seg_ctx_heads,
            dim_feedforward=config.hidden_size * 2,
            dropout=seg_ctx_dropout,
            batch_first=True,
        )
        self.segment_context = nn.TransformerEncoder(encoder_layer, num_layers=seg_ctx_layers)
        if getattr(config, "use_intra_line_boundary", False):
            self.boundary_classifier = nn.Sequential(
                nn.Linear(config.hidden_size * 2, config.hidden_size),
                nn.ReLU(),
                nn.Linear(config.hidden_size, 1)
            )
            self.lambda_bound = nn.Parameter(
                torch.tensor(getattr(config, "lambda_bound_init", 0.1))
            )
        else:
            self.boundary_classifier = None
            self.lambda_bound = None
        
        # ====== SEMANTIC-GEOMETRY DISENTANGLE ======
        if getattr(config, "use_semantic_geometry_disentangle", False):
            self.geo_head = nn.Linear(config.hidden_size, config.hidden_size)
            self.geo_line_classifier = nn.Linear(config.hidden_size, config.max_line_position)
            self.geo_block_classifier = nn.Linear(config.hidden_size, config.max_block_position)
            self.semi_head = nn.Linear(config.hidden_size, config.hidden_size)
            
            # ====== SỬA: Dùng float thay vì Parameter ======
            self.lambda_geo = float(getattr(config, "lambda_geo_init", 0.1))
            self.lambda_orth = float(getattr(config, "lambda_orth_init", 0.1))
        else:
            self.geo_head = None
            self.geo_line_classifier = None
            self.geo_block_classifier = None
            self.semi_head = None
            self.lambda_geo = 0.0
            self.lambda_orth = 0.0
                # ReZero-style gate: starts at 0 so at step 0 the context module is a
        # NO-OP (output == plain mean-pooled vector, i.e. identical to a
        # "segment pooling only, no inter-segment context" ablation). Training
        # then gradually learns how much of the (initially random) context
        # transform to blend in. This avoids injecting a large random
        # perturbation into a well-pretrained backbone's features right at
        # the start of fine-tuning -- important on tiny datasets like FUNSD
        # (149 docs) where a high-variance early gradient can permanently
        # damage the pretrained representation.
        self.segment_context_gate = nn.Parameter(torch.zeros(1))

        # Small embedding so the classifier can still tell "first token of the
        # segment" (-> should predict B-xxx) apart from the rest (-> I-xxx),
        # even though every token in the segment otherwise shares one pooled
        # vector. Initialized near zero so early training resembles the
        # unmodified baseline.
        self.is_first_token_embedding = nn.Embedding(2, config.hidden_size)
        nn.init.normal_(self.is_first_token_embedding.weight, mean=0.0, std=0.02)

        self.init_weights()
        # for param in self.layoutlmv3.parameters():
        #     param.requires_grad = False

    def _segment_pool_and_contextualize(self, text_hidden, seg_id):
        """
        text_hidden: (B, L, H) hidden states for the TEXT part only
                     (image-patch positions, if any, are handled separately
                     by the caller and never enter this function).
        seg_id:      (B, L) long tensor. -1 marks tokens that do not belong
                     to any segment (special tokens / padding). Non-negative
                     values are LOCAL segment indices per example, assigned
                     in reading order (0, 1, 2, ...), exactly matching the
                     bbox-equality grouping used in run_funsd_cord.py's
                     tokenize_and_align_labels (see patch).

        Returns:
            broadcast_hidden: (B, L, H) -- every token belonging to the same
                segment gets an IDENTICAL context-enriched vector (before the
                is-first-token embedding is added back in `forward`).
        """
        B, L, H = text_hidden.shape
        device = text_hidden.device
        broadcast_hidden = text_hidden.clone()

        for b in range(B):
            ids = seg_id[b]
            valid = ids >= 0
            if valid.sum() == 0:
                continue

            uniq_segs = torch.unique(ids[valid], sorted=True)  # reading order
            n_seg = uniq_segs.shape[0]

            seg_vecs = torch.zeros(n_seg, H, device=device, dtype=text_hidden.dtype)
            seg_masks = []
            for i, s in enumerate(uniq_segs):
                mask = ids == s
                seg_masks.append(mask)
                seg_vecs[i] = text_hidden[b, mask].mean(dim=0)

            # The only place adjacent segments exchange information.
            # Cheap: n_seg is typically tens, not hundreds, per document.
            ctx_out = self.segment_context(seg_vecs.unsqueeze(0)).squeeze(0)  # (n_seg, H)
            # ReZero blend: at init (gate=0) this reduces to seg_vecs_ctx == seg_vecs
            # (pure mean-pooling, no context) -- see comment on self.segment_context_gate.
            seg_vecs_ctx = seg_vecs + self.segment_context_gate * (ctx_out - seg_vecs)

            for i, mask in enumerate(seg_masks):
                broadcast_hidden[b, mask] = seg_vecs_ctx[i]

        return broadcast_hidden
    def _compute_boundary_loss(self, text_hidden, line_ids, labels, attention_mask, text_len):
        """
        Tính Intra-Line Boundary Loss.
        
        Với mỗi cặp token liên tiếp cùng dòng:
        - Label = 0 nếu cùng entity
        - Label = 1 nếu khác entity
        """
        device = text_hidden.device
        B = text_hidden.shape[0]
        
        boundary_loss = 0.0
        n_pairs = 0
        
        for b in range(B):
            for i in range(text_len - 1):
                # Kiểm tra token hợp lệ
                if attention_mask is not None:
                    if attention_mask[b, i] != 1 or attention_mask[b, i+1] != 1:
                        continue
                
                # Kiểm tra cùng dòng
                if line_ids is not None:
                    if line_ids[b, i] < 0 or line_ids[b, i+1] < 0:
                        continue
                    if line_ids[b, i] != line_ids[b, i+1]:
                        continue
                
                # Kiểm tra label hợp lệ
                if labels[b, i] < 0 or labels[b, i+1] < 0:
                    continue
                
                # Tạo feature cho cặp
                h_pair = torch.cat([text_hidden[b, i], text_hidden[b, i+1]], dim=-1)
                boundary_logit = self.boundary_classifier(h_pair).squeeze(-1)
                
                # Xác định label boundary
                label_i = labels[b, i].item()
                label_i_next = labels[b, i+1].item()
                
                # Cùng entity nếu:
                # - Cả 2 cùng label (cùng I-X)
                # - Hoặc label_i là B-X và label_i_next là I-X
                same_entity = False
                
                # Case 1: Cả 2 cùng label
                if label_i == label_i_next:
                    # Nhưng nếu cả 2 đều là B-X thì khác entity
                    # (vì B- là bắt đầu entity mới)
                    # Giả sử label_list có dạng: O, B-X, I-X
                    # Ta cần biết label_i có phải B- không
                    # Đơn giản: nếu label_i % 2 == 0 thì là B- hoặc O
                    # (tùy vào label_list cụ thể)
                    if label_i == 0:  # O
                        same_entity = True
                    # Nếu label_i là B- (thường là label chẵn), thì label_i_next 
                    # cũng B- → khác entity
                    # Ta cần kiểm tra cụ thể hơn
                    same_entity = (label_i == label_i_next)
                
                # Case 2: label_i là B-X, label_i_next là I-X
                # Trong FUNSD: O=0, B-HEADER=1, I-HEADER=2, B-QUESTION=3, I-QUESTION=4, ...
                # B-X có label lẻ, I-X có label chẵn (label_i + 1)
                if label_i % 2 == 1 and label_i_next == label_i + 1:
                    same_entity = True
                
                boundary_label = 0.0 if same_entity else 1.0
                
                boundary_loss += F.binary_cross_entropy_with_logits(
                    boundary_logit, 
                    torch.tensor(boundary_label, device=device)
                )
                n_pairs += 1
        
        if n_pairs > 0:
            boundary_loss = boundary_loss / n_pairs
        else:
            boundary_loss = torch.tensor(0.0, device=device)
        
        return boundary_loss

    def _compute_disentangle_loss(self, text_hidden, line_ids, block_ids, 
                                attention_mask, text_len):
        device = text_hidden.device
        B = text_hidden.shape[0]
        
        # Cắt line_ids, block_ids về text_len
        if line_ids is not None and line_ids.shape[1] > text_len:
            line_ids = line_ids[:, :text_len]
        if block_ids is not None and block_ids.shape[1] > text_len:
            block_ids = block_ids[:, :text_len]
        
        h_geo = self.geo_head(text_hidden)
        h_semi = self.semi_head(text_hidden)
        
        # ====== GEOMETRY LOSS ======
        geo_loss = torch.tensor(0.0, device=device)
        
        if line_ids is not None:
            line_logits = self.geo_line_classifier(h_geo)
            block_logits = self.geo_block_classifier(h_geo)
            
            valid_mask = (line_ids >= 0) & (block_ids >= 0)
            if attention_mask is not None:
                text_attention_mask = attention_mask[:, :text_len]
                valid_mask = valid_mask & (text_attention_mask == 1)
            
            if valid_mask.sum() > 0:
                line_ids_clamped = torch.clamp(line_ids, 0, self.geo_line_classifier.out_features - 1)
                block_ids_clamped = torch.clamp(block_ids, 0, self.geo_block_classifier.out_features - 1)
                
                geo_loss_line = F.cross_entropy(
                    line_logits[valid_mask], 
                    line_ids_clamped[valid_mask]
                )
                geo_loss_block = F.cross_entropy(
                    block_logits[valid_mask],
                    block_ids_clamped[valid_mask]
                )
                geo_loss = geo_loss_line + geo_loss_block
                
                # ====== KIỂM TRA NaN ======
                if torch.isnan(geo_loss) or torch.isinf(geo_loss):
                    geo_loss = torch.tensor(0.0, device=device)
        
        # ====== ORTHOGONALITY LOSS ======
        h_geo_norm = F.normalize(h_geo, dim=-1)
        h_semi_norm = F.normalize(h_semi, dim=-1)
        
        cos_sim = torch.matmul(h_geo_norm, h_semi_norm.transpose(1, 2))
        
        if attention_mask is not None:
            text_attention_mask = attention_mask[:, :text_len]
            valid_mask_2d = text_attention_mask.bool()
            pair_mask = valid_mask_2d.unsqueeze(1) & valid_mask_2d.unsqueeze(2)
            cos_sim = cos_sim * pair_mask.float()
        
        orth_loss = (cos_sim ** 2).mean()
        
        # ====== KIỂM TRA NaN ======
        if torch.isnan(orth_loss) or torch.isinf(orth_loss):
            orth_loss = torch.tensor(0.0, device=device)
        
        return geo_loss, orth_loss

    def forward(
        self,
        input_ids=None,
        bbox=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        valid_span=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        seg_id=None,
        line_ids=None,
        block_ids=None,
        column_ids=None, 
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        images=None,
        
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.layoutlmv3(
            input_ids,
            bbox=bbox,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            images=images,
            valid_span=valid_span,
            line_ids=line_ids,
            block_ids=block_ids,
            column_ids=column_ids, 
        )

        sequence_output = outputs[0]  # (B, text_len + image_len, H)
        text_len = input_ids.shape[1]
        text_hidden = sequence_output[:, :text_len, :]
        image_hidden = sequence_output[:, text_len:, :]

        # ====== SỬA: Cắt seg_id để chỉ lấy phần text ======
        if seg_id is not None:
            # Đảm bảo seg_id có đúng độ dài text
            if seg_id.shape[1] != text_len:
                # Nếu seg_id dài hơn text_len, chỉ lấy phần text
                if seg_id.shape[1] > text_len:
                    seg_id = seg_id[:, :text_len]
                else:
                    # Nếu seg_id ngắn hơn, pad với -1
                    pad_len = text_len - seg_id.shape[1]
                    pad_tensor = torch.ones(seg_id.shape[0], pad_len, device=seg_id.device, dtype=seg_id.dtype) * -1
                    seg_id = torch.cat([seg_id, pad_tensor], dim=1)
            
            text_hidden = self._segment_pool_and_contextualize(text_hidden, seg_id)

            # Add the is-first-token-of-segment signal
            is_first = torch.zeros_like(seg_id, dtype=torch.long)
            is_first[:, 0] = 0
            if seg_id.shape[1] > 1:
                prev = seg_id[:, :-1]
                cur = seg_id[:, 1:]
                changed = (cur != prev) & (cur >= 0)
                is_first[:, 1:] = changed.long()
            is_first = is_first * (seg_id >= 0).long()

            text_hidden = text_hidden + self.is_first_token_embedding(is_first)

        if image_hidden.shape[1] > 0:
            pooled_sequence = torch.cat([text_hidden, image_hidden], dim=1)
        else:
            pooled_sequence = text_hidden

        pooled_sequence = self.dropout(pooled_sequence)
        logits = self.classifier(pooled_sequence)
        # ====== TÍNH CÁC LOSS PHỤ ======
        aux_loss = torch.tensor(0.0, device=logits.device)

        if labels is not None:
            # ====== INTRA-LINE BOUNDARY LOSS ======
            if self.boundary_classifier is not None:
                boundary_loss = self._compute_boundary_loss(
                    text_hidden=sequence_output[:, :text_len, :],
                    line_ids=line_ids if line_ids is not None else None,
                    labels=labels,
                    attention_mask=attention_mask,
                    text_len=text_len,
                )
                aux_loss = aux_loss + self.lambda_bound * boundary_loss
            
            # ====== SEMANTIC-GEOMETRY DISENTANGLE LOSS ======
            if self.geo_head is not None:
                geo_loss, orth_loss = self._compute_disentangle_loss(
                    text_hidden=sequence_output[:, :text_len, :],
                    line_ids=line_ids if line_ids is not None else None,
                    block_ids=block_ids if block_ids is not None else None,
                    attention_mask=attention_mask,
                    text_len=text_len,
                )
                aux_loss = aux_loss + self.lambda_geo * geo_loss + self.lambda_orth * orth_loss
        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            if attention_mask is not None:
                active_loss = attention_mask.view(-1) == 1
                active_logits = logits.view(-1, self.num_labels)
                active_labels = torch.where(
                    active_loss, labels.view(-1), torch.tensor(loss_fct.ignore_index).type_as(labels)
                )
                loss = loss_fct(active_logits, active_labels)
            else:
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

            loss = loss + aux_loss

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )