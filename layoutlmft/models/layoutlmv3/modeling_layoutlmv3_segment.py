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
            self.lambda_bound = float(getattr(config, "lambda_bound_init", 0.1))
        else:
            self.boundary_classifier = None
            self.lambda_bound = 0.0
        
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
        device = text_hidden.device
        
        # 1. Cắt tất cả tensor về giới hạn text_len để loại bỏ visual tokens (giải quyết triệt để lỗi 708 vs 511)
        labels = labels[:, :text_len]
        if attention_mask is not None:
            attention_mask = attention_mask[:, :text_len]
        if line_ids is not None:
            line_ids = line_ids[:, :text_len]
            
        # 2. Trượt tensor để tạo cặp (i, i+1) cho phần văn bản
        h_i = text_hidden[:, :-1, :]
        h_j = text_hidden[:, 1:, :]
        h_pair = torch.cat([h_i, h_j], dim=-1)  # (B, L-1, 2H)
        
        boundary_logit = self.boundary_classifier(h_pair).squeeze(-1)  # (B, L-1)
        
        li = labels[:, :-1]
        lj = labels[:, 1:]
        
        # 3. Tạo mask hợp lệ
        valid = (li >= 0) & (lj >= 0)
        
        if attention_mask is not None:
            valid = valid & (attention_mask[:, :-1] == 1) & (attention_mask[:, 1:] == 1)
            
        if line_ids is not None:
            valid = valid & (line_ids[:, :-1] >= 0) & (line_ids[:, :-1] == line_ids[:, 1:])
            
        # 4. Xác định target cho Boundary Loss chuẩn xác
        is_O = (li == 0)  # O luôn có ID = 0 nhờ assert của bạn
        is_B = (li % 2 == 1)
        is_I = (li > 0) & (li % 2 == 0)

        # Cùng entity khi: (O kề O) HOẶC (B kề I tương ứng) HOẶC (I kề I tương ứng)
        same_entity = (is_O & (lj == 0)) | \
                    (is_B & (lj == li + 1)) | \
                    (is_I & (lj == li))

        target = (~same_entity).float()
        
        # 5. Tính Loss (BCE)
        loss_per_pair = F.binary_cross_entropy_with_logits(boundary_logit, target, reduction="none")
        loss_per_pair = loss_per_pair * valid.float()
        
        denom = valid.float().sum().clamp(min=1.0)
        return loss_per_pair.sum() / denom

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
        
        # ====== ORTHOGONALITY LOSS (FIXED: 1-to-1 Token Mapping) ======
        h_geo_norm = F.normalize(h_geo, dim=-1)
        h_semi_norm = F.normalize(h_semi, dim=-1)
        
        # Tính Cosine Sim giữa geo và semi của CÙNG MỘT token (B, L)
        cos_sim = (h_geo_norm * h_semi_norm).sum(dim=-1)
        
        if attention_mask is not None:
            valid_mask = attention_mask[:, :text_len].bool()
        else:
            valid_mask = torch.ones_like(cos_sim, dtype=torch.bool)
        
        # Tính trung bình bình phương Cosine Similarity trên các token hợp lệ
        valid_cos_sim = cos_sim[valid_mask]
        if valid_cos_sim.numel() > 0:
            orth_loss = (valid_cos_sim ** 2).mean()
        else:
            orth_loss = torch.tensor(0.0, device=device)
            
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
        ce_loss_val = 0.0
        
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            if attention_mask is not None:
                active_loss = attention_mask.view(-1) == 1
                active_logits = logits.view(-1, self.num_labels)
                active_labels = torch.where(
                    active_loss, labels.view(-1), torch.tensor(loss_fct.ignore_index).type_as(labels)
                )
                ce_loss = loss_fct(active_logits, active_labels)
            else:
                ce_loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

            ce_loss_val = ce_loss.item()
            loss = ce_loss + aux_loss

        # ====== LƯU VẾT CÁC THÀNH PHẦN LOSS ĐỂ TRACKING ======
        self.loss_tracker = {
            "ce_loss": round(ce_loss_val, 4),
            "boundary_loss": round(boundary_loss.item(), 4) if 'boundary_loss' in locals() else 0.0,
            "geo_loss": round(geo_loss.item(), 4) if 'geo_loss' in locals() else 0.0,
            "orth_loss": round(orth_loss.item(), 4) if 'orth_loss' in locals() else 0.0,
            "total_loss": round(loss.item(), 4) if loss is not None else 0.0,
            "lam_bnd": getattr(self, "lambda_bound", 0.0),
            "lam_geo": getattr(self, "lambda_geo", 0.0),
            "lam_orth": getattr(self, "lambda_orth", 0.0)
        }

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )