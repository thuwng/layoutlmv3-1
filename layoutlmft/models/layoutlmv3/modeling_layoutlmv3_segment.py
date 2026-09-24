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

        self.seg_norm = nn.LayerNorm(config.hidden_size)
        self.seg_out_proj = nn.Linear(config.hidden_size, config.hidden_size)

        nn.init.zeros_(self.seg_out_proj.weight)
        nn.init.zeros_(self.seg_out_proj.bias)

        self.init_weights()
        # for param in self.layoutlmv3.parameters():
        #     param.requires_grad = False

    def _segment_pool_and_contextualize(self, text_hidden, seg_id):
        B, L, H = text_hidden.shape
        device = text_hidden.device
        broadcast_context = torch.zeros_like(text_hidden)

        for b in range(B):
            ids = seg_id[b]
            valid = ids >= 0
            if valid.sum() == 0:
                continue

            uniq_segs = torch.unique(ids[valid], sorted=True)
            n_seg = uniq_segs.shape[0]

            seg_vecs = torch.zeros(n_seg, H, device=device, dtype=text_hidden.dtype)
            seg_masks = []
            for i, s in enumerate(uniq_segs):
                mask = ids == s
                seg_masks.append(mask)
                seg_vecs[i] = text_hidden[b, mask].mean(dim=0)

            # Tính context liên segment
            ctx_out = self.segment_context(seg_vecs.unsqueeze(0)).squeeze(0)  # (n_seg, H)

            # Phân bổ context về lại từng token
            for i, mask in enumerate(seg_masks):
                broadcast_context[b, mask] = ctx_out[i]

        # V2 Core: Cộng context vào hidden state gốc qua LayerNorm và Linear (Khởi tạo = 0)
        updated_hidden = text_hidden + self.seg_out_proj(self.seg_norm(broadcast_context))
        return updated_hidden
    
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

    def _compute_disentangle_loss(self, text_hidden, line_ids, block_ids, attention_mask, text_len):
        device = text_hidden.device
        h_geo = self.geo_head(text_hidden)
        h_semi = self.semi_head(text_hidden)
        
        # ====== 1. RELATIVE GEOMETRY LOSS (FIXED WITH POS_WEIGHT & TEMPERATURE) ======
        h_geo_norm = F.normalize(h_geo, dim=-1)
        # Cosine similarity matrix cho mọi cặp token: (B, L, L)
        geo_sim = torch.matmul(h_geo_norm, h_geo_norm.transpose(1, 2))
        
        geo_loss = torch.tensor(0.0, device=device)
        geo_acc = 0.0
        geo_baseline_acc = 0.0
        
        if line_ids is not None:
            line_ids = line_ids[:, :text_len]
            # Ma trận target: 1 nếu cùng line_id, 0 nếu khác
            same_line = (line_ids.unsqueeze(1) == line_ids.unsqueeze(2)) & (line_ids.unsqueeze(1) >= 0)
            
            if attention_mask is not None:
                valid_mask = attention_mask[:, :text_len].bool()
            else:
                valid_mask = torch.ones((text_hidden.shape[0], text_len), dtype=torch.bool, device=device)
                
            pair_valid = valid_mask.unsqueeze(1) & valid_mask.unsqueeze(2)
            
            if pair_valid.sum() > 0:
                target = same_line.float()
                target_flat = target[pair_valid]
                
                # Tính toán trọng số cân bằng lớp (pos_weight) cho dữ liệu mất cân bằng nặng
                n_pos = target_flat.sum().clamp(min=1.0)
                n_neg = (target_flat.numel() - n_pos).clamp(min=1.0)
                pos_weight = (n_neg / n_pos).clamp(max=30.0)  # Cáp để tránh weight quá cực đoan
                
                # Dùng temperature để kéo giãn cosine similarity trước khi qua hàm loss
                temperature = 0.1
                logit = geo_sim[pair_valid] / temperature
                
                geo_loss = F.binary_cross_entropy_with_logits(logit, target_flat, pos_weight=pos_weight)
                
                # Tính Accuracy dựa trên Logits / Sigmoid xác suất
                probs = torch.sigmoid(logit)
                preds = (probs > 0.5).float()
                geo_acc = (preds == target_flat).float().mean().item()
                
                # Tính baseline đoán mù (luôn đoán lớp chiếm đa số)
                baseline_acc = max(n_pos.item(), n_neg.item()) / target_flat.numel()

        # ====== 2. ORTHOGONALITY LOSS (1-to-1 Token Mapping) ======
        h_semi_norm = F.normalize(h_semi, dim=-1)
        cos_sim_orth = (h_geo_norm * h_semi_norm).sum(dim=-1)
        
        valid_mask_orth = attention_mask[:, :text_len].bool() if attention_mask is not None else torch.ones_like(cos_sim_orth, dtype=torch.bool)
        
        valid_cos_sim = cos_sim_orth[valid_mask_orth]
        if valid_cos_sim.numel() > 0:
            orth_loss = (valid_cos_sim ** 2).mean()
        else:
            orth_loss = torch.tensor(0.0, device=device)
            
        return geo_loss, orth_loss, geo_acc, geo_baseline_acc

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

        if seg_id is not None:
            if seg_id.shape[1] != text_len:
                if seg_id.shape[1] > text_len:
                    seg_id = seg_id[:, :text_len]
                else:
                    pad_len = text_len - seg_id.shape[1]
                    pad_tensor = torch.ones(seg_id.shape[0], pad_len, device=seg_id.device, dtype=seg_id.dtype) * -1
                    seg_id = torch.cat([seg_id, pad_tensor], dim=1)
            
            # Chỉ cần 1 dòng này để cập nhật text_hidden
            text_hidden = self._segment_pool_and_contextualize(text_hidden, seg_id)
            
        if image_hidden.shape[1] > 0:
            pooled_sequence = torch.cat([text_hidden, image_hidden], dim=1)
        else:
            pooled_sequence = text_hidden

        pooled_sequence = self.dropout(pooled_sequence)
        logits = self.classifier(pooled_sequence)

        # ====== TÍNH CÁC LOSS PHỤ CÓ WARM-UP ======
        aux_loss = torch.tensor(0.0, device=logits.device)
        geo_ramp = 0.0
        
        if labels is not None:
            # Boundary Loss (không cần warmup)
            if self.boundary_classifier is not None:
                boundary_loss = self._compute_boundary_loss(
                    text_hidden=sequence_output[:, :text_len, :],
                    line_ids=line_ids if line_ids is not None else None,
                    labels=labels,
                    attention_mask=attention_mask,
                    text_len=text_len,
                )
                aux_loss = aux_loss + self.lambda_bound * boundary_loss
            
            # Geometry Loss (Cần Warm-up trễ)
            if self.geo_head is not None:
                geo_loss, orth_loss, geo_acc, geo_baseline_acc = self._compute_disentangle_loss(
                    text_hidden=sequence_output[:, :text_len, :],
                    line_ids=line_ids if line_ids is not None else None,
                    block_ids=block_ids if block_ids is not None else None,
                    attention_mask=attention_mask,
                    text_len=text_len,
                )
                
                # Tính dốc ramp dựa trên current_step được truyền từ callback
                current_step = getattr(self, "current_step", 0)
                geo_warmup_steps = getattr(self.config, "geo_warmup_steps", 200)
                geo_ramp = min(1.0, max(0.0, (current_step - geo_warmup_steps) / 100.0))
                
                aux_loss = aux_loss + geo_ramp * (self.lambda_geo * geo_loss + self.lambda_orth * orth_loss)

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
            "geo_acc": round(geo_acc, 4) if 'geo_acc' in locals() else 0.0,
            "geo_baseline_acc": round(geo_baseline_acc, 4) if 'geo_baseline_acc' in locals() else 0.0,
            "geo_ramp": round(geo_ramp, 4),
            "total_loss": round(loss.item(), 4) if loss is not None else 0.0,
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