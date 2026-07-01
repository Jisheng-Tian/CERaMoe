import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.nn import LayerNorm
from torch.autograd import Variable
import math
import logging


class LearnablePositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0, max_len=1000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.embeddings = nn.Embedding(max_len, d_model)
        initrange = 0.1
        self.embeddings.weight.data.uniform_(-initrange, initrange)

    def forward(self, x):
        pos = torch.arange(0, x.size(1), device=x.device).int().unsqueeze(0)
        x = x + self.embeddings(pos).expand_as(x)
        return x


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) *
            -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        pe *= 0.1
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + Variable(self.pe[:, :x.size(1)], requires_grad=False)
        return self.dropout(x)


def count_parameters(model, args=None):
    use_moe = getattr(args, 'use_moe', False) if args is not None else False

    detail = {
        'embedding':      0,
        'attention':      0,
        'router':         0,
        'rarity_encoder': 0,
        'shared_expert':  0,
        'routed_experts': 0,
        'ffn_standard':   0,
        'output_heads':   0,
        'causal_module':  0,  # CERaMoE 新增
        'other':          0,
    }

    for name, param in model.named_parameters():
        n = param.numel()
        if any(k in name for k in ['embeddings', 'special_embeddings',
                                    'segment_embedding', 'positional']):
            detail['embedding'] += n
        elif 'self_attn' in name or 'norm1' in name or 'norm2' in name:
            detail['attention'] += n
        elif '.moe_ffn.router.' in name or '.moe_ffn.rarity_bias' in name \
                or '.moe_ffn.rarity_scale' in name:
            detail['router'] += n
        elif 'rarity_context_encoder' in name:
            detail['rarity_encoder'] += n
        elif '.moe_ffn.shared_expert.' in name:
            detail['shared_expert'] += n
        elif '.moe_ffn.experts.' in name:
            detail['routed_experts'] += n
        elif '.linear' in name and 'transformer' in name:
            detail['ffn_standard'] += n
        elif name.startswith('cls_'):
            detail['output_heads'] += n
        elif 'causal_bias' in name or 'causal_scale' in name:
            detail['causal_module'] += n
        else:
            detail['other'] += n

    total_params = sum(detail.values())

    if not use_moe:
        activated_params = total_params
    else:
        num_experts = getattr(args, 'num_experts', 4)
        top_k = getattr(args, 'top_k', 2)
        activated_expert = int(round(detail['routed_experts'] * top_k / num_experts))
        activated_params = (
            detail['embedding']
            + detail['attention']
            + detail['router']
            + detail['rarity_encoder']
            + detail['shared_expert']
            + activated_expert
            + detail['ffn_standard']
            + detail['output_heads']
            + detail['causal_module']
            + detail['other']
        )

    return total_params, activated_params, detail


def format_param_count(n):
    if n >= 1_000_000:
        return f'{n / 1_000_000:.2f}M'
    elif n >= 1_000:
        return f'{n / 1_000:.1f}K'
    return str(n)


class ExpertFFN(nn.Module):
    def __init__(self, d_model, d_ff=None, dropout=0.1):
        super().__init__()
        d_ff = d_ff or d_model * 4
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.dropout(F.relu(self.linear1(x)))
        x = self.linear2(x)
        return x


class RarityContextEncoder(nn.Module):

    def __init__(self, d_model, in_dim=3):
        super().__init__()
        self.in_dim = in_dim
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
        )
        for m in self.mlp:
            if isinstance(m, nn.Linear):
                nn.init.uniform_(m.weight, -0.01, 0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.mlp(x)


class MoEFFN(nn.Module):

    def __init__(self, d_model, num_experts=4, top_k=2, d_ff=None, dropout=0.1,
                 use_shared_expert=False, use_rarity_router=False):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.d_model = d_model
        self.use_shared_expert = use_shared_expert
        self.use_rarity_router = use_rarity_router

        if use_shared_expert:
            self.shared_expert = ExpertFFN(d_model, d_ff, dropout)

        self.experts = nn.ModuleList([
            ExpertFFN(d_model, d_ff, dropout) for _ in range(num_experts)
        ])

        self.router = nn.Linear(d_model, num_experts, bias=False)

        if use_rarity_router:
            self.rarity_bias = nn.Linear(d_model, num_experts, bias=False)
            self.rarity_scale = nn.Parameter(torch.tensor(0.1))

        self._rarity_context = None

        self.register_buffer('aux_loss', torch.tensor(0.0))
        self.register_buffer('router_entropy', torch.tensor(0.0), persistent=False)
        self.register_buffer('expert_usage', torch.zeros(num_experts), persistent=False)
        self.register_buffer('routed_usage', torch.zeros(num_experts), persistent=False)

        self._last_f = None
        self._last_P = None

    def set_rarity_context(self, context):
        self._rarity_context = context

    def forward(self, x):
        seq_len, batch_size, d_model = x.shape
        x_flat = x.reshape(-1, d_model)

        router_logits = self.router(x_flat)

        if self.use_rarity_router and self._rarity_context is not None:
            rarity_bias_logits = self.rarity_bias(self._rarity_context)
            router_logits = router_logits + self.rarity_scale * rarity_bias_logits

        router_probs = F.softmax(router_logits, dim=-1)

        top_k_probs, top_k_indices = torch.topk(router_probs, self.top_k, dim=-1)
        top_k_gates = top_k_probs / (top_k_probs.sum(dim=-1, keepdim=True) + 1e-9)

        top1_indices = top_k_indices[:, 0]
        expert_mask = F.one_hot(top1_indices, self.num_experts).float()
        f = expert_mask.mean(dim=0)
        P = router_probs.mean(dim=0)

        self._last_f = f.detach()
        self._last_P = P

        token_entropy = -(router_probs * torch.log(router_probs + 1e-9)).sum(dim=-1)
        self.router_entropy = token_entropy.mean().detach()
        self.expert_usage = f.detach()

        routed_usage = torch.zeros(self.num_experts, device=x.device)
        routed_usage.scatter_add_(0, top_k_indices.reshape(-1),
                                   top_k_gates.reshape(-1).detach())
        self.routed_usage = (routed_usage / (routed_usage.sum() + 1e-9)).detach()

        if self.training:
            self.aux_loss = self.num_experts * (f.detach() * P).sum()
        else:
            self.aux_loss.fill_(0.0)

        output = torch.zeros_like(x_flat)
        for i, expert in enumerate(self.experts):
            mask = (top_k_indices == i)
            if not mask.any():
                continue
            token_indices = mask.any(dim=1).nonzero(as_tuple=True)[0]
            gate_weights = (top_k_gates * mask.float()).sum(dim=1)[token_indices]
            expert_out = expert(x_flat[token_indices])
            output[token_indices] += gate_weights.unsqueeze(-1) * expert_out

        if self.use_shared_expert:
            output = output + self.shared_expert(x_flat)

        return output.reshape(seq_len, batch_size, d_model)


class MoETransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, num_experts=4, top_k=2, d_ff=None, dropout=0.1,
                 use_shared_expert=False, use_rarity_router=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.moe_ffn = MoEFFN(d_model, num_experts, top_k, d_ff, dropout,
                              use_shared_expert, use_rarity_router)
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        src2 = self.self_attn(src, src, src,
                              attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)

        src2 = self.moe_ffn(src)
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src


class MoETransformerEncoder(nn.Module):
    def __init__(self, encoder_layer_args, num_layers):
        super().__init__()
        self.layers = nn.ModuleList([
            MoETransformerEncoderLayer(**encoder_layer_args)
            for _ in range(num_layers)
        ])
        self.num_layers = num_layers

    def set_rarity_context(self, context):
        for layer in self.layers:
            layer.moe_ffn.set_rarity_context(context)

    def clear_rarity_context(self):
        for layer in self.layers:
            layer.moe_ffn._rarity_context = None

    def forward(self, src, mask=None, src_key_padding_mask=None):
        output = src
        for layer in self.layers:
            output = layer(output, src_mask=mask,
                           src_key_padding_mask=src_key_padding_mask)

        avg_f = torch.zeros(self.layers[0].moe_ffn.num_experts, device=output.device)
        avg_P = torch.zeros(self.layers[0].moe_ffn.num_experts, device=output.device)
        for layer in self.layers:
            avg_f = avg_f + layer.moe_ffn._last_f
            avg_P = avg_P + layer.moe_ffn._last_P
        avg_f = avg_f / self.num_layers
        avg_P = avg_P / self.num_layers

        self._last_visit_f = avg_f
        self._last_visit_P = avg_P

        return output

    def get_aux_loss(self):
        total_aux_loss = torch.tensor(0.0, device=next(self.parameters()).device)
        for layer in self.layers:
            total_aux_loss = total_aux_loss + layer.moe_ffn.aux_loss
        return total_aux_loss / self.num_layers

    def get_last_visit_routing(self):
        return self._last_visit_f, self._last_visit_P

    def get_diagnostics(self):
        device = next(self.parameters()).device
        total_entropy = torch.tensor(0.0, device=device)
        num_experts = self.layers[0].moe_ffn.num_experts
        total_top1_usage = torch.zeros(num_experts, device=device)
        total_routed_usage = torch.zeros(num_experts, device=device)
        for layer in self.layers:
            total_entropy = total_entropy + layer.moe_ffn.router_entropy
            total_top1_usage = total_top1_usage + layer.moe_ffn.expert_usage
            total_routed_usage = total_routed_usage + layer.moe_ffn.routed_usage
        return {
            'router_entropy': total_entropy / self.num_layers,
            'expert_usage': total_top1_usage / self.num_layers,
            'routed_usage': total_routed_usage / self.num_layers,
        }


class PatientEncoder(nn.Module):
    def __init__(self, args, voc_size):
        super().__init__()
        self.args = args
        self.voc_size = voc_size
        self.emb_dim = args.embed_dim
        self.device = torch.device('cuda:{}'.format(args.cuda))

        self.use_moe = getattr(args, 'use_moe', False)
        self.num_experts = getattr(args, 'num_experts', 4)
        self.top_k = getattr(args, 'top_k', 2)

        self.use_shared_expert = getattr(args, 'use_shared_expert', False)
        self.use_rarity_router = getattr(args, 'use_rarity_router', False)
        self.use_mono_constraint = getattr(args, 'use_mono_constraint', False)

        self.use_causal_router = getattr(args, 'use_causal_router', False)

        rarity_in_dim = 4 if self.use_causal_router else 3
        self.rarity_in_dim = rarity_in_dim

        self.special_tokens = {
            'CLS': torch.LongTensor([0,]).to(self.device),
            'SEP': torch.LongTensor([1,]).to(self.device)
        }
        self.segment_embedding = nn.Embedding(2, self.emb_dim)

        if args.patient_seperate == False:
            self.embeddings = nn.ModuleList(
                [nn.Embedding(voc_size[i], self.emb_dim) for i in range(2)])
            self.special_embeddings = nn.Embedding(2, self.emb_dim)

            if self.use_moe:
                encoder_layer_args = {
                    'd_model': self.emb_dim,
                    'nhead': args.nhead,
                    'num_experts': self.num_experts,
                    'top_k': self.top_k,
                    'd_ff': self.emb_dim * 4,
                    'dropout': args.dropout,
                    'use_shared_expert': self.use_shared_expert,
                    'use_rarity_router': self.use_rarity_router,
                }
                self.transformer_visit = MoETransformerEncoder(
                    encoder_layer_args, num_layers=args.encoder_layers
                )
                if self.use_rarity_router:
                    self.rarity_context_encoder = RarityContextEncoder(
                        self.emb_dim, in_dim=rarity_in_dim
                    )
            else:
                self.transformer_visit = nn.TransformerEncoder(
                    nn.TransformerEncoderLayer(
                        d_model=self.emb_dim, nhead=args.nhead, dropout=args.dropout
                    ),
                    num_layers=args.encoder_layers
                )

            self.positional_embedding_layer_disease = LearnablePositionalEncoding(
                d_model=args.embed_dim)
            self.positional_embedding_layer_procedure = LearnablePositionalEncoding(
                d_model=args.embed_dim)
            self.patient_encoder = self.patient_encoder_unified
        else:
            self.embeddings = nn.ModuleList(
                [nn.Embedding(voc_size[i], self.emb_dim // 2) for i in range(2)])
            self.special_embeddings = nn.Embedding(2, self.emb_dim // 2)
            self.transformer_disease = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(
                    d_model=self.emb_dim // 2, nhead=args.nhead, dropout=args.dropout),
                num_layers=args.encoder_layers
            )
            self.transformer_procedure = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(
                    d_model=self.emb_dim // 2, nhead=args.nhead, dropout=args.dropout),
                num_layers=args.encoder_layers
            )
            self.patient_layer = nn.Sequential(
                nn.Linear(self.emb_dim, self.emb_dim),
                nn.ReLU(),
                nn.Linear(self.emb_dim, self.emb_dim),
            )
            self.positional_embedding_layer_disease = LearnablePositionalEncoding(
                d_model=args.embed_dim // 2)
            self.positional_embedding_layer_procedure = LearnablePositionalEncoding(
                d_model=args.embed_dim // 2)
            self.patient_encoder = self.patient_encoder_seperate


    def set_causal_matrices(self, M_dm_pos, M_dm_signed, M_pm_pos, M_pm_signed):
        device = self.device
        self.register_buffer('M_dm_pos',
            torch.from_numpy(M_dm_pos).float().to(device), persistent=True)
        self.register_buffer('M_dm_signed',
            torch.from_numpy(M_dm_signed).float().to(device), persistent=True)
        self.register_buffer('M_pm_pos',
            torch.from_numpy(M_pm_pos).float().to(device), persistent=True)
        self.register_buffer('M_pm_signed',
            torch.from_numpy(M_pm_signed).float().to(device), persistent=True)
        self._causal_matrices_loaded = True
        logging.info(f"[CERaMoE] 因果矩阵已注入模型 buffer: "
                     f"M_dm{tuple(self.M_dm_pos.shape)}, "
                     f"M_pm{tuple(self.M_pm_pos.shape)}")

    def _compute_causal_salience_for_visit(self, diag_ids, proc_ids):
        if not self._causal_matrices_loaded or len(diag_ids) + len(proc_ids) == 0:
            return 0.0

        with torch.no_grad():
            n_m = self.M_dm_pos.shape[1]
            raw = torch.zeros(n_m, device=self.device, dtype=self.M_dm_pos.dtype)
            if len(diag_ids) > 0:
                d_idx = torch.LongTensor(diag_ids).to(self.device)
                raw = raw + self.M_dm_pos[d_idx].sum(0)
            if len(proc_ids) > 0:
                p_idx = torch.LongTensor(proc_ids).to(self.device)
                raw = raw + self.M_pm_pos[p_idx].sum(0)

            total = raw.sum() + 1e-9
            if total < 1e-6:
                return 0.0
            peak = raw.max()
            return float(peak / total)

    def patient_encoder_unified(self, batch_visits):
        batch_repr = []
        self._visit_rarity_scores = []
        self._visit_causal_saliences = []  # ★ CERaMoE 新增

        for adm in batch_visits:
            diseases = adm[0]
            procedures = adm[1]

            if self.use_causal_router and self._causal_matrices_loaded:
                causal_sal = self._compute_causal_salience_for_visit(diseases, procedures)
            else:
                causal_sal = 0.0
            self._visit_causal_saliences.append(causal_sal)

            # === Rarity Context（CERaMoE: 4 维 / MoE baseline: 3 维）===
            if self.use_moe and self.use_rarity_router and hasattr(
                    self, 'rarity_context_encoder'):
                visit_weight = adm[3] if len(adm) > 3 else 0.0
                if self.use_causal_router:
                    rarity_features = torch.tensor(
                        [float(visit_weight),
                         float(len(diseases)),
                         float(len(procedures)),
                         float(causal_sal)],
                        dtype=torch.float32, device=self.device
                    ).unsqueeze(0)
                else:
                    rarity_features = torch.tensor(
                        [float(visit_weight),
                         float(len(diseases)),
                         float(len(procedures))],
                        dtype=torch.float32, device=self.device
                    ).unsqueeze(0)
                rarity_context = self.rarity_context_encoder(rarity_features)
                self.transformer_visit.set_rarity_context(rarity_context)

            if len(adm) > 3:
                self._visit_rarity_scores.append(float(adm[3]))
            else:
                self._visit_rarity_scores.append(0.0)

            disease_embedding = self.embeddings[0](
                torch.LongTensor(diseases).unsqueeze(dim=1).to(self.device))
            procedure_embedding = self.embeddings[1](
                torch.LongTensor(procedures).unsqueeze(dim=1).to(self.device))

            cls_embedding = self.special_embeddings(
                self.special_tokens['CLS']).unsqueeze(dim=1)
            sep_embedding = self.special_embeddings(
                self.special_tokens['SEP']).unsqueeze(dim=1)

            disease_embedding = torch.cat((cls_embedding, disease_embedding), dim=0)
            procedure_embedding = torch.cat((sep_embedding, procedure_embedding), dim=0)

            disease_embedding = self.positional_embedding_layer_disease(disease_embedding)
            procedure_embedding = self.positional_embedding_layer_procedure(
                procedure_embedding)

            combined_embedding = torch.cat((disease_embedding, procedure_embedding), dim=0)

            segments = torch.tensor(
                [0] * (len(diseases) + 2) + [1] * len(procedures)).to(self.device)
            segment_embedding = self.segment_embedding(segments).unsqueeze(dim=1)
            input_embedding = combined_embedding + segment_embedding

            visit_representation = self.transformer_visit(input_embedding)[0]
            visit_representation = torch.reshape(visit_representation, (1, 1, -1))
            batch_repr.append(visit_representation)

        if self.use_moe and hasattr(self.transformer_visit, 'clear_rarity_context'):
            self.transformer_visit.clear_rarity_context()

        batch_repr = torch.cat(batch_repr, dim=1).to(self.device)
        batch_repr = batch_repr.squeeze(dim=0)
        return batch_repr

    def patient_encoder_seperate(self, batch_visits):
        device = self.device
        batch_disease_repr, batch_procedure_repr = [], []
        for adm in batch_visits:
            diseases = adm[0]
            procedures = adm[1]
            disease_embedding = self.embeddings[0](
                torch.LongTensor(diseases).unsqueeze(dim=1).to(self.device))
            procedure_embedding = self.embeddings[1](
                torch.LongTensor(procedures).unsqueeze(dim=1).to(self.device))
            cls_embedding_dis = self.special_embeddings(
                self.special_tokens['CLS']).unsqueeze(dim=1)
            cls_embedding_pro = self.special_embeddings(
                self.special_tokens['SEP']).unsqueeze(dim=1)
            disease_embedding = torch.cat((cls_embedding_dis, disease_embedding), dim=0)
            procedure_embedding = torch.cat((cls_embedding_pro, procedure_embedding), dim=0)
            disease_embedding = self.positional_embedding_layer_disease(disease_embedding)
            procedure_embedding = self.positional_embedding_layer_procedure(procedure_embedding)
            disease_representation = self.transformer_disease(disease_embedding)[0]
            procedure_representation = self.transformer_procedure(procedure_embedding)[0]
            disease_representation = disease_representation.mean(dim=0)
            procedure_representation = procedure_representation.mean(dim=0)
            disease_representation = torch.reshape(disease_representation, (1, 1, -1))
            procedure_representation = torch.reshape(procedure_representation, (1, 1, -1))
            batch_disease_repr.append(disease_representation)
            batch_procedure_repr.append(procedure_representation)
        batch_disease_repr = torch.cat(batch_disease_repr, dim=1).to(device)
        batch_procedure_repr = torch.cat(batch_procedure_repr, dim=1).to(device)
        batch_repr = torch.cat((batch_disease_repr, batch_procedure_repr), dim=-1)
        batch_repr = batch_repr.squeeze(dim=0)
        return batch_repr

    def get_moe_aux_loss(self):
        if self.use_moe and hasattr(self.transformer_visit, 'get_aux_loss'):
            return self.transformer_visit.get_aux_loss()
        return torch.tensor(0.0, device=self.device)

    def get_last_visit_routing(self):
        if not self.use_moe or not hasattr(
                self.transformer_visit, 'get_last_visit_routing'):
            return [], None, None
        f, P = self.transformer_visit.get_last_visit_routing()
        return self._visit_rarity_scores, f, P

    def get_moe_diagnostics(self):
        if self.use_moe and hasattr(self.transformer_visit, 'get_diagnostics'):
            return self.transformer_visit.get_diagnostics()
        return {
            'router_entropy': torch.tensor(0.0, device=self.device),
            'expert_usage': torch.zeros(self.num_experts, device=self.device),
            'routed_usage': torch.zeros(self.num_experts, device=self.device),
        }



class CERaMoE(PatientEncoder):
    def __init__(self, args, voc_size, ddi_adj):
        super().__init__(args, voc_size)
        self.tensor_ddi_adj = torch.FloatTensor(ddi_adj).to(self.device)

        self.pretrain_cer_enabled = getattr(args, 'pretrain_cer', False)
        self.cer_loss_weight = getattr(args, 'cer_loss_weight', 1.0)
        self.cer_temperature = getattr(args, 'cer_temperature', 1.0)
        self.cer_relevance_decay = getattr(args, 'cer_relevance_decay', 0.1)

        self.use_causal_bias = getattr(args, 'use_causal_bias', False)

        self.init_weights()

        self.cls_mask = nn.Linear(self.emb_dim, self.voc_size[0] + self.voc_size[1])
        self.cls_final = nn.Linear(self.emb_dim, self.voc_size[2])

        self.cls_cer = nn.Linear(self.emb_dim, self.voc_size[2])

        if self.use_causal_bias:
            self.causal_scale = nn.Parameter(torch.tensor(0.1))
        else:
            self.register_parameter('causal_scale', None)


    def parameter_summary(self):
        total, activated, detail = count_parameters(self, self.args)

        lines = [
            "=" * 64,
            f"  CERaMoE  参数统计",
            "=" * 64,
            f"  {'类别':<22} {'参数量':>12}",
            f"  {'-'*22}   {'-'*12}",
        ]

        label_map = {
            'embedding':      '嵌入层（词+位置+段落）',
            'attention':      '多头注意力 + LayerNorm',
            'router':         'MoE Router（含 Rarity Bias）',
            'rarity_encoder': 'Rarity Context Encoder',
            'shared_expert':  'Shared Expert（始终激活）',
            'routed_experts': f'路由 Expert × {self.num_experts}（稀疏）',
            'ffn_standard':   '标准 FFN（非 MoE）',
            'output_heads':   '输出头（Mask/Final/CER）',
            'causal_module':  '因果偏置 / 因果缩放',
            'other':          '其他',
        }

        for key, label in label_map.items():
            v = detail[key]
            if v > 0:
                lines.append(f"  {label:<24} {format_param_count(v):>10}")

        lines += [
            f"  {'─'*24}   {'─'*10}",
            f"  {'总参数量（Total）':<24} {format_param_count(total):>10}  ({total:,})",
        ]

        if self.use_moe:
            sparsity = 1.0 - activated / total if total > 0 else 0.0
            lines += [
                f"  {'激活参数量':<24} {format_param_count(activated):>10}  ({activated:,})",
                f"  {'MoE 稀疏率':<24} {sparsity * 100:>9.1f}%",
                f"  （top_k={self.top_k}, num_experts={self.num_experts}）",
            ]
        else:
            lines.append(f"  （标准 Transformer，激活参数量 = 总参数量）")

        # CERaMoE 创新点摘要
        lines += [
            "-" * 64,
            "  CERaMoE 状态:",
            f"    [Shared Expert]           {'ON' if self.use_shared_expert else 'OFF'}",
            f"    [Rarity-Aware Router]     {'ON' if self.use_rarity_router else 'OFF'}",
            f"    [Rarity-Monotonic Routing]{'ON' if self.use_mono_constraint else 'OFF'}",
            f"    [CER 预训练]              {'ON' if self.pretrain_cer_enabled else 'OFF'}",
            f"    [Causal-Augmented Router] {'ON' if getattr(self, 'use_causal_router', False) else 'OFF'}",
            f"    [Causal Bias Correction]  {'ON' if self.use_causal_bias else 'OFF'}",
            f"    [因果矩阵已加载]          {'YES' if self._causal_matrices_loaded else 'NO'}",
        ]

        lines.append("=" * 64)

        report = "\n".join(lines)
        logging.info("\n" + report)
        return report


    def _compute_batch_causal_bias(self, batch_visits):
        if not self._causal_matrices_loaded:
            n_m = self.voc_size[2]
            return torch.zeros(len(batch_visits), n_m, device=self.device)

        biases = []
        n_m = self.M_dm_signed.shape[1]
        for adm in batch_visits:
            diag_ids = adm[0]
            proc_ids = adm[1]
            bias = torch.zeros(n_m, device=self.device, dtype=self.M_dm_signed.dtype)
            if len(diag_ids) > 0:
                d_idx = torch.LongTensor(diag_ids).to(self.device)
                bias = bias + self.M_dm_signed[d_idx].sum(0)
            if len(proc_ids) > 0:
                p_idx = torch.LongTensor(proc_ids).to(self.device)
                bias = bias + self.M_pm_signed[p_idx].sum(0)
            # z-score 标准化（visit 内）
            mu = bias.mean()
            sigma = bias.std() + 1e-6
            bias = (bias - mu) / sigma
            biases.append(bias)

        return torch.stack(biases, dim=0)


    def forward_finetune(self, input):
        patient_repr = self.patient_encoder(input)
        result = self.cls_final(patient_repr)

        if self.use_causal_bias and self._causal_matrices_loaded:
            causal_bias = self._compute_batch_causal_bias(input)  # [B, |M|]
            # result = result + self.causal_scale * causal_bias
            result = result + torch.sigmoid(self.causal_scale) * causal_bias


        neg_pred_prob = F.sigmoid(result)
        neg_pred_prob = torch.matmul(neg_pred_prob.t(), neg_pred_prob)
        batch_neg = 0.0005 * neg_pred_prob.mul(self.tensor_ddi_adj).sum()

        moe_aux_loss = self.get_moe_aux_loss()
        return result, batch_neg, moe_aux_loss

    def forward(self, input, mode='fine-tune'):
        assert mode in ['fine-tune', 'pretrain_mask', 'pretrain_cer'], \
            f"Unknown mode: {mode}"

        if mode == 'fine-tune':
            return self.forward_finetune(input)

        elif mode == 'pretrain_mask':
            patient_repr = self.patient_encoder(input)
            result = self.cls_mask(patient_repr)
            moe_aux_loss = self.get_moe_aux_loss()
            return result, moe_aux_loss


        elif mode == 'pretrain_cer':
            # ★ CERaMoE: CER 预训练分支
            patient_repr = self.patient_encoder(input)
            result = self.cls_cer(patient_repr)  # [B, |M|]
            moe_aux_loss = self.get_moe_aux_loss()
            return result, moe_aux_loss

    def init_weights(self):
        initrange = 0.1
        self.embeddings[0].weight.data.uniform_(-initrange, initrange)
        self.embeddings[1].weight.data.uniform_(-initrange, initrange)
        self.segment_embedding.weight.data.uniform_(-initrange, initrange)
        self.special_embeddings.weight.data.uniform_(-initrange, initrange)
