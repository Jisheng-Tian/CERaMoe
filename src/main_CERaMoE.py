import os
import sys
import dill
import logging
import argparse
import numpy as np
from tqdm import tqdm
from copy import deepcopy
from collections import deque
import random
import math

import torch
import torch.nn.functional as F
from torch.optim import AdamW as Optimizer
from torch.utils.tensorboard.writer import SummaryWriter

from models.CERaMoE import CERaMoE
from utils.util import (multi_label_metric, ddi_rate_score, get_n_params,
                         create_log_id, logging_config, get_grouped_metrics,
                         get_model_path, get_pretrained_model_path)


# CERaMoE 因果预训练模块
from causal_pretrain import (
    build_causal_matrices, prepare_cer_batch_targets, cer_loss
)


# ============================================================================
# RoutingBuffer（仅用于 Rarity-Monotonic Routing Constraint）
# ============================================================================

class RoutingBuffer:
    def __init__(self, buffer_size=64, num_experts=4):
        self.buffer_size = buffer_size
        self.num_experts = num_experts
        self.rarity_scores = deque(maxlen=buffer_size)
        self.P_list = deque(maxlen=buffer_size)

    def add(self, rarity_score, P_detached):
        self.rarity_scores.append(rarity_score)
        self.P_list.append(P_detached)

    def clear(self):
        self.rarity_scores.clear()
        self.P_list.clear()

    def __len__(self):
        return len(self.rarity_scores)


def compute_buffered_mono_loss(current_rarity, current_P,
                                routing_buffer, num_experts, margin=0.1,
                                min_buffer=8):
    if len(routing_buffer) < min_buffer:
        return torch.tensor(0.0, device=current_P.device)

    device = current_P.device
    buf_rarity = torch.tensor(list(routing_buffer.rarity_scores),
                              dtype=torch.float32, device=device)
    buf_P = torch.stack(list(routing_buffer.P_list))

    rarity_weights = F.softmax(buf_rarity, dim=0)
    rare_expert_pref = (rarity_weights.unsqueeze(1) * buf_P).sum(0)

    k_rare = max(1, num_experts // 2)
    rare_mask = torch.zeros(num_experts, device=device)
    rare_mask[rare_expert_pref.topk(k_rare).indices] = 1.0

    current_rare_score = (current_P * rare_mask).sum()
    buf_rare_scores = (buf_P * rare_mask.unsqueeze(0)).sum(dim=1)

    rarity_diff = current_rarity - buf_rarity
    score_diff = current_rare_score - buf_rare_scores

    valid = rarity_diff.abs() > 1e-6
    if valid.sum() == 0:
        return torch.tensor(0.0, device=device)

    signed_score_diff = rarity_diff.sign() * score_diff
    hinge = torch.clamp(margin - signed_score_diff, min=0)
    mono_loss = (hinge * valid.float()).sum() / (valid.sum().float() + 1e-9)
    return mono_loss


# ============================================================================
# Argparse
# ============================================================================

def get_args():
    parser = argparse.ArgumentParser()

    # === 基础 ===
    parser.add_argument('-n', '--note', type=str, default='', help="User notes")
    parser.add_argument('--model_name', type=str, default='CERaMoE',
                        help="model name")
    parser.add_argument('--dataset', type=str, default='mimic-iii', help='dataset')
    parser.add_argument('--early_stop', type=int, default=10)
    parser.add_argument('-t', '--test', action='store_true', help="test mode")
    parser.add_argument('-l', '--log_dir_prefix', type=str, default=None)
    parser.add_argument('-p', '--pretrain_prefix', type=str, default=None)
    parser.add_argument('--cuda', type=int, default=0)

    # === 预训练任务（推荐 mask + CER 组合）===
    parser.add_argument('-mask', '--pretrain_mask', action='store_true')
    parser.add_argument('-cer', '--pretrain_cer', action='store_true',)
    parser.add_argument('--pretrain_epochs', type=int, default=30)
    parser.add_argument('--mask_prob', type=float, default=0)

    # === 模型架构 ===
    parser.add_argument('--embed_dim', type=int, default=512)
    parser.add_argument('--encoder_layers', type=int, default=3)
    parser.add_argument('--nhead', type=int, default=4)
    parser.add_argument('--batch_size', type=int, default=1)

    # === 训练超参 ===
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--dropout', type=float, default=0.3)
    parser.add_argument('--weight_decay', type=float, default=0.1)
    parser.add_argument('--weight_multi', type=float, default=0.03)
    parser.add_argument('--weight_ddi', type=float, default=0.85)

    # === 消融 ===
    parser.add_argument('-s', '--patient_seperate', action='store_true')
    parser.add_argument('-e', '--seg_rel_emb', action='store_false', default=True)

    # === MoE===
    parser.add_argument('--use_moe', action='store_true', default=False)
    parser.add_argument('--num_experts', type=int, default=4)
    parser.add_argument('--top_k', type=int, default=2)
    parser.add_argument('--moe_loss_weight', type=float, default=0.01)
    parser.add_argument('--use_shared_expert', action='store_true', default=False,)
    parser.add_argument('--use_rarity_router', action='store_true', default=False,)
    parser.add_argument('--use_mono_constraint', action='store_true', default=False,)
    parser.add_argument('--mono_loss_weight', type=float, default=0.005)
    parser.add_argument('--mono_margin', type=float, default=0.1)
    parser.add_argument('--routing_buffer_size', type=int, default=64)


    # === CERaMoE===
    parser.add_argument('--cer_loss_weight', type=float, default=1.0,)
    parser.add_argument('--cer_temperature', type=float, default=1.0,help='CER 软目标分布的温度（小→尖锐，大→平滑）')
    parser.add_argument('--cer_relevance_decay', type=float, default=0.1,)
    parser.add_argument('--causal_pmi_method', type=str, default='ipw_cpmi',
                        choices=['cpmi', 'ipw_cpmi'],
                        help='因果矩阵估计方法（推荐 ipw_cpmi）')
    parser.add_argument('--causal_min_co_count', type=int, default=1)

    parser.add_argument('--use_causal_router', action='store_true', default=False,
                        help='【创新点2】启用 Causal-Augmented Rarity Router')

    parser.add_argument('--use_causal_bias', action='store_true', default=False,
                        help='【创新点3】启用 fine-tune 阶段的因果偏置校准')

    parser.add_argument('--cer_early_stop', type=int, default=5,
                        help='CER 预训练早停 patience')
    parser.add_argument('--cer_ema_window', type=int, default=3,
                        help='CER 验证 KL 的滑动平滑窗口')

    # === 工具 ===
    parser.add_argument('--count_params_only', action='store_true', default=False)

    args = parser.parse_args()
    return args


# ============================================================================
# 评估函数
# ============================================================================

@torch.no_grad()
def evaluator(args, model, data_val, voc_size, epoch, ddi_adj_path, device,
              rec_results_path=''):
    model.eval()
    ja, prauc, avg_p, avg_r, avg_f1 = [[] for _ in range(5)]
    visit_weights = []
    smm_record = []
    med_cnt, visit_cnt = 0, 0
    recommended_drugs = set()
    loss_val_bce = loss_val_milti = loss_val_ddi = loss_val_sum = 0
    len_val = len(data_val)

    rec_results = []
    all_pred_prob = []
    ja_visit = [[] for _ in range(5)]

    for patient in tqdm(data_val, ncols=60, total=len(data_val), desc="Evaluation"):
        y_gt, y_pred, y_pred_prob, y_pred_label = [], [], [], []
        visit_weights_patient = []
        all_diseases, all_procedures, all_medications = [], [], []

        for adm in patient:
            if args.test:
                all_diseases.append(adm[0])
                all_procedures.append(adm[1])
                all_medications.append(adm[2])
            y_gt_tmp = np.zeros(voc_size[2])
            y_gt_tmp[adm[2]] = 1
            y_gt.append(y_gt_tmp)
            visit_weights.append(adm[3])
            visit_weights_patient.append(adm[3])

        results, loss_ddi, _moe_aux_loss = model(patient)

        loss_bce, loss_multi = loss_func(voc_size, patient, results, device)
        loss_val_bce += loss_bce.item() / len_val
        loss_val_milti += loss_multi.item() / len_val
        loss_val_ddi += loss_ddi.item() / len_val
        y_pred_prob = F.sigmoid(results).detach().cpu().numpy()
        for target_output in y_pred_prob:
            y_pred_tmp = target_output.copy()
            all_pred_prob.append(list(y_pred_tmp))
            y_pred_tmp[y_pred_tmp >= 0.5] = 1
            y_pred_tmp[y_pred_tmp < 0.5] = 0
            y_pred.append(y_pred_tmp)
            y_pred_label_tmp = np.where(y_pred_tmp == 1)[0]
            recommended_drugs = set(y_pred_label_tmp) | recommended_drugs
            y_pred_label.append(sorted(y_pred_label_tmp))
            visit_cnt += 1
            med_cnt += len(y_pred_label_tmp)

        smm_record.append(y_pred_label)
        adm_ja, adm_prauc, adm_avg_p, adm_avg_r, adm_avg_f1 = multi_label_metric(
            np.array(y_gt), np.array(y_pred), np.array(y_pred_prob))

        if args.test:
            if len(patient) < 5:
                ja_visit[len(patient) - 1].append(adm_ja)
            else:
                ja_visit[4].append(adm_ja)
            records = [all_diseases, all_procedures, all_medications,
                       y_pred_label, visit_weights_patient, [adm_ja]]
            rec_results.append(records)

        ja.append(adm_ja)
        prauc.append(adm_prauc)
        avg_p.append(adm_avg_p)
        avg_r.append(adm_avg_r)
        avg_f1.append(adm_avg_f1)

    if args.test:
        os.makedirs(rec_results_path, exist_ok=True)
        rec_results_file = rec_results_path + '/' + 'rec_results.pkl'
        dill.dump(rec_results, open(rec_results_file, 'wb'))
        ja_result_file = rec_results_path + '/' + 'ja_result.pkl'
        dill.dump(ja_visit, open(ja_result_file, 'wb'))
        for i in range(5):
            logging.info(str(i + 1) +
                         f'visit\t mean: {np.mean(ja_visit[i]):.4},'
                         f' std: {np.std(ja_visit[i]):.4}')

    ddi_rate = ddi_rate_score(smm_record, path=ddi_adj_path)
    get_grouped_metrics(ja, visit_weights)
    logging.info(
        f'Epoch {epoch:03d}, Jaccard: {np.mean(ja):.4}, DDI Rate: {ddi_rate:.4}, '
        f'PRAUC: {np.mean(prauc):.4}, AVG_F1: {np.mean(avg_f1):.4}, '
        f'AVG_PRC: {np.mean(avg_p):.4f}, AVG_RECALL: {np.mean(avg_r):.4f}, '
        f'AVG_MED: {med_cnt / visit_cnt:.4}')

    loss_val_sum = ((1 - args.weight_multi) * loss_val_bce
                    + args.weight_multi * loss_val_milti
                    + args.weight_ddi * loss_val_ddi)

    return (ddi_rate, np.mean(ja), np.mean(prauc), np.mean(avg_f1),
            med_cnt / visit_cnt,
            loss_val_bce, loss_val_milti, loss_val_ddi, loss_val_sum)


@torch.no_grad()
def evaluator_mask(model, data_val, voc_size, epoch, device, mode='pretrain'):
    model.eval()
    loss_val = 0
    dis_ja_list, dis_prauc_list, dis_p_list, dis_r_list, dis_f1_list = [[] for _ in range(5)]
    pro_ja_list, pro_prauc_list, pro_p_list, pro_r_list, pro_f1_list = [[] for _ in range(5)]
    dis_cnt, pro_cnt, visit_cnt = 0, 0, 0
    len_val = len(data_val)

    for batch in tqdm(data_val, ncols=60, desc=mode, total=len_val):
        batch_size = len(batch)
        result, _moe_aux_loss = model(batch, mode)

        dis_gt = np.zeros((batch_size, voc_size[0]))
        pro_gt = np.zeros((batch_size, voc_size[1]))
        for i in range(batch_size):
            dis_gt[i, batch[i][0]] = 1
            pro_gt[i, batch[i][1]] = 1
        target = np.concatenate((dis_gt, pro_gt), axis=1)
        loss = F.binary_cross_entropy_with_logits(
            result, torch.tensor(target, device=device))
        loss_val += loss.item()

        dis_logit = result[:, :voc_size[0]]
        pro_logit = result[:, voc_size[0]:]
        dis_pred_prob = F.sigmoid(dis_logit).cpu().numpy()
        pro_pred_prob = F.sigmoid(pro_logit).cpu().numpy()

        visit_cnt += batch_size
        dis_pred, dis_pred_label = [], []
        pro_pred, pro_pred_label = [], []
        for i in range(batch_size):
            dis_pred_temp = dis_pred_prob[i].copy()
            dis_pred_temp[dis_pred_temp >= 0.5] = 1
            dis_pred_temp[dis_pred_temp < 0.5] = 0
            dis_pred.append(dis_pred_temp)
            dis_pred_label.append(sorted(np.where(dis_pred_temp == 1)[0]))
            dis_cnt += int(dis_pred_temp.sum())

            pro_pred_temp = pro_pred_prob[i].copy()
            pro_pred_temp[pro_pred_temp >= 0.5] = 1
            pro_pred_temp[pro_pred_temp < 0.5] = 0
            pro_pred.append(pro_pred_temp)
            pro_pred_label.append(sorted(np.where(pro_pred_temp == 1)[0]))
            pro_cnt += int(pro_pred_temp.sum())

        dis_ja, dis_prauc, dis_avg_p, dis_avg_r, dis_avg_f1 = multi_label_metric(
            np.array(dis_gt), np.array(dis_pred), np.array(dis_pred_prob))
        pro_ja, pro_prauc, pro_avg_p, pro_avg_r, pro_avg_f1 = multi_label_metric(
            np.array(pro_gt), np.array(pro_pred), np.array(pro_pred_prob))

        dis_ja_list.append(dis_ja); dis_prauc_list.append(dis_prauc)
        dis_p_list.append(dis_avg_p); dis_r_list.append(dis_avg_r); dis_f1_list.append(dis_avg_f1)
        pro_ja_list.append(pro_ja); pro_prauc_list.append(pro_prauc)
        pro_p_list.append(pro_avg_p); pro_r_list.append(pro_avg_r); pro_f1_list.append(pro_avg_f1)

    avg_ja = (np.mean(dis_ja_list) + np.mean(pro_ja_list)) / 2
    avg_prauc = (np.mean(dis_prauc_list) + np.mean(pro_prauc_list)) / 2
    avg_p = (np.mean(dis_p_list) + np.mean(pro_p_list)) / 2
    avg_r = (np.mean(dis_r_list) + np.mean(pro_r_list)) / 2
    avg_f1 = (np.mean(dis_f1_list) + np.mean(pro_f1_list)) / 2
    avg_cnt = (dis_cnt / visit_cnt + pro_cnt / visit_cnt) / 2

    logging.info(f'Epoch {epoch:03d}   Jaccard: {avg_ja:.4f}, PRAUC: {avg_prauc:.4f}, '
                 f'F1: {avg_f1:.4f}, AVG_CNT: {avg_cnt:.4f}')
    return loss_val / len_val, avg_ja, avg_prauc, avg_p, avg_r, avg_f1, avg_cnt



@torch.no_grad()
def evaluator_cer(model, data_val, voc_size, M_dm_pos, M_pm_pos,
                   epoch, device, args):
    model.eval()
    loss_val = 0
    n_batches = len(data_val)

    top5_hit = 0
    top10_hit = 0
    n_visits = 0

    for batch in tqdm(data_val, ncols=60, desc='pretrain_cer_eval', total=n_batches):
        target_dist = prepare_cer_batch_targets(
            batch, M_dm_pos, M_pm_pos,
            relevance_decay=args.cer_relevance_decay,
            temperature=args.cer_temperature,
            device=device,
        )
        result, _moe_aux_loss = model(batch, mode='pretrain_cer')
        loss = cer_loss(result, target_dist, temperature=args.cer_temperature)
        loss_val += loss.item()

        pred_probs = F.softmax(result, dim=-1)
        for i, visit in enumerate(batch):
            gt_meds = set(visit[2]) if len(visit) > 2 else set()
            if len(gt_meds) == 0:
                continue
            top5 = set(pred_probs[i].topk(5).indices.cpu().tolist())
            top10 = set(pred_probs[i].topk(10).indices.cpu().tolist())
            top5_hit += len(top5 & gt_meds) / max(len(gt_meds), 1)
            top10_hit += len(top10 & gt_meds) / max(len(gt_meds), 1)
            n_visits += 1

    avg_loss = loss_val / max(n_batches, 1)
    avg_top5 = top5_hit / max(n_visits, 1)
    avg_top10 = top10_hit / max(n_visits, 1)

    logging.info(f'[CER] Epoch {epoch:03d}   val_KL: {avg_loss:.4f}, '
                 f'top5_recall: {avg_top5:.4f}, top10_recall: {avg_top10:.4f}')
    return avg_loss, avg_top5, avg_top10


# ============================================================================
# 工具函数
# ============================================================================

def get_moe_diagnostics(model, num_experts):
    if not hasattr(model, 'get_moe_diagnostics'):
        return 0.0, np.zeros(num_experts), np.zeros(num_experts)
    diag = model.get_moe_diagnostics()
    router_entropy = float(diag['router_entropy'].detach().cpu().item())
    expert_usage = diag['expert_usage'].detach().cpu().numpy()
    routed_usage = diag.get('routed_usage', diag['expert_usage']).detach().cpu().numpy()
    return router_entropy, expert_usage, routed_usage


def get_moe_grad_norms(model):
    router_sq, experts_sq, shared_sq, rarity_sq = 0.0, 0.0, 0.0, 0.0
    for name, param in model.named_parameters():
        if param.grad is None:
            continue
        grad_sq = float(param.grad.detach().float().pow(2).sum().item())
        if '.moe_ffn.router.' in name:
            router_sq += grad_sq
        elif '.moe_ffn.experts.' in name:
            experts_sq += grad_sq
        elif '.moe_ffn.shared_expert.' in name:
            shared_sq += grad_sq
        elif '.moe_ffn.rarity_bias.' in name or '.moe_ffn.rarity_scale' in name:
            rarity_sq += grad_sq
        elif '.rarity_context_encoder.' in name:
            rarity_sq += grad_sq
    return (math.sqrt(router_sq), math.sqrt(experts_sq),
            math.sqrt(shared_sq), math.sqrt(rarity_sq))


def random_mask_word(seq, vocab, mask_prob=0.15):
    mask_idx = vocab.word2idx['[MASK]']
    for i, _ in enumerate(seq):
        prob = random.random()
        if prob < mask_prob:
            prob /= mask_prob
            if prob < 0.8:
                seq[i] = mask_idx
            elif prob < 0.9:
                seq[i] = random.choice(list(vocab.word2idx.items()))[1]
    return seq


def mask_batch_data(batch_data, diag_voc, pro_voc, mask_prob):
    masked_data = []
    for visit in batch_data:
        diag = random_mask_word(list(visit[0]), diag_voc, mask_prob)
        pro = random_mask_word(list(visit[1]), pro_voc, mask_prob)
        masked_visit = [diag, pro] + list(visit[2:])
        masked_data.append(masked_visit)
    return masked_data



# ============================================================================
# Main
# ============================================================================

def main(args):
    # 日志配置
    if args.test:
        args.note = 'test of ' + args.log_dir_prefix
    log_directory_path = os.path.join('log', args.dataset, args.model_name)
    log_save_id = create_log_id(log_directory_path)
    save_dir = os.path.join(log_directory_path,
                            'log' + str(log_save_id) + '_' + args.note)
    logging_config(folder=save_dir, name='log{:d}'.format(log_save_id),
                   note=args.note, no_console=False)
    logging.info("当前进程的PID为: %s", os.getpid())
    logging.info(args)

    # 创新点配置打印
    logging.info("=" * 70)
    logging.info("CERaMoE  完整创新点配置")
    logging.info("=" * 70)
    logging.info("【MoE 基础】")
    logging.info(f"  use_moe:                  {args.use_moe}")
    logging.info(f"  num_experts / top_k:      {args.num_experts} / {args.top_k}")
    logging.info("【内化 MoE 创新点】")
    logging.info(f"  [1] Shared Expert:        {args.use_shared_expert}")
    logging.info(f"  [2] Rarity-Aware Router:  {args.use_rarity_router}")
    logging.info(f"  [3] Mono Constraint:      {args.use_mono_constraint} "
                 f"(weight={args.mono_loss_weight}, margin={args.mono_margin})")
    logging.info("【CERaMoE 因果增强模块】")
    logging.info(f"  [1]  CER Pretraining:     {args.pretrain_cer} "
                 f"(weight={args.cer_loss_weight}, T={args.cer_temperature})")
    logging.info(f"  [2]  Causal Router:       {args.use_causal_router}")
    logging.info(f"  [3]  Causal Bias (FT):    {args.use_causal_bias}")
    logging.info(f"  PMI 估计方法:             {args.causal_pmi_method}")
    logging.info("=" * 70)

    # 加载数据
    data_path = f'../data/output/{args.dataset}/records_final.pkl'
    voc_path = f'../data/output/{args.dataset}/voc_final.pkl'
    ddi_adj_path = f'../data/output/{args.dataset}/ddi_A_final.pkl'

    device = torch.device('cuda:{}'.format(args.cuda))

    data = dill.load(open(data_path, 'rb'))
    voc = dill.load(open(voc_path, 'rb'))
    diag_voc, pro_voc, med_voc = voc['diag_voc'], voc['pro_voc'], voc['med_voc']

    def add_word(word, voc):
        voc.word2idx[word] = len(voc.word2idx)
        voc.idx2word[len(voc.idx2word)] = word
        return voc

    add_word('[MASK]', diag_voc)
    add_word('[MASK]', pro_voc)

    ddi_adj = dill.load(open(ddi_adj_path, 'rb'))

    split_point = int(len(data) * 2 / 3)
    data_train_raw = data[:split_point]
    val_len = int(len(data[split_point:]) / 2)
    data_val_raw = data[split_point:split_point + val_len]
    data_pretrain_patients = data[:split_point + val_len]
    data_test_raw = data[split_point + val_len:]

    data_pretrain = [visit for patient in data_pretrain_patients for visit in patient]
    data_train = [visit for patient in data_train_raw for visit in patient]
    data_val = [visit for patient in data_val_raw for visit in patient]
    data_test = [visit for patient in data_test_raw for visit in patient]

    def batchify(data, batch_size):
        return [data[i:min(i + batch_size, len(data))]
                for i in range(0, len(data), batch_size)]

    data_train = batchify(data_train, args.batch_size)
    data_val = batchify(data_val, args.batch_size)
    data_test = batchify(data_test, args.batch_size)

    voc_size = (len(diag_voc.idx2word), len(pro_voc.idx2word), len(med_voc.idx2word))

    # ========================================================================
    # CERaMoE 关键步骤：构建因果效应矩阵
    # ========================================================================
    causal_needed = (args.pretrain_cer
                     or args.use_causal_router or args.use_causal_bias)

    M_dm_pos = M_dm_signed = M_pm_pos = M_pm_signed = None

    if causal_needed:
        logging.info("[CERaMoE] 正在构建因果效应矩阵...")
        M_dm_pos, M_dm_signed, M_pm_pos, M_pm_signed, causal_stats = \
            build_causal_matrices(
                data_pretrain, voc_size,
                method=args.causal_pmi_method,
                min_co_count=args.causal_min_co_count,
            )
    else:
        logging.info("[CERaMoE] 未启用任何因果模块，跳过因果矩阵构建")

    # 模型初始化
    model = CERaMoE(args, voc_size, ddi_adj)

    # ★ 注入因果矩阵
    if causal_needed:
        model.set_causal_matrices(M_dm_pos, M_dm_signed, M_pm_pos, M_pm_signed)

    logging.info(model)
    model.parameter_summary()

    if args.count_params_only:
        logging.info("[CERaMoE] --count_params_only 模式，参数统计完成，正常退出。")
        return

    # 测试模式
    if args.test:
        model_path = get_model_path(log_directory_path, args.log_dir_prefix)
        load_checkpoint_compat(model, model_path, device=device)
        model.to(device=device)
        logging.info("load model from %s", model_path)
        rec_results_path = save_dir + '/' + 'rec_results'
        evaluator(args, model, data_test, voc_size, 0, ddi_adj_path, device,
                  rec_results_path)
        return
    else:
        writer = SummaryWriter(save_dir)

    # === 准备 CER 用的因果矩阵 tensor 引用 ===
    M_dm_pos_t = model.M_dm_pos if causal_needed else None
    M_pm_pos_t = model.M_pm_pos if causal_needed else None


    model.to(device=device)
    logging.info(f'n_parameters: {get_n_params(model)}')
    optimizer = Optimizer(model.parameters(), lr=args.lr,
                          weight_decay=args.weight_decay)
    logging.info(f'Optimizer: {optimizer}')

    # ========================================================================
    # 预训练阶段
    # ========================================================================

    # 推荐顺序：Mask → CER
    if args.pretrain_mask:
        logging.info("\n" + "=" * 50 + "\n  [Stage 1] Mask 预训练\n" + "=" * 50)
        main_mask(args, model, optimizer, writer, diag_voc, pro_voc,
                  data_train, data_val, voc_size, device, save_dir, log_save_id)

    if args.pretrain_cer:
        logging.info("\n" + "=" * 50 + "\n  [Stage 2] CER 预训练\n" + "=" * 50)
        main_cer(args, model, optimizer, writer, data_train, data_val,
                 voc_size, M_dm_pos_t, M_pm_pos_t, device, save_dir, log_save_id)


    # 加载已有的 pretrained model（如指定）
    if not (args.pretrain_mask or args.pretrain_cer) \
            and args.pretrain_prefix is not None:
        pretrained_model_path = get_pretrained_model_path(
            log_directory_path, args.pretrain_prefix)
        load_pretrained_model(model, pretrained_model_path)

    # ========================================================================
    # ★ Fine-tune 阶段
    # ========================================================================

    logging.info("\n" + "=" * 50 + "\n  [Stage 3] Fine-tune\n" + "=" * 50)

    EPOCH = 60
    best_epoch, best_ja = 0, 0
    best_ddi_rate = 0.0
    best_model_state = None

    routing_buffer = RoutingBuffer(
        buffer_size=args.routing_buffer_size,
        num_experts=args.num_experts
    )

    for epoch in range(EPOCH):
        epoch += 1
        print(f'\nepoch {epoch} ----- model={args.model_name}, logger={log_save_id}')

        model.train()
        loss_train_bce = loss_train_multi = loss_train_ddi = loss_train_all = 0
        loss_train_moe = loss_train_mono = 0
        moe_router_entropy = moe_router_grad_norm = 0
        moe_expert_grad_norm = moe_shared_grad_norm = moe_rarity_grad_norm = 0
        moe_expert_usage = np.zeros(args.num_experts, dtype=np.float64)
        moe_routed_usage = np.zeros(args.num_experts, dtype=np.float64)

        # 监控因果偏置缩放因子
        causal_scale_val = (model.causal_scale.item()
                            if model.causal_scale is not None else 0.0)

        routing_buffer.clear()

        for step, patient in tqdm(enumerate(data_train), ncols=60,
                                  desc="finetune", total=len(data_train)):
            result, loss_ddi, moe_aux_loss = model(patient)
            router_entropy, expert_usage, routed_usage = get_moe_diagnostics(
                model, args.num_experts)

            loss_bce, loss_multi = loss_func(voc_size, patient, result, device)

            mono_loss = torch.tensor(0.0, device=device)
            cur_P = None
            cur_rarity = 0.0

            if args.use_moe and args.use_mono_constraint:
                rarity_scores, _, cur_P = model.get_last_visit_routing()
                if cur_P is not None and len(rarity_scores) > 0:
                    cur_rarity = rarity_scores[0]
                    mono_loss = compute_buffered_mono_loss(
                        cur_rarity, cur_P, routing_buffer,
                        args.num_experts, args.mono_margin)

            loss_all = ((1 - args.weight_multi) * loss_bce
                        + args.weight_multi * loss_multi
                        + args.weight_ddi * loss_ddi
                        + args.moe_loss_weight * moe_aux_loss
                        + args.mono_loss_weight * mono_loss)

            loss_final = loss_all / args.batch_size
            loss_final.backward()

            (router_grad_norm, expert_grad_norm,
             shared_grad_norm, rarity_grad_norm) = get_moe_grad_norms(model)

            optimizer.step()
            optimizer.zero_grad()

            if args.use_moe and args.use_mono_constraint and cur_P is not None:
                routing_buffer.add(cur_rarity, cur_P.detach())

            n = len(data_train)
            loss_train_bce += loss_bce.item() / n
            loss_train_multi += loss_multi.item() / n
            loss_train_ddi += loss_ddi.item() / n
            loss_train_all += loss_all.item() / n
            loss_train_moe += moe_aux_loss.item() / n
            loss_train_mono += mono_loss.item() / n
            moe_router_entropy += router_entropy / n
            moe_router_grad_norm += router_grad_norm / n
            moe_expert_grad_norm += expert_grad_norm / n
            moe_shared_grad_norm += shared_grad_norm / n
            moe_rarity_grad_norm += rarity_grad_norm / n
            moe_expert_usage += expert_usage / n
            moe_routed_usage += routed_usage / n

        ddi_rate, ja, prauc, avg_f1, avg_med, \
        loss_val_bce, loss_val_multi, loss_val_ddi, loss_val_all = evaluator(
            args, model, data_val, voc_size, epoch, ddi_adj_path, device)

        moe_top1_msg = ", ".join(
            [f"E{i}:{u:.3f}" for i, u in enumerate(moe_expert_usage.tolist())])
        moe_routed_msg = ", ".join(
            [f"E{i}:{u:.3f}" for i, u in enumerate(moe_routed_usage.tolist())])
        logging.info(
            f'loss_train_all:{loss_train_all:.4f}, bce:{loss_train_bce:.4f}, '
            f'multi:{loss_train_multi:.4f}, ddi:{loss_train_ddi:.4f}, '
            f'moe_aux:{loss_train_moe:.6f}, mono:{loss_train_mono:.6f}\n'
            f'                loss_val_all:{loss_val_all:.4f}, bce:{loss_val_bce:.4f}, '
            f'multi:{loss_val_multi:.4f}, ddi:{loss_val_ddi:.4f}\n'
            f'                router_entropy:{moe_router_entropy:.6f}, '
            f'router_grad:{moe_router_grad_norm:.6f}, '
            f'expert_grad:{moe_expert_grad_norm:.6f}, '
            f'shared_grad:{moe_shared_grad_norm:.6f}, '
            f'rarity_grad:{moe_rarity_grad_norm:.6f}\n'
            f'                top1:[{moe_top1_msg}], routed:[{moe_routed_msg}]\n'
            f'                causal_scale: {causal_scale_val:.4f}')

        tensorboard_write(writer, ja, prauc, ddi_rate, avg_med, epoch,
                          loss_train_bce, loss_train_multi, loss_train_ddi,
                          loss_train_all, loss_val_bce, loss_val_multi,
                          loss_val_ddi, loss_val_all, loss_train_moe, loss_train_mono,
                          moe_router_entropy, moe_router_grad_norm,
                          moe_expert_grad_norm, moe_shared_grad_norm,
                          moe_rarity_grad_norm, moe_expert_usage, moe_routed_usage,
                          causal_scale_val)

        if epoch != 0 and best_ja < ja:
            best_epoch = epoch
            best_ja, best_ddi_rate = ja, ddi_rate
            best_model_state = deepcopy(model.state_dict())
        logging.info(f'best_epoch: {best_epoch}, best_ja: {best_ja:.4f}\n')

        if epoch - best_epoch > args.early_stop:
            break

    logging.info('Train finished')
    if best_model_state is not None:
        torch.save(best_model_state, open(os.path.join(
            save_dir, 'Epoch_{}_JA_{:.4}_DDI_{:.4}.model'.format(
                best_epoch, best_ja, best_ddi_rate)), 'wb'))


# ============================================================================
# CER 预训练循环
# ============================================================================

def main_cer(args, model, optimizer, writer, data_train, data_val,
             voc_size, M_dm_pos_t, M_pm_pos_t, device, save_dir, log_save_id):
    """
    CER (Causal Effect Reconstruction) 预训练循环。

    每个 step:
      1. 取 batch 中所有 visit
      2. 用因果矩阵计算 batch 软目标分布（无梯度）
      3. 模型前向 cls_cer → logits
      4. KL(target || softmax(logits / T))
      5. + MoE aux loss
      6. 反向传播

    评估 + early stopping:
      - 每 epoch 在 data_val 上计算 KL + top-K recall
      - 监控 val_KL 平滑值，若多个 epoch 不降则早停
    """
    EPOCH = args.pretrain_epochs
    best_epoch, best_val_kl = 0, float('inf')
    best_state = None
    val_kl_history = []

    for epoch in range(1, EPOCH + 1):
        print(f'\nepoch {epoch} ----- model={args.model_name}, '
              f'logger={log_save_id}, mode=pretrain_cer')

        model.train()
        train_kl = train_moe = train_total = 0.0
        moe_router_entropy = moe_router_grad_norm = moe_expert_grad_norm = 0.0
        moe_expert_usage = np.zeros(args.num_experts, dtype=np.float64)

        n = len(data_train)

        for batch in tqdm(data_train, ncols=60, desc="pretrain_cer", total=n):
            # 1. 构造软目标（无梯度）
            target_dist = prepare_cer_batch_targets(
                batch, M_dm_pos_t, M_pm_pos_t,
                relevance_decay=args.cer_relevance_decay,
                temperature=args.cer_temperature,
                device=device,
            )

            # 2. 前向
            result, moe_aux_loss = model(batch, mode='pretrain_cer')

            router_entropy, expert_usage, _ = get_moe_diagnostics(
                model, args.num_experts)

            # 3. 损失
            loss_kl = cer_loss(result, target_dist, temperature=args.cer_temperature)
            loss_total = (args.cer_loss_weight * loss_kl
                         + args.moe_loss_weight * moe_aux_loss)

            loss_total.backward()
            router_grad_norm, expert_grad_norm, _, _ = get_moe_grad_norms(model)
            optimizer.step()
            optimizer.zero_grad()

            train_kl += loss_kl.item() / n
            train_moe += moe_aux_loss.item() / n
            train_total += loss_total.item() / n
            moe_router_entropy += router_entropy / n
            moe_router_grad_norm += router_grad_norm / n
            moe_expert_grad_norm += expert_grad_norm / n
            moe_expert_usage += expert_usage / n

        # 评估
        val_kl, val_top5, val_top10 = evaluator_cer(
            model, data_val, voc_size, M_dm_pos_t, M_pm_pos_t,
            epoch, device, args)

        val_kl_history.append(val_kl)
        window = max(1, getattr(args, 'cer_ema_window', 3))
        smoothed = float(np.mean(val_kl_history[-window:]))

        improved = ""
        if smoothed < best_val_kl:
            best_val_kl = smoothed
            best_epoch = epoch
            best_state = deepcopy(model.state_dict())
            improved = " ★ new best"

        moe_msg = ", ".join(
            [f"E{i}:{u:.3f}" for i, u in enumerate(moe_expert_usage.tolist())])

        logging.info(
            f'[CER] Epoch {epoch:03d}   '
            f'train_kl: {train_kl:.4f}, train_moe: {train_moe:.4f}, '
            f'train_total: {train_total:.4f}, val_kl: {val_kl:.4f}, '
            f'val_kl_smooth: {smoothed:.4f} (best: {best_val_kl:.4f} @ '
            f'ep{best_epoch}){improved}, '
            f'val_top5: {val_top5:.4f}, val_top10: {val_top10:.4f}, '
            f'router_entropy: {moe_router_entropy:.4f}, '
            f'router_grad: {moe_router_grad_norm:.4f}, '
            f'expert_grad: {moe_expert_grad_norm:.4f}, '
            f'expert_usage: [{moe_msg}]')

        # TensorBoard
        writer.add_scalar('CER/train_kl', train_kl, epoch)
        writer.add_scalar('CER/val_kl', val_kl, epoch)
        writer.add_scalar('CER/val_kl_smooth', smoothed, epoch)
        writer.add_scalar('CER/val_top5_recall', val_top5, epoch)
        writer.add_scalar('CER/val_top10_recall', val_top10, epoch)
        writer.add_scalar('CER/train_moe_aux', train_moe, epoch)
        writer.add_scalar('CER/router_entropy', moe_router_entropy, epoch)
        writer.add_scalar('CER/router_grad_norm', moe_router_grad_norm, epoch)
        writer.add_scalar('CER/expert_grad_norm', moe_expert_grad_norm, epoch)
        for i, u in enumerate(moe_expert_usage.tolist()):
            writer.add_scalar(f'CER/expert_usage/E{i}', u, epoch)

        # Early stopping
        if epoch - best_epoch >= args.cer_early_stop:
            logging.info(f'[CER] Early stopping at epoch {epoch}. '
                         f'Best smoothed val_kl: {best_val_kl:.4f} @ ep{best_epoch}.')
            break

    if best_state is not None:
        model.load_state_dict(best_state)
        logging.info(f'[CER] Restored best model from epoch {best_epoch} '
                     f'(smoothed val_kl={best_val_kl:.4f})')

    save_pretrained_model(model, save_dir, suffix='cer')


# ============================================================================
# Mask 预训练循环（baseline 兼容）
# ============================================================================

def main_mask(args, model, optimizer, writer, diag_voc, pro_voc,
              data_train, data_val, voc_size, device, save_dir, log_save_id):
    best_epoch_mask, best_ja_mask = 0, 0
    EPOCH = args.pretrain_epochs

    for epoch in range(EPOCH):
        epoch += 1
        print(f'\nepoch {epoch} ----- mode=pretrain_mask')
        model.train()
        loss_train = 0
        moe_router_entropy = moe_router_grad_norm = moe_expert_grad_norm = 0
        moe_expert_usage = np.zeros(args.num_experts, dtype=np.float64)

        for batch in tqdm(data_train, ncols=60, desc="pretrain_mask",
                          total=len(data_train)):
            batch_size = len(batch)
            masked_batch = (mask_batch_data(batch, diag_voc, pro_voc, args.mask_prob)
                            if args.mask_prob > 0 else batch)

            result, moe_aux_loss = model(masked_batch, mode='pretrain_mask')
            router_entropy, expert_usage, _ = get_moe_diagnostics(model, args.num_experts)

            bce_target_dis = np.zeros((batch_size, voc_size[0]))
            bce_target_pro = np.zeros((batch_size, voc_size[1]))
            for i in range(batch_size):
                bce_target_dis[i, batch[i][0]] = 1
                bce_target_pro[i, batch[i][1]] = 1
            bce_target = np.concatenate((bce_target_dis, bce_target_pro), axis=1)

            loss_bce = F.binary_cross_entropy_with_logits(
                result, torch.tensor(bce_target, device=device))
            loss = loss_bce + args.moe_loss_weight * moe_aux_loss
            loss.backward()

            router_grad_norm, expert_grad_norm, _, _ = get_moe_grad_norms(model)
            optimizer.step()
            optimizer.zero_grad()
            loss_train += loss.item()
            moe_router_entropy += router_entropy / len(data_train)
            moe_router_grad_norm += router_grad_norm / len(data_train)
            moe_expert_grad_norm += expert_grad_norm / len(data_train)
            moe_expert_usage += expert_usage / len(data_train)

        loss_train /= len(data_train)
        loss_val, ja, prauc, avg_p, avg_r, avg_f1, avg_cnt = evaluator_mask(
            model, data_val, voc_size, epoch, device, mode='pretrain_mask')

        if ja > best_ja_mask:
            best_epoch_mask, best_ja_mask = epoch, ja

        moe_msg = ", ".join(
            [f"E{i}:{u:.3f}" for i, u in enumerate(moe_expert_usage.tolist())])
        logging.info(
            f'[Mask] Epoch {epoch:03d}   train_loss: {loss_train:.4f}, '
            f'val_loss: {loss_val:.4f}, val_ja: {ja:.4f}, '
            f'best_ja: {best_ja_mask:.4f}@ep{best_epoch_mask}, '
            f'expert_usage: [{moe_msg}]')

        writer.add_scalar('Mask/Loss_Train', loss_train, epoch)
        writer.add_scalar('Mask/Loss_Val', loss_val, epoch)
        writer.add_scalar('Mask/Jaccard', ja, epoch)
        writer.add_scalar('Mask/prauc', prauc, epoch)

    save_pretrained_model(model, save_dir, suffix='mask')



# ============================================================================
# 模型存取
# ============================================================================

def save_pretrained_model(model, save_dir, suffix='cer'):
    model_path = os.path.join(save_dir, f'saved.pretrained_model.{suffix}')
    torch.save(model.state_dict(), open(model_path, 'wb'))
    logging.info(f'Pretrained model ({suffix}) saved to {model_path}')


def load_checkpoint_compat(model, model_path, device=None):
    map_location = device if device is not None else 'cpu'
    state_dict = torch.load(open(model_path, 'rb'), map_location=map_location)
    incompatible = model.load_state_dict(state_dict, strict=False)
    if incompatible.missing_keys:
        logging.warning('Missing keys (%d): %s',
                        len(incompatible.missing_keys), incompatible.missing_keys)
    if incompatible.unexpected_keys:
        logging.warning('Unexpected keys (%d): %s',
                        len(incompatible.unexpected_keys), incompatible.unexpected_keys)
    logging.info('Loaded checkpoint from %s (strict=False)', model_path)


def load_pretrained_model(model, model_path):
    load_checkpoint_compat(model, model_path, device=None)


# ============================================================================
# 损失函数
# ============================================================================

def loss_func(voc_size, patient, results, device):
    loss_bce_lst = loss_multi_lst = torch.Tensor().double().to(device)
    for idx, adm in enumerate(patient):
        result = results[idx].unsqueeze(dim=0)
        loss_bce_target = np.zeros((1, voc_size[2]))
        loss_bce_target[:, adm[2]] = 1
        loss_multi_target = np.full((1, voc_size[2]), -1)
        loss_multi_target[0][0:len(adm[2])] = adm[2]
        loss_bce = F.binary_cross_entropy_with_logits(
            result, torch.tensor(loss_bce_target, device=device))
        loss_multi = F.multilabel_margin_loss(
            F.sigmoid(result), torch.tensor(loss_multi_target, device=device))
        loss_bce_lst = torch.cat([loss_bce_lst, loss_bce.view(-1)])
        loss_multi_lst = torch.cat([loss_multi_lst, loss_multi.view(-1)])
    return loss_bce_lst.sum(), loss_multi_lst.sum()


# ============================================================================
# TensorBoard 写入
# ============================================================================

def tensorboard_write(writer, ja, prauc, ddi_rate, avg_med, epoch,
                      loss_train_bce=0.0, loss_train_multi=0.0,
                      loss_train_ddi=0.0, loss_train_all=0.0,
                      loss_val_bce=0.0, loss_val_multi=0.0,
                      loss_val_ddi=0.0, loss_val_all=0.0,
                      loss_train_moe=0.0, loss_train_mono=0.0,
                      moe_router_entropy=0.0, moe_router_grad_norm=0.0,
                      moe_expert_grad_norm=0.0, moe_shared_grad_norm=0.0,
                      moe_rarity_grad_norm=0.0,
                      moe_expert_usage=None, moe_routed_usage=None,
                      causal_scale_val=0.0):
    if epoch > 0:
        writer.add_scalar('Loss_Train/bce', loss_train_bce, epoch)
        writer.add_scalar('Loss_Train/multi', loss_train_multi, epoch)
        writer.add_scalar('Loss_Train/ddi', loss_train_ddi, epoch)
        writer.add_scalar('Loss_Train/sum', loss_train_all, epoch)
        writer.add_scalar('Loss_Train/moe_aux', loss_train_moe, epoch)
        writer.add_scalar('Loss_Train/mono_constraint', loss_train_mono, epoch)
        writer.add_scalar('MoE/router_entropy', moe_router_entropy, epoch)
        writer.add_scalar('MoE/router_grad_norm', moe_router_grad_norm, epoch)
        writer.add_scalar('MoE/expert_grad_norm', moe_expert_grad_norm, epoch)
        writer.add_scalar('MoE/shared_grad_norm', moe_shared_grad_norm, epoch)
        writer.add_scalar('MoE/rarity_grad_norm', moe_rarity_grad_norm, epoch)
        writer.add_scalar('Causal/scale', causal_scale_val, epoch)
        if moe_expert_usage is not None:
            for i, u in enumerate(moe_expert_usage.tolist()):
                writer.add_scalar(f'MoE/top1_usage/E{i}', u, epoch)
        if moe_routed_usage is not None:
            for i, u in enumerate(moe_routed_usage.tolist()):
                writer.add_scalar(f'MoE/routed_usage/E{i}', u, epoch)
        writer.add_scalar('Loss_Val/bce', loss_val_bce, epoch)
        writer.add_scalar('Loss_Val/multi', loss_val_multi, epoch)
        writer.add_scalar('Loss_Val/ddi', loss_val_ddi, epoch)
        writer.add_scalar('Loss_Val/sum', loss_val_all, epoch)
    writer.add_scalar('Metrics/Jaccard', ja, epoch)
    writer.add_scalar('Metrics/prauc', prauc, epoch)
    writer.add_scalar('Metrics/DDI', ddi_rate, epoch)
    writer.add_scalar('Metrics/Med_count', avg_med, epoch)


# ============================================================================
# Entry Point
# ============================================================================

if __name__ == '__main__':
    sys.path.append("..")
    torch.manual_seed(1203)
    np.random.seed(1203)
    random.seed(1203)

    args = get_args()
    main(args)
