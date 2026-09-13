"""Training entry point for DCR-GRN.

The released DCR-GRN model corresponds to the historical ``spore_cib``
implementation.  Public command-line aliases are translated below so that
existing checkpoints remain loadable without exposing experimental names to
users.
"""

import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = os.environ.get(
    "DCRGRN_CUDA_VISIBLE_DEVICES",
    "0",
)
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import random
import time
import torch
import copy

from torch.nn import BCEWithLogitsLoss
from torch.optim import lr_scheduler
from torch_geometric import seed_everything
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from utils.data_utils import load_data, ATFGRN_Dataset
from utils.eval_utils import evaluate_auc_ap
from utils.train_utils1 import construct_knn_graph, train_node2vec_emb
from models import DCRGRN
from config import parser
import numpy as np
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import logging


PUBLIC_VARIANTS = {
    "dcrgrn": "spore_cib",
    "backbone": "directed_idpath_evidential_pcconv_lowdeg",
}


def normalize_variant_name(name):
    """Map documented model names to checkpoint-compatible internal names."""
    normalized = str(name).strip().lower()
    return PUBLIC_VARIANTS.get(normalized, normalized)


def checkpoint_variant_tag(name):
    """Return stable public checkpoint/log names for released variants."""
    normalized = normalize_variant_name(name)
    if normalized == "spore_cib":
        return "DCRGRN"
    if normalized == "directed_idpath_evidential_pcconv_lowdeg":
        return "backbone"
    return normalized
def seed_all(seed):
    """Fix random seeds for reproducibility across CPU and GPU."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)



def parse_seed_list(args):
    raw = str(getattr(args, "seed_list", "")).strip()
    if raw:
        seeds = [int(x.strip()) for x in raw.split(",") if x.strip()]
    else:
        base_seed = int(getattr(args, "seed", 2022))
        seeds = [base_seed + i for i in range(int(args.runs))]
    assert len(seeds) == int(args.runs), f"seed_list length ({len(seeds)}) must equal runs ({args.runs})"
    return seeds


def is_spore_variant(args):
    return str(getattr(args, 'variant', '')).lower() in (
        'spore_grn', 'spore', 'spore_nf',
        'spore_uot', 'spore_epid', 'spore_curo', 'spore_cur', 'spore_mcg',
        'spore_cib', 'spore_cib_safecap', 'spore_cib_rankguard',
        'spore_cib_stablemask', 'spore_cib_scondrank', 'spore_cib_puauc',
        'spore_cib_edgeprompt', 'spore_cib_confdec', 'spore_cib_topopure',
        'spore_cib_dualcontrol', 'spore_cib_gradalign', 'spore_cib_stabledecomp',
        'spore_cib_crsb', 'spore_cib_uti', 'spore_cib_ugrt', 'spore_cib_dehg',
        'spore_cib_per',
        'spore_cib_wo_edecomp', 'spore_cib_wo_tdecomp',
        'spore_cib_wo_degreecond', 'spore_cib_wo_topodeg', 'spore_cib_wo_ib',
        'spore_hcsp'
    )


def is_spore_nf_variant(args):
    return str(getattr(args, 'variant', '')).lower() in (
        'spore_nf', 'spore_uot', 'spore_epid', 'spore_curo', 'spore_cur',
        'spore_mcg', 'spore_cib', 'spore_cib_safecap',
        'spore_cib_rankguard', 'spore_cib_stablemask', 'spore_cib_scondrank',
        'spore_cib_puauc', 'spore_cib_edgeprompt', 'spore_cib_confdec',
        'spore_cib_topopure', 'spore_cib_dualcontrol', 'spore_cib_gradalign',
        'spore_cib_stabledecomp', 'spore_cib_crsb', 'spore_cib_uti',
        'spore_cib_ugrt', 'spore_cib_dehg', 'spore_cib_per', 'spore_cib_wo_edecomp', 'spore_cib_wo_tdecomp', 'spore_cib_wo_degreecond', 'spore_cib_wo_topodeg', 'spore_cib_wo_ib', 'spore_hcsp'
    )


def resolve_retained_checkpoint(args, seed=None, run_idx=None):
    retained_variant = 'directed_idpath_evidential_pcconv_lowdeg'
    seed_tag = 'noseed' if seed is None else str(seed)
    run_tag = 'norun' if run_idx is None else str(run_idx)
    ckpt_dir = getattr(args, 'teacher_dir', '../results/checkpoints')
    candidate_names = (
        f"backbone_{args.netType}_{args.dataset}{args.num}_seed{seed_tag}_run{run_tag}_best_model.pt",
        f"{retained_variant}_{args.netType}_{args.dataset}{args.num}_seed{seed_tag}_run{run_tag}_best_model.pt",
        f"backbone_{args.netType}_{args.dataset}{args.num}_seed{seed_tag}_run1_best_model.pt",
        f"{retained_variant}_{args.netType}_{args.dataset}{args.num}_seed{seed_tag}_run1_best_model.pt",
    )
    for name in candidate_names:
        retained_path = os.path.join(ckpt_dir, name)
        if os.path.exists(retained_path):
            return retained_path
    return os.path.join(ckpt_dir, candidate_names[0])


def load_shape_matched_state(model, state):
    """Load checkpoint tensors whose names and shapes still match this variant.

    New non-evidential SPORE variants intentionally change the final fusion
    head. PyTorch's strict=False still raises on same-name shape mismatches, so
    we explicitly filter those tensors and keep the retained backbone weights.
    """
    if isinstance(state, dict) and 'state_dict' in state:
        state = state['state_dict']
    model_state = model.state_dict()
    compatible = {}
    skipped = []
    for key, value in state.items():
        if key in model_state and hasattr(value, 'shape') and tuple(value.shape) == tuple(model_state[key].shape):
            compatible[key] = value
        else:
            skipped.append(key)
    missing, unexpected = model.load_state_dict(compatible, strict=False)
    return missing, unexpected, skipped, len(compatible)


def update_ema_model(ema_model, model, decay=0.99):
    if ema_model is None:
        return
    with torch.no_grad():
        for ema_p, p in zip(ema_model.parameters(), model.parameters()):
            ema_p.data.mul_(decay).add_(p.data, alpha=1.0 - decay)


def train(model, train_loader,grn_data, knn_graph, device, optimizer, train_dataset, epoch=None, args=None, teacher_model=None, ema_model=None):
    """Execute one training epoch."""
    num=0
    model.train()
    total_loss = 0
    y_pred, y_true = [], []
    for data in tqdm(train_loader, ncols=70):
        data = data.to(device)
        optimizer.zero_grad()
        # Forward pass: get logits from global feature (g) and different views (1, 2, 3)
        logits_g,logits_1, logits_2, logits_3 = model(data,grn_data, knn_graph)
        stablemask_extra = logits_3.new_tensor(0.0)
        variant_name = str(getattr(args, 'variant', '')).lower() if args is not None else ''
        if args is not None and variant_name in ('spore_cib_stablemask', 'spore_cib_stabledecomp'):
            first_spore = getattr(model, 'last_spore', None)
            _, _, _, logits_3_second = model(data, grn_data, knn_graph)
            second_spore = getattr(model, 'last_spore', None)
            if first_spore is not None and second_spore is not None and hasattr(model, 'compute_stablemask_loss'):
                stablemask_extra = model.compute_stablemask_loss(
                    first_spore, second_spore, logits_3, logits_3_second, data.y.to(torch.float), args=args
                )
                model.last_spore = first_spore
        if hasattr(model, 'compute_evidential_loss') and model.evidential_enabled:
            loss = model.compute_evidential_loss(data.y.to(torch.float), epoch=epoch)
        else:
            loss_1 = BCEWithLogitsLoss()(logits_1.view(-1), data.y.to(torch.float))
            loss_2 = BCEWithLogitsLoss()(logits_2.view(-1), data.y.to(torch.float))
            loss_3 = BCEWithLogitsLoss()(logits_3.view(-1), data.y.to(torch.float))
            loss_g = BCEWithLogitsLoss()(logits_g.view(-1), data.y.to(torch.float))

            # Aggregate losses (Multi-task learning strategy)
            loss = loss_3+loss_2+loss_1+loss_g
        if hasattr(model, 'auxiliary_loss'):
            loss = loss + model.auxiliary_loss()
        if args is not None and is_spore_variant(args) and hasattr(model, 'compute_spore_loss'):
            loss = loss + model.compute_spore_loss(
                data.y.to(torch.float),
                lambda_rec=getattr(args, 'spore_lambda_rec', 0.05),
                lambda_orth=getattr(args, 'spore_lambda_orth', 0.01),
                lambda_share=getattr(args, 'spore_lambda_share', 0.01),
                lambda_syn=getattr(args, 'spore_lambda_syn', 0.02),
                lambda_degree=getattr(args, 'spore_lambda_degree', 0.005),
            )
        if args is not None and hasattr(model, 'compute_variant_loss'):
            loss = loss + model.compute_variant_loss(data.y.to(torch.float), args=args)
        loss = loss + stablemask_extra
        if ema_model is not None and variant_name == 'spore_cib_stabledecomp':
            lam_ema = float(getattr(args, 'cib_stable_ema_lambda', 0.02))
            conf_thr = float(getattr(args, 'cib_stable_ema_conf', 0.70))
            if lam_ema > 0:
                ema_model.eval()
                with torch.no_grad():
                    _, _, _, ema_logits = ema_model(data, grn_data, knn_graph)
                    ema_prob = torch.sigmoid(ema_logits.view(-1)).clamp(min=1e-6, max=1.0 - 1e-6)
                    conf = torch.maximum(ema_prob, 1.0 - ema_prob)
                    mask = conf > conf_thr
                if mask.any():
                    student_prob = torch.sigmoid(logits_3.view(-1)).clamp(min=1e-6, max=1.0 - 1e-6)
                    ema_kl = ema_prob * torch.log(ema_prob / student_prob) + (1.0 - ema_prob) * torch.log((1.0 - ema_prob) / (1.0 - student_prob))
                    loss = loss + lam_ema * ema_kl[mask].mean()
        if teacher_model is not None and args is not None and is_spore_nf_variant(args):
            anchor_weight = float(getattr(args, 'spore_nf_anchor_weight', 0.02))
            if anchor_weight > 0:
                with torch.no_grad():
                    _, _, _, teacher_logits = teacher_model(data, grn_data, knn_graph)
                    teacher_prob = torch.sigmoid(teacher_logits.view(-1))
                    label = data.y.to(torch.float).view(-1)
                    high_conf = torch.abs(teacher_prob - 0.5) > float(getattr(args, 'spore_nf_anchor_delta', 0.25))
                    correct = (teacher_prob >= 0.5).eq(label >= 0.5)
                    mask = high_conf & correct
                if mask.any():
                    if str(getattr(args, 'variant', '')).lower() == 'spore_cib_safecap':
                        suf_weight = float(getattr(args, 'cib_safecap_lambda_suf', anchor_weight))
                        rho = float(getattr(args, 'cib_safecap_pos_rho', 1.5))
                        if epoch is None:
                            suf_decay = 1.0
                        else:
                            pivot = 0.30 * float(getattr(args, 'epochs', 80))
                            if float(epoch) <= pivot:
                                suf_decay = 1.0
                            else:
                                suf_decay = max(0.0, 1.0 - (float(epoch) - pivot) / max(1.0, float(getattr(args, 'epochs', 80)) - pivot))
                        student_prob = torch.sigmoid(logits_3.view(-1)).clamp(min=1e-6, max=1.0 - 1e-6)
                        tprob = teacher_prob.detach().clamp(min=1e-6, max=1.0 - 1e-6)
                        bern_kl = tprob * torch.log(tprob / student_prob) + (1.0 - tprob) * torch.log((1.0 - tprob) / (1.0 - student_prob))
                        weights = (1.0 + rho * label) * mask.to(student_prob.dtype)
                        anchor_loss = (weights * bern_kl).sum() / weights.sum().clamp(min=1.0)
                        loss = loss + suf_weight * suf_decay * anchor_loss
                    else:
                        anchor_loss = torch.mean((logits_3.view(-1)[mask] - teacher_logits.detach().view(-1)[mask]).pow(2))
                        loss = loss + anchor_weight * anchor_loss
        if variant_name == 'spore_cib_gradalign' and hasattr(model, 'compute_cib_rank_raw_loss') and hasattr(model, 'gradalign_parameters'):
            rank_loss = model.compute_cib_rank_raw_loss(data.y.to(torch.float), args=args)
            shared_params = model.gradalign_parameters()
            if shared_params and getattr(rank_loss, 'requires_grad', False):
                main_grads = torch.autograd.grad(loss, shared_params, retain_graph=True, allow_unused=True)
                rank_grads = torch.autograd.grad(rank_loss, shared_params, retain_graph=True, allow_unused=True)
                dot = loss.new_tensor(0.0)
                main_norm_sq = loss.new_tensor(0.0)
                rank_norm_sq = loss.new_tensor(0.0)
                for gm, gr in zip(main_grads, rank_grads):
                    if gm is None or gr is None:
                        continue
                    dot = dot + (gm.detach() * gr.detach()).sum()
                    main_norm_sq = main_norm_sq + gm.detach().pow(2).sum()
                    rank_norm_sq = rank_norm_sq + gr.detach().pow(2).sum()
                loss.backward(retain_graph=True)
                lam_rank = float(getattr(args, 'cib_gradalign_rank_lambda', 0.08))
                kappa = float(getattr(args, 'cib_gradalign_kappa', 0.50))
                rho = float(getattr(args, 'cib_gradalign_rho', 0.25))
                eps = 1e-12
                proj_grads = []
                for gm, gr in zip(main_grads, rank_grads):
                    if gr is None:
                        proj_grads.append(None)
                    elif gm is None or dot.item() >= 0:
                        proj_grads.append(gr.detach())
                    else:
                        proj_grads.append((gr - kappa * dot / (main_norm_sq + eps) * gm).detach())
                proj_norm_sq = loss.new_tensor(0.0)
                for pg in proj_grads:
                    if pg is not None:
                        proj_norm_sq = proj_norm_sq + pg.pow(2).sum()
                scale = 1.0
                if proj_norm_sq.item() > 0 and main_norm_sq.item() > 0:
                    max_norm = rho * torch.sqrt(main_norm_sq + eps)
                    cur_norm = lam_rank * torch.sqrt(proj_norm_sq + eps)
                    if cur_norm > max_norm:
                        scale = (max_norm / (cur_norm + eps)).detach().item()
                for p, pg in zip(shared_params, proj_grads):
                    if pg is None:
                        continue
                    if p.grad is None:
                        p.grad = torch.zeros_like(p)
                    p.grad.add_(pg, alpha=lam_rank * scale)
                if hasattr(model, 'last_spore'):
                    model.last_spore['gradalign_cos'] = (dot / (torch.sqrt(main_norm_sq * rank_norm_sq) + eps)).detach()
                    model.last_spore['gradalign_scale'] = loss.new_tensor(scale)
            else:
                loss.backward()
        else:
            loss.backward()
        optimizer.step()
        if ema_model is not None and variant_name == 'spore_cib_stabledecomp':
            update_ema_model(ema_model, model, decay=float(getattr(args, 'cib_stable_ema_decay', 0.99)))
        num+=1
        y_pred.append(logits_3.detach().view(-1).cpu())
        y_true.append(data.y.detach().view(-1).cpu().to(torch.float))
        total_loss += loss.item()
    y_true, y_pred = torch.cat(y_true), torch.cat(y_pred)

    return total_loss/num,evaluate_auc_ap(y_pred, y_true)

@torch.no_grad()
def test(args, loader, grn_data,knn_graph, model, device,dataset):
    """Evaluate model performance on Validation or Test sets."""
    model.eval()
    num=0
    total_loss = 0
    y_pred, y_true = [], []
    for data in tqdm(loader, ncols=70):
        data = data.to(device)
        logits_g,logits1,logits2, logits3 = model(data, grn_data,knn_graph)

        loss_3 = BCEWithLogitsLoss()(logits3.view(-1), data.y.to(torch.float))


        loss = loss_3
        total_loss += loss.item()
        num+=1
        y_pred.append(logits3.detach().view(-1).cpu())
        y_true.append(data.y.detach().view(-1).cpu().to(torch.float))


    y_true, y_pred = torch.cat(y_true), torch.cat(y_pred)
    return total_loss/num, evaluate_auc_ap(y_pred, y_true)

def Adj(args):
    """Load positive edges from the training file to construct the known GRN structure."""
    data_dir = '../data/' + args.netType + '/' + args.dataset + ' ' + args.num
    train_file = data_dir + '/Train_set.csv'

    df = pd.read_csv(train_file, index_col=0, header=0)

    # Filter only positive interactions
    pos_edges = df[df["Label"] == 1][["TF", "Target"]].values.tolist()
    pos_edge_index = torch.tensor(pos_edges, dtype=torch.long).t().contiguous()
    return pos_edge_index


def coexpr_edge_weight(edge_index, expr):
    """Scheme 10 edge confidence: fallback confidence from absolute Pearson co-expression.

    STRING can use native scores when available; the current ATFGRN Specific CSV has
    binary labels only, so this faithful fallback follows the proposal: use
    expression consistency as edge_attr for TransformerConv.
    """
    src, dst = edge_index.to(expr.device)
    x = expr[src]
    y = expr[dst]
    x = x - x.mean(dim=1, keepdim=True)
    y = y - y.mean(dim=1, keepdim=True)
    w = (x * y).sum(dim=1) / (x.norm(dim=1) * y.norm(dim=1) + 1e-8)
    w = w.abs().clamp(0.0, 1.0)
    return w.view(-1, 1)


def run(args, seed=None, run_idx=None):
    """Main pipeline: Data loading, Pre-processing, Training, and Evaluation."""
    if seed is not None:
        seed_all(int(seed))

    # 1. Load gene expression data
    expfile = "../Benchmark Dataset/"+args.netType+' Dataset/'+args.dataset+'/TFs+'+args.num+'/BL--ExpressionData.csv'
    save_path = '../Data_process'+ '/' +args.netType+ '/' + args.dataset + ' ' + args.num
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    dataset = load_data(expfile,save_path)
    learning_rate = float(args.lr)
    logger.info(f"Learning rate: {learning_rate}")

    # 2. Construct KNN graph and learn initial node embeddings (Node2Vec)
    knn_graph = construct_knn_graph(data=dataset[0])
    emb = train_node2vec_emb(knn_graph, seed=seed)
    knn_graph.x = emb
    if seed is not None:
        seed_all(int(seed))

    # 3. Initialize Dataset objects for Train/Val/Test for Subgraph
    train_dataset = ATFGRN_Dataset(dataset, args, num_hops=2, split='train')
    val_dataset = ATFGRN_Dataset(dataset, args, num_hops=2, split='val')
    test_dataset = ATFGRN_Dataset(dataset, args, num_hops=2, split='test')

    loader_generator = torch.Generator()
    if seed is not None:
        loader_generator.manual_seed(int(seed))
    train_loader = DataLoader(train_dataset, batch_size=args.bs, shuffle=True, generator=loader_generator)
    val_loader = DataLoader(val_dataset, batch_size=args.bs, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.bs, shuffle=False)

    device = torch.device('cuda:0' if args.cuda else 'cpu')

    train_data = Adj(args)
    feature = dataset[0].x.to(device)
    grn_data= Data(x=feature, edge_index=train_data).to(device)
    grn_data.edge_index_directed = train_data.to(device)
    if getattr(args, 'variant', 'baseline') in ('edgeconf', 'evidential_edgeconf'):
        grn_data.edge_attr = coexpr_edge_weight(grn_data.edge_index, grn_data.x)

    # 4. Initialize Model, Optimizer, and Scheduler
    variant = getattr(args, 'variant', 'baseline')
    model = DCRGRN(train_dataset, feature.size(1), train_dataset[0].num_features, hidden_channels=32, out_channels=32, num_layers=args.num_layers, variant=variant, net_type=getattr(args, 'netType', 'Specific')).to(device)
    if is_spore_variant(args) and getattr(args, 'spore_load_retained', True):
        retained_path = resolve_retained_checkpoint(args, seed=seed, run_idx=run_idx)
        if os.path.exists(retained_path):
            state = torch.load(retained_path, map_location=device)
            missing, unexpected, skipped, loaded = load_shape_matched_state(model, state)
            logger.info(f"SPORE initialized from retained checkpoint: {retained_path}")
            logger.info(f"SPORE load_state loaded={loaded} missing={len(missing)} unexpected={len(unexpected)} skipped_shape_or_name={len(skipped)}")
            if skipped:
                logger.info(f"SPORE skipped checkpoint keys sample: {skipped[:8]}")
        else:
            logger.info(f"SPORE retained checkpoint not found, training from random init: {retained_path}")
    teacher_model = None
    if is_spore_nf_variant(args) and float(getattr(args, 'spore_nf_anchor_weight', 0.02)) > 0:
        retained_path = resolve_retained_checkpoint(args, seed=seed, run_idx=run_idx)
        if os.path.exists(retained_path):
            teacher_model = DCRGRN(
                train_dataset,
                feature.size(1),
                train_dataset[0].num_features,
                hidden_channels=32,
                out_channels=32,
                num_layers=args.num_layers,
                variant='directed_idpath_evidential_pcconv_lowdeg',
                net_type=getattr(args, 'netType', 'Specific')
            ).to(device)
            teacher_model.load_state_dict(torch.load(retained_path, map_location=device), strict=False)
            teacher_model.eval()
            for p in teacher_model.parameters():
                p.requires_grad_(False)
            logger.info(f"SPORE-NF teacher anchor loaded from retained checkpoint: {retained_path}")
        else:
            logger.info(f"SPORE-NF teacher anchor skipped; retained checkpoint not found: {retained_path}")
    ema_model = None
    if str(getattr(args, 'variant', '')).lower() == 'spore_cib_stabledecomp':
        ema_model = copy.deepcopy(model).to(device)
        ema_model.eval()
        for p in ema_model.parameters():
            p.requires_grad_(False)
        logger.info(
            f"StableDecomp-CIB EMA teacher initialized from current student; "
            f"decay={getattr(args, 'cib_stable_ema_decay', 0.99)}"
        )
    logger.info(model)
    knn_graph = knn_graph.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=args.wd)
    schedular = lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.99)

    # Metrics initialization
    final_test_auc = final_test_ap = 0
    early_stop_best = 0
    patience = 0

    ckpt_dir = '../results/checkpoints'
    os.makedirs(ckpt_dir, exist_ok=True)
    seed_tag = 'noseed' if seed is None else str(seed)
    run_tag = 'norun' if run_idx is None else str(run_idx)
    checkpoint_tag = checkpoint_variant_tag(getattr(args, 'variant', 'dcrgrn'))
    best_model_path = os.path.join(
        ckpt_dir,
        f"{checkpoint_tag}_{args.netType}_{args.dataset}{args.num}_seed{seed_tag}_run{run_tag}_best_model.pt",
    )

    # model.load_state_dict(torch.load(best_model_path))
    # _, final_test_results = test(args, test_loader, grn_data, knn_graph, model, device, test_dataset)

    # 5. Training Loop
    for epoch in range(1, args.epochs):
        schedular.step()
        loss, result = train(
            model, train_loader, grn_data, knn_graph, device, optimizer, train_dataset,
            epoch=epoch, args=args, teacher_model=teacher_model, ema_model=ema_model
        )
        _, val_results = test(args, val_loader, grn_data, knn_graph, model, device, val_dataset)
        train_auc, train_ap = result['AUC'], result['AP']
        # Early stopping & model selection: based on Val AUC
        if args.metric == 'auc_ap':
            val_auc, val_ap = val_results['AUC'], val_results['AP']
            if round(val_auc, 4) > round(early_stop_best, 4):
                early_stop_best = val_auc
                patience = 0
                torch.save(model.state_dict(), best_model_path)
            else:
                patience += 1

            logger.info(
                f'Epoch: {epoch:02d}, trainLoss: {loss:.4f}, train_AUC: {train_auc:.4f}, train_AP: {train_ap:.4f},'
                f'Val_AUC: {val_auc:.4f}, Val_AP: {val_ap:.4f}'
            )
            if patience >= args.patience:
                break
    else:
        logger.info('All epochs done.')

    # Final evaluation
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    _, final_test_results = test(args, test_loader, grn_data, knn_graph, model, device, test_dataset)
    final_test_auc = final_test_results['AUC']
    final_test_ap = final_test_results['AP']
    fe10 = final_test_results.get('EPR@10', 0); fe5 = final_test_results.get('EPR@05', 0); fe1 = final_test_results.get('EPR@01', 0); ff1 = final_test_results.get('F1@10', 0)
    logger.info(f'Final AUC:{final_test_auc:.4f} AP:{final_test_ap:.4f} EPR@01:{fe1:.4f} EPR@05:{fe5:.4f} EPR@10:{fe10:.4f} F1@10:{ff1:.4f}')
    return [final_test_auc, final_test_ap, fe10, ff1]


if __name__ == '__main__':
    args = parser.parse_args()
    args.variant = normalize_variant_name(args.variant)

    # Setup Logging
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    exp_time = '_'.join(time.asctime().split(' '))

    log_dir = '../results/logs/'
    log_file = log_dir + 'Log_{}_{}_{}{}__{}.txt'.format(checkpoint_variant_tag(args.variant), args.netType, args.dataset.capitalize(), args.num,exp_time)
    os.makedirs(log_dir, exist_ok=True)

    # File and Console handlers
    handler = logging.FileHandler(log_file)
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    logger.addHandler(console)

    logger.info(args)
    res = []

    fixed_seeds = parse_seed_list(args)
    logger.info(f'FIXED_SEEDS:{fixed_seeds}')
    for i in range(args.runs):
        seed = fixed_seeds[i]
        logger.info(f'===== RUN {i + 1}/{args.runs} SEED {seed} =====')
        seed_all(seed)
        try:
            seed_everything(seed)
        except NameError:
            pass
        results = run(args, seed=seed, run_idx=i + 1)
        res.append(results)

    # Calculate and report average metrics
    if args.metric == 'auc_ap':
        for i in range(len(res)):
            logger.info(f'Run: {i + 1:2d}, AUC={res[i][0]:.4f} AP={res[i][1]:.4f} EPR={res[i][2]:.4f} F1={res[i][3]:.4f}')
        auc = sum(r[0] for r in res) / args.runs
        ap  = sum(r[1] for r in res) / args.runs
        epr = sum(r[2] for r in res) / args.runs
        f1  = sum(r[3] for r in res) / args.runs
        logger.info(f"Avg AUC:{auc:.4f} AP:{ap:.4f} EPR@10:{epr:.4f} F1@10:{f1:.4f}")
