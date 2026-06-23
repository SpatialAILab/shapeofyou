import torch
import torch.nn.functional as F

def get_sparse_contrastive_loss(
    src_feats, trg_feats, logit_scale=None, bin_score=None, pseudo_labels=None, trg_mask=None, include_unmatch=True, ot_iters=10, beta=0.3, sparse_loss="contrastive", group_ids=None
):
    if sparse_loss == "ot":
        C, P, _ = trg_feats.shape
        device = src_feats.device

        pseudo_label_src_coords, pseudo_label_trg_coords = pseudo_labels
        # src_feats_sampled = src_feats[:, pseudo_label_src_coords[:, 1], pseudo_label_src_coords[:, 0]].T
        # trg_mask_flat = trg_mask.view(-1)
        # trg_feats_sampled = trg_feats.reshape(C, P * P)[:, trg_mask_flat].T

        # num_src_points = src_feats_sampled.shape[0]
        # num_trg_points = trg_feats_sampled.shape[0]

        # full_to_masked_map = torch.full((P * P,), -1, dtype=torch.long, device=device)
        # original_fg_indices = torch.where(trg_mask_flat)[0]
        # new_masked_indices = torch.arange(num_trg_points, device=device)
        # full_to_masked_map[original_fg_indices] = new_masked_indices
        # gt_trg_indices_full = pseudo_label_trg_coords[:, 1] * P + pseudo_label_trg_coords[:, 0]
        # gt_trg_indices_masked = full_to_masked_map[gt_trg_indices_full]

        # valid_gt_mask = (gt_trg_indices_masked != -1)
        # if not valid_gt_mask.any():
        #     return torch.tensor(0.0, device=device)

        # gt_src_indices = torch.arange(num_src_points, device=device)[valid_gt_mask]
        # gt_trg_indices_masked = gt_trg_indices_masked[valid_gt_mask]

        # log_assignment = get_ot_plan(src_feats_sampled, trg_feats_sampled).squeeze(0)

        # log_p_matched = log_assignment[gt_src_indices, gt_trg_indices_masked]
        # loss = -log_p_matched.mean()

        # return loss
        src_feats_sampled = src_feats[:, pseudo_label_src_coords[:, 1], pseudo_label_src_coords[:, 0]]
        num_gt_matches = src_feats_sampled.shape[1]
        trg_feats_flat = trg_feats.reshape(C, P * P)

        src_feats_normalized = F.normalize(src_feats_sampled.T, p=2, dim=1)
        trg_feats_normalized = F.normalize(trg_feats_flat.T, p=2, dim=1)

        scores = torch.matmul(src_feats_normalized, trg_feats_normalized.T)
        log_assignment = log_optimal_transport(
            scores.unsqueeze(0), bin_score, iters=ot_iters
        ).squeeze(0)

        gt_src_indices = torch.arange(num_gt_matches, device=device)
        gt_trg_indices = pseudo_label_trg_coords[:, 1] * P + pseudo_label_trg_coords[:, 0]

        log_p_matched = log_assignment[gt_src_indices, gt_trg_indices]
        loss_matched = -log_p_matched.mean()
        trg_background_mask = (trg_mask.view(-1) == 0)
        log_p_unmatched = log_assignment[-1, :-1][trg_background_mask]

        if trg_background_mask.sum().item() > 0 and include_unmatch:
            loss_unmatched = -log_p_unmatched.mean()
        else:
            loss_unmatched = torch.tensor(0.0, device=device)

        total_loss = loss_matched + 0.01 * loss_unmatched
        return total_loss

    elif sparse_loss == "soft_target":
        C, P, _ = trg_feats.shape
        device = src_feats.device

        pseudo_label_src_coords, pseudo_label_trg_coords = pseudo_labels  # (N,2), (x,y)

        def _flat_idx(xy):
            return xy[:, 1] * P + xy[:, 0]

        dual_mask = (trg_mask is not None and trg_mask.shape[0] == 2)

        if dual_mask:
            src_mask = trg_mask[0].to(torch.bool)
            t_mask   = trg_mask[1].to(torch.bool)
            src_mask_flat = src_mask.view(-1)
            trg_mask_flat = t_mask.view(-1)

            src_feats_sampled = src_feats.reshape(C, P * P)[:, src_mask_flat].T  # (Ns, C)
            trg_feats_sampled = trg_feats.reshape(C, P * P)[:, trg_mask_flat].T  # (Nt, C)

            full2masked_src = torch.full((P * P,), -1, dtype=torch.long, device=device)
            full2masked_trg = torch.full((P * P,), -1, dtype=torch.long, device=device)
            full2masked_src[torch.where(src_mask_flat)[0]] = torch.arange(src_feats_sampled.size(0), device=device)
            full2masked_trg[torch.where(trg_mask_flat)[0]] = torch.arange(trg_feats_sampled.size(0), device=device)

            gt_src_indices = full2masked_src[_flat_idx(pseudo_label_src_coords)]  # (N,)
            gt_trg_indices = full2masked_trg[_flat_idx(pseudo_label_trg_coords)]  # (N,)

            src_pl = src_feats_sampled[gt_src_indices]  # (N, C)
            trg_pl = trg_feats_sampled[gt_trg_indices]  # (N, C)

            src_pl = F.normalize(src_pl, p=2, dim=1)
            trg_pl = F.normalize(trg_pl, p=2, dim=1)

            with torch.no_grad():
                log_Q_pl = get_ot_plan(src_pl, trg_pl, iters=ot_iters)  # (1, N, N)
            Q_pl = torch.exp(log_Q_pl).squeeze(0)  # (N, N)

            N = src_pl.size(0)
            hard_target = torch.zeros(N, N, device=device)
            diag = torch.arange(N, device=device)
            hard_target[diag, diag] = 1

            soft_target = (1 - beta) * hard_target + beta * Q_pl

            logits = logit_scale * (src_pl @ trg_pl.t())  # (N, N)
            loss_src2trg = -(soft_target * F.log_softmax(logits, dim=1)).sum(dim=1).mean()
            loss_trg2src = -(soft_target.t() * F.log_softmax(logits.t(), dim=1)).sum(dim=1).mean()
        else:
            # ---- single mask: keep the existing Ns x Nt formulation ----
            src_feats_sampled = src_feats[:, pseudo_label_src_coords[:, 1], pseudo_label_src_coords[:, 0]].T  # (Ns, C)

            trg_mask_flat = trg_mask.view(-1).to(torch.bool)
            trg_feats_sampled = trg_feats.reshape(C, P * P)[:, trg_mask_flat].T  # (Nt, C)

            full2masked_trg = torch.full((P * P,), -1, dtype=torch.long, device=device)
            full2masked_trg[torch.where(trg_mask_flat)[0]] = torch.arange(trg_feats_sampled.size(0), device=device)

            gt_src_indices = torch.arange(src_feats_sampled.size(0), device=device)  # (Ns,)
            gt_trg_indices = full2masked_trg[_flat_idx(pseudo_label_trg_coords)]    # (Ns,)

            src_feats_norm = F.normalize(src_feats_sampled, p=2, dim=1)  # (Ns, C)
            trg_feats_norm = F.normalize(trg_feats_sampled, p=2, dim=1)  # (Nt, C)

            with torch.no_grad():
                log_Q_star = get_ot_plan(src_feats_norm, trg_feats_norm, iters=ot_iters)  # (1, Ns, Nt)
            Q_star = torch.exp(log_Q_star).squeeze(0)  # (Ns, Nt)

            hard_target = torch.zeros(src_feats_norm.size(0), trg_feats_norm.size(0), device=device)
            hard_target[gt_src_indices, gt_trg_indices] = 1

            soft_target = (1 - beta) * hard_target + beta * Q_star

            logits = logit_scale * (src_feats_norm @ trg_feats_norm.t())  # (Ns, Nt)
            loss_src2trg = -(soft_target * F.log_softmax(logits, dim=1)).sum(dim=1).mean()
            loss_trg2src = -(soft_target.t() * F.log_softmax(logits.t(), dim=1)).sum(dim=1).mean()

        return (loss_src2trg + loss_trg2src) / 2.0

    elif sparse_loss == "supcon":
        # Supervised Contrastive Loss for handling multiple positives from same source
        # src_feats, trg_feats: (N, C) - features for source and target
        # group_ids: (N,) - group assignment for each sample

        if group_ids is None:
            # Fallback to standard contrastive if no group_ids provided
            src_feats_norm = F.normalize(src_feats, p=2, dim=1)
            trg_feats_norm = F.normalize(trg_feats, p=2, dim=1)
            logits_per_src = logit_scale * src_feats_norm @ trg_feats_norm.T
            labels = torch.arange(src_feats_norm.shape[0], device=src_feats_norm.device, dtype=torch.long)
            loss_src2trg = F.cross_entropy(logits_per_src, labels)
            loss_trg2src = F.cross_entropy(logits_per_src.T, labels)
            return (loss_src2trg + loss_trg2src) / 2.0

        # Filter out padded entries (group_id == -1)
        valid_mask = (group_ids != -1)
        if not valid_mask.any():
            return torch.tensor(0.0, device=src_feats.device)

        src_feats_valid = src_feats[valid_mask]
        trg_feats_valid = trg_feats[valid_mask]
        group_ids_valid = group_ids[valid_mask]

        # Normalize features
        src_feats_norm = F.normalize(src_feats_valid, p=2, dim=1)  # (M, C)
        trg_feats_norm = F.normalize(trg_feats_valid, p=2, dim=1)  # (M, C)

        N = src_feats_norm.shape[0]
        temperature = 1.0 / logit_scale if logit_scale is not None else 0.07

        # Compute similarity matrices
        sim_src2trg = (src_feats_norm @ trg_feats_norm.T) / temperature  # (M, M)
        sim_trg2src = sim_src2trg.T

        # Create masks for same group (positives) and different group (negatives)
        # mask_pos[i, j] = 1 if group_ids[i] == group_ids[j] and i != j
        group_matrix = group_ids_valid.unsqueeze(0) == group_ids_valid.unsqueeze(1)  # (M, M)
        mask_self = torch.eye(N, dtype=torch.bool, device=src_feats.device)
        mask_pos = group_matrix & (~mask_self)  # Same group but not self

        # For each anchor, compute SupCon loss
        # L_i = -log( sum_{p in P(i)} exp(sim(i,p)) / sum_{a != i} exp(sim(i,a)) )

        def supcon_loss_one_way(sim_matrix, mask_pos):
            # sim_matrix: (M, M)
            # mask_pos: (M, M) - positive pairs mask

            # Numerator: sum of exp(sim) for positive pairs
            exp_sim = torch.exp(sim_matrix)  # (M, M)

            # For numerical stability, use log-sum-exp trick
            # Denominator: sum of exp(sim) for all pairs except self
            mask_valid = ~mask_self  # All except diagonal

            # Compute log of denominator
            log_denominator = torch.logsumexp(sim_matrix.masked_fill(~mask_valid, float('-inf')), dim=1)  # (M,)

            # For each anchor, compute log(sum of positives)
            # If an anchor has no positives, skip it
            num_positives = mask_pos.sum(dim=1)  # (M,)
            has_positive = num_positives > 0

            if not has_positive.any():
                return torch.tensor(0.0, device=src_feats.device)

            # Compute log of numerator for anchors with positives
            log_numerator = torch.zeros(N, device=src_feats.device)
            log_numerator[has_positive] = torch.logsumexp(
                sim_matrix[has_positive].masked_fill(~mask_pos[has_positive], float('-inf')),
                dim=1
            )

            # Loss: -log(numerator / denominator) = -(log_numerator - log_denominator)
            loss_per_anchor = -(log_numerator - log_denominator)

            # Average over anchors that have positives
            loss = loss_per_anchor[has_positive].mean()
            return loss

        loss_src2trg = supcon_loss_one_way(sim_src2trg, mask_pos)
        loss_trg2src = supcon_loss_one_way(sim_trg2src, mask_pos)

        return (loss_src2trg + loss_trg2src) / 2.0

    elif sparse_loss == "soft_supcon":
        # Soft Supervised Contrastive Loss
        # Build soft positives by linearly mixing the hard positive mask with the OT plan
        # soft_pos = (1 - beta) * hard_pos + beta * Q_ot

        if group_ids is None:
            # Fallback to standard contrastive if no group_ids provided
            src_feats_norm = F.normalize(src_feats, p=2, dim=1)
            trg_feats_norm = F.normalize(trg_feats, p=2, dim=1)
            logits_per_src = logit_scale * src_feats_norm @ trg_feats_norm.T
            labels = torch.arange(src_feats_norm.shape[0], device=src_feats_norm.device, dtype=torch.long)
            loss_src2trg = F.cross_entropy(logits_per_src, labels)
            loss_trg2src = F.cross_entropy(logits_per_src.T, labels)
            return (loss_src2trg + loss_trg2src) / 2.0

        # Filter out padded entries (group_id == -1)
        valid_mask = (group_ids != -1)
        if not valid_mask.any():
            return torch.tensor(0.0, device=src_feats.device)

        src_feats_valid = src_feats[valid_mask]
        trg_feats_valid = trg_feats[valid_mask]
        group_ids_valid = group_ids[valid_mask]

        # Normalize features
        src_feats_norm = F.normalize(src_feats_valid, p=2, dim=1)
        trg_feats_norm = F.normalize(trg_feats_valid, p=2, dim=1)

        N = src_feats_norm.shape[0]
        temperature = 1.0 / logit_scale if logit_scale is not None else 0.07

        # Compute similarity matrix
        sim_src2trg = (src_feats_norm @ trg_feats_norm.T) / temperature
        sim_trg2src = sim_src2trg.T

        # Hard positive mask (same group, excluding self-pairs)
        group_matrix = group_ids_valid.unsqueeze(0) == group_ids_valid.unsqueeze(1)
        mask_self = torch.eye(N, dtype=torch.bool, device=src_feats.device)
        hard_pos = (group_matrix & (~mask_self)).float()

        # Compute the OT plan
        with torch.no_grad():
            log_Q = get_ot_plan(src_feats_norm, trg_feats_norm, iters=ot_iters)
        Q_ot = torch.exp(log_Q).squeeze(0)  # (N, N)

        # Exclude self-pairs
        Q_ot = Q_ot * (~mask_self).float()

        # Soft positive mask (linear mixture)
        soft_pos = (1 - beta) * hard_pos + beta * Q_ot

        def soft_supcon_loss_one_way(sim_matrix, soft_pos_matrix):
            mask_valid = ~mask_self

            # Log softmax over all valid pairs (excluding self-pairs)
            log_prob = sim_matrix - torch.logsumexp(
                sim_matrix.masked_fill(~mask_valid, float('-inf')), dim=1, keepdim=True
            )

            # Weighted sum of log probabilities
            pos_weight_sum = soft_pos_matrix.sum(dim=1)
            has_positive = pos_weight_sum > 1e-8

            if not has_positive.any():
                return torch.tensor(0.0, device=src_feats.device)

            # Loss = -sum(soft_pos * log_prob) / sum(soft_pos)
            weighted_log_prob = (soft_pos_matrix * log_prob).sum(dim=1)
            loss_per_anchor = -weighted_log_prob / pos_weight_sum.clamp(min=1e-8)

            return loss_per_anchor[has_positive].mean()

        loss_src2trg = soft_supcon_loss_one_way(sim_src2trg, soft_pos)
        loss_trg2src = soft_supcon_loss_one_way(sim_trg2src, soft_pos.T)

        return (loss_src2trg + loss_trg2src) / 2.0

    else:
        src_feats = F.normalize(src_feats, p=2, dim=1)
        trg_feats = F.normalize(trg_feats, p=2, dim=1)

        logits_per_src = logit_scale * src_feats @ trg_feats.T
        labels = torch.arange(
            src_feats.shape[0], device=src_feats.device, dtype=torch.long
        )

        loss_src2trg = F.cross_entropy(logits_per_src, labels)
        loss_trg2src = F.cross_entropy(logits_per_src.T, labels)

        return (loss_src2trg + loss_trg2src) / 2.0

def get_dense_loss(src_feat_map, tgt_feat_map, src_coords, tgt_coords, corr_map_net, std=0.1, threshold=1.0):
    C, P, _ = src_feat_map.shape

    src_flat = src_feat_map.view(C, -1).T
    tgt_flat = tgt_feat_map.view(C, -1)
    corr_map = torch.matmul(src_flat.contiguous(), tgt_flat.contiguous())

    corr_map = corr_map.reshape(1, P, P, P, P)
    predicted_flow_map = corr_map_net(corr_map)

    gt_flow = (tgt_coords - src_coords).float()

    if std > 0:
        std = std * threshold / 2
        noise = torch.randn_like(gt_flow, dtype=torch.float32) * std
        gt_flow = gt_flow + noise

    pred_flow_at_coords = predicted_flow_map[0, src_coords[:, 1], src_coords[:, 0], :]
    epe_loss = torch.norm(pred_flow_at_coords - gt_flow, dim=-1).mean()

    return epe_loss

def log_sinkhorn_iterations(Z, log_mu, log_nu, iters=100):
    device = Z.device
    u, v = (
        torch.zeros_like(log_mu, device=device),
        torch.zeros_like(log_nu, device=device),
    )
    with torch.no_grad():
        for _ in range(iters - 1):
            u = log_mu - torch.logsumexp(Z + v.unsqueeze(1), dim=2)
            v = log_nu - torch.logsumexp(Z + u.unsqueeze(2), dim=1)
    u = log_mu - torch.logsumexp(Z + v.unsqueeze(1), dim=2)
    v = log_nu - torch.logsumexp(Z + u.unsqueeze(2), dim=1)
    return Z + u.unsqueeze(2) + v.unsqueeze(1)

def get_ot_plan(features_S, features_T, iters=10):
    device = features_S.device
    sim_1_to_2 = features_S @ features_T.t()
    scores = sim_1_to_2.unsqueeze(0)
    b, m, n = scores.shape
    log_mu = scores.new_zeros((b, m))
    log_nu = scores.new_zeros((b, n))
    Z = log_sinkhorn_iterations(scores, log_mu, log_nu, iters)
    return Z

def log_optimal_transport(scores, alpha, iters: int):
    """ Perform Differentiable Optimal Transport in Log-space for stability"""
    b, m, n = scores.shape
    one = scores.new_tensor(1)
    ms, ns = (m*one).to(scores), (n*one).to(scores)

    bins0 = alpha.expand(b, m, 1)
    bins1 = alpha.expand(b, 1, n)
    alpha = alpha.expand(b, 1, 1)

    couplings = torch.cat([torch.cat([scores, bins0], -1),
                           torch.cat([bins1, alpha], -1)], 1)

    norm = - (ms + ns).log()
    log_mu = torch.cat([norm.expand(m), ns.log()[None] + norm])
    log_nu = torch.cat([norm.expand(n), ms.log()[None] + norm])
    log_mu, log_nu = log_mu[None].expand(b, -1), log_nu[None].expand(b, -1)

    Z = log_sinkhorn_iterations(couplings, log_mu, log_nu, iters)
    Z = Z - norm  # multiply probabilities by M+N
    return Z