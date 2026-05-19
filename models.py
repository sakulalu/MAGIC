import time
import math
import numpy as np
import torch

from dataprocessing import get_multiview_data
from loss import MultiPathConsensusLoss


def pre_train(network_model, mv_data, batch_size, epochs, optimizer):
    """Pretrain view-specific autoencoders with reconstruction loss only."""
    start_time = time.time()
    loader, num_views, num_samples, _ = get_multiview_data(mv_data, batch_size)
    criterion = torch.nn.MSELoss()

    for epoch in range(epochs):
        total_loss = 0.0
        network_model.train()
        network_model.set_impute(False)

        for sub_data_views, _ in loader:
            reconstructions, _ = network_model.reconstruct_only(sub_data_views)

            loss = 0.0
            for v in range(num_views):
                mask = (
                    (sub_data_views[v].abs().sum(dim=1) > 1e-6)
                    & (~torch.isnan(sub_data_views[v]).any(dim=1))
                ).unsqueeze(1).float()
                loss = loss + criterion(sub_data_views[v] * mask, reconstructions[v] * mask)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if epoch % 10 == 0 or epoch == epochs - 1:
            print(f"Pre-training, epoch {epoch + 1}, loss={total_loss / num_samples:.7f}")

    print("Pre-training finished. Time: {:.2f}s".format(time.time() - start_time))


def contrastive_train(network_model, mv_data, alignment_loss, batch_size,
                      beta, gamma, temperature_l, normalized, epoch, optimizer,
                      epoch_open_ot=30, epoch_open_knn=50,
                      impute_th_start=0.65, impute_th_end=0.55):
    """Train MAGIC with multi-path consensus, semantic alignment, and reconstruction."""
    network_model.train()
    loader, num_views, num_samples, _ = get_multiview_data(mv_data, batch_size)
    criterion = torch.nn.MSELoss()

    if epoch < epoch_open_ot:
        network_model.set_impute(False)
    else:
        network_model.set_impute(True)
        network_model.set_impute_knn(epoch >= epoch_open_knn)
        ratio = min(1.0, max(0.0, (epoch - epoch_open_ot) / max(1, epoch_open_knn - epoch_open_ot)))
        threshold = impute_th_start * (1.0 - ratio) + impute_th_end * ratio
        network_model.set_impute_conf(threshold)

    consensus_loss = MultiPathConsensusLoss()
    total_loss = 0.0

    for sub_data_views, _ in loader:
        probs, reconstructions, _, _, view_sims, aug = network_model(sub_data_views, return_aug=True)

        loss = gamma * consensus_loss(aug)["loss_total"]

        sim_idx = 0
        for i in range(num_views):
            for j in range(i + 1, num_views):
                sim = view_sims[sim_idx]
                loss = loss + sim * beta * alignment_loss.forward_label(
                    probs[i], probs[j], temperature_l, normalized
                )
                loss = loss + sim * beta * alignment_loss.forward_prob(probs[i], probs[j])
                sim_idx += 1

        for v in range(num_views):
            mask = (
                (sub_data_views[v].abs().sum(dim=1) > 1e-6)
                & (~torch.isnan(sub_data_views[v]).any(dim=1))
            ).unsqueeze(1).float()
            loss = loss + criterion(sub_data_views[v] * mask, reconstructions[v] * mask)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    loss_per_sample = total_loss / num_samples
    print(
        f"Contrastive training, epoch {epoch + 1}, "
        f"loss={loss_per_sample:.7f}, "
        f"impute={network_model.impute_enabled}, "
        f"knn={network_model.impute_enable_knn}"
    )

    if (epoch + 1) % 10 == 0:
        acc, nmi, pur, ari = valid(network_model, mv_data, batch_size)
        print(f"[Eval] epoch {epoch + 1}: ACC={acc:.4f} NMI={nmi:.4f} PUR={pur:.4f} ARI={ari:.4f}")

    return total_loss


def inference(network_model, mv_data, batch_size):
    network_model.eval()
    loader, num_views, _, _ = get_multiview_data(mv_data, batch_size)
    pred_vectors = [[] for _ in range(num_views)]
    soft_vector, labels_vector = [], []

    for sub_data_views, sub_labels in loader:
        with torch.no_grad():
            probs, _, _, _, _ = network_model(sub_data_views)

            weights = []
            for p in probs:
                entropy = -(p * p.clamp_min(1e-12).log()).sum(dim=1).mean()
                weights.append(1.0 - entropy.item() / math.log(p.size(1) + 1e-12))

            weights = torch.tensor(weights, device=probs[0].device, dtype=probs[0].dtype)
            weights = weights / (weights.sum() + 1e-12)

            log_probs = torch.stack([torch.log(p + 1e-12) for p in probs], dim=0)
            fused_logits = (weights.view(-1, 1, 1) * log_probs).sum(dim=0)
            fused_probs = torch.softmax(fused_logits, dim=1)

        for v in range(num_views):
            pred_vectors[v].extend(torch.argmax(probs[v], dim=1).cpu().numpy())

        soft_vector.extend(fused_probs.cpu().numpy())
        labels_vector.extend(sub_labels)

    labels_vector = np.asarray(labels_vector).reshape(len(soft_vector))
    total_pred = np.argmax(np.asarray(soft_vector), axis=1)
    pred_vectors = [np.asarray(pred) for pred in pred_vectors]
    return total_pred, pred_vectors, labels_vector


def valid(network_model, mv_data, batch_size):
    from metrics import calculate_metrics

    total_pred, _, labels_vector = inference(network_model, mv_data, batch_size)
    acc, nmi, pur, ari = calculate_metrics(labels_vector, total_pred)
    print(f"ACC = {acc:.4f} NMI = {nmi:.4f} PUR = {pur:.4f} ARI = {ari:.4f}")
    return acc, nmi, pur, ari
