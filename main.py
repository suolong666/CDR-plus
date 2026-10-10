import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.utils.data as data
from tqdm import tqdm

from config import parse_args
from data_processing import data_param_prepare
from evaluation import get_coverage, get_entropy, get_gini, test_one_batch
from model import CDRPlus


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def collate(batch):
    batch = torch.stack(batch, dim=1)
    return batch[0], batch[1]


def evaluate(args, model, users, ground_truth, mask, topk):
    model.eval()
    with torch.no_grad():
        scores = model.test_forward(range(args.n_head)).cpu() + mask
        metrics = {}
        for k in topk:
            recommendations = torch.topk(scores, k=k).indices.numpy()
            selected = recommendations[users]
            truth = [ground_truth[user] for user in users]
            precision, recall, ndcg = test_one_batch(selected, truth, k)
            precision /= len(users)
            recall /= len(users)
            metrics[k] = {
                "precision": precision,
                "recall": recall,
                "ndcg": ndcg / len(users),
                "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
                "entropy": get_entropy(recommendations),
                "coverage": get_coverage(recommendations, args.n_tail),
                "gini": get_gini(recommendations, args.n_tail),
            }
    return metrics


def train_stage(args, model, loader, valid_users, test_users, valid_mask, test_mask, valid_truth, test_truth, name, checkpoint):
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    best_recall, stale_epochs, epoch = -float("inf"), 0, 0
    while stale_epochs < args.patience:
        epoch += 1
        model.train()
        for nodes, positives in tqdm(loader, desc=f"{name} epoch {epoch}"):
            optimizer.zero_grad()
            loss = model(nodes.to(args.device), positives.to(args.device))
            loss.backward()
            optimizer.step()
        validation = evaluate(args, model, valid_users, valid_truth, valid_mask, [args.top_k[-1]])
        recall = validation[args.top_k[-1]]["recall"]
        print(f"{name} epoch {epoch}: Recall@{args.top_k[-1]}={recall:.5f}, "
              f"CD={model.latest_losses['cd'].item():.4f}, "
              f"Ortho={model.latest_losses['ortho'].item():.4f}, "
              f"Recon={model.latest_losses['recon'].item():.4f}")
        if recall > best_recall:
            best_recall, stale_epochs = recall, 0
            torch.save(model.state_dict(), checkpoint)
        else:
            stale_epochs += 1
    model.load_state_dict(torch.load(checkpoint, map_location=args.device))
    results = evaluate(args, model, test_users, test_truth, test_mask, args.top_k)
    print(f"{name} test metrics: {results}")
    return results


def make_loader(heads, tails, batch_size):
    pairs = torch.from_numpy(np.column_stack((heads, tails)))
    return data.DataLoader(pairs, batch_size=batch_size, shuffle=True, collate_fn=collate)


def main():
    args = parse_args()
    args.device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    setup_seed(args.seed)
    run_dir = Path("save") / args.dataname / time.strftime("%m-%d-%H-%M-%S")
    run_dir.mkdir(parents=True, exist_ok=False)
    heads, tails, valid_users, test_users, alpha, beta, valid_mask, test_mask, valid_truth, test_truth = data_param_prepare(args)
    model = CDRPlus(args, alpha["member"], beta["member"], args.temperature["member"]).to(args.device)

    train_stage(
        args, model, make_loader(heads["member"], tails["member"], args.batch_size["member"]),
        valid_users, test_users, valid_mask, test_mask, valid_truth, test_truth,
        "pretrain", run_dir / "pretrain.pt"
    )
    # The trained embedding table is retained: pre-training initializes fine-tuning.
    model.set_contrastive_data(alpha["tuple"], beta["tuple"], args.temperature["tuple"])
    results = train_stage(
        args, model, make_loader(heads["tuple"], tails["tuple"], args.batch_size["tuple"]),
        valid_users, test_users, valid_mask, test_mask, valid_truth, test_truth,
        "finetune", run_dir / "finetune.pt"
    )
    np.save(run_dir / "final_projected_embeddings.npy", model.projected_embeddings().detach().cpu().numpy())
    print(f"Completed CDR+ run. Outputs: {run_dir}; final metrics: {results}")


if __name__ == "__main__":
    main()
