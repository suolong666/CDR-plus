import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="CDR+: geometry-aware tripartite recommendation")
    parser.add_argument("--dataname", choices=["mafengwo", "youshu"], default="mafengwo")
    parser.add_argument("--embedding-dim", dest="embedding_dim", type=int, default=64)
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--train-ratio", type=float, default=0.05)
    parser.add_argument("--test-ratio", type=float, default=0.20)
    parser.add_argument("--lambda-ortho", type=float, default=0.5)
    parser.add_argument("--lambda-recon", type=float, default=0.5)
    parser.add_argument("--gamma", type=float, default=1e-4)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--device", default=None, help="e.g. cpu or cuda:0; auto-detect when omitted")
    args = parser.parse_args()
    args.split_ratio = {"train": args.train_ratio, "test": args.test_ratio}
    args.require_data = ["member", "tuple"]
    args.top_k = [10, 20, 30]
    if args.dataname == "mafengwo":
        args.batch_size = {"member": 512, "tuple": 96}
        # CDR+ implementation details: tau=1.0 for pre-training and 0.3
        # for fine-tuning on both Mafengwo and Youshu.
        args.temperature = {"member": 1.0, "tuple": 0.3}
    else:
        args.batch_size = {"member": 512, "tuple": 128}
        args.temperature = {"member": 1.0, "tuple": 0.3}
    return args
