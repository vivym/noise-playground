import argparse

import torch
import pandas as pd
from torchvision import datasets, transforms

from noise_playground import models
from noise_playground.metrics import accuracy, AverageMeter


def build_dataloader_and_model(
    dataset_name: str,
    batch_size: int,
    num_workers: int,
    model_name: str,
    dataset_root: str = "./data",
):
    if dataset_name == "cifar10":
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                (0.4914, 0.4822, 0.4465),
                (0.2023, 0.1994, 0.201),
            ),
        ])
        dataset = datasets.CIFAR10(
            root=dataset_root,
            train=False,
            download=True,
            transform=transform,
        )
    elif dataset_name == "cifar100":
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                (0.507, 0.4865, 0.4409),
                (0.2673, 0.2564, 0.2761),
            ),
        ])
        dataset = datasets.CIFAR100(
            root=dataset_root,
            train=False,
            download=True,
            transform=transform,
        )
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    model_fn = getattr(models, f"{dataset_name}_{model_name}")
    model = model_fn(pretrained=True)

    return dataloader, model


@torch.no_grad()
def inference(dataloader, model, device):
    acc1 = AverageMeter("Acc@1", ":6.2f")
    acc5 = AverageMeter("Acc@5", ":6.2f")
    labels_list = []
    for (inputs, labels) in dataloader:
        labels_list.append(labels)
        inputs = inputs.to(device)
        labels = labels.to(device)

        outputs = model(inputs)

        acc1_, acc5_ = accuracy(outputs, labels, topk=(1, 5))
        acc1.update(acc1_.item(), inputs.shape[0])
        acc5.update(acc5_.item(), inputs.shape[0])

    return labels_list


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="vgg16")
    parser.add_argument("--dataset", type=str, default="cifar10")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    device = torch.device("cpu" if args.cpu else "cuda")

    dataloader, model = build_dataloader_and_model(
        dataset_name=args.dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        model_name=args.model,
    )
    model.eval()
    model.to(device)
    model.register_importance_hooks()

    labels_list = inference(dataloader, model, device)
    labels = torch.cat(labels_list, dim=0)

    importances = model.get_importances()
    importances = [
        torch.cat(scores, dim=0)
        for scores in importances
    ]

    scores_per_class = []
    for c in range(10):
        mask = labels == c
        scores_per_class.append([scores[mask].mean(0) for scores in importances])

    results = []
    for layer_idx in range(len(importances)):
        scores = torch.stack([
            scores_per_class[c][layer_idx]
            for c in range(10)
        ], dim=0)
        scores = scores.max(0)[0]
        results.append(scores)

    max_rows = max(map(lambda x: x.shape[0], results))

    df = pd.DataFrame(index=range(max_rows))
    for idx, layer_scores in enumerate(results):
        layer_scores = layer_scores.cpu().sort(descending=True)[0].numpy()
        df[f"Layer {idx + 1}"] = pd.Series(layer_scores)

    df.to_excel(excel_writer="data/importances2.xlsx")


if __name__ == "__main__":
    main()
