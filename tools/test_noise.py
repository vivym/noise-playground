import argparse

from tqdm import tqdm
import torch
from torchvision import datasets, transforms

from noise_playground import models
from noise_playground.metrics import accuracy, AverageMeter


def build_dataloader_and_model(
    dataset_name: str,
    batch_size: int,
    num_workers: int,
    model_name: str,
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
            root="./data",
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
            root="./data",
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
    for (inputs, labels) in dataloader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        outputs = model(inputs)

        acc1_, acc5_ = accuracy(outputs, labels, topk=(1, 5))
        acc1.update(acc1_.item(), inputs.shape[0])
        acc5.update(acc5_.item(), inputs.shape[0])

    return acc1.avg, acc5.avg


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

    for layer_idx in range(model.num_layers):
        for noise_type in ["feature_noise", "weight_noise"]:
            model.noise_layer_idx = layer_idx
            model.noise_factor = 1.0
            if noise_type == "feature_noise":
                model.apply_feature_noise = True
            if noise_type == "weight_noise":
                model.apply_weight_noise = True

            acc1, acc5 = inference(dataloader, model, device)
            print(noise_type, "\t", layer_idx, "\t", acc1, acc5)
        print("-" * 50)


if __name__ == "__main__":
    main()
