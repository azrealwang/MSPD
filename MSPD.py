import os
import torch
import argparse
import math
from time import time
from utils import save_all_images,load_samples,predict
from robustbench.utils import load_model

def parse_args_and_config():
    parser = argparse.ArgumentParser()
    
    # Attack Settings
    parser.add_argument('--reversion', help='if this is reversion attack', action='store_true')
    parser.add_argument('--eps', help='budget', type=float, default=8)
    parser.add_argument('--N', help='average N gradients for smoothing', type=int, default='20')
    parser.add_argument('--LR', help='epoch learning rate', type=float, default=1.0)
    parser.add_argument('--batch_size', help='batch size depends on memory', type=int, default=1)
    # Model Settings
    parser.add_argument('--data', help='cifar10 or imagenet', type=str, required=True)
    parser.add_argument('--backbone', help='backbone model', type=str, required=True)
    parser.add_argument('--classifier', help='GT, decouple, couple', type=str, default='decouple')
    # Input and Output
    parser.add_argument('--input', help='input path', type=str, required=True)
    parser.add_argument('--output', help='output path', type=str, required=True)
    parser.add_argument('--start_idx', help='start idx', type=int, default=0)
    parser.add_argument('--end_idx', help='end idx, id+1', type=int, default=1000)

    args = parser.parse_args()

    return args


def SmoothGrad(model, x, y, epsilon, alpha, K, N=20, forward=False):
    """
    SmoothGrad-PGD: Run PGD independently on N noisy initializations, then average the final outputs.

    Args:
        model   : torch.nn.Module
        x       : input tensor (B, C, H, W)
        y       : labels tensor (B,)
        epsilon : L-infinity perturbation bound
        alpha   : PGD step size
        K       : number of PGD steps
        N       : number of noisy trajectories (EOT samples)
        forward : if True, targeted (minimize loss); if False, untargeted (maximize loss)

    Returns:
        x_adv   : adversarial examples (B, C, H, W)
    """
    import torch.nn.functional as F
    model.eval().to(x.device)
    B, C, H, W = x.shape

    # Step 1: Repeat input x for N noisy copies → (N, B, C, H, W)
    x_repeat = x.unsqueeze(0).repeat(N, 1, 1, 1, 1)

    # Step 2: Add uniform noise ξ_i ∼ U(-ε, ε) for random start
    noise = torch.empty_like(x_repeat).uniform_(-epsilon, epsilon)
    x_adv_all = (x_repeat + noise).clamp(0, 1).detach()

    # Step 3: Run PGD independently for each noisy copy
    for _ in range(K):
        x_adv_all = x_adv_all.clone().detach().requires_grad_(True)  # (N, B, C, H, W)

        # Flatten for parallel processing: (N * B, C, H, W)
        x_flat = x_adv_all.view(-1, C, H, W)
        y_flat = y.repeat(N)

        logits = model(x_flat)
        loss = F.cross_entropy(logits, y_flat, reduction='sum')
        # print(loss)

        grad = torch.autograd.grad(loss, x_adv_all, retain_graph=False, create_graph=False)[0]
        direction = -grad.sign() if forward else grad.sign()

        # PGD step
        x_adv_all = x_adv_all + alpha * direction

        # Project back to ε-ball around clean input x
        x_center = x.unsqueeze(0).repeat(N, 1, 1, 1, 1)
        x_adv_all = torch.max(torch.min(x_adv_all, x_center + epsilon), x_center - epsilon)
        x_adv_all = x_adv_all.clamp(0, 1).detach()

    # Step 4: Average final results over N → (B, C, H, W)
    x_adv = x_adv_all.mean(dim=0)

    return x_adv


def main() -> None:
    # Settings
    args = parse_args_and_config()
    print(args)
    eps = args.eps/255
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Loading data
    x_test, y_test_GT = load_samples(args.input,args.start_idx,args.end_idx)
    x_test = torch.Tensor(x_test)
    y_test_GT = torch.Tensor(y_test_GT).long()

    # Loading backbone model
    if args.data == 'cifar10':
        backbone = load_model(args.backbone, dataset="cifar10", threat_model="Linf")
    elif args.data == 'imagenet':
        backbone = load_model(args.backbone, dataset="imagenet", threat_model="Linf")
    else:
        raise ValueError("Unsupported dataset")

    # Labeling
    if args.classifier == 'GT':
        y_test = y_test_GT
    else:
        if args.classifier == 'decouple':
            if args.data == 'cifar10':
                classifier = load_model('Hendrycks2020AugMix_ResNeXt', dataset="cifar10", threat_model="corruptions")
            elif args.data == 'imagenet':
                classifier = load_model('Tian2022Deeper_DeiT-B', dataset="imagenet", threat_model="corruptions")
        elif args.classifier == 'couple':
            classifier = backbone
        predictions = predict(classifier,x_test,batch_size=args.batch_size)
        y_test = predictions.max(1)[1]
    
    # MSPD
    if not os.path.exists(args.output):
        os.makedirs(args.output)
    batches = math.ceil(len(y_test)/args.batch_size)
    x_test_def = torch.Tensor([])
    start_time = time()

    for b in range(batches):
        print(f'Processing {b+1}th Batch....')
        if b == batches-1:
            idx = range(b*args.batch_size,len(y_test))
        else:
            idx = range(b*args.batch_size,(b+1)*args.batch_size)
        x_batch = x_test[idx].clone().to(device)
        y_batch = y_test[idx].clone().to(device)
        # Forward one epoch
        x_batch_FK = SmoothGrad(backbone, x_batch, y_batch, epsilon=eps, alpha=eps, K=1, N=args.N, forward=True)
        x_batch_F = x_batch + args.LR * (x_batch_FK - x_batch)
        x_batch_F = torch.max(torch.min(x_batch_F, x_batch + eps), x_batch - eps)
        x_batch_F = x_batch_F.clamp(0, 1)
        # Backward one epoch
        x_batch_advK = SmoothGrad(backbone, x_batch_F, y_batch, epsilon=eps, alpha=eps, K=1, N=args.N, forward=False)
        x_batch_B = x_batch_F - args.LR * (x_batch_advK - x_batch_F)
        x_batch_B = torch.max(torch.min(x_batch_B, x_batch + eps), x_batch - eps)
        x_batch_B = x_batch_B.clamp(0, 1)
        # defense or reversion
        if args.reversion:
            x_batch_output = x_batch * 2 - x_batch_B
        else:
            x_batch_output = x_batch_B.clone()
        # Save preemptive examples
        save_all_images(x_batch_output,y_test_GT[idx],args.output,args.start_idx+b*args.batch_size)
        x_test_def = torch.cat((x_test_def, x_batch_output.cpu()), 0)

    end_time = time()
    time_cost = end_time - start_time # record time cost
    d_linf = (x_test_def-x_test).abs().max()*255
    print(f"Time cost: {time_cost}s; Linf Distance: {d_linf}")

if __name__ == "__main__":
    main()