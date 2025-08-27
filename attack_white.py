import os
import torch
import argparse
from time import time
from utils import save_all_images,load_samples,predict
from robustbench import load_model
from autoattack import AutoAttack

def parse_args_and_config():
    parser = argparse.ArgumentParser()
    
    # Attack Settings
    parser.add_argument('--attack', help='AutoAttack', type=str, required=True)
    parser.add_argument('--version', help='standard, rand for autoattack only', type=str, default='standard')
    parser.add_argument('--norm', help='Linf, L2', type=str, default='linf')
    parser.add_argument('--eps', help='budget = eps/255', type=float, default=8)
    parser.add_argument('--batch_size', help='batch size', type=int, default=1)
    # Model Settings
    parser.add_argument('--data', help='cifar10 or imagenet', type=str, required=True)
    parser.add_argument('--model', help='target model', type=str, required=True)
    # Input & Output
    parser.add_argument('--input', help='input path', type=str, required=True)
    parser.add_argument('--output', help='output path', type=str, required=True)
    parser.add_argument('--start_idx', help='start idx', type=int, default=0)
    parser.add_argument('--end_idx', help='end idx, id+1', type=int, default=1000)

    args = parser.parse_args()

    return args

def main() -> None:
    # Settings
    args = parse_args_and_config()
    print(args)
    eps = args.eps/255 # e.g., if expecting L2=0.5, args.eps should be set at args.eps=127.5

    # Loading model
    if args.data == 'cifar10':
        model = load_model(args.model, dataset="cifar10", threat_model="Linf")
    elif args.data == 'imagenet':
        if args.model == 'ViT':
            import timm
            model = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=1000)
        elif args.model == 'VGG':
            import timm
            model = timm.create_model('vgg19_bn.tv_in1k', pretrained=True, num_classes=1000)
        else:
            model = load_model(args.model, dataset="imagenet", threat_model="Linf")
    else:
        raise ValueError("Unsupported dataset")

    # Load examples
    x_test, y_test = load_samples(args.input,args.start_idx,args.end_idx)
    x_test = torch.Tensor(x_test)
    y_test = torch.Tensor(y_test).long()
    if args.batch_size is None:
        args.batch_size = len(y_test)

    # Clean accuracy
    predictions = predict(model,x_test,batch_size=args.batch_size)
    accuracy = (predictions.max(1)[1] == y_test).float().mean()
    print(f"Clean accuracy of benign test examples: {accuracy}")

    # Attack
    start_time = time()
    if args.attack == 'AutoAttack':
        attack = AutoAttack(model,norm=args.norm,eps=eps,version=args.version)
        x_test_adv = attack.run_standard_evaluation(x_test,y_test,bs=args.batch_size).cpu()
    else:
        raise ValueError("unsupported attack")
    end_time = time()
    time_cost = end_time-start_time # record time cost
    print(f"Time cost: {time_cost}s")
    # Save adversarial examples
    if not os.path.exists(args.output):
        os.makedirs(args.output)
    save_all_images(x_test_adv,y_test,args.output,args.start_idx)

    # Robust accuracy
    d_linf = (x_test_adv-x_test).abs().max()*255
    predictions = predict(model,x_test_adv,batch_size=args.batch_size)
    accuracy = (predictions.max(1)[1] == y_test).float().mean() 
    print(f"Robust accuracy on test examples: {accuracy}; Linf Distance: {d_linf}")
    x_test_adv_load, _= load_samples(args.output,args.start_idx,args.end_idx)
    x_test_adv_load = torch.Tensor(x_test_adv_load)
    load_err = (x_test_adv_load-x_test_adv).abs().max()*255
    predictions = predict(model,x_test_adv_load,batch_size=args.batch_size)
    accuracy = (predictions.max(1)[1] == y_test).float().mean()  
    print(f"Robust accuracy on load examples: {accuracy}; Load error: {load_err}")

if __name__ == "__main__":
    main()