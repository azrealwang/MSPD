import argparse
import torch
from utils import load_samples,predict
from robustbench import load_model

def parse_args_and_config():
    parser = argparse.ArgumentParser()

    parser.add_argument('--data', help='imagenet or cifar10', type=str, required=True)
    parser.add_argument('--model', help='model name', type=str, required=True)
    parser.add_argument('--input', help='input path', type=str, required=True)
    parser.add_argument('--start_idx', help='start idx', type=int, default=0)
    parser.add_argument('--end_idx', help='end idx, id+1', type=int, default=1000)
    parser.add_argument('--batch_size', help='batch size depends on memory', type=int, default=None)

    args = parser.parse_args()

    return args

def main() -> None:
    # Settings
    args = parse_args_and_config()
    print(args)

    # Load Model
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

    # Load Inputs
    x_test, y_test = load_samples(args.input,args.start_idx,args.end_idx)
    x_test = torch.Tensor(x_test)
    y_test = torch.Tensor(y_test)
    if args.batch_size is None:
        args.batch_size = len(y_test)

    # Compute Accuracy
    predictions = predict(model,x_test,batch_size=args.batch_size)
    accuracy = (predictions.max(1)[1] == y_test).float().mean()
    print(f"Accuracy of test examples: {accuracy}")

if __name__ == "__main__":
    main()