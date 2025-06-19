#!/usr/bin/env python3

import yaml
import argparse

def main():
    parser = argparse.ArgumentParser(description='Generate YAML file with updated parameters.')
    parser.add_argument('--default_yaml', required=True, help='Path to default YAML file')
    parser.add_argument('--output_yaml', required=True, help='Path to output YAML file')
    parser.add_argument('--dataset', help='Dataset value')
    parser.add_argument('--lr', help='Learning rate')
    parser.add_argument('--wd', help='Weight decay')
    parser.add_argument('--epochs', help='Number of epochs')
    parser.add_argument('--save_steps', help='Save steps')
    parser.add_argument('--max_samples', help='Max number of samples')
    parser.add_argument('--model', help='Model name')
    parser.add_argument('--output_dir', help='Output directory')
    parser.add_argument('--per_device_train_batch_size', help='Per device train batch size')
    parser.add_argument('--template', help='chat template')
    parser.add_argument('--gradient_accumulation_steps', help='Gradient accumulation steps')
    parser.add_argument('--eval_dataset', help='Evaluation dataset')
    parser.add_argument('--max_steps', help='Max steps')
    parser.add_argument('--media_dir', help='Media directory')
    args = parser.parse_args()

    # Read default YAML file
    with open(args.default_yaml, 'r') as f:
        config = yaml.safe_load(f)

    # Update the config with provided arguments
    if args.dataset:
        config['dataset'] = args.dataset
    if args.lr:
        config['learning_rate'] = float(args.lr)
    if args.wd:
        config['weight_decay'] = float(args.wd)
    if args.epochs:
        config['num_train_epochs'] = float(args.epochs)
    if args.save_steps:
        config['save_steps'] = int(args.save_steps)
    if args.max_samples:
        config['max_samples'] = int(args.max_samples)
    if args.model:
        config['model_name_or_path'] = args.model
    if args.output_dir:
        config['output_dir'] = args.output_dir
    if args.per_device_train_batch_size:
        config['per_device_train_batch_size'] = int(args.per_device_train_batch_size)
    if args.template:
        config['template'] = args.template
    if args.gradient_accumulation_steps:
        config['gradient_accumulation_steps'] = int(args.gradient_accumulation_steps)
    if args.eval_dataset and args.eval_dataset != "none":
        config['eval_dataset'] = args.eval_dataset
    if args.max_steps:
        if args.max_steps == "none":
            config['max_steps'] = 0
        else:
            config['max_steps'] = int(args.max_steps)
    if args.media_dir:
        config['media_dir'] = args.media_dir
    
    # Write updated YAML file
    with open(args.output_yaml, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)

if __name__ == '__main__':
    main()