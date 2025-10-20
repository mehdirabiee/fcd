import argparse
import os
from typing import Dict, Any
import argparse
import os

def parse_args(default_params) -> argparse.Namespace:
    """Parse command line arguments with validation"""
   
    parser = argparse.ArgumentParser(description='Train and Test Model for FCD Detection.')

    # dataset handling
    parser.add_argument('--data_dir', type=str, required=True, help='Path to dataset root directory')
    parser.add_argument('--split_file', type=str, required=True, help='Path to split file')
    parser.add_argument('--splits', nargs='+', default=['train', 'val', 'test'],
                        help="Which splits to load (any of: train, val, test). Default: all")

    # model and training
    parser.add_argument('--checkpoint_path', type=str, help='Path to model checkpoint')
    parser.add_argument('--save_dir', type=str, help='Output directory')
    parser.add_argument('--model_type', type=str, default=default_params['model_type'])
    parser.add_argument('--gpus', type=str, default='0')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--prefix', type=str, default='')
    parser.add_argument("--emission_tracking", action='store_true', help="Enable carbon emission tracking")

    # param overrides
    parser.add_argument('--kwargs', nargs='*', help='key=value pairs to override params')

    args = parser.parse_args()
    validate_args(args)
    return args

def validate_args(args: argparse.Namespace) -> None:
    """Validate command line arguments"""

    # dataset + split file must exist
    if not os.path.exists(args.data_dir):
        raise ValueError(f"--data_dir not found: {args.data_dir}")
    if not os.path.exists(args.split_file):
        raise ValueError(f"--split_file not found: {args.split_file}")

    # must specify at least one split
    valid_splits = {"train", "val", "test"}
    requested_splits = {s.lower() for s in args.splits}
    invalid = requested_splits - valid_splits
    if invalid:
        raise ValueError(f"Invalid split(s): {invalid}. Must be subset of {valid_splits}")

    # if training, require save_dir and val split
    if "train" in requested_splits:
        if "val" not in requested_splits:
            raise ValueError("--splits must include 'val' when using 'train'")
        if not args.save_dir:
            raise ValueError("--save_dir required when training")

    # resume requires save_dir to exist
    if args.resume and (not args.save_dir or not os.path.exists(args.save_dir)):
        raise ValueError("--save_dir must exist when using --resume")

    # testing requires a checkpoint or training
    if "test" in requested_splits and not (args.checkpoint_path or "train" in requested_splits):
        raise ValueError("--splits includes 'test' but no --checkpoint_path or 'train' split provided")

def parse_kwargs(params: Dict[str, Any], kwargs_list: list) -> Dict[str, Any]:
    """Parse key=value arguments with type preservation"""
    if not kwargs_list:
        return params
        
    for kv in kwargs_list:
        if '=' not in kv:
            raise ValueError(f"Invalid kwargs format: {kv}. Use key=value")
            
        key, value = kv.split('=', 1)
        if key not in params:
            print(f"Warning: Unknown parameter '{key}'")
            continue
            
        try:
            orig_type = type(params[key])
            if orig_type == bool:
                params[key] = value.lower() in {'true', '1', 'yes'}
            else:
                params[key] = orig_type(value)
        except Exception as e:
            raise ValueError(f"Cannot convert '{value}' for '{key}': {e}")
            
    return params
