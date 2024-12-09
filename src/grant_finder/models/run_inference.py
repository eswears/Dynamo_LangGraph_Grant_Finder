#!/usr/bin/env python3
import os
import sys
import argparse
import subprocess
import logging.handlers
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

def setup_logging():
    """Setup logging to both file and console"""
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    log_file = log_dir / "bitnet_inference.log"
    logger = logging.Logger("bitnet_inference")
    
    file_handler = logging.FileHandler(log_file)
    console_handler = logging.StreamHandler(sys.stdout)
    
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    logger.setLevel(logging.INFO)
    
    return logger

def run_inference(args):
    """Run inference using the BitNet model"""
    logger = setup_logging()
    try:
        cli_path = Path(args.cli_path)
        model_path = Path(args.model_path)
        
        if not cli_path.exists():
            raise RuntimeError(f"llama-cli not found at: {cli_path}")
        if not model_path.exists():
            raise RuntimeError(f"Model file not found at: {model_path}")

        logger.info(f"Using llama-cli at: {cli_path}")
        logger.info(f"Using model at: {model_path}")

        # Build command with layer-specific context size
        if "high-level" in args.prompt:
            ctx_size = int(os.getenv('HIGH_LEVEL_CONTEXT', 2048))
        elif "mid-level" in args.prompt:
            ctx_size = int(os.getenv('MID_LEVEL_CONTEXT', 1536))
        else:
            ctx_size = int(os.getenv('LOW_LEVEL_CONTEXT', 1024))

        command = [
            str(cli_path),
            '-m', str(model_path),
            '-t', str(os.getenv('THREADS', 4)),
            '-p', args.prompt,
            '-ngl', '0',
            '-c', str(ctx_size),
            '--temp', str(args.temperature),
            "-b", "1"
        ]

        if args.embedding:
            command.extend([
                '--embedding',
                '--embedding-dim', str(args.embedding_dim)
            ])
        else:
            command.extend([
                '-n', str(args.n_predict)
            ])

        logger.info(f"Running command: {' '.join(command)}")
        result = subprocess.run(command, capture_output=True, text=True)
        
        logger.info(f"Return code: {result.returncode}")
        logger.info(f"stdout: {result.stdout}")
        logger.info(f"stderr: {result.stderr}")
        
        if result.returncode != 0:
            error_msg = f"Command failed with return code {result.returncode}\nstderr: {result.stderr}\nstdout: {result.stdout}"
            logger.error(error_msg)
            raise RuntimeError(error_msg)
            
        if not result.stdout:
            error_msg = "No output from command"
            logger.error(error_msg)
            raise RuntimeError(error_msg)
            
        print(result.stdout)
        return 0

    except Exception as e:
        logger.error(f"Error during inference: {str(e)}")
        return 1
    finally:
        # Clean up any large objects
        import gc
        gc.collect()

def setup_args():
    parser = argparse.ArgumentParser(description='Run BitNet inference')
    parser.add_argument("-m", "--model-path", type=str, required=True,
                      help="Path to model file")
    parser.add_argument("--cli-path", type=str, required=True,
                      help="Path to llama-cli executable")
    parser.add_argument("-n", "--n-predict", type=int, default=128,
                      help="Number of tokens to predict")
    parser.add_argument("-p", "--prompt", type=str, required=True,
                      help="Prompt to generate text from")
    parser.add_argument("-t", "--threads", type=int, default=4,
                      help="Number of threads to use")
    parser.add_argument("-c", "--ctx-size", type=int, default=2048,
                      help="Size of the prompt context")
    parser.add_argument("-temp", "--temperature", type=float, default=0.0,
                      help="Temperature for sampling")
    parser.add_argument("--embedding", action="store_true",
                      help="Generate embeddings instead of text")
    parser.add_argument("--embedding-dim", type=int, default=1536,
                      help="Dimension of generated embeddings")
    return parser.parse_args()

def main():
    args = setup_args()
    return run_inference(args)

if __name__ == "__main__":
    sys.exit(main()) 