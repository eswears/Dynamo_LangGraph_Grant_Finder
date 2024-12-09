from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field, ConfigDict
from langchain_core.language_models import BaseLLM
from langchain_core.outputs import LLMResult, Generation
import subprocess
import sys
import os
import logging
from pathlib import Path
import json
import time
from dotenv import load_dotenv
from langchain.embeddings.base import Embeddings

load_dotenv()

class BitNetConfig(BaseModel):
    """Configuration for BitNet model"""
    cli_path: str
    model_path: str
    model_size: str = "0.7B"
    quantization: str = "i2_s"
    kernel_type: str = "i2_s"
    threads: int = 4
    ctx_size: int = 2048
    temperature: float = 0.0
    n_predict: int = 128
    n_prompt: int = 512
    quant_embd: bool = False
    use_pretuned: bool = False

class BitNetLLM(BaseLLM):
    """BitNet language model implementation."""
    
    # Define model configuration
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    # Define fields
    model_config_: BitNetConfig = Field(default_factory=lambda: BitNetConfig(cli_path="", model_path=""))
    _logger: Optional[logging.Logger] = None
    
    def __init__(self, **kwargs):
        super().__init__()
        # Set up logging
        self._logger = logging.getLogger(__name__)
        self._logger.setLevel(logging.DEBUG)
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.DEBUG)
        console_formatter = logging.Formatter('%(name)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(console_formatter)
        self._logger.addHandler(console_handler)
        
        # File handler
        file_handler = logging.FileHandler('bitnet_debug.log')
        file_handler.setLevel(logging.DEBUG)
        file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(file_formatter)
        self._logger.addHandler(file_handler)
        
        self._logger.debug(f"Initializing BitNetLLM with config: {kwargs}")
        
        # Initialize config
        self.model_config_ = BitNetConfig(**kwargs)
    
    def _call(self, prompt: str, stop: Optional[List[str]] = None, **kwargs) -> str:
        """Call BitNet model with prompt"""
        max_retries = int(os.getenv('MAX_RETRIES', 3))
        retry_delay = int(os.getenv('RETRY_DELAY', 1))
        
        try:
            for attempt in range(max_retries):
                try:
                    run_inference_path = str(Path(__file__).parent / "run_inference.py")
                    self._logger.debug(f"run_inference.py path: {run_inference_path}")
                    
                    # For document analysis prompts, use hierarchical approach
                    if "documents" in prompt:
                        # Parse the prompt
                        import ast
                        prompt_dict = ast.literal_eval(prompt)
                        query = prompt_dict["query"]
                        
                        # Process each layer's results
                        results = []
                        for layer in ["high", "mid", "low"]:
                            try:
                                layer_prompt = f"Analyze the following {layer}-level information:\n\n{query}"
                                result = self._process_layer(layer_prompt, layer)
                                results.append(result)
                            except Exception as e:
                                self._logger.error(f"Error processing {layer} layer (attempt {attempt + 1}): {str(e)}")
                                if attempt < max_retries - 1:
                                    time.sleep(retry_delay * (2 ** attempt))  # Exponential backoff
                                    continue
                                raise
                        
                        # Combine results from all layers
                        combined_output = self._combine_layer_results(results)
                        return combined_output
                        
                    else:
                        # For non-document prompts, use standard approach
                        command = [
                            sys.executable,
                            run_inference_path,
                            "--cli-path", self.model_config_.cli_path,
                            "--model-path", self.model_config_.model_path,
                            "-p", prompt,
                            "-t", str(self.model_config_.threads),
                            "-c", str(self.model_config_.ctx_size),
                            "--temp", str(self.model_config_.temperature),
                        ]
                        
                        self._logger.debug(f"Running command: {' '.join(command)}")
                        result = subprocess.run(command, capture_output=True, text=True, check=False)
                        
                        if result.returncode != 0:
                            error_msg = (
                                f"Subprocess failed in {__file__}:_call() at line {sys._getframe().f_lineno}\n"
                                f"Return code: {result.returncode}\n"
                                f"Command: {' '.join(command)}\n"
                                f"stdout: {result.stdout}\n"
                                f"stderr: {result.stderr}"
                            )
                            self._logger.error(error_msg)
                            raise RuntimeError(error_msg)
                        
                        return result.stdout
                        
                except Exception as e:
                    if attempt < max_retries - 1:
                        self._logger.warning(f"Attempt {attempt + 1} failed, retrying...")
                        time.sleep(retry_delay * (2 ** attempt))
                        continue
                    raise
        finally:
            # Clean up any large objects
            import gc
            gc.collect()

    def _process_layer(self, prompt: str, layer: str) -> str:
        """Process a single layer with retries"""
        command = [
            sys.executable,
            str(Path(__file__).parent / "run_inference.py"),
            "--cli-path", self.model_config_.cli_path,
            "--model-path", self.model_config_.model_path,
            "-p", prompt,
            "-t", str(self.model_config_.threads),
            "-c", self._get_layer_context_size(layer),
            "--temp", str(self.model_config_.temperature),
        ]
        
        result = subprocess.run(command, capture_output=True, text=True, check=False)
        
        if result.returncode != 0:
            error_msg = (
                f"Layer processing failed in {__file__}:_process_layer() at line {sys._getframe().f_lineno}\n"
                f"Layer: {layer}\n"
                f"Return code: {result.returncode}\n"
                f"Command: {' '.join(command)}\n"
                f"stdout: {result.stdout}\n"
                f"stderr: {result.stderr}"
            )
            raise RuntimeError(error_msg)
        
        return result.stdout

    def _get_layer_context_size(self, layer: str) -> int:
        """Get context size for specific layer"""
        context_sizes = {
            "high": int(os.getenv('HIGH_LEVEL_CONTEXT', 2048)),
            "mid": int(os.getenv('MID_LEVEL_CONTEXT', 1536)),
            "low": int(os.getenv('LOW_LEVEL_CONTEXT', 1024))
        }
        return context_sizes.get(layer, int(os.getenv('LOW_LEVEL_CONTEXT', 1024)))

    def _combine_layer_results(self, results: List[str]) -> str:
        """Combine results from different layers into coherent output"""
        try:
            # Parse JSON results
            parsed_results = []
            for result in results:
                try:
                    parsed = json.loads(result)
                    parsed_results.append(parsed)
                except json.JSONDecodeError:
                    self._logger.warning(f"Could not parse result as JSON: {result}")
                    continue
            
            # Combine information from different layers
            combined = {
                "high_level_summary": parsed_results[0] if len(parsed_results) > 0 else {},
                "mid_level_details": parsed_results[1] if len(parsed_results) > 1 else {},
                "low_level_specifics": parsed_results[2] if len(parsed_results) > 2 else {}
            }
            
            return json.dumps(combined, indent=2)
            
        except Exception as e:
            self._logger.error(f"Error combining layer results: {str(e)}")
            raise

    def _generate(self, prompts: List[str], stop: Optional[List[str]] = None, **kwargs) -> LLMResult:
        """Implementation of BaseLLM's abstract _generate method"""
        try:
            self._logger.debug(f"Generating responses for {len(prompts)} prompts")
            generations = []
            for i, prompt in enumerate(prompts):
                self._logger.debug(f"Processing prompt {i+1}/{len(prompts)}")
                text = self._call(prompt, stop=stop, **kwargs)
                generations.append([Generation(text=text)])
            return LLMResult(generations=generations)
        except Exception as e:
            self._logger.error(f"Error in _generate: {str(e)}", exc_info=True)
            raise

    @property
    def _llm_type(self) -> str:
        """Required by BaseLLM"""
        return "bitnet"

class BitNetEmbeddings(Embeddings):
    def __init__(self, model_config: BitNetConfig, logger: logging.Logger):
        self.model_config = model_config
        self.logger = logger
        self.provider = "bitnet"
        # Default batch size or get from environment
        self._batch_size = int(os.getenv('CHUNK_BATCH_SIZE', 50))

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a list of documents"""
        embeddings = []
        for i, text in enumerate(texts):
            try:
                embedding = self.embed_query(text)
                embeddings.append(embedding)
            except Exception as e:
                self.logger.error(f"Error embedding document {i}: {str(e)}")
                # Return empty embedding of correct dimension to maintain consistency
                embeddings.append([0.0] * 1536)
        return embeddings

    def embed_query(self, text: str) -> List[float]:
        """Generate embedding for a single text"""
        try:
            command = [
                sys.executable,
                str(Path(__file__).parent / "run_inference.py"),
                "--cli-path", self.model_config.cli_path,
                "--model-path", self.model_config.model_path,
                "-p", text,
                "-t", str(self.model_config.threads),
                "--embedding",  # Request embedding output
                "--embedding-dim", "1536"  # Match OpenAI embedding dimension
            ]
            
            self.logger.debug(f"Running embedding command: {' '.join(command)}")
            result = subprocess.run(command, capture_output=True, text=True, check=False)
            
            if result.returncode != 0:
                error_msg = (
                    f"Embedding generation failed:\n"
                    f"Return code: {result.returncode}\n"
                    f"Command: {' '.join(command)}\n"
                    f"stdout: {result.stdout}\n"
                    f"stderr: {result.stderr}"
                )
                self.logger.error(error_msg)
                raise RuntimeError(error_msg)
            
            # Parse embedding from output
            embedding = json.loads(result.stdout)
            if not isinstance(embedding, list) or len(embedding) != 1536:
                raise ValueError(f"Invalid embedding format. Expected list of 1536 floats, got: {type(embedding)}")
            
            return embedding
            
        except Exception as e:
            self.logger.error(f"Error generating embedding: {str(e)}")
            raise
