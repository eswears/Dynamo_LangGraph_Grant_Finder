import subprocess
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
from langchain_core.language_models.llms import BaseLLM
from langchain_core.outputs import Generation, LLMResult
from pydantic import Field, PrivateAttr

class BitNetLLM(BaseLLM):
    """BitNet implementation of LangChain's LLM interface"""
    
    # Define fields using Pydantic Field
    model_path: str = Field(description="Path to the BitNet model")
    model_size: str = Field(default="0.7B", description="Model size (0.7B, 3.3B, or 8.0B)")
    quantization: str = Field(default="i2_s", description="Quantization type (i2_s or tl1)")
    kernel_type: str = Field(default="i2_s", description="Kernel type (i2_s, tl1, or tl2)")
    threads: int = Field(default=4, description="Number of threads to use")
    ctx_size: int = Field(default=2048, description="Context size")
    temperature: float = Field(default=0.0, description="Temperature for sampling")
    n_predict: int = Field(default=128, description="Number of tokens to predict")
    n_prompt: int = Field(default=512, description="Number of prompt tokens")
    quant_embd: bool = Field(default=False, description="Whether to quantize embeddings")
    use_pretuned: bool = Field(default=False, description="Whether to use pretuned kernels")
    
    # Use PrivateAttr for non-serializable attributes
    _logger: logging.Logger = PrivateAttr()
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._logger = logging.getLogger(__name__)
    
    def _call(self, prompt: str, stop: Optional[List[str]] = None, **kwargs) -> str:
        """Implementation of BaseLLM's abstract _call method"""
        try:
            # Add JSON formatting instruction to prompt
            formatted_prompt = f"""
            {prompt}
            
            IMPORTANT: Your response MUST be valid JSON. Format your response like this:
            ```json
            {{
                "action": "the action to take",
                "action_input": "the input to the action",
                "thought": "your reasoning"
            }}
            ```
            """
            
            self._logger.info(f"BitNetLLM input prompt: {formatted_prompt}")
            
            cmd = [
                "python", "run_inference.py",
                "-m", str(self.model_path),
                "-p", formatted_prompt,
                "-t", str(self.threads),
                "-c", str(self.ctx_size),
                "-temp", str(self.temperature),
                "-n", str(self.n_predict)
            ]
            
            self._logger.info(f"Running command: {' '.join(cmd)}")
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode != 0:
                self._logger.error(f"BitNetLLM subprocess error: {result.stderr}")
                raise RuntimeError(f"BitNetLLM subprocess failed: {result.stderr}")
            
            output = result.stdout.strip()
            self._logger.info(f"BitNetLLM raw output: {output}")
            
            return output
            
        except Exception as e:
            self._logger.error(f"BitNetLLM error: {str(e)}", exc_info=True)
            raise

    def _generate(self, prompts: List[str], stop: Optional[List[str]] = None, **kwargs) -> LLMResult:
        """Implementation of BaseLLM's abstract _generate method"""
        try:
            generations = []
            for prompt in prompts:
                text = self._call(prompt, stop=stop, **kwargs)
                generations.append([Generation(text=text)])
            return LLMResult(generations=generations)
        except Exception as e:
            self._logger.error(f"BitNetLLM generate error: {str(e)}", exc_info=True)
            raise

    @property
    def _llm_type(self) -> str:
        """Required by BaseLLM"""
        return "bitnet"
