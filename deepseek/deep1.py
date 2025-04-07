import os
import torch
import logging
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    pipeline,
    StoppingCriteria,
    StoppingCriteriaList
)
from huggingface_hub import snapshot_download
from langchain_community.llms import HuggingFacePipeline

# Configuration
os.environ["TOKENIZERS_PARALLELISM"] = "false"
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("deepseek_engine")

MODEL_NAME = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
CACHE_DIR = "/home/ubuntu/deepseek_models"
MAX_INPUT_TOKENS = 4096  # Reduced from 7500
MAX_NEW_TOKENS = 1024    # Reduced from 3000
DEFAULT_TEMPERATURE = 0.6

class StopOnTokens(StoppingCriteria):
    def __init__(self, stop_token_ids):
        self.stop_token_ids = stop_token_ids

    def __call__(self, input_ids, scores, **kwargs):
        return input_ids[0][-1] in self.stop_token_ids

class DeepSeekLangChain:
    def __init__(self):
        self.model, self.tokenizer = self.load_model()
        self.pipeline = self.create_pipeline()

    def load_model(self):
        """Optimized model loading with caching"""
        if not os.path.exists(CACHE_DIR):
            snapshot_download(
                repo_id=MODEL_NAME,
                local_dir=CACHE_DIR,
                ignore_patterns=["*.md", "*.txt"]
            )

        tokenizer = AutoTokenizer.from_pretrained(
            CACHE_DIR,
            padding_side="left",
            model_max_length=MAX_INPUT_TOKENS + MAX_NEW_TOKENS
        )
        tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            CACHE_DIR,
            device_map="auto",
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True
        )
        return model, tokenizer

    def create_pipeline(self):
        """Optimized pipeline configuration"""
        stop_tokens = [self.tokenizer.eos_token_id]
        stopping_criteria = StoppingCriteriaList([StopOnTokens(stop_tokens)])

        return pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=DEFAULT_TEMPERATURE,
            top_p=0.9,
            top_k=50,
            do_sample=True,
            stopping_criteria=stopping_criteria,
            device_map="auto",
            batch_size=1
        )

    def generate(self, prompt, system_prompt="You are a helpful AI assistant."):
        try:
            # Format the prompt
            formatted_prompt = f"[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n\n{prompt} [/INST]"
            
            # Log token counts before generation
            input_tokens = self.tokenizer(formatted_prompt, return_tensors="pt").input_ids
            input_token_count = input_tokens.shape[1]
            
            logger.info(f"Input prompt tokens: {input_token_count}")
            logger.info(f"System prompt: {system_prompt}")
            logger.info(f"User prompt: {prompt}")

            # Warm-up pass (removes initial lag)
            if not hasattr(self, '_warmed_up'):
                self.pipeline("Warm-up", max_new_tokens=10)
                self._warmed_up = True

            # Generate response
            result = self.pipeline(
                formatted_prompt,
                max_new_tokens=MAX_NEW_TOKENS,
                num_return_sequences=1,
                pad_token_id=self.tokenizer.eos_token_id
            )[0]['generated_text']
            
            # Log output tokens
            full_response_tokens = self.tokenizer(result, return_tensors="pt").input_ids
            output_token_count = full_response_tokens.shape[1] - input_token_count
            
            logger.info(f"Output tokens generated: {output_token_count}")
            
            return result.split("[/INST]")[-1].strip()

        except Exception as e:
            logger.error(f"Generation failed: {e}")
            raise

# Singleton instance
deepseek_model = DeepSeekLangChain()
