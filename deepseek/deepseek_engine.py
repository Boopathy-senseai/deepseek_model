import os
import torch
import logging
import re
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    pipeline,
    StoppingCriteria,
    StoppingCriteriaList
)
from huggingface_hub import snapshot_download
from langchain_community.llms import HuggingFacePipeline
from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain
from typing import List

# Environment Configuration
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:256"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Logger Configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("deepseek_engine")

# Constants
MODEL_NAME = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
CACHE_DIR = "/home/ubuntu/deepseek_models"
MAX_INPUT_TOKENS = 7500
MAX_NEW_TOKENS = 3000
DEFAULT_TEMPERATURE = 0.6

class StopOnTokens(StoppingCriteria):
    def __init__(self, stop_token_ids: List[int]):
        self.stop_token_ids = stop_token_ids

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        return any(stop_id in input_ids[0] for stop_id in self.stop_token_ids)

class DeepSeekLangChain:
    def __init__(self):
        self.model_name = MODEL_NAME
        self.cache_dir = CACHE_DIR
        os.makedirs(self.cache_dir, exist_ok=True)
        self.download_model()
        self.initialize_pipeline()
        self.create_chain()

    def download_model(self):
        try:
            config_path = os.path.join(self.cache_dir, "config.json")
            if not os.path.exists(config_path):
                logger.info(f"Downloading model: {self.model_name}")
                snapshot_download(
                    repo_id=self.model_name,
                    local_dir=self.cache_dir,
                    ignore_patterns=["*.md", "*.txt"],
                    local_dir_use_symlinks=False
                )
        except Exception as e:
            logger.error(f"Model download failed: {e}")
            raise

    def initialize_pipeline(self):
        try:
            # Initialize tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.cache_dir,
                trust_remote_code=True,
                padding_side="left",
                model_max_length=MAX_INPUT_TOKENS + MAX_NEW_TOKENS
            )

            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            # Load model without quantization first to check
            model = AutoModelForCausalLM.from_pretrained(
                self.cache_dir,
                device_map="auto",
                torch_dtype=torch.float16,
                trust_remote_code=True,
                low_cpu_mem_usage=True
            )

            # Configure stopping criteria
            stop_token_ids = [self.tokenizer.eos_token_id]
            stopping_criteria = StoppingCriteriaList([StopOnTokens(stop_token_ids)])

            # Create transformers pipeline with explicit settings
            text_pipeline = pipeline(
                "text-generation",
                model=model,
                tokenizer=self.tokenizer,
                max_new_tokens=MAX_NEW_TOKENS,
                temperature=DEFAULT_TEMPERATURE,
                top_p=0.9,
                top_k=40,
                repetition_penalty=1.15,
                stopping_criteria=stopping_criteria,
                pad_token_id=self.tokenizer.pad_token_id,
                device_map="auto",
                batch_size=1,
                return_full_text=False
            )

            # Wrap in LangChain pipeline using updated import
            self.llm = HuggingFacePipeline(pipeline=text_pipeline)

        except Exception as e:
            logger.error(f"Pipeline initialization failed: {e}")
            raise

    def create_chain(self):
        template = """[INST] <<SYS>>
{system_prompt}
<</SYS>>

{user_prompt} [/INST]"""

        self.prompt = PromptTemplate(
            template=template,
            input_variables=["system_prompt", "user_prompt"]
        )

        self.chain = LLMChain(
            llm=self.llm,
            prompt=self.prompt,
            verbose=False
        )

    def generate(self, prompt: str, system_prompt: str = "You are a helpful AI assistant.", 
                temperature: float = DEFAULT_TEMPERATURE, max_new_tokens: int = MAX_NEW_TOKENS) -> str:
        try:
            if not prompt.strip():
                raise ValueError("User prompt is empty.")
            
            logger.info(f"System prompt: {system_prompt}")
            logger.info(f"User prompt: {prompt}")

            # Prepare input with proper formatting
            formatted_input = self.prompt.format(
                system_prompt=system_prompt,
                user_prompt=prompt
            )

            # Generate response directly through pipeline
            response = self.llm(formatted_input)

            # Clean response
            if isinstance(response, dict):
                generated_text = response.get("generated_text", str(response))
            else:
                generated_text = str(response)

            cleaned = re.sub(r'\n{2,}', '\n', generated_text)
            cleaned = re.sub(r'\b(Hmm+|Okay|Alright|Wait)\b[\s,.]*', '', cleaned, flags=re.IGNORECASE)
            return cleaned.strip()

        except Exception as e:
            logger.error(f"Generation failed: {e}", exc_info=True)
            raise

# Global instance
deepseek_model = DeepSeekLangChain()
