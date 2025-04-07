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
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Logger Configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("deepseek_engine")

# Constants
MODEL_NAME = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
CACHE_DIR = "/Users/sridhar/Desktop/fast/deepseek_model"
MAX_INPUT_TOKENS = 200
MAX_NEW_TOKENS = 100
DEFAULT_TEMPERATURE = 0.6

# Detect device
if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
    TORCH_DTYPE = torch.float32  # MPS does not support float16 well
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    TORCH_DTYPE = torch.float16
else:
    DEVICE = torch.device("cpu")
    TORCH_DTYPE = torch.float32

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
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.cache_dir,
                trust_remote_code=True,
                padding_side="left",
                model_max_length=MAX_INPUT_TOKENS + MAX_NEW_TOKENS
            )

            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            model = AutoModelForCausalLM.from_pretrained(
                self.cache_dir,
                torch_dtype=TORCH_DTYPE,
                trust_remote_code=True
            ).to(DEVICE)

            stop_token_ids = [self.tokenizer.eos_token_id]
            stopping_criteria = StoppingCriteriaList([StopOnTokens(stop_token_ids)])

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
                device=0 if DEVICE.type == "cuda" else -1,  # -1 = CPU, 0 = GPU
                return_full_text=False
            )

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

            formatted_input = self.prompt.format(
                system_prompt=system_prompt,
                user_prompt=prompt
            )

            response = self.llm(formatted_input)

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
