# import os
# import torch
# import logging
# from transformers import (
#     AutoTokenizer,
#     AutoModelForCausalLM,
#     TextIteratorStreamer,
#     TextGenerationPipeline,
#     StoppingCriteria,
#     StoppingCriteriaList
# )
# from typing import List

# from huggingface_hub import snapshot_download
# from langchain_community.llms import HuggingFacePipeline
# from langchain_core.prompts import PromptTemplate
# from langchain.chains import LLMChain
# from threading import Thread

# # Configuration
# os.environ["TOKENIZERS_PARALLELISM"] = "false"
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger("deepseek_engine")

# MODEL_NAME = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
# CACHE_DIR = "/Users/sridhar/Desktop/fast/deepseek_model/cache"
# MAX_INPUT_TOKENS = 7500
# MAX_NEW_TOKENS = 3000
# DEFAULT_TEMPERATURE = 0.6
# class StopOnTokens(StoppingCriteria):
#     def __init__(self, stop_token_ids: List[int]):
#         self.stop_token_ids = stop_token_ids

#     def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
#         return any(stop_id in input_ids[0] for stop_id in self.stop_token_ids)

# from datetime import datetime

# # Detect device
# if torch.backends.mps.is_available():
#     DEVICE = torch.device("mps")
#     TORCH_DTYPE = torch.float32
# elif torch.cuda.is_available():
#     DEVICE = torch.device("cuda")
#     TORCH_DTYPE = torch.float16
# else:
#     DEVICE = torch.device("cpu")
#     TORCH_DTYPE = torch.float32


# class DeepSeekLangChain:
#     def __init__(self):
#         self.model_name = MODEL_NAME
#         self.cache_dir = CACHE_DIR
#         os.makedirs(self.cache_dir, exist_ok=True)
#         self.download_model()
#         self.initialize_pipeline()
#         self.create_chain()

#     def download_model(self):
#         try:
#             if not os.path.exists(os.path.join(self.cache_dir, "config.json")):
#                 logger.info(f"Downloading model: {self.model_name}")
#                 snapshot_download(
#                     repo_id=self.model_name,
#                     local_dir=self.cache_dir,
#                     ignore_patterns=["*.md", "*.txt"],
#                     local_dir_use_symlinks=False
#                 )
#         except Exception as e:
#             logger.error(f"Model download failed: {e}")
#             raise

#     def initialize_pipeline(self):
#         try:
#             self.tokenizer = AutoTokenizer.from_pretrained(
#                 self.cache_dir,
#                 trust_remote_code=True,
#                 padding_side="left",
#                 model_max_length=MAX_INPUT_TOKENS + MAX_NEW_TOKENS
#             )

#             if self.tokenizer.pad_token is None:
#                 self.tokenizer.pad_token = self.tokenizer.eos_token

#             self.model = AutoModelForCausalLM.from_pretrained(
#                 self.cache_dir,
#                 torch_dtype=TORCH_DTYPE,
#                 trust_remote_code=True
#             ).to(DEVICE)
#             stop_token_ids = [self.tokenizer.eos_token_id]
#             stopping_criteria = StoppingCriteriaList([StopOnTokens(stop_token_ids)])
#             self.text_pipeline = TextGenerationPipeline(
#                 model=self.model,
#                 tokenizer=self.tokenizer,
#                 device=0 if DEVICE.type == "cuda" else -1,
#                 stopping_criteria=stopping_criteria,
#             )

#         except Exception as e:
#             logger.error(f"Pipeline initialization failed: {e}")
#             raise

#     def create_chain(self):
#         template = """[INST] <<SYS>>
# {system_prompt}
# <</SYS>>

# {user_prompt} [/INST]"""

#         self.prompt = PromptTemplate(
#             template=template,
#             input_variables=["system_prompt", "user_prompt"]
#         )

#         self.chain = LLMChain(
#             llm=HuggingFacePipeline(pipeline=self.text_pipeline),
#             prompt=self.prompt,
#             verbose=False
#         )

#     def generate(self, prompt: str, system_prompt: str = "You are a helpful AI assistant.",
#                  temperature: float = DEFAULT_TEMPERATURE,
#                  max_new_tokens: int = MAX_NEW_TOKENS):
#         try:
#             print("step 1")
#             if not prompt.strip():
#                 raise ValueError("User prompt is empty.")
#             print("step 2")
#             formatted_input = self.prompt.format(
#                 system_prompt=system_prompt,
#                   user_prompt=prompt
#                   )

#             inputs = self.tokenizer(formatted_input, return_tensors="pt", padding=True).to(DEVICE)
#             print("step 3")
#             streamer = TextIteratorStreamer(self.tokenizer, skip_special_tokens=True)
#             print("step 4",max_new_tokens)
#             generation_thread = Thread(
#                 target=self.model.generate,
#                 kwargs={
#                     "input_ids": inputs["input_ids"],
#                     "attention_mask": inputs["attention_mask"],
#                     "max_new_tokens": max_new_tokens,
#                     "temperature": temperature,
#                     "do_sample": True,
#                     "pad_token_id": self.tokenizer.pad_token_id,
#                     "streamer": streamer,
#                 }
#             )
#             generation_thread.start()

#             for token in streamer:
#                 print(f"django model Yielding token:{datetime.now()}", token)
#                 yield token

#         except Exception as e:
#             logger.error(f"Streaming generation failed: {e}", exc_info=True)
#             yield "[Error] Generation failed."


# # Global instance
# deepseek_model = DeepSeekLangChain() 
import os
import torch
import logging
from datetime import datetime
from threading import Thread
from typing import List

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TextIteratorStreamer,
    StoppingCriteria,
    StoppingCriteriaList
)
from huggingface_hub import snapshot_download
from transformers import TextGenerationPipeline
from langchain_community.llms import HuggingFacePipeline
from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain

# Set environment variables
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("deepseek_engine")

# Constants
MODEL_NAME = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
CACHE_DIR = "/Users/sridhar/Desktop/fast/deepseek_model/cache"  # <-- Change this path
MAX_INPUT_TOKENS = 7500
MAX_NEW_TOKENS = 3000
DEFAULT_TEMPERATURE = 0.6

# Device selection
if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
    TORCH_DTYPE = torch.float32
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    TORCH_DTYPE = torch.float16
else:
    DEVICE = torch.device("cpu")
    TORCH_DTYPE = torch.float32

# Custom stopping criteria
class StopOnTokens(StoppingCriteria):
    def __init__(self, stop_token_ids: List[int]):
        self.stop_token_ids = stop_token_ids

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        return any(stop_id in input_ids[0] for stop_id in self.stop_token_ids)

# DeepSeek streaming wrapper
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
            if not os.path.exists(os.path.join(self.cache_dir, "config.json")):
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

            self.model = AutoModelForCausalLM.from_pretrained(
                self.cache_dir,
                torch_dtype=TORCH_DTYPE,
                trust_remote_code=True,
                
            ).to(DEVICE)

            stop_token_ids = [self.tokenizer.eos_token_id]
            self.stopping_criteria = StoppingCriteriaList([StopOnTokens(stop_token_ids)])

        except Exception as e:
            logger.error(f"Pipeline initialization failed: {e}")
            raise

    def create_chain(self):
        template = """[INST] 
{system_prompt}

{user_prompt} [/INST]"""

        self.prompt = PromptTemplate(
            template=template,
            input_variables=["system_prompt", "user_prompt"]
        )

    def generate(self, prompt: str, system_prompt: str = "You are a helpful AI assistant.",
                 temperature: float = DEFAULT_TEMPERATURE,
                 max_new_tokens: int = MAX_NEW_TOKENS):
        try:
            if not prompt.strip():
                raise ValueError("User prompt is empty.")

            formatted_input = self.prompt.format(
                system_prompt=system_prompt,
                user_prompt=prompt
            )

            inputs = self.tokenizer(
                formatted_input,
                return_tensors="pt",
                padding=True
            ).to(DEVICE)

            # Set up streaming
            streamer = TextIteratorStreamer(self.tokenizer, skip_special_tokens=True)

            generation_thread = Thread(
                target=self.model.generate,
                kwargs={
                    "input_ids": inputs["input_ids"],
                    "attention_mask": inputs["attention_mask"],
                    "max_new_tokens": max_new_tokens,
                    "temperature": temperature,
                    "do_sample": True,
                    "pad_token_id": self.tokenizer.pad_token_id,
                    "streamer": streamer,
                    "stopping_criteria": self.stopping_criteria,
                    "repetition_penalty": 1.15,
                    "num_beams":1
                }
            )
            generation_thread.start()

            # Yield each token as it’s generated
            for token in streamer:
                print(f"django model Yielding token: {datetime.now()} {token}")
                yield token

        except Exception as e:
            logger.error(f"Streaming generation failed: {e}", exc_info=True)
            yield "[Error] Generation failed."


# Global instance
deepseek_model = DeepSeekLangChain()
