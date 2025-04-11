
# # import os
# # import torch
# # import logging
# # import re
# # from transformers import (
# #     AutoTokenizer,
# #     AutoModelForCausalLM,
# #     pipeline,
# #     StoppingCriteria,
# #     StoppingCriteriaList,
# #     TextIteratorStreamer
# # )
# # from huggingface_hub import snapshot_download
# # from langchain_community.llms import HuggingFacePipeline
# # from langchain_core.prompts import PromptTemplate
# # from langchain.chains import LLMChain
# # from typing import List
# # from threading import Thread
# # import time

# # # Environment Configuration
# # os.environ["TOKENIZERS_PARALLELISM"] = "false"

# # # Logger Configuration
# # logging.basicConfig(level=logging.INFO)
# # logger = logging.getLogger("deepseek_engine")

# # # Constants
# # MODEL_NAME = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
# # CACHE_DIR = "/Users/sridhar/Desktop/fast/deepseek_model/cache"
# # MAX_INPUT_TOKENS = 200
# # MAX_NEW_TOKENS = 100
# # DEFAULT_TEMPERATURE = 0.6

# # # Detect Device
# # if torch.backends.mps.is_available():
# #     DEVICE = torch.device("mps")
# #     TORCH_DTYPE = torch.float32
# # elif torch.cuda.is_available():
# #     DEVICE = torch.device("cuda")
# #     TORCH_DTYPE = torch.float16
# # else:
# #     DEVICE = torch.device("cpu")
# #     TORCH_DTYPE = torch.float32


# # # Custom Stopping Criteria
# # class StopOnTokens(StoppingCriteria):
# #     def __init__(self, stop_token_ids: List[int]):
# #         self.stop_token_ids = stop_token_ids

# #     def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
# #         return any(stop_id in input_ids[0] for stop_id in self.stop_token_ids)


# # # DeepSeek Wrapper Class
# # class DeepSeekLangChain:
# #     def __init__(self):
# #         self.model_name = MODEL_NAME
# #         self.cache_dir = CACHE_DIR
# #         os.makedirs(self.cache_dir, exist_ok=True)
# #         self.download_model()
# #         self.initialize_pipeline()
# #         self.create_chain()

# #     def download_model(self):
# #         try:
# #             config_path = os.path.join(self.cache_dir, "config.json")
# #             if not os.path.exists(config_path):
# #                 logger.info(f"Downloading model: {self.model_name}")
# #                 snapshot_download(
# #                     repo_id=self.model_name,
# #                     local_dir=self.cache_dir,
# #                     ignore_patterns=["*.md", "*.txt"],
# #                     local_dir_use_symlinks=False
# #                 )
# #         except Exception as e:
# #             logger.error(f"Model download failed: {e}")
# #             raise

# #     def initialize_pipeline(self):
# #         try:
# #             self.tokenizer = AutoTokenizer.from_pretrained(
# #                 self.cache_dir,
# #                 trust_remote_code=True,
# #                 padding_side="left",
# #                 model_max_length=MAX_INPUT_TOKENS + MAX_NEW_TOKENS
# #             )

# #             if self.tokenizer.pad_token is None:
# #                 self.tokenizer.pad_token = self.tokenizer.eos_token

# #             self.model = AutoModelForCausalLM.from_pretrained(
# #                 self.cache_dir,
# #                 torch_dtype=TORCH_DTYPE,
# #                 trust_remote_code=True
# #             ).to(DEVICE)

# #         except Exception as e:
# #             logger.error(f"Pipeline initialization failed: {e}")
# #             raise

# #     def create_chain(self):
# #         template = """[INST] <<SYS>>
# # {system_prompt}
# # <</SYS>>

# # {user_prompt} [/INST]"""

# #         self.prompt = PromptTemplate(
# #             template=template,
# #             input_variables=["system_prompt", "user_prompt"]
# #         )

# #         # chain setup for synchronous LLM usage
# #         self.chain = LLMChain(
# #             llm=HuggingFacePipeline(pipeline=None),
# #             prompt=self.prompt,
# #             verbose=False
# #         )

# #     def generate(self, prompt: str, system_prompt: str = "You are a helpful AI assistant.",
# #                  temperature: float = DEFAULT_TEMPERATURE,
# #                  max_new_tokens: int = MAX_NEW_TOKENS):
# #         try:
# #             if not prompt.strip():
# #                 raise ValueError("User prompt is empty.")

# #             logger.info(f"System prompt: {system_prompt}")
# #             logger.info(f"User prompt: {prompt}")

# #             # Format input with prompt template
# #             formatted_input = self.prompt.format(
# #                 system_prompt=system_prompt,
# #                 user_prompt=prompt
# #             )
# #             response=self.llm(formatted_input)
# #             logger.info(f"Formatted input: {formatted_input}")
# #             logger.info(f"Response: {response}")
# #             if isinstance(response, dict):
# #                 generated_text = response.get("generated_text", str(response))
# #             else:
# #                 generated_text = str(response)
# #             logger.info(f"Generated text: {generated_text}")
# #             # Tokenize input
# #             inputs = self.tokenizer(formatted_input, return_tensors="pt", padding=True).to(DEVICE)

# #             # Create streamer for live token output
# #             streamer = TextIteratorStreamer(self.tokenizer, skip_special_tokens=True)

# #             # Start model generation in a separate thread
# #             generation_thread = Thread(
# #                 target=self.model.generate,
# #                 kwargs={
# #                     "input_ids": inputs["input_ids"],
# #                     "attention_mask": inputs["attention_mask"],
# #                     "max_new_tokens": max_new_tokens,
# #                     "temperature": temperature,
# #                     "do_sample": True,
# #                     "pad_token_id": self.tokenizer.pad_token_id,
# #                     "streamer": streamer,
# #                 }
# #             )
# #             generation_thread.start()

# #             # Yield tokens live as they are generated
# #             for token in streamer:
# #                 print("Yielding token:", token)
# #                 yield token

# #         except Exception as e:
# #             logger.error(f"Streaming generation failed: {e}", exc_info=True)
# #             yield "[Error] Generation failed."


# # # Global instance
# # deepseek_model = DeepSeekLangChain()
# '''
# import os
# import torch
# import logging
# from transformers import (
#     AutoTokenizer,
#     AutoModelForCausalLM,
#     TextIteratorStreamer,
#     TextGenerationPipeline
# )
# from huggingface_hub import snapshot_download
# from langchain_community.llms import HuggingFacePipeline
# from langchain_core.prompts import PromptTemplate
# from langchain.chains import LLMChain
# from typing import List
# from threading import Thread

# # Environment Configuration
# os.environ["TOKENIZERS_PARALLELISM"] = "false"

# # Logger Configuration
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger("deepseek_engine")

# # Constants
# MODEL_NAME = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
# CACHE_DIR = "/Users/sridhar/Desktop/fast/deepseek_model/cache"
# MAX_INPUT_TOKENS = 300
# MAX_NEW_TOKENS = 150
# DEFAULT_TEMPERATURE = 0.6

# # Detect Device
# if torch.backends.mps.is_available():
#     DEVICE = torch.device("mps")
#     TORCH_DTYPE = torch.float32
# elif torch.cuda.is_available():
#     DEVICE = torch.device("cuda")
#     TORCH_DTYPE = torch.float16
# else:
#     DEVICE = torch.device("cpu")
#     TORCH_DTYPE = torch.float32


# # Custom Stopping Criteria
# class StopOnTokens(torch.nn.Module):
#     def __init__(self, stop_token_ids: List[int]):
#         super().__init__()
#         self.stop_token_ids = stop_token_ids

#     def forward(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
#         return any(stop_id in input_ids[0] for stop_id in self.stop_token_ids)


# # DeepSeek Wrapper Class
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
#             config_path = os.path.join(self.cache_dir, "config.json")
#             if not os.path.exists(config_path):
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

#             # Define HF pipeline
#             self.text_pipeline = TextGenerationPipeline(
#                 model=self.model,
#                 tokenizer=self.tokenizer,
#                 device=0 if DEVICE.type == "cuda" else -1
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
#             print("step     1")
#             if not prompt.strip():
#                 raise ValueError("User prompt is empty.")
#             print("step     2")
#             logger.info(f"System prompt: {system_prompt}")
#             logger.info(f"User prompt: {prompt}")
#             print("step     3")
#             formatted_input = self.prompt.format(
#                 system_prompt=system_prompt,
#                 user_prompt=prompt
#             )

#             logger.info(f"Formatted input: {formatted_input}")
#             print("step     4")
#             # Generate response (synchronously)
#             try:
#                 response = self.chain.run({
#                     "system_prompt": system_prompt,
#                     "user_prompt": prompt
#                 })
#             except Exception as e:
#                 print("Error in chain.run:", e)
#             logger.info(f"Response: {response}")
#             print("step     5")
#             # Tokenize input
#             inputs = self.tokenizer(formatted_input, return_tensors="pt", padding=True).to(DEVICE)

#             # Create streamer for live token output
#             streamer = TextIteratorStreamer(self.tokenizer, skip_special_tokens=True)
#             print("step     6")
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
#             print("step     7")
#             for token in streamer:
#                 print("Yielding token:", token)
#                 yield token

#         except Exception as e:
#             logger.error(f"Streaming generation failed: {e}", exc_info=True)
#             yield "[Error] Generation failed."


# # Global instance
# deepseek_model = DeepSeekLangChain()


# '''

# # import os
# # import torch
# # import logging
# # from transformers import (
# #     AutoTokenizer,
# #     AutoModelForCausalLM,
# #     TextIteratorStreamer,
# #     TextGenerationPipeline
# # )
# # from huggingface_hub import snapshot_download
# # from langchain_community.llms import HuggingFacePipeline
# # from langchain_core.prompts import PromptTemplate
# # from langchain.chains import LLMChain
# # from typing import List
# # from threading import Thread

# # # Environment Configuration
# # os.environ["TOKENIZERS_PARALLELISM"] = "false"

# # # Logger Configuration
# # logging.basicConfig(level=logging.INFO)
# # logger = logging.getLogger("deepseek_engine")

# # # Constants
# # MODEL_NAME = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
# # CACHE_DIR = "/Users/sridhar/Desktop/fast/deepseek_model/cache"
# # MAX_INPUT_TOKENS = 300
# # MAX_NEW_TOKENS = 150
# # DEFAULT_TEMPERATURE = 0.6

# # # Detect Device
# # if torch.backends.mps.is_available():
# #     DEVICE = torch.device("mps")
# #     TORCH_DTYPE = torch.float32
# # elif torch.cuda.is_available():
# #     DEVICE = torch.device("cuda")
# #     TORCH_DTYPE = torch.float16
# # else:
# #     DEVICE = torch.device("cpu")
# #     TORCH_DTYPE = torch.float32


# # # Custom Stopping Criteria (Optional for advanced usage)
# # class StopOnTokens(torch.nn.Module):
# #     def __init__(self, stop_token_ids: List[int]):
# #         super().__init__()
# #         self.stop_token_ids = stop_token_ids

# #     def forward(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
# #         return any(stop_id in input_ids[0] for stop_id in self.stop_token_ids)


# # # DeepSeek Wrapper Class
# # class DeepSeekLangChain:
# #     def __init__(self):
# #         self.model_name = MODEL_NAME
# #         self.cache_dir = CACHE_DIR
# #         os.makedirs(self.cache_dir, exist_ok=True)
# #         self.download_model()
# #         self.initialize_pipeline()
# #         self.create_chain()

# #     def download_model(self):
# #         try:
# #             config_path = os.path.join(self.cache_dir, "config.json")
# #             if not os.path.exists(config_path):
# #                 logger.info(f"Downloading model: {self.model_name}")
# #                 snapshot_download(
# #                     repo_id=self.model_name,
# #                     local_dir=self.cache_dir,
# #                     ignore_patterns=["*.md", "*.txt"],
# #                     local_dir_use_symlinks=False
# #                 )
# #         except Exception as e:
# #             logger.error(f"Model download failed: {e}")
# #             raise

# #     def initialize_pipeline(self):
# #         try:
# #             self.tokenizer = AutoTokenizer.from_pretrained(
# #                 self.cache_dir,
# #                 trust_remote_code=True,
# #                 padding_side="left",
# #                 model_max_length=MAX_INPUT_TOKENS + MAX_NEW_TOKENS
# #             )

# #             if self.tokenizer.pad_token is None:
# #                 self.tokenizer.pad_token = self.tokenizer.eos_token

# #             self.model = AutoModelForCausalLM.from_pretrained(
# #                 self.cache_dir,
# #                 torch_dtype=TORCH_DTYPE,
# #                 trust_remote_code=True
# #             ).to(DEVICE)

# #             # Define HF pipeline for LLMChain
# #             self.text_pipeline = TextGenerationPipeline(
# #                 model=self.model,
# #                 tokenizer=self.tokenizer,
# #                 device=0 if DEVICE.type == "cuda" else -1,
# #                 max_new_tokens=MAX_NEW_TOKENS, 
# #                  do_sample=True,   
# #                  temperature=DEFAULT_TEMPERATURE
# #             )

# #         except Exception as e:
# #             logger.error(f"Pipeline initialization failed: {e}")
# #             raise

# #     def create_chain(self):
# #         template = """[INST] <<SYS>>
# # {system_prompt}
# # <</SYS>>

# # {user_prompt} [/INST]"""

# #         self.prompt = PromptTemplate(
# #             template=template,
# #             input_variables=["system_prompt", "user_prompt"]
# #         )

# #         self.chain = LLMChain(
# #             llm=HuggingFacePipeline(pipeline=self.text_pipeline),
# #             prompt=self.prompt,
# #             verbose=False
# #         )

# #     def generate(self, prompt: str, system_prompt: str = "You are a helpful AI assistant.",
# #                  temperature: float = DEFAULT_TEMPERATURE,
# #                  max_new_tokens: int = MAX_NEW_TOKENS,
# #                  stream: bool = False):
# #         try:
# #             if not prompt.strip():
# #                 raise ValueError("User prompt is empty.")

# #             logger.info(f"System prompt: {system_prompt}")
# #             logger.info(f"User prompt: {prompt}")

# #             # Format the prompt
# #             formatted_input = self.prompt.format(
# #                 system_prompt=system_prompt,
# #                 user_prompt=prompt
# #             )
# #             print("step     1",formatted_input)
# #             if not stream:
# #                 # Use LangChain LLMChain (sync response)
# #                 response = self.chain.run(system_prompt=system_prompt, user_prompt=prompt)
# #                 return response

# #             else:
# #                 # Tokenize for streaming
# #                 inputs = self.tokenizer(formatted_input, return_tensors="pt", padding=True).to(DEVICE)

# #                 # Streamer for real-time output
# #                 streamer = TextIteratorStreamer(self.tokenizer, skip_special_tokens=True)
# #                 thread = Thread(
# #                     target=self.model.generate,
# #                     kwargs={
# #                         "input_ids": inputs["input_ids"],
# #                         "attention_mask": inputs["attention_mask"],
# #                         "max_new_tokens": max_new_tokens,
# #                         "temperature": temperature,
# #                         "do_sample": True,
# #                         "streamer": streamer,
# #                         "pad_token_id": self.tokenizer.pad_token_id
# #                     }
# #                 )
# #                 thread.start()

# #                 # Yield tokens one by one
# #                 for token in streamer:
# #                     print("Yielding token:", token)
# #                     yield token

# #         except Exception as e:
# #             logger.error(f"Generation failed: {e}", exc_info=True)
# #             yield "[Error] Generation failed."

# # deepseek_model = DeepSeekLangChain()

# import os
# import torch
# import logging
# from transformers import (
#     AutoTokenizer,
#     AutoModelForCausalLM,
#     TextIteratorStreamer,
#     TextGenerationPipeline
# )
# from huggingface_hub import snapshot_download
# from langchain_community.llms import HuggingFacePipeline
# from langchain_core.prompts import PromptTemplate
# from langchain.chains import LLMChain
# from typing import List
# from threading import Thread

# # Environment Configuration
# os.environ["TOKENIZERS_PARALLELISM"] = "false"

# # Logger Configuration
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger("deepseek_engine")

# # Constants
# MODEL_NAME = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
# CACHE_DIR = "/Users/sridhar/Desktop/fast/deepseek_model/cache"
# MAX_INPUT_TOKENS = 300
# MAX_NEW_TOKENS = 250
# DEFAULT_TEMPERATURE = 0.6

# # Detect Device
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
#             config_path = os.path.join(self.cache_dir, "config.json")
#             if not os.path.exists(config_path):
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

#             # Define HF pipeline for LLMChain
#             hf_pipeline = TextGenerationPipeline(
#                 model=self.model,
#                 tokenizer=self.tokenizer,
#                 device=0 if DEVICE.type == "cuda" else -1,
#                 max_new_tokens=MAX_NEW_TOKENS,
#                 do_sample=True,
#                 temperature=DEFAULT_TEMPERATURE
#             )

#             # Wrap pipeline to return only generated text
#             def wrapped_pipeline(inputs, **kwargs):
#                 outputs = hf_pipeline(inputs, **kwargs)
#                 return [{"text": o["generated_text"]} for o in outputs]

#             self.text_pipeline = wrapped_pipeline

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
#                  max_new_tokens: int = MAX_NEW_TOKENS,
#                  stream: bool = False):
#         try:
#             if not prompt.strip():
#                 raise ValueError("User prompt is empty.")

#             logger.info(f"System prompt: {system_prompt}")
#             logger.info(f"User prompt: {prompt}")

#             # Format the prompt
#             formatted_input = self.prompt.format(
#                 system_prompt=system_prompt,
#                 user_prompt=prompt

#             )
            
#             if not stream:
#                 # Use LangChain LLMChain (sync response)
#                 response = self.chain.run(system_prompt=system_prompt, user_prompt=prompt)
#                 print("step     2",response)
#                 return response

#             else:
#                 # Tokenize for streaming
#                 inputs = self.tokenizer(formatted_input, return_tensors="pt", padding=True).to(DEVICE)

#                 # Streamer for real-time output
#                 streamer = TextIteratorStreamer(self.tokenizer, skip_special_tokens=True)
#                 thread = Thread(
#                     target=self.model.generate,
#                     kwargs={
#                         "input_ids": inputs["input_ids"],
#                         "attention_mask": inputs["attention_mask"],
#                         "max_new_tokens": max_new_tokens,
#                         "temperature": temperature,
#                         "do_sample": True,
#                         "streamer": streamer,
#                         "pad_token_id": self.tokenizer.pad_token_id
#                     }
#                 )
#                 thread.start()

#                 # Yield tokens one by one
#                 for token in streamer:
#                     yield token

#         except Exception as e:
#             logger.error(f"Generation failed: {e}", exc_info=True)
#             yield "[Error] Generation failed."


# # Export model instance
# deepseek_model = DeepSeekLangChain()

import os
import torch
import logging
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TextIteratorStreamer,
    TextGenerationPipeline
)
from huggingface_hub import snapshot_download
from langchain_community.llms import HuggingFacePipeline
from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain
from threading import Thread

# Configuration
os.environ["TOKENIZERS_PARALLELISM"] = "false"
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("deepseek_engine")

MODEL_NAME = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
CACHE_DIR = "/Users/sridhar/Desktop/fast/deepseek_model/cache"
MAX_INPUT_TOKENS = 300
MAX_NEW_TOKENS = 150
DEFAULT_TEMPERATURE = 0.6

# Detect device
if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
    TORCH_DTYPE = torch.float32
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    TORCH_DTYPE = torch.float16
else:
    DEVICE = torch.device("cpu")
    TORCH_DTYPE = torch.float32


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
                trust_remote_code=True
            ).to(DEVICE)

            self.text_pipeline = TextGenerationPipeline(
                model=self.model,
                tokenizer=self.tokenizer,
                device=0 if DEVICE.type == "cuda" else -1
            )

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
            llm=HuggingFacePipeline(pipeline=self.text_pipeline),
            prompt=self.prompt,
            verbose=False
        )

    def generate(self, prompt: str, system_prompt: str = "You are a helpful AI assistant.",
                 temperature: float = DEFAULT_TEMPERATURE,
                 max_new_tokens: int = MAX_NEW_TOKENS):
        try:
            print("step 1")
            if not prompt.strip():
                raise ValueError("User prompt is empty.")
            print("step 2")
            formatted_input = self.prompt.format(system_prompt=system_prompt, user_prompt=prompt)

            inputs = self.tokenizer(formatted_input, return_tensors="pt", padding=True).to(DEVICE)
            print("step 3")
            streamer = TextIteratorStreamer(self.tokenizer, skip_special_tokens=True)
            print("step 4")
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
                }
            )
            generation_thread.start()

            for token in streamer:
                print("Yielding token:", token)
                yield token

        except Exception as e:
            logger.error(f"Streaming generation failed: {e}", exc_info=True)
            yield "[Error] Generation failed."


# Global instance
deepseek_model = DeepSeekLangChain()
