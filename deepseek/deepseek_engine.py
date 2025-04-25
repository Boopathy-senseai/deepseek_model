# import os
# import re
# import torch
# import logging
# from typing import List
# from transformers import (
#     AutoTokenizer,
#     AutoModelForCausalLM,
#     pipeline,
#     StoppingCriteria,
#     StoppingCriteriaList,
#     BitsAndBytesConfig
# )
# from peft import PeftModel, PeftConfig
# from langchain_huggingface import HuggingFacePipeline
# from langchain_core.prompts import PromptTemplate

# # === Constants ===
# BASE_MODEL = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
# ADAPTER_PATH = "/home/ubuntu/deepseek_models/fine_tuned"
# OFFLOAD_DIR = "/home/ubuntu/deepseek_offload"
# MAX_NEW_TOKENS = 3000
# DEFAULT_TEMPERATURE = 0.8

# # === Logging Setup ===
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger("deepseek_engine")

# # === Environment ===
# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:256"
# os.environ["TOKENIZERS_PARALLELISM"] = "false"


# class StopOnTokens(StoppingCriteria):
#     def __init__(self, stop_token_ids: List[int]):
#         self.stop_token_ids = stop_token_ids

#     def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
#         return any(stop_id in input_ids[0] for stop_id in self.stop_token_ids)


# class DeepSeekLangChain:
#     def __init__(self):
#         self.adapter_path = ADAPTER_PATH
#         self.offload_dir = OFFLOAD_DIR

#         print("Cache contents:", os.listdir(self.adapter_path))
#         os.makedirs(self.offload_dir, exist_ok=True)

#         self.initialize_pipeline()
#         self.create_chain()

#     def initialize_pipeline(self):
#         print("Initializing tokenizer...")
#         try:
#             self.tokenizer = AutoTokenizer.from_pretrained(
#                 BASE_MODEL,
#                 trust_remote_code=True,
#                 padding_side="left"
#             )

#             # Ensure pad_token is set correctly
#             if self.tokenizer.pad_token is None:
#                 self.tokenizer.pad_token = self.tokenizer.eos_token

#             print("Loading PEFT config...")
#             peft_config = PeftConfig.from_pretrained(self.adapter_path)

#             # Quantization configuration with FP32 CPU offload enabled
#             quant_config = BitsAndBytesConfig(
#                 load_in_4bit=True,
#                 bnb_4bit_quant_type="nf4",
#                 bnb_4bit_use_double_quant=True,
#                 bnb_4bit_compute_dtype=torch.float16,
#                 llm_int8_enable_fp32_cpu_offload=True  # Enable FP32 offloading
#             )

#             print("Loading base model...")
#             base_model = AutoModelForCausalLM.from_pretrained(
#                 peft_config.base_model_name_or_path,
#                 torch_dtype=torch.float16,
#                 device_map="auto",  # Automatically distribute the model across available devices
#                 trust_remote_code=True,
#                 quantization_config=quant_config,
#                 offload_folder=self.offload_dir,
#                 offload_state_dict=True
#             )

#             # Resize token embeddings to match tokenizer size
#             base_model.resize_token_embeddings(len(self.tokenizer))

#             print("Merging PEFT adapter...")
#             peft_model = PeftModel.from_pretrained(
#                 base_model,
#                 self.adapter_path,
#                 offload_folder=self.offload_dir,
#                 device_map="auto",
#             )
#             model = peft_model.merge_and_unload()

#             stop_token_ids = [self.tokenizer.eos_token_id]
#             stopping_criteria = StoppingCriteriaList([StopOnTokens(stop_token_ids)])

#             print("Creating HuggingFace pipeline...")
#             text_pipeline = pipeline(
#                 "text-generation",
#                 model=model,
#                 tokenizer=self.tokenizer,
#                 max_new_tokens=MAX_NEW_TOKENS,
#                 temperature=DEFAULT_TEMPERATURE,
#                 top_p=0.8,
#                 top_k=40,
#                 repetition_penalty=1.1,
#                 padtoken_id=self.tokenizer.pad_token_id,
#                 eos_token_id=self.tokenizer.eos_token_id,
#                 do_sample=True,
#                 return_full_text=False,
#                 stopping_criteria=stopping_criteria,
#                 device_map="auto"  # Automatically distribute the workload across available GPUs
#             )

#             self.llm = HuggingFacePipeline(pipeline=text_pipeline)
#         except Exception as e:
#             logger.error(f"Error during pipeline initialization: {e}", exc_info=True)
#             raise RuntimeError(f"Failed to initialize the DeepSeek model pipeline: {e}")

#     def create_chain(self):
#         print("Creating LangChain chain...")
#         template = "{user_prompt}"
#         self.prompt = PromptTemplate(template=template, input_variables=["user_prompt"])
#         self.chain = self.prompt | self.llm

#     def generate(self, prompt: str, system_prompt: str = "", chat_history=None) -> str:
#         try:
#             logger.info(f"Prompt: {prompt}")
#             if chat_history:
#                 system_prompt = system_prompt + "\n" + f"Current Conversation Context: {chat_history}"
#             full_prompt = f"{system_prompt}\n{prompt}" if system_prompt else prompt
#             result = self.chain.invoke({"user_prompt": full_prompt})
#             output = result.strip() if isinstance(result, str) else str(result)

#             # Clean up
#             output = re.sub(r'User:.*?Assistant:', '', output, flags=re.DOTALL)
#             output = re.sub(r'\n{2,}', '\n', output)
#             output = re.sub(r'\b(Hmm+|Okay|Alright|Wait)\b[\s,.]*', '', output, flags=re.IGNORECASE)
#             return output.strip()

#         except Exception as e:
#             logger.error(f"Text generation failed: {e}", exc_info=True)
#             return "An error occurred during generation."


# # Initialize and test the generation
# try:
#     deepseek_model = DeepSeekLangChain()
#     print("DeepSeek model initialized successfully.")
#     while True:
#         data = input("Enter your prompt: ")
#         if data.lower() == 'exit':
#             break
#         # Generate response
#         print("Generating response...")
#         result = deepseek_model.generate(data)
#         print("Generated response:", result)
# except Exception as e:
#     print(f"Failed to initialize DeepSeek model: {e}")
import os
import re
import torch
import logging
from typing import List
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    pipeline,
    TextIteratorStreamer,
    StoppingCriteria,
    StoppingCriteriaList,
    BitsAndBytesConfig
)
import threading
from sty import fg
from peft import PeftModel, PeftConfig
from langchain_huggingface import HuggingFacePipeline
from langchain_core.prompts import PromptTemplate

# === Constants ===
BASE_MODEL = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
ADAPTER_PATH = "/home/ubuntu/deepseek_llama/HQ_devi_intake_finetuned_model"
OFFLOAD_DIR = "/home/ubuntu/deepseek_offload"
MAX_NEW_TOKENS = 3000
DEFAULT_TEMPERATURE = 0.8

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
 
# === Logging Setup ===
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("deepseek_engine")

# === Environment ===
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:256"
os.environ["TOKENIZERS_PARALLELISM"] = "false"


class StopOnTokens(StoppingCriteria):
    def __init__(self, stop_token_ids: List[int]):
        self.stop_token_ids = stop_token_ids

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        return any(stop_id in input_ids[0] for stop_id in self.stop_token_ids)


class DeepSeekLangChain:
    def __init__(self):
        self.adapter_path = ADAPTER_PATH
        self.offload_dir = OFFLOAD_DIR

        print("Cache contents:", os.listdir(self.adapter_path))
        os.makedirs(self.offload_dir, exist_ok=True)

        self.initialize_pipeline()
        self.create_chain()

    def initialize_pipeline(self):
        print("Initializing tokenizer...")
        try:
            # First try loading tokenizer from adapter
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.adapter_path,
                    trust_remote_code=True,
                    padding_side="left"
                )
                print("Loaded tokenizer from adapter directory")
            except:
                self.tokenizer = AutoTokenizer.from_pretrained(
                    BASE_MODEL,
                    trust_remote_code=True,
                    padding_side="left"
                )
                print("Loaded tokenizer from base model")

            # Ensure pad_token is set correctly
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            print(f"Tokenizer vocab size: {len(self.tokenizer)}")

            print("Loading PEFT config...")
            peft_config = PeftConfig.from_pretrained(self.adapter_path)

            # Quantization configuration
            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.float16,
                llm_int8_enable_fp32_cpu_offload=True
            )

            print("Loading base model...")
            base_model = AutoModelForCausalLM.from_pretrained(
                peft_config.base_model_name_or_path,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True,
                quantization_config=quant_config,
                offload_folder=self.offload_dir,
                offload_state_dict=True
            )

            # Get expected vocab size from adapter config
            adapter_config = PeftConfig.from_pretrained(self.adapter_path)
            expected_vocab_size = adapter_config.vocab_size if hasattr(adapter_config, 'vocab_size') else len(self.tokenizer)
            
            print(f"Resizing embeddings to: {expected_vocab_size}")
            base_model.resize_token_embeddings(expected_vocab_size)

            print("Merging PEFT adapter...")
            peft_model = PeftModel.from_pretrained(
                base_model,
                self.adapter_path,
                offload_folder=self.offload_dir,
                device_map="auto",
            )
            
            # Merge and unload
            model = peft_model.merge_and_unload()
            self.model = peft_model.merge_and_unload()

            stop_token_ids = [self.tokenizer.eos_token_id]
            stopping_criteria = StoppingCriteriaList([StopOnTokens(stop_token_ids)])
            self.stopping_criteria = StoppingCriteriaList([StopOnTokens(stop_token_ids)])

            print("Creating HuggingFace pipeline...")
            text_pipeline = pipeline(
                "text-generation",
                model=model,
                tokenizer=self.tokenizer,
                max_new_tokens=MAX_NEW_TOKENS,
                temperature=DEFAULT_TEMPERATURE,
                top_p=0.8,
                top_k=40,
                repetition_penalty=1.1,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                do_sample=True,
                return_full_text=False,
                stopping_criteria=stopping_criteria,
                device_map="auto"
            )

            self.llm = HuggingFacePipeline(pipeline=text_pipeline)
        except Exception as e:
            logger.error(f"Error during pipeline initialization: {e}", exc_info=True)
            raise RuntimeError(f"Failed to initialize the DeepSeek model pipeline: {e}")

    def create_chain(self):
        print("Creating LangChain chain...")
        template = "{user_prompt}"
        self.prompt = PromptTemplate(template=template, input_variables=["user_prompt"])
        self.chain = self.prompt | self.llm

    def generate(self, prompt: str ,system_prompt:str, temperature:float=0.7,max_new_tokens:int=MAX_NEW_TOKENS,chat_history=None) -> str:
        try:
            logger.info(f"Prompt: {prompt}")
            if chat_history:
                system_prompt = system_prompt + "\n" + f"Current Conversation Context: {chat_history}"
                
            full_prompt = (
                f"<|system|>{system_prompt}<|end|>\n"
                f"<|user|>{prompt}<|end|>\n"
                "<|assistant|><think>"
            )

            # Tokenize and move to device
            inputs = self.tokenizer(full_prompt, return_tensors="pt").to(device)
            streamer = TextIteratorStreamer(self.tokenizer, skip_special_tokens=True)
    
            # Generation configuration
            generation_kwargs = {
                "input_ids": inputs["input_ids"],
                "attention_mask": inputs["attention_mask"],
                "max_new_tokens": max_new_tokens,
                "do_sample": True,
                "top_k": 50,
                "top_p": 0.95,
                "temperature": temperature,
                "pad_token_id": self.tokenizer.pad_token_id,
                "streamer": streamer,
                "repetition_penalty": 1.15,
            }

            print("p----->",generation_kwargs)
            # Run generation in a separate thread
            thread = threading.Thread(target=self.model.generate, kwargs=generation_kwargs)
            thread.start()
    
            buffer = ""
            start_printing = False
            end_think = False
    
            for token in streamer:
                buffer += token
    
                # Start streaming after <think>
                if not start_printing and "<think>" in buffer:
                    start_printing = True
                    buffer = buffer.split("<think>", 1)[1]
    
                    yield "<think>"  # Send opening tag
                    print(fg.green + "<think>", flush=True)
    
                    if buffer:
                        yield buffer
                        print(buffer, end="", flush=True)
                    buffer = ""
    
                elif start_printing:
                    # End streaming at </think>
                    if not end_think and "</think>" in buffer:
                        parts = buffer.split("</think>", 1)
    
                        yield parts[0]
                        print(parts[0], end="", flush=True)
    
                        yield "\n</think>\n"
                        print("\n</think>\n", flush=True)
    
                        buffer = parts[1]
                        end_think = True
                    else:
                        yield token
                        print(token, end="", flush=True)

        except Exception as e:
            logger.error(f"Text generation failed: {e}", exc_info=True)
            return "An error occurred during generation."

deepseek_model = DeepSeekLangChain()


    # def generate(self, prompt: str, system_prompt: str , chat_history=None) -> str:
    #     try:
    #         logger.info(f"Prompt: {prompt}")
    #         if chat_history:
    #             system_prompt = system_prompt + "\n" + f"Current Conversation Context: {chat_history}"
    #         full_prompt = (
    #         f"<|system|>{system_prompt}<|end|>\n"
    #         f"<|user|>{prompt}<|end|>\n"
    #         "<|assistant|><think>"
    #     )
    #         #full_prompt = f"{system_prompt}\n{prompt}" if system_prompt else prompt
    #         result = self.chain.invoke({"user_prompt": full_prompt})
    #         output = result.strip() if isinstance(result, str) else str(result)

    #         # Clean up
    #         output = re.sub(r'User:.*?Assistant:', '', output, flags=re.DOTALL)
    #         output = re.sub(r'\n{2,}', '\n', output)
    #         output = re.sub(r'\b(Hmm+|Okay|Alright|Wait)\b[\s,.]*', '', output, flags=re.IGNORECASE)
    #         return output.strip()

    #     except Exception as e:
    #         logger.error(f"Text generation failed: {e}", exc_info=True)
    #         return "An error occurred during generation."


# # Initialize and test the generation
# if __name__ == "__main__":
#     try:
#         deepseek_model = DeepSeekLangChain()
#         print("DeepSeek model initialized successfully.")
#         print("Test prompt: 'Explain quantum computing in simple terms'")
#         test_response = deepseek_model.generate("Explain quantum computing in simple terms")
#         print("Test response:", test_response)
        
#         # while True:
#         #     data = input("\nEnter your prompt (or 'exit' to quit): ")
#         #     if data.lower() == 'exit':
#         #         break
#         #     print("Generating response...")
#         #     result = deepseek_model.generate(data)
#         #     print("\nGenerated response:", result)
#     except Exception as e:
#         print(f"Failed to initialize DeepSeek model: {e}")
