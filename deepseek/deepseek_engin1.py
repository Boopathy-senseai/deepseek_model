import os
import torch
import logging
import re
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    AutoConfig,
    StoppingCriteria,
    StoppingCriteriaList
)
from huggingface_hub import snapshot_download
from typing import List

# ✅ Environment & Memory Optimization
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:256"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# ✅ Logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("deepseek_engine")

# ✅ Constants
MODEL_NAME = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
CACHE_DIR = "/home/ubuntu/deepseek_models"
MAX_INPUT_TOKENS = 7500
MAX_NEW_TOKENS = 3000
DEFAULT_TEMPERATURE = 0.6

# ✅ Custom stopping condition
class StopOnTokens(StoppingCriteria):
    def __init__(self, stop_token_ids: List[int]):
        self.stop_token_ids = stop_token_ids

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        return input_ids[0][-1].item() in self.stop_token_ids

# ✅ Quantization config
# bnb_config = BitsAndBytesConfig(
#     load_in_8bit=True,
#     bnb_8bit_quant_type="nf4",
#     bnb_8bit_compute_dtype=torch.float16,
#     bnb_8bit_use_double_quant=True,
#     llm_int8_enable_fp32_cpu_offload=True
# )
bnb_config = BitsAndBytesConfig(
    load_in_8bit=True,
    llm_int8_enable_fp32_cpu_offload=True
)

class DeepSeekModel:
    def __init__(self):
        self.model_name = MODEL_NAME
        self.cache_dir = CACHE_DIR
        os.makedirs(self.cache_dir, exist_ok=True)
        self.download_model()
        self.load_model()

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

    def load_model(self):
        try:
            logger.info("Loading tokenizer...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.cache_dir,
                trust_remote_code=True,
                padding_side="left",
                model_max_length=MAX_INPUT_TOKENS + MAX_NEW_TOKENS
            )

            # if not self.tokenizer.pad_token:
            #     self.tokenizer.pad_token = self.tokenizer.eos_token
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            logger.info("Loading config...")
            config = AutoConfig.from_pretrained(
                self.cache_dir,
                trust_remote_code=True
            )
            config.rope_scaling = {
                "type": "dynamic",
                "factor": 4.0,
                "original_max_position_embeddings": 2048
            }

            logger.info("Loading 4-bit model...")
            self.model = AutoModelForCausalLM.from_pretrained(
                self.cache_dir,
                config=config,
                quantization_config=bnb_config,
                device_map="auto",
                torch_dtype=torch.float16,
                trust_remote_code=True,
                low_cpu_mem_usage=True,
                max_memory={0: "20GiB", "cpu": "64GiB"}
            )

            # stop_token_ids = [
            #     self.tokenizer.eos_token_id,
            #     self.tokenizer.convert_tokens_to_ids(["<|endoftext|>"])[0]
            # 
            # ]
            custom_stop_token_id = self.tokenizer.convert_tokens_to_ids("<|endoftext|>")
            stop_token_ids = [self.tokenizer.eos_token_id]

            if custom_stop_token_id and custom_stop_token_id != self.tokenizer.eos_token_id:
                stop_token_ids.append(custom_stop_token_id)

            self.stopping_criteria = StoppingCriteriaList([StopOnTokens(stop_token_ids)])

            self.warmup_model()

        except Exception as e:
            logger.error(f"Model loading failed: {e}")
            raise

    def warmup_model(self):
        try:
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
            test_input = self.tokenizer(
                "Warmup: Explain GxP compliance.",
                return_tensors="pt",
                max_length=1024,
                truncation=True
            ).to(device)
            _ = self.model.generate(
                **test_input,
                max_new_tokens=512,
                temperature=0.1,
                do_sample=True
            )
            logger.info("Warmup complete ✅")
        except Exception as e:
            logger.warning(f"Warmup failed: {e}")

    def generate(self, prompt: str,system_prompt: str, temperature: float = DEFAULT_TEMPERATURE, max_new_tokens: int = MAX_NEW_TOKENS):
        try:


            full_prompt = f"<|system|>\n{system_prompt}\n<|user|>\n{prompt}\n<|assistant|>"
            logger.info(f"System prompt: {system_prompt}")
            logger.info(f"User prompt: {prompt}")
            logger.info(f"Full prompt: {full_prompt}")

            # inputs = self.tokenizer(
            #     full_prompt,
            #     return_tensors="pt",
            #     truncation=True,
            #     max_length=MAX_INPUT_TOKENS,
            # ).to("cuda" if torch.cuda.is_available() else "cpu")
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            inputs = self.tokenizer(
                full_prompt,
                return_tensors="pt",
                truncation=True,
                max_length=MAX_INPUT_TOKENS,
                padding=True,
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}

            if inputs["input_ids"].shape[1] < 5:
                logger.error(f"Tokenized input too short: shape={inputs.input_ids.shape}")
                raise ValueError("Prompt is too short or malformed.")
            torch.cuda.empty_cache()
            if torch.cuda.is_available():
                torch.cuda.synchronize()

            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=True,
                top_p=0.9,
                top_k=40,
                repetition_penalty=1.15,
                # length_penalty=0.8,
                stopping_criteria=self.stopping_criteria,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                use_cache=True,
                num_beams=1,
                output_scores=True,
                return_dict_in_generate=True
            )

            full_sequence = outputs.sequences[0]
            # new_tokens = full_sequence[inputs.input_ids.shape[-1]:]
            new_tokens = full_sequence[inputs["input_ids"].shape[-1]:]

            decoded = self.tokenizer.decode(
                new_tokens,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True
            ).strip()

            # Clean response
            cleaned = re.sub(r'\n{2,}', '\n', decoded)
            cleaned = re.sub(r'\b(Hmm+|Okay|Alright|Wait)\b[\s,.]*', '', cleaned, flags=re.IGNORECASE)

            logger.info(f"Generated {len(new_tokens)} tokens.")
            return cleaned

        except torch.cuda.OutOfMemoryError:
            logger.error("CUDA OOM. Retrying with fewer tokens.")
            torch.cuda.empty_cache()
            return self.generate(prompt, temperature, max_new_tokens // 2)
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            raise

# ✅ Global model instance
deepseek_model = DeepSeekModel()


