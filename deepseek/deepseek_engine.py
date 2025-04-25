# from transformers import AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer
# import torch
# import threading
# from sty import fg

# # Set device
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # Load tokenizer & model
# tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")
# model = AutoModelForCausalLM.from_pretrained(
#     "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
#     torch_dtype=torch.float16,
#     trust_remote_code=True
# ).to(device)
# model.eval()

# class DeepSeekModel:
#     @staticmethod
#     def generate(prompt, system_prompt="You are an AI assistant. Provide clear and accurate responses.", 
#                  temperature=0.7, max_new_tokens=1024):
#         # ... [previous setup code remains the same] ...

#         # Initialize streamer with proper newline handling
#         system_prompt = "You are an AI assistant. Provide clear and accurate responses."
#         full_prompt = (
#             f"<|system|>{system_prompt}<|end|>\n"
#             f"<|user|>{prompt}<|end|>\n"
#             "<|assistant|><think>"
#         )
#         inputs = tokenizer(full_prompt, return_tensors="pt").to(device)
#         streamer = TextIteratorStreamer(
#             tokenizer, 
#             skip_special_tokens=False,  # Preserve newlines
#             skip_prompt=True
#         )

#         # ... [generation kwargs remain the same] ...
#         generation_kwargs = {
#             "input_ids": inputs["input_ids"],
#             "attention_mask": inputs["attention_mask"],
#             "max_new_tokens": max_new_tokens,
#             "do_sample": True,
#             "top_k": 50,
#             "top_p": 0.95,
#             "temperature": temperature,
#             "pad_token_id": tokenizer.eos_token_id,
#             "streamer": streamer,
#             "repetition_penalty": 1.15,
#         }
#         thread = threading.Thread(target=model.generate, kwargs=generation_kwargs)
#         thread.start()
    
#         buffer = ""
#         in_think_block = False

#         for token in streamer:
#             buffer += token

#             # Handle think block boundaries
#             if "<think>" in buffer and not in_think_block:
#                 prefix, think_content = buffer.split("<think>", 1)
#                 yield prefix  # Send any content before <think>
#                 buffer = think_content
#                 in_think_block = True

#             if "</think>" in buffer and in_think_block:
#                 think_content, postfix = buffer.split("</think>", 1)

#                 # Process think content with newlines
#                 print(think_content)
#                 yield f"\n{think_content}\n"  # Wrap think content in newlines
#                 buffer = postfix
#                 in_think_block = False

#             # Send intermediate content
#             if not in_think_block and buffer:
#                 yield buffer.replace('\n', '\n')  # Explicit newline preservation
#                 buffer = ""

#         # Yield remaining content
#         if buffer:
#             yield buffer
# # Global instance
# deepseek_model = DeepSeekModel()
"""
from transformers import AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer
import torch
import threading
from sty import fg

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load tokenizer & model
tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")
model = AutoModelForCausalLM.from_pretrained(
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
    torch_dtype=torch.float16,
    trust_remote_code=True
).to(device)
model.eval()

class DeepSeekModel:
    @staticmethod
    def generate(prompt, system_prompt="You are an AI assistant. Provide clear and accurate responses.", 
                 temperature=0.7, max_new_tokens=1024):
        # Format the input prompt for the model
        system_prompt = "You are an AI assistant. Provide clear and accurate responses."
        full_prompt = (
            f"<|system|>{system_prompt}<|end|>\n"
            f"<|user|>{prompt}<|end|>\n"
            "<|assistant|><think>"
        )

        # Tokenize and move to device
        inputs = tokenizer(full_prompt, return_tensors="pt").to(device)
        streamer = TextIteratorStreamer(tokenizer, skip_special_tokens=True)

        # Generation configuration
        generation_kwargs = {
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs["attention_mask"],
            "max_new_tokens": max_new_tokens,
            "do_sample": True,
            "top_k": 50,
            "top_p": 0.95,
            "temperature": temperature,
            "pad_token_id": tokenizer.eos_token_id,
            "streamer": streamer,
            "repetition_penalty": 1.15,
        }

        # Run generation in a separate thread
        thread = threading.Thread(target=model.generate, kwargs=generation_kwargs)
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

                    yield "</think>"
                    print("</think>", flush=True)

                    buffer = parts[1]
                    end_think = True
                else:
                    yield token
                    print(token, end="", flush=True)

# Global instance
deepseek_model = DeepSeekModel()
"""
from transformers import AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer, BitsAndBytesConfig
import torch
import threading
from sty import fg
 
# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
 
# Quantization configuration (4-bit for memory efficiency)
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
)
 
# Load tokenizer & model (Llama 8B instead of DeepSeek)
model_id = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"  # Changed to Llama 8B
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=quantization_config,
    torch_dtype=torch.float16,
    trust_remote_code=True,
    device_map="auto"  # Automatically distributes layers across GPUs
).to(device)
model.eval()
 
class DeepSeekModel:
    @staticmethod
    def generate(prompt, system_prompt="You are an AI assistant. Provide clear and accurate responses.",
                 temperature=0.7, max_new_tokens=1024):
        # Format the input prompt for the model
        full_prompt = (
            f"<|system|>{system_prompt}<|end|>\n"
            f"<|user|>{prompt}<|end|>\n"
            "<|assistant|><think>"
        )
 
        # Tokenize and move to device
        inputs = tokenizer(full_prompt, return_tensors="pt").to(device)
        streamer = TextIteratorStreamer(tokenizer, skip_special_tokens=True)
 
        # Generation configuration
        generation_kwargs = {
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs["attention_mask"],
            "max_new_tokens": max_new_tokens,
            "do_sample": True,
            "top_k": 50,
            "top_p": 0.95,
            "temperature": temperature,
            "pad_token_id": tokenizer.eos_token_id,
            "streamer": streamer,
            "repetition_penalty": 1.15,
        }
 
        # Run generation in a separate thread
        thread = threading.Thread(target=model.generate, kwargs=generation_kwargs)
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
 
# Global instance (kept the same name for compatibility)
deepseek_model = DeepSeekModel()

