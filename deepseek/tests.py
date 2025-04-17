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

# Function to stream and print only after <think>
def stream_response(prompt):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    streamer = TextIteratorStreamer(tokenizer, skip_special_tokens=True)

    thread = threading.Thread(
        target=model.generate,
        kwargs={
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs["attention_mask"],
            "max_new_tokens": 1024,
            "do_sample": True,
            "top_k": 50,
            "top_p": 0.95,
            "temperature": 0.7,
            "pad_token_id": tokenizer.eos_token_id,
            "streamer": streamer,
            "repetition_penalty": 1.15,
        }
    )
    thread.start()

    buffer = ""
    start_printing = False
    end_think = False

    for token in streamer:
        buffer += token

        # Start printing only after <think>
        if not start_printing and "<think>" in buffer:
            start_printing = True
            buffer = buffer.split("<think>", 1)[1]
            print(fg.green + "<think>\n" + buffer, end="", flush=True)
        elif start_printing:
            if not end_think and "</think>" in buffer:
                parts = buffer.split("\n</think>", 1)
                print(parts[0], end="", flush=True)
                print("\n</think>")
                print(fg.yellow, end="", flush=True)
                buffer = parts[1]
                end_think = True
            else:
                print(token, end="", flush=True)

    print(fg.rs)  # Reset color

# Ask model using chat-style prompt
def ask_question(user_input):
    prompt = (
        "<|system|>You are a friendly chatbot named Chatty.<|end|>\n"
        f"<|user|>{user_input}<|end|>\n"
        "<|assistant|><think>"
    )
    print(fg.cyan + "\n[Model is thinking...]" + fg.rs)
    stream_response(prompt)
 

# Initial chat
# ask_question("Please introduce yourself and say 'How can I help you today?'")

# Loop for more questions
while True:
    user_input = input(fg.cyan + "\nYour question: " + fg.rs)
    if user_input.lower() == "exit":
        break
    ask_question(user_input)

 