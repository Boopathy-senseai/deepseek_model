# import json
# import logging
# from django.http import JsonResponse, HttpResponseBadRequest
# from django.views.decorators.csrf import csrf_exempt
# from .deepseek_engine import deepseek_model  # Your model instance

# logger = logging.getLogger("gxp_model")

# @csrf_exempt
# def generate_text(request):
#     if request.method != "POST":
#         return HttpResponseBadRequest("Only POST allowed")

#     try:
#         data = json.loads(request.body.decode("utf-8"))
#         prompt = data.get("prompt")
#         system_prompt = data.get(
#             "system_prompt",
#             "You are an AI assistant. Provide clear and accurate responses."
#         )
#         chat_history=data.get("chat_history",None)
#         temperature = float(data.get("temperature", 0.6))
#         max_tokens = int(data.get("max_tokens", 3000))

#         if not prompt:
#             return JsonResponse({"success": False, "error": "Prompt is required"}, status=422)

#         temperature = max(0.1, min(1.0, temperature))
#         max_tokens = min(3000, max_tokens)

#         result = deepseek_model.generate(
#             prompt=prompt,
#             system_prompt=system_prompt,
#             temperature=temperature,
#             max_new_tokens=max_tokens,
#             chat_history=chat_history
#         )

#         return JsonResponse({
#             "success": True,
#             "response": result,
#             "parameters": {
#                 "temperature": temperature,
#                 "max_tokens": max_tokens
#             }
#         })

#     except Exception as e:
#         logger.error(f"Generation failed: {str(e)}")
#         return JsonResponse({"success": False, "error": str(e)}, status=500)


import json,time
import logging
import traceback
from typing import Dict
from django.http import JsonResponse, HttpResponseBadRequest
from django.views.decorators.csrf import csrf_exempt
from .deepseek_engine import deepseek_model
from .rag_engine import rag_generate

# Setup logging
logger = logging.getLogger("gxp_model")

# Initialize model
try:
    print("Step 1: Initializing DeepSeek model...")
    
    print("Step 2: Model initialized successfully.")
    chat_memory: Dict[str, list] = {}  # Store conversations as lists for each session
except Exception as e:
    traceback.print_exc()
    logger.error(f"Failed to initialize DeepSeek model: {str(e)}")
    deepseek_model = None

@csrf_exempt
def generate_text(request):
    if request.method != "POST":
        return HttpxResponseBadRequest("Only POST allowed")

    if deepseek_model is None:

        return JsonResponse({
            "success": False,
            "error": "Model initialization failed"
        }, status=500)

    try:
        data = json.loads(request.body.decode("utf-8"))
        prompt = data.get("prompt")
        system_prompt = data.get("system_prompt"," ")
        session_id = data.get("session_id")
        temperature = float(data.get("temperature", 0.6))
        max_tokens = int(data.get("max_tokens", 1000))
        chat_history = data.get("chat_history", None)
        use_rag = int(data.get("use_rag", False))
        time.sleep(4)
        print("&*&*&*&*&*&*&",use_rag)
        time.sleep(4)
        if not prompt:
            return JsonResponse({"success": False, "error": "Prompt is required"}, status=422)

        temperature = max(0.1, min(1.0, temperature))
        max_tokens = min(1000, max_tokens)

        # Retrieve or initialize the conversation history for the session
        if session_id not in chat_memory:
            chat_memory[session_id] = []

        # Build conversation string
        conversation = '\n'.join(f"User: {item[0]}\nAssistant: {item[1]}" for item in chat_memory[session_id])
        
        # Prepare the full prompt with conversation context
        full_prompt = f"{conversation}\nUser: {prompt}\nAssistant:"

        '''system_prompt = f"""
        {system_prompt}
        The initial query is: {prompt}
        The conversation so far is:
        {conversation}
        """
        '''
        system_prompt=system_prompt
        # Log the full prompt for debugging
        logger.info(f"Full Prompt: {full_prompt}")

        if use_rag==1:
            print("**************************")
            result = rag_generate(
                user_query=prompt,
                system_prompt=system_prompt,
                temperature=temperature,
                max_tokens=max_tokens
            )
        else:
            print("!!!!!!!!!!!!!!!!!!")
            result = deepseek_model.generate(
            prompt=prompt,
            system_prompt=system_prompt,
            chat_history=chat_history

        )
        print("*****************************************", result)
        # Append the user input and the assistant's response to the conversation history
        if len(chat_memory[session_id]) >= 8:  # Keep only the last 8 interactions
            chat_memory[session_id].pop(0)  # Remove the oldest entry
        
        # Assume result contains the assistant's response
        chat_memory[session_id].append((prompt, result))
        print("*********************rjhjhjh********************", result)
        return JsonResponse({
            "success": True,
            "response": result,
            "parameters": {
                "temperature": temperature,
                "max_tokens": max_tokens
            }
        })
    
    except Exception as e:
        logger.error(f"Generation failed: {str(e)}", exc_info=True)
        return JsonResponse({
            "success": False,
            "error": str(e)
        }, status=500)

