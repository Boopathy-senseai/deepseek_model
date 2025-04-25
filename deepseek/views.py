# import json
# import logging
# import traceback
# from typing import Dict
# from django.http import JsonResponse, HttpResponseBadRequest
# from django.views.decorators.csrf import csrf_exempt
# from .deepseek_engine import DeepSeekLangChain

# # Setup logging
# logger = logging.getLogger("gxp_model")

# # Initialize model
# try:
#     print("Step 1: Initializing DeepSeek model...")
#     deepseek_model = DeepSeekLangChain()
#     print("Step 2: Model initialized successfully.")
#     chat_memory: Dict[str, str] = {}
# except Exception as e:
#     traceback.print_exc()
#     logger.error(f"Failed to initialize DeepSeek model: {str(e)}")
#     deepseek_model = None

# @csrf_exempt
# def generate_text(request):
#     if request.method != "POST":
#         return HttpResponseBadRequest("Only POST allowed")

#     if deepseek_model is None:
#         return JsonResponse({
#             "success": False,
#             "error": "Model initialization failed"
#         }, status=500)

#     try:
#         data = json.loads(request.body.decode("utf-8"))
#         prompt = data.get("prompt")
#         system_prompt = data.get(
#             "system_prompt",
#             """You are a GxP risk assessment assistant trained to conduct structured WRPN assessments for GMP, GCP, GLP, GVP, CSV, and GDP deviations. Your job is to guide users through input collection, interpret their responses into structured risk scores (Severity, Probability, Detection), calculate the WRPN, and report the risk level and business impact clearly.\n\nBehavioral constraints:\n- Never ask users for scores; extract meaning from natural language.\n- Ask one simple, clear question at a time.\n- Avoid jargon unless already introduced.\n- Use friendly tone.\n- If deviation intake data is already provided, skip basic data gathering and focus on clarification and assessment.\n- Always return a final WRPN calculation, risk level, rationale for scoring, and a clear business impact summary.\n- **Stop asking questions once all scoring inputs are clear.** Conclude with risk assessment summary and ask if the user would like next-step help (e.g., CAPA)."""
#         )
#         session_id = data.get("session_id")
#         temperature = float(data.get("temperature", 0.6))
#         max_tokens = int(data.get("max_tokens", 1000))
#         chat_history=data.get("chat_history",None)
#         if not prompt:
#             return JsonResponse({"success": False, "error": "Prompt is required"}, status=422)

#         temperature = max(0.1, min(1.0, temperature))
#         max_tokens = min(1000, max_tokens)

#         user_input = prompt
#         full_prompt = f"{chat_memory.get(session_id, '')}User: {user_input}\nAssistant:"
#         logger.info(f"Full Prompt: {full_prompt}")

#         result = deepseek_model.generate(
#             prompt=full_prompt,
#             system_prompt=system_prompt
#         )

#         memory_key = f"User: {user_input}\nAssistant: {result}\n"
#         if session_id in chat_memory:
#             exchanges = chat_memory[session_id].count("User:")
#             if exchanges >= 3:
#                 parts = chat_memory[session_id].split("User:", 3)
#                 chat_memory[session_id] = "User:" + parts[-1]

#         chat_memory[session_id] = chat_memory.get(session_id, '') + memory_key

#         return JsonResponse({
#             "success": True,
#             "response": result,
#             "parameters": {
#                 "temperature": temperature,
#                 "max_tokens": max_tokens
#             }
#         })

#     except Exception as e:
#         logger.error(f"Generation failed: {str(e)}", exc_info=True)
#         return JsonResponse({
#             "success": False,
#             "error": str(e)
#         }, status=500)



import json
import logging
import traceback
from typing import Dict
from django.http import JsonResponse, HttpResponseBadRequest, StreamingHttpResponse

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
        return HttpResponseBadRequest("Only POST allowed")

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
        use_rag = bool(data.get("use_rag", False))
        print("&*&*&*&*&*&*&",use_rag)
        if not prompt:
            return JsonResponse({"success": False, "error": "Prompt is required"}, status=422)

        temperature = max(0.1, min(1.0, temperature))
        max_tokens = min(1000, max_tokens)

        # Retrieve or initialize the conversation history for the session
        if session_id not in chat_memory:
            chat_memory[session_id] = []
        
        # Build conversation string
        conversation = '\n'.join(f"User: {item[0]}\nAssistant: {item[1]}" for item in chat_memory[session_id])
        
        #    the full prompt with conversation context
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

        def event_stream():
            try:
                for chunk in deepseek_model.generate(
                    prompt=prompt,
                    system_prompt=system_prompt,
                    temperature=temperature,
                    max_new_tokens=max_tokens
                ):
                   
                    if len(chunk)>0:
                        yield f"{chunk}\n"
                    #else:
                    #    yield "//n" 
            except GeneratorExit:
                logger.warning("Streaming interrupted by client")
            except Exception as e:
                logger.error(f"Error in generator: {e}", exc_info=True)
                yield "[ERROR]: Streaming interrupted.\n"

        return StreamingHttpResponse(event_stream(), content_type="text/plain")


        # if use_rag:
        #     print("**************************")
        #     result = rag_generate(
        #         user_query=prompt,
        #         system_prompt=system_prompt,
        #         temperature=temperature,
        #         max_tokens=max_tokens
        #     )
        # else:
        #     print("!!!!!!!!!!!!!!!!!!")
        #     result = deepseek_model.generate(
        #     prompt=prompt,
        #     system_prompt=system_prompt,
        #     chat_history=chat_history

        # )
        # print("*****************************************", result)
        # # Append the user input and the assistant's response to the conversation history
        # if len(chat_memory[session_id]) >= 8:  # Keep only the last 8 interactions
        #     chat_memory[session_id].pop(0)  # Remove the oldest entry
        
        # # Assume result contains the assistant's response
        # chat_memory[session_id].append((prompt, result))

        # return JsonResponse({
        #     "success": True,
        #     "response": result,
        #     "parameters": {
        #         "temperature": temperature,
        #         "max_tokens": max_tokens
        #     }
        # })

    except json.JSONDecodeError:
        return JsonResponse({"success": False, "error": "Invalid JSON format"}, status=400)
    except ValueError as e:
        return JsonResponse({"success": False, "error": str(e)}, status=400)
    except Exception as e:
        logger.error(f"Generation failed: {str(e)}", exc_info=True)
        return JsonResponse({"success": False,"error": str(e)}, status=500)
