# import json
# import logging
# from django.http import JsonResponse, HttpResponseBadRequest
# from django.views.decorators.csrf import csrf_exempt
# from .deepseek_engine import deepseek_model
# from django.http import StreamingHttpResponse

# logger = logging.getLogger("gxp_model")

# @csrf_exempt
# def generate_text(request):
#     if request.method != "POST":
#         return HttpResponseBadRequest("Only POST requests are allowed")

#     try:
#         # Parse request data
#         data = json.loads(request.body.decode("utf-8"))
#         prompt = data.get("prompt", "").strip()
#         system_prompt = data.get(
#             "system_prompt",
#             "You are an AI assistant. Provide clear and accurate responses."
#         )
#         temperature = float(data.get("temperature", 0.6))
#         max_tokens = int(data.get("max_tokens", 3000))

#         # Validate inputs
#         if not prompt:
#             return JsonResponse(
#                 {"success": False, "error": "Prompt is required"},
#                 status=422
#             )

#         # Clamp values to safe ranges
#         temperature = max(0.1, min(1.0, temperature))
#         max_tokens = min(3000, max(100, max_tokens))

#         print(")))))0",prompt,system_prompt)
#         # Generate response
#         generator = deepseek_model.generate(
#             prompt=prompt,
#             system_prompt=system_prompt
#            # temperature=temperature,
#            # max_new_tokens=max_tokens
#         )

#         def token_stream():
#             for token in generator:
#                 yield token  # or yield token + " " for space between words

#         return StreamingHttpResponse(token_stream(), content_type="text/plain")

#     except json.JSONDecodeError:
#         return JsonResponse(
#             {"success": False, "error": "Invalid JSON format"},
#             status=400
#         )
#     except ValueError as e:
#         return JsonResponse(
#             {"success": False, "error": str(e)},
#             status=400
#         )
#     except Exception as e:
#         logger.error(f"Generation failed: {str(e)}", exc_info=True)
#         return JsonResponse(
#             {"success": False, "error": "Internal server error"},
#             status=500
#         )

# import json
# import logging
# from django.http import JsonResponse, HttpResponseBadRequest, StreamingHttpResponse
# from django.views.decorators.csrf import csrf_exempt
# from .deepseek_engine import deepseek_model
# import sys
# import time
# logger = logging.getLogger("gxp_model")

# @csrf_exempt
# def generate_text(request):
#     if request.method != "POST":
#         return HttpResponseBadRequest("Only POST requests are allowed")

#     try:
#         data = json.loads(request.body.decode("utf-8"))
#         prompt = data.get("prompt", "").strip()
#         system_prompt = data.get("system_prompt", "You are an AI assistant. Provide clear and accurate responses.")
#         temperature = float(data.get("temperature", 0.6))
#         max_tokens = int(data.get("max_tokens", 200))

#         if not prompt:
#             return JsonResponse({"success": False, "error": "Prompt is required"}, status=422)

#         # Clamp values
#         temperature = max(0.1, min(1.0, temperature))
#         max_tokens = min(200, max(100, max_tokens))

#         logger.info(f"Prompt: {prompt}")
#         logger.info(f"System prompt: {system_prompt}")
#         logger.info(f"Temperature: {temperature}, Max Tokens: {max_tokens}")

#         # Define the generator function to stream data
        # def event_stream():
        #     try:
        #         for chunk in deepseek_model.generate(
        #             prompt=prompt,
        #             system_prompt=system_prompt,
        #             temperature=temperature,
        #             max_new_tokens=max_tokens
        #         ):
        #             if chunk:
        #                 sys.stdout.flush()  # Ensures flushing from backend
        #                 yield f"{chunk}\n"
        #                 time.sleep(0.01)    # Optional: simulate slight delay
        #     except Exception as e:
        #         yield f"[ERROR]: {str(e)}\n"

        # response = StreamingHttpResponse(event_stream(), content_type="text/plain")
        # response["X-Accel-Buffering"] = "no"  # Disable buffering in nginx/gunicorn
        # return response


#     except json.JSONDecodeError:
#         return JsonResponse({"success": False, "error": "Invalid JSON format"}, status=400)
#     except ValueError as e:
#         return JsonResponse({"success": False, "error": str(e)}, status=400)
#     except Exception as e:
#         logger.error(f"Generation failed: {str(e)}", exc_info=True)
#         return JsonResponse({"success": False, "error": "Internal server error"}, status=500)
# import json
# import logging
# from django.http import JsonResponse, HttpResponseBadRequest
# from django.views.decorators.csrf import csrf_exempt
# from channels.layers import get_channel_layer
# from asgiref.sync import async_to_sync
# from .deepseek_engine import deepseek_model
# logger = logging.getLogger("gxp_model")

# @csrf_exempt
# def generate_text(request):
#     if request.method != "POST":
#         return HttpResponseBadRequest("Only POST requests are allowed")

#     try:
#         data = json.loads(request.body.decode("utf-8"))
#         prompt = data.get("prompt", "").strip()
#         system_prompt = data.get(
#             "system_prompt",
#             "You are an AI assistant. Provide clear and accurate responses."
#         )
#         session_id = data.get("session_id", "").strip()
#         print("))))))))))))))))))))))))))))))))))",prompt,session_id)
#         if not prompt:
#             return JsonResponse({"success": False, "error": "Prompt is required"}, status=422)
#         if not session_id:
#             return JsonResponse({"success": False, "error": "Session ID is required"}, status=422)
#         generator = deepseek_model.generate(
#             prompt=prompt,
#             system_prompt=system_prompt
#            # temperature=temperature,
#            # max_new_tokens=max_tokens
#         )
#         # print("****************",generator,json.dumps(generator))
#         tokens = list(generator)
#         def token_stream():
#             print("***************",tokens)
#             for token in tokens:
#                 print("***************",token)
#                 yield token  # or yield token + " " for space between words

#         #  StreamingHttpResponse(token_stream(), content_type="text/plain")

#         # Send prompt to the WebSocket consumer group
#         print("step one")
#         channel_layer = get_channel_layer()
#         print("Step two")
#         async_to_sync(channel_layer.group_send)(
#             f"stream_{session_id}",
#             {
#                 "type": "send.prompt",
#                 "prompt": prompt,
#                 "data": token_stream(),
#             }
#         )

#         return JsonResponse({"success": True, "message": "Prompt sent to WebSocket"})

#     except json.JSONDecodeError:
#         return JsonResponse({"success": False, "error": "Invalid JSON format"}, status=400)
#     except Exception as e:
#         logger.error(f"Failed to send prompt: {str(e)}", exc_info=True)
#         return JsonResponse({"success": False, "error": "Internal server error"}, status=500)
import json
import logging
from django.http import JsonResponse, HttpResponseBadRequest, StreamingHttpResponse
from django.views.decorators.csrf import csrf_exempt
from .deepseek_engine import deepseek_model  # Assuming this is your token generator

logger = logging.getLogger("gxp_model")

def stream_response(generator):
    try:
        for token in generator:
            # print("Streaming token:", token)
            yield token + "\n"  # newline delimited for FastAPI to parse line-by-line
        yield "[[END]]\n"  # Optional: signal end of stream
    except Exception as e:
        logger.error(f"Error during streaming: {str(e)}", exc_info=True)
        yield "[[ERROR]]\n"

@csrf_exempt
def generate_text(request):
    if request.method != "POST":
        return HttpResponseBadRequest("Only POST requests are allowed")

    try:
        data = json.loads(request.body.decode("utf-8"))
        prompt = data.get("prompt", "").strip()
        system_prompt = data.get(
            "system_prompt",
            "You are an AI assistant. Provide clear and accurate responses."
        )
        session_id = data.get("session_id", "").strip()

        if not prompt:
            return JsonResponse({"success": False, "error": "Prompt is required"}, status=422)
        if not session_id:
            return JsonResponse({"success": False, "error": "Session ID is required"}, status=422)

        # Create generator from deepseek_model
        generator = deepseek_model.generate(prompt=prompt, system_prompt=system_prompt)

        # Return streaming response
        return StreamingHttpResponse(
            stream_response(generator),
            content_type="text/plain"
        )

    except json.JSONDecodeError:
        return JsonResponse({"success": False, "error": "Invalid JSON format"}, status=400)
    except Exception as e:
        logger.error(f"Failed to stream prompt: {str(e)}", exc_info=True)
        return JsonResponse({"success": False, "error": "Internal server error"}, status=500)
