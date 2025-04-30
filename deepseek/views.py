import json
import logging
from django.http import JsonResponse, HttpResponseBadRequest, StreamingHttpResponse
from django.views.decorators.csrf import csrf_exempt
from .deepseek_engine import deepseek_model
import time
logger = logging.getLogger("gxp_model")

@csrf_exempt
def generate_text(request):

    if request.method != "POST":
        return HttpResponseBadRequest("Only POST requests are allowed")

    try:
        print("!!!!!!!!!!!!!!!!!!!!!!")
        data = json.loads(request.body.decode("utf-8"))
        print(data)
        
        print("!!!!!!!!!!!!!!!!!!!!!!")
       
        prompt = data.get("prompt", "").strip()
        system_prompt = data.get("system_prompt", "")
        temperature = float(data.get("temperature", 0.6))
        max_tokens = int(data.get("max_tokens", 3000))
    
        if not prompt:
            return JsonResponse({"success": False, "error": "Prompt is required"}, status=422)

        # Clamp values
    
        temperature = max(0.1, min(1.0, temperature))
        max_tokens = min(3000, max_tokens)
        
        logger.info(f"Prompt: {prompt}")
        logger.info(f"System prompt: {system_prompt}")
        logger.info(f"Temperature: {temperature}, Max Tokens: {max_tokens}")

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

    except json.JSONDecodeError:
        return JsonResponse({"success": False, "error": "Invalid JSON format"}, status=400)
    except ValueError as e:
        return JsonResponse({"success": False, "error": str(e)}, status=400)
    except Exception as e:
        logger.error(f"Generation failed: {str(e)}", exc_info=True)
        return JsonResponse({"success": False, "error": "Internal server error"}, status=500)

 