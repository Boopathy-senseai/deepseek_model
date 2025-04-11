import json
import logging
from django.http import JsonResponse, HttpResponseBadRequest, StreamingHttpResponse
from django.views.decorators.csrf import csrf_exempt
from .deepseek_engine import deepseek_model

logger = logging.getLogger("gxp_model")

@csrf_exempt
def generate_text(request):
    if request.method != "POST":
        return HttpResponseBadRequest("Only POST requests are allowed")

    try:
        data = json.loads(request.body.decode("utf-8"))
        prompt = data.get("prompt", "").strip()
        system_prompt = data.get("system_prompt", "You are an AI assistant. Provide clear and accurate responses.")
        temperature = float(data.get("temperature", 0.6))
        max_tokens = int(data.get("max_tokens", 300))

        if not prompt:
            return JsonResponse({"success": False, "error": "Prompt is required"}, status=422)

        # Clamp values
        temperature = max(0.1, min(1.0, temperature))
        max_tokens = min(200, max(100, max_tokens))

        logger.info(f"Prompt: {prompt}")
        logger.info(f"System prompt: {system_prompt}")
        logger.info(f"Temperature: {temperature}, Max Tokens: {max_tokens}")

        # Wrap the generator to ensure each chunk is followed by newline and flushed immediately
        def event_stream():
            try:
                for chunk in deepseek_model.generate(
                    prompt=prompt,
                    system_prompt=system_prompt,
                    temperature=temperature,
                    max_new_tokens=max_tokens
                ):
                    if chunk:
                        yield f"{chunk}\n"
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

 