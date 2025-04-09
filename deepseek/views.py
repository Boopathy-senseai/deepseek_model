import json
import logging
from django.http import JsonResponse, HttpResponseBadRequest
from django.views.decorators.csrf import csrf_exempt
from .deepseek_engine import deepseek_model
from django.http import StreamingHttpResponse

logger = logging.getLogger("gxp_model")

@csrf_exempt
def generate_text(request):
    if request.method != "POST":
        return HttpResponseBadRequest("Only POST requests are allowed")

    try:
        # Parse request data
        data = json.loads(request.body.decode("utf-8"))
        prompt = data.get("prompt", "").strip()
        system_prompt = data.get(
            "system_prompt",
            "You are an AI assistant. Provide clear and accurate responses."
        )
        temperature = float(data.get("temperature", 0.6))
        max_tokens = int(data.get("max_tokens", 3000))

        # Validate inputs
        if not prompt:
            return JsonResponse(
                {"success": False, "error": "Prompt is required"},
                status=422
            )

        # Clamp values to safe ranges
        temperature = max(0.1, min(1.0, temperature))
        max_tokens = min(3000, max(100, max_tokens))

        print(")))))0",prompt,system_prompt)
        # Generate response
        generator = deepseek_model.generate(
            prompt=prompt,
            system_prompt=system_prompt
           # temperature=temperature,
           # max_new_tokens=max_tokens
        )

        def token_stream():
            for token in generator:
                yield token  # or yield token + " " for space between words

        return StreamingHttpResponse(token_stream(), content_type="text/plain")

    except json.JSONDecodeError:
        return JsonResponse(
            {"success": False, "error": "Invalid JSON format"},
            status=400
        )
    except ValueError as e:
        return JsonResponse(
            {"success": False, "error": str(e)},
            status=400
        )
    except Exception as e:
        logger.error(f"Generation failed: {str(e)}", exc_info=True)
        return JsonResponse(
            {"success": False, "error": "Internal server error"},
            status=500
        )


