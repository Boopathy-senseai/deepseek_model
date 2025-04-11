# import json
# from channels.generic.websocket import AsyncWebsocketConsumer
# from deepseek.deepseek_engine import DeepSeekLangChain

# class ChatConsumer(AsyncWebsocketConsumer):
#     async def connect(self):
#         await self.accept()
#         await self.send("Connected to DeepSeeek Moddel")


#     async def disconnect(self, close_code):
#         pass

#     async def receive(self, text_data):
#         data = json.loads(text_data)
#         prompt = data.get("prompt")
#         print("prompt",prompt)
#         # Start generating tokens using DeepSeekLangChain
#         model = DeepSeekLangChain()

#         try:
#             for token in model.generate(prompt):
#                 print("token send))))))",token)
#                 await self.send(text_data=json.dumps({"token": token}))
#         except Exception as e:
#             await self.send(text_data=json.dumps({"error": str(e)}))
# import json
# from channels.generic.websocket import AsyncWebsocketConsumer
# from deepseek.deepseek_engine import deepseek_model
# import asyncio

# class ChatConsumer(AsyncWebsocketConsumer):
#     async def connect(self):
#         await self.accept()
#         await self.send(text_data=json.dumps({"message": "Connected to DeepSeek Model"}))

#     async def disconnect(self, close_code):
#         print("Disconnected:", close_code)

#     async def receive(self, text_data):
#         try:
#             data = json.loads(text_data)
#             prompt = data.get("prompt", "").strip()

#             if not prompt:
#                 await self.send(text_data=json.dumps({"error": "Prompt is empty"}))
#                 return

#             # Start streaming tokens from the DeepSeekLangChain generator
#             loop = asyncio.get_event_loop()
#             generator = deepseek_model.generate(prompt)

#             async def stream_tokens():
#                 print("Starting token stream...",generator)
#                 for token in generator:

#                     print("Streaming token:", token)
#                     await self.send(text_data=json.dumps({"token": token}))
#                     await asyncio.sleep(0)  # yield control to event loop

#                 await self.send(text_data=json.dumps({"token": "[END]"}))  # Send end signal

#             await stream_tokens()

#         except Exception as e:
#             await self.send(text_data=json.dumps({"error": str(e)}))
# import json
# from channels.generic.websocket import AsyncWebsocketConsumer
# from deepseek.deepseek_engine import deepseek_model
# import asyncio

# class ChatConsumer(AsyncWebsocketConsumer):
#     async def connect(self):
#         await self.accept()
#         await self.send(text_data=json.dumps({"message": "Connected to DeepSeek Model"}))

#     async def disconnect(self, close_code):
#         print("Disconnected:", close_code)

#     async def receive(self, text_data):
#         try:
#             data = json.loads(text_data)
#             prompt = data.get("prompt", "").strip()

#             if not prompt:
#                 await self.send(text_data=json.dumps({"error": "Prompt is empty"}))
#                 return

#             generator = deepseek_model.generate(prompt, stream=True)

#             async def stream_tokens():
#                 for token in generator:
#                     print("Streaming token:", token)
#                     await self.send(text_data=json.dumps({"token": token}))
#                     await asyncio.sleep(0)  # Yield control to event loop

#                 await self.send(text_data=json.dumps({"token": "[END]"}))

#             await stream_tokens()

#         except Exception as e:
#             await self.send(text_data=json.dumps({"error": str(e)}))

import asyncio
import json
from channels.generic.websocket import AsyncWebsocketConsumer
from deepseek.deepseek_engine import deepseek_model
from threading import Thread


class ChatConsumer(AsyncWebsocketConsumer):
    async def connect(self):
        self.session_id = self.scope['url_route']['kwargs'].get('session_id', 'default')
        self.group_name = f"stream_{self.session_id}"
        await self.channel_layer.group_add(self.group_name, self.channel_name)
        await self.accept()
        await self.send(text_data=json.dumps({
            "message": f"Connected to DeepSeek model for session: {self.session_id}"
        }))

    async def disconnect(self, close_code):
        await self.channel_layer.group_discard(self.group_name, self.channel_name)

    async def receive(self, text_data):
        data = json.loads(text_data)
        prompt = data.get("prompt", "").strip()
        if not prompt:
            await self.send(text_data=json.dumps({"error": "Prompt is empty"}))
            return
        await self.stream_tokens(prompt)

    async def stream_tokens(self, prompt):
        queue = asyncio.Queue()
        loop = asyncio.get_event_loop()  # Get current loop to use in thread

        def run_generation():
            try:
                for token in deepseek_model.generate(prompt):
                    # Push token to the async queue from the thread safely
                    asyncio.run_coroutine_threadsafe(queue.put(token), loop)
                asyncio.run_coroutine_threadsafe(queue.put("[END]"), loop)
            except Exception as e:
                asyncio.run_coroutine_threadsafe(queue.put(f"[ERROR] {str(e)}"), loop)

        # Start generation in a background thread
        thread = Thread(target=run_generation)
        thread.start()

        # Consume from queue and send to WebSocket
        while True:
            token = await queue.get()
            await self.send(text_data=json.dumps({"token": token}))
            if token == "[END]" or token.startswith("[ERROR]"):
                break
