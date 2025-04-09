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
import json
from channels.generic.websocket import AsyncWebsocketConsumer
from deepseek.deepseek_engine import deepseek_model
import asyncio

class ChatConsumer(AsyncWebsocketConsumer):
    async def connect(self):
        await self.accept()
        await self.send(text_data=json.dumps({"message": "Connected to DeepSeek Model"}))

    async def disconnect(self, close_code):
        print("Disconnected:", close_code)

    async def receive(self, text_data):
        try:
            data = json.loads(text_data)
            prompt = data.get("prompt", "").strip()

            if not prompt:
                await self.send(text_data=json.dumps({"error": "Prompt is empty"}))
                return

            # Start streaming tokens from the DeepSeekLangChain generator
            loop = asyncio.get_event_loop()
            generator = deepseek_model.generate(prompt)

            async def stream_tokens():
                for token in generator:
                    print("Streaming token:", token)
                    await self.send(text_data=json.dumps({"token": token}))
                    await asyncio.sleep(0)  # yield control to event loop

                await self.send(text_data=json.dumps({"token": "[END]"}))  # Send end signal

            await stream_tokens()

        except Exception as e:
            await self.send(text_data=json.dumps({"error": str(e)}))
