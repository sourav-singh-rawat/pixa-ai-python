import asyncio
import websockets
from src.audio_processor import AudioProcessor
from src.helper import CHUNK_SIZE

class WebSocketServer:
    def __init__(self, host, port):
        self.host = host
        self.port = port
        self.processor = AudioProcessor()

    async def handle_client(self, websocket, path):
        try:
            async for message in websocket:
                await self.process_message(websocket, message)
        except websockets.exceptions.ConnectionClosed:
            print("[WebSocketServer]:[handle_client]: WebSocket connection closed")
        except Exception as e:
            print(f"[WebSocketServer]:[handle_client]: Error in processing: {e}")
            raise e

    async def process_message(self, websocket, message):
        if isinstance(message, str):
            message = message.encode('utf-8')

        if not isinstance(message, bytes):
            raise ValueError(f"[WebSocketServer]:[process_message]: Expected string or bytes, got {type(message)}")

        if message == b"stop-consuming":
            await self.handle_end_of_speech(websocket)
        else:
            await self.handle_audio_chunk(websocket, message)

    async def handle_end_of_speech(self, websocket):
        print("[WebSocketServer]:[handle_end_of_speech] End of speech signal received")
        await self.send_processed_speech(websocket)

    async def handle_audio_chunk(self, websocket, chunk):
        is_speech = await self.processor.is_speech(chunk)
        
        if is_speech:
            await self.handle_speech(chunk)
        else:
            await self.handle_silence(websocket)
                        
    async def handle_speech(self,chunk):
        print("[WebSocketServer]:[handle_speech]: Speech detected")
        await self.processor.process_chunk(chunk)
        
    async def handle_silence(self,websocket):
        print("[WebSocketServer]:[handle_silence]: Silence detected")
        
        if await self.processor.is_silence_timeout():
            await self.send_processed_speech(websocket)

    async def send_processed_speech(self, websocket):
        response = await self.processor.process_speech()
        if response:
            await websocket.send("start-generative-response")
            for response_chunk in response.iter_bytes(chunk_size=CHUNK_SIZE):
                await websocket.send(response_chunk)
            await websocket.send("stop-consuming")

    def run(self):
        try:
            start_server = websockets.serve(self.handle_client, self.host, self.port)
            asyncio.get_event_loop().run_until_complete(start_server)
            asyncio.get_event_loop().run_forever()
        except Exception as e:
            print(f"\n\n[STT]:[run]: Error: {e}\n\n")
            raise e