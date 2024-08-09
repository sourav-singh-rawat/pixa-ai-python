from pysilero_vad import SileroVoiceActivityDetector
from src.llm import LLM
from src.stt import STT
from src.tts import TTS
import asyncio

SPEECH_TIMEOUT = 1.5
VAD_THRESHOLD = 0.2

class AudioProcessor:
    def __init__(self):
        self.stt = STT(self.get_conversation_transcript,self.update_converstaion_transcript)
        self.llm = LLM(self.update_converstaion_transcript)
        self.tts = TTS()
        self.vad = SileroVoiceActivityDetector()
        self.speech_buffer = []
        self.last_speech_time = None
        self._coversation_transcript = []
        
    async def is_speech(self, chunk):
        is_speech = self.vad(chunk) >= VAD_THRESHOLD
        if is_speech:
            self.last_speech_time = asyncio.get_event_loop().time()
        return is_speech

    async def process_chunk(self, chunk):   
        self.speech_buffer.append(chunk)
        self.stt.feed_chunk(chunk)

    async def process_speech(self):
        stt_response = self.stt.transcribe()
        if stt_response == "":
            return None
                
        llm_response = self.llm.feed_content(stt_response)
        tts_response = self.tts.feed_text(llm_response)
        
        self.reset()
        return tts_response
    
    def reset(self):
        self.speech_buffer.clear()
        self.last_speech_time = None
        
    def get_conversation_transcript(self) -> list:
        return self._coversation_transcript
        
    def update_converstaion_transcript(self,conversation_transcript:list) -> None:
        self._coversation_transcript.extend(conversation_transcript)
    
    async def is_silence_timeout(self):
        if self.last_speech_time is None:
            return False
        
        current_time = asyncio.get_event_loop().time()
        return (current_time - self.last_speech_time) >= SPEECH_TIMEOUT