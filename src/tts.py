from openai import OpenAI
import wave
import os

class TTS:         
    def __init__(self):
        api_key = os.getenv("OPENAI_API_KEY")
        
        self.client = OpenAI(api_key=api_key)
        
    def feed_text(self, text_data):
        try:
            response = self.client.audio.speech.create(
                model="tts-1",
                voice="alloy",
                input=text_data,
                response_format="pcm",
            )
            
            return response
        except Exception as e:
            print(f"\n\n[TTS]:[feed_text] Response Error: {e}\n\n")
            raise e
    
    def _write_to_wav(self,chunks):
        SAMPLE_RATE = 11000
        SAMPLE_WIDTH = 2
        CHANNELS = 2

        """Write audio chunks to a WAV file."""
        with wave.open(f"output{len(chunks)}.wav", 'wb') as wav_file:
            wav_file.setnchannels(CHANNELS)
            wav_file.setsampwidth(SAMPLE_WIDTH)
            wav_file.setframerate(SAMPLE_RATE)
            for chunk in chunks:
                wav_file.writeframes(chunk)
       