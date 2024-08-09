from deepgram import (
    DeepgramClient,
    DeepgramClientOptions,
    LiveTranscriptionEvents,
    LiveOptions,
)
from typing import Callable
from fireworks.client import Fireworks
import json
import os

class STT: 
    config = DeepgramClientOptions(
        options={
            "keepalive": "true",
        }
    )
    
    def __init__(self,get_converstaion_transcript_callback:Callable[[], list],update_conversation_transcript_callback: Callable[[list], None]):
        try:
            print("[STT]:[init]")
            
            self.words_transcripted = []
            self.text_generated = ""
                    
            api_key = os.getenv("DEEPGRAM_API_KEY")
        
            self.deepgram = DeepgramClient(api_key, self.config)
            
            self.dg_connection = self.deepgram.listen.live.v("1")
            
            self._configure_deepgram()
            
            FIREWORKS_API_KEY = os.getenv("FIREWORKS_API_KEY")
            
            self.fireworks_client = Fireworks(api_key=FIREWORKS_API_KEY)
            
            self._get_converstaion_transcript_callback = get_converstaion_transcript_callback
            self._update_conversation_transcript_callback = update_conversation_transcript_callback
        except Exception as e:
            print(f"\n\n[STT]:[_init]: Error: {e}\n\n")
            raise e
                        
    def _configure_deepgram(self):
        print("[STT]:[_configure_deepgram] Configuring deepgram connection")
        try:
            self.dg_connection.on(LiveTranscriptionEvents.Transcript, self._on_message)
            self.dg_connection.on(LiveTranscriptionEvents.Metadata, self._on_metadata)
            self.dg_connection.on(LiveTranscriptionEvents.Error, self._on_error)

            options = LiveOptions(
                model="nova-2-general", 
                language="en-US", 
                encoding="linear16",
                sample_rate="16000",
                smart_format=True,
                diarize= True,
                punctuate= True, 
                filler_words= True,
                endpointing= True, 
                multichannel= False,
                alternatives= 1, 
                interim_results= True,
            )
                        
            self.dg_connection.start(options)            
        except Exception as e:
            print(f"\n\n[STT]:[_configure_deepgram]: Error: {e}\n\n")
            raise e
        
    def feed_chunk(self,bytes_data):
        try:            
            self.dg_connection.send(bytes_data)
        except Exception as e:
            print(f"\n\n[STT]:[feed_chunk]: Error: {e}\n\n")
            raise e              
    
    def transcribe(self):  
        try:       
            print("\n[STT]:[transcribe]: End signal detected")
            print(f"\n[STT]:[transcribe]: full sentence {self.text_generated}\n")

            final_text = self.text_generated
            self.text_generated = ""
            self.words_transcripted = []
            
            return final_text
        except Exception as e:
                print(f"\n\n[STT]:[transcribe]: Error: {e}\n\n")
                raise e
            
    ################################# With Completion only ###############################
         
    def render_transcript(self, transcript):
        previous_speaker = None
        transcript_text = ""
        if not transcript or len(transcript) == 0:
            return ""
        
        for word in transcript:
            if previous_speaker != word['speaker']:
                transcript_text += f"\n{word['speaker']}: "
                previous_speaker = word['speaker']
            transcript_text += word['punctuated_word'] + " "
        return transcript_text.strip()
                
    def _on_message(self, *args, **kwargs):
        try:
            candidate = kwargs.get('result').channel.alternatives[0]
            
            if len(candidate.words) > 0:
                is_final = kwargs.get('result').is_final
                
                if is_final:
                    sentance = candidate.transcript
                    print(f"new sentance: {sentance}")
                    
                    # conversation_transcript = [
                    #     {
                    #         'speaker': 'Speaker 1',
                    #         'punctuated_word': word.punctuated_word,
                    #     }
                    #     for word in candidate.words
                    # ]
                    
                    # self._update_conversation_transcript_callback(conversation_transcript)
                    
                    
                    # has_speaker_0 = any('Speaker 0' in item.values() for item in conversation_transcript)
                    
                    # if has_speaker_0:
                        # self.text_generated = sentance
                    # else:
                    #     self.predict()
                    
                    
                    self.words_transcripted.extend(word.punctuated_word for word in candidate.words)
                    
                    self.predict()
            
        except Exception as e:
            print(f"\n\n[STT]:[_on_message]: Error: {e}\n\n")
            raise e
        
    def __parsed_output(self,data):
        start = data.find('{')
        if start == -1:
            return None
        
        count = 1
        for i in range(start + 1, len(data)):
            if data[i] == '{':
                count += 1
            elif data[i] == '}':
                count -= 1
            
            if count == 0:
                return json.loads(data[start:i+1])
    
        return None

    def predict(self):        
        # conversation_transcript = self._get_converstaion_transcript_callback()
                
        # transcript = self.render_deepgram_words_collection_transcript(conversation_transcript)
        
        transcript = " ".join(self.words_transcripted)
        
        print(f"transcript: {transcript}")
        
        prompt = (
        f"""
            <s> [INST]

            A Dialog is provided to you which may or may not be completed. Completion is marked with punctuation signs such as period, question mark or exclamation mark.
            A Dialog that lacks context but is marked with  period, question mark or exclamation mark, is regarded as complete only.

            Your role is to determine the most suitable words to complete the incomplete Dialog, maintaining a conversational tone. Then Output the complete predicted sentence as per JSON Object with the key "output" enclosed in curly brackets.

            JSON OBJECT FORMAT:

            {{ "output" : "NEXT_WORDS_OF_THE_DIALOG" }}

            (Use output examples to understand the format and structure and not to base the content)

            Output Example 1:
            Dialog : "How are you ".
            {{ "output" : "How are you going to shift the sofa?" }}
            [Note: In this example since the sentence is grammatically incomplete, the output contains the completed sentence as whole.]

            Output Example 2:
            Dialog : "How are you doing that?".
            {{ "output" : "How are you doing that?" }}
            [Note: In this example since the sentence is already grammatically complete, the output will be the same sentence.]

            Output Example 3:
            Dialog : "Hello!"
            {{ "output" : "Hello!" }}
            [Note: In this example since the sentence is already grammatically complete regardless of its contextual meaning, the output will be the same sentence.]

            Below given is the Dialog delimited by # .

            Dialog: #{transcript}#
            [/INST] </s>
        """
        )
        
        # completion = self.openai_client.completions.create(
        #     model="mixtral-8x7b-instruct",
        #     prompt=prompt,
        #     # stop=[ 'Speaker 1', 'Speaker 2', 'Speaker 3', 'Speaker 4'],
        #     max_tokens=4096,
        # )
        
        # prediction = completion.choices[0].message.content
        # print(f"\n\nprediction: {prediction}\n\n")
        
        completion = self.fireworks_client.completion.create(
            model="accounts/fireworks/models/mixtral-8x7b-instruct",
            prompt=prompt,
            max_tokens=4096,
            top_p=0.3,
            top_k=16,
            presence_penalty=2,
            frequency_penalty= 2,
            temperature=0.1,
        )
        
        parsed_response = self.__parsed_output(completion.choices[0].text)
        if parsed_response:
            prediction = parsed_response['output']
            if prediction:
                print(prediction)
                
                self.text_generated = prediction
            
    ##################### With Chat Completion (need to update according to above later changes) ##################3
    
    # def render_transcript(self, transcript):
    #     result = []
    #     previous_speaker = None
    #     transcript_text = ""
    #     if not transcript or len(transcript) == 0:
    #         return []  # Return an empty list instead of an empty string
        
    #     for word in transcript:
    #         if previous_speaker != word['role']:
    #             if previous_speaker is not None:
    #                 result.append({"role": previous_speaker, "content": transcript_text.strip()})
    #                 transcript_text = ""
    #             previous_speaker = word['role']
    #         transcript_text += word['punctuated_word'] + " "
        
    #     # Don't forget to append the last segment
    #     if previous_speaker is not None:
    #         result.append({"role": previous_speaker, "content": transcript_text.strip()})
        
    #     return result
                
    # def _on_message(self, *args, **kwargs):
    #     try:
    #         candidate = kwargs.get('result').channel.alternatives[0]
            
    #         if len(candidate.words) > 0:
    #             is_final = kwargs.get('result').is_final
                
    #             if is_final:
    #                 conversation_transcript = [
    #                     {
    #                         'role': 'user',
    #                         'punctuated_word': word.punctuated_word,
    #                     }
    #                     for word in candidate.words
    #                 ]
                    
    #                 self._update_conversation_transcript_callback(conversation_transcript)
                    
    #                 self.predict()
            
    #     except Exception as e:
    #         print(f"\n\n[STT]:[_on_message]: Error: {e}\n\n")
    #         raise e

    # def predict(self):
    #     conversation_transcript = self._get_converstaion_transcript_callback()
        
    #     transcript = self.render_transcript(conversation_transcript)
        
    #     print(transcript)
        
    #     system_prompt = (
    #         "Given the following conversation transcript:\n\n"
    #         "Focus on predicting what (the user) might say next or how they might complete their current thought. "
    #         "Consider the following guidelines:\n"
    #         "1. Only predict user's next words or completion of their current statement.\n"
    #         "2. Take into account the entire conversation context.\n"
    #         "3. The user may not have finished speaking, so anticipate possible completions or continuations.\n"
    #         "4. If the user's last statement seems complete, suggest a natural continuation of their thoughts.\n"
    #         "5. Maintain consistency with the user's speaking style and the conversation topic.\n"
    #         "6. Do not generate responses for other user.\n\n"
    #         "Predicted continuation or next statement for user:"
    #     )
        
    #     messages = [{"role": "system", "content": system_prompt}]
    #     messages.extend(transcript)
        
    #     completion = self.openai_client.chat.completions.create(
    #         model="text-davinci-002",
    #         messages=messages,
    #         max_tokens=100,
    #     )
        
    #     prediction = completion.choices[0].message.content  # For chat completions, use message.content
    #     print(f"\n\nprediction: {prediction}\n\n")
        
    #     self.text_generated = prediction

    def _on_metadata(self, *args, **kwargs):
        print(f"\n\n[STT]:[_on_metadata]: Meta-Data:{kwargs.get('metadata')}\n\n")

    def _on_error(self,  *args, **kwargs):
        print(f"\n\n[STT]:[_on_error]: Meta-Data:{kwargs.get('error')}\n\n")
    
    def dispose(self):
        try:
            self.dg_connection.finish()
        except Exception as e:
            print(f"\n\n[STT]:[dispose]: Error: {e}\n\n")
            raise e
            