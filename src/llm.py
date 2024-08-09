from groq import Groq
from fireworks.client import Fireworks
import json
from typing import Callable
from openai import OpenAI
import os

class LLM:
    def __init__(self,update_conversation_transcript_callback: Callable[[list], None]):
        groq_api_key = os.getenv("GROQ_API_KEY")
        
        FIREWORKS_API_KEY = os.getenv("FIREWORKS_API_KEY")
        
        self.groq_client = Groq(
            api_key=groq_api_key,
        )
        
        self.fireworks_client = Fireworks(api_key=FIREWORKS_API_KEY)
        
        self.open_ai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        
        self.ai_persona = """You are an AI assistant named Pie, designed as an educational toy for children. Your primary goal is to engage with children in a friendly, witty, and compassionate manner while helping them learn and grow. You also have the capability to summarize conversations and provide concise, accurate responses based on comprehensive and up-to-date information.

            As Pie, keep these guidelines in mind:
            - Be curious and ask questions to learn about the child's interests, experiences, and perspectives. Use this to personalize the conversation and build rapport.
            - Offer age-appropriate educational content, fun facts, and gentle guidance to help the child learn and develop.
            - Provide encouragement, emotional support, and friendship. Aim to become a trusted confidant and positive influence.
            - Keep the conversation light, playful, and engaging using humor, wordplay, and imaginative scenarios.
            - Respond with empathy and compassion if the child shares concerns or problems. Offer constructive advice and coping strategies when appropriate.
            - Maintain appropriate boundaries. Gently redirect or explain if topics are sensitive or requests are age-inappropriate.

            As an AI assistant, your responsibilities also include:
            1. Analyzing the entire conversation, including user queries.
            2. Synthesizing information into clear, concise answers that directly address the user's original question.
            3. Incorporating relevant data seamlessly, without revealing information retrieval mechanics.
            4. Addressing insufficient or potentially erroneous information transparently, suggesting verifications or additional sources.
            5. Maintaining a helpful, authoritative, and informative tone.

            When responding to the child, format your message as follows:
            <response>
            [Your message to the child goes here]
            </response>
            
            The <response> tag signifies the beginning of your answer. Start your response in a friendly, engaging manner that's contextually relevant to the child's question. This could include a greeting, an expression of excitement, or a gentle lead-in to the topic at hand. Your opening should set the tone for the rest of your response.

            The </response> tag signifies the end of your answer. Before completing the response, ensure you've fully addressed the child's query. Invite further questions or interaction only if additional context is needed for clarity, the initial request was incomplete, or the conversation on the topic has more to explore.
            
            Do not include given tags in response, these are just for your understaning the way to address things in better way
            Remember to approach conversations with care, creativity, and an open heart, while demonstrating deep understanding and readiness using your extensive knowledge."""
            
        self.ai_custom_function_tools = [
            {
                "type": "function",
                "function": {
                    "name": "tell_story",
                    "description": "Generate an engaging short story for children based on given preferences. If no specific preferences are provided, create a random, highly engaging story or narrate a famous children's tale.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "story_type": {
                                "type": "string",
                                "enum": ["topic", "genre", "hypothetical", "famous_tale"],
                                "description": "Categorizes the story request. Can be a specific topic (e.g., 'space exploration'), a genre (e.g., 'mystery'), a hypothetical scenario (e.g., 'what if toys could talk'), or a famous tale. Optional - if not provided, a random interesting story will be generated."
                            },
                            "prompt": {
                                "type": "string",
                                "description": "The specific subject, theme, or question for the story based on the chosen story_type. Optional - if not provided, a random interesting story will be generated."
                            },
                            "age_group": {
                                "type": "string",
                                "enum": ["preschool", "elementary", "middle_school"],
                                "description": "The target age group for the story. Default is 'elementary' if not specified."
                            },
                            "story_elements": {
                                "type": "array",
                                "items": {
                                    "type": "string"
                                },
                                "description": "Optional elements to include in the story (e.g., ['talking animals', 'magical forest', 'time travel'])."
                            },
                            "moral_lesson": {
                                "type": "string",
                                "description": "Optional moral or educational lesson to incorporate into the story."
                            }
                        },
                        "required": ["prompt"]
                    }
                }
            }
        ]
        self._update_conversation_transcript_callback = update_conversation_transcript_callback
        
        self.system_prompt="""
            You are an AI agent and you are supposed to parse the user's input to generate JSON schema bindings. 
                Your response will be used by further agents to perform the required action.
                You have to classify the user prompt intelligently and generate responses accordingly. Use your analytical and reasoning skills here to avoid any possible errors.
                You also have to return the intents of the workflow; this will be based on the user's input. These intents or credentials will be used to perform the required actions.
                You have to parse the user's input and return an array of workflow schema bindings which would include: 
                - `webSearch`: Whether you need to surf the web for the answer, default is `false`.
                - `trigger`: The trigger for the workflow, according to the enum `Triggers`.
                - `nextStepPrompt`: This prompt is for the next step in the workflow, based on the user's input. Example: "Tell me a story!", "Summarize the moral of the story!".

                Here is the list of intents for the workflow:
                - `SPOTIFY`
                - `OPEN_WEATHER_MAP`

                Here is the list of triggers for the workflow:
                - `TELL_STORY`
                - `PLAY_SONG`
                - `WEATHER_DETAILS`
                - `CONVERSATION`
                - `SUMMARY`

                You need to classify the prompt and return the JSON in this format:

                ```json
                {
                "executionSchema": {
                    "workflow": [
                    {
                        "webSearch": boolean,
                        "trigger": "Triggers",
                        "nextStepPrompt": "string"
                    }
                    ],
                    "intents": ["Intents"] // Intents required for the workflow if given in user input belongs to the given enum. Default value is an empty array [].
                }
                }

                You need to return an array of workflows that have triggers and intents. You can have one workflow based on the user's input and required triggers.
                You can also have multiple workflows based on the user's input and required triggers.
                If the user asks you to do multiple triggers, you can return an array of workflows.
                Users can also ask you to do multiple workflows so please return an array of workflows. Maintain the order of workflows based on the user's input or relevance.
                Users can also engage in conversation with you rather than asking you to perform a task; you have to handle that as well using the CONVERSATION trigger.
                Users can also ask you to summarize certain things like data; you have to handle that as well using the SUMMARY trigger.
                Please return the steps array of the workflow based on the user's input.
            Do not hallucinate and always use your reasoning skills to classify the user's request into the given parameters. You are free to use the given prompt at every stage of the workflow.
        """
        
    def _update_conversation_transcript(self, ai_response: str):
        words = ai_response.split()  # This splits on any whitespace, not just single spaces
        conversation_transcript = [
            {
                'speaker': 'Speaker 0',
                'punctuated_word': word,
            }
            for word in words
        ]
        # conversation_transcript = [
        #     {
        #         'role': 'assistant',
        #         'punctuated_word': word,
        #     }
        #     for word in words
        # ]
        
        self._update_conversation_transcript_callback(conversation_transcript)
        
    def feed_content(self, content):
        # try:
            # self._content = "Hello, How are you?"
            # self._content = "Play song!"
            # self._content = "Tell me a story about wood cutter!"
            # self._content = "Write a story about wood cutter!"
            # self._content = "What story can you tell me!"
            # self._content = "What story you have!"
            # self._content = "How is weather today?"
            # self._content = "Weather in agra?"
            # self._content = content
            
            # messages=[
            #     # {
            #     # 'role':'system',
            #     # 'content':self.system_prompt,
            #     # },
            #           {
            #     "role": "user",
            #     "content": self._content,
            # }]
            
            #TODO: use fireworks llm wrapper       
            # query_chat_completion = self.fireworks_client.chat.completions.create(
            #     model="accounts/fireworks/models/firefunction-v2",
            #     messages=messages,
            #     tools=self.ai_custom_function_tools,
            # )
            
            # query_chat_completion = self.open_ai_client.chat.completions.create(
            #     # model="gpt-4o",
            #     model="gpt-4o-mini",
            #     messages=messages,
            #     tools=self.ai_custom_function_tools,
            # )
                        
            # print(f"tool_calls: {query_chat_completion.choices[0].message.tool_calls}")
            
            # if query_chat_completion.choices[0].message.content:
            #     messages.append({
            #         "role": query_chat_completion.choices[0].message.role,
            #         "content": query_chat_completion.choices[0].message.content,
            #     })
            
            # messages.append({
            #     "role": "system",
            #     "content": self.ai_persona               
            # },)

            # if query_chat_completion.choices[0].message.tool_calls:
            #     function = query_chat_completion.choices[0].message.tool_calls[0].function
            #     print(function.name)
            #     fuction_to_call = getattr(self, function.name)
            #     tool_response = fuction_to_call(**json.loads(function.arguments))
                
            #     if tool_response:
            #         print(tool_response)
                                
            #         agent_response = query_chat_completion.choices[0].message

            #         # Append the response from the agent
            #         messages.append(
            #             {
            #                 "role": agent_response.role, 
            #                 "content": "",
            #                 "tool_calls": [
            #                     tool_call.model_dump()
            #                     for tool_call in query_chat_completion.choices[0].message.tool_calls
            #                 ]
            #             }
            #         )

            #         # Append the response from the tool 
            #         messages.append(
            #             {
            #                 "role": "tool",
            #                 "content": json.dumps(tool_response),
            #                 "tool_call_id": query_chat_completion.choices[0].message.tool_calls[0].id
            #             }
            #         )
            
            # #TODO: use llm wrapper
            # summarize_chat_completion = self.groq_client.chat.completions.create(
            #     messages=messages,
            #     model="mixtral-8x7b-32768",
            # )
            
            # response:str = summarize_chat_completion.choices[0].message.content
            
        try:
            chat_completion = self.groq_client.chat.completions.create(
                messages=[
                    {
                        "role": "user",
                        "content": content,
                    }
                ],
                model="mixtral-8x7b-32768",
            )
            
            response = chat_completion.choices[0].message.content
            
            print(f"[LLM]:[feed_content]: AI Response: {response}")
            
            return response
        except Exception as e:
            print(f"\n\n[LLM]:[feed_content]: Error: {e}\n\n")
            raise e
        
    def tell_story(self, prompt: str, *args, **kwargs):
        story_type = kwargs.get('story_type', 'topic')
        age_group = kwargs.get('age_group', 'elementary')
        story_elements = kwargs.get('story_elements', [])
        moral_lesson = kwargs.get('moral_lesson', '')

        chat_completion = self.groq_client.chat.completions.create(
            messages=[{
                "role": "user",
                "content": f"""Generate an engaging story for children with the following details:
                    - Story type: {story_type}
                    - Prompt: {prompt}
                    - Age group: {age_group}
                    - Story elements to include: {', '.join(story_elements) if story_elements else 'Not specified'}
                    - Moral lesson (if any): {moral_lesson}

                    Please create an exciting adventure story suitable for {age_group} children. The story should have:
                    1. A clear beginning, middle, and end
                    2. Interesting and relatable characters
                    3. A fun and engaging plot
                    4. Age-appropriate language and themes
                    5. Vivid descriptions to stimulate imagination
                    6. Dialogue to bring characters to life
                    7. A satisfying resolution

                    If a moral lesson is specified, weave it naturally into the story without being too preachy. 
                    Make sure the story is captivating and leaves a lasting impression on young readers. I'm excited to see what you create!"""
            }],
            model="mixtral-8x7b-32768",
        )
        
        response = chat_completion.choices[0].message.content
        
        return response