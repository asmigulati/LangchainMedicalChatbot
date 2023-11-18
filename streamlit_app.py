import streamlit as st
import re
import pandas as pd
import numpy as np
import tensorflow_hub as hub
import datetime
from langchain.tools import BaseTool
from langchain.schema import SystemMessage
from langchain.chat_models import ChatOpenAI
from langchain.agents import OpenAIMultiFunctionsAgent, AgentExecutor
from langchain.memory import ConversationBufferMemory
from langchain.prompts import MessagesPlaceholder
import openai
from langchain.llms import OpenAI
from langchain.agents import AgentType, initialize_agent, load_tools
from langchain.callbacks import StreamlitCallbackHandler
# Initialize the model
openai.api_key = st.secrets["OPENAI_API_KEY"]
@st.cache(allow_output_mutation=True)
def load_model():
    module_url = "https://tfhub.dev/google/universal-sentence-encoder/4"
    model = hub.load(module_url)
    print(f"Module {module_url} loaded")
    return model
#Utilities
model= load_model()
class VectorDatabase:
    def __init__(self):
        # Load the Universal Sentence Encoder model
        self.model = model
        self.df = pd.DataFrame()

    def embed(self, input_text):
        # Embed the input text using the model
        return self.model(input_text).numpy()

    def add_entries(self, timestamp_list, activity_list):
        # Add new entries to the database
        new_data = {'Timestamp': timestamp_list, 'Activity': activity_list}
        new_df = pd.DataFrame(new_data)
        new_df['Embed'] = list(self.embed(new_df['Activity']))
        self.df = pd.concat([self.df, new_df], ignore_index=True)

    def cosine_similarity(self, A, B):
        # Calculate cosine similarity
        dot_product = np.dot(A, B)
        norm_A = np.linalg.norm(A)
        norm_B = np.linalg.norm(B)
        return dot_product / (norm_A * norm_B)

    def query_by_text(self, input_text, lim=5):
        # Query the database by text
        input_embedding = self.embed([input_text])[0]
        self.df['Similarity'] = self.df['Embed'].apply(lambda x: self.cosine_similarity(x, input_embedding))
        result = self.df.sort_values(by="Similarity", ascending=False)['Activity']
        return result.head(lim).tolist()

    def query_by_timestamp(self, start_time, end_time):
        # Query the database by timestamp range
        mask = (self.df['Timestamp'] >= start_time) & (self.df['Timestamp'] <= end_time)
        return self.df.loc[mask]['Activity'].tolist()
def read_text_file(file_path):
    """
    Reads a text file and returns its content as a string.

    Parameters:
    file_path (str): The path to the text file to be read.

    Returns:
    str: The content of the file.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()
    except FileNotFoundError:
        return "The file was not found."
    except Exception as e:
        return f"An error occurred: {e}"
class MemGPT:
    def __init__(self, file_name):
        # Some required variables
        self.max_no_of_tokens = 8000
        self.current_no_of_tokens = 300
        # main context management
        self.core_memory = ''  # it alters the system_instruction
        self.system_instruction = read_text_file(
            file_name) + 'Core Memory Section (If you have appended here already, no Need to append again!:\n'
        self.conversational_context = []
        # External Context
        self.recall_db = VectorDatabase()
        self.archival_storage = VectorDatabase()

    # a basic function to keep track of tokens
    def token_counter(self, text):
        num_characters = len(text)
        estimated_tokens = num_characters / 4
        return estimated_tokens

    # functions that the agent can call!
    def conversation_search(self, query="", query_by="", start_time=0, end_time=0):
        if query:
            return ' '.join(self.recall_db.query_by_text(query))
        else:  # if query is by time
            return ' '.join(self.recall_db.query_by_timestamp(start_time, end_time))

    def send_message(self, message):
        self.recall_db.add_entries([datetime.datetime.now()], [message])
        return message

    def core_memory_append(self, text_to_append):
        self.system_instruction += text_to_append

    def core_memory_replace(self, text_to_replace, text_to_be_filled_in):
        self.core_memory.replace(text_to_replace, text_to_be_filled_in)

    def archival_memory_insert(self, memory):
        self.archival_storage.add_entries([datetime.datetime.now()], [memory])

    def archival_memory_search(self, memory="", by="", start_time=0, end_time=0):
        if memory:
            return ' '.join(self.archival_storage.query_by_text(memory))
        else:  # if query is by time
            return ' '.join(self.archival_storage.query_by_timestamp(start_time, end_time))

memgpt=MemGPT('medgpt_chat.txt')
class SendMessageTool(BaseTool):
    name = "send_message"
    description = "ONLY Way to send a message to the human"
    def _run(self,user_message: str) -> str:
        return memgpt.send_message(user_message)
class ConversationSearchTool(BaseTool):
    name = "conversation_search"
    description = "useful to search past conversation history in the recall storage storage"
    def _run(self, query: str = "", query_by: str = "", start_time: int = 0, end_time: int = 0) -> str:
        if query:
            return memgpt.conversation_search(query=query)
        elif query_by == 'time':
            return memgpt.conversation_search(start_time=start_time, end_time=end_time)
        else:
            return "No valid query provided"
class CoreMemoryAppendTool(BaseTool):
    name="core_memory_append"
    description = "add useful details or notes about the user or conversation to the core memory"
    def _run(self, TextToAppend: str) -> str:
        memgpt.core_memory_append(TextToAppend)
        return "Done! Appended to core memory"
class CoreMemoryReplaceTool(BaseTool):
    name="core_memory_replace"
    description = "add useful details or notes about the user or conversation to the core memory"
    def _run(self, TextToReplace: str,TextToReplaceWith: str) -> str:
        memgpt.core_memory_replace(TextToReplace,TextToReplaceWith)
        return "Done! replaced in core memory"
class ArchivalMemoryInsertTool(BaseTool):
    name="archival_memory_insert"
    description = "add details that should be stored long term"
    def _run(self, memory: str) -> str:
        memgpt.archival_memory_insert(memory)
        return "Done!added to archival memory"
class ArchivalMemorySearchTool(BaseTool):
    name = "archival_memory_search"
    description = "Retrieve memories from long term data storage (archival storage)"
    def _run(self, query: str = "", query_by: str = "", start_time: int = 0, end_time: int = 0) -> str:
        if query:
            return memgpt.archival_memory_search(query=query)
        elif query_by == 'time':
            return memgpt.archival_memory_search(start_time=start_time, end_time=end_time)
        else:
            return "No valid query provided"
tools_list = [
    SendMessageTool(),
    ConversationSearchTool(),
    CoreMemoryAppendTool(),
    CoreMemoryReplaceTool(),
    ArchivalMemoryInsertTool(),
    ArchivalMemorySearchTool()]
llm = ChatOpenAI(temperature=0, model="gpt-4-0613")
_system_message = memgpt.system_instruction
memory = ConversationBufferMemory(memory_key="memory", return_messages=True)
prompt = OpenAIMultiFunctionsAgent.create_prompt(
    system_message=SystemMessage(content=_system_message),
    extra_prompt_messages=[MessagesPlaceholder(variable_name="memory")]
    )

agent = OpenAIMultiFunctionsAgent(llm=llm, tools=tools_list, prompt=prompt)

agent_executor = AgentExecutor(
    agent=agent,
    tools=tools_list,
    memory=memory,
    verbose=True,
)
if prompt := st.chat_input():
    st.chat_message("user").write(prompt)
    with st.chat_message("assistant"):
        st_callback = StreamlitCallbackHandler(st.container())
        response = agent_executor.run(prompt, callbacks=[st_callback])
        st.write(response)
