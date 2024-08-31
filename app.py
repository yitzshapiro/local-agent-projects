from llama_index.llms.ollama import Ollama
from llama_index.core.bridge.pydantic import BaseModel, Field
from llama_index.core.tools import FunctionTool
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.core.llms import ChatMessage
from mem0 import Memory
from typing import List
import json
import whisper
import os
from mem0_config import config, AGENT_ID

m = Memory.from_config(config)


class Task(BaseModel):
    name: str = Field(..., description="The name of the task")
    description: str = Field(..., description="The description of the task")
    platform: str = Field(..., description="The platform of the task")
    deadline: str = Field(..., description="The deadline of the task")
    status: str = Field(..., description="The status of the task")
    source: str = Field(..., description="The source of the task")
    requester: str = Field(..., description="The requester of the task")
    addToMemory: bool = Field(..., description="Whether to add the task to the memory")
    memory: str = Field(
        ..., description="The memories, listed as a string, to add the task to"
    )


class TaskList(BaseModel):
    tasks: List[Task] = Field(..., description="The list of tasks")


llm = Ollama(
    model="llama3.1:latest",
    base_url="http://localhost:11434",
    is_function_calling_model=False,
)
embed_model = OllamaEmbedding(
    model_name="llama3.1:latest", base_url="http://localhost:11434"
)

sllm = llm.as_structured_llm(TaskList)


def transcribe_audio(audio_path):
    model = whisper.load_model("base")
    result = model.transcribe(audio_path)
    return result["text"]


def save_json_output(file_path, data):
    if not os.path.exists(file_path):
        with open(file_path, "w") as file:
            json.dump([], file)

    with open(file_path, "r") as file:
        existing_data = json.load(file)

    existing_data.append(data)

    with open(file_path, "w") as file:
        json.dump(existing_data, file, indent=2)


audio_path = input("Enter the path to the audio file (m4a): ")

try:
    transcription = transcribe_audio(audio_path)
    print("Transcription:")
    print(transcription)

    response = sllm.chat(
        [
            ChatMessage(
                role="user",
                content=f"""Based on the following information, create a list of tasks that need to be done. 
                For each task, provide a name, description, platform, deadline, status, source, and requester and output in JSON format. 
                You will also add a list of memories as one long string to the task if addToMemory is true. Here's the information: {transcription}""",
            )
        ]
    )
    print("Raw response:")
    print(response)

    response_str = str(response)
    json_start = response_str.find("{")
    json_end = response_str.rfind("}") + 1
    json_content = response_str[json_start:json_end]

    tasks_dict = json.loads(json_content)

    print("Parsed response:")
    print(json.dumps(tasks_dict, indent=2))

    json_output_path = "tasks_output.json"
    save_json_output(json_output_path, tasks_dict)

    for task in tasks_dict.get("tasks", []):
        if task.get("addToMemory", False):
            m.add(task.get("memory", ""), agent_id=AGENT_ID)
            print(f"Added to memory: {task.get('memory', '')}")

except Exception as e:
    print(f"An error occurred: {e}")
