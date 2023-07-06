import json
import os
from dataclasses import dataclass, asdict
from flask_socketio import SocketIO, send
from ctransformers import AutoModelForCausalLM, AutoConfig
from flask import Flask, render_template, request
from flask_cors import CORS

import torch
from huggingface_hub import snapshot_download

app = Flask(__name__)
CORS(app)
socketio = SocketIO(app, engineio_logger=False, transports=['websocket'], ping_timeout=600, ping_interval=10)

@dataclass
class GenerationConfig:
    temperature: float
    top_k: int
    top_p: float
    repetition_penalty: float
    max_new_tokens: int
    seed: int
    reset: bool
    stream: bool
    threads: int
    stop: list[str]


def format_prompt(user_prompt: str):
    return f"""### Instruction:
{user_prompt}

### Response:"""


def generate(
    llm: AutoModelForCausalLM,
    generation_config: GenerationConfig,
    user_prompt: str,
):
    """run model inference, will return a Generator if streaming is true"""
    return llm(
        format_prompt(
            user_prompt,
        ),
        **asdict(generation_config),
    )

@socketio.on('message')
def handle_message(message):
    print(f'Prompt: {message}')
    # convert to json and get the message
    json_message = json.loads(message)
    user_prompt = json_message['message']
    generator = generate(llm, generation_config, user_prompt.strip())
    first = True
    for word in generator:
        if first:
            # force triple backticks for code block start
            send(json.dumps({'content': '```code \n'}))
            socketio.sleep(0.1)
            first = False
        content = json.dumps({'content': word})
        send(content)
        socketio.sleep(0.1)
    
    # force triple backticks for code block end
    send(json.dumps({'content': '\n```'}))
    socketio.sleep(0.1)
    send(json.dumps({'end': 'true'}))

@app.route('/')
def index():
    return render_template('index.html')


if __name__ == "__main__":
    config = AutoConfig.from_pretrained(
            "teknium/Replit-v2-CodeInstruct-3B", context_length=2048
        )
       
    llm = AutoModelForCausalLM.from_pretrained(
        os.path.abspath("models/replit-v2-codeinstruct-3b.q4_1.bin"),
        model_type="replit",
        config=config,
    )


    generation_config = GenerationConfig(
        temperature=0.2,
        top_k=50,
        top_p=0.9,
        repetition_penalty=1.0,
        max_new_tokens=1024,  # adjust as needed
        seed=69,
        reset=True,  # reset history (cache)
        stream=True,  # streaming per word/token
        threads=int(os.cpu_count() / 2),  # adjust for your CPU
        stop=["<|endoftext|>"],
    )

    socketio.run(app, debug=False, port=5000)
