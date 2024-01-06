from papote.train import LogCtxLoss
import os
from flask import Flask, render_template
from flask_socketio import SocketIO, send, emit
import time
from typing import Optional
import readline
import torch
import sys
from types import SimpleNamespace
from colored import fg, bg, attr
from papote.bpe import BPE
from papote.model import Transformer, transformer_from_checkpoint
import papote.sampler as S
from papote.utils import OptionsBase

app = Flask(__name__)
socketio = SocketIO(app, logger=True, engineio_logger=True)


class Printer:

    colors = [
        'DarkRed',
        'Crimson',
        'OrangeRed',
        'Coral',
        'LimeGreen',
        'DarkGreen',
    ]

    def __init__(self, send, bpe, separator='', print_prompt=True):
        self.bpe = bpe
        self.separator = separator
        self.print_prompt = print_prompt
        self.send = send
        self.num_sent = 0

    def __call__(self, prompt, next_token, prob, logit):
        if self.num_sent == 0:
            if self.print_prompt:
                self.send(
                    '<b>' +
                    self.bpe.decode_text(prompt, self.separator.encode()) +
                    '</b>')
            self.num_sent += len(prompt)
        color = self.colors[int(min(prob, 0.99) * len(self.colors))]
        self.send('<span style="color: ' + color + '">' +
                  self.bpe.vocab[next_token].decode('utf-8', 'ignore') +
                  '</span>')
        socketio.sleep(0.01)
        self.num_sent += 1


class Options(OptionsBase):
    sep: Optional[str] = ''
    temperature: float = 0.7
    top_k: int = 100
    top_p: float = 0.95
    typical_p: Optional[float] = None
    cfg: Optional[float] = 3
    repeat_penalty: float = 1
    repeat_window: int = 16
    length: int

    def __init__(self, length):
        super().__init__()
        self.length = length


def build_sampler(model, bpe, send, **kwargs):
    sep = kwargs.pop('sep', '')
    sampler = S.default_sampler(model, bpe, **kwargs)
    sampler.event_handler = Printer(send, bpe, sep)
    return sampler


import threading

thread = None
thread_lock = threading.Lock()
current_story = ''


def start_background_thread(req):
    global thread
    global current_story
    opts = Options(model.context_size.item())
    prompt = req['prompt']
    opts.cfg = float(req.get('cfg', opts.cfg))
    opts.temperature = 0.3

    def ontoken(token):
        global current_story
        socketio.emit('token_generated', token)
        current_story += token

    sampler = build_sampler(model, bpe, ontoken, **vars(opts))
    current_story = ''
    socketio.emit('start', None)
    with torch.inference_mode():
        sampler.sample(prompt)
    socketio.emit('token_generated', ' EOF')
    socketio.emit('end', None)
    filename = str(int(time.time())) + ''.join(
        c for c in prompt if c.isalnum() or c in ' _-')
    with open(f'generated/{filename}.txt', 'w') as f:
        f.write(current_story)
    thread = None


@socketio.on('connect')
def test_connect():
    if thread is None:
        emit('end', None)
    else:
        emit('start', None)
    emit('token_generated', current_story)


@socketio.on('request')
def start(prompt):
    print('start')
    global thread
    with thread_lock:
        if thread is None:
            thread = socketio.start_background_task(
                target=start_background_thread, req=prompt)


@app.route('/generated/<filename>')
def generated(filename):
    with open(f'generated/{filename}') as f:
        return """
<!doctype html>
<html lang="fr">
<body>
        """ + f.read().replace('\n', '<br/>') + """
</body>
</html>"""


@app.route('/')
def index():
    return """
<!doctype html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <title>My Website</title>
    <meta name="description" content="My Website">
    <meta name="author" content="SitePoint">
</head>
<body>
    <input type="text" id="prompt" placeholder="prompt">
    <button id="send" onclick="generate()">Send</button>
    <label>
        CFG:
        <input type="range" id="cfg" placeholder="cfg" value="1" min="0.0" max="10" step="0.1" oninput="this.nextElementSibling.value = this.value">
        <output>1</output>
    </label>
    <div id="output"></div>
    <ul>""" + '\n'.join(
        f'<li><a href="generated/{filename}">{filename}</a></li>'
        for filename in sorted(os.listdir('generated'),
                               reverse=True)) + """</ul>
</body>
<script src="https://cdnjs.cloudflare.com/ajax/libs/socket.io/4.0.1/socket.io.js"></script>
<script>
// Establish WebSocket connection
const socket = io();

// Handle WebSocket connection open event
socket.on('connect', () => {
    document.getElementById('output').innerHTML = '';
    console.log('WebSocket connection established.');
});

// Handle incoming WebSocket messages
socket.on('token_generated', (data) => {
    const token = data;
    // Handle the received token, update the web interface, etc.
    document.getElementById('output').innerHTML += token.replace(/\\n/g, "<br />");;
    console.log(token);
});

// Handle WebSocket connection close event
socket.on('disconnect', () => {
    console.log('WebSocket connection closed.');
});

socket.on('start', () => {
    document.getElementById('output').innerHTML = '';
    document.getElementById('send').disabled = true;
});

socket.on('end', () => {
    document.getElementById('send').disabled = false;
});

function generate() {
    const prompt = document.getElementById('prompt').value;
    socket.emit('request', {
        'prompt':prompt,
        'cfg':document.getElementById('cfg').value
    });
}

</script>
</html>
"""


if __name__ == '__main__':
    # Load the BPE
    checkpoint = torch.load(sys.argv[1], map_location='cpu')

    bpe = BPE()
    bpe.load_state_dict(checkpoint['bpe'])

    # Load the model
    model = transformer_from_checkpoint(checkpoint)
    model.eval()
    modelc = model  #torch.compile(model)
    print(modelc.load_state_dict(checkpoint['model']))
    del checkpoint

    # Sample from the model
    socketio.run(app, debug=True, host='0.0.0.0')
