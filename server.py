from flask import Flask, render_template
from flask_socketio import SocketIO, send
from generate_one_ir import init_models, evaluate
import threading

app = Flask(__name__)
# CORS(app)
# app.config['SECRET_KEY'] = 'your_secret_key'
socketio = SocketIO(app, cors_allowed_origins='*')
port = 5000

mesh_embed, netG = init_models()
model_lock = threading.Lock()
# model_lock.acquire()

# Handle messages received via WebSockets
@socketio.on('message')
def handle_message(msg):
    print('Message received: ' + msg)
    send('Message received: ' + msg, broadcast=True)
    
@socketio.on('ask-for-IR')
def handle_ask_for_IR(recv_pos, src_pos, id):
    print(f'recv_pos: {recv_pos}')
    print(f'src_pos: {src_pos}')
    try:
        source = [src_pos['x'], src_pos['y'], src_pos['z']]
        receiver = [recv_pos['x'], recv_pos['y'], recv_pos['z']]
    except KeyError as e:
        print(f'KeyError: {e}')
        return
    if model_lock.acquire(False):
        ir_buffer = evaluate(mesh_embed, netG, receiver, source)
        # socketio.emit('send-IR', {'ir_array': ir_array.tobytes(), 'id': id})
        model_lock.release()
        return {'ir_buffer': ir_buffer.tobytes(), 'id': id}

if __name__ == '__main__':
    socketio.run(app, debug=True, port=port)
