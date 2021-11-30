
import websockets
import asyncio

from flask import Flask



app = Flask(__name__)


@app.route('/')
def hello():
    return 'Hello, World!'

if __name__ == '__main__':
   app.run(host="localhost", port=3002, debug=True)


   PORT = 3000

async def echo(websocket, path):
    print("A client just connected")
    try:
        async for message in websocket:
            print("Received message from client: " + message)
    except websockets.exceptions.ConnectionClosed as e:
        print("A client just disconnected")

start_server = websockets.serve(echo, "localhost", PORT)

asyncio.get_event_loop().run_until_complete(start_server)
asyncio.get_event_loop().run_forever()