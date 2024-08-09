from dotenv import load_dotenv
from src.websocket_server import WebSocketServer

def main():
    try:
        load_dotenv()
        # host = "0.0.0.0"
        host = "localhost"
        server = WebSocketServer(host, 8765)
        server.run()
    except Exception as e:
        print(f"\n\nError Found : {e}\n\n")

if __name__ == "__main__":
    main()
