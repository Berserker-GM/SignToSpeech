"""Run API + serve built frontend. Dev: run API here and `npm run dev` in frontend/."""
import socket
import sys


def _lan_ip() -> str:
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.connect(("8.8.8.8", 80))
        ip = sock.getsockname()[0]
        sock.close()
        return ip
    except OSError:
        return "YOUR_LAN_IP"


if __name__ == "__main__":
    sys.path.insert(0, ".")
    from api.server import run

    lan = _lan_ip()
    print("API: http://127.0.0.1:8000")
    print(f"Phone / mobile app: http://{lan}:8000  (same Wi‑Fi)")
    print("Dev UI: cd frontend && npm run dev  ->  http://127.0.0.1:5173")
    run()
