"""Run API + serve built frontend. Dev: run API here and `npm run dev` in frontend/."""
import sys

if __name__ == "__main__":
    sys.path.insert(0, ".")
    from api.server import run

    print("API: http://127.0.0.1:8000")
    print("Dev UI: cd frontend && npm run dev  ->  http://127.0.0.1:5173")
    run()
