SignToSpeech 🤟🔊

Real-time Sign Language to Speech using MediaPipe, Random Forest, LSTM, and a React frontend.

✨ Features:
    1. Real-time hand landmark detection using MediaPipe,
    2. Static sign recognition using Random Forest,
    3, Dynamic sign recognition using LSTM,
    4. AUTO / STATIC / DYNAMIC recognition modes,
    5. Sign-to-sentence conversion,
    6. Text-to-Speech,
    7. React + Vite frontend,
    8. FastAPI + WebSocket backend,
    9. Current Models.
    
Static Model — 13 Signs:
    DRINK · EAT · HELLO · HELP · MORE
    NO · PAIN · PLEASE · RESTROOM · SORRY
    STOP · THANK_YOU · YES
    
Dynamic LSTM — 7 Signs
    HELLO · HELP · NO · PLEASE
    SORRY · THANK_YOU · YES

Dynamic input:
  (T, 63)
  where 63 = 21 landmarks × (x, y, z).

🚀 Run Locally
  1. Clone
    git clone https://github.com/Berserker-GM/SignToSpeech.git
    cd SignToSpeech

  2. Install dependencies
    pip install -r requirements-web.txt
    cd frontend
    npm install
    cd ..

  3. Start backend
    python run_web.py

  4. Start frontend:
    In a new terminal:
        1. cd frontend
        2. npm run dev

Open the URL shown by Vite, usually:
    http://localhost:5173

-----------------preview----------------
1. LandMarks
   <img width="1919" height="1073" alt="Screenshot 2026-06-02 231019" src="https://github.com/user-attachments/assets/947b3e1d-40d2-4bc1-90de-6ee1867283d6" />

2. UI and detection in action
 <img width="1917" height="962" alt="Screenshot 2026-09-13 144740" src="https://github.com/user-attachments/assets/1d18f143-fc73-42b4-b8dd-38d1238f7387" />

3. History of the chat
<img width="1593" height="962" alt="Screenshot 2026-09-13 144758" src="https://github.com/user-attachments/assets/cc3b858e-5888-41ae-8994-c71bb069825c" />

4. Voices from ElevenLabs
<img width="1592" height="965" alt="Screenshot 2026-09-13 144805" src="https://github.com/user-attachments/assets/d4400e78-7ba2-4273-ad47-7928b2da49e0" />

5. Models
<img width="1917" height="986" alt="Screenshot 2026-09-13 144810" src="https://github.com/user-attachments/assets/a54336fc-915e-4712-a397-b5d01c11bc1d" />


⚠️ Note

  This is an actively developed prototype with a limited vocabulary. Recognition performance depends on lighting, camera quality, hand position, and training data.
  More signs and improved dynamic recognition are planned.

🤝 Contributing

  Contributions are welcome, especially for:  
    -Expanding the sign vocabulary,
    -Improving dynamic recognition,
    -Adding diverse training data,
    -Improving accessibility,
    -Mobile support,
    -Model evaluation and optimization.

