# 🤖 NCT Chatbot

สวัสดีงับ! นี่คือแชทบอทที่จะช่วยตอบคำถามเกี่ยวกับ NCT โดยใช้ AI ในการค้นหาและสร้างคำตอบที่เหมาะสม แต่ตอบได้เพียงข้อมูลที่มีอยู่นะ

## ✨ ทำอะไรได้บ้าง?

- ค้นหาข้อมูลด้วย FAISS
- สร้างคำตอบด้วย Gemini จาก Google
- มีหน้าเว็บให้แชทได้เลย

## 📦 ต้องติดตั้งอะไรบ้าง?

ต้องมี Python 3.8 ขึ้นไปนะ แล้วก็พวกนี้:
- Flask
- sentence-transformers
- faiss-cpu
- google-generativeai
- NumPy

## 🚀 วิธีติดตั้ง

1. ก็อปโค้ดมาก่อน:
```bash
git clone https://github.com/thanatchaphon-23/zen-bot.git
cd zen-bot
```

2. ตั้งค่า API:
- ไปเอา API key จาก Google AI Studio มา
- เอาไปใส่ใน `nctbot.py`

## 📁 มีไฟล์อะไรบ้าง

```
ZEN-BOT/
├── templates/  <- หน้าเว็บอยู่นี่
│   ├── index.html
│   └── style.css
├── main.py     <- รันตัวนี้นะจ้ะ 🙆‍♀️
├── nctbot.py   <- โค้ดหลัก
└── NEO.json    <- ข้อมูล
```

## 🙎‍♀️ จะใช้ยังไง?

1. รันเลย:
```bash
python main.py
ถ้าเป็นระบบ macOS ใช้ python3 น้าา
```

2. เข้าเว็บที่ `http://localhost:8080`

3. หรือจะเขียนโค้ดเรียกใช้เองก็ได้:
```python
from nctbot import NCTChatbot

chatbot = NCTChatbot("NEO.json")
response = chatbot.chat("ถามอะไรก็ได้")
print(response)
```

## 🙏 😊

ไลบารี่ที่ใช้:
- Sentence Transformers
- FAISS
- Google Gemini
- Flask
