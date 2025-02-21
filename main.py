from flask import Flask, request, jsonify, send_from_directory
from nctbot import NCTChatbot

# สร้าง chatbot
chatbot = NCTChatbot("NEO.json")

# สร้างแอปพลิเคชัน Flask
app = Flask(__name__, static_folder='templates')

@app.route('/')
def index():
    return send_from_directory('templates', 'index.html')

@app.route('/api/ask', methods=['POST'])
def ask():
    data = request.json
    query = data.get('query', '')
    if not query:
        return jsonify({'error': 'กรุณาระบุคำถาม'}), 400
    # ใช้ chatbot ในการประมวลผลคำถาม
    answer = chatbot.generate_response(query)
    return jsonify({'answer': answer})

if __name__ == "__main__":
    print("🚀 เริ่มต้นระบบ...")
    app.run(debug=True, host='0.0.0.0', port=8080)
