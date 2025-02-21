from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
import json
from typing import List, Dict, Tuple
from google import genai

# กำหนด API client สำหรับ Gemini
client = genai.Client(api_key="AIzaSyAnMoWyTZYVT0nx7n9-zPAVFsCenlHmRhc")

class NCTChatbot:
    def __init__(self, vector_path: str):
        # โหลดโมเดลสำหรับ embedding
        self.encoder = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')
        
        # โหลดข้อมูลจากไฟล์ JSON
        self.data = self.load_data(vector_path)
        self.index, self.passages = self.build_index()

    def load_data(self, vector_path: str) -> List[Dict]:
        """โหลดข้อมูลจากไฟล์ JSON"""
        with open(vector_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data

    def build_index(self) -> Tuple[faiss.IndexFlatL2, List[str]]:
        """สร้าง FAISS index สำหรับการค้นหาแบบ semantic"""
        passages = [entry['content'] for entry in self.data]
        embeddings = np.array([self.encoder.encode([entry['content']])[0] for entry in self.data], dtype=np.float32)
        
        # สร้าง FAISS index
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(embeddings)
        
        return index, passages

    def retrieve_relevant_passages(self, query: str, k: int = 3) -> List[Dict]:
        """ค้นหาข้อความที่เกี่ยวข้องที่สุด"""
        # สร้าง embedding สำหรับ query
        query_embedding = self.encoder.encode([query])
        
        # ค้นหาข้อความที่ใกล้เคียงที่สุด
        distances, indices = self.index.search(
            np.array(query_embedding).astype('float32'), k
        )
        
        # รวบรวมผลลัพธ์
        results = []
        for idx in indices[0]:
            if idx >= 0 and idx < len(self.data):  # ตรวจสอบความถูกต้องของ index
                results.append({
                    'title': self.data[idx]['title'],
                    'content': self.data[idx]['content']
                })
        
        return results

    def generate_response(self, query: str) -> str:
        """สร้างคำตอบโดยใช้ Gemini LLM"""
        try:
            # ค้นหาข้อความที่เกี่ยวข้อง
            relevant_passages = self.retrieve_relevant_passages(query)
            
            # สร้าง prompt
            context = " ".join([f"หัวข้อ: {p['title']}\nข้อมูล: {p['content']}" for p in relevant_passages])
            prompt = f"""คำถาม: {query}

ข้อมูลที่เกี่ยวข้อง:
{context}

โปรดช่วยให้คำตอบเป็นธรรมชาติและเหมาะสมกับสถานการณ์นี้:
"""

            # ใช้ Gemini เพื่อสร้างคำตอบ
            response = client.models.generate_content(
                model="gemini-2.0-flash", contents=prompt
            )
            return response.text
        
        except Exception as e:
            return f"เกิดข้อผิดพลาดในการสร้างคำตอบ: {str(e)}"

    def chat(self, query: str) -> Dict:
        """ฟังก์ชันหลักสำหรับการแชท"""
        try:
            # ค้นหาข้อความที่เกี่ยวข้อง
            relevant_info = self.retrieve_relevant_passages(query)
            
            # สร้างคำตอบ
            response = self.generate_response(query)
            
            # ระบุว่าข้อมูลมาจากสมาชิกคนไหนบ้าง
            return {
                'status': 'success',
                'response': response,
                'confidence': 'high' if len(relevant_info) >= 2 else 'medium'
            }
            
        except Exception as e:
            return {
                'status': 'error',
                'response': 'ขออภัย ฉันไม่สามารถประมวลผลคำถามนี้ได้',
                'error': str(e)
            }
