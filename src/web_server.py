from __future__ import annotations

import base64
import json
import os
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from io import BytesIO
from typing import Any, Dict

import numpy as np
from PIL import Image

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.inference import DigitRecognizer


recognizer = DigitRecognizer()


HTML_PAGE = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>CNN手写数字识别</title>
  <style>
    * { margin: 0; padding: 0; box-sizing: border-box; }
    body { 
      font-family: 'Segoe UI', 'Microsoft YaHei', sans-serif; 
      min-height: 100vh;
      background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
      color: #2d3748;
    }
    .container { 
      max-width: 1200px; 
      margin: 0 auto; 
      padding: 20px;
    }
    .header {
      text-align: center;
      padding: 30px 0;
      color: white;
    }
    .header h1 {
      font-size: 2.5rem;
      margin-bottom: 10px;
      text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
    }
    .header p {
      font-size: 1.1rem;
      opacity: 0.9;
    }
    .main-content {
      display: grid;
      grid-template-columns: 1.5fr 1fr;
      gap: 20px;
    }
    .card {
      background: white;
      border-radius: 20px;
      padding: 25px;
      box-shadow: 0 10px 40px rgba(0,0,0,0.15);
    }
    .tab-container {
      display: flex;
      gap: 10px;
      margin-bottom: 20px;
      border-bottom: 2px solid #e2e8f0;
      padding-bottom: 10px;
    }
    .tab {
      padding: 12px 24px;
      border: none;
      border-radius: 25px;
      background: #f7fafc;
      color: #4a5568;
      font-weight: 600;
      cursor: pointer;
      transition: all 0.3s ease;
    }
    .tab:hover {
      background: #edf2f7;
    }
    .tab.active {
      background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
      color: white;
    }
    .panel {
      display: none;
    }
    .panel.active {
      display: block;
    }
    #canvas {
      width: 100%;
      max-width: 300px;
      height: 300px;
      background: white;
      border-radius: 15px;
      border: 3px solid #e2e8f0;
      display: block;
      margin: 0 auto;
      touch-action: none;
    }
    #preview {
      display: none;
      width: 200px;
      height: 200px;
      object-fit: contain;
      margin: 15px auto;
      border-radius: 12px;
      border: 2px solid #e2e8f0;
    }
    .btn-group {
      display: flex;
      gap: 12px;
      justify-content: center;
      margin-top: 20px;
      flex-wrap: wrap;
    }
    .btn {
      padding: 12px 28px;
      border: none;
      border-radius: 12px;
      font-weight: 600;
      cursor: pointer;
      transition: all 0.3s ease;
    }
    .btn-primary {
      background: linear-gradient(135deg, #48bb78 0%, #38a169 100%);
      color: white;
    }
    .btn-primary:hover {
      transform: translateY(-2px);
      box-shadow: 0 5px 20px rgba(72, 187, 120, 0.4);
    }
    .btn-secondary {
      background: linear-gradient(135deg, #ed8936 0%, #dd6b20 100%);
      color: white;
    }
    .btn-secondary:hover {
      transform: translateY(-2px);
      box-shadow: 0 5px 20px rgba(237, 137, 54, 0.4);
    }
    .result-section {
      text-align: center;
    }
    .prediction {
      font-size: 4rem;
      font-weight: 800;
      color: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
      background: -webkit-linear-gradient(135deg, #667eea 0%, #764ba2 100%);
      -webkit-background-clip: text;
      -webkit-text-fill-color: transparent;
      margin: 15px 0;
    }
    .confidence {
      font-size: 1.1rem;
      color: #718096;
      margin-bottom: 25px;
    }
    .sub-title {
      font-size: 1.2rem;
      font-weight: 600;
      margin: 20px 0 15px 0;
      padding-bottom: 10px;
      border-bottom: 2px solid #e2e8f0;
    }
    .prob-bar-container {
      display: flex;
      align-items: center;
      margin: 10px 0;
    }
    .digit-label {
      width: 35px;
      font-weight: 600;
      color: #4a5568;
    }
    .prob-bar {
      flex: 1;
      height: 20px;
      background: #e2e8f0;
      border-radius: 10px;
      overflow: hidden;
      margin: 0 10px;
    }
    .prob-fill {
      height: 100%;
      background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
      border-radius: 10px;
      transition: width 0.3s ease;
    }
    .prob-value {
      width: 60px;
      text-align: right;
      font-weight: 600;
      color: #2d3748;
    }
    table {
      width: 100%;
      border-collapse: collapse;
      margin-top: 10px;
    }
    th, td {
      padding: 12px;
      text-align: left;
      border-bottom: 1px solid #e2e8f0;
    }
    th {
      background: #f7fafc;
      font-weight: 600;
    }
    .history-list {
      max-height: 250px;
      overflow-y: auto;
    }
    .file-input {
      display: none;
    }
    .file-label {
      display: inline-block;
      padding: 12px 24px;
      background: #f7fafc;
      border: 2px dashed #cbd5e0;
      border-radius: 12px;
      cursor: pointer;
      transition: all 0.3s ease;
      margin-bottom: 15px;
    }
    .file-label:hover {
      border-color: #667eea;
      background: #f0f4ff;
    }
    @media (max-width: 850px) {
      .main-content {
        grid-template-columns: 1fr;
      }
      .header h1 {
        font-size: 1.8rem;
      }
    }
  </style>
</head>
<body>
  <div class="container">
    <div class="header">
      <h1>🔢 CNN手写数字识别系统</h1>
      <p>支持上传图片或在线手写，实时识别手写数字（0-9）</p>
    </div>
    
    <div class="main-content">
      <div class="card">
        <div class="tab-container">
          <button class="tab active" data-target="upload-panel">📷 上传图片</button>
          <button class="tab" data-target="draw-panel">✏️ 在线手写</button>
        </div>
        
        <div id="upload-panel" class="panel active">
          <label for="fileInput" class="file-label">点击选择图片文件</label>
          <input type="file" id="fileInput" accept="image/*" class="file-input" />
          <img id="preview" alt="预览图片" />
          <div class="btn-group">
            <button class="btn btn-primary" id="predictUploadBtn">开始识别</button>
          </div>
        </div>
        
        <div id="draw-panel" class="panel">
          <p style="text-align:center; color:#718096; margin-bottom:15px;">在画布上用黑色笔迹手写数字（白底黑字）</p>
          <canvas id="canvas" width="300" height="300"></canvas>
          <div class="btn-group">
            <button class="btn btn-primary" id="predictCanvasBtn">识别手写</button>
            <button class="btn btn-secondary" id="clearCanvasBtn">清空画布</button>
          </div>
        </div>
      </div>
      
      <div class="card">
        <div class="result-section">
          <div class="prediction" id="predictionValue">-</div>
          <div class="confidence" id="confidenceValue">置信度: -</div>
        </div>
        
        <div class="sub-title">🏆 Top 3 预测</div>
        <table id="top3Table">
          <thead>
            <tr><th>排名</th><th>数字</th><th>概率</th></tr>
          </thead>
          <tbody></tbody>
        </table>
        
        <div class="sub-title">📊 概率分布</div>
        <div id="probabilityBars"></div>
        
        <div class="sub-title">📝 识别历史</div>
        <div class="history-list">
          <table id="historyTable">
            <thead>
              <tr><th>模式</th><th>预测</th><th>置信度</th></tr>
            </thead>
            <tbody></tbody>
          </table>
        </div>
      </div>
    </div>
  </div>

  <script>
    const tabs = document.querySelectorAll('.tab');
    const panels = document.querySelectorAll('.panel');
    const history = [];
    
    const canvas = document.getElementById('canvas');
    const ctx = canvas.getContext('2d');
    ctx.fillStyle = '#ffffff';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    ctx.lineWidth = 18;
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';
    ctx.strokeStyle = '#000000';
    
    let isDrawing = false;
    
    function getPoint(event) {
      const rect = canvas.getBoundingClientRect();
      const touch = event.touches ? event.touches[0] : event;
      return {
        x: touch.clientX - rect.left,
        y: touch.clientY - rect.top
      };
    }
    
    function startDrawing(event) {
      isDrawing = true;
      const point = getPoint(event);
      ctx.beginPath();
      ctx.moveTo(point.x, point.y);
      event.preventDefault();
    }
    
    function draw(event) {
      if (!isDrawing) return;
      const point = getPoint(event);
      ctx.lineTo(point.x, point.y);
      ctx.stroke();
      event.preventDefault();
    }
    
    function stopDrawing() {
      isDrawing = false;
    }
    
    canvas.addEventListener('mousedown', startDrawing);
    canvas.addEventListener('mousemove', draw);
    window.addEventListener('mouseup', stopDrawing);
    canvas.addEventListener('touchstart', startDrawing, {passive: false});
    canvas.addEventListener('touchmove', draw, {passive: false});
    window.addEventListener('touchend', stopDrawing);
    
    tabs.forEach(tab => {
      tab.addEventListener('click', () => {
        tabs.forEach(t => t.classList.remove('active'));
        panels.forEach(p => p.classList.remove('active'));
        tab.classList.add('active');
        document.getElementById(tab.dataset.target).classList.add('active');
      });
    });
    
    const preview = document.getElementById('preview');
    const fileInput = document.getElementById('fileInput');
    
    fileInput.addEventListener('change', () => {
      const file = fileInput.files[0];
      if (!file) {
        preview.style.display = 'none';
        return;
      }
      preview.src = URL.createObjectURL(file);
      preview.style.display = 'block';
    });
    
    function updateHistory() {
      const tbody = document.querySelector('#historyTable tbody');
      tbody.innerHTML = '';
      history.forEach(item => {
        const row = document.createElement('tr');
        row.innerHTML = `
          <td>${item.mode === 'Upload' ? '📷' : '✏️'}</td>
          <td style="font-weight:600;">${item.prediction}</td>
          <td>${item.confidence}</td>
        `;
        tbody.appendChild(row);
      });
    }
    
    function showResult(data, mode) {
      document.getElementById('predictionValue').textContent = data.prediction;
      document.getElementById('confidenceValue').textContent = `置信度: ${(data.confidence * 100).toFixed(2)}%`;
      
      const top3Body = document.querySelector('#top3Table tbody');
      top3Body.innerHTML = '';
      data.top3.forEach((item, index) => {
        const row = document.createElement('tr');
        row.innerHTML = `
          <td>Top ${index + 1}</td>
          <td style="font-weight:600;">${item.digit}</td>
          <td>${(item.probability * 100).toFixed(2)}%</td>
        `;
        top3Body.appendChild(row);
      });
      
      const barsContainer = document.getElementById('probabilityBars');
      barsContainer.innerHTML = '';
      data.probabilities.forEach(item => {
        const barRow = document.createElement('div');
        barRow.className = 'prob-bar-container';
        barRow.innerHTML = `
          <div class="digit-label">${item.digit}</div>
          <div class="prob-bar"><div class="prob-fill" style="width:${(item.probability * 100).toFixed(2)}%"></div></div>
          <div class="prob-value">${(item.probability * 100).toFixed(1)}%</div>
        `;
        barsContainer.appendChild(barRow);
      });
      
      history.unshift({
        mode: mode,
        prediction: data.prediction,
        confidence: `${(data.confidence * 100).toFixed(2)}%`
      });
      
      if (history.length > 8) {
        history.pop();
      }
      
      updateHistory();
    }
    
    async function predictUpload() {
      const file = fileInput.files[0];
      if (!file) {
        alert('请先选择图片文件！');
        return;
      }
      
      const formData = new FormData();
      formData.append('file', file);
      
      const response = await fetch('/api/predict-upload', {
        method: 'POST',
        body: formData
      });
      
      const data = await response.json();
      showResult(data, 'Upload');
    }
    
    async function predictCanvas() {
      const dataUrl = canvas.toDataURL('image/png');
      const response = await fetch('/api/predict-canvas', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ data_url: dataUrl })
      });
      
      const data = await response.json();
      showResult(data, 'Canvas');
    }
    
    document.getElementById('predictUploadBtn').addEventListener('click', predictUpload);
    document.getElementById('predictCanvasBtn').addEventListener('click', predictCanvas);
    document.getElementById('clearCanvasBtn').addEventListener('click', () => {
      ctx.fillStyle = '#ffffff';
      ctx.fillRect(0, 0, canvas.width, canvas.height);
    });
  </script>
</body>
</html>"""


class DigitHandler(BaseHTTPRequestHandler):
    def _send_json(self, payload: Dict[str, Any], status: int = 200) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_html(self, html: str) -> None:
        body = html.encode("utf-8")
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        if self.path == "/":
            self._send_html(HTML_PAGE)
            return
        self.send_error(HTTPStatus.NOT_FOUND, "Not Found")

    def do_POST(self) -> None:
        if self.path == "/api/predict-upload":
            self._handle_upload()
            return
        if self.path == "/api/predict-canvas":
            self._handle_canvas()
            return
        self.send_error(HTTPStatus.NOT_FOUND, "Not Found")

    def _handle_upload(self) -> None:
        content_type = self.headers.get("Content-Type", "")
        content_length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(content_length)

        boundary = None
        for part in content_type.split(";"):
            part = part.strip()
            if part.startswith("boundary="):
                boundary = part[9:].strip().encode()
                break

        if boundary is None:
            self._send_json({"error": "缺少 boundary 参数"}, status=400)
            return

        file_data = None
        for chunk in body.split(b"--" + boundary):
            if b'name="file"' in chunk:
                sep = chunk.find(b"\r\n\r\n")
                if sep != -1:
                    file_data = chunk[sep + 4:]
                    if file_data.endswith(b"\r\n"):
                        file_data = file_data[:-2]
                    break

        if not file_data:
            self._send_json({"error": "未找到文件数据"}, status=400)
            return

        image = Image.open(BytesIO(file_data)).convert("L")
        result = recognizer.predict_uploaded_image(image)
        self._send_json(result)

    def _handle_canvas(self) -> None:
        content_length = int(self.headers.get("Content-Length", "0"))
        raw_body = self.rfile.read(content_length)
        payload = json.loads(raw_body.decode("utf-8"))
        _, encoded = payload["data_url"].split(",", 1)
        image = Image.open(BytesIO(base64.b64decode(encoded))).convert("RGBA")
        result = recognizer.predict_canvas({"composite": np.array(image)})
        self._send_json(result)

    def log_message(self, format: str, *args) -> None:
        return


def main() -> None:
    port = int(os.getenv("PORT", "8080"))
    server = ThreadingHTTPServer(("0.0.0.0", port), DigitHandler)
    print(f"服务已启动: http://localhost:{port}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
