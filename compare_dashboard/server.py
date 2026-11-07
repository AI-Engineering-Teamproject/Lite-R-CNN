import base64
import io
import json
import os
import sys
import threading
import time
import secrets
import socket
import re
import zipfile
from urllib.parse import parse_qs, urlsplit, urlencode
from urllib.request import Request, build_opener, HTTPRedirectHandler
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parent
BASE = ROOT.parent
sys.path.insert(0, str(BASE / 'src'))
os.environ.setdefault('OMP_NUM_THREADS', '4')

import torch
import torchvision
from PIL import Image, ImageOps
from torchvision.transforms.functional import pil_to_tensor
from light_rcnn.model.detector import light_rcnn_from_config

torch.set_num_threads(min(4, os.cpu_count() or 1))
V3 = BASE / 'outputs' / 'coco_4090_v3' / 'best.pth'
models = {}
lock = threading.Lock()
ACCESS_KEY = None
COCO_ZIP = 'https://huggingface.co/datasets/Hyeon1222/COCO/resolve/main/val2017.zip'
COCO_ANNOTATIONS_ZIP = 'https://huggingface.co/datasets/Hyeon1222/COCO/resolve/main/annotations.zip'
annotations_cache = None
annotations_lock = threading.Lock()

class SafeRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        target = urlsplit(newurl)
        if target.scheme != 'https' or not (target.hostname == 'huggingface.co' or target.hostname.endswith('.hf.co') or target.hostname == 'datasets-server.huggingface.co'):
            raise ValueError('지원하지 않는 이미지 주소입니다.')
        return super().redirect_request(req, fp, code, msg, headers, newurl)

HF = build_opener(SafeRedirect())
def hf_read(url, limit):
    parsed = urlsplit(url)
    if parsed.scheme != 'https' or parsed.hostname != 'datasets-server.huggingface.co': raise ValueError('지원하지 않는 이미지 주소입니다.')
    with HF.open(Request(url, headers={'User-Agent':'LiteRCNN-compare/1.0'}), timeout=30) as response:
        data = response.read(limit + 1)
    if len(data) > limit: raise ValueError('불러온 이미지가 너무 큽니다.')
    return data

class RemoteZip(io.RawIOBase):
    def __init__(self, url=COCO_ZIP):
        self.pos = 0
        self.url = url
        with HF.open(Request(self.url, headers={'Range':'bytes=0-0','User-Agent':'LiteRCNN-compare/1.0'}), timeout=45) as r:
            match = re.fullmatch(r'bytes 0-0/(\d+)', r.headers.get('Content-Range',''))
            if r.status != 206 or not match: raise ValueError('원격 데이터셋이 부분 읽기를 지원하지 않습니다.')
            self.size = int(match.group(1))
    def seekable(self): return True
    def readable(self): return True
    def tell(self): return self.pos
    def seek(self, offset, whence=0):
        self.pos = (0 if whence == 0 else self.pos if whence == 1 else self.size) + offset
        if self.pos < 0: raise ValueError('잘못된 파일 위치입니다.')
        return self.pos
    def read(self, size=-1):
        if self.pos >= self.size: return b''
        n = min(self.size - self.pos, self.size if size < 0 else size)
        if n > 16 * 1024 * 1024: raise ValueError('요청한 이미지가 너무 큽니다.')
        start, end = self.pos, self.pos+n-1
        if not n: return b''
        with HF.open(Request(self.url,headers={'Range':f'bytes={start}-{end}','User-Agent':'LiteRCNN-compare/1.0'}),timeout=45) as r:
            expected=f'bytes {start}-{end}/{self.size}'
            if r.status != 206 or r.headers.get('Content-Range') != expected: raise ValueError('원격 데이터셋의 부분 읽기에 실패했습니다.')
            data=r.read(n+1)
        if len(data) != n: raise ValueError('이미지를 완전히 불러오지 못했습니다.')
        self.pos += n
        return data

def load_annotations():
    global annotations_cache
    with annotations_lock:
        if annotations_cache is not None: return annotations_cache
        with zipfile.ZipFile(RemoteZip(COCO_ANNOTATIONS_ZIP)) as archive:
            entry=archive.getinfo('annotations/instances_val2017.json')
            if entry.file_size > 30 * 1024 * 1024: raise ValueError('주석 파일이 너무 큽니다.')
            data=json.loads(archive.read(entry))
        categories={x['id']:x['name'] for x in data['categories']}
        files={x['file_name']:x['id'] for x in data['images']}
        by_image={}
        for ann in data['annotations']:
            x,y,w,h=ann['bbox']
            by_image.setdefault(ann['image_id'], []).append({'box':[x,y,x+w,y+h], 'label':categories.get(ann['category_id'],str(ann['category_id'])), 'iscrowd':bool(ann.get('iscrowd',0))})
        annotations_cache=(files,by_image)
        return annotations_cache

def random_image(dataset_key):
    if dataset_key != 'coco-val2017': raise ValueError('COCO 2017 검증 데이터만 사용할 수 있습니다.')
    with zipfile.ZipFile(RemoteZip()) as archive:
        images=[item for item in archive.infolist() if item.filename.lower().endswith(('.jpg','.jpeg','.png')) and not item.is_dir()]
        if not images: raise ValueError('검증 이미지를 찾지 못했습니다.')
        item=secrets.choice(images)
        if item.file_size > 12 * 1024 * 1024: raise ValueError('불러온 이미지가 너무 큽니다.')
        raw=archive.read(item)
    with Image.open(io.BytesIO(raw)) as image:
        if image.width * image.height > 40000000: raise ValueError('불러온 이미지의 해상도가 너무 큽니다.')
        image.verify()
    filename=Path(item.filename).name
    files,by_image=load_annotations()
    if filename not in files: raise ValueError('이 이미지에 해당하는 검증 주석을 찾지 못했습니다.')
    image_id=files[filename]
    return {'image':base64.b64encode(raw).decode('ascii'), 'dataset':'Hyeon1222/COCO', 'split':'val2017', 'filename':filename, 'total':len(images), 'image_id':image_id, 'ground_truth':by_image.get(image_id,[])}

def load_models(device):
    global models
    if device in models: return models[device]
    if not V3.is_file(): raise RuntimeError('v3 가중치를 찾을 수 없습니다: outputs/coco_4090_v3/best.pth')
    ckpt = torch.load(V3, map_location='cpu', weights_only=True)
    lite = light_rcnn_from_config(ckpt['config'], pretrained=False)
    lite.load_state_dict(ckpt['model']); lite.eval()
    weights = torchvision.models.detection.FasterRCNN_ResNet50_FPN_Weights.COCO_V1
    # Torchvision retrieves the official COCO_V1 weights into its cache on first use.
    faster = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=weights, min_size=640, max_size=640).eval()
    faster.roi_heads.score_thresh = 0.05
    lite.roi_heads.score_thresh = 0.05
    models[device] = (faster.to(device), lite.to(device), weights.meta['categories'])
    return models[device]

def infer(data, device, threshold=0.50):
    if device not in ('cpu', 'cuda'): raise ValueError('지원하지 않는 실행 장치입니다.')
    if device == 'cuda' and not torch.cuda.is_available(): raise ValueError('사용 가능한 CUDA GPU가 없습니다.')
    if not 0.05 <= threshold <= 0.95: raise ValueError('신뢰도는 5%에서 95% 사이여야 합니다.')
    raw = base64.b64decode(data, validate=True)
    if len(raw) > 12 * 1024 * 1024: raise ValueError('사진 크기는 12MB 이하여야 합니다.')
    with Image.open(io.BytesIO(raw)) as img:
        img = ImageOps.exif_transpose(img).convert('RGB')
        if img.width * img.height > 40000000: raise ValueError('사진 해상도가 너무 큽니다.')
        tensor = pil_to_tensor(img).float().div(255)
    with lock, torch.inference_mode():
        faster, lite, names = load_models(device)
        tensor = tensor.to(device)
        results = []
        for model in (faster, lite):
            # Warm both models on this input before measuring; loading excluded.
            model([tensor])
            if device == 'cuda': torch.cuda.synchronize()
            start = time.perf_counter(); pred = model([tensor])[0]
            if device == 'cuda': torch.cuda.synchronize()
            ms = (time.perf_counter() - start)*1000
            boxes = []
            for box, score, label in zip(pred['boxes'].tolist(), pred['scores'].tolist(), pred['labels'].tolist()):
                if score < threshold: continue
                idx = int(label)
                boxes.append({'box': box, 'score': score, 'label': names[idx] if idx < len(names) else str(idx)})
            results.append({'ms': round(ms, 2), 'boxes': boxes})
    return {'width': img.width, 'height': img.height, 'results': results, 'device': device.upper(), 'score_threshold': threshold, 'model_score_threshold': 0.05}

class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        parsed = urlsplit(self.path)
        if ACCESS_KEY and not secrets.compare_digest(parse_qs(parsed.query).get('key', [''])[0], ACCESS_KEY): return self.send_error(403)
        if parsed.path == '/':
            body=(ROOT / 'index.html').read_text(encoding='utf-8').replace('__ACCESS_KEY__', ACCESS_KEY or '').encode('utf-8'); self.send_response(200); self.send_header('Content-Type','text/html; charset=utf-8'); self.send_header('Referrer-Policy','no-referrer'); self.send_header('Content-Length',str(len(body))); self.end_headers(); self.wfile.write(body)
        elif parsed.path == '/status':
            body=json.dumps({'cuda':torch.cuda.is_available(), 'device_name':torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}).encode()
            self.send_response(200); self.send_header('Content-Type','application/json'); self.send_header('Content-Length',str(len(body))); self.end_headers(); self.wfile.write(body)
        else: self.send_error(404)
    def do_POST(self):
        if self.path not in ('/analyze','/random-image'): return self.send_error(404)
        if ACCESS_KEY and not secrets.compare_digest(self.headers.get('X-Access-Key',''), ACCESS_KEY): return self.send_error(403)
        try:
            n=int(self.headers.get('Content-Length','0'))
            if n > 18000000: raise ValueError('사진 크기는 12MB 이하여야 합니다.')
            request=json.loads(self.rfile.read(n))
            result=random_image(request.get('dataset')) if self.path == '/random-image' else infer(request['image'],request.get('device','cpu'),float(request.get('threshold',0.50)))
            status=200
        except Exception as e: result={'error':str(e)}; status=400
        data=json.dumps(result,ensure_ascii=False).encode('utf-8'); self.send_response(status); self.send_header('Content-Type','application/json; charset=utf-8'); self.send_header('Content-Length',str(len(data))); self.end_headers(); self.wfile.write(data)

if __name__ == '__main__':
    phone = '--phone' in sys.argv
    if phone:
        ACCESS_KEY = secrets.token_urlsafe(24)
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
                s.connect(('1.1.1.1', 80)); ip = s.getsockname()[0]
        except OSError: ip = '<this-computer-ip>'
        print(f'Phone: http://{ip}:8766/?key={ACCESS_KEY}', flush=True)
        print('Open this address on a phone on the same Wi-Fi. Keep this window open.', flush=True)
        ThreadingHTTPServer(('0.0.0.0',8766),Handler).serve_forever()
    else:
        if '--open' in sys.argv:
            import webbrowser
            threading.Timer(1.0, lambda: webbrowser.open('http://127.0.0.1:8765/')).start()
        print('http://127.0.0.1:8765',flush=True)
        ThreadingHTTPServer(('127.0.0.1',8765),Handler).serve_forever()
