# -*- coding: utf-8 -*-
"""
Spyderエディタ

これは一時的なスクリプトファイルです。
"""
from ultralytics import YOLO

def train():
    model = YOLO('yolov8n.pt')

    results = model.train(data='lvis.yaml', epochs=20, imgsz=640, device=0)

if __name__ == '__main__':
    train()


