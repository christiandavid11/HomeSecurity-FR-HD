import os
import urllib.request as rq
from typing import Optional

import cv2
from pymongo import MongoClient

from config import settings

client = MongoClient(settings.mongodb)

print("[INFO] Connecting database...")

db = client.get_database(settings.db)

def get_classes():
    return [class_name.strip() for class_name in open(settings.classes).readlines()]


async def is_ready(_type: str, ready: bool) -> None:
    collection = db.detect

    ready = "online" if ready else "offline" # type: ignore

    print(f"[INFO] {_type} is {ready}...")

    if _type == "face-recognized":
        collection.update_one(
            { 'tech': 'face_recognition' },
            { '$set': {'is_ready': ready},
            '$currentDate': { 'last_modified': True}}
        )

    elif _type == "human-detected":
        collection.update_one(
            { 'tech': 'human_detection' },
            { '$set': {'is_ready': ready},
            '$currentDate': { 'last_modified': True}}
        )


async def detected(_type: str, is_detected: bool, name: Optional[str]=None) -> None:
    collection = db.detect

    is_detected = "detected" if is_detected else "not detected" # type: ignore

    print(f"[INFO] {_type} is {is_detected}...")

    if _type == "face-recognized":
        collection.update_one(
            { 'tech': 'face_recognition' },
            { '$set': {
                'is_detected': is_detected,
                'last_recognized': name
            },
            '$currentDate': { 'last_modified': True}}
        )

    elif _type == "human-detected":
        collection.update_one(
            { 'tech': 'human_detection' },
            { '$set': {'is_detected': is_detected},
            '$currentDate': { 'last_modified': True}}
        )


async def draw_border(img, pt1, pt2, color, thickness, r, d):
    x1, y1 = pt1
    x2, y2 = pt2

    cv2.line(img, (x1 + r, y1), (x1 + r + d, y1), color, thickness)
    cv2.line(img, (x1, y1 + r), (x1, y1 + r + d), color, thickness)
    cv2.ellipse(img, (x1 + r, y1 + r), (r, r), 180, 0, 90, color, thickness)

    cv2.line(img, (x2 - r, y1), (x2 - r - d, y1), color, thickness)
    cv2.line(img, (x2, y1 + r), (x2, y1 + r + d), color, thickness)
    cv2.ellipse(img, (x2 - r, y1 + r), (r, r), 270, 0, 90, color, thickness)

    cv2.line(img, (x1 + r, y2), (x1 + r + d, y2), color, thickness)
    cv2.line(img, (x1, y2 - r), (x1, y2 - r - d), color, thickness)
    cv2.ellipse(img, (x1 + r, y2 - r), (r, r), 90, 0, 90, color, thickness)

    cv2.line(img, (x2 - r, y2), (x2 - r - d, y2), color, thickness)
    cv2.line(img, (x2, y2 - r), (x2, y2 - r - d), color, thickness)
    cv2.ellipse(img, (x2 - r, y2 - r), (r, r), 0, 0, 90, color, thickness)


async def get_images():
    try:
        images = db.tests
        images.count_documents({})
        images = list(images.find())

        if os.path.isdir('dataset'):
            for _, v in enumerate(images):
                for name, image in v['Photos'].items():
                    if not os.path.exists(f'dataset/{name}'):
                        os.mkdir(f'dataset/{name}')

                    for i, iu in enumerate(image):
                        image_url = iu['image_url'].split('images/')[1]
                        file = os.path.join(os.getcwd(), f'dataset/{name}', image_url)
                        rq.urlretrieve(iu['image_url'], file)
                        print(f"[INFO] Retrieving {name} images {i+1}/{len(image)}...")

    except Exception as exc:
        print(f"[ERROR] Something happened on getting image...\n{exc}")

    else:
        print(f"[INFO] Dataset complete...")

async def create_env():
    if not os.path.exists('.env'):
        print('[INFO] Creating .env file...')
        f = open('.env','a')
        f.write('mongodb=\n')
        f.write('db=ReactNativeApp')
        f.close()
        print('[INFO] .env file is created...')