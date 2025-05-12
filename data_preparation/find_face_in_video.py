import os
import gc
import dlib
import cv2
import numpy as np
from PIL import Image
import scipy.ndimage

# ─── Config ────────────────────────────────────────────────────────────────
VIDEO_PATH     = "/content/drive/MyDrive/AA/The.Simpsons.Movie.2007.1080p.BluRay.x264-[YTS.AM].mp4"  # /content/drive/MyDrive/AA/videoplayback.mp4
OUTPUT_FOLDER  = "/content/drive/MyDrive/AA/output"
PREDICTOR_PATH = "/content/shape_predictor_68_face_landmarks.dat"
SAVENAME       = "Simpsons"
BLACK_WIDTH    = 0      # disable black‐bar cropping
MAX_WIDTH      = 1080   # downscale each frame to this width
FRAME_INTERVAL = 6      # process every 6th frame
# ────────────────────────────────────────────────────────────────────────────

def align_face(img: Image.Image, lm: np.ndarray) -> Image.Image:
    lm_chin          = lm[0:17]
    lm_eyebrow_left  = lm[17:22]
    lm_eyebrow_right = lm[22:27]
    lm_nose          = lm[27:31]
    lm_nostrils      = lm[31:36]
    lm_eye_left      = lm[36:42]
    lm_eye_right     = lm[42:48]
    lm_mouth_outer   = lm[48:60]
    lm_mouth_inner   = lm[60:68]

    eye_left   = np.mean(lm_eye_left, axis=0)
    eye_right  = np.mean(lm_eye_right, axis=0)
    eye_avg    = (eye_left + eye_right) * 0.5
    eye_to_eye = eye_right - eye_left
    mouth_left = lm_mouth_outer[0]
    mouth_right= lm_mouth_outer[6]
    mouth_avg  = (mouth_left + mouth_right) * 0.5
    eye_to_mouth = mouth_avg - eye_avg

    x = eye_to_eye - np.flipud(eye_to_mouth) * [-1,1]
    x /= np.hypot(*x)
    x *= max(np.hypot(*eye_to_eye) * 2.0, np.hypot(*eye_to_mouth) * 1.8)
    y = np.flipud(x) * [-1,1]
    c = eye_avg + eye_to_mouth * 0.1
    quad = np.stack([c - x - y, c - x + y, c + x + y, c + x - y])
    qsize = np.hypot(*x)

    output_size    = 512
    transform_size = 512
    enable_padding = True

    # shrink
    shrink = int(np.floor(qsize / output_size * 0.5))
    if shrink > 1:
        img    = img.resize((int(img.size[0]/shrink), int(img.size[1]/shrink)), Image.ANTIALIAS)
        quad  /= shrink
        qsize /= shrink

    # crop
    border = max(int(np.rint(qsize * 0.1)), 3)
    crop   = (
        max(int(np.floor(min(quad[:,0]))) - border, 0),
        max(int(np.floor(min(quad[:,1]))) - border, 0),
        min(int(np.ceil(max(quad[:,0]))) + border, img.size[0]),
        min(int(np.ceil(max(quad[:,1]))) + border, img.size[1])
    )
    if crop[2] - crop[0] < img.size[0] or crop[3] - crop[1] < img.size[1]:
        img  = img.crop(crop)
        quad -= crop[0:2]

    # pad
    pad = (
        max(-int(np.floor(min(quad[:,0]))) + border, 0),
        max(-int(np.floor(min(quad[:,1]))) + border, 0),
        max(int(np.ceil(max(quad[:,0]))) - img.size[0] + border, 0),
        max(int(np.ceil(max(quad[:,1]))) - img.size[1] + border, 0)
    )
    if enable_padding and max(pad) > border - 4:
        pad = np.maximum(pad, int(np.rint(qsize * 0.3)))
        arr = np.float32(img)
        img = np.pad(arr, ((pad[1], pad[3]), (pad[0], pad[2]), (0,0)), 'reflect')
        h, w, _ = img.shape
        yv, xv, _ = np.ogrid[:h, :w, :1]
        mask = np.maximum(
            1.0 - np.minimum(xv/pad[0], (w-1-xv)/pad[2]),
            1.0 - np.minimum(yv/pad[1], (h-1-yv)/pad[3])
        )
        blur = qsize * 0.02
        img = (img + (scipy.ndimage.gaussian_filter(img, [blur,blur,0]) - img)
               * np.clip(mask*3.0+1.0, 0.0, 1.0))
        img = (img + (np.median(img, (0,1)) - img) * np.clip(mask, 0.0, 1.0))
        img = Image.fromarray(np.uint8(np.clip(np.rint(img), 0, 255)), 'RGB')
        quad += pad[:2]

    # transform
    img = img.transform((transform_size, transform_size),
                        Image.QUAD, (quad+0.5).flatten(), Image.BILINEAR)
    if output_size < transform_size:
        img = img.resize((output_size, output_size), Image.ANTIALIAS)

    return img

def process_video(video_path, savename, detector, predictor):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video {video_path}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Opened video ({total} frames). Processing every {FRAME_INTERVAL}th frame…")

    idx = 0
    face_index = 0
    while True:
        idx += 1
        ret, frame = cap.read()
        if not ret:
            break

        if idx % FRAME_INTERVAL != 0:
            continue

        # downscale
        h0, w0 = frame.shape[:2]
        if w0 > MAX_WIDTH:
            scale = MAX_WIDTH / w0
            frame = cv2.resize(frame, (MAX_WIDTH, int(h0 * scale)), interpolation=cv2.INTER_AREA)

        # crop
        h, _ = frame.shape[:2]
        frame_crop = frame[BLACK_WIDTH : h - BLACK_WIDTH]

        # detect
        gray = cv2.cvtColor(frame_crop, cv2.COLOR_BGR2GRAY)
        dets = detector(gray, 1)
        if not dets:
            print(f"[✗] Frame {idx:05d}: no faces detected")
            continue

        d = dets[0]
        x1, y1, x2, y2 = d.left(), d.top(), d.right(), d.bottom()

        # landmarks
        shape = predictor(gray, d)
        lm    = np.array([[p.x, p.y] for p in shape.parts()])

        # align
        rgb     = cv2.cvtColor(frame_crop, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(rgb)
        aligned = align_face(img_pil, lm)

        out_name = f"{savename}_face{face_index:03d}_frame{idx:05d}.jpg"
        aligned.save(os.path.join(OUTPUT_FOLDER, out_name), quality=95)

        print(f"[{face_index}] Frame {idx:05d}, bbox {x1},{y1}→{x2},{y2}, saved → {out_name}")
        
        face_index += 1
        del gray, rgb, lm, img_pil, aligned
        gc.collect()

    cap.release()
    print(f"Done: {face_index} faces aligned & saved to {OUTPUT_FOLDER!r}")

if __name__ == "__main__":
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    detector  = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(PREDICTOR_PATH)

    process_video(VIDEO_PATH, SAVENAME, detector, predictor)
