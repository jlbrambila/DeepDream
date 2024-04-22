import cv2
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.inception_v3 import preprocess_input
from moviepy.editor import ImageSequenceClip

# Define paths
deep_dream_folder = os.path.join(os.path.expanduser("~"), "Desktop", "deepdreamS")
input_video_path = os.path.join(deep_dream_folder, 'deep_dream_vid.mp4')
output_frame_path = os.path.join(deep_dream_folder, 'frames/')
output_dream_frame_path = os.path.join(deep_dream_folder, 'dream_frames/')

# Create directories for frames and dream frames
os.makedirs(output_frame_path, exist_ok=True)
os.makedirs(output_dream_frame_path, exist_ok=True)

# Function to extract frames from video
def extract_frames(video_path, output_path):
    cap = cv2.VideoCapture(video_path)
    count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imwrite(f'{output_path}frame_{count:04d}.png', frame)
        count += 1
    cap.release()
    return count

# Load the InceptionV3 model
def load_model():
    model = InceptionV3(include_top=False, weights='imagenet')
    dream_layers = ['mixed6', 'mixed7', 'mixed8', 'mixed9', 'mixed10'] # You can adjust which layers are used
    outputs = [model.get_layer(name).output for name in dream_layers]
    deep_dream_model = tf.keras.Model(inputs=model.input, outputs=outputs)
    return deep_dream_model

# Deep Dream function
def deep_dream_effect(img, model, steps=50, step_size=0.05):
    img = preprocess_input(img)
    img = tf.convert_to_tensor(img)
    for _ in range(steps):
        with tf.GradientTape() as tape:
            tape.watch(img)
            loss = tf.reduce_sum([tf.reduce_mean(layer) for layer in model(img)])
        gradients = tape.gradient(loss, img)
        gradients /= tf.math.reduce_std(gradients) + 1e-8
        img = img + gradients * step_size
        img = tf.clip_by_value(img, -1, 1)
    return img.numpy()

# Apply Deep Dream to each frame
def process_frames(frame_folder, dream_folder, model, frame_count):
    for i in range(frame_count):
        frame_path = os.path.join(frame_folder, f'frame_{i:04d}.png')
        img = cv2.imread(frame_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = np.expand_dims(img, axis=0).astype(np.float32)
        processed_img = deep_dream_effect(img, model)
        processed_img = np.squeeze(processed_img)
        processed_img = ((processed_img + 1) / 2.0 * 255).astype(np.uint8)
        processed_img = cv2.cvtColor(processed_img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(dream_folder, f'dream_frame_{i:04d}.png'), processed_img)

# Reassemble video from frames
def create_video_from_frames(frame_folder, output_video_path, fps=30):
    frame_files = [os.path.join(frame_folder, f) for f in sorted(os.listdir(frame_folder)) if f.endswith('.png')]
    clip = ImageSequenceClip(frame_files, fps=fps)
    clip.write_videofile(output_video_path, codec='libx264')

# Main processing steps
frame_count = extract_frames(input_video_path, output_frame_path)
model = load_model()
process_frames(output_frame_path, output_dream_frame_path, model, frame_count)
create_video_from_frames(output_dream_frame_path, os.path.join(deep_dream_folder, 'deep_dream_video.mp4'))
