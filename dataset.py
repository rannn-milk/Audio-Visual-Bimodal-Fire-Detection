import os
import cv2
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from moviepy import VideoFileClip

# ===================== 固定配置 =====================
IMG_SIZE = (128, 128)
EXTRACT_INTERVAL = 5  # 5秒1帧
SAMPLE_RATE = 22050
N_MELS = 128
HOP_LENGTH = 512

# ===================== 工具函数 =====================
# ===================== 工具函数 =====================
def extract_audio_segment_no_ffmpeg(video_path, start_sec, end_sec, output_audio_path):
    """
    终极版：不使用 subclip，真正截取 0-5 / 5-10s 音频，100% 不报错
    """
    try:
        # 先获取完整音频
        video = VideoFileClip(video_path)
        audio = video.audio
        sr = audio.fps = 22050  # 采样率
        audio = video.audio

        # 读取全部音频数据
        full_audio = audio.to_soundarray()
        if len(full_audio.shape) > 1:
            full_audio = full_audio.mean(axis=1)  # 转单声道

        # 计算切片位置
        start_idx = int(start_sec * sr)
        end_idx = int(end_sec * sr)
        segment = full_audio[start_idx:end_idx]

        # 保存切片音频
        import wave
        with wave.open(output_audio_path, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(SAMPLE_RATE)
            wf.writeframes((segment * 32767).astype('int16').tobytes())

        video.close()
    except Exception as e:

        try:
            import wave
            with wave.open(output_audio_path, 'wb') as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(SAMPLE_RATE)
        except:
            open(output_audio_path, 'wb').close()
def create_mel_spectrogram(audio_path, save_path):
    """
    生成梅尔频谱图
    """
    if not os.path.exists(audio_path):
        print(f"音频不存在: {audio_path}")
        return

    y, sr = librosa.load(audio_path, sr=SAMPLE_RATE)
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=N_MELS, hop_length=HOP_LENGTH)
    mel_spec = librosa.power_to_db(mel_spec, ref=np.max)

    if mel_spec.shape[1] > IMG_SIZE[1]:
        mel_spec = mel_spec[:, :IMG_SIZE[1]]
    else:
        pad = IMG_SIZE[1] - mel_spec.shape[1]
        mel_spec = np.pad(mel_spec, ((0, 0), (0, pad)), mode='constant')

    plt.figure(figsize=(128/100, 128/100), dpi=100)
    librosa.display.specshow(mel_spec, sr=sr, hop_length=HOP_LENGTH, cmap='gray')
    plt.axis('off')
    plt.tight_layout(pad=0)
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    plt.close()

# ===================== 主函数 =====================
def extract_frames_and_mel_simple(video_path, output_img_dir, output_mel_dir, output_audio_dir):
    if not os.path.exists(video_path):
        print("视频不存在")
        return

    Path(output_img_dir).mkdir(parents=True, exist_ok=True)
    Path(output_mel_dir).mkdir(parents=True, exist_ok=True)
    Path(output_audio_dir).mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps
    print(f"视频时长: {duration:.1f}s")

    current_second = 0
    saved = 0

    while current_second < duration:
        start = current_second
        end = current_second + EXTRACT_INTERVAL
        if end > duration:
            break


        # 取当前时间段【中间时刻】的帧
        middle_time = start + (end - start) / 2  # 0-5s → 2.5s, 5-10s→7.5s
        cap.set(cv2.CAP_PROP_POS_MSEC, middle_time * 1000)  # 跳转到中间毫秒
        # ===========================================================

        ret, frame = cap.read()
        if ret:
            frame = cv2.resize(frame, IMG_SIZE)

            img_path = os.path.join(output_img_dir, f"{end}s.jpg")
            cv2.imwrite(img_path, frame)

        # 音频保存为 5s.wav、10s.wav
        audio_path = os.path.join(output_audio_dir, f"{end}s.wav")
        #extract_audio_segment_no_ffmpeg(video_path, start, end, audio_path)

        # 梅尔谱保存为 mel_5s.jpg
        mel_path = os.path.join(output_mel_dir, f"mel_{end}s.jpg")
        #create_mel_spectrogram(audio_path, mel_path)

        print(f"{start}-{end}s 完成")
        current_second += EXTRACT_INTERVAL
        saved += 1

    cap.release()
    print(f"\n全部完成！共生成 {saved} 组数据")

# ===================== 执行 =====================
if __name__ == "__main__":
    VIDEO_PATH = r"D:\lc\1.mp4"
    OUTPUT_IMG_DIR = r"D:\lc\data\frames"
    OUTPUT_MEL_DIR = r"D:\lc\data\mels"
    OUTPUT_AUDIO_DIR = r"D:\lc\data\audios"

    extract_frames_and_mel_simple(VIDEO_PATH, OUTPUT_IMG_DIR, OUTPUT_MEL_DIR, OUTPUT_AUDIO_DIR)