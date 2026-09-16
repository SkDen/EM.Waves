
from EMVisualizer   import EMVisualizer


video = "./out_video/test.mp4"
output_dir = './results'

amp = 0.0005

print("Создание видео...")
try:
    viz = EMVisualizer(output_dir, vmin=-amp, vmax=amp)
    viz.create_video(video, fps=20, use_opencv=True)
except Exception as e:
    print(f"Ошибка при создании видео (OpenCV): {e}")
    try:
        viz.create_video(video, fps=20, use_opencv=False)
    except Exception as e2:
        print(f"Не удалось создать видео (matplotlib): {e2}")