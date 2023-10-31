import cv2
import os

# Путь к папке с видеофайлами
video_folder = "video_sources"

# Получаем список видеофайлов в папке
video_files = [f for f in os.listdir(video_folder) if f.endswith(".mp4")]

# Проходимся по первым 5 видеофайлам (или меньшему количеству, если их меньше)
for video_file in video_files[:5]:
    video_path = os.path.join(video_folder, video_file)

    # Открываем видеофайл с помощью OpenCV
    video_capture = cv2.VideoCapture(video_path)

    # Проверяем, было ли успешное открытие видео
    if not video_capture.isOpened():
        print(f"Ошибка при открытии видеофайла: {video_file}")
    else:
        # Получаем информацию о видео
        frame_count = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_rate = int(video_capture.get(cv2.CAP_PROP_FPS))
        video_duration = frame_count / frame_rate

        # Получаем информацию о кодеке
        codec_num = int(video_capture.get(cv2.CAP_PROP_FOURCC))
        codec_str = chr(codec_num & 0xFF) + chr((codec_num >> 8) & 0xFF) + chr((codec_num >> 16) & 0xFF) + chr(
            (codec_num >> 24) & 0xFF)

        # Выводим полученные параметры
        print(f"Информация о видеофайле: {video_file}")
        print(f"Ширина кадра: {frame_width}")
        print(f"Высота кадра: {frame_height}")
        print(f"Частота кадров: {frame_rate} кадров в секунду")
        print(f"Длительность видео: {video_duration} секунд")
        print(f"Кодек видео: {codec_str}")
        print("-" * 40)

    # Закрываем видеофайл
    video_capture.release()
