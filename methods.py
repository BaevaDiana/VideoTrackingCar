# методы
#MedianFlow Tracker
#KLT (KLT Tracker):
#CSRT (Discriminative Correlation Filter with Channel and Spatial Reliability)


#KCF
#
#

#пример для получения данных видео
# Загрузка видеофайла
video_path = 'путь_к_вашему_видеофайлу.mp4'
cap = cv2.VideoCapture(video_path)

# Проверка на успешное открытие видеофайла
if not cap.isOpened():
    print("Ошибка при открытии видеофайла.")
else:
    # Определение параметров видео
    codec = int(cap.get(cv2.CAP_PROP_FOURCC))
    frame_rate = cap.get(cv2.CAP_PROP_FPS)
    duration = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) / frame_rate

    print(f"Кодек: {codec}")
    print(f"Частота кадров: {frame_rate} FPS")
    print(f"Длительность видео: {duration} секунд")

