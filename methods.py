# методы
#MedianFlow Tracker
#KLT (KLT Tracker):
#CSRT (Discriminative Correlation Filter with Channel and Spatial Reliability)


#KCF
#DCF
#

#пример для получения данных видео
# Загрузка видеофайла
# video_path = 'путь_к_вашему_видеофайлу.mp4'
# cap = cv2.VideoCapture(video_path)
#
# # Проверка на успешное открытие видеофайла
# if not cap.isOpened():
#     print("Ошибка при открытии видеофайла.")
# else:
#     # Определение параметров видео
#     codec = int(cap.get(cv2.CAP_PROP_FOURCC))
#     frame_rate = cap.get(cv2.CAP_PROP_FPS)
#     duration = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) / frame_rate
#
#     print(f"Кодек: {codec}")
#     print(f"Частота кадров: {frame_rate} FPS")
#     print(f"Длительность видео: {duration} секунд")

import cv2
import numpy as np

# Функция для инициализации трекера KLT
def init_klt_tracker(image, bbox):
    x, y, w, h = bbox
    roi = image[y:y+h, x:x+w]
    gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    # Определение угловых точек (features) в ROI
    corners = cv2.goodFeaturesToTrack(gray_roi, maxCorners=100, qualityLevel=0.01, minDistance=10)

    # Преобразование координат угловых точек в глобальные координаты
    corners = corners + np.array([x, y])

    # Инициализация трекера KLT
    p0 = corners.reshape(-1, 1, 2).astype(np.float32)
    return p0, gray_roi

# Функция для обновления KLT трекера
def update_klt_tracker(image, p0, prev_gray):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Определение новых положений угловых точек
    p1, st, err = cv2.calcOpticalFlowPyrLK(prev_gray, gray, p0, None)

    # Выбор хороших точек
    p0 = p0[st == 1]
    p1 = p1[st == 1]

    return p0, p1, gray

# Функция для визуализации трека объекта
def draw_tracks(frame, p0, p1):
    for i, (new, old) in enumerate(zip(p1, p0)):
        a, b = new.ravel()
        c, d = old.ravel()
        frame = cv2.line(frame, (a, b), (c, d), (0, 0, 255), 2)
        frame = cv2.circle(frame, (a, b), 5, (0, 0, 255), -1)
    return frame

# Основная функция для трекинга объекта на видео
def track_object_klt(video_path):
    cap = cv2.VideoCapture(video_path)

    # Чтение первого кадра
    ret, frame = cap.read()
    if not ret:
        print("Не удалось прочитать видео.")
        return

    # Выбор объекта для трекинга
    bbox = cv2.selectROI('Select Object', frame)
    p0, prev_gray = init_klt_tracker(frame, bbox)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        p0, p1, prev_gray = update_klt_tracker(frame, p0, prev_gray)

        frame = draw_tracks(frame, p0, p1)

        cv2.imshow('Object Tracking', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Путь к видеофайлу, в котором нужно отслеживать объект
video_path = 'video_sources/example_1.mp4'
track_object_klt(video_path)
