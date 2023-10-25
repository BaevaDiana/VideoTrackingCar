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

# Функция для отслеживания объекта на видео
def track_object(video_path):
    cap = cv2.VideoCapture(video_path)
    tracker = cv2.TrackerMedianFlow_create()

    # Выбор объекта для трекинга
    ret, frame = cap.read()
    if not ret:
        print("Не удалось прочитать видео.")
        return
    bbox = cv2.selectROI('Select Object', frame)
    tracker.init(frame, bbox)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        success, bbox = tracker.update(frame)

        if success:
            # Отрисовка прямоугольника вокруг объекта
            x, y, w, h = [int(i) for i in bbox]
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        cv2.imshow('Object Tracking', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Путь к видеофайлу, в котором нужно отслеживать объект
video_path = 'путь_к_вашему_видеофайлу.mp4'
track_object(video_path)
