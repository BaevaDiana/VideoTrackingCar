import cv2
import numpy as np

# Открываем видеопоток
cap = cv2.VideoCapture('video_sources/example_1.mp4')  # Замените 'path_to_video.mp4' на путь к видео

# Определяем параметры для cv2.goodFeaturesToTrack
feature_params = dict(
    maxCorners=100,     # Максимальное количество точек, которые мы хотим найти
    qualityLevel=0.3,   # Минимальное качество точки (0-1)
    minDistance=7,      # Минимальное расстояние между точками
    blockSize=7         # Размер окна для вычисления углов
)

# Инициализируем начальные точки для отслеживания
initial_points = None

while True:
    ret, frame = cap.read()

    if not ret:
        break

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Ищем хорошие точки для отслеживания в области машины
    if initial_points is None:
        # Определите область, где находится машина (например, через ROI)
        # Найдите хорошие точки в этой области
        roi = gray_frame[y_start:y_end, x_start:x_end]
        initial_points = cv2.goodFeaturesToTrack(roi, mask=None, **feature_params)

        # Преобразуйте координаты точек обратно в координаты кадра
        initial_points += (x_start, y_start)

    if initial_points is not None:
        # Вычисляем оптический поток для отслеживания точек
        next_points, status, _ = cv2.calcOpticalFlowPyrLK(gray_frame, previous_frame, initial_points, None)

        # Отфильтруйте точки, которые не были успешно отслежены
        good_points = next_points[status == 1]
        initial_points = good_points

        # Отрисовываем отслеживаемые точки
        for point in initial_points:
            x, y = point.ravel()
            cv2.circle(frame, (x, y), 3, (0, 255, 0), -1)

    previous_frame = gray_frame

    cv2.imshow('Car Tracking', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
