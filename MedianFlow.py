import cv2

# Загрузка видеофайла
video = cv2.VideoCapture('./video_sources/example_4.mp4')

# Инициализация MedianFlow Tracker
tracker = cv2.legacy.TrackerMedianFlow_create()

# Задание начального прямоугольника для отслеживания
_, frame = video.read()
bbox = cv2.selectROI(frame, False)
tracker.init(frame, bbox)

# Чтение видеопотока и отслеживание объектов
while True:
    # Чтение кадра из видеопотока
    ret, frame = video.read()
    if not ret:
        break

    # Обновление трекера и получение нового прямоугольника
    success, bbox = tracker.update(frame)

    # Отображение прямоугольника вокруг объекта
    if success:
        x, y, w, h = [int(i) for i in bbox]
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Отображение кадра с выделенными объектами
    cv2.imshow('Tracking', frame)

    # Выход из цикла по нажатию клавиши 'q'
    if cv2.waitKey(1) & 0xff == ord('q'):
        break

# Освобождение ресурсов и закрытие окон
video.release()
cv2.destroyAllWindows()
