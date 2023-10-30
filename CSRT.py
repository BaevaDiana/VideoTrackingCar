import cv2

# Открываем видеофайл
cap = cv2.VideoCapture('./video_sources/example_3.mp4')

# Инициализируем трекер CSRT
tracker = cv2.TrackerCSRT_create()


# Читаем первый кадр из видеофайла
ret, frame = cap.read()

# Выбираем область интереса (ROI) для отслеживания
bbox = cv2.selectROI(frame, False)

# Инициализируем трекер с помощью первого кадра и ROI
tracker.init(frame, bbox)

while True:
    # Читаем кадр из видеофайла
    ret, frame = cap.read()

    # Обновляем трекер на текущем кадре
    success, bbox = tracker.update(frame)

    # Отображаем ROI на текущем кадре
    if success:
        x, y, w, h = [int(i) for i in bbox]
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    else:
        cv2.putText(frame, "Tracking failed!", (100, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)

    # Отображаем текущий кадр
    cv2.imshow('frame', frame)

    # Выходим из цикла при нажатии клавиши 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Освобождаем ресурсы и закрываем окна
cap.release()
cv2.destroyAllWindows()

