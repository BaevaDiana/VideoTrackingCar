import cv2

# загрузка исходного видеофайла
cap = cv2.VideoCapture('./video_sources/example_5.mp4')

# инициализация MedianFlow Tracker
tracker = cv2.legacy.TrackerMedianFlow_create()

# получение ширины и высоты кадра
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# задание начального прямоугольника для отслеживания
_, frame = cap.read()
bbox = cv2.selectROI(frame, False)
tracker.init(frame, bbox)

# создание объекта cv2.VideoWriter
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('./video_results/output_medianflow_5.avi', fourcc, 20.0, (width, height))

# чтение видеопотока и отслеживание объектов
while True:
    # чтение кадра из видеофайла
    ret, frame = cap.read()

    if not ret:
        break

    # обновление трекера и получение нового прямоугольника
    success, bbox = tracker.update(frame)

    # отображение прямоугольника вокруг объекта
    if success:
        x, y, w, h = [int(i) for i in bbox]
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # запись текущий кадр в файл
    out.write(frame)

    # отображение кадра с выделенными объектами
    cv2.imshow('Tracking', frame)

    # выход из цикла по нажатию клавиши 'q'
    if cv2.waitKey(1) & 0xff == ord('q'):
        break

# освобождение ресурсов и закрытие окон
cap.release()
out.release()
cv2.destroyAllWindows()
