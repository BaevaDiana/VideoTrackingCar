import cv2
import time

# загрузка исходного видеофайла
cap = cv2.VideoCapture('./video_sources/example_4.mp4')

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
out = cv2.VideoWriter('./video_results/output_medianflow_4.avi', fourcc, 20.0, (width, height))

start_time = time.time()
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

end_time = time.time()
# вывод сравнительных характеристик
if cap.get(cv2.CAP_PROP_FRAME_COUNT) != 0:
    print(f"Время работы метода MedianFlow: {end_time - start_time:.5f} секунд")
    print(f"Скорость обработки: {cap.get(cv2.CAP_PROP_FPS):.0f} кадров/секунду")
    print(f"Частота потери изображения: {1 / ((end_time - start_time) / cap.get(cv2.CAP_PROP_POS_FRAMES)):.0f} кадров/секунду")
else:
    print("Видеофайл не содержит кадров.")
# освобождение ресурсов и закрытие окон
cap.release()
out.release()
cv2.destroyAllWindows()
