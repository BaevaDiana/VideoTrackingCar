import cv2
import time

# загрузка исходного видеофайла
cap = cv2.VideoCapture('./video_sources/example_5.mp4')

# создание объекта вычитания фона
fgbg = cv2.createBackgroundSubtractorMOG2()

# получение ширины и высоты кадра
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# создание объекта cv2.VideoWriter
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('./video_results/output_mog2_5.avi', fourcc, 20.0, (width, height))

start_time = time.time()
# чтение видеопотока и отслеживание объектов
while True:
    # чтение кадра из видеофайла
    ret, frame = cap.read()

    if not ret:
        break

    # применение вычитания фона к кадру
    fgmask = fgbg.apply(frame)

    # поиск контуров на маске переднего плана
    contours, hierarchy = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # перебор контуров
    for contour in contours:
        # получение ограничивающего прямоугольника каждого контура
        x, y, w, h = cv2.boundingRect(contour)

        # отрисовка прямоугольника вокруг контура, если он достаточно большой и имеет определенное соотношение сторон
        if w > 50 and h > 50 and w/h > 1.3:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # запись текущего кадра в файл
    out.write(frame)

    # отображение текущего кадра
    cv2.imshow('Tracking', frame)

    # выход из цикла при нажатии пользователем клавиши 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

end_time = time.time()
# вывод сравнительных характеристик
if cap.get(cv2.CAP_PROP_FRAME_COUNT) != 0:
    print(f"Время работы метода MOG2: {end_time - start_time:.5f} секунд")
    print(f"Скорость обработки: {cap.get(cv2.CAP_PROP_FPS):.0f} кадров/секунду")
    print(f"Частота потери изображения: {1 / ((end_time - start_time) / cap.get(cv2.CAP_PROP_POS_FRAMES)):.0f} кадров/секунду")
else:
    print("Видеофайл не содержит кадров.")

# освобождение ресурсов и закрытие всех окон
cap.release()
out.release()
cv2.destroyAllWindows()
