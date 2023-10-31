import cv2
import numpy as np
import time

# обнаружение объектов с помощью YOLO
def apply_yolo_object_detection(image_to_process):
    # получение высоты и ширины изображения
    height, width, _ = image_to_process.shape
    # предварительная обработку изображения, включая масштабирование, нормализацию и изменение порядка цветовых каналов
    # и создание 4D-массив (блоб)
    blob = cv2.dnn.blobFromImage(image_to_process, 1 / 255, (608, 608),
                                 (0, 0, 0), swapRB=True, crop=False)
    # передача массива в нейронную сеть
    net.setInput(blob)
    # прямой проход по сети и получение выходных данных
    outs = net.forward(out_layers)
    # сохранение индексов классов, оценки классов и координат ограничивающих рамок
    class_indexes, class_scores, boxes = ([] for i in range(3))
    objects_count = 0

    # поиск объектов на изображении
    """ 
    Переменная outs содержит выходные данные модели нейронной сети для каждого объекта, обнаруженного на изображении.
    Код перебирает каждый объект и извлекает координаты его ограничивающей рамки, индекс класса и оценку класса. 
    Ограничивающая рамка определяется своими координатами центра, шириной и высотой. 
    Полученная ограничивающая рамка затем добавляется в список всех обнаруженных рамок вместе с индексом класса и оценкой.
    """
    for out in outs:
        for obj in out:
            scores = obj[5:]
            class_index = np.argmax(scores)
            class_score = scores[class_index]
            if class_score > 0:
                center_x = int(obj[0] * width)
                center_y = int(obj[1] * height)
                obj_width = int(obj[2] * width)
                obj_height = int(obj[3] * height)
                box = [center_x - obj_width // 2, center_y - obj_height // 2,
                       obj_width, obj_height]
                boxes.append(box)
                class_indexes.append(class_index)
                class_scores.append(float(class_score))

    # выборка только наиболее точных рамок
    chosen_boxes = cv2.dnn.NMSBoxes(boxes, class_scores, 0.0, 0.4)
    # перебор каждой рамки и добавление в список всех обнаруженных рамок
    for box_index in chosen_boxes:
        box_index = box_index
        box = boxes[box_index]
        class_index = class_indexes[box_index]
        # если класс объекта находится в списке классов, которые нужно обнаружить
        if classes[class_index] in classes_to_look_for:
            # увеличение счётчика обнаруженных объектов
            objects_count += 1
            # отрисовка прямоугольника возле отслеживаемого объекта
            image_to_process = draw_object_bounding_box(image_to_process,
                                                        class_index, box)
    # отрисовка количества найденных объектов на изображении
    final_image = draw_object_count(image_to_process, objects_count)

    # возвращение изменённого изображения
    return final_image

# рисование границ объекта с подписью отслеживаемого класса
def draw_object_bounding_box(image_to_process, index, box):
    # задание параметров отображаемых прямоугольников
    x, y, w, h = box
    start = (x, y)
    end = (x + w, y + h)
    color = (0, 255, 0)
    width = 2
    # отображение прямоугольника
    final_image = cv2.rectangle(image_to_process, start, end, color, width)

    # задание параметров отображаемого текста
    start = (x, y - 10)
    font_size = 1
    font = cv2.FONT_HERSHEY_SIMPLEX
    width = 2
    text = classes[index]
    # отображение надписи класса
    final_image = cv2.putText(final_image, text, start, font,
                              font_size, color, width, cv2.LINE_AA)

    # возвращение изменённого изображения
    return final_image

# отрисовка количества найденных объектов на изображении
def draw_object_count(image_to_process, objects_count):
    # задание параметров отображаемого текста
    start = (10, 120)
    font_size = 1.5
    font = cv2.FONT_HERSHEY_SIMPLEX
    width = 3
    text = "Objects found: " + str(objects_count)
    white_color = (255, 255, 255)
    black_outline_color = (0, 0, 0)

    # создание текстовых блоков
    final_image = cv2.putText(image_to_process, text, start, font, font_size,
                              black_outline_color, width * 3, cv2.LINE_AA)
    final_image = cv2.putText(final_image, text, start, font, font_size,
                              white_color, width, cv2.LINE_AA)

    # возвращение изменённого изображения
    return final_image

# захват и анализ видео в режиме реального времени
def start_video_object_detection():
    # загрузка исходного видеофайла
    cap = cv2.VideoCapture('./video_sources/example_5.mp4')
    # чтение первого кадра из видеофайла
    ret, frame = cap.read()
    # создание объекта VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('./video_results/output_yolo_1.avi', fourcc, 20.0, (frame.shape[1], frame.shape[0]))
    start_time = time.time()
    while True:
        ret, frame = cap.read()

        if not ret:
            break

        # применение методов обнаружения объектов YOLO на каждом кадре
        frame = apply_yolo_object_detection(frame)

        # # изменение размера обработанного видео и отображение его на экране
        # frame = cv2.resize(frame, (1920 // 2, 1080 // 2))

        # запись обработанного кадра в файл
        out.write(frame)

        cv2.imshow("Tracking", frame)

        # выход из цикла по нажатию клавиши 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    end_time = time.time()
    # вывод сравнительных характеристик
    if cap.get(cv2.CAP_PROP_FRAME_COUNT) != 0:
        print(f"Время работы метода MOG2: {end_time - start_time:.5f} секунд")
        print(f"Скорость обработки: {cap.get(cv2.CAP_PROP_FPS):.0f} кадров/секунду")
        print(
            f"Частота потери изображения: {1 / ((end_time - start_time) / cap.get(cv2.CAP_PROP_POS_FRAMES)):.0f} кадров/секунду")
    else:
        print("Видеофайл не содержит кадров.")

    # освобождение ресурсов и закрытие окон
    cap.release()
    out.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    # загрузка предварительно обученной модели YOLOv4-tiny, используя файлы конфигурации и весов
    net = cv2.dnn.readNetFromDarknet("resources_for_yolo/yolov4-tiny.cfg",
                                     "resources_for_yolo/yolov4-tiny.weights")
    # возвращение списка имен всех слоев в сети
    layer_names = net.getLayerNames()
    # возвращение индексов выходных слоев, которые не соединены с другими слоями
    out_layers_indexes = net.getUnconnectedOutLayers()
    # создание списка имен выходных слоев, которые используются для обнаружения объектов на изображении
    out_layers = [layer_names[index - 1] for index in out_layers_indexes]

    # загрузка из файла классов объектов, обнаруживаемых YOLO
    # определение классов объектов, которые мы будем искать
    with open("resources_for_yolo/coco.names.txt") as file:
        classes = file.read().split("\n")

    look_for = input("Что мы отслеживаем: ").split(',')

    # удаление пробелов
    list_look_for = []
    for look in look_for:
        list_look_for.append(look.strip())
    classes_to_look_for = list_look_for

    # вызов функции детекции
    start_video_object_detection()

# закрытие окон
cv2.destroyAllWindows()