import org.opencv.core.*;
import org.opencv.dnn.Dnn;
import org.opencv.dnn.Net;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import org.opencv.videoio.VideoCapture;

import javax.swing.*;
import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.Scanner;

public class ObjectsDetection {
    static {
        // Загружаем OpenCV и проверяем её версию
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
        System.out.println("Version: " + Core.VERSION);
    }

    public static void main(String[] args) {
        Scanner scanner = new Scanner(System.in);
        JFrame window = new JFrame();                          // Создаём окно для просмотра изображений
        JLabel screen = new JLabel();                          // Создаём контейнер для изображения
        window.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE); // Указывам дефолтное действие при закрытии окна
        window.setVisible(true);                               // Делаем созданное окно видимым

        // Инициализируем переменную для захвата кадров из видеопотока
        VideoCapture cap = null;

        System.out.println("Выберите источник видео (камера / файл (должен лежать по адресу \"src/testVideo.mp4\") ?:");

        switch (scanner.nextLine()) {
            case "камера":
                cap = new VideoCapture(0);
                break;
            case "файл":
                cap = new VideoCapture("src/testVideo.mp4");
                break;
            default:
                System.out.println("Третьего не дано");
        }

        // Инициализируем переменные.
        Mat frame = new Mat();           // Кадр изображения
        Mat frameResized = new Mat();    // Кадр изображения после ресайза
        MatOfByte buf = new MatOfByte(); // Конвертим эту матрицу в понятную для жавы байтовую матрицу

        /* Эти пороговые значения определяют точность нейросети, т.е., если нейросеть предполагает, что на изображении
        с вероятностью 80% какой-то пёсон, то она нарисует вокруг него идентификационную рамочку */
        float minProbability = 0.5f;
        float threshold = 0.3f;

        int height; // Содержит высоту кадра
        int width;  // Содержит ширину кадра
        ImageIcon ic;

        List<String> labels = labels("src/coco.names"); // Извлекаем из coco.names распознаваемые объекты
        int amountOfClasses = labels.size();                 // Содержит количество этих объектов

        // Для каждого объекта случайным образом генерируем свой цвет ограничительной рамки в RGB-пространстве
        Random random = new Random();
        Scalar[] colors = new Scalar[amountOfClasses];

        for (int i = 0; i < amountOfClasses; i++) {
            colors[i] = new Scalar(
                    random.nextInt(256),
                    random.nextInt(256),
                    random.nextInt(256)
            );
        }

        // Инициализируем нейронную сеть
        String cfgPath = "src/yolov4-p6.cfg";                           // Содержит структуру (слои) нейросети
        String weightsPath = "src/yolov4-p6.weights";                   // Содержит результат обучения нейросети
        // Указываем, что распознавание будет производиться при помощи yolo4-нейросети
        Net network = Dnn.readNetFromDarknet(cfgPath, weightsPath);
        // Извлекаем наименования всех слоёв нейросети
        List<String> namesOfAllLayers = network.getLayerNames();
        // Извлекаем индексы выходных слоёв нейросети (их всего 3)
        MatOfInt outputLayersIndexes = network.getUnconnectedOutLayers();
        int amountOfOutputLayers = outputLayersIndexes.toArray().length; // Содержит количество выходных слоёв

        // В цикле извлекаем наименования выходных слоёв из namesOfAllLayers
        List<String> outputLayersNames = new ArrayList<>();

        for (int i = 0; i < amountOfOutputLayers; i++) {
            outputLayersNames.add(namesOfAllLayers.get(outputLayersIndexes.toList().get(i) - 1));
        }

        if (!cap.isOpened()) {
            System.out.println("Could not open video device");
        } else {

            // В бесконечном цикле обрабатываем поступающие кадры из видеопотока.
            while (true) {
                cap.read(frame);         // Извлекаем кадр из видеопотока
                height = frame.height(); // Извлекаем высоту кадра
                width = frame.width();   // Извлекаем ширину кадра
                /* Берём текущий видеокадр, уменьшаем его (кратно 32), чтобы снизить нагрузку на нейросеть (стандартные
                640х480 для неё тяжеловаты) и записываем в новую переменную */
                Imgproc.resize(frame, frameResized, new Size(192, 192));
                // Подготавливаем партию изображений (blob)
                Mat blob = Dnn.blobFromImage(frameResized, 1 / 255.0);
                // Подаём blob на вход нейросети.
                network.setInput(blob);

                // Извлекаем данные с выходных слоёв нейросети в отдельный список
                List<Mat> outputFromNetwork = new ArrayList<>();

                for (int i = 0; i < amountOfOutputLayers; i++) {
                    outputFromNetwork.add(network.forward(outputLayersNames.get(i)));
                }

                // Координаты обнаруженных ограничительных рамок заносим в список и конвертируем при помощи MatOfRect2d
                List<Rect2d> boundingBoxesList = new ArrayList<>();
                MatOfRect2d boundingBoxes = new MatOfRect2d();
                // Предсказаные вероятности заносим в список и конвертируем при помощи MatOfFloat
                List<Float> confidencesList = new ArrayList<>();
                MatOfFloat confidences = new MatOfFloat();
                // Индексы предсказаных объектов также заносим в отдельный список.
                List<Integer> classIndexes = new ArrayList<>();

                // Проходим в цикле через все предсказания из выходных слоёв
                for (int i = 0; i < amountOfOutputLayers; i++) {
                    System.out.println(outputFromNetwork.get(i).size());

                    // Проходим через все предсказания из слоя:
                    for (int b = 0; b < outputFromNetwork.get(i).size().height; b++) {
                        // Заносим из слоя в список вероятность для каждого объекта
                        double[] scores = new double[amountOfClasses];

                        for (int c = 0; c < amountOfClasses; c++) {
                            scores[c] = outputFromNetwork.get(i).get(b, c + 5)[0];
                        }

                        // Получаем индекс объекта с максимальной вероятностью
                        int indexOfMaxValue = 0;

                        for (int c = 0; c < amountOfClasses; c++) {
                            indexOfMaxValue = (scores[c] > scores[indexOfMaxValue]) ? c : indexOfMaxValue;
                        }

                        // Сохраняем значение максимальной вероятности
                        double maxProbability = scores[indexOfMaxValue];

                        // Если вероятность больше заданого минимума,
                        if (maxProbability > minProbability) {
                        /* то извлекаем значения точек ограничительной рамки из слоя, расчитываем нужные значения,
                           получаем ширину, высоту, начальные координаты по "x" и "y",
                           заносим значения в объект типа Rect2d (проще говоря - рисуем рамки вокруг объектов) */
                            double boxWidth = outputFromNetwork.get(i).get(b, 2)[0] * width;
                            double boxHeight = outputFromNetwork.get(i).get(b, 3)[0] * height;
                            Rect2d boxRect2d = new Rect2d(
                                    (outputFromNetwork.get(i).get(b, 0)[0] * width) - (boxWidth / 2),
                                    (outputFromNetwork.get(i).get(b, 1)[0] * height) - (boxHeight / 2),
                                    boxWidth,
                                    boxHeight
                            );

                            // Заносим в список параметры ограничительной рамки
                            boundingBoxesList.add(boxRect2d);
                            // Заносим в список максимальную вероятность
                            confidencesList.add((float) maxProbability);
                            // Заносим в список индекс предполагаемого класса
                            classIndexes.add(indexOfMaxValue);
                        }
                    }
                }

                // Конвертируем списки в соответствующие матрицы
                boundingBoxes.fromList(boundingBoxesList);
                confidences.fromList(confidencesList);


            /* Так как каждому объекту на изображении как правило могут соответствовать несколько ограничительных рамок,
               то необходимо выбирать для каждого обьекта наиболее подходящую.
               Для этого пропускаем все обнаруженные рамки через алгоритм "non-maximum suppression".
               Функция Dnn.NMSBoxes возвращает матрицу с индексами (MatOfInt indices) для наиболее подходящих рамок. */

                // Инициализируем матрицу для NMSBoxes.
                MatOfInt indices = new MatOfInt();
                Dnn.NMSBoxes(boundingBoxes, confidences, minProbability, threshold, indices);

                // Если алгоритм ограничительные рамки, то наносим их на изображение
                if (indices.size().height > 0) {

                    for (int i = 0; i < indices.toList().size(); i++) {
                        // Создаём объект класса Rect, на основе которого будет нанесена ограничительная рамка
                        Rect rect = new Rect(
                                (int) boundingBoxes.toList().get(indices.toList().get(i)).x,
                                (int) boundingBoxes.toList().get(indices.toList().get(i)).y,
                                (int) boundingBoxes.toList().get(indices.toList().get(i)).width,
                                (int) boundingBoxes.toList().get(indices.toList().get(i)).height
                        );

                        // Извлекаем индекс выявленного на изображении объекта
                        int classIndex = classIndexes.get(indices.toList().get(i));

                        // Наносим ограничительную рамку
                        Imgproc.rectangle(frame, rect, colors[classIndex], 2);

                        // Форматируем строку для нанесения на изображение: <распознанный объект>: <вероятность>
                        String Text = labels.get(classIndex) + ": " + confidences.toList().get(i);
                        // Инициализируем точку для нанесения текста.
                        Point textPoint = new Point(
                                (int) boundingBoxes.toList().get(indices.toList().get(i)).x,
                                (int) boundingBoxes.toList().get(indices.toList().get(i)).y - 10
                        );
                        // Наносим текст на изображение
                        Imgproc.putText(frame, Text, textPoint, 1, 1.5, colors[classIndex]);
                    }
                }

                // Преобразуем изображение в матрицу байтов для получения массива байтов (пикселей)
                Imgcodecs.imencode(".png", frame, buf);
                // Преобразуем массив пикселей в отображаемое изображение
                ic = new ImageIcon(buf.toArray());
                // Кладём изображение в контейнер
                screen.setIcon(ic);
                screen.repaint();
                // Привязываем контейнер к окну отображения
                window.setContentPane(screen);
                window.pack();
            }
        }
    }

    // Функция для парсинга coco.names
    public static List<String> labels(String path) {
        List<String> labels = new ArrayList<>();

        try {
            Scanner scanner = new Scanner(new File(path));

            while (scanner.hasNext()) {
                String label = scanner.nextLine();
                labels.add(label);
            }
        } catch (IOException e) {
            e.printStackTrace();
        }

        return labels;
    }
}