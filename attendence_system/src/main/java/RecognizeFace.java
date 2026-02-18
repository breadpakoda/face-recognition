import org.bytedeco.opencv.global.opencv_core;
import org.bytedeco.opencv.global.opencv_imgproc;
import org.bytedeco.opencv.global.opencv_highgui;
import org.bytedeco.opencv.opencv_core.*;
import org.bytedeco.opencv.opencv_face.LBPHFaceRecognizer;
import org.bytedeco.opencv.opencv_objdetect.CascadeClassifier;
import org.bytedeco.opencv.opencv_videoio.VideoCapture;

import java.io.File;
import java.util.HashMap;
import java.util.Map;

public class RecognizeFace {

    public static void main(String[] args) {
        // Path setup
        String modelPath = "trained_model.xml";
        String facesPath = "faces";
        String cascadePath = "haarcascade_frontalface_alt.xml";

        // Check model
        File modelFile = new File(modelPath);
        if (!modelFile.exists()) {
            System.out.println("❌ Trained model not found: " + modelPath);
            return;
        }

        // Load face recognizer
        LBPHFaceRecognizer faceRecognizer = LBPHFaceRecognizer.create();
        faceRecognizer.read(modelPath);

        // Load Haar Cascade
        CascadeClassifier faceCascade = new CascadeClassifier(cascadePath);
        if (faceCascade.empty()) {
            System.out.println("❌ Haar Cascade file not found: " + cascadePath);
            return;
        }

        // Build mapping (label -> folder name)
        Map<Integer, String> labelNames = new HashMap<>();
        File facesDir = new File(facesPath);
        File[] personDirs = facesDir.listFiles(File::isDirectory);

        if (personDirs == null) {
            System.out.println("❌ No folders found inside " + facesPath);
            return;
        }

        int labelCounter = 1;
        for (File personDir : personDirs) {
            labelNames.put(labelCounter, personDir.getName());
            labelCounter++;
        }

        // Start webcam
        VideoCapture camera = new VideoCapture(0);
        if (!camera.isOpened()) {
            System.out.println("❌ Cannot open camera");
            return;
        }

        Mat frame = new Mat();
        Mat gray = new Mat();

        System.out.println("🎥 Starting recognition... Press 'q' to quit.");

        while (true) {
            if (!camera.read(frame) || frame.empty()) continue;

            opencv_imgproc.cvtColor(frame, gray, opencv_imgproc.COLOR_BGR2GRAY);
            RectVector faces = new RectVector();
            faceCascade.detectMultiScale(gray, faces);

            for (int i = 0; i < faces.size(); i++) {
                Rect faceRect = faces.get(i);
                Mat face = new Mat(gray, faceRect);
                opencv_imgproc.resize(face, face, new Size(200, 200));

                int[] label = new int[1];
                double[] confidence = new double[1];
                faceRecognizer.predict(face, label, confidence);

                String name;
                if (confidence[0] < 80) {
                    name = labelNames.getOrDefault(label[0], "Unknown");
                } else {
                    name = "Unknown";
                }

                // Draw rectangle & name
                opencv_imgproc.rectangle(frame, faceRect, new Scalar(0, 255, 0, 0), 2, 8, 0);
                opencv_imgproc.putText(
                        frame,
                        name + " (" + String.format("%.1f", confidence[0]) + ")",
                        new Point(faceRect.x(), faceRect.y() - 10),
                        opencv_imgproc.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        new Scalar(0, 255, 0, 0)
                );
            }

            opencv_highgui.imshow("Face Recognition", frame);

            if (opencv_highgui.waitKey(20) == 'q') break;
        }

        camera.release();
        opencv_highgui.destroyAllWindows();
        System.out.println("🟢 Recognition ended.");
    }
}
