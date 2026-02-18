import org.bytedeco.opencv.global.opencv_core;
import org.bytedeco.opencv.global.opencv_imgcodecs;
import org.bytedeco.opencv.global.opencv_imgproc;
import org.bytedeco.opencv.opencv_core.*;
import org.bytedeco.opencv.opencv_face.LBPHFaceRecognizer;

import java.io.File;
import java.nio.IntBuffer;
import java.util.ArrayList;
import java.util.List;

public class TrainModel {

    public static void main(String[] args) {
        String datasetPath = "faces"; // Main folder with subfolders person1, person2, etc.

        List<Mat> images = new ArrayList<>();
        List<Integer> labelsList = new ArrayList<>();

        File mainDir = new File(datasetPath);
        File[] personFolders = mainDir.listFiles(File::isDirectory);

        if (personFolders == null || personFolders.length == 0) {
            System.out.println(" No person folders found inside " + datasetPath);
            return;
        }

        int label = 0;
        for (File personFolder : personFolders) {
            label++; // assign an incremental numeric label
            File[] imageFiles = personFolder.listFiles((dir, name) -> name.endsWith(".jpg") || name.endsWith(".png"));

            if (imageFiles == null) continue;

            for (File imageFile : imageFiles) {
                Mat img = opencv_imgcodecs.imread(imageFile.getAbsolutePath(), opencv_imgcodecs.IMREAD_GRAYSCALE);
                if (img.empty()) {
                    System.out.println("Could not read: " + imageFile.getAbsolutePath());
                    continue;
                }

                opencv_imgproc.resize(img, img, new Size(200, 200));
                images.add(img);
                labelsList.add(label);
            }

            System.out.println(" Loaded " + images.size() + " images so far (Person label: " + label + ")");
        }

        if (images.isEmpty()) {
            System.out.println(" No valid images found for training.");
            return;
        }

        // Convert labels list to Mat
        Mat labels = new Mat(labelsList.size(), 1, opencv_core.CV_32SC1);
        IntBuffer labelsBuf = labels.createBuffer();
        for (int i = 0; i < labelsList.size(); i++) {
            labelsBuf.put(i, labelsList.get(i));
        }

        // Convert images list to MatVector
        MatVector matVector = new MatVector(images.size());
        for (int i = 0; i < images.size(); i++) {
            matVector.put(i, images.get(i));
        }

        // Train LBPH recognizer
        LBPHFaceRecognizer faceRecognizer = LBPHFaceRecognizer.create();

        System.out.println("🔹 Training model on " + images.size() + " images...");
        faceRecognizer.train(matVector, labels);

        // Save the trained model
        faceRecognizer.save("trained_model.xml");
        System.out.println(" Model trained and saved successfully as 'trained_model.xml'");
    }
}
