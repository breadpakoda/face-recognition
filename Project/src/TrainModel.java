import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.MatOfInt;
import org.opencv.core.Size;
import org.opencv.face.LBPHFaceRecognizer;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class TrainModel {

    public static void main(String[] args) {
        // Load the OpenCV native library
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);

        String datasetPath = "faces";
        List<Mat> faces = new ArrayList<>();
        List<Integer> labelsList = new ArrayList<>();
        Map<Integer, String> namesMap = new HashMap<>();
        int currentId = 0;

        System.out.println("ℹ️ Starting to read images from " + datasetPath + "...");

        File rootDir = new File(datasetPath);
        File[] personDirs = rootDir.listFiles();

        if (personDirs == null) {
            System.out.println("❌ Error: Could not list directories in " + datasetPath);
            System.out.println("Make sure the 'faces' folder exists and you have run CaptureFaces.java");
            return;
        }

        // Loop through all subfolders (each person)
        for (File personDir : personDirs) {
            if (personDir.isDirectory()) {
                String personName = personDir.getName();
                namesMap.put(currentId, personName);
                System.out.println(" -> Training on: " + personName + " (Label: " + currentId + ")");

                File[] imageFiles = personDir.listFiles();
                if (imageFiles == null) continue;

                // Loop through all images for that person
                for (File imageFile : imageFiles) {
                    // Simple check for image file extensions
                    String fileName = imageFile.getName().toLowerCase();
                    if (fileName.endsWith(".jpg") || fileName.endsWith(".png") || fileName.endsWith(".pgm")) {

                        // Read the image in grayscale
                        Mat img = Imgcodecs.imread(imageFile.getAbsolutePath(), Imgcodecs.IMREAD_GRAYSCALE);
                        if (img.empty()) {
                            System.out.println("⚠️ Warning: Could not read " + imageFile.getAbsolutePath());
                            continue;
                        }

                        // Resize the image (must be same size as in recognition)
                        Mat resizedImg = new Mat();
                        Size size = new Size(200, 200);
                        Imgproc.resize(img, resizedImg, size);

                        faces.add(resizedImg);
                        labelsList.add(currentId);
                    }
                }
                currentId++;
            }
        }

        if (faces.isEmpty()) {
            System.out.println("❌ Error: No faces found to train. Did you run CaptureFaces first?");
            return;
        }

        System.out.println("✅ Images read. Found " + faces.size() + " total images.");
        System.out.println("⏳ Starting training...");

        // Convert the labels list to the Mat format OpenCV needs
        // This is the Java equivalent of np.array(labels)
        MatOfInt labelsMat = new MatOfInt();
        labelsMat.fromList(labelsList);

        // Create and train the LBPH Face Recognizer
        // This line requires the opencv_contrib module!
        LBPHFaceRecognizer recognizer = LBPHFaceRecognizer.create();

        // The Java `train` method directly accepts a List<Mat> for faces
        recognizer.train(faces, labelsMat);

        // Save the trained model
        recognizer.save("trained_model.yml");

        // Save the label map (ID -> Name)
        try (PrintWriter pw = new PrintWriter(new FileWriter("labels.txt"))) {
            for (Map.Entry<Integer, String> entry : namesMap.entrySet()) {
                // Same format as your Python script: ID,Name
                pw.println(entry.getKey() + "," + entry.getValue());
            }
        } catch (IOException e) {
            System.out.println("❌ Error writing labels file!");
            e.printStackTrace();
        }

        System.out.println("✅ Training complete! Model saved as 'trained_model.yml'");
        System.out.println("✅ Labels saved as 'labels.txt'");
    }
}