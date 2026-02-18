import org.opencv.core.*;
import org.opencv.highgui.HighGui;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import org.opencv.objdetect.CascadeClassifier;
import org.opencv.videoio.VideoCapture;

import java.io.File;
import java.util.Scanner;

public class CaptureFaces {
    public static void main(String[] args) {
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);

        // Ask user for person name
        Scanner scanner = new Scanner(System.in);
        System.out.print("Enter the person's name: ");
        String personName = scanner.nextLine().trim();
        scanner.close();

        // Create folder to save faces
        String folder = "faces/" + personName;
        File dir = new File(folder);
        if (!dir.exists()) {
            dir.mkdirs();
            System.out.println("✅ Created folder: " + folder);
        } else {
            System.out.println("ℹ️ Folder already exists, adding more images...");
        }

        // Load Haar Cascade
        CascadeClassifier faceDetector = new CascadeClassifier("src/haarcascade_frontalface_alt.xml");
        if (faceDetector.empty()) {
            System.out.println("❌ Error: Haar Cascade file not loaded!");
            return;
        }

        // Open webcam
        VideoCapture camera = new VideoCapture(0);
        if (!camera.isOpened()) {
            System.out.println("❌ Error: Cannot open camera!");
            return;
        }

        Mat frame = new Mat();
        int count = 0;
        int target = 70; // Number of images to capture

        System.out.println("📸 Capturing faces for " + personName + "... Look at the camera!");
        System.out.println("Press ESC to stop early.");

        while (count < target) {
            if (camera.read(frame)) {
                Mat gray = new Mat();
                Imgproc.cvtColor(frame, gray, Imgproc.COLOR_BGR2GRAY);

                MatOfRect faces = new MatOfRect();
                faceDetector.detectMultiScale(gray, faces);

                for (Rect rect : faces.toArray()) {
                    // Draw rectangle
                    Imgproc.rectangle(frame, rect, new Scalar(0, 255, 0), 2);

                    // Crop the face
                    Mat face = new Mat(gray, rect);

                    // Save the image
                    String filename = folder + "/" + (count + 1) + ".jpg";
                    Imgcodecs.imwrite(filename, face);
                    count++;

                    System.out.println("💾 Saved: " + filename);

                    // Wait a bit to avoid duplicates
                    try { Thread.sleep(200); } catch (InterruptedException e) { e.printStackTrace(); }

                    // Stop if we’ve reached the target
                    if (count >= target) break;
                }

                Imgproc.putText(frame, "Images captured: " + count + "/" + target,
                        new Point(10, 25), Imgproc.FONT_HERSHEY_SIMPLEX, 0.8,
                        new Scalar(0, 255, 255), 2);

                HighGui.imshow("Face Capture", frame);
                if (HighGui.waitKey(30) == 27) break; // ESC key
            }
        }

        camera.release();
        HighGui.destroyAllWindows();
        System.out.println("✅ Face capture completed for " + personName);
    }
}
