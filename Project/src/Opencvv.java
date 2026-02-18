import org.opencv.core.*;
import org.opencv.highgui.HighGui;
import org.opencv.imgproc.Imgproc;
import org.opencv.objdetect.CascadeClassifier;
import org.opencv.videoio.VideoCapture;

public class Opencvv {
    public static void main(String[] args) {
        // Load OpenCV native library
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);

        // Load the Haar Cascade face detection model
        CascadeClassifier faceDetector = new CascadeClassifier("src/haarcascade_frontalface_alt.xml");

        // Check if the classifier is loaded correctly
        if (faceDetector.empty()) {
            System.out.println("Error: Could not load Haar Cascade file!");
            return;
        }

        // Open the default camera (0)
        VideoCapture camera = new VideoCapture(0);
        if (!camera.isOpened()) {
            System.out.println("Error: Cannot open camera!");
            return;
        }

        Mat frame = new Mat();

        System.out.println("Press ESC to exit...");

        // Start capturing frames
        while (true) {
            if (camera.read(frame)) {
                // Convert the frame to grayscale (improves detection speed and accuracy)
                Mat gray = new Mat();
                Imgproc.cvtColor(frame, gray, Imgproc.COLOR_BGR2GRAY);

                // Detect faces in the frame
                MatOfRect faces = new MatOfRect();
                faceDetector.detectMultiScale(gray, faces);

                // Draw rectangles around detected faces
                Rect[] facesArray = faces.toArray();
                for (Rect rect : facesArray) {
                    Imgproc.rectangle(frame,
                            new Point(rect.x, rect.y),
                            new Point(rect.x + rect.width, rect.y + rect.height),
                            new Scalar(0, 255, 0), 2);
                }

                // Show number of faces detected
                Imgproc.putText(frame, "Faces detected: " + facesArray.length,
                        new Point(10, 25), Imgproc.FONT_HERSHEY_SIMPLEX, 0.8,
                        new Scalar(0, 255, 255), 2);

                // Display the video feed with rectangles
                HighGui.imshow("Face Detection", frame);

                // Exit on ESC key
                if (HighGui.waitKey(30) == 27) { // 27 = ESC key
                    System.out.println("Exiting...");
                    break;
                }
            } else {
                System.out.println("No frame captured!");
                break;
            }
        }

        // Release camera and close all OpenCV windows
        camera.release();
        HighGui.destroyAllWindows();
    }
}
