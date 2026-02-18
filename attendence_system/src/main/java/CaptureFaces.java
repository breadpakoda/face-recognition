import org.bytedeco.opencv.opencv_core.*;
import org.bytedeco.opencv.opencv_imgproc.*;
import org.bytedeco.opencv.opencv_objdetect.*;
import org.bytedeco.opencv.opencv_videoio.*;
import org.bytedeco.opencv.global.opencv_imgcodecs;
import org.bytedeco.opencv.global.opencv_imgproc;
import org.bytedeco.opencv.global.opencv_highgui;

import java.io.File;
import java.sql.*;
import java.util.Scanner;

public class CaptureFaces {
    public static void main(String[] args) {
        Scanner scanner = new Scanner(System.in);
        System.out.print("Enter the student's name: ");
        String studentName = scanner.nextLine().trim();
        scanner.close();

        //  Insert student name into database
        int studentId = addStudentToDB(studentName);
        if (studentId == -1) {
            System.err.println(" Could not add student to database. Exiting...");
            return;
        }

        // Create folder to save faces
        String folder = "faces/" + studentName;
        File dir = new File(folder);
        if (!dir.exists()) {
            dir.mkdirs();
            System.out.println(" Created folder: " + folder);
        } else {
            System.out.println(" Folder already exists, adding more images...");
        }

        // Load Haar Cascade for face detection
        String cascadeFile = "haarcascade_frontalface_alt.xml";
        CascadeClassifier faceDetector = new CascadeClassifier(cascadeFile);
        if (faceDetector.empty()) {
            System.err.println(" Error: Haar Cascade file not loaded!");
            return;
        }

        // Open webcam
        VideoCapture camera = new VideoCapture(0);
        if (!camera.isOpened()) {
            System.err.println(" Error: Cannot open camera!");
            return;
        }

        Mat frame = new Mat();
        Mat gray = new Mat();
        int count = 0;
        int target = 70; // Number of images to capture

        System.out.println(" Capturing faces for " + studentName + "... Look at the camera!");
        System.out.println("Press Stop button to stop early.");

        while (count < target) {
            if (!camera.read(frame) || frame.empty()) continue;

            opencv_imgproc.cvtColor(frame, gray, opencv_imgproc.COLOR_BGR2GRAY);

            RectVector faces = new RectVector();
            faceDetector.detectMultiScale(gray, faces);

            for (int i = 0; i < faces.size(); i++) {
                Rect rect = faces.get(i);
                opencv_imgproc.rectangle(frame, rect, new Scalar(0, 255, 0, 0), 2, 8, 0);
                Mat face = new Mat(gray, rect);

                String filename = folder + "/" + (count + 1) + ".jpg";
                boolean ok = opencv_imgcodecs.imwrite(filename, face);
                if (ok) {
                    count++;
                    System.out.println(" Saved: " + filename);
                } else {
                    System.err.println(" Failed to save: " + filename);
                }

                try { Thread.sleep(200); } catch (InterruptedException ignored) {}
                if (count >= target) break;
            }

            opencv_imgproc.putText(frame, "Images captured: " + count + "/" + target,
                    new Point(10, 25), opencv_imgproc.FONT_HERSHEY_SIMPLEX, 0.8,
                    new Scalar(0, 255, 255, 0), 2, opencv_imgproc.LINE_AA, false);

            opencv_highgui.imshow("Face Capture", frame);
            int key = opencv_highgui.waitKey(30);
            if (key == 27) break; // ESC pressed
        }

        camera.release();
        opencv_highgui.destroyAllWindows();
        System.out.println(" Face capture completed for " + studentName + " — total saved: " + count);
    }

    //  Add student to database and return student_id
    private static int addStudentToDB(String studentName) {
        String url = "jdbc:mysql://localhost:3306/fdbas";
        String user = "root";
        String password = "12345";
        int studentId = -1;

        try (Connection conn = DriverManager.getConnection(url, user, password)) {
            // Check if student already exists
            PreparedStatement check = conn.prepareStatement(
                    "SELECT student_id FROM students WHERE name = ?");
            check.setString(1, studentName);
            ResultSet rs = check.executeQuery();

            if (rs.next()) {
                studentId = rs.getInt("student_id");
                System.out.println(" Student already exists with ID: " + studentId);
            } else {
                // Insert new student
                PreparedStatement insert = conn.prepareStatement(
                        "INSERT INTO students (name) VALUES (?)", Statement.RETURN_GENERATED_KEYS);
                insert.setString(1, studentName);
                insert.executeUpdate();
                ResultSet keys = insert.getGeneratedKeys();
                if (keys.next()) studentId = keys.getInt(1);
                System.out.println(" Added student to database with ID: " + studentId);
            }
        } catch (SQLException e) {
            e.printStackTrace();
        }
        return studentId;
    }
}
