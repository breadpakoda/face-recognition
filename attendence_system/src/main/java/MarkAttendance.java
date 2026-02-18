import org.bytedeco.opencv.opencv_core.*;
import org.bytedeco.opencv.opencv_face.LBPHFaceRecognizer;
import org.bytedeco.opencv.opencv_objdetect.CascadeClassifier;
import org.bytedeco.opencv.opencv_videoio.VideoCapture;
import org.bytedeco.opencv.global.opencv_imgproc;
import org.bytedeco.opencv.global.opencv_highgui;
import org.bytedeco.opencv.global.opencv_imgcodecs;

import java.io.FileWriter;
import java.io.IOException;
import java.sql.*;
import java.time.LocalDateTime;
import java.util.HashMap;
import java.util.Map;
import java.util.Scanner;

public class MarkAttendance {

    private static final String DB_URL = "jdbc:mysql://localhost:3306/fdbas";
    private static final String DB_USER = "root";
    private static final String DB_PASSWORD = "12345";

    public static void main(String[] args) {
        Scanner scanner = new Scanner(System.in);

        //  Fetch available courses
        Map<Integer, String> coursesMap = new HashMap<>();
        try (Connection conn = DriverManager.getConnection(DB_URL, DB_USER, DB_PASSWORD)) {
            Statement stmt = conn.createStatement();
            ResultSet rs = stmt.executeQuery("SELECT course_id, course_name FROM courses");

            System.out.println("Available Courses:");
            while (rs.next()) {
                int courseId = rs.getInt("course_id");
                String courseName = rs.getString("course_name");
                coursesMap.put(courseId, courseName);
                System.out.println(courseId + " -> " + courseName);
            }

            if (coursesMap.isEmpty()) {
                System.out.println(" No courses found. Exiting...");
                return;
            }

        } catch (SQLException e) {
            e.printStackTrace();
            return;
        }

        //  Select course
        System.out.print("Enter the course ID to mark attendance for: ");
        int selectedCourseId = scanner.nextInt();
        scanner.close();

        if (!coursesMap.containsKey(selectedCourseId)) {
            System.out.println(" Invalid course ID. Exiting...");
            return;
        }
        String selectedCourseName = coursesMap.get(selectedCourseId);
        System.out.println(" Attendance will be marked for: " + selectedCourseName);

        //  Load face recognizer and Haar Cascade
        String modelPath = "D:/finall/attendence_system/trained_model.xml";
        LBPHFaceRecognizer recognizer = LBPHFaceRecognizer.create();
        recognizer.read(modelPath);

        String cascadePath = "D:/finall/attendence_system/haarcascade_frontalface_alt.xml";
        CascadeClassifier faceDetector = new CascadeClassifier(cascadePath);

        //  Map labels to student names
        Map<Integer, String> labelToName = new HashMap<>();
        labelToName.put(1, "aditya");
        labelToName.put(2, "abhishek");
        labelToName.put(3, "chandu");
        labelToName.put(4, "Garvit");
        labelToName.put(5, "krish");

        //  Track recognized students for 3-time accuracy
        Map<String, Integer> recognitionCount = new HashMap<>();

        //  Open webcam
        VideoCapture camera = new VideoCapture(0);
        if (!camera.isOpened()) {
            System.out.println(" Cannot open camera");
            return;
        }

        Mat frame = new Mat();
        System.out.println(" Camera started. Attendance will run for 120 seconds...");

        long startTime = System.currentTimeMillis();
        long duration = 120 * 1000; // 120 seconds

        while (System.currentTimeMillis() - startTime < duration) {
            if (!camera.read(frame)) continue;

            Mat gray = new Mat();
            opencv_imgproc.cvtColor(frame, gray, opencv_imgproc.COLOR_BGR2GRAY);

            RectVector faces = new RectVector();
            faceDetector.detectMultiScale(gray, faces);

            for (int i = 0; i < faces.size(); i++) {
                Rect face = faces.get(i);
                Mat faceROI = new Mat(gray, face);

                int[] label = new int[1];
                double[] confidence = new double[1];
                recognizer.predict(faceROI, label, confidence);

                if (confidence[0] < 80) {
                    String studentName = labelToName.get(label[0]);

                    // count recognitions
                    recognitionCount.put(studentName, recognitionCount.getOrDefault(studentName, 0) + 1);

                    if (recognitionCount.get(studentName) == 3) {
                        System.out.println(" Recognized: " + studentName + " (attendance marked)");
                        markAttendance(studentName, selectedCourseId);
                    }
                }
            }
        }

        camera.release();
        opencv_highgui.destroyAllWindows();
        System.out.println("⏹ Attendance session ended.");

        //  Generate CSV
        generateCSV(selectedCourseId);
    }

    private static void markAttendance(String studentName, int courseId) {
        try (Connection conn = DriverManager.getConnection(DB_URL, DB_USER, DB_PASSWORD)) {
            // Get student_id
            PreparedStatement psStudent = conn.prepareStatement(
                    "SELECT student_id FROM students WHERE name = ?");
            psStudent.setString(1, studentName);
            ResultSet rs = psStudent.executeQuery();

            int studentId = -1;
            if (rs.next()) studentId = rs.getInt("student_id");
            else {
                System.out.println(" Student not found in database: " + studentName);
                return;
            }

            // Insert attendance log (avoid duplicates)
            PreparedStatement checkLog = conn.prepareStatement(
                    "SELECT * FROM attendance_log WHERE student_id=? AND course_id=?");
            checkLog.setInt(1, studentId);
            checkLog.setInt(2, courseId);
            ResultSet rsCheck = checkLog.executeQuery();
            if (rsCheck.next()) return; // already marked

            PreparedStatement psLog = conn.prepareStatement(
                    "INSERT INTO attendance_log (student_id, course_id, timestamp) VALUES (?, ?, ?)");
            psLog.setInt(1, studentId);
            psLog.setInt(2, courseId);
            psLog.setTimestamp(3, Timestamp.valueOf(LocalDateTime.now()));

            psLog.executeUpdate();

        } catch (SQLException e) {
            e.printStackTrace();
        }
    }

    private static void generateCSV(int courseId) {
        try (Connection conn = DriverManager.getConnection(DB_URL, DB_USER, DB_PASSWORD);
             FileWriter csvWriter = new FileWriter("attendance_report_course_" + courseId + ".csv")) {

            csvWriter.append("Student Name,Attendance,Date & Time\n");

            Statement stmt = conn.createStatement();
            ResultSet rsStudents = stmt.executeQuery("SELECT name, student_id FROM students");

            while (rsStudents.next()) {
                String name = rsStudents.getString("name");
                int studentId = rsStudents.getInt("student_id");

                PreparedStatement psCheck = conn.prepareStatement(
                        "SELECT timestamp FROM attendance_log WHERE student_id=? AND course_id=?");
                psCheck.setInt(1, studentId);
                psCheck.setInt(2, courseId);
                ResultSet rsCheck = psCheck.executeQuery();

                if (rsCheck.next()) {
                    Timestamp ts = rsCheck.getTimestamp("timestamp");
                    csvWriter.append(name).append(",P,").append(ts.toString()).append("\n");
                } else {
                    csvWriter.append(name).append(",A,\n");
                }
            }

            System.out.println(" CSV generated: attendance_report_course_" + courseId + ".csv");

        } catch (SQLException | IOException e) {
            e.printStackTrace();
        }
    }
}
