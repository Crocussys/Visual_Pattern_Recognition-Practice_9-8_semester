#include <iostream>
#include <opencv2/objdetect.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

using namespace std;
using namespace cv;

int main()
{
    CascadeClassifier face_cascade;
    if (!face_cascade.load(samples::findFile("/home/rozgor/opencv/opencv-4.9.0/data/haarcascades/haarcascade_frontalface_default.xml"))){
        cout << "File Error" << endl;
        return -1;
    }
    CascadeClassifier eye_cascade;
    if (!eye_cascade.load(samples::findFile("/home/rozgor/opencv/opencv-4.9.0/data/haarcascades/haarcascade_eye_tree_eyeglasses.xml"))){
        cout << "File Error" << endl;
        return -1;
    }
    CascadeClassifier smile_cascade;
    if (!smile_cascade.load(samples::findFile("/home/rozgor/opencv/opencv-4.9.0/data/haarcascades/haarcascade_smile.xml"))){
        cout << "File Error" << endl;
        return -1;
    }
    VideoCapture cap("../Visual_Pattern_Recognition-Practice_9-8_semester/ZUA.mp4");
    if(!cap.isOpened()){
        cout << "Error" << endl;
        return -1;
    }

    VideoWriter out("../Visual_Pattern_Recognition-Practice_9-8_semester/output.mp4", cap.get(CAP_PROP_FOURCC), cap.get(CAP_PROP_FPS), Size(cap.get(CAP_PROP_FRAME_WIDTH), cap.get(CAP_PROP_FRAME_HEIGHT)));

    bool start = true;
    Mat frame, new_image, gray_image;
    while(true){
        cap >> frame;
        if(frame.empty()) break;
        GaussianBlur(frame, new_image, Size(0, 0), 3);
        cvtColor(new_image, gray_image, COLOR_BGR2GRAY);
        vector<Rect> faces, eyes, smiles;
        face_cascade.detectMultiScale(gray_image, faces, 1.1, 5);
        eye_cascade.detectMultiScale(gray_image, eyes, 1.1, 5);
        smile_cascade.detectMultiScale(gray_image, smiles, 1.9, 25);
        new_image = frame.clone();
        for (const auto& face: faces){
            rectangle(new_image, face, Scalar(0, 255, 0), 2);
        }
        for (const auto& eye: eyes){
            Point eye_center(eye.x + eye.width / 2, eye.y + eye.height / 2);
            int radius = cvRound((eye.width + eye.height) * 0.25);
            circle(new_image, eye_center, radius, Scalar(255, 0, 0), 2);
        }
        for (const auto& smile: smiles){
            rectangle(new_image, smile, Scalar(0, 0, 255), 2);
        }
        imshow("Faces Detected", new_image);
        out.write(new_image);
        char c = (char) waitKey(30);
        if (c == 27) break;
        if (c == 32 || start){
            while(true){
                char c = (char) waitKey(30);
                if (c == 32) break;
            }
            start = false;
        }
    }
    cap.release();
    out.release();
    destroyAllWindows();
    return 0;
}
