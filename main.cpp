#include "opencv2/objdetect.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/videoio.hpp"

#include <atomic>
#include <thread>
#include <bits/stdc++.h>
#include <iostream>
#include <vector>
#include <string>
#include <mutex>

using namespace std;

mutex mu;

struct Pictures
{
    cv::Mat frame;
    string name;
};


Pictures GetPicture (vector<Pictures> &pictures)
{
    Pictures pic;
    lock_guard<mutex> lg(mu);
    if (pictures.size() > 0)
    {
        pic = pictures.back();
        pictures.pop_back();
        return pic;
     }
    return nullopt;
}


void LoadPicture (string &inpath, vector<Pictures> &pictures, atomic<bool> &progress)
{
    string path = inpath;
    for (const auto & file : filesystem::directory_iterator(path))
    {
        cv::Mat frame = cv::imread(file.path());
        if (frame.data)
        {
            lock_guard<mutex> lg(mu);
            pictures.push_back(Pictures{frame, file.path().filename()});
        }
    }
    progress = false;
}



void DetectAndSave(vector<Pictures> &pictures, string &outpath, atomic<bool> &progress)
{
    cv::CascadeClassifier facecascade;
    string  face_cascade_name = "/usr/share/opencv4/haarcascades/haarcascade_frontalface_alt2.xml";
    if(!facecascade.load(face_cascade_name))
    {
        cout << "Error loading face cascade\n";
    }
    else while (progress || pictures.size() > 0)
    {
        Pictures pic = GetPicture(pictures);
         if (pic.frame.data)
         {
            cout << "number of images : " << pictures.size() << "\tnumber tread : " << this_thread::get_id() << endl;
            cv::Mat frame_gray;
            cv::cvtColor(pic.frame, frame_gray, cv::COLOR_BGR2GRAY);
            cv::equalizeHist(frame_gray, frame_gray);

            vector<cv::Rect> faces;
            facecascade.detectMultiScale(frame_gray, faces);
            for (size_t i = 0; i < faces.size(); i++)
            {
                cv::Point center(faces[i].x + faces[i].width/2, faces[i].y + faces[i].height/2);
                cv::ellipse(pic.frame, center, cv::Size(faces[i].width/2, faces[i].height/2),
                                                        0, 0, 360, cv::Scalar(255, 0, 255), 4);

                cv::imwrite(outpath + "/" + pic.name, pic.frame);
            }
        }
    }
}

int main(int argc, char *argv[])
{
    cv::CommandLineParser parser(argc, argv,
            "{i|/home/kostya/Загрузки/dataset| path input }"
            "{o|/home/kostya/Загрузки/vivod| path output }"
            "{j|4| count thread }");


    atomic<bool> progress = true;
    vector <Pictures> pictures;
    vector <thread*> th;

    string inpath = parser.get<string> ("i");
    string outpath = parser.get<string>("o");
    int count_thread = parser.get<int>("j");

    th.push_back(new thread (LoadPicture, ref(inpath), ref(pictures), ref(progress)));

    cout <<  pictures.size() << endl;

    for (int i = 0; i < count_thread;i++)
    {
        th.push_back(new thread (DetectAndSave, ref(pictures), ref(outpath), ref(progress)));
    }

    while (th.size() > 0)
    {
        th[0]->join();
        th.erase(th.begin());
    }
    return 0;
}
