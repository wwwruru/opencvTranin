#include "opencv2/objdetect.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/videoio.hpp"

#include <thread>
#include <bits/stdc++.h>
#include <iostream>
#include <dirent.h>
#include <vector>
#include <string>
#include <mutex>

using namespace std;

cv::CascadeClassifier fasecascad;
mutex mu;
bool progress = true;
struct Pictures
{
    cv::Mat frame;
    string name;
};

vector<Pictures>  pictures;

void Loadpicture (string inpath, vector<Pictures> &pictures, bool &progress)
{
    DIR *path = opendir(inpath.c_str());
    struct dirent *dir;
    if (path)
    {
        while ((dir = readdir(path)) != NULL)
        {
            cv::Mat frame = cv::imread(inpath + "/" + dir->d_name);
            if (frame.data)
            {
                mu.lock();
                pictures.push_back(Pictures{frame, dir->d_name});
                mu.unlock();
            }
        }
    }
    else
    {
        cout << "failed to open directory"<< endl;
    }
    closedir(path);
    progress = false;
}



void DetectAndSave(vector<Pictures> &pictures, string outpath)
{
    while (progress || pictures.size() > 0)
    {
        Pictures pic;
        mu.lock();
        if (pictures.size() > 0)
        {
            pic = pictures.back();
            pictures.pop_back();
         }
         mu.unlock();
         if (pic.frame.data)
         {
            cout <<  pictures.size() << "\t" << this_thread::get_id() << endl;
            cv::Mat frame_gray;
            cv::cvtColor(pic.frame, frame_gray, cv::COLOR_BGR2GRAY);
            cv::equalizeHist(frame_gray, frame_gray);

            vector<cv::Rect> faces;
            fasecascad.detectMultiScale(frame_gray, faces);
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
            "{-i|/home/kostya/Загрузки/dataset| path input }"
            "{-o|/home/kostya/Загрузки/vivod| path output }"
            "{-j|4 | number }");

    string  face_cascade_name = "/usr/share/opencv4/haarcascades/haarcascade_frontalface_alt2.xml";
    if( !fasecascad.load(face_cascade_name) )
    {
        cout << "Error loading face cascade\n";
        return -1;
    };

    vector <thread*> th;

    string inpath = parser.get<string> ("-i");
    string outpath = parser.get<string>("-o");
    int count_thread = parser.get<int>("-j");

    th.push_back(new thread (Loadpicture, inpath, ref(pictures), ref(progress)));

    cout <<  pictures.size() << endl;

    for (int i = 0; i < count_thread;i++)
    {
        th.push_back(new thread (DetectAndSave, ref(pictures), outpath));
    }

    for (thread* thr: th)
        thr->join();
    return 0;
}
