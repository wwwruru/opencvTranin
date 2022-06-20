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

using namespace std;

cv::CascadeClassifier fasecascad;

void LoadPictureName (string inpath, vector<vector<string>*> &name)
{
    DIR *path = opendir(inpath.c_str());
    struct dirent *dir;
    if (path)
    {
        int i = 0;
        while ((dir = readdir(path)) != NULL)
        {
            name[i]->push_back(dir->d_name);
            i++;
            if (i == name.size())
                i = 0;
        }
    }
    else
    {
        cout << "faund path"<< endl;
    }
    closedir(path);
}



void DetectAndSave(string inpath, vector<string> *namefile, string outpath)
{
    for (string name : *namefile)
    {
        cout << name << endl;
        cv::Mat frame = cv::imread(inpath + "/" + name );
        if (frame.data)
        {
            cv::Mat frame_gray;
            cv::cvtColor( frame, frame_gray, cv::COLOR_BGR2GRAY );
            cv::equalizeHist( frame_gray, frame_gray );

            vector<cv::Rect> faces;
            fasecascad.detectMultiScale(frame_gray, faces);
            for ( size_t i = 0; i < faces.size(); i++ )
            {
                cv::Point center(faces[i].x + faces[i].width/2, faces[i].y + faces[i].height/2);
                cv::ellipse( frame, center, cv::Size(faces[i].width/2, faces[i].height/2),
                                                      0, 0, 360, cv::Scalar(255, 0, 255), 4);

                cv::imwrite(outpath + "/" + name, frame);
            }
        }
    }
}

int main(int argc, char *argv[])
{
    string  face_cascade_name = "/usr/share/opencv4/haarcascades/haarcascade_frontalface_alt2.xml";
    if( !fasecascad.load(face_cascade_name) )
    {
        cout << "Error loading face cascade\n";
        return -1;
    };

    vector <thread*> th;

    string inpath = "/home/kostya/Загрузки/dataset";
    string outpath = "/home/kostya/Загрузки/vivod";
    int count_thread = 4;

    if(argc!=4) {
            cout << "input path"<<"\n";
            cin >> inpath;
            cout << "output path"<<"\n";
            cin >> outpath;
            cout << "count thread"<<"\n";
            cin >> count_thread;
        }else
        {
            inpath =  argv[1];
            outpath =  argv[2];
            count_thread = atoi(argv[3]);
        }

    vector<vector<string>*> namefile;
    for (int i = 0; i < count_thread; i++)
        namefile.push_back(new vector<string>);
    LoadPictureName(inpath, namefile);

    for (int i = 0; i < count_thread; i++)
    {
        th.push_back(new thread (DetectAndSave, inpath, namefile[i], outpath));
    }

    for (thread* thr: th)
        thr->join();
    return 0;
}
