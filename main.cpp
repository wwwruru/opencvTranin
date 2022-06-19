#include "opencv2/objdetect.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/videoio.hpp"

#include <thread>
#include <bits/stdc++.h>
#include <iostream>
#include <dirent.h>
#include <list>
#include <string>

using namespace std;

cv::CascadeClassifier fasecascad;

list<string> picture (string inpath)
{
    DIR *path = opendir(inpath.c_str());
    list<string> name;
    struct dirent *dir;
    if (path)
    {
        while ((dir = readdir(path)) != NULL)
        {
            name.push_front(dir->d_name);
        }
    }
    else
    {
        cout << "faund path"<< endl;
    }
    closedir(path);
    return name;
}



void detectAndSave(string inpath, list<string> namefile, string outpath)
{
    for (string name : namefile)
    {
        cout << name << endl;
        cv::Mat frame = cv::imread(inpath + "/" + name );
        if (frame.data)
        {
            cv::Mat frame_gray;
            cv::cvtColor( frame, frame_gray, cv::COLOR_BGR2GRAY );
            cv::equalizeHist( frame_gray, frame_gray );

            std::vector<cv::Rect> faces;
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

    list <thread*> th;

    string inpath = "/home/kostya/Загрузки/dataset";
    string outpath = "/home/kostya/Загрузки/vivod";
    int count_thread = 10;

    /*if(argc!=4) {
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
        }*/

    list<string> namefile = picture(inpath);
    if (namefile.size() > 0){
        int minsize = namefile.size() / count_thread;
        for (int i = 0; i < count_thread; i++)
        {
            if (i < count_thread - 1){
                auto it =  namefile.begin();
                advance(it, minsize * i);
                auto it2 = namefile.begin();
                advance(it2, minsize * (i + 1 ) - 1);
                list<string> namefile2;
                namefile2.assign(it, it2);
                th.push_front(new thread (detectAndSave, inpath, namefile2, outpath));
            }
            else{
                auto it =  namefile.begin();
                advance(it, minsize * i);
                list<string> namefile2;
                namefile2.assign(it, namefile.end());
                th.push_front(new thread (detectAndSave, inpath, namefile2, outpath));
            }
        }
    } else cout << "file miss" << endl;

    for (thread* thr: th)
        thr->join();
    return 0;
}
