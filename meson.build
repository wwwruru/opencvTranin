project('tutorial', 'cpp', version : '0.1.0')

cv_dep = dependency('opencv4', 'lpthread')
thread_dep = dependency('threads')


executable('facedetected', 'main.cpp',
           dependencies : [cv_dep, thread_dep])