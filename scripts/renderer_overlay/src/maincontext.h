/*
    Copyright (c) 2024 Anthony J. Thibault
    This software is licensed under the MIT License. See LICENSE for more details.

    Headless overlay: removes the X11/GLX-dependent fields from the Linux
    MainContext so the renderer builds on HPC nodes without X11 dev headers.
    Original lives in the upstream stochasticsplats repo.
*/

#pragma once

#if defined(__ANDROID__)
#include <EGL/egl.h>
#include <EGL/eglext.h>
#include <GLES3/gl3.h>
#include <GLES3/gl3ext.h>
#include <jni.h>
#include <android_native_app_glue.h>
#elif defined(__linux__)
#include <GL/glew.h>
#endif

#if defined(__ANDROID__)
    struct MainContext
    {
        EGLDisplay display;
        EGLConfig config;
        EGLContext context;
        android_app* androidApp;
    };
#else
    struct MainContext
    {
    };
#endif
