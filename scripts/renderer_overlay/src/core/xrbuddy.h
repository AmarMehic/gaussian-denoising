/*
    Headless overlay: no-op stub of XrBuddy that satisfies all call sites in
    app.cpp without pulling in OpenXR or X11. When the renderer is launched
    without the --openxr flag, none of these methods are reached at runtime;
    they only exist to make the original VR code paths compile.

    Original (full OpenXR implementation) lives in the upstream
    stochasticsplats repo at src/core/xrbuddy.{h,cpp}.
*/

#pragma once

#include <array>
#include <cstdint>
#include <functional>
#include <map>
#include <string>
#include <vector>

#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>

#ifndef __ANDROID__
#include <GL/glew.h>
#else
#include <GLES3/gl3.h>
#include <GLES3/gl3ext.h>
#endif

#include "maincontext.h"

struct SuperSampleBuffers
{
    GLuint framebuffer = 0;
    GLuint colorTexture = 0;
    GLuint depthTexture = 0;
    GLuint resolveFramebuffer = 0;
    GLsizei targetWidth = 0;
    GLsizei targetHeight = 0;
    GLsizei superWidth = 0;
    GLsizei superHeight = 0;
};

class XrBuddy
{
public:
    using RenderCallback = std::function<void(const glm::mat4& projMat,
                                              const glm::mat4& eyeMat,
                                              const glm::vec4& viewport,
                                              const glm::vec2& nearFar,
                                              int32_t viewNum)>;

    XrBuddy(MainContext& /*mainContextIn*/,
            const glm::vec2& /*nearFarIn*/,
            int /*sampleCountIn*/) {}

    bool Init() { return false; }
    bool PollEvents() { return false; }
    bool SyncInput() { return false; }
    bool SessionReady() const { return false; }
    bool RenderFrame() { return false; }
    bool Shutdown() { return true; }

    void SetRenderCallback(RenderCallback) {}

    bool GetActionBool(const std::string&, bool*, bool*, bool*) const { return false; }
    bool GetActionFloat(const std::string&, float*, bool*, bool*) const { return false; }
    bool GetActionVec2(const std::string&, glm::vec2*, bool*, bool*) const { return false; }
    bool GetActionPosition(const std::string&, glm::vec3*, bool*, bool*) const { return false; }
    bool GetActionOrientation(const std::string&, glm::quat*, bool*, bool*) const { return false; }
    bool GetActionLinearVelocity(const std::string&, glm::vec3*, bool*) const { return false; }
    bool GetActionAngularVelocity(const std::string&, glm::vec3*, bool*) const { return false; }

    uint32_t GetColorTexture() const { return 0; }

    void CycleColorSpace() {}
};
