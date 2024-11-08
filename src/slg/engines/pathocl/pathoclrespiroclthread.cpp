/***************************************************************************
 * Copyright 1998-2020 by authors (see AUTHORS.txt)                        *
 *                                                                         *
 *   This file is part of LuxCoreRender.                                   *
 *                                                                         *
 * Licensed under the Apache License, Version 2.0 (the "License");         *
 * you may not use this file except in compliance with the License.        *
 * You may obtain a copy of the License at                                 *
 *                                                                         *
 *     http://www.apache.org/licenses/LICENSE-2.0                          *
 *                                                                         *
 * Unless required by applicable law or agreed to in writing, software     *
 * distributed under the License is distributed on an "AS IS" BASIS,       *
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.*
 * See the License for the specific language governing permissions and     *
 * limitations under the License.                                          *
 ***************************************************************************/

#if !defined(LUXRAYS_DISABLE_OPENCL)

#include <boost/lexical_cast.hpp>

#include "luxrays/core/geometry/transform.h"
#include "luxrays/core/randomgen.h"
#include "luxrays/utils/ocl.h"
#include "luxrays/devices/ocldevice.h"
#include "luxrays/kernels/kernels.h"

#include "slg/slg.h"
#include "slg/kernels/kernels.h"
#include "slg/renderconfig.h"
#include "slg/engines/pathocl/pathocl.h"

using namespace std;
using namespace luxrays;
using namespace slg;

//------------------------------------------------------------------------------
// PathOCLRespirOCLRenderThread
//------------------------------------------------------------------------------

PathOCLRespirOCLRenderThread::PathOCLRespirOCLRenderThread(const u_int index, luxrays::HardwareIntersectionDevice *device,
        PathOCLRenderEngine *re)
    : PathOCLOpenCLRenderThread(index, device, re) {

}

PathOCLRespirOCLRenderThread::~PathOCLRespirOCLRenderThread() {

}

void PathOCLRespirOCLRenderThread::StartRenderThread() {
    PathOCLOpenCLRenderThread::StartRenderThread();
}

void PathOCLRespirOCLRenderThread::GetThreadFilmSize(u_int *filmWidth, u_int *filmHeight, u_int *filmSubRegion) {
    PathOCLOpenCLRenderThread::GetThreadFilmSize(filmWidth, filmHeight, filmSubRegion);
}

void PathOCLRespirOCLRenderThread::RenderThreadImpl() {
    PathOCLOpenCLRenderThread::RenderThreadImpl();
}

#endif
