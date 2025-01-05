#if !defined(LUXRAYS_DISABLE_OPENCL)

#include "luxrays/devices/ocldevice.h"

#include "slg/slg.h"
#include "slg/engines/respirpathocl/respirpathocl.h"

using namespace std;
using namespace luxrays;
using namespace slg;

//------------------------------------------------------------------------------
// RespirPathOCLRenderEngine
//------------------------------------------------------------------------------

PathOCLBaseOCLRenderThread* RespirOCLRenderEngine::CreateOCLThread(const u_int index,
            HardwareIntersectionDevice *device) {
        return new PathOCLRespirOCLRenderThread(index, device, this);
}

#endif