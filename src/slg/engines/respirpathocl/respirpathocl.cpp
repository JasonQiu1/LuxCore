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

RespirPathOCLRenderEngine::RespirPathOCLRenderEngine(const RenderConfig *rcfg) :
		PathOCLRenderEngine(rcfg) {
    const Properties &cfg = renderConfig->cfg;
    numSpatialReuseIterations = cfg.Get(GetDefaultProps().Get("respirpathocl.spatialreuse.numiterations")).Get<int>();
}

RespirPathOCLRenderEngine::~RespirPathOCLRenderEngine() {
}

PathOCLBaseOCLRenderThread* RespirPathOCLRenderEngine::CreateOCLThread(const u_int index,
            HardwareIntersectionDevice *device) {
    return new RespirPathOCLRenderThread(index, device, this);
}

//------------------------------------------------------------------------------
// Static methods used by RenderEngineRegistry
//------------------------------------------------------------------------------

Properties RespirPathOCLRenderEngine::ToProperties(const Properties &cfg) {
	Properties props;

	props <<
			OCLRenderEngine::ToProperties(cfg) <<
			cfg.Get(GetDefaultProps().Get("renderengine.type")) <<
			PathTracer::ToProperties(cfg) <<
			cfg.Get(GetDefaultProps().Get("pathocl.pixelatomics.enable")) <<
			cfg.Get(GetDefaultProps().Get("opencl.task.count")) <<
            cfg.Get(GetDefaultProps().Get("respirpathocl.spatialreuse.numiterations")) <<
			Sampler::ToProperties(cfg);
	return props;
}

RenderEngine* RespirPathOCLRenderEngine::FromProperties(const RenderConfig *rcfg) {
	return new RespirPathOCLRenderEngine(rcfg);
}

const Properties &RespirPathOCLRenderEngine::GetDefaultProps() {
	static Properties props = Properties() <<
			OCLRenderEngine::GetDefaultProps() <<
			Property("renderengine.type")(GetObjectTag()) <<
			PathTracer::GetDefaultProps() <<
			Property("pathocl.pixelatomics.enable")(true) <<
			Property("opencl.task.count")("AUTO") <<
            Property("respirpathocl.spatialreuse.numiterations")(1);
	return props;
}

#endif