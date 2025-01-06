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
}

RespirPathOCLRenderEngine::~RespirPathOCLRenderEngine()
{
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
			Sampler::ToProperties(cfg) <<
			PhotonGICache::ToProperties(cfg);

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
			PhotonGICache::GetDefaultProps();

	return props;
}

#endif