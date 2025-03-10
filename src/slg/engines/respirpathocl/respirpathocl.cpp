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
	spatialRadius = cfg.Get(GetDefaultProps().Get("respirpathocl.spatialreuse.radius")).Get<int>();
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
			cfg.Get(GetDefaultProps().Get("respirpathocl.spatialreuse.radius")) <<
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
            Property("respirpathocl.spatialreuse.numiterations")(1) <<
			Property("respirpathocl.spatialreuse.radius")(1);
	return props;
}

void RespirPathOCLRenderEngine::UpdateTaskCount() {
	const Properties &cfg = renderConfig->cfg;
	if (!cfg.IsDefined("opencl.task.count")) {
		taskCount = film->GetWidth() * film->GetHeight() / intersectionDevices.size();
	} else {
		const u_int defaultTaskCount = 512ull * 1024ull;

		// Compute the cap to the number of tasks
		u_int taskCap = defaultTaskCount;
		BOOST_FOREACH(DeviceDescription *devDesc, selectedDeviceDescs) {
			if (devDesc->GetMaxMemory() <= 8ull* 1024ull * 1024ull * 1024ull) // For 8GB cards
				taskCap = Min(taskCap, 256u * 1024u);
			if (devDesc->GetMaxMemory() <= 4ull * 1024ull * 1024ull * 1024ull) // For 4GB cards
				taskCap = Min(taskCap, 128u * 1024u);
			if (devDesc->GetMaxMemory() <= 2ull * 1024ull * 1024ull * 1024ull) // For 2GB cards
				taskCap = Min(taskCap, 64u * 1024u);
		}

		if (cfg.Get(Property("opencl.task.count")("AUTO")).Get<string>() == "AUTO")
			taskCount = taskCap;
		else
			taskCount = cfg.Get(Property("opencl.task.count")(taskCap)).Get<u_int>();
	}

	// I don't know yet the workgroup size of each device so I can not
	// round up task count to be a multiple of workgroups size of all devices
	// used. Rounding to 8192 is a simple trick based on the assumption that
	// workgroup size is a power of 2 and <= 8192.
	taskCount = RoundUp<u_int>(taskCount, 8192);
	if(GetType() != RTPATHOCL)
		SLG_LOG("[PathOCLRenderEngine] OpenCL task count: " << taskCount);
}

#endif