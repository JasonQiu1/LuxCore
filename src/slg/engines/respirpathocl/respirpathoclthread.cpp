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
#include "slg/engines/respirpathocl/respirpathocl.h"

using namespace std;
using namespace luxrays;
using namespace slg;

//------------------------------------------------------------------------------
// RespirPathOCLRenderThread
//------------------------------------------------------------------------------

RespirPathOCLRenderThread::RespirPathOCLRenderThread(const u_int index, luxrays::HardwareIntersectionDevice *device,
        RespirPathOCLRenderEngine *re)
    : PathOCLOpenCLRenderThread(index, device, re), 
	  spatialRadius(re->spatialRadius),
	  numSpatialNeighbors(re->numSpatialNeighbors)
{
	pixelIndexMapBuff = nullptr;
	spatialReuseDataBuff = nullptr;
	shiftInOutDataBuff = nullptr;

    spatialReuseKernel_MK_INIT = nullptr;
	spatialReuseKernel_MK_NEXT_NEIGHBOR = nullptr;
	spatialReuseKernel_MK_SHIFT = nullptr;
	spatialReuseKernel_MK_CHECK_VISIBILITY = nullptr;
	spatialReuseKernel_MK_RESAMPLE = nullptr;
	spatialReuseKernel_MK_FINISH_RESAMPLE = nullptr;
	spatialReuseKernel_MK_FINISH_ITERATION = nullptr;
	spatialReuseKernel_MK_FINISH_REUSE = nullptr;
	spatialReuseKernel_MK_SET_SPLAT = nullptr;
}

RespirPathOCLRenderThread::~RespirPathOCLRenderThread() {
    delete spatialReuseKernel_MK_INIT;
	delete spatialReuseKernel_MK_NEXT_NEIGHBOR;
	delete spatialReuseKernel_MK_SHIFT;
	delete spatialReuseKernel_MK_CHECK_VISIBILITY;
	delete spatialReuseKernel_MK_RESAMPLE;
	delete spatialReuseKernel_MK_FINISH_RESAMPLE;
	delete spatialReuseKernel_MK_FINISH_ITERATION;
	delete spatialReuseKernel_MK_FINISH_REUSE;
	delete spatialReuseKernel_MK_SET_SPLAT;
}

void RespirPathOCLRenderThread::Stop() {
	StopRenderThread();

	// Transfer the films
	TransferThreadFilms(intersectionDevice);
	FreeThreadFilmsOCLBuffers();

	// Scene buffers
	intersectionDevice->FreeBuffer(&materialsBuff);
	intersectionDevice->FreeBuffer(&materialEvalOpsBuff);
	intersectionDevice->FreeBuffer(&materialEvalStackBuff);
	intersectionDevice->FreeBuffer(&texturesBuff);
	intersectionDevice->FreeBuffer(&textureEvalOpsBuff);
	intersectionDevice->FreeBuffer(&textureEvalStackBuff);
	intersectionDevice->FreeBuffer(&meshDescsBuff);
	intersectionDevice->FreeBuffer(&scnObjsBuff);
	intersectionDevice->FreeBuffer(&normalsBuff);
	intersectionDevice->FreeBuffer(&triNormalsBuff);
	intersectionDevice->FreeBuffer(&uvsBuff);
	intersectionDevice->FreeBuffer(&colsBuff);
	intersectionDevice->FreeBuffer(&alphasBuff);
	intersectionDevice->FreeBuffer(&triAOVBuff);
	intersectionDevice->FreeBuffer(&trianglesBuff);
	intersectionDevice->FreeBuffer(&interpolatedTransformsBuff);
	intersectionDevice->FreeBuffer(&vertsBuff);
	intersectionDevice->FreeBuffer(&lightsBuff);
	intersectionDevice->FreeBuffer(&envLightIndicesBuff);
	intersectionDevice->FreeBuffer(&lightsDistributionBuff);
	intersectionDevice->FreeBuffer(&infiniteLightSourcesDistributionBuff);
	intersectionDevice->FreeBuffer(&dlscAllEntriesBuff);
	intersectionDevice->FreeBuffer(&dlscDistributionsBuff);
	intersectionDevice->FreeBuffer(&dlscBVHNodesBuff);
	intersectionDevice->FreeBuffer(&elvcAllEntriesBuff);
	intersectionDevice->FreeBuffer(&elvcDistributionsBuff);
	intersectionDevice->FreeBuffer(&elvcTileDistributionOffsetsBuff);
	intersectionDevice->FreeBuffer(&elvcBVHNodesBuff);
	intersectionDevice->FreeBuffer(&envLightDistributionsBuff);
	intersectionDevice->FreeBuffer(&cameraBuff);
	intersectionDevice->FreeBuffer(&cameraBokehDistributionBuff);
	intersectionDevice->FreeBuffer(&lightIndexOffsetByMeshIndexBuff);
	intersectionDevice->FreeBuffer(&lightIndexByTriIndexBuff);
	intersectionDevice->FreeBuffer(&imageMapDescsBuff);

	for (u_int i = 0; i < imageMapsBuff.size(); ++i)
		intersectionDevice->FreeBuffer(&imageMapsBuff[i]);
	imageMapsBuff.resize(0);
	intersectionDevice->FreeBuffer(&pgicRadiancePhotonsBuff);
	intersectionDevice->FreeBuffer(&pgicRadiancePhotonsValuesBuff);
	intersectionDevice->FreeBuffer(&pgicRadiancePhotonsBVHNodesBuff);
	intersectionDevice->FreeBuffer(&pgicCausticPhotonsBuff);
	intersectionDevice->FreeBuffer(&pgicCausticPhotonsBVHNodesBuff);

	// OpenCL task related buffers
	intersectionDevice->FreeBuffer(&pathStatesBuff);
	intersectionDevice->FreeBuffer(&raysBuff);
	intersectionDevice->FreeBuffer(&hitsBuff);
	intersectionDevice->FreeBuffer(&taskConfigBuff);
	intersectionDevice->FreeBuffer(&tasksBuff);
	intersectionDevice->FreeBuffer(&tasksDirectLightBuff);
	intersectionDevice->FreeBuffer(&tasksStateBuff);
	intersectionDevice->FreeBuffer(&samplerSharedDataBuff);
	intersectionDevice->FreeBuffer(&samplesBuff);
	intersectionDevice->FreeBuffer(&sampleDataBuff);
	intersectionDevice->FreeBuffer(&sampleResultsBuff);
	intersectionDevice->FreeBuffer(&taskStatsBuff);
	intersectionDevice->FreeBuffer(&eyePathInfosBuff);
	intersectionDevice->FreeBuffer(&directLightVolInfosBuff);
	intersectionDevice->FreeBuffer(&pixelFilterBuff);

	// Respir buffers
	intersectionDevice->FreeBuffer(&pixelIndexMapBuff);
	intersectionDevice->FreeBuffer(&spatialReuseDataBuff);
	intersectionDevice->FreeBuffer(&shiftInOutDataBuff);

	started = false;

	// Film is deleted in the destructor to allow image saving after
	// the rendering is finished
}

void RespirPathOCLRenderThread::StartRenderThread() {
    PathOCLOpenCLRenderThread::StartRenderThread();
}

void RespirPathOCLRenderThread::GetThreadFilmSize(u_int *filmWidth, u_int *filmHeight, u_int *filmSubRegion) {
    PathOCLOpenCLRenderThread::GetThreadFilmSize(filmWidth, filmHeight, filmSubRegion);
}

// Check that all path states are equal to the target
bool RespirPathOCLRenderThread::CheckSyncedPathStates(slg::ocl::respir::RespirAsyncState* pathStatesReadBuffer, 
		const u_int taskCount, slg::ocl::respir::RespirAsyncState targetState) {
	intersectionDevice->EnqueueReadBuffer(pathStatesBuff, true,
		sizeof(slg::ocl::respir::RespirAsyncState) * taskCount,
		pathStatesReadBuffer);

	for (u_int i = 0; i < taskCount; i++) {
		//SLG_LOG("[PathOCLRespirOCLRenderThread::" << threadIndex << "] TaskState(" << i << ") PathState=" << tasksState[i].state);
		if (pathStatesReadBuffer[i] != targetState) {
			SLG_LOG("[PathOCLRespirOCLRenderThread::" << threadIndex << "] TaskState(" << i << ") PathState=" << pathStatesReadBuffer[i] << " Not synced.");
			return false;
		}
	}
	return true;
}

void RespirPathOCLRenderThread::RenderThreadImpl() {
    SLG_LOG("[PathOCLRespirOCLRenderThread::" << threadIndex << "] Rendering thread started");

	RespirPathOCLRenderEngine *engine = (RespirPathOCLRenderEngine *)renderEngine;
	const u_int taskCount = engine->taskCount;

	intersectionDevice->PushThreadCurrentDevice();

    // Keep track of rendered samples per pixel for this thread.
    // TODO: remove, since this should already be tracked in the film class I think
    u_int numFrames = 0;
	const u_int haltSpp = engine->film->GetHaltSpp();

	try {
		//----------------------------------------------------------------------
		// Execute initialization kernels
		//----------------------------------------------------------------------

		// Clear the frame buffer
		const u_int filmPixelCount = threadFilms[0]->film->GetWidth() * threadFilms[0]->film->GetHeight();
		intersectionDevice->EnqueueKernel(filmClearKernel,
			HardwareDeviceRange(RoundUp<u_int>(filmPixelCount, filmClearWorkGroupSize)),
			HardwareDeviceRange(filmClearWorkGroupSize));

		// Initialize random number generator seeds
		intersectionDevice->EnqueueKernel(initSeedKernel,
				HardwareDeviceRange(taskCount), HardwareDeviceRange(initWorkGroupSize));

		// Initialize the tasks buffer
		intersectionDevice->EnqueueKernel(initKernel,
				HardwareDeviceRange(taskCount), HardwareDeviceRange(initWorkGroupSize));

		// Check if I have to load the start film
		if (engine->hasStartFilm && (threadIndex == 0))
			threadFilms[0]->SendFilm(intersectionDevice);

		ocl::respir::RespirAsyncState* pathStatesReadBuffer = (ocl::respir::RespirAsyncState*)malloc(sizeof(*pathStatesReadBuffer) * taskCount);

		//----------------------------------------------------------------------
		// Rendering loop
		//----------------------------------------------------------------------

		// The film refresh time target
		const double targetTime = 0.2; // 200ms

        const u_int numSpatialReuseIterations = engine->numSpatialReuseIterations;

		u_int iterations = 1;

		double totalTransferTime = 0.0;
		double totalKernelTime = 0.0;

		while (!boost::this_thread::interruption_requested()) {
			//if (threadIndex == 0)
			//	SLG_LOG("[DEBUG] =================================");

			// Check if we are in pause mode
			if (engine->pauseMode) {
				// Check every 100ms if I have to continue the rendering
				while (!boost::this_thread::interruption_requested() && engine->pauseMode)
					boost::this_thread::sleep(boost::posix_time::millisec(100));

				if (boost::this_thread::interruption_requested())
					break;
			}

			//------------------------------------------------------------------
			// This is required for updating film denoiser parameter
			if (threadFilms[0]->film->GetDenoiser().IsEnabled()) {
				boost::unique_lock<boost::mutex> lock(engine->setKernelArgsMutex);
				SetAllAdvancePathsKernelArgs(0);
			}

            SLG_LOG("[PathOCLRespirOCLRenderThread::" << threadIndex << "] Beginning rendering for frame " << numFrames << ".");

			const double timeKernelStart = WallClockTime();

			// Get next sample if this is not the first iteration of this loop.
	        intersectionDevice->EnqueueKernel(advancePathsKernel_MK_NEXT_SAMPLE,
			    HardwareDeviceRange(taskCount), HardwareDeviceRange(advancePathsWorkGroupSize));

            // Generate camera rays for each pixel in this frame.
            intersectionDevice->EnqueueKernel(advancePathsKernel_MK_GENERATE_CAMERA_RAY,
			    HardwareDeviceRange(taskCount), HardwareDeviceRange(advancePathsWorkGroupSize));

            // Perform initial path resampling to get canonical samples for each pixel this frame.
            bool isInitialPathResamplingDone = false;
			bool firstLoop = false;
            u_int totalIterationsThisFrame = 0;

			SLG_LOG("[PathOCLRespirOCLRenderThread::" << threadIndex << "] Generating canonical initial path samples: " << taskCount);
			while (!isInitialPathResamplingDone) {
				SLG_LOG("[PathOCLRespirOCLRenderThread::" << threadIndex << "] Queuing advance paths kernels for " << iterations << " iterations");

                // Trace until all paths are completed for this frame.
                for (u_int i = 0; i < iterations; ++i) {
                    // Trace rays
                    intersectionDevice->EnqueueTraceRayBuffer(raysBuff, hitsBuff, taskCount);

                    // Advance to next path state
                    EnqueueAdvancePathsKernel();
                }

				//SLG_LOG("[PathOCLRespirOCLRenderThread::" << threadIndex << "] Finished queuing advance paths kernels");

				//SLG_LOG("[PathOCLRespirOCLRenderThread::" << threadIndex << "] All advance paths kernels finished execution");

                // Check if initial path resampling for all pixels is complete
				// This is blocking and waits for queue to finish
				intersectionDevice->FinishQueue();
                isInitialPathResamplingDone = CheckSyncedPathStates(pathStatesReadBuffer, taskCount, ocl::respir::RespirAsyncState::SYNC);

                totalIterationsThisFrame += iterations;

				if (isInitialPathResamplingDone) {
					break;
				}
				firstLoop = false;
				iterations++;
            }

			SLG_LOG("[PathOCLRespirOCLRenderThread::" << threadIndex << "] Number of iterations for initial path sampling: " << totalIterationsThisFrame);
            
            // Perform spatial reuse.
			if (numSpatialReuseIterations > 0) {
				SLG_LOG("[PathOCLRespirOCLRenderThread::" << threadIndex << "] Spatial reuse is enabled, performing " << numSpatialReuseIterations << " iterations");
				// Initialize variables and find spatial neighbors
				intersectionDevice->EnqueueKernel(spatialReuseKernel_MK_INIT,
						HardwareDeviceRange(taskCount), HardwareDeviceRange(spatialReuseWorkGroupSize));
				// Ensure all paths are synced before continuing
				intersectionDevice->FinishQueue();
			} else {
				SLG_LOG("[PathOCLRespirOCLRenderThread::" << threadIndex << "] Spatial reuse is disabled, configure with respirpathocl.spatialreuse.numiterations");
			}

			// Iterate x times
			for (u_int i = 0; i < numSpatialReuseIterations; i++) {
				SLG_LOG("[PathOCLRespirOCLRenderThread::" << threadIndex << "] Beginning spatial reuse iteration " << i);
				bool isSpatialReuseDone = false;
				u_int visibilityIterations = 0;
				// Resample neighboring pixels (ASYNC WORK-ITEMS)
				while (!isSpatialReuseDone) {
					SLG_LOG("[PathOCLRespirOCLRenderThread::" << threadIndex << "] Spatial reuse (iteration " << i << "): tracing shadow paths");
					for (int i = 0; i < totalIterationsThisFrame; i++) {
						// Find next neighboring pixel
						intersectionDevice->EnqueueKernel(spatialReuseKernel_MK_NEXT_NEIGHBOR,
							HardwareDeviceRange(taskCount), HardwareDeviceRange(spatialReuseAsyncWorkGroupSize));
						
						// Shift from/to neighbor
						intersectionDevice->EnqueueKernel(spatialReuseKernel_MK_SHIFT,
							HardwareDeviceRange(taskCount), HardwareDeviceRange(spatialReuseAsyncWorkGroupSize));
						// Trace shadow rays to reconnection vertices
						intersectionDevice->EnqueueTraceRayBuffer(raysBuff, hitsBuff, taskCount);
						// Check visibility and update reservoirs if successful
						intersectionDevice->EnqueueKernel(spatialReuseKernel_MK_CHECK_VISIBILITY,
							HardwareDeviceRange(taskCount), HardwareDeviceRange(spatialReuseAsyncWorkGroupSize));
							
						// Use shifted result for canonical pairwise MIS calculation and resample for actual replacement
						intersectionDevice->EnqueueKernel(spatialReuseKernel_MK_RESAMPLE,
							HardwareDeviceRange(taskCount), HardwareDeviceRange(spatialReuseAsyncWorkGroupSize));
						// Finish resampling neighbor into current reservoir
						intersectionDevice->EnqueueKernel(spatialReuseKernel_MK_FINISH_RESAMPLE,
							HardwareDeviceRange(taskCount), HardwareDeviceRange(spatialReuseAsyncWorkGroupSize));
					}
					// This is blocking and waits for queue to finish
					intersectionDevice->FinishQueue();
					visibilityIterations += totalIterationsThisFrame;
					isSpatialReuseDone = CheckSyncedPathStates(pathStatesReadBuffer, taskCount, ocl::respir::RespirAsyncState::SYNC);
				}

				SLG_LOG("[PathOCLRespirOCLRenderThread::" << threadIndex << "] Number of iterations for visibility checks: " << visibilityIterations);
				
				// Finish the iteration
				intersectionDevice->EnqueueKernel(spatialReuseKernel_MK_FINISH_ITERATION,
					HardwareDeviceRange(taskCount), HardwareDeviceRange(spatialReuseWorkGroupSize));

				// Ensure all paths are synced before continuing
				intersectionDevice->FinishQueue();
			}
			
			// Finish up reuse
			if (numSpatialReuseIterations > 0) {
				SLG_LOG("[PathOCLRespirOCLRenderThread::" << threadIndex << "] Spatial reuse passes are complete, finishing reuse.");
				intersectionDevice->EnqueueKernel(spatialReuseKernel_MK_FINISH_REUSE,
                    HardwareDeviceRange(taskCount), HardwareDeviceRange(spatialReuseWorkGroupSize));
			}

            // Splat pixels.
			SLG_LOG("[PathOCLRespirOCLRenderThread::" << threadIndex << "] Splatting pixels.");
			intersectionDevice->EnqueueKernel(spatialReuseKernel_MK_SET_SPLAT,
                    HardwareDeviceRange(taskCount), HardwareDeviceRange(spatialReuseWorkGroupSize));
            intersectionDevice->EnqueueKernel(advancePathsKernel_MK_SPLAT_SAMPLE,
			    HardwareDeviceRange(taskCount), HardwareDeviceRange(advancePathsWorkGroupSize));
			
			const double timeKernelEnd = WallClockTime();
			totalKernelTime += timeKernelEnd - timeKernelStart;

			//------------------------------------------------------------------

            numFrames++;
			threadFilms[0]->film->SetSampleCount(numFrames, numFrames, 0.0);

			// Check halt conditions
			if (engine->film->GetConvergence() == 1.f || numFrames >= haltSpp)
				break;
		}
        free(pathStatesReadBuffer);

	} catch (boost::thread_interrupted) {
		SLG_LOG("[PathOCLRenderThread::" << threadIndex << "] Rendering thread halted");
	}
    SLG_LOG("[PathOCLRespirRenderThread::" << threadIndex << "]: Rendered " << numFrames << " frames in total.");

	threadFilms[0]->RecvFilm(intersectionDevice);
	intersectionDevice->FinishQueue();
	
	threadDone = true;
	
	intersectionDevice->PopThreadCurrentDevice();
}

#endif
