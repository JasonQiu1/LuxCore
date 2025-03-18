#line 2 "respir_kernels_micro.cl"

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

//------------------------------------------------------------------------------
// SpatialReuse (Micro-Kernels)
//------------------------------------------------------------------------------

#define DEBUG_PRINTF_SR_KERNEL_NAME 1
#ifndef DEBUG_GID
#define DEBUG_GID 159982
#endif

//------------------------------------------------------------------------------
// SpatialReuse_MK_INIT Kernel
//
// Initializes the spatial reuse pass.
//------------------------------------------------------------------------------
__kernel void SpatialReuse_MK_INIT(
    KERNEL_ARGS
    KERNEL_ARGS_SPATIALREUSE
) {
    const size_t gid = get_global_id(0);

    // Read the path state
#if defined(DEBUG_PRINTF_SR_KERNEL_NAME)
    if (gid == DEBUG_GID)
        printf("Kernel: SpatialReuse_MK_INIT(state = %d)\n", pathStates[gid]);
#endif

    //--------------------------------------------------------------------------
    // Start of variables setup
    //--------------------------------------------------------------------------
    SampleResult *sampleResult = &sampleResultsBuff[gid];
    const Film* restrict film = &taskConfig->film;
    RespirReservoir* reservoir = &tasksState[gid].reservoir;
    SpatialReuseData* restrict spatialReuseData = &spatialReuseDatas[gid];
    const Ray* ray = &rays[gid];

    //--------------------------------------------------------------------------
    // End of variables setup
    //--------------------------------------------------------------------------

    reservoir->M = 1;

    // no reconnection vertex, no invertible shift mapping
    // CAN shift FROM other domains, CANNOT shift TO other domains
    if (reservoir->sample.rc.pathDepth == -1 
        || reservoir->sample.rc.pathDepth > reservoir->sample.pathDepth) {
        if (gid == DEBUG_GID)
            printf("SpatialReuse_MK_INIT: No reconnection vertex.\n");
        // keep pathstate to SYNC so that resampling and visibility kernels do not run
        // keep pixelIndexMap to be -1 so this pixel isn't resampled from
        return;
    }

    // Finalize initial path resampling RIS: calculate unbiased contribution weight of final sample
    const float integrand = Radiance_Filter(film, reservoir->sample.integrand);
    if (reservoir->weight != 0) {
        reservoir->weight /= integrand;
    }

    // Save ray time state
    spatialReuseData->preSpatialReuseTime = ray->time;

    /*
    // Set up for first spatial reuse iteration.
    */

    // Prime neighbor search
    spatialReuseData->numNeighborsLeft = numSpatialNeighbors;
    spatialReuseData->neighborGid = -1;
    spatialReuseData->numValidNeighbors = 0;
    PixelIndexMap_Set(pixelIndexMap, filmWidth, 
            sampleResult->pixelX, sampleResult->pixelY, 
            gid);

    // Init resampling reservoir and canonical MIS weight
    RespirReservoir_Init(&spatialReuseData->spatialReuseReservoir);
    spatialReuseData->canonicalMisWeight = 1.f;

    // Prime pathstate
    pathStates[gid] = (PathState) SR_MK_NEXT_NEIGHBOR;
}

//------------------------------------------------------------------------------
// SpatialReuse_MK_NEXT_NEIGHBOR Kernel
//
// Generates a candidate neighbor and if valid stores it in ShiftInOutData.
// Shift from current domain to neighbor domain for canonical pairwise MIS calculation.
//
// FROM: SpatialReuse_MK_INIT (begin the pass)
// FROM: SpatialReuse_MK_FINISH_RESAMPLE (loop through all neighbors)
// TO: SpatialReuse_MK_NEXT_NEIGHBOR (invalid neighbor generated)
// TO: SpatialReuse_MK_SHIFT (-> SpatialReuse_MK_FINISH_RESAMPLE)
// TO: SYNC (no more neighbors)
//------------------------------------------------------------------------------
__kernel void SpatialReuse_MK_NEXT_NEIGHBOR(
    KERNEL_ARGS
    KERNEL_ARGS_SPATIALREUSE
    KERNEL_ARGS_SHIFT
) {
    const size_t gid = get_global_id(0);

    if (pathStates[gid] != SR_MK_NEXT_NEIGHBOR)
        return;

#if defined(DEBUG_PRINTF_SR_KERNEL_NAME)
    if (gid == DEBUG_GID)
        printf("Kernel: SpatialReuse_MK_NEXT_NEIGHBOR(state = %d)\n", pathStates[gid]);
#endif

    //--------------------------------------------------------------------------
    // Start of variables setup
    //--------------------------------------------------------------------------
    GPUTask* restrict task = &tasks[gid];
    const SampleResult* restrict sampleResult = &sampleResultsBuff[gid];
    SpatialReuseData* restrict spatialReuseData = &spatialReuseDatas[gid];
    ShiftInOutData* restrict shiftInOutData = &shiftInOutDatas[gid];
    //--------------------------------------------------------------------------
    // End of variables setup
    //--------------------------------------------------------------------------

    // Generate a valid neighbor
    while (true) {
        if (spatialReuseData->numNeighborsLeft == 0) {
            // No more neighbors, this iteration is finished
            pathStates[gid] = (PathState) SYNC;
            return;
        }

        // Generate a candidate neighbor gid
        if (Respir_UpdateNextNeighborGid(
            spatialReuseData, sampleResult, 
            spatialRadius, pixelIndexMap, filmWidth, filmHeight, &task->seed))
        {
            // Found valid neighbor
            break;
        }
    }

    /*
    /	Set up CENTRAL (current/canonical pixel) -> NEIGHBOR shift 
    / 	for canonical pairwise MIS accumulation
    */
    // Set up inputs to MK_SHIFT
    shiftInOutData->shiftSrcGid = gid;
    shiftInOutData->shiftDstGid = spatialReuseData->neighborGid;
    shiftInOutData->afterShiftState = SR_MK_RESAMPLE;

    pathStates[gid] = (PathState) SR_MK_SHIFT;
}

//------------------------------------------------------------------------------
// SpatialReuse_MK_SHIFT Kernel
//
// Shift from one reservoir (domain/pixel) SRC to another (domain/pixel) DST.
// This means the resulting path will start at DST's path.
// Stores shifted integrand and jacobian into ShiftInOutData->shiftReservoir.
//
// FROM: SpatialReuse_MK_NEXT_NEIGHBOR (for canonical pairwise MIS calculation)
// FROM: SpatialReuse_MK_RESAMPLE (for resampling neighbors into current reservoir)
// TO: SpatialReuse_MK_CHECK_VISIBILITY (if jacobian is valid)
// TO: PathState stored in ShiftInOutData (if jacobian is invalid)
//		- SpatialReuse_MK_RESAMPLE if coming from SpatialReuse_MK_NEXT_NEIGHBOR
//		- SpatialReuse_MK_FINISH_RESAMPLE if coming from SpatialReuse_MK_RESAMPLE
//------------------------------------------------------------------------------
__kernel void SpatialReuse_MK_SHIFT(
    KERNEL_ARGS
    KERNEL_ARGS_SHIFT
) {
    const size_t gid = get_global_id(0);

    // Check correct pathstate during async execution
    if (pathStates[gid] != SR_MK_SHIFT)
        return;

    #if defined(DEBUG_PRINTF_SR_KERNEL_NAME)
    if (gid == DEBUG_GID)
        printf("Kernel: SpatialReuse_MK_SHIFT(state = %d)\n", pathStates[gid]);
    #endif

    //--------------------------------------------------------------------------
    // Start of variables setup
    //--------------------------------------------------------------------------
    
    const Film* restrict film = &taskConfig->film;
    const Scene* restrict scene = &taskConfig->scene;
    ShiftInOutData* shiftInOutData = &shiftInOutDatas[gid];

    // Initialize shift reservoir to use as output for shifted integrand and jacobian
    RespirReservoir* restrict out = &shiftInOutData->shiftReservoir;
    RespirReservoir_Init(out);
    const RespirReservoir* src = &tasksState[shiftInOutData->shiftSrcGid].reservoir;
    const RespirReservoir* dst = &tasksState[shiftInOutData->shiftDstGid].reservoir;
    const RcVertex* rc = &src->sample.rc;

    if (rc->pathDepth == -1 || rc->pathDepth > src->sample.pathDepth) {
        // No reconnection vertex, invalid shift
        Respir_HandleInvalidShift(shiftInOutData, out, &pathStates[gid]);
        return;
    }

    bool isRcVertexFinal = src->sample.pathDepth == rc->pathDepth;
    bool isRcVertexEscapedVertex = src->sample.pathDepth + 1 == rc->pathDepth && !src->sample.isLastVertexNee;
    bool isRcVertexNEE = isRcVertexFinal && src->sample.isLastVertexNee;

    // Initialize image maps page pointer table
    INIT_IMAGEMAPS_PAGES

    //--------------------------------------------------------------------------
    // End of variables setup
    //--------------------------------------------------------------------------

    /*
    /	Calculate and verify valid Jacobian determinant for the shift.
    */

    const float3 rcPoint = VLOAD3F(&rc->bsdf.hitPoint.p.x);
    const float3 dstPoint = VLOAD3F(&dst->sample.prefixBsdf.hitPoint.p.x);

    float3 dstToRc = rcPoint - dstPoint;
    const float dstDistanceSquared = dot(dstToRc, dstToRc);
    const float dstDistance = sqrt(dstDistanceSquared);
    dstToRc /= dstDistance;

    // absolute value of Cos(angle from surface normal of rc point to prefix point) 
    const float3 rcGeometricN = HitPoint_GetGeometryN(&rc->bsdf.hitPoint);
    const float dstCosW = abs(dot(dstToRc, rcGeometricN));

    // Cached jacobian is src: (sqr distance) / (cos angle)
    out->sample.rc.jacobian = rc->jacobian * (dstCosW / dstDistanceSquared);

    if (get_global_id(0) == DEBUG_GID) {
        const float3 srcPoint = VLOAD3F(&src->sample.prefixBsdf.hitPoint.p.x);
        printf("Src prefix point: (%f, %f, %f)\n", srcPoint.x, srcPoint.y, srcPoint.z);
        printf("Dst prefix point: (%f, %f, %f)\n", dstPoint.x, dstPoint.y, dstPoint.z);
        printf("Src rc vertex point: (%f, %f, %f)\n", rcPoint.x, rcPoint.y, rcPoint.z);
        printf("Src rc geometric normal: (%f, %f, %f)\n", rcGeometricN.x, rcGeometricN.y, rcGeometricN.z);
        printf("Initial jacobian determinant: %f\n", out->sample.rc.jacobian);
    }

    // Check if shifted jacobian is valid
    if (Respir_IsInvalidJacobian(out->sample.rc.jacobian)) 
    {
        // Invalid Jacobian, shift fails
        Respir_HandleInvalidShift(shiftInOutData, out, &pathStates[gid]);
        return;
    }

    /*
    /	Verify distance connectability condition.
    */

    // distance threshold of 2-5% world size recommended by GRIS paper
    // TODO: make distance threshold configurable as percent world size
    const float3 srcToDstRc = VLOAD3F(&dst->sample.rc.bsdf.hitPoint.p.x) 
            - VLOAD3F(&src->sample.prefixBsdf.hitPoint.p.x);
    const float srcToDstRcDistance = sqrt(dot(srcToDstRc, srcToDstRc));
    const float minDistance = worldRadius * 2 * 0.025; 
    if (srcToDstRcDistance < minDistance || dstDistance < minDistance) {
        // Shift failed or noninvertible
        Respir_HandleInvalidShift(shiftInOutData, out, &pathStates[gid]);
        return;
    }

    /*
    /	Calculate and verify valid, nonblack shifted integrand.
    */

    // Correct jacobian for scattering pdf from prefix vertex towards reconnection
    // Use cached BSDF info from src/base path
    const float srcPdf = rc->prefixToRcPdf;
    const BSDF* dstBsdf = &dst->sample.prefixBsdf;
    float dstPdf = 1.0f;
    BSDFEvent event;
    const float3 dstBsdfValue = BSDF_Evaluate(dstBsdf,
            dstToRc, &event, &dstPdf
            MATERIALS_PARAM);
    out->sample.rc.jacobian *= dstPdf / srcPdf;
    if (Respir_IsInvalidJacobian(out->sample.rc.jacobian) || Spectrum_IsBlack(dstBsdfValue)) {
        Respir_HandleInvalidShift(shiftInOutData, out, &pathStates[gid]);
        return;
    }

    if (get_global_id(0) == DEBUG_GID) {
        printf("Correct difference in prefix scatter pdfs, jacobian: %f\n", out->sample.rc.jacobian);
    }

    // Correct jacobian for bsdf scattering value from prefix vertex to reconnection to incident dir
    // Use cached BSDF info from src/base path
    float dstPdf2 = 1.0f;
    float srcRcIncidentPdf = 1.0f;
    float dstRcIncidentPdf = 1.0f;
    float3 dstRcIncidentBsdfValue = WHITE;
    
    const BSDF* rcBsdf = &rc->bsdf;
    if (!isRcVertexEscapedVertex) {
        srcRcIncidentPdf = rc->incidentPdf;
        dstRcIncidentBsdfValue = BSDF_Evaluate(rcBsdf,
            VLOAD3F(&rc->incidentDir.x), 
            &event, &dstRcIncidentPdf
            MATERIALS_PARAM);

        if (!isRcVertexNEE) {
            dstPdf2 = dstRcIncidentPdf;
        } else {
            dstPdf2 = src->sample.lightPdf;
        }
    }
    if (Spectrum_IsBlack(dstRcIncidentBsdfValue)) {
        Respir_HandleInvalidShift(shiftInOutData, out, &pathStates[gid]);
        return;
    }

    // Scale by difference
    Radiance_ScaleGroup(film, rc->irradiance,
        (dstBsdfValue / dstPdf) * (dstRcIncidentBsdfValue / dstPdf2),
        out->sample.integrand
    );

    if (get_global_id(0) == DEBUG_GID) {
        printf("Correct difference in rc scatter bsdfs, jacobian: %f\n", out->sample.rc.jacobian);
    }

    // TODO: might not need this if we're not using multi-lobed materials
    if (isRcVertexEscapedVertex) {
        Radiance_Scale(film, out->sample.integrand,
            PowerHeuristic(dstPdf, src->sample.lightPdf), // supposed to be dstPdf evaluated on all lobes at dst prefix vertex
            out->sample.integrand);
        if (get_global_id(0) == DEBUG_GID) {
            printf("Correct mis weights for NEE paths ending with rc vertex, jacobian: %f\n", out->sample.rc.jacobian);
        }
    }

    if (isRcVertexFinal) {
        float misWeight = 1.0f;
        if (isRcVertexNEE) {
            misWeight = PowerHeuristic(src->sample.lightPdf, dstRcIncidentPdf); // supposed to be dstPdf evaluated on all lobes at dst prefix vertex
        } else {
            misWeight = PowerHeuristic(dstRcIncidentPdf, src->sample.lightPdf); // supposed to be dstPdf evaluated on all lobes at dst prefix vertex
        }
        // TODO: might not need this if we're not using multi-lobed materials
        Radiance_Scale(film, out->sample.integrand,
            misWeight, 
            out->sample.integrand);

        if (get_global_id(0) == DEBUG_GID) {
            printf("Correct mis weights for paths ending with rc vertex, jacobian: %f\n", out->sample.rc.jacobian);
        }
    }

    if ((isRcVertexFinal && !isRcVertexNEE) || (!isRcVertexFinal && !isRcVertexEscapedVertex)) {
        out->sample.rc.jacobian *= dstRcIncidentPdf / srcRcIncidentPdf;
        if (get_global_id(0) == DEBUG_GID) {
            printf("Correct difference in rc scatter pdfs for paths, jacobian: %f\n", out->sample.rc.jacobian);
        }
    }

    if (Respir_IsInvalidJacobian(out->sample.rc.jacobian) || Radiance_IsBlack(film, out->sample.integrand)) {
        Respir_HandleInvalidShift(shiftInOutData, out, &pathStates[gid]);
        return;
    }

    /*
    /	Set up shadow ray for visibility check.
    */

    // Do visibility check from dst primary hit vertex to src secondary hit vertex
    // Initialize the trough a shadow transparency flag used by Scene_Intersect()
    tasksDirectLight[gid].throughShadowTransparency = false;

    // Make a copy of current PathVolumeInfo for tracing the
    // shadow ray
    directLightVolInfos[gid] = eyePathInfos[gid].volume;

    const float3 shadowRayOrigin = BSDF_GetRayOrigin(&dst->sample.prefixBsdf, dstToRc);
    float3 shadowRayDir = rcPoint + (BSDF_GetLandingGeometryN(&dst->sample.prefixBsdf) 
            * MachineEpsilon_E_Float3(rcPoint) * (rcBsdf->hitPoint.intoObject ? 1.f : -1.f)) - 
            shadowRayOrigin;
    const float shadowRayDirDistanceSquared = dot(shadowRayDir, shadowRayDir);
    const float shadowRayDirDistance = sqrt(shadowRayDirDistanceSquared);
    shadowRayDir /= shadowRayDirDistance;
    Ray_Init4(&rays[gid], shadowRayOrigin, shadowRayDir, 0.f, shadowRayDirDistance, dst->sample.hitTime);

    pathStates[gid] = (PathState) SR_MK_CHECK_VISIBILITY;
    return;
}

//------------------------------------------------------------------------------
// SpatialReuse_MK_CHECK_VISIBILITY Kernel
//
// Checks if the shadow ray shot to the reconnection vertex is blocked or not.
// If visible, then update ShiftInOutData->shiftReservoir appropriately (TODO EXACTLY WHAT??).
//
// FROM: SpatialReuse_MK_SHIFT
// TO: SpatialReuse_MK_CHECK_VISIBILITY (if ray is not finished tracing)
// TO: PathState stored in ShiftInOutData (if ray has finished tracing)
//		- SpatialReuse_MK_RESAMPLE if coming from SpatialReuse_MK_NEXT_NEIGHBOR
//		- SpatialReuse_MK_FINISH_RESAMPLE if coming from SpatialReuse_MK_RESAMPLE
//
//------------------------------------------------------------------------------
__kernel void SpatialReuse_MK_CHECK_VISIBILITY(
        KERNEL_ARGS
        KERNEL_ARGS_SPATIALREUSE
        KERNEL_ARGS_SHIFT
) {
    const size_t gid = get_global_id(0);

    if (pathStates[gid] != SR_MK_CHECK_VISIBILITY)
        return;

    #if defined(DEBUG_PRINTF_SR_KERNEL_NAME)
    if (gid == DEBUG_GID)
        printf("Kernel: SpatialReuse_MK_CHECK_VISIBILITY(state = %d)\n", pathStates[gid]);
    #endif

    //--------------------------------------------------------------------------
    // Start of variables setup
    //--------------------------------------------------------------------------
    GPUTask* restrict task = &tasks[gid];
    GPUTaskDirectLight *taskDirectLight = &tasksDirectLight[gid];
    const Scene* restrict scene = &taskConfig->scene;
    ShiftInOutData* shiftInOutData = &shiftInOutDatas[gid];

    // Initialize image maps page pointer table
    INIT_IMAGEMAPS_PAGES

    //--------------------------------------------------------------------------
    // End of variables setup
    //--------------------------------------------------------------------------

    float3 connectionThroughput = WHITE;

    // TODO: SAVE THE PASSTHROUGH SEED WHEN CACHING RECONNECTION VERTEX
    Seed seedPassThroughEvent = taskDirectLight->seedPassThroughEvent;
    const float passThroughEvent = Rnd_FloatValue(&seedPassThroughEvent);
    taskDirectLight->seedPassThroughEvent = seedPassThroughEvent;

    // TODO: SAVE THE THROUGHSHADOWTRANSPARENCY WHEN CACHING (?)
    int throughShadowTransparency = taskDirectLight->throughShadowTransparency;
    const bool continueToTrace =
        Scene_Intersect(taskConfig,
            EYE_RAY | SHADOW_RAY,
            &throughShadowTransparency,
            &directLightVolInfos[gid],
            &task->tmpHitPoint,
            passThroughEvent,
            &rays[gid], &rayHits[gid], &task->tmpBsdf,
            &connectionThroughput, WHITE,
            NULL,
            true
            MATERIALS_PARAM
            );

    const bool rayMiss = (rayHits[gid].meshIndex == NULL_INDEX);

    // If continueToTrace, there is nothing to do, just keep the same state
    if (continueToTrace) {
        return;
    }

    if (!rayMiss) {
        // Hit something, meaning reconnection vertex is not visible
        // Shift fails due to occlusion
        taskDirectLight->directLightResult = SHADOWED;
        Respir_HandleInvalidShift(shiftInOutData, &shiftInOutData->shiftReservoir, &pathStates[gid]);
        return;
    }

    // Shift is successful, keep shifted integranad and jacobian in shiftReservoir 
    taskDirectLight->directLightResult = ILLUMINATED;
    pathStates[gid] = (PathState) shiftInOutData->afterShiftState;
}

//------------------------------------------------------------------------------
// SpatialReuse_MK_RESAMPLE Kernel
//
// Update the canonical pairwise MIS weight.
// Resample the neighbor reservoir into the current reservoir.
//
// FROM: (SpatialReuse_MK_NEXT_NEIGHBOR->) SpatialReuse_MK_SHIFT
// TO: SpatialReuse_MK_SHIFT (-> SpatialReuse_MK_FINISH_RESAMPLE)
//------------------------------------------------------------------------------
__kernel void SpatialReuse_MK_RESAMPLE(
    KERNEL_ARGS
    KERNEL_ARGS_SPATIALREUSE
    KERNEL_ARGS_SHIFT
) {
    const size_t gid = get_global_id(0);

    if (pathStates[gid] != SR_MK_RESAMPLE)
        return;

    #if defined(DEBUG_PRINTF_SR_KERNEL_NAME)
    if (gid == DEBUG_GID)
        printf("Kernel: SpatialReuse_MK_RESAMPLE(state = %d)\n", pathStates[gid]);
    #endif
        
    //--------------------------------------------------------------------------
    // Start of variables setup
    //--------------------------------------------------------------------------
    GPUTask* restrict task = &tasks[gid];
    const Film* restrict film = &taskConfig->film;
    SpatialReuseData* spatialReuseData = &spatialReuseDatas[gid];
    ShiftInOutData* shiftInOutData = &shiftInOutDatas[gid];

    const RespirReservoir* restrict shifted = &shiftInOutData->shiftReservoir;
    const RespirReservoir* central = &tasksState[shiftInOutData->shiftSrcGid].reservoir;
    const RespirReservoir* neighbor = &tasksState[shiftInOutData->shiftDstGid].reservoir;

    //--------------------------------------------------------------------------
    // End of variables setup
    //--------------------------------------------------------------------------

    /*
    // 	Update canonical pairwise MIS weight with shifted integrand and jacobian.
    */
    spatialReuseData->canonicalMisWeight += 1.0f;
    const float prefixApproximateIntegrandM = neighbor->M * Radiance_Filter(film, shifted->sample.integrand) * shifted->sample.rc.jacobian;
    if (prefixApproximateIntegrandM >= 0.0f) {
        const float centralIntegrandM = central->M * Radiance_Filter(film, central->sample.integrand);
        spatialReuseData->canonicalMisWeight -= prefixApproximateIntegrandM
                / (prefixApproximateIntegrandM + centralIntegrandM / numSpatialNeighbors);
    } 

    /*
    //	Set up NEIGHBOR -> CENTRAL (current/canonical pixel) shift 
    //	for neighbor resampling
    */

    // Set up inputs to MK_SHIFT
    shiftInOutData->shiftSrcGid = spatialReuseData->neighborGid;
    shiftInOutData->shiftDstGid = gid;
    shiftInOutData->afterShiftState = SR_MK_FINISH_RESAMPLE;

    pathStates[gid] = (PathState) SR_MK_SHIFT;
    }

//------------------------------------------------------------------------------
// SpatialReuse_MK_FINISH_RESAMPLE Kernel
//
// Finish resampling neighbor into current based off of shift mapping results.
// Merge with spatialReuseReservoir using pairwise MIS.
//
// FROM: (SpatialReuse_MK_RESAMPLE->) SpatialReuse_MK_SHIFT
// TO: SR_MK_NEXT_NEIGHBOR (continue iterating through other neighbors)
//------------------------------------------------------------------------------
__kernel void SpatialReuse_MK_FINISH_RESAMPLE(
    KERNEL_ARGS
    KERNEL_ARGS_SPATIALREUSE
    KERNEL_ARGS_SHIFT
) {
    const size_t gid = get_global_id(0);

    GPUTask *task = &tasks[gid];
    if (pathStates[gid] != SR_MK_FINISH_RESAMPLE)
        return;

    #if defined(DEBUG_PRINTF_SR_KERNEL_NAME)
    if (gid == DEBUG_GID)
        printf("Kernel: SpatialReuse_MK_FINISH_RESAMPLE(state = %d)\n", pathStates[gid]);
    #endif

    //--------------------------------------------------------------------------
    // Start of variables setup
    //--------------------------------------------------------------------------
    __constant const Film* restrict film = &taskConfig->film;
    SpatialReuseData* spatialReuseData = &spatialReuseDatas[gid];
    ShiftInOutData* shiftInOutData = &shiftInOutDatas[gid];

    RespirReservoir* restrict shifted = &shiftInOutData->shiftReservoir;
    const RespirReservoir* neighbor = &tasksState[shiftInOutData->shiftSrcGid].reservoir;
    const RespirReservoir* central = &tasksState[shiftInOutData->shiftDstGid].reservoir;
    //--------------------------------------------------------------------------
    // End of variables setup
    //--------------------------------------------------------------------------

    /*
    // Calculate pairwise resampling weight for the neighbor sample.
    */
    
    // Save shifted output.
    const float shiftedJacobian = shifted->sample.rc.jacobian;
    Spectrum shiftedIntegrand[FILM_MAX_RADIANCE_GROUP_COUNT];
    Radiance_Copy(film, shifted->sample.integrand, shiftedIntegrand);

    if (get_global_id(0) == DEBUG_GID) {
        printf("Shift neighbor->central\n");
        printf("\tIntegrand: %f\n", Radiance_Filter(film, shiftedIntegrand));
        printf("\tJacobian: %f\n", shiftedJacobian);
    }

    // Make a copy of neighbor so that I can update it with some info before merging.
    *shifted = *neighbor;

    float neighborWeight = 0.0f;

    // Make sure that it's even possible to select by checking that resampling weight is valid.
    const float weight = neighbor->M * Radiance_Filter(film, shiftedIntegrand)
            * shiftedJacobian * neighbor->weight;
    if (weight <= 0.0f || isnan(weight) || isinf(weight)) {
        Radiance_Clear(shiftedIntegrand);
    } else {
        const float neighborIntegrandM = neighbor->M * Radiance_Filter(film, neighbor->sample.integrand) / shiftedJacobian;
        const float shiftedIntegrandM = central->M * Radiance_Filter(film, shiftedIntegrand);
        neighborWeight = neighborIntegrandM / (neighborIntegrandM + shiftedIntegrandM / numSpatialNeighbors);
        if (isnan(neighborWeight) || isinf(neighborWeight)) {
            neighborWeight = 0.0f;
        }
    }

    /*
    //	Resample the shifted reservoir into the spatial reuse reservoir.
    */

    // Set shifted integrand back before merging neighbor with spatial reuse data.
    Radiance_Copy(film, shiftedIntegrand, shifted->sample.integrand);

    RespirReservoir_Merge(&spatialReuseData->spatialReuseReservoir, 
        shifted->sample.integrand, shiftedJacobian, shifted,
        neighborWeight, &task->seed, film);

    pathStates[gid] = (PathState) SR_MK_NEXT_NEIGHBOR;
}

//------------------------------------------------------------------------------
// SpatialReuse_MK_FINISH_ITERATION Kernel
//
// Completes the spatial reuse iteration by finalizing GRIS and setting up the next iteration.
//------------------------------------------------------------------------------
__kernel void SpatialReuse_MK_FINISH_ITERATION(
    KERNEL_ARGS
    KERNEL_ARGS_SPATIALREUSE
) {
    const size_t gid = get_global_id(0);

    RespirReservoir* restrict central = &tasksState[gid].reservoir;
    if (central->sample.rc.pathDepth == -1 
        || central->sample.rc.pathDepth > central->sample.pathDepth) {
        return;
    }

    #if defined(DEBUG_PRINTF_SR_KERNEL_NAME)
    if (gid == DEBUG_GID)
        printf("Kernel: SpatialReuse_MK_FINISH_ITERATION(state = %d)\n", pathStates[gid]);
    #endif

    //--------------------------------------------------------------------------
    // Start of variables setup
    //--------------------------------------------------------------------------
    GPUTask *task = &tasks[gid];
    __constant const Film* restrict film = &taskConfig->film;
    const SampleResult* restrict sampleResult = &sampleResultsBuff[gid];  
    SpatialReuseData* spatialReuseData = &spatialReuseDatas[gid];
    RespirReservoir* restrict srReservoir = &spatialReuseData->spatialReuseReservoir;
    //--------------------------------------------------------------------------
    // End of variables setup
    //--------------------------------------------------------------------------

    /*
    //	Resample the canonical reservoir into the spatial reuse reservoir.
    */
    RespirReservoir_Merge(srReservoir, 
        central->sample.integrand, 1.0f, central,
        spatialReuseData->canonicalMisWeight, &task->seed, film);

    /*
    // 	Finalize GRIS by calculating unbiased contribution weight.
    */
    float srIntegrand = Radiance_Filter(film, srReservoir->sample.integrand);
    if (srIntegrand <= 0.f || isnan(srIntegrand) || isinf(srIntegrand)) {
        srIntegrand = 0.0f;
        srReservoir->weight = 0.0f;
    } else {
        srReservoir->weight /= srIntegrand * (spatialReuseData->numValidNeighbors + 1);
    }

    /*
    // Set up for next spatial reuse iteration.
    */

    // Copy spatial reuse reservoir to central reservoir.
    *central = *srReservoir;

    // Prime neighbor search
    spatialReuseData->numNeighborsLeft = numSpatialNeighbors;
    spatialReuseData->neighborGid = -1;
    spatialReuseData->numValidNeighbors = 0;
    PixelIndexMap_Set(pixelIndexMap, filmWidth, 
            sampleResult->pixelX, sampleResult->pixelY, 
            gid);

    // Init resampling reservoir and canonical MIS weight
    RespirReservoir_Init(srReservoir);
    spatialReuseData->canonicalMisWeight = 1.f;

    pathStates[gid] = (PathState) SR_MK_NEXT_NEIGHBOR;
}

//------------------------------------------------------------------------------
// SpatialReuse_MK_FINISH_REUSE Kernel
//
// Runs after all iterations are complete.
//------------------------------------------------------------------------------
__kernel void SpatialReuse_MK_FINISH_REUSE(
    KERNEL_ARGS
    KERNEL_ARGS_SPATIALREUSE
) {
    const size_t gid = get_global_id(0);

    //--------------------------------------------------------------------------
    // Start of variables setup
    //--------------------------------------------------------------------------
    SpatialReuseData* spatialReuseData = &spatialReuseDatas[gid];
    const RespirReservoir* reservoir = &tasksState[gid].reservoir;
    Ray *ray = &rays[gid];
    SampleResult *sampleResult = &sampleResultsBuff[gid];
    const Film* restrict film = &taskConfig->film;
    //--------------------------------------------------------------------------
    // End of variables setup
    //--------------------------------------------------------------------------

    // Copy final sample's radiance from reservoir to sampleResultsBuff[gid] to be splatted like normal
    Radiance_Copy(film,
        reservoir->sample.integrand,
        sampleResult->radiancePerPixelNormalized);

    if (reservoir->sample.rc.pathDepth == -1 
        || reservoir->sample.rc.pathDepth > reservoir->sample.pathDepth) {
        return;
    }

    Radiance_Scale(film,
        sampleResult->radiancePerPixelNormalized,
        reservoir->weight,
        sampleResult->radiancePerPixelNormalized);

    // maintain integrity of pathtracer by using time from before spatial reuse
    ray->time = spatialReuseData->preSpatialReuseTime;

    // Reinitialize PixelIndexMap state in case the pixel this task is working on changes
    PixelIndexMap_Set(pixelIndexMap, filmWidth, 
        sampleResult->pixelX, sampleResult->pixelY, 
        -1);
    }

//------------------------------------------------------------------------------
// SpatialReuse_MK_SET_SPLAT Kernel
//
// Sets path state to splat.
//------------------------------------------------------------------------------
__kernel void SpatialReuse_MK_SET_SPLAT(
    KERNEL_ARGS
) {
    pathStates[get_global_id(0)] = MK_SPLAT_SAMPLE;
}