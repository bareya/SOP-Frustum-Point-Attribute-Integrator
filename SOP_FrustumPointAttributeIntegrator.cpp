/*MIT License
 *
 * Copyright (c) 2017 Piotr Barejko
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

#include <UT/UT_DSOVersion.h>

#include <GEO/GEO_PrimVolume.h>
#include <GU/GU_PrimVolume.h>
#include <OBJ/OBJ_Camera.h>
#include <OP/OP_AutoLockInputs.h>
#include <OP/OP_Director.h>
#include <OP/OP_Network.h>
#include <OP/OP_Node.h>
#include <OP/OP_OperatorTable.h>
#include <PRM/PRM_SpareData.h>
#include <SOP/SOP_Node.h>
#include <UT/UT_ParallelUtil.h>

#include <tbb/tbb.h>

#include <iostream>
#include <valarray>

struct FrustumPointAttributeIntegrator
{
    using DIPair = std::pair<fpreal32, GA_Index>; // distance - index pair
    using TileData = tbb::concurrent_vector<DIPair>;

    FrustumPointAttributeIntegrator() = default;
    FrustumPointAttributeIntegrator(const char* cameraPath, OP_Context& context);

    void addDependency(OP_Node* node);
    void prependTransform(OBJ_Node* node, OP_Context& context);

    void appendPosition(const UT_Vector4D& position, GA_Index index)
    {
        UT_Vector4D camPos = position * m_worldToCamera;
        UT_Vector4 pos = camPos * m_projection;
        pos /= pos.w();
        pos.x() = 0;

        UT_Vector3F posF(pos);
        if (m_volume->isInside(posF))
        {
            int z;
            m_volume->posToIndex(posF, m_indexx[index], m_indexy[index], z);
            size_t tileIndex = m_volumeWriteHandle->indexToLinearTile(m_indexx[index], m_indexy[index], z);
            m_tiles[tileIndex].emplace_back(-camPos.z(), index);
        }
    }

    void assignPointsToTiles(const GU_Detail* gdp)
    {
        size_t npt = static_cast<size_t>(gdp->getNumPoints());
        m_indexx.resize(npt);
        m_indexy.resize(npt);

        UTparallelForLightItems(UT_BlockedRange<GA_Index>(0, gdp->getNumPoints()),
                                [&](const UT_BlockedRange<GA_Index>& r) {
                                    for (auto it = r.begin(); it != r.end(); ++it)
                                    {
                                        appendPosition(gdp->getPos4(gdp->pointOffset(it)), it);
                                    }
                                });
    }

    void setAttribute(GU_Detail* gdp)
    {
        GA_RWHandleF occlusion = gdp->addFloatTuple(GA_ATTRIB_POINT, GA_SCOPE_PUBLIC, "occlusion", 1);

        // iterate over tails
        UTparallelForHeavyItems(UT_BlockedRange<size_t>(0, m_tiles.size()),
                                [&](const UT_BlockedRange<size_t>& r) {
                                    for (auto tileIt = r.begin(); tileIt != r.end(); ++tileIt)
                                    {
                                        TileData* tile = &m_tiles[tileIt];

                                        // sort points in the tile, sort them by z
                                        std::sort(tile->begin(), tile->end(), [](const DIPair& left, const DIPair& right) {
                                            return left.first < right.first;
                                        });

                                        // iterate over sotred points and increment attribute
                                        for (auto it = tile->begin(); it != tile->end(); ++it)
                                        {
                                            const DIPair* pair = &(*it);
                                            fpreal32 value = m_volumeWriteHandle->getValue(m_indexx[pair->second], m_indexy[pair->second], 0);
                                            occlusion.set(gdp->pointOffset(pair->second), value);
                                            m_volumeWriteHandle->setValue(m_indexx[pair->second], m_indexy[pair->second], 0, value + 1);
                                        }
                                    }
                                });
    }

private:
    OBJ_Camera* m_camera{};
    UT_Matrix4D m_projection;
    UT_Matrix4D m_worldToCamera;

    GU_DetailHandle m_volumeDetail;
    GEO_PrimVolume* m_volume;
    UT_VoxelArrayWriteHandleF m_volumeWriteHandle;

    std::vector<TileData> m_tiles;
    std::vector<int> m_indexx;
    std::vector<int> m_indexy;
};

class SOP_FrustumPointAttributeIntegrator : public SOP_Node
{
private:
    SOP_FrustumPointAttributeIntegrator(OP_Network* parent, const char* name, OP_Operator* entry);

public:
    static OP_Node* constructor(OP_Network* parent, const char* name, OP_Operator* entry);
    virtual ~SOP_FrustumPointAttributeIntegrator() = default;

    static PRM_Template templates[];

    virtual OP_ERROR cookMySop(OP_Context& context);
};

OP_ERROR SOP_FrustumPointAttributeIntegrator::cookMySop(OP_Context& context)
{
    OP_AutoLockInputs inputsLock(this);
    if (inputsLock.lock(context) >= UT_ERROR_ABORT)
    {
        return error();
    }

    fpreal time = context.getTime();

    // parms
    UT_String cameraPath;
    evalString(cameraPath, "camera", 0, time);

    FrustumPointAttributeIntegrator integrator;
    try
    {
        integrator = FrustumPointAttributeIntegrator(cameraPath.c_str(), context);
        integrator.addDependency(this);
        integrator.prependTransform(static_cast<OBJ_Node*>(this->getCreator()), context);
    }
    catch (const std::exception& e)
    {
        addError(SOP_MESSAGE, e.what());
        return error();
    }

    duplicateSource(0, context);

    integrator.assignPointsToTiles(gdp);
    integrator.setAttribute(gdp);

    return error();
}

SOP_FrustumPointAttributeIntegrator::SOP_FrustumPointAttributeIntegrator(OP_Network* parent, const char* name, OP_Operator* entry)
    : SOP_Node(parent, name, entry)
{
}

OP_Node* SOP_FrustumPointAttributeIntegrator::constructor(OP_Network* parent, const char* name, OP_Operator* entry)
{
    return new SOP_FrustumPointAttributeIntegrator(parent, name, entry);
}

namespace
{
PRM_Name parmNames[]{
    PRM_Name("camera", "Camera"),
    PRM_Name("attribute", "Attribute") //
};

PRM_Default parmDefaults[]{
    PRM_Default(0, "/obj/cam1"),
    PRM_Default(0, "pscale") //
};

} // namespace

PRM_Template SOP_FrustumPointAttributeIntegrator::templates[] = {
    PRM_Template(PRM_STRING, PRM_TYPE_DYNAMIC_PATH, 1, &parmNames[0], &parmDefaults[0], 0, 0, 0, &PRM_SpareData::objCameraPath),
    PRM_Template() //
};

void newSopOperator(OP_OperatorTable* table)
{
    table->addOperator(new OP_Operator("SOP_FrustumPointAttributeIntegrator", "SOP_FrustumPointAttributeIntegrator",
                                       SOP_FrustumPointAttributeIntegrator::constructor,
                                       SOP_FrustumPointAttributeIntegrator::templates,
                                       1u, 1u));
}

FrustumPointAttributeIntegrator::FrustumPointAttributeIntegrator(const char* cameraPath, OP_Context& context)
{
    m_camera = static_cast<OBJ_Camera*>(OPgetDirector()->findOBJNode(cameraPath));
    if (!m_camera)
    {
        throw std::runtime_error("Camera not found!");
    }
    if (m_camera->getObjectType() != OBJ_CAMERA)
    {
        throw std::runtime_error("Provided object is not a camera object!");
    }

    m_camera->getProjectionMatrix(context, m_projection);
    m_camera->getInverseLocalToWorldTransform(context, m_worldToCamera);

    // allocate camera volume and set size of the volume
    m_volumeDetail.allocateAndSet(new GU_Detail());
    GU_Detail* geo = m_volumeDetail.writeLock();
    m_volume = GU_PrimVolume::build(geo);
    if (!m_volume)
    {
        throw std::runtime_error("Failed to create camera volume!");
    }
    m_volumeWriteHandle = m_volume->getVoxelWriteHandle();
    m_volumeWriteHandle->size(m_camera->RESX(context.getTime()), m_camera->RESY(context.getTime()), 1);

    // tiles allocation, each tile stores points
    m_tiles.resize(static_cast<size_t>(m_volumeWriteHandle->numTiles()));
    for (auto it = m_tiles.begin(); it != m_tiles.end(); ++it)
    {
        (*it).reserve(2000);
    }
}

void FrustumPointAttributeIntegrator::addDependency(OP_Node* node)
{
    node->addExtraInput(static_cast<OBJ_Node*>(m_camera), OP_INTEREST_DATA);
    const PRM_ParmList* parmList = m_camera->getParmList();
    for (int i = 0; i < parmList->getEntries(); i++)
    {
        node->addExtraInput(*m_camera, i, -1);
    }
}

void FrustumPointAttributeIntegrator::prependTransform(OBJ_Node* node, OP_Context& context)
{
    UT_Matrix4D parentXform;
    node->getInverseLocalToWorldTransform(context, parentXform);

    // update world and current
    m_worldToCamera = parentXform * m_worldToCamera;
}
