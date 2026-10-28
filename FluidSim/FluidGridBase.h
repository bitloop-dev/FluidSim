#pragma once
#include "FluidGridTypes.h"
#include "FluidGridEdit.h"
#include "ElementCatalogue.h"
#include "FluidMixture.h"
#include "FluidCheckpoint.h"
#include <bitloop/core/types.h>

namespace fluid
{

// Backend-independent simulation operations. Native storage stays in each engine.
class FluidGridBase
{
public:
    using Scalar           = double;
    using State            = FluidGridState<Scalar>;
    using ProjectionResult = FluidProjectionResult;
    using StepTimings      = FluidStepTimings;
    using Edit             = FluidGridEdit<FluidGridBase>;

    virtual ~FluidGridBase() = default;

    virtual int getWidth() const = 0;
    virtual int getHeight() const = 0;
    virtual double getCellSize() const = 0;
    virtual bl::DVec2 getOrigin() const = 0;
    virtual void setOrigin(bl::DVec2 origin) = 0;
    // A GL context requirement, not permission to move a live device between threads.
    // WebGPU operations/readback/destruction stay on the chosen simulation owner.
    virtual bool requiresRenderThread() const = 0;

    // Configure membership/order before upload; changing mixtures requires another upload.
    virtual void configureMaterials(ElementCatalogue elements, FluidMixture liquid, FluidMixture gas) = 0;
    virtual const ElementCatalogue& getElements() const = 0;
    virtual const FluidMixture& getLiquidMixture() const = 0;
    virtual const FluidMixture& getGasMixture() const = 0;
    virtual double getGravityY() const = 0;
    virtual void setGravityY(double acceleration) = 0;

    virtual void upload(const State& state) = 0;
    virtual State download() const = 0;

    // Completed-step snapshots on the simulation owner. False means unsupported;
    // incompatible/corrupt checkpoints throw before applying state. No backend migration.
    virtual bool saveCheckpoint(FluidCheckpoint&) const { return false; }
    virtual bool loadCheckpoint(const FluidCheckpoint&) { return false; }

    Edit beginEdit() { return Edit(*this); }

    virtual void destroy() = 0;

    // Optional timings observe this substep; GPU stages include completion waits.
    virtual ProjectionResult step(double dt, int max_iterations = 200, double divergence_tolerance = 1e-6, StepTimings* timings = nullptr) = 0;
    virtual double maxDivergence() const = 0;

    // Derived RGBA display output; reuse caller storage without downloading the physical state.
    virtual void writeColorPixels(std::vector<float>& pixels) const = 0;

protected:
    virtual void commitEdits(detail::CellEditBatch& batch) = 0;

private:
    friend Edit;
    friend class FluidGrid;
};

}
