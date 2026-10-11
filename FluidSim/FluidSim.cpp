#include "FluidSim.h"

SIM_BEG;

using namespace bl;

/// ─────────────────────── project setup ───────────────────────

void FluidSim_Project::UI::sidebar()
{
    bl_scoped(viewport_count);

    ImGui::SliderInt("Viewport Count", &viewport_count, 1, 8);
}

void FluidSim_Project::projectPrepare(Layout& layout)
{
    for (int i = 0; i < viewport_count; i++)
        layout << create<FluidSim_Scene>();
}

/// ────────────────────────── scene ──────────────────────────

void FluidSim_Scene::UI::sidebar()
{
    if (ImGui::CollapsingHeaderBox("View", true))
    {
        bl_scoped(camera);

        camera.populateUI();
        ImGui::EndCollapsingHeaderBox();
    }

}

void FluidSim_Scene::uploadDyeTexture() const
{
    // convert dye concentrations to display bytes
    dye_pixels.resize(fluid.dye.size());

    for (size_t i = 0; i < fluid.dye.size(); ++i)
    {
        double value = std::clamp(fluid.dye[i], 0.0, 1.0);
        dye_pixels[i] = static_cast<unsigned char>(value * 255.0 + 0.5);
    }

    // restore previous GL state on exit and allow tightly packed single-channel rows
    GLStateGuard guard;
    glActiveTexture(GL_TEXTURE0);
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);

    if (dye_texture == 0)
    {
        // create the texture on first use
        glGenTextures(1, &dye_texture);
        glBindTexture(GL_TEXTURE_2D, dye_texture);

        // blend neighbouring cell values and prevent sampling across opposite edges
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

        // allocate one normalized byte per cell for the fixed-size fluid grid
        glTexImage2D(
            GL_TEXTURE_2D, 0, GL_R8,
            fluid.width, fluid.height, 0,
            GL_RED, GL_UNSIGNED_BYTE, nullptr);
    }
    else
    {
        glBindTexture(GL_TEXTURE_2D, dye_texture);
    }

    // replace the texture contents with the current dye values without reallocating
    glTexSubImage2D(
        GL_TEXTURE_2D, 0, 0, 0,
        fluid.width, fluid.height,
        GL_RED, GL_UNSIGNED_BYTE, dye_pixels.data());
}

void FluidSim_Scene::sceneStart()
{
    if (show_particles)
    {
        for (int y = 0; y < particles_y; y++)
        {
            for (int x = 0; x < particles_x; x++)
            {
                double tx = double(x) / double(particles_x);
                double ty = double(y) / double(particles_y);
                double px = std::lerp(0.0, world_w, tx);
                double py = std::lerp(0.0, world_h, ty);

                particles.push_back(Particle{px, py});
            }
        }
    }
    
    bool clockwise = false;
    for (int i = 0; i < 20; i++)
    {
        double vortex_x = this->random(0.0, world_w);
        double vortex_y = this->random(0.0, world_h);
        double vortex_radius = random(0.1, 0.3) * world_hypot;
        double vortex_strength = (clockwise ? 1.0 : -1.0) * 0.2;// random(0.1, 1.0);

        fluid.addVortex(
            DVec2{ vortex_x, vortex_y },
            vortex_radius,
            vortex_strength
        );
        clockwise = !clockwise;
    }

    surface.setFragmentSource(fluid_shader_src);
}

void FluidSim_Scene::sceneMounted([[maybe_unused]] Viewport* ctx)
{
    /// Initialize viewport.
    camera.setOriginViewportAnchor(Anchor::CENTER);
    camera.focusWorldRect(*ctx, 0.0, 0.0, world_w, world_h);

    /// Enable reverting to the current camera view.
    camera.uiSetCurrentAsDefault();

    /// Configure default navigator.
    navigator.setTarget(camera);
    navigator.setDirectCameraPanning(true);
}

void FluidSim_Scene::sceneUnmounted([[maybe_unused]] Viewport* ctx)
{
    /// Release state owned specifically for this Viewport.
}

void FluidSim_Scene::sceneDestroy()
{
    renderThreadQueue().post([this]()
    {
        if (dye_texture)
        {
            glDeleteTextures(1, &dye_texture);
            dye_texture = 0;
        }

        surface.destroy();
    });
}

void FluidSim_Scene::sceneProcess(
    [[maybe_unused]] double dt
)
{
    // redraw the scene every frame
    requestRedraw(true);
}

void FluidSim_Scene::viewportProcess(
    [[maybe_unused]] Viewport* ctx,
    [[maybe_unused]] double dt)
{
    fluid.advect(dt);

    //double before = fluid.maxDivergence();
    fluid.project(dt);
    //double after = fluid.maxDivergence();

    fluid.advectDye(dt);

    //blPrint() << "divergance before: " << before << "  after: " << after;
    if (show_particles)
    {
        for (auto& particle : particles)
        {
            DVec2 velocity = fluid.sampleVelocity(particle.pos);
            particle.pos += velocity * dt;
        }
    }
}

void FluidSim_Scene::viewportDraw(Viewport* ctx) const
{
    /// Draw the Scene on this viewport without modifying simulation state.
    ctx->transform(camera);

    uploadDyeTexture();
    surface.render(512, 512, [&](const ShaderSurface& shader)
    {
        shader.bindTexture2D("u_dye", dye_texture);
    });

    DQuad bounds{
        fluid.origin.x,
        fluid.origin.y,
        fluid.origin.x + fluid.width * fluid.cell_size,
        fluid.origin.y + fluid.height * fluid.cell_size
    };

    ctx->worldMode();
    ctx->drawShaderSurface(surface, bounds);
    ctx->drawWorldAxis();

    if (show_particles)
    {
        ctx->worldHudMode();
        for (auto& particle : particles)
        {
            ctx->fillEllipse(particle.pos, 0.5, Color::white);
        }
    }
}

void FluidSim_Scene::onEvent(Event e)
{
    if (e.ownedBy(this))
        navigator.handleWorldNavigation(e, true);
}

// void FluidSim_Scene::onPointerDown(PointerEvent e) {}
// void FluidSim_Scene::onPointerUp(PointerEvent e) {}
// void FluidSim_Scene::onPointerMove(PointerEvent e) {}
// void FluidSim_Scene::onWheel(PointerEvent e) {}
// void FluidSim_Scene::onKeyDown(KeyEvent e) {}
// void FluidSim_Scene::onKeyUp(KeyEvent e) {}


SIM_END;
