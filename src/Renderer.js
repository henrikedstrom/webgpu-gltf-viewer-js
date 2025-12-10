/**
 * @file Renderer.js
 * @brief Main rendering class for WebGPU-based glTF viewer with IBL support.
 */

const { mat4, vec4 } = glMatrix;
import PanoramaToCubemapConverter from './PanoramaToCubemapConverter.js';
import MipmapGenerator, { MipKind } from './MipmapGenerator.js';
import EnvironmentPreprocessor from './EnvironmentPreprocessor.js';

// IBL texture size constants
const kIrradianceMapSize = 64;
const kPrecomputedSpecularMapSize = 512;
const kBRDFIntegrationLUTMapSize = 128;

export default class Renderer {
  /**
   * @brief Constructs a new renderer instance.
   */
  constructor() {
    // WebGPU resources
    this.device = null;
    this.context = null;
    this.format = null;
    this.depthTexture = null;
    this.depthTextureView = null;
    this.renderPassDescriptor = null;

    // Global data
    this.globalUniformBuffer = null;
    this.globalBindGroupLayout = null;
    this.globalBindGroup = null;

    // Environment and IBL related data
    this.environmentTexture = null;
    this.environmentTextureView = null;
    this.iblIrradianceTexture = null;
    this.iblIrradianceTextureView = null;
    this.iblSpecularTexture = null;
    this.iblSpecularTextureView = null;
    this.iblBrdfIntegrationLUT = null;
    this.iblBrdfIntegrationLUTView = null;
    this.environmentCubeSampler = null;
    this.iblBrdfIntegrationLUTSampler = null;
    this.environmentShaderModule = null;
    this.environmentPipeline = null;

    // Model related data
    this.modelShaderModule = null;
    this.materialBindGroupLayout = null;
    this.modelPipelineOpaque = null;
    this.modelPipelineTransparent = null;
    this.vertexBuffer = null;
    this.indexBuffer = null;
    this.modelUniformBuffer = null;
    this.modelTextureSampler = null;

    // Default textures
    this.defaultSRGBTexture = null;
    this.defaultSRGBTextureView = null;
    this.defaultUNormTexture = null;
    this.defaultUNormTextureView = null;
    this.defaultNormalTexture = null;
    this.defaultNormalTextureView = null;
    this.defaultCubeTexture = null;

    // Model-specific helper data 
    this.modelUpdateComplete = false; // Flag to track if updateModel() has completed

    // Meshes and Materials
    this.opaqueMeshes = [];
    this.transparentMeshes = [];
    this.materials = [];
    this.transparentMeshesDepthSorted = [];
  }

  /**
   * @brief Initializes the renderer with WebGPU resources, pipelines, and initial model/environment.
   * @param {HTMLCanvasElement} canvas - The canvas element to render to
   * @param {Object} camera - The camera object
   * @param {Object} environment - The environment object
   * @param {Object} model - The model object
   */
  async initialize(canvas, camera, environment, model) {
    // Initialize model update flag
    this.modelUpdateComplete = false;

    // Initialize WebGPU
    await this.#initWebGPU(canvas);

    this.#createDepthTexture(canvas.width, canvas.height);

    this.#createBindGroupLayouts();

    this.#createSamplers();

    this.#createRenderPassDescriptor();

    this.#createDefaultTextures();
    
    await this.#createModelRenderPipelines();
    await this.#createEnvironmentRenderPipeline();

    this.#createUniformBuffers();

    await this.updateEnvironment(environment);
    
    await this.updateModel(model);
  }

  /**
   * @brief Updates the model WebGPUresources (vertices, indices, materials, meshes).
   * @param {Object} model - The model object to render
   */
  async updateModel(model) {
    if (!model.isLoaded()) return;

    const t0 = performance.now();
    
    // Mark model update as in progress
    this.modelUpdateComplete = false;

    // Create vertex buffer
    this.#createVertexBuffer(model);

    // Create index buffer
    this.#createIndexBuffer(model);

    // Create materials with textures and bind groups
    await this.#createMaterials(model);

    // Build opaque/transparent mesh lists
    this.#createSubMeshes(model);

    // Mark model update as complete (after all async operations finish)
    this.modelUpdateComplete = true;

    console.log(`Updated Model WebGPU resources in ${(performance.now() - t0).toFixed(2)} ms`);
  }

  /**
   * @brief Updates the environment WebGPU resources (cubemap, IBL maps).
   * @param {Object} environment - The environment object
   */
  async updateEnvironment(environment) {
    if (!environment || !environment.getTexture().m_data) {
      console.warn("No environment data to process");
      return;
    }

    const t0 = performance.now();

    // Destroy the existing environment resources
    this.environmentTexture = null;
    this.environmentTextureView = null;
    this.iblIrradianceTexture = null;
    this.iblIrradianceTextureView = null;
    this.iblSpecularTexture = null;
    this.iblSpecularTextureView = null;
    this.iblBrdfIntegrationLUT = null;
    this.iblBrdfIntegrationLUTView = null;

    // Create environment textures and convert panorama to cubemap
    await this.#createEnvironmentTextures(environment);
    
    // Create global bind group with environment resources
    this.#createGlobalBindGroup();
    
    console.log(`Updated Environment WebGPU resources in ${(performance.now() - t0).toFixed(2)} ms`);
  }

  /**
   * @brief Renders a frame (environment background + model).
   * @param {Array<number>} modelMatrix - The model transformation matrix (16-element array)
   * @param {Object} cameraUniforms - Camera uniforms object with viewMatrix, projectionMatrix, and cameraPosition
   */
  render(modelMatrix, cameraUniforms) {
    // Guard against race condition: ensure updateModel() has completed
    if (!this.modelUpdateComplete) {
      return; // updateModel() is still in progress, skip rendering
    }

    // Skip rendering if pipeline is not ready
    if (!this.environmentPipeline) {
      console.warn("Environment pipeline not ready, skipping render");
      return;
    }
    if (!this.modelPipelineOpaque || !this.modelPipelineTransparent) {
      console.warn("Model pipelines not ready, skipping render");
      return;
    }

    // Update view dependent data
    this.#updateUniforms(modelMatrix, cameraUniforms);
    this.#sortTransparentMeshes(modelMatrix, cameraUniforms.viewMatrix);

    // Get current texture
    const currentTexture = this.context.getCurrentTexture();
    this.renderPassDescriptor.colorAttachments[0].view =
      currentTexture.createView();

    // Create command encoder and render pass
    const encoder = this.device.createCommandEncoder();
    const pass = encoder.beginRenderPass(
      this.renderPassDescriptor
    );

    // Set global bind group (group 0)
    pass.setBindGroup(0, this.globalBindGroup);
    
    // Render environment background first
    pass.setPipeline(this.environmentPipeline);
    pass.draw(3, 1, 0, 0); // Fullscreen triangle
    
    // Set up vertex and index buffers
    pass.setVertexBuffer(0, this.vertexBuffer);
    pass.setIndexBuffer(this.indexBuffer, "uint32");

    // Draw opaque submeshes
    pass.setPipeline(this.modelPipelineOpaque);
    for (const sm of this.opaqueMeshes) {
      pass.setBindGroup(1, this.materials[sm.materialIndex].bindGroup);
      pass.drawIndexed(sm.indexCount, 1, sm.firstIndex, 0, 0);
    }

    // Draw transparent submeshes back-to-front
    pass.setPipeline(this.modelPipelineTransparent);
    for (const depthInfo of this.transparentMeshesDepthSorted) {
      const sm = this.transparentMeshes[depthInfo.meshIndex];
      pass.setBindGroup(1, this.materials[sm.materialIndex].bindGroup);
      pass.drawIndexed(sm.indexCount, 1, sm.firstIndex, 0, 0);
    }

    // End the pass
    pass.end();

    // Submit commands
    this.device.queue.submit([encoder.finish()]);
  }

  /**
   * @brief Resizes the renderer's depth texture and updates render pass descriptor.
   * @param {number} width - New width in pixels
   * @param {number} height - New height in pixels
   */
  resize(width, height) {
    // Recreate depth texture with new size
    this.#createDepthTexture(width, height);

    // Update render pass descriptor
    this.renderPassDescriptor.depthStencilAttachment.view =
      this.depthTextureView;
  }

  // Private methods

  /**
   * @brief Creates bind group layouts for global and material resources.
   * @private
   */
  #createBindGroupLayouts() {
    // Global bind group layout (group 0) - environment and camera resources
    this.globalBindGroupLayout = this.device.createBindGroupLayout({
      entries: [
        {
          binding: 0,
          visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT,
          buffer: { type: "uniform" },
        }, // Global uniforms (camera matrices)
        {
          binding: 1,
          visibility: GPUShaderStage.FRAGMENT,
          sampler: { type: 'filtering' }
        }, // Environment sampler
        {
          binding: 2,
          visibility: GPUShaderStage.FRAGMENT,
          texture: { 
            sampleType: 'float',
            viewDimension: 'cube'
          }
        }, // Environment cubemap
        {
          binding: 3,
          visibility: GPUShaderStage.FRAGMENT,
          texture: {
            sampleType: 'float',
            viewDimension: 'cube'
          }
        }, // IBL irradiance cubemap
        {
          binding: 4,
          visibility: GPUShaderStage.FRAGMENT,
          texture: {
            sampleType: 'float',
            viewDimension: 'cube'
          }
        }, // IBL specular cubemap
        {
          binding: 5,
          visibility: GPUShaderStage.FRAGMENT,
          texture: {
            sampleType: 'float',
            viewDimension: '2d'
          }
        }, // IBL BRDF integration LUT texture
        {
          binding: 6,
          visibility: GPUShaderStage.FRAGMENT,
          sampler: { type: 'filtering' }
        }, // IBL BRDF integration LUT sampler
      ],
    });

    // Material bind group layout (group 1) - model and material resources
    this.materialBindGroupLayout = this.device.createBindGroupLayout({
      entries: [
        {
          binding: 0,
          visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT,
          buffer: { type: "uniform" },
        }, // Model uniforms
        {
          binding: 1,
          visibility: GPUShaderStage.FRAGMENT,
          buffer: { type: "uniform" },
        }, // Material uniforms
        { binding: 2, visibility: GPUShaderStage.FRAGMENT, sampler: {} }, // Sampler
        { binding: 3, visibility: GPUShaderStage.FRAGMENT, texture: {} }, // BaseColor
        { binding: 4, visibility: GPUShaderStage.FRAGMENT, texture: {} }, // MetallicRoughness
        { binding: 5, visibility: GPUShaderStage.FRAGMENT, texture: {} }, // Normal
        { binding: 6, visibility: GPUShaderStage.FRAGMENT, texture: {} }, // Occlusion
        { binding: 7, visibility: GPUShaderStage.FRAGMENT, texture: {} }, // Emissive
      ],
    });
  }

  /**
   * @brief Creates the global bind group with environment and IBL resources.
   * @private
   */
  #createGlobalBindGroup() {
    if (!this.globalBindGroupLayout || !this.globalUniformBuffer) {
      console.warn("Cannot create global bind group: missing layout or uniform buffer");
      return;
    }

    // Use environment resources if available, otherwise use fallbacks
    const envTexture = this.environmentTextureView || this.defaultCubeTexture.createView();
    
    // Use IBL resources if available, otherwise use fallbacks
    const iblIrradianceTexture = this.iblIrradianceTextureView || this.defaultCubeTexture.createView();
    const iblSpecularTexture = this.iblSpecularTextureView || this.defaultCubeTexture.createView();
    const iblBrdfLUTTexture = this.iblBrdfIntegrationLUTView || this.defaultUNormTexture.createView();

    this.globalBindGroup = this.device.createBindGroup({
      layout: this.globalBindGroupLayout,
      entries: [
        { binding: 0, resource: { buffer: this.globalUniformBuffer } },
        { binding: 1, resource: this.environmentCubeSampler },
        { binding: 2, resource: envTexture },
        { binding: 3, resource: iblIrradianceTexture },
        { binding: 4, resource: iblSpecularTexture },
        { binding: 5, resource: iblBrdfLUTTexture },
        { binding: 6, resource: this.iblBrdfIntegrationLUTSampler },
      ],
    });
  }

  /**
   * @brief Calculates the largest power of 2 less than or equal to x.
   * @param {number} x - Input value
   * @returns {number} Largest power of 2 <= x
   * @private
   */
  #floorPow2(x) {
    let power = 1;
    while (power * 2 <= x) {
      power *= 2;
    }
    return power;
  }

  /**
   * @brief Creates an environment cubemap texture and view.
   * @param {Array<number>} size - Texture size [width, height, depthOrArrayLayers]
   * @param {boolean} mipmapping - Whether to enable mipmapping
   * @returns {Object} Object with texture and textureView properties
   * @private
   */
  #createEnvironmentTexture(size, mipmapping = false) {
    // Calculate mip levels if mipmapping enabled
    const mipLevelCount = mipmapping ? 
      Math.floor(Math.log2(Math.max(size[0], size[1]))) + 1 : 1;

    // Create texture with rgba16float format
    const texture = this.device.createTexture({
      size: size, // [width, height, depthOrArrayLayers]
      format: 'rgba16float',
      usage: GPUTextureUsage.TEXTURE_BINDING | 
             GPUTextureUsage.STORAGE_BINDING |
             GPUTextureUsage.COPY_DST |
             GPUTextureUsage.COPY_SRC,
      mipLevelCount: mipLevelCount,
    });

    // Create appropriate texture view
    const dimension = size[2] === 6 ? 'cube' : '2d';
    const textureView = texture.createView({
      format: 'rgba16float',
      dimension: dimension,
      mipLevelCount: mipLevelCount,
      arrayLayerCount: size[2],
    });

    return { texture, textureView };
  }

  /**
   * @brief Creates a sampler for environment cubemap textures.
   * @returns {GPUSampler} The created sampler
   * @private
   */
  #createEnvironmentSampler() {
    return this.device.createSampler({
      addressModeU: 'repeat',
      addressModeV: 'repeat', 
      addressModeW: 'repeat',
      minFilter: 'linear',
      magFilter: 'linear',
      mipmapFilter: 'linear',
    });
  }

  /**
   * @brief Creates environment textures and IBL maps from panorama.
   * @param {Object} environment - The environment object
   * @private
   */
  async #createEnvironmentTextures(environment) {
    // Create helpers
    const mipmapGenerator = new MipmapGenerator(this.device);
    const panoramaConverter = new PanoramaToCubemapConverter(this.device);
    const environmentPreprocessor = new EnvironmentPreprocessor(this.device);

    // Get panorama texture from environment
    const panoramaTexture = environment.getTexture();
    
    // Calculate environment cubemap size (power of 2, based on panorama width)
    const environmentCubeSize = this.#floorPow2(panoramaTexture.m_width);
    
    // Create environment cubemap texture with mipmaps
    const envResult = this.#createEnvironmentTexture([environmentCubeSize, environmentCubeSize, 6], true);
    this.environmentTexture = envResult.texture;
    this.environmentTextureView = envResult.textureView;
    
    // Convert panorama to cubemap
    await panoramaConverter.uploadAndConvert(panoramaTexture, this.environmentTexture);
    
    // Generate mipmaps for the environment cubemap
    await mipmapGenerator.generateMipmaps(
      this.environmentTexture,
      { width: environmentCubeSize, height: environmentCubeSize },
      MipKind.Float16Cube
    );
    
    // Create IBL textures
    const irradianceResult = this.#createEnvironmentTexture([kIrradianceMapSize, kIrradianceMapSize, 6], true);
    this.iblIrradianceTexture = irradianceResult.texture;
    this.iblIrradianceTextureView = irradianceResult.textureView;
    
    const specularResult = this.#createEnvironmentTexture([kPrecomputedSpecularMapSize, kPrecomputedSpecularMapSize, 6], true);
    this.iblSpecularTexture = specularResult.texture;
    this.iblSpecularTextureView = specularResult.textureView;
    
    const brdfLUTResult = this.#createEnvironmentTexture([kBRDFIntegrationLUTMapSize, kBRDFIntegrationLUTMapSize, 1], false);
    this.iblBrdfIntegrationLUT = brdfLUTResult.texture;
    this.iblBrdfIntegrationLUTView = brdfLUTResult.textureView;
    
    // Precompute IBL maps
    await environmentPreprocessor.generateMaps(
      this.environmentTexture,
      this.iblIrradianceTexture,
      this.iblSpecularTexture,
      this.iblBrdfIntegrationLUT
    );
    
    // Generate mipmaps for irradiance texture
    await mipmapGenerator.generateMipmaps(
      this.iblIrradianceTexture,
      { width: kIrradianceMapSize, height: kIrradianceMapSize },
      MipKind.Float16Cube
    );
  }

  /**
   * @brief Creates the render pipeline for environment background rendering.
   * @private
   */
  async #createEnvironmentRenderPipeline() {
    try {
      // Load environment shader
      const shaderCode = await this.#loadShaderFile("./shaders/environment.wgsl");
      this.environmentShaderModule = this.device.createShaderModule({ 
        code: shaderCode 
      });

      // Create pipeline layout using the shared global bind group layout
      const pipelineLayout = this.device.createPipelineLayout({
        bindGroupLayouts: [this.globalBindGroupLayout],
      });

      // Create environment render pipeline
      this.environmentPipeline = this.device.createRenderPipeline({
        layout: pipelineLayout,
        vertex: {
          module: this.environmentShaderModule,
          entryPoint: "vs_main",
          // No vertex buffers - vertices generated in shader
        },
        fragment: {
          module: this.environmentShaderModule,
          entryPoint: "fs_main",
          targets: [{ format: this.format }],
        },
        primitive: {
          topology: "triangle-list",
        },
        depthStencil: {
          format: "depth24plus-stencil8",
          depthWriteEnabled: false, // Don't write depth for environment background
          depthCompare: "less-equal",
        },
      });

    } catch (error) {
      console.error("Failed to create environment pipeline:", error);
      throw error;
    }
  }

  /**
   * @brief Initializes WebGPU adapter, device, and canvas context.
   * @param {HTMLCanvasElement} canvas - The canvas element
   * @private
   */
  async #initWebGPU(canvas) {
    // Get adapter
    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter) {
      throw new Error("WebGPU adapter not available.");
    }

    // Get device
    this.device = await adapter.requestDevice();
    if (!this.device) {
      throw new Error("Failed to create WebGPU device.");
    }

    // Configure canvas context
    this.format = navigator.gpu.getPreferredCanvasFormat();
    this.context = canvas.getContext("webgpu");
    this.context.configure({
      device: this.device,
      format: this.format,
      alphaMode: "opaque",
    });
  }

  /**
   * @brief Creates or recreates the depth texture with the specified dimensions.
   * @param {number} width - Texture width in pixels
   * @param {number} height - Texture height in pixels
   * @private
   */
  #createDepthTexture(width, height) {
    if (this.depthTexture) {
      this.depthTexture.destroy();
    }

    this.depthTexture = this.device.createTexture({
      size: [width, height, 1],
      format: "depth24plus-stencil8",
      usage: GPUTextureUsage.RENDER_ATTACHMENT,
    });

    this.depthTextureView = this.depthTexture.createView();
  }

  /**
   * @brief Creates render pipelines for opaque and transparent model rendering.
   * @private
   */
  async #createModelRenderPipelines() {
    // Load shader code from file
    const shaderCode = await this.#loadShaderFile("./shaders/gltf_pbr.wgsl");

    // Create shader module
    const shaderModule = this.device.createShaderModule({ code: shaderCode });

    // Create pipeline layout
    const pipelineLayout = this.device.createPipelineLayout({
      bindGroupLayouts: [this.globalBindGroupLayout, this.materialBindGroupLayout],
    });

    // Base vertex state
    const vertexState = {
      module: shaderModule,
      entryPoint: "vs_main",
      buffers: [
        {
          arrayStride: 18 * Float32Array.BYTES_PER_ELEMENT, // Full vertex: 18 floats
          attributes: [
            { shaderLocation: 0, format: "float32x3", offset: 0 }, // Position
            { shaderLocation: 1, format: "float32x3", offset: 3 * Float32Array.BYTES_PER_ELEMENT }, // Normal
            { shaderLocation: 2, format: "float32x4", offset: 6 * Float32Array.BYTES_PER_ELEMENT }, // Tangent
            { shaderLocation: 3, format: "float32x2", offset: 10 * Float32Array.BYTES_PER_ELEMENT }, // TexCoord0
            { shaderLocation: 4, format: "float32x2", offset: 12 * Float32Array.BYTES_PER_ELEMENT }, // TexCoord1
            { shaderLocation: 5, format: "float32x4", offset: 14 * Float32Array.BYTES_PER_ELEMENT }, // Color
          ],
        },
      ],
    };

    // Opaque pipeline
    this.modelPipelineOpaque = this.device.createRenderPipeline({
      vertex: vertexState,
      fragment: {
        module: shaderModule,
        entryPoint: "fs_main",
        targets: [{ format: this.format }],
      },
      primitive: { topology: "triangle-list" },
      depthStencil: {
        format: "depth24plus-stencil8",
        depthWriteEnabled: true,
        depthCompare: "less-equal",
      },
      layout: pipelineLayout,
    });

    // Transparent pipeline (alpha blending, no depth writes)
    const blendComponent = {
      operation: "add",
      srcFactor: "src-alpha",
      dstFactor: "one-minus-src-alpha",
    };
    const blendState = { color: blendComponent, alpha: blendComponent };

    this.modelPipelineTransparent = this.device.createRenderPipeline({
      vertex: vertexState,
      fragment: {
        module: shaderModule,
        entryPoint: "fs_main",
        targets: [{ format: this.format, blend: blendState }],
      },
      primitive: { topology: "triangle-list" },
      depthStencil: {
        format: "depth24plus-stencil8",
        depthWriteEnabled: false,
        depthCompare: "less-equal",
      },
      layout: pipelineLayout,
    });
  }

  /**
   * @brief Creates uniform buffers for global and model uniforms.
   * @private
   */
  #createUniformBuffers() {
    // Global uniforms: 5 mat4 + vec3 + padding = 5*64 + 16 = 336 bytes, round up to 512 for alignment
    this.globalUniformBuffer = this.device.createBuffer({
      size: 512, // viewMatrix, projectionMatrix, inverseViewMatrix, inverseProjectionMatrix, cameraPosition
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    // Model uniforms: 2 mat4 = 2*64 = 128 bytes, round up to 256 for alignment
    this.modelUniformBuffer = this.device.createBuffer({
      size: 256, // modelMatrix, normalMatrix
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    // Per-material uniform buffers created later in #createMaterialBindGroups
  }

  /**
   * @brief Creates materials with textures, uniform buffers, and bind groups.
   * @param {Object} model - The model object
   * @private
   */
  async #createMaterials(model) {
    // Create mipmap generator helper
    const mipmapGenerator = new MipmapGenerator(this.device);

    this.materials = [];
    if (!this.materialBindGroupLayout) return; // layout must exist

    const srcMaterials = model.getMaterials();
    if (!srcMaterials || srcMaterials.length === 0) return;

    for (let i = 0; i < srcMaterials.length; i++) {
      const srcMat = srcMaterials[i];
      const material = {
        uniforms: {
          baseColorFactor: srcMat.baseColorFactor,
          emissiveFactor: srcMat.emissiveFactor,
          metallicFactor: srcMat.metallicFactor,
          roughnessFactor: srcMat.roughnessFactor,
          normalScale: srcMat.normalScale,
          occlusionStrength: srcMat.occlusionStrength,
          alphaCutoff: srcMat.alphaCutoff,
          alphaMode: srcMat.alphaMode,
        },
        uniformBuffer: null,
        baseColorTexture: null,
        metallicRoughnessTexture: null,
        normalTexture: null,
        occlusionTexture: null,
        emissiveTexture: null,
        bindGroup: null,
      };

      // Create per-material uniform buffer
      const ub = this.device.createBuffer({
        size: 256,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      });
      const data = new Float32Array(16);
      let o = 0;
      data.set(material.uniforms.baseColorFactor, o);
      o += 4;
      data[o++] = material.uniforms.emissiveFactor[0];
      data[o++] = material.uniforms.emissiveFactor[1];
      data[o++] = material.uniforms.emissiveFactor[2];
      data[o++] = material.uniforms.alphaMode;
      data[o++] = material.uniforms.metallicFactor;
      data[o++] = material.uniforms.roughnessFactor;
      data[o++] = material.uniforms.normalScale;
      data[o++] = material.uniforms.occlusionStrength;
      data[o++] = material.uniforms.alphaCutoff;
      while (o < 16) data[o++] = 0.0;
      this.device.queue.writeBuffer(ub, 0, data.buffer);
      material.uniformBuffer = ub;

      // Create textures for this material
      // Base Color Texture
      if (srcMat.baseColorTexture !== undefined && srcMat.baseColorTexture >= 0) {
        const texture = model.getTexture(srcMat.baseColorTexture);
        if (texture && texture.image) {
          // Set type for proper format/mip handling
          const textureWithType = { ...texture, type: "baseColor" };
          material.baseColorTexture = await this.#createModelTexture(textureWithType, mipmapGenerator);
        }
      }
      if (!material.baseColorTexture) {
        material.baseColorTexture = this.defaultSRGBTexture;
      }

      // Metallic-Roughness Texture
      if (srcMat.metallicRoughnessTexture !== undefined && srcMat.metallicRoughnessTexture >= 0) {
        const texture = model.getTexture(srcMat.metallicRoughnessTexture);
        if (texture && texture.image) {
          const textureWithType = { ...texture, type: "metallicRoughness" };
          material.metallicRoughnessTexture = await this.#createModelTexture(textureWithType, mipmapGenerator);
        }
      }
      if (!material.metallicRoughnessTexture) {
        material.metallicRoughnessTexture = this.defaultUNormTexture;
      }

      // Normal Texture
      if (srcMat.normalTexture !== undefined && srcMat.normalTexture >= 0) {
        const texture = model.getTexture(srcMat.normalTexture);
        if (texture && texture.image) {
          const textureWithType = { ...texture, type: "normal" };
          material.normalTexture = await this.#createModelTexture(textureWithType, mipmapGenerator);
        }
      }
      if (!material.normalTexture) {
        material.normalTexture = this.defaultNormalTexture;
      }

      // Occlusion Texture
      if (srcMat.occlusionTexture !== undefined && srcMat.occlusionTexture >= 0) {
        const texture = model.getTexture(srcMat.occlusionTexture);
        if (texture && texture.image) {
          const textureWithType = { ...texture, type: "occlusion" };
          material.occlusionTexture = await this.#createModelTexture(textureWithType, mipmapGenerator);
        }
      }
      if (!material.occlusionTexture) {
        material.occlusionTexture = this.defaultUNormTexture;
      }

      // Emissive Texture
      if (srcMat.emissiveTexture !== undefined && srcMat.emissiveTexture >= 0) {
        const texture = model.getTexture(srcMat.emissiveTexture);
        if (texture && texture.image) {
          const textureWithType = { ...texture, type: "emissive" };
          material.emissiveTexture = await this.#createModelTexture(textureWithType, mipmapGenerator);
        }
      }
      if (!material.emissiveTexture) {
        material.emissiveTexture = this.defaultSRGBTexture;
      }

      // Create bind group
      const bg = this.device.createBindGroup({
        layout: this.materialBindGroupLayout,
        entries: [
          { binding: 0, resource: { buffer: this.modelUniformBuffer } }, // Model uniforms
          { binding: 1, resource: { buffer: material.uniformBuffer } }, // Material uniforms
          { binding: 2, resource: this.modelTextureSampler }, // Model texture sampler
          { binding: 3, resource: material.baseColorTexture.createView() }, // BaseColor
          { binding: 4, resource: material.metallicRoughnessTexture.createView() }, // MetallicRoughness
          { binding: 5, resource: material.normalTexture.createView() }, // Normal
          { binding: 6, resource: material.occlusionTexture.createView() }, // Occlusion
          { binding: 7, resource: material.emissiveTexture.createView() }, // Emissive
        ],
      });
      material.bindGroup = bg;

      this.materials.push(material);
    }
  }

  /**
   * @brief Create a model texture with mipmaps using the appropriate MipKind.
   * @param {Object} texture - Texture info with image, width, height, and type.
   * @param {MipmapGenerator} mipmapGenerator - Mipmap generator instance.
   * @returns {Promise<GPUTexture>} The created WebGPU texture with mipmaps.
   * @private
   */
  async #createModelTexture(texture, mipmapGenerator) {
    const { image, width, height, type } = texture;

    // Calculate mip level count
    const mipLevelCount = 1 + Math.floor(Math.log2(Math.max(width, height)));

    // Determine format and mip kind based on texture type
    const needsSrgb = type === "baseColor" || type === "emissive";
    const isNormal = type === "normal";
    
    let mipKind;
    if (needsSrgb) {
      mipKind = MipKind.SRGB2D;
    } else if (isNormal) {
      mipKind = MipKind.Normal2D;
    } else {
      mipKind = MipKind.LinearUNorm2D;
    }

    if (mipKind === MipKind.SRGB2D) {
      // sRGB textures: create directly with render attachment usage for render-based mipmaps
      const finalTexture = this.device.createTexture({
        size: [width, height, 1],
        format: 'rgba8unorm-srgb',
        usage: GPUTextureUsage.TEXTURE_BINDING | 
               GPUTextureUsage.RENDER_ATTACHMENT | 
               GPUTextureUsage.COPY_DST,
        mipLevelCount,
      });

      // Upload level 0
      this.device.queue.copyExternalImageToTexture(
        { source: image },
        { texture: finalTexture },
        [width, height]
      );

      // Generate mipmaps via render path
      await mipmapGenerator.generateMipmaps(
        finalTexture,
        { width, height },
        MipKind.SRGB2D
      );

      return finalTexture;
    } else {
      // Non-sRGB textures: use compute-based mipmaps via intermediate texture
      // Create intermediate RGBA8Unorm texture with storage binding for compute mipmaps
      // Note: RENDER_ATTACHMENT is required for copyExternalImageToTexture
      const intermediateTexture = this.device.createTexture({
        size: [width, height, 1],
        format: 'rgba8unorm',
        usage: GPUTextureUsage.TEXTURE_BINDING | 
               GPUTextureUsage.STORAGE_BINDING | 
               GPUTextureUsage.COPY_DST | 
               GPUTextureUsage.COPY_SRC |
               GPUTextureUsage.RENDER_ATTACHMENT,
        mipLevelCount,
      });

      // Upload level 0
      this.device.queue.copyExternalImageToTexture(
        { source: image },
        { texture: intermediateTexture },
        [width, height]
      );

      // Generate mipmaps via compute (normal-aware or linear)
      await mipmapGenerator.generateMipmaps(
        intermediateTexture,
        { width, height },
        mipKind
      );

      // Copy into a final texture without STORAGE usage for sampling
      const finalTexture = this.device.createTexture({
        size: [width, height, 1],
        format: 'rgba8unorm',
        usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
        mipLevelCount,
      });

      const encoder = this.device.createCommandEncoder();
      for (let level = 0; level < mipLevelCount; level++) {
        const mipWidth = Math.max(width >> level, 1);
        const mipHeight = Math.max(height >> level, 1);
        encoder.copyTextureToTexture(
          { texture: intermediateTexture, mipLevel: level },
          { texture: finalTexture, mipLevel: level },
          { width: mipWidth, height: mipHeight, depthOrArrayLayers: 1 }
        );
      }
      this.device.queue.submit([encoder.finish()]);

      return finalTexture;
    }
  }

  /**
   * @brief Creates samplers for model textures, environment cubemap, and BRDF LUT.
   * @private
   */
  #createSamplers() {
    // Model textures sampler
    this.modelTextureSampler = this.device.createSampler({
      magFilter: "linear",
      minFilter: "linear",
      mipmapFilter: "linear",
      addressModeU: "repeat",
      addressModeV: "repeat",
    });

    // Environment cube sampler
    this.environmentCubeSampler = this.#createEnvironmentSampler();

    // BRDF LUT sampler
    this.iblBrdfIntegrationLUTSampler = this.device.createSampler({
      addressModeU: 'clamp-to-edge',
      addressModeV: 'clamp-to-edge',
      addressModeW: 'clamp-to-edge',
      minFilter: 'linear',
      magFilter: 'linear',
      mipmapFilter: 'nearest',
    });
  }

  /**
   * @brief Creates default fallback textures (sRGB, UNORM, normal, cube).
   * @private
   */
  #createDefaultTextures() {
    // 1x1 white sRGB texture (base color/emissive default)
    {
      this.defaultSRGBTexture = this.device.createTexture({
        size: [1, 1, 1],
        format: "rgba8unorm-srgb",
        usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
      });
      const whitePixel = new Uint8Array([255, 255, 255, 255]);
      this.device.queue.writeTexture(
        { texture: this.defaultSRGBTexture },
        whitePixel,
        { bytesPerRow: 4 },
        { width: 1, height: 1 }
      );
    }

    // 1x1 white UNORM texture (MR/Occlusion default)
    {
      this.defaultUNormTexture = this.device.createTexture({
        size: [1, 1, 1],
        format: "rgba8unorm",
        usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
      });
      const whitePixel = new Uint8Array([255, 255, 255, 255]);
      this.device.queue.writeTexture(
        { texture: this.defaultUNormTexture },
        whitePixel,
        { bytesPerRow: 4 },
        { width: 1, height: 1 }
      );
    }

    // 1x1 UNORM flat normal texture (normal default)
    {
      this.defaultNormalTexture = this.device.createTexture({
        size: [1, 1, 1],
        format: "rgba8unorm",
        usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
      });
      const flatNormal = new Uint8Array([128, 128, 255, 255]);
      this.device.queue.writeTexture(
        { texture: this.defaultNormalTexture },
        flatNormal,
        { bytesPerRow: 4 },
        { width: 1, height: 1 }
      );
    }

    // 1x1x6 white cube texture (environment fallback)
    {
      this.defaultCubeTexture = this.device.createTexture({
        size: [1, 1, 6],
        format: "rgba8unorm",
        usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
      });
      const whitePixel = new Uint8Array([255, 255, 255, 255]);
      for (let face = 0; face < 6; face++) {
        this.device.queue.writeTexture(
          { texture: this.defaultCubeTexture, origin: { x: 0, y: 0, z: face } },
          whitePixel,
          { bytesPerRow: 4 },
          { width: 1, height: 1, depthOrArrayLayers: 1 }
        );
      }
    }
  }

  /**
   * @brief Updates all uniform buffers (global and model).
   * @param {Array<number>} modelMatrix - The model transformation matrix (16-element array)
   * @param {Object} cameraUniforms - Camera uniforms object with viewMatrix, projectionMatrix, and cameraPosition
   * @private
   */
  #updateUniforms(modelMatrix, cameraUniforms) {
    this.#updateGlobalUniforms(cameraUniforms);
    this.#updateModelUniforms(modelMatrix);
  }

  /**
   * @brief Updates the global uniform buffer with camera matrices and position.
   * @param {Object} cameraUniforms - Camera uniforms object with viewMatrix, projectionMatrix, and cameraPosition
   * @private
   */
  #updateGlobalUniforms(cameraUniforms) {
    // Global uniforms: camera matrices and position only
    const globalData = new Float32Array(80); // 5 matrices * 16 floats + 4 floats for camera pos

    const viewMatrix = cameraUniforms.viewMatrix;
    const projectionMatrix = cameraUniforms.projectionMatrix;
    const cameraPos = cameraUniforms.cameraPosition;

    // Compute inverse matrices
    const inverseViewMatrix = mat4.create();
    mat4.invert(inverseViewMatrix, viewMatrix);
    
    const inverseProjectionMatrix = mat4.create();
    mat4.invert(inverseProjectionMatrix, projectionMatrix);

    // Pack matrices (each matrix is 16 floats)
    globalData.set(viewMatrix, 0);                    // offset 0-15
    globalData.set(projectionMatrix, 16);             // offset 16-31  
    globalData.set(inverseViewMatrix, 32);            // offset 32-47
    globalData.set(inverseProjectionMatrix, 48);      // offset 48-63
    globalData.set([cameraPos[0], cameraPos[1], cameraPos[2], 0.0], 64); // offset 64-67

    // Upload to GPU
    this.device.queue.writeBuffer(this.globalUniformBuffer, 0, globalData);
  }

  /**
   * @brief Updates the model uniform buffer with model and normal matrices.
   * @param {Array<number>} modelMatrix - The model transformation matrix (16-element array)
   * @private
   */
  #updateModelUniforms(modelMatrix) {
    // Model uniforms: model transform matrices only
    const modelData = new Float32Array(32); // 2 matrices * 16 floats
    
    // Compute normal matrix: inverse transpose of upper 3x3 of model matrix
    const modelMatrix3x3 = [
      modelMatrix[0], modelMatrix[1], modelMatrix[2],
      modelMatrix[4], modelMatrix[5], modelMatrix[6], 
      modelMatrix[8], modelMatrix[9], modelMatrix[10],
    ];

    // Create 3x3 matrix, invert it, then transpose
    const temp3x3 = mat4.create();
    mat4.set(
      temp3x3,
      modelMatrix3x3[0], modelMatrix3x3[1], modelMatrix3x3[2], 0,
      modelMatrix3x3[3], modelMatrix3x3[4], modelMatrix3x3[5], 0,
      modelMatrix3x3[6], modelMatrix3x3[7], modelMatrix3x3[8], 0,
      0, 0, 0, 1
    );

    const normalMatrix3x3 = mat4.create();
    mat4.transpose(normalMatrix3x3, mat4.invert(normalMatrix3x3, temp3x3));

    // Convert back to 4x4 with identity for the 4th row/column
    const normalMatrix = mat4.create();
    mat4.identity(normalMatrix);
    normalMatrix[0] = normalMatrix3x3[0];
    normalMatrix[1] = normalMatrix3x3[1];
    normalMatrix[2] = normalMatrix3x3[2];
    normalMatrix[4] = normalMatrix3x3[4];
    normalMatrix[5] = normalMatrix3x3[5];
    normalMatrix[6] = normalMatrix3x3[6];
    normalMatrix[8] = normalMatrix3x3[8];
    normalMatrix[9] = normalMatrix3x3[9];
    normalMatrix[10] = normalMatrix3x3[10];

    // Pack matrices
    modelData.set(modelMatrix, 0);    // offset 0-15
    modelData.set(normalMatrix, 16);  // offset 16-31

    // Upload to GPU
    this.device.queue.writeBuffer(this.modelUniformBuffer, 0, modelData);
  }

  /**
   * @brief Creates submesh lists (opaque and transparent) from model data.
   * @param {Object} model - The model object
   * @private
   */
  #createSubMeshes(model) {
    this.opaqueMeshes = [];
    this.transparentMeshes = [];
    const subMeshes = model.getSubMeshes();
    const materials = model.getMaterials();
    if (!subMeshes || !materials) return;

    for (const sm of subMeshes) {
      const mat = materials[sm.materialIndex];
      const centroid = [
        (sm.minBounds[0] + sm.maxBounds[0]) * 0.5,
        (sm.minBounds[1] + sm.maxBounds[1]) * 0.5,
        (sm.minBounds[2] + sm.maxBounds[2]) * 0.5,
      ];
      const dst = {
        firstIndex: sm.firstIndex,
        indexCount: sm.indexCount,
        materialIndex: sm.materialIndex,
        centroid: centroid,
      };
      if (mat && mat.alphaMode === 2) {
        this.transparentMeshes.push(dst);
      } else {
        this.opaqueMeshes.push(dst);
      }
    }
  }

  /**
   * @brief Sorts transparent meshes by depth for back-to-front rendering.
   * @param {Array<number>} modelMatrix - Model transformation matrix (16-element array)
   * @param {Array<number>} viewMatrix - View transformation matrix (16-element array)
   * @private
   */
  #sortTransparentMeshes(modelMatrix, viewMatrix) {
    const modelView = mat4.create();
    mat4.multiply(modelView, viewMatrix, modelMatrix);
  
    // Attempt to reuse array and objects from previous frame
    const sorted = this.transparentMeshesDepthSorted;
  
    let count = 0;
    for (let i = 0; i < this.transparentMeshes.length; i++) {
      const sm = this.transparentMeshes[i];
      const c = sm.centroid;
      const centroidWorld = vec4.fromValues(c[0], c[1], c[2], 1.0);
      const centroidView = vec4.create();
      vec4.transformMat4(centroidView, centroidWorld, modelView);
      const depth = centroidView[2];
  
      if (depth < 0.0) {
        let entry = sorted[count];
        if (!entry) { // More entries than previous frame, create new object
          entry = {};
          sorted[count] = entry;
        }
        entry.depth = depth;
        entry.meshIndex = i;
        count++;
      }
    }
  
    sorted.length = count; // truncate if fewer than previous frame
    sorted.sort((a, b) => a.depth - b.depth);
  }  

  /**
   * @brief Creates the vertex buffer from model vertex data.
   * @param {Object} model - The model object
   * @private
   */
  #createVertexBuffer(model) {
    if (this.vertexBuffer) {
      this.vertexBuffer.destroy();
    }

    const vertexData = model.getVertices();

    this.vertexBuffer = this.device.createBuffer({
      size: vertexData.byteLength,
      usage: GPUBufferUsage.VERTEX,
      mappedAtCreation: true,
    });

    new Float32Array(this.vertexBuffer.getMappedRange()).set(vertexData);
    this.vertexBuffer.unmap();
  }

  /**
   * @brief Creates the index buffer from model index data.
   * @param {Object} model - The model object
   * @private
   */
  #createIndexBuffer(model) {
    if (this.indexBuffer) {
      this.indexBuffer.destroy();
    }

    const indices = model.getIndices();

    if (indices.length > 0) {
      this.indexBuffer = this.device.createBuffer({
        size: indices.length * 4, // Always use 32-bit (4 bytes per index)
        usage: GPUBufferUsage.INDEX | GPUBufferUsage.COPY_DST,
        mappedAtCreation: true,
      });

      // Convert to 32-bit typed array
      const typedIndices = new Uint32Array(indices);
      new Uint32Array(this.indexBuffer.getMappedRange()).set(typedIndices);
      this.indexBuffer.unmap();
    }
  }

  /**
   * @brief Creates the render pass descriptor for color and depth attachments.
   * @private
   */
  #createRenderPassDescriptor() {
    this.renderPassDescriptor = {
      colorAttachments: [
        {
          view: undefined, // Assigned per-frame
          loadOp: "clear",
          clearValue: { r: 0.0, g: 0.2, b: 0.4, a: 1.0 },
          storeOp: "store",
        },
      ],
      depthStencilAttachment: {
        view: this.depthTextureView,
        depthLoadOp: "clear",
        depthClearValue: 1.0,
        depthStoreOp: "store",
        stencilLoadOp: "clear",
        stencilClearValue: 0,
        stencilStoreOp: "store",
      },
    };
  }

  /**
   * @brief Reloads shaders from disk and recreates pipelines.
   */
  async reloadShaders() {
    console.log("Destroying existing shader resources...");

    // Store references to current pipelines in case reload fails
    const oldModelPipelineOpaque = this.modelPipelineOpaque;
    const oldModelPipelineTransparent = this.modelPipelineTransparent;
    const oldEnvPipeline = this.environmentPipeline;
    const oldModelShaderModule = this.shaderModule;
    const oldEnvShaderModule = this.environmentShaderModule;

    try {
      // Clear current pipeline and shader module
      this.modelPipelineOpaque = null;
      this.modelPipelineTransparent = null;
      this.environmentPipeline = null;
      this.shaderModule = null;
      this.environmentShaderModule = null;

      console.log("Loading shader from disk...");
      
      // Recreate the model pipeline with new shader code
      await this.#createModelRenderPipelines();
      // Recreate the environment pipeline with new shader code
      await this.#createEnvironmentRenderPipeline();
      
      console.log("Shader pipeline recreated successfully!");
    } catch (error) {
      console.error("Failed to reload shaders, restoring previous pipeline:", error);
      
      // Restore previous pipelines if reload failed
      this.modelPipelineOpaque = oldModelPipelineOpaque;
      this.modelPipelineTransparent = oldModelPipelineTransparent;
      this.environmentPipeline = oldEnvPipeline;
      this.shaderModule = oldModelShaderModule;
      this.environmentShaderModule = oldEnvShaderModule;
      
      throw error; // Re-throw so caller knows it failed
    }
  }

  /**
   * @brief Loads a shader file from disk with cache-busting.
   * @param {string} path - Path to the shader file
   * @returns {Promise<string>} The shader source code
   * @private
   */
  async #loadShaderFile(path) {
    try {
      // Add cache-busting parameter to force reload from disk
      const cacheBuster = `?t=${Date.now()}`;
      const response = await fetch(path + cacheBuster, {
        cache: 'no-cache'
      });
      if (!response.ok) {
        throw new Error(
          `Failed to load shader: ${response.status} ${response.statusText}`
        );
      }
      return await response.text();
    } catch (error) {
      console.error(`Error loading shader file ${path}:`, error);
      throw error;
    }
  }
}
