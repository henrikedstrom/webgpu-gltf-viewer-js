const { mat4 } = glMatrix;
import PanoramaToCubemapConverter from './PanoramaToCubemapConverter.js';
import MipmapGenerator, { MipKind } from './MipmapGenerator.js';

export default class Renderer {
  constructor() {
    // WebGPU resources
    this.device = null;
    this.context = null;
    this.format = null;

    // Rendering resources
    this.depthTexture = null;
    this.depthTextureView = null;
    this.pipeline = null;
    this.globalUniformBuffer = null;
    this.modelUniformBuffer = null;
    this.sampler = null;
    this.defaultTexture = null;
    this.defaultNormalTexture = null;
    this.defaultCubeTexture = null;
    // Per-material resources
    this.materialBindGroups = []; // bind group per material
    this.materialUniformBuffers = []; // uniform buffer per material

    // Environment resources
    this.environmentTexture = null;
    this.environmentTextureView = null;
    this.environmentCubeSampler = null;
    this.environmentShaderModule = null;
    this.environmentPipeline = null;
    this.panoramaConverter = null;
    this.mipmapGenerator = null;

    // Bind group layouts
    this.globalBindGroupLayout = null;
    this.materialBindGroupLayout = null;
    this.globalBindGroup = null;

    // Model-specific resources
    this.vertexBuffer = null;
    this.indexBuffer = null;
    this.indices = null;
    this.vertexData = null;
    this.modelTextures = [];
    this.modelTextureTypes = [];

    // Render pass descriptor
    this.renderPassDescriptor = null;

    // Canvas dimensions
    this.width = 0;
    this.height = 0;
  }

  async initialize(canvas, camera, environment, model) {
    // Store references to dependencies
    this.camera = camera;
    this.environment = environment;
    this.model = model;
    
    // Store canvas dimensions
    this.width = canvas.width;
    this.height = canvas.height;

    // Initialize WebGPU
    await this.#initWebGPU(canvas);

    // Create depth buffer
    this.#createDepthTexture();

    // Create resources first
    this.mipmapGenerator = new MipmapGenerator(this.device);
    this.#createDefaultTexture();
    this.#createDefaultCubeTexture();
    this.#createSampler();

    // Create rendering pipeline
    await this.#createPipeline();

    // Create uniform buffers (after pipeline, so we can create bind groups)
    this.#createUniformBuffers();

    // Create environment resources
    await this.updateEnvironment(environment);
    
    // Create initial global bind group
    if (!this.globalBindGroup) {
      this.#createGlobalBindGroup();
    }

    // Setup model resources
    await this.updateModel(model);

    // Create render pass descriptor
    this.#createRenderPassDescriptor();
  }

  async updateModel(model) {
    if (!model.isLoaded()) return;

    const t0 = performance.now();
   
    // Store the new model reference
    this.model = model;

    // Get model data
    this.vertexData = model.getVertices();
    this.indices = model.getIndices();

    // Create vertex buffer
    this.#createVertexBuffer();

    // Create index buffer
    this.#createIndexBuffer();

    // Load model textures then build per-material bind groups
    await this.#loadModelTextures(model);
    this.#createMaterialBindGroups(model);

    console.log(`Updated Model WebGPU resources in ${(performance.now() - t0).toFixed(2)} ms`);
  }

  async updateEnvironment(environment) {
    if (!environment || !environment.getTexture().m_data) {
      console.warn("No environment data to process");
      return;
    }
    // Store the new environment reference
    this.environment = environment;

    try {
      const t0 = performance.now();

      // Create environment textures and convert panorama to cubemap
      await this.#createEnvironmentTexturesAndSamplers();
      
      // Create environment rendering pipeline
      await this.#createEnvironmentPipeline();
      
      // Create global bind group with environment resources
      this.#createGlobalBindGroup();
      
      console.log(`Updated Environment WebGPU resources in ${(performance.now() - t0).toFixed(2)} ms`);
    } catch (error) {
      console.error("Failed to update environment:", error);
      throw error;
    }
  }

  render() {
    if (!this.model || !this.model.isLoaded()) return;

    // Skip rendering if pipeline is not ready
    if (!this.pipeline) {
      console.warn("Pipeline not ready, skipping render");
      return;
    }

    this.#updateUniforms();

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
    pass.draw(3, 1, 0, 0); // 3 vertices for fullscreen triangle
    
    // Set up vertex and index buffers
    pass.setVertexBuffer(0, this.vertexBuffer);
    pass.setIndexBuffer(this.indexBuffer, "uint32");

    // Draw opaque submeshes
    pass.setPipeline(this.pipeline);
    const subMeshes = this.model.getSubMeshes();
    for (const sm of subMeshes) {
      pass.setBindGroup(1, this.materialBindGroups[sm.materialIndex]);
      pass.drawIndexed(sm.indexCount, 1, sm.firstIndex, 0, 0);
    }

    pass.end();

    // Submit commands
    this.device.queue.submit([encoder.finish()]);
  }

  resize(width, height) {
    this.width = width;
    this.height = height;

    // Recreate depth texture with new size
    this.#createDepthTexture();

    // Update render pass descriptor
    this.renderPassDescriptor.depthStencilAttachment.view =
      this.depthTextureView;
  }

  // Private methods

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

  #createGlobalBindGroup() {
    if (!this.globalBindGroupLayout || !this.globalUniformBuffer) {
      console.warn("Cannot create global bind group: missing layout or uniform buffer");
      return;
    }

    // Use environment resources if available, otherwise use fallbacks
    const envSampler = this.environmentCubeSampler || this.sampler;
    const envTexture = this.environmentTextureView || this.defaultCubeTexture.createView();

    this.globalBindGroup = this.device.createBindGroup({
      layout: this.globalBindGroupLayout,
      entries: [
        { binding: 0, resource: { buffer: this.globalUniformBuffer } }, // Global uniforms
        { binding: 1, resource: envSampler }, // Environment sampler (or fallback)
        { binding: 2, resource: envTexture }, // Environment cubemap (or fallback)
      ],
    });
  }

  // Helper function for power-of-2 calculation
  #floorPow2(x) {
    let power = 1;
    while (power * 2 <= x) {
      power *= 2;
    }
    return power;
  }

  // Create environment cubemap texture and view
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

  // Create environment sampler
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

  // Create environment textures and samplers
  async #createEnvironmentTexturesAndSamplers() {
    // Get panorama texture from environment
    const panoramaTexture = this.environment.getTexture();
    
    // Calculate environment cubemap size (power of 2, based on panorama width)
    const environmentCubeSize = this.#floorPow2(panoramaTexture.m_width);
    
    // Create environment cubemap texture with mipmaps
    const envResult = this.#createEnvironmentTexture([environmentCubeSize, environmentCubeSize, 6], true);
    this.environmentTexture = envResult.texture;
    this.environmentTextureView = envResult.textureView;
    
    // Initialize panorama converter if not already created
    if (!this.panoramaConverter) {
      this.panoramaConverter = new PanoramaToCubemapConverter(this.device);
    }
    
    // Convert panorama to cubemap
    await this.panoramaConverter.uploadAndConvert(panoramaTexture, this.environmentTexture);
    
    // Generate mipmaps for the environment cubemap
    await this.mipmapGenerator.generateMipmaps(
      this.environmentTexture,
      { width: environmentCubeSize, height: environmentCubeSize },
      MipKind.Float16Cube
    );
    
    // Create environment sampler
    this.environmentCubeSampler = this.#createEnvironmentSampler();
  }

  // Create environment rendering pipeline
  async #createEnvironmentPipeline() {
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
          format: "depth24plus",
          depthWriteEnabled: false, // Don't write depth for environment background
          depthCompare: "less-equal", // Render at far plane
        },
      });

    } catch (error) {
      console.error("Failed to create environment pipeline:", error);
      throw error;
    }
  }

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

  #createDepthTexture() {
    if (this.depthTexture) {
      this.depthTexture.destroy();
    }

    this.depthTexture = this.device.createTexture({
      size: [this.width, this.height, 1],
      format: "depth24plus",
      usage: GPUTextureUsage.RENDER_ATTACHMENT,
    });

    this.depthTextureView = this.depthTexture.createView();
  }

  async #createPipeline() {
    // Load shader code from file
    const shaderCode = await this.#loadShaderFile("./shaders/gltf_pbr.wgsl");

    // Create shader module
    const shaderModule = this.device.createShaderModule({ code: shaderCode });

    // Create separate bind group layouts
    this.#createBindGroupLayouts();

    // Create pipeline layout
    const pipelineLayout = this.device.createPipelineLayout({
      bindGroupLayouts: [this.globalBindGroupLayout, this.materialBindGroupLayout],
    });

    // Create render pipeline
    this.pipeline = this.device.createRenderPipeline({
      vertex: {
        module: shaderModule,
        entryPoint: "vs_main",
        buffers: [
          {
            arrayStride: 18 * Float32Array.BYTES_PER_ELEMENT, // Full vertex: 18 floats
            attributes: [
              {
                shaderLocation: 0, // Position
                format: "float32x3",
                offset: 0,
              },
              {
                shaderLocation: 1, // Normal
                format: "float32x3",
                offset: 3 * Float32Array.BYTES_PER_ELEMENT,
              },
              {
                shaderLocation: 2, // Tangent
                format: "float32x4",
                offset: 6 * Float32Array.BYTES_PER_ELEMENT,
              },
              {
                shaderLocation: 3, // TexCoord0
                format: "float32x2",
                offset: 10 * Float32Array.BYTES_PER_ELEMENT,
              },
              {
                shaderLocation: 4, // TexCoord1
                format: "float32x2",
                offset: 12 * Float32Array.BYTES_PER_ELEMENT,
              },
              {
                shaderLocation: 5, // Color
                format: "float32x4",
                offset: 14 * Float32Array.BYTES_PER_ELEMENT,
              },
            ],
          },
        ],
      },
      fragment: {
        module: shaderModule,
        entryPoint: "fs_main",
        targets: [{ format: this.format }],
      },
      primitive: {
        topology: "triangle-list",
      },
      depthStencil: {
        format: "depth24plus",
        depthWriteEnabled: true,
        depthCompare: "less",
      },
      layout: pipelineLayout,
    });
  }

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

  #createMaterialBindGroups(model) {
    this.materialBindGroups = [];
    this.materialUniformBuffers = [];
    if (!this.materialBindGroupLayout) return; // layout must exist

    const materials = model.getMaterials();
    if (!materials || materials.length === 0) return;

    // Helper to fetch texture by direct index (already mapped) or fallback
    const texOrDefault = (idx, kind) => {
      if (idx !== undefined && idx >= 0 && idx < this.modelTextures.length) {
        return this.modelTextures[idx];
      }
      if (kind === "normal") return this.defaultNormalTexture;
      return this.defaultTexture;
    };

    for (let i = 0; i < materials.length; i++) {
      const m = materials[i];

      // Create per-material uniform buffer
      const ub = this.device.createBuffer({
        size: 256,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      });
      const data = new Float32Array(16);
      let o = 0;
      data.set(m.baseColorFactor, o);
      o += 4;
      data[o++] = m.emissiveFactor[0];
      data[o++] = m.emissiveFactor[1];
      data[o++] = m.emissiveFactor[2];
      data[o++] = m.alphaMode;
      data[o++] = m.metallicFactor;
      data[o++] = m.roughnessFactor;
      data[o++] = m.normalScale;
      data[o++] = m.occlusionStrength;
      data[o++] = m.alphaCutoff;
      while (o < 16) data[o++] = 0.0;
      this.device.queue.writeBuffer(ub, 0, data.buffer);
      this.materialUniformBuffers.push(ub);

      const baseColorTex = texOrDefault(m.baseColorTexture, "baseColor");
      const mrTex = texOrDefault(
        m.metallicRoughnessTexture,
        "metallicRoughness"
      );
      const normalTex = texOrDefault(m.normalTexture, "normal");
      const occTex = texOrDefault(m.occlusionTexture, "occlusion");
      const emissiveTex = texOrDefault(m.emissiveTexture, "emissive");

      const bg = this.device.createBindGroup({
        layout: this.materialBindGroupLayout,
        entries: [
          { binding: 0, resource: { buffer: this.modelUniformBuffer } }, // Model uniforms
          { binding: 1, resource: { buffer: ub } }, // Material uniforms
          { binding: 2, resource: this.sampler }, // Sampler
          { binding: 3, resource: baseColorTex.createView() }, // BaseColor
          { binding: 4, resource: mrTex.createView() }, // MetallicRoughness
          { binding: 5, resource: normalTex.createView() }, // Normal
          { binding: 6, resource: occTex.createView() }, // Occlusion
          { binding: 7, resource: emissiveTex.createView() }, // Emissive
        ],
      });
      this.materialBindGroups.push(bg);
    }
  }

  async #loadModelTextures(model) {
    this.modelTextures = [];
    this.modelTextureTypes = [];

    const textures = model.getTextures();

    for (const texture of textures) {
      if (texture.image) {
        const webgpuTexture = await this.#createModelTexture(texture);
        this.modelTextures.push(webgpuTexture);
        this.modelTextureTypes.push(texture.type);
      }
    }
  }

  /**
   * @brief Create a model texture with mipmaps using the appropriate MipKind.
   * @param {Object} texture - Texture info with image, width, height, and type.
   * @returns {Promise<GPUTexture>} The created WebGPU texture with mipmaps.
   * @private
   */
  async #createModelTexture(texture) {
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
      await this.mipmapGenerator.generateMipmaps(
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
      await this.mipmapGenerator.generateMipmaps(
        intermediateTexture,
        { width, height },
        mipKind
      );

      // For non-sRGB data textures, we can use the intermediate directly
      // since the final format is also rgba8unorm
      return intermediateTexture;
    }
  }

  #createDefaultCubeTexture() {
    // Create a 1x1x6 white cube texture as default for environment
    this.defaultCubeTexture = this.device.createTexture({
      size: [1, 1, 6],
      format: "rgba8unorm",
      usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
    });

    // Write white pixel data to all 6 faces
    const whitePixel = new Uint8Array([255, 255, 255, 255]);
    for (let face = 0; face < 6; face++) {
      this.device.queue.writeTexture(
        { 
          texture: this.defaultCubeTexture,
          origin: { x: 0, y: 0, z: face }
        },
        whitePixel,
        { bytesPerRow: 4 },
        { width: 1, height: 1, depthOrArrayLayers: 1 }
      );
    }
  }

  #createSampler() {
    this.sampler = this.device.createSampler({
      magFilter: "linear",
      minFilter: "linear",
      mipmapFilter: "linear",
      addressModeU: "repeat",
      addressModeV: "repeat",
    });
  }

  #createDefaultTexture() {
    // Create a 1x1 white texture as default for base color (sRGB format)
    this.defaultTexture = this.device.createTexture({
      size: [1, 1, 1],
      format: "rgba8unorm-srgb",
      usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
    });

    // Write white pixel data
    const whitePixel = new Uint8Array([255, 255, 255, 255]);
    this.device.queue.writeTexture(
      { texture: this.defaultTexture },
      whitePixel,
      { bytesPerRow: 4 },
      { width: 1, height: 1 }
    );

    // Create a 1x1 "flat" normal texture (128, 128, 255, 255 = [0, 0, 1] in tangent space)
    this.defaultNormalTexture = this.device.createTexture({
      size: [1, 1, 1],
      format: "rgba8unorm",
      usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
    });

    // Write flat normal data (pointing straight up in tangent space)
    const flatNormal = new Uint8Array([128, 128, 255, 255]); // [0.5, 0.5, 1.0, 1.0] in [0,1] range
    this.device.queue.writeTexture(
      { texture: this.defaultNormalTexture },
      flatNormal,
      { bytesPerRow: 4 },
      { width: 1, height: 1 }
    );
  }

  #updateUniforms() {
    this.#updateGlobalUniforms();
    this.#updateModelUniforms();
  }

  #updateGlobalUniforms() {
    // Global uniforms: camera matrices and position only
    const globalData = new Float32Array(80); // 5 matrices * 16 floats + 4 floats for camera pos

    const viewMatrix = this.camera.getViewMatrix();
    const projectionMatrix = this.camera.getProjectionMatrix();
    const cameraPos = this.camera.getWorldPosition();

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

  #updateModelUniforms() {
    // Model uniforms: model transform matrices only
    const modelData = new Float32Array(32); // 2 matrices * 16 floats

    const modelMatrix = this.model.getTransform();
    
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

  #createVertexBuffer() {
    if (this.vertexBuffer) {
      this.vertexBuffer.destroy();
    }

    this.vertexBuffer = this.device.createBuffer({
      size: this.vertexData.byteLength,
      usage: GPUBufferUsage.VERTEX,
      mappedAtCreation: true,
    });

    new Float32Array(this.vertexBuffer.getMappedRange()).set(this.vertexData);
    this.vertexBuffer.unmap();
  }

  #createIndexBuffer() {
    if (this.indexBuffer) {
      this.indexBuffer.destroy();
    }

    if (this.indices.length > 0) {
      this.indexBuffer = this.device.createBuffer({
        size: this.indices.length * 4, // Always use 32-bit (4 bytes per index)
        usage: GPUBufferUsage.INDEX | GPUBufferUsage.COPY_DST,
        mappedAtCreation: true,
      });

      // Convert to 32-bit typed array
      const typedIndices = new Uint32Array(this.indices);
      new Uint32Array(this.indexBuffer.getMappedRange()).set(typedIndices);
      this.indexBuffer.unmap();

      // Store the typed array for format detection in render()
      this.indices = typedIndices;
    }
  }

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
      },
    };
  }

  // Reload shaders from disk
  async reloadShaders() {
    console.log("Destroying existing shader resources...");
    
    // Store references to current pipeline in case reload fails
    const oldPipeline = this.pipeline;
    const oldShaderModule = this.shaderModule;
    
    try {
      // Clear current pipeline and shader module
      this.pipeline = null;
      this.shaderModule = null;
      
      console.log("Loading shader from disk...");
      
      // Recreate the pipeline with fresh shader code
      await this.#createPipeline();
      
      console.log("Shader pipeline recreated successfully!");
    } catch (error) {
      console.error("Failed to reload shaders, restoring previous pipeline:", error);
      
      // Restore previous pipeline if reload failed
      this.pipeline = oldPipeline;
      this.shaderModule = oldShaderModule;
      
      throw error; // Re-throw so caller knows it failed
    }
  }

  // Load shader file from disk
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
