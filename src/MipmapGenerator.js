/**
 * @file MipmapGenerator.js
 * @brief Generates mipmaps for textures using compute and render shaders.
 *        Supports multiple texture types: linear UNORM, normal maps, cubemaps, and sRGB.
 */

/**
 * @brief Enumeration of mipmap generation modes.
 * @readonly
 * @enum {string}
 */
export const MipKind = Object.freeze({
  /** Generic linear UNORM 2D data (e.g., ORM/AO) */
  LinearUNorm2D: 'LinearUNorm2D',
  /** Normal maps (decode-average-renormalize-reencode) */
  Normal2D: 'Normal2D',
  /** Float cube textures (HDR/environment) */
  Float16Cube: 'Float16Cube',
  /** sRGB color textures (albedo/emissive) via render downsample */
  SRGB2D: 'SRGB2D',
});

export default class MipmapGenerator {
  // === WebGPU Core ===
  #device;

  // === Compute Pipelines ===
  #pipeline2D;
  #pipelineCube;
  #pipelineNormal2D;

  // === Bind Group Layouts ===
  #bindGroupLayout2D;
  #bindGroupLayoutCube;
  #bindGroupLayoutFace;
  #renderBindGroupLayout;

  // === Uniform Buffers & Bind Groups ===
  #uniformBuffers; // One per cubemap face (0..5)
  #faceBindGroups; // One per cubemap face (0..5)

  // === Render Pipeline (sRGB) ===
  #renderPipelineSRGB2D;
  #renderColorFormatSRGB;

  // === Initialization State ===
  #isInitialized;
  #initializationPromise;

  /**
   * @brief Constructs a new mipmap generator using the provided WebGPU device.
   * @param {GPUDevice} device - The WebGPU device
   */
  constructor(device) {
    this.#device = device;

    // Only initialize fields with non-null defaults
    this.#uniformBuffers = [];
    this.#faceBindGroups = [];
    this.#renderColorFormatSRGB = 'rgba8unorm-srgb';
    this.#isInitialized = false;

    // Initialize synchronous resources only
    this.#initUniformBuffers();
    this.#initBindGroupLayouts();
  }

  /**
   * @brief Generates a full mip chain for a texture.
   * @param {GPUTexture} texture - Texture created with STORAGE_BINDING (for compute) or RENDER_ATTACHMENT (for sRGB) and mip levels > 1.
   * @param {{width:number, height:number}} size - Base level size (level 0).
   * @param {MipKind} kind - The type of mipmap generation to perform.
   *
   * Notes:
   * - For LinearUNorm2D and Normal2D: uses compute shaders with textureLoad()/textureStore().
   * - For Float16Cube: uses compute shader for HDR cubemap downsampling.
   * - For SRGB2D: uses render pipeline to leverage hardware sRGB encode/decode.
   */
  async generateMipmaps(texture, size, kind = MipKind.LinearUNorm2D) {
    await this.#ensureInitialized();

    switch (kind) {
      case MipKind.LinearUNorm2D:
        this.#generate2DCompute(texture, size, this.#pipeline2D, this.#bindGroupLayout2D);
        break;
      case MipKind.Normal2D:
        this.#generate2DCompute(texture, size, this.#pipelineNormal2D, this.#bindGroupLayout2D);
        break;
      case MipKind.Float16Cube:
        this.#generateCubeCompute(texture, size);
        break;
      case MipKind.SRGB2D:
        this.#generate2DRenderSRGB(texture, size);
        break;
      default:
        this.#generate2DCompute(texture, size, this.#pipeline2D, this.#bindGroupLayout2D);
        break;
    }
  }

  /**
   * @brief Ensure async initialization is complete (lazy initialization)
   * @private
   */
  async #ensureInitialized() {
    if (this.#isInitialized) {
      return;
    }

    if (this.#initializationPromise) {
      await this.#initializationPromise;
      return;
    }

    this.#initializationPromise = this.#initPipelines();
    await this.#initializationPromise;
    this.#isInitialized = true;
  }

  /**
   * @brief Create one u32 face-index uniform buffer per cubemap face (0..5).
   * @private
   */
  #initUniformBuffers() {
    const bufferDescriptor = {
      size: 4, // Face id (u32)
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    };

    for (let face = 0; face < 6; face++) {
      const uniformBuffer = this.#device.createBuffer(bufferDescriptor);
      const faceIndexData = new Uint32Array([face]);
      this.#device.queue.writeBuffer(uniformBuffer, 0, faceIndexData.buffer);
      this.#uniformBuffers.push(uniformBuffer);
    }
  }

  /**
   * @brief Create bind group layouts for 2D/cubemap compute and render paths.
   * @private
   */
  #initBindGroupLayouts() {
    // Common input texture layout (compute)
    const inputTexture = {
      binding: 0,
      visibility: GPUShaderStage.COMPUTE,
      texture: {
        sampleType: 'unfilterable-float',
        multisampled: false,
      },
    };

    // Common output texture layout (compute)
    const outputTexture = {
      binding: 1,
      visibility: GPUShaderStage.COMPUTE,
      storageTexture: {
        access: 'write-only',
      },
    };

    // Setup 2D bind group layout
    inputTexture.texture.viewDimension = '2d';
    outputTexture.storageTexture.viewDimension = '2d';
    outputTexture.storageTexture.format = 'rgba8unorm';

    this.#bindGroupLayout2D = this.#device.createBindGroupLayout({
      entries: [
        { ...inputTexture },
        { ...outputTexture },
      ],
    });

    // Setup Cube bind group layout
    inputTexture.texture.viewDimension = '2d-array';
    outputTexture.storageTexture.viewDimension = '2d-array';
    outputTexture.storageTexture.format = 'rgba16float';

    this.#bindGroupLayoutCube = this.#device.createBindGroupLayout({
      entries: [
        { ...inputTexture },
        { ...outputTexture },
      ],
    });

    // Face index layout (only for cube maps)
    this.#bindGroupLayoutFace = this.#device.createBindGroupLayout({
      entries: [
        {
          binding: 0,
          visibility: GPUShaderStage.COMPUTE,
          buffer: {
            type: 'uniform',
            minBindingSize: 4,
          },
        },
      ],
    });

    // Create bind groups for each face
    for (let face = 0; face < 6; face++) {
      this.#faceBindGroups.push(
        this.#device.createBindGroup({
          layout: this.#bindGroupLayoutFace,
          entries: [
            { binding: 0, resource: { buffer: this.#uniformBuffers[face] } },
          ],
        })
      );
    }
  }

  /**
   * @brief Initialize all pipelines (compute and render).
   * @private
   */
  async #initPipelines() {
    await Promise.all([
      this.#initComputePipelines(),
      this.#initRenderPipeline(),
    ]);
  }

  /**
   * @brief Create compute pipelines for 2D, normal, and cubemap mip generation.
   * @private
   */
  async #initComputePipelines() {
    const [pipeline2D, pipelineCube, pipelineNormal2D] = await Promise.all([
      this.#createComputePipeline('./shaders/mipmap_generator_2d.wgsl', [this.#bindGroupLayout2D]),
      this.#createComputePipeline('./shaders/mipmap_generator_cube.wgsl', [this.#bindGroupLayoutCube, this.#bindGroupLayoutFace]),
      this.#createComputePipeline('./shaders/mipmap_generator_normal_2d.wgsl', [this.#bindGroupLayout2D]),
    ]);

    this.#pipeline2D = pipeline2D;
    this.#pipelineCube = pipelineCube;
    this.#pipelineNormal2D = pipelineNormal2D;
  }

  /**
   * @brief Create the render pipeline for sRGB mipmap generation.
   * @private
   */
  async #initRenderPipeline() {
    // Bind group layout: texture only (using textureLoad, no sampler needed)
    this.#renderBindGroupLayout = this.#device.createBindGroupLayout({
      entries: [
        {
          binding: 0,
          visibility: GPUShaderStage.FRAGMENT,
          texture: {
            sampleType: 'float',
            viewDimension: '2d',
            multisampled: false,
          },
        },
      ],
    });

    // Create render pipeline targeting sRGB RGBA8 color
    this.#renderPipelineSRGB2D = await this.#createRenderPipeline(
      './shaders/mipmap_downsample_render.wgsl',
      this.#renderColorFormatSRGB
    );
  }

  /**
   * @brief Create a compute pipeline from a shader file.
   * @param {string} shaderPath - Path to the WGSL shader file.
   * @param {GPUBindGroupLayout[]} bindGroupLayouts - Bind group layouts for the pipeline.
   * @returns {Promise<GPUComputePipeline>} The created compute pipeline.
   * @private
   */
  async #createComputePipeline(shaderPath, bindGroupLayouts) {
    const shaderCode = await this.#loadShaderFile(shaderPath);
    const shaderModule = this.#device.createShaderModule({ code: shaderCode });
    const pipelineLayout = this.#device.createPipelineLayout({ bindGroupLayouts });

    return this.#device.createComputePipeline({
      layout: pipelineLayout,
      compute: {
        module: shaderModule,
        entryPoint: 'computeMipMap',
      },
    });
  }

  /**
   * @brief Create a render pipeline from a shader file.
   * @param {string} shaderPath - Path to the WGSL shader file.
   * @param {GPUTextureFormat} colorFormat - The target color format.
   * @returns {Promise<GPURenderPipeline>} The created render pipeline.
   * @private
   */
  async #createRenderPipeline(shaderPath, colorFormat) {
    const shaderCode = await this.#loadShaderFile(shaderPath);
    const shaderModule = this.#device.createShaderModule({ code: shaderCode });

    const pipelineLayout = this.#device.createPipelineLayout({
      bindGroupLayouts: [this.#renderBindGroupLayout],
    });

    return this.#device.createRenderPipeline({
      layout: pipelineLayout,
      vertex: {
        module: shaderModule,
        entryPoint: 'vs_main',
        buffers: [],
      },
      fragment: {
        module: shaderModule,
        entryPoint: 'fs_main',
        targets: [{ format: colorFormat }],
      },
      primitive: {
        topology: 'triangle-list',
      },
    });
  }

  /**
   * @brief Load shader file from the shaders directory.
   * @param {string} shaderPath - Path to the shader file.
   * @returns {Promise<string>} The shader source code.
   * @private
   */
  async #loadShaderFile(shaderPath) {
    const response = await fetch(shaderPath);
    return response.text();
  }

  /**
   * @brief Generate mipmaps for a 2D texture using a compute shader.
   * @param {GPUTexture} texture - The texture to generate mipmaps for.
   * @param {{width:number, height:number}} size - Base level size.
   * @param {GPUComputePipeline} pipeline - The compute pipeline to use.
   * @param {GPUBindGroupLayout} layout - The bind group layout.
   * @private
   */
  #generate2DCompute(texture, size, pipeline, layout) {
    const mipLevelCount = 1 + Math.floor(Math.log2(Math.max(size.width, size.height)));

    // Create mip level views
    const viewDescriptor = {
      format: 'rgba8unorm',
      dimension: '2d',
      baseMipLevel: 0,
      mipLevelCount: 1,
      baseArrayLayer: 0,
      arrayLayerCount: 1,
    };

    const mipLevelViews = [];
    for (let i = 0; i < mipLevelCount; i++) {
      viewDescriptor.baseMipLevel = i;
      mipLevelViews.push(texture.createView(viewDescriptor));
    }

    const encoder = this.#device.createCommandEncoder();
    const computePass = encoder.beginComputePass();
    computePass.setPipeline(pipeline);

    for (let nextLevel = 1; nextLevel < mipLevelViews.length; nextLevel++) {
      const width = Math.max(1, size.width >> nextLevel);
      const height = Math.max(1, size.height >> nextLevel);

      const bindGroup = this.#device.createBindGroup({
        layout,
        entries: [
          { binding: 0, resource: mipLevelViews[nextLevel - 1] },
          { binding: 1, resource: mipLevelViews[nextLevel] },
        ],
      });
      computePass.setBindGroup(0, bindGroup);

      const workgroupSize = 8;
      const workgroupCountX = Math.ceil(width / workgroupSize);
      const workgroupCountY = Math.ceil(height / workgroupSize);
      computePass.dispatchWorkgroups(workgroupCountX, workgroupCountY, 1);
    }

    computePass.end();
    const commandBuffer = encoder.finish();
    this.#device.queue.submit([commandBuffer]);
  }

  /**
   * @brief Generate mipmaps for a cubemap texture using a compute shader.
   * @param {GPUTexture} texture - The cubemap texture to generate mipmaps for.
   * @param {{width:number, height:number}} size - Base level size (per face).
   * @private
   */
  #generateCubeCompute(texture, size) {
    const mipLevelCount = 1 + Math.floor(Math.log2(Math.max(size.width, size.height)));

    // Create views per mip level (2D array views over 6 faces)
    const viewDescriptor = {
      format: 'rgba16float',
      dimension: '2d-array',
      baseMipLevel: 0,
      mipLevelCount: 1,
      baseArrayLayer: 0,
      arrayLayerCount: 6,
    };

    const mipLevelViews = [];
    for (let i = 0; i < mipLevelCount; i++) {
      viewDescriptor.baseMipLevel = i;
      mipLevelViews.push(texture.createView(viewDescriptor));
    }

    // Command encoding
    const encoder = this.#device.createCommandEncoder();
    const computePass = encoder.beginComputePass();
    computePass.setPipeline(this.#pipelineCube);

    // For each face and mip level
    for (let face = 0; face < 6; face++) {
      // Set per-face uniform (group 1)
      computePass.setBindGroup(1, this.#faceBindGroups[face]);

      for (let nextLevel = 1; nextLevel < mipLevelViews.length; nextLevel++) {
        const width = Math.max(1, size.width >> nextLevel);
        const height = Math.max(1, size.height >> nextLevel);

        // Bind prev/next level views (group 0)
        const bindGroup = this.#device.createBindGroup({
          layout: this.#bindGroupLayoutCube,
          entries: [
            { binding: 0, resource: mipLevelViews[nextLevel - 1] },
            { binding: 1, resource: mipLevelViews[nextLevel] },
          ],
        });
        computePass.setBindGroup(0, bindGroup);

        const workgroupSize = 8;
        const workgroupCountX = Math.ceil(width / workgroupSize);
        const workgroupCountY = Math.ceil(height / workgroupSize);
        computePass.dispatchWorkgroups(workgroupCountX, workgroupCountY, 1);
      }
    }

    computePass.end();
    const commandBuffer = encoder.finish();
    this.#device.queue.submit([commandBuffer]);
  }

  /**
   * @brief Generate mipmaps for an sRGB 2D texture using render passes.
   *        Uses hardware sRGB encode/decode for correct gamma handling.
   * @param {GPUTexture} texture - The sRGB texture to generate mipmaps for.
   * @param {{width:number, height:number}} size - Base level size.
   * @private
   */
  #generate2DRenderSRGB(texture, size) {
    const mipLevelCount = 1 + Math.floor(Math.log2(Math.max(size.width, size.height)));

    // Create command encoder
    const encoder = this.#device.createCommandEncoder();

    // Iterate over mip levels
    for (let nextLevel = 1; nextLevel < mipLevelCount; nextLevel++) {
      // Views for prev (sampled) and next (render target) levels
      const prevView = texture.createView({
        format: 'rgba8unorm-srgb',
        dimension: '2d',
        baseMipLevel: nextLevel - 1,
        mipLevelCount: 1,
        baseArrayLayer: 0,
        arrayLayerCount: 1,
      });

      const nextView = texture.createView({
        format: 'rgba8unorm-srgb',
        dimension: '2d',
        baseMipLevel: nextLevel,
        mipLevelCount: 1,
        baseArrayLayer: 0,
        arrayLayerCount: 1,
      });

      // Create bind group for prev level (texture only, using textureLoad)
      const bindGroup = this.#device.createBindGroup({
        layout: this.#renderBindGroupLayout,
        entries: [
          { binding: 0, resource: prevView },
        ],
      });

      // Render pass to write next level
      const renderPass = encoder.beginRenderPass({
        colorAttachments: [
          {
            view: nextView,
            loadOp: 'clear',
            storeOp: 'store',
            clearValue: { r: 0.0, g: 0.0, b: 0.0, a: 0.0 },
          },
        ],
      });

      renderPass.setPipeline(this.#renderPipelineSRGB2D);
      renderPass.setBindGroup(0, bindGroup);
      renderPass.draw(3, 1, 0, 0); // Fullscreen triangle
      renderPass.end();
    }

    // Submit
    const commandBuffer = encoder.finish();
    this.#device.queue.submit([commandBuffer]);
  }
}
