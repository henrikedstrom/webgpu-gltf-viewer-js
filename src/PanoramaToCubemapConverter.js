/**
 * @file PanoramaToCubemapConverter.js
 * @brief Uploads a panorama texture and converts it to a cubemap using a compute shader.
 */

export default class PanoramaToCubemapConverter {
  // === WebGPU Core ===
  #device;

  // === Compute Pipeline ===
  #computePipeline;

  // === Bind Groups & Layouts ===
  #bindGroupLayouts; // [common parameters, per-face uniforms]
  #perFaceBindGroups; // 6 bind groups for per-face parameters

  // === Uniform Buffers ===
  #perFaceUniformBuffers; // 6 buffers, one per cubemap face

  // === Samplers ===
  #sampler;

  // === Initialization State ===
  #isInitialized;
  #initializationPromise;

  /**
   * @brief Constructs a new converter using the provided WebGPU device.
   * @param {GPUDevice} device - The WebGPU device
   */
  constructor(device) {
    this.#device = device;
    
    // Only initialize fields with non-null defaults
    this.#bindGroupLayouts = [];
    this.#perFaceUniformBuffers = [];
    this.#perFaceBindGroups = [];
    this.#isInitialized = false;
    
    // Initialize synchronous resources only
    this.#initUniformBuffers();
    this.#initSampler();
    this.#initBindGroupLayouts();
    this.#initBindGroups();
  }

  /**
   * @brief Uploads the panorama texture and converts it into the provided cubemap texture.
   * @param {Object} panoramaTextureInfo - The source panorama texture data from Environment
   * @param {GPUTexture} environmentCubemap - The destination cubemap texture
   */
  async uploadAndConvert(panoramaTextureInfo, environmentCubemap) {
    // Ensure async initialization is complete
    await this.#ensureInitialized();
    
    const width = panoramaTextureInfo.width;
    const height = panoramaTextureInfo.height;
    const data = panoramaTextureInfo.data; // Float32Array

    // Validate data buffer size
    const expectedSize = width * height * 4; // 4 components (RGBA)
    if (data.length !== expectedSize) {
      throw new Error(`Data buffer size mismatch: expected ${expectedSize} floats, got ${data.length}`);
    }

    // Validate texture size for web constraints
    if (width > 8192 || height > 8192) {
      console.warn(`Texture size ${width}x${height} may exceed browser limits`);
    }

    // Create WebGPU texture descriptor for the input panorama texture
    const textureDescriptor = {
      usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.COPY_DST | GPUTextureUsage.COPY_SRC,
      size: [width, height, 1],
      format: 'rgba32float',
      mipLevelCount: 1,
    };
    const panoramaTexture = this.#device.createTexture(textureDescriptor);
    
    // Upload the texture data
    // Use the typed array directly or create a new buffer if it's a view
    const dataBuffer = data.byteOffset === 0 && data.byteLength === data.buffer.byteLength 
      ? data.buffer 
      : data.buffer.slice(data.byteOffset, data.byteOffset + data.byteLength);
      
    this.#device.queue.writeTexture(
      { texture: panoramaTexture },
      dataBuffer,
      { bytesPerRow: 4 * width * 4 }, // 4 components * 4 bytes per float
      { width, height }
    );

    // Create texture views
    const inputViewDesc = {
      format: 'rgba32float',
      dimension: '2d',
    };
    const outputCubeViewDesc = {
      format: 'rgba16float',
      dimension: '2d-array',
      baseMipLevel: 0,
      mipLevelCount: 1,
      baseArrayLayer: 0,
      arrayLayerCount: 6,
    };

    // Create bind group 0 (common for all faces)
    const bindGroup0 = this.#device.createBindGroup({
      layout: this.#bindGroupLayouts[0],
      entries: [
        { binding: 0, resource: this.#sampler },
        { binding: 1, resource: panoramaTexture.createView(inputViewDesc) },
        { binding: 2, resource: environmentCubemap.createView(outputCubeViewDesc) },
      ],
    });

    // Create command encoder and compute pass
    const encoder = this.#device.createCommandEncoder();
    const computePass = encoder.beginComputePass();

    // Set the compute pipeline
    computePass.setPipeline(this.#computePipeline);

    // Set bind group common to all faces
    computePass.setBindGroup(0, bindGroup0);

    // Dispatch compute shader for each face
    const workgroupSize = 8;
    const cubemapSize = environmentCubemap.width; // Assuming square faces
    const workgroupCountX = Math.ceil(cubemapSize / workgroupSize);
    const workgroupCountY = Math.ceil(cubemapSize / workgroupSize);

    for (let face = 0; face < 6; face++) {
      // Set per-face bind group
      computePass.setBindGroup(1, this.#perFaceBindGroups[face]);
      
      // Dispatch workgroups
      computePass.dispatchWorkgroups(workgroupCountX, workgroupCountY, 1);
    }

    // Finish and submit
    computePass.end();
    const commandBuffer = encoder.finish();
    this.#device.queue.submit([commandBuffer]);
  }

  /**
   * @brief Ensure async initialization is complete (lazy initialization)
   * @private
   */
  async #ensureInitialized() {
    if (this.#isInitialized) {
      return; // Already initialized
    }
    
    if (this.#initializationPromise) {
      // Initialization in progress, wait for it
      await this.#initializationPromise;
      return;
    }
    
    // Start initialization
    this.#initializationPromise = this.#initComputePipeline();
    await this.#initializationPromise;
    this.#isInitialized = true;
  }

  /**
   * @brief Initialize uniform buffers for per-face parameters
   * @private
   */
  #initUniformBuffers() {
    const bufferDescriptor = {
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      size: 4, // face index is a u32
    };
    
    // Each buffer contains the face index (u32)
    for (let face = 0; face < 6; face++) {
      this.#perFaceUniformBuffers[face] = this.#device.createBuffer(bufferDescriptor);
      const faceIndexData = new Uint32Array([face]);
      this.#device.queue.writeBuffer(this.#perFaceUniformBuffers[face], 0, faceIndexData.buffer);
    }
  }

  /**
   * @brief Initialize sampler for the input panorama texture
   * @private
   */
  #initSampler() {
    this.#sampler = this.#device.createSampler({
      addressModeU: 'repeat',
      addressModeV: 'clamp-to-edge',
      addressModeW: 'repeat',
      minFilter: 'nearest',
      magFilter: 'nearest',
      mipmapFilter: 'nearest',
    });
  }

  /**
   * @brief Initialize bind group layouts
   * @private
   */
  #initBindGroupLayouts() {
    // Bind group layout 0 (common parameters):
    // - binding 0: sampler
    // - binding 1: input texture (2D)
    // - binding 2: output texture (2D array storage)

    const samplerEntry = {
      binding: 0,
      visibility: GPUShaderStage.COMPUTE, 
      sampler: {
        type: 'non-filtering',
      },
    };
    const inputTextureEntry = {
      binding: 1,
      visibility: GPUShaderStage.COMPUTE, 
      texture: {
        sampleType: 'unfilterable-float',
        viewDimension: '2d',
        multisampled: false,
      },
    };
    const outputCubemapEntry = {
      binding: 2,
      visibility: GPUShaderStage.COMPUTE,
      storageTexture: {
        access: 'write-only',
        format: 'rgba16float',
        viewDimension: '2d-array',
      },
    };

    const group0Entries = [samplerEntry, inputTextureEntry, outputCubemapEntry];
    const group0LayoutDesc = {
      entries: group0Entries,
    };
    this.#bindGroupLayouts[0] = this.#device.createBindGroupLayout(group0LayoutDesc);

    
    // Bind group layout 1 (per-face parameters):
    // - binding 0: uniform buffer (face index)

    const faceIndexEntry = {
      binding: 0,
      visibility: GPUShaderStage.COMPUTE,
      buffer: {
        type: 'uniform',
        minBindingSize: 4,
      },
    };
    const group1Entries = [faceIndexEntry];
    const group1LayoutDesc = {
      entries: group1Entries
    };
    this.#bindGroupLayouts[1] = this.#device.createBindGroupLayout(group1LayoutDesc); 
  }

  /**
   * @brief Initialize bind groups for per-face parameters
   * @private
   */
  #initBindGroups() {
    
    // Create bind groups for per-face uniform buffers
    for (let face = 0; face < 6; face++) {
      const bindGroupEntries = [{
        binding: 0,
        resource: { buffer: this.#perFaceUniformBuffers[face] },
      }];
      const bindGroupDescriptor = {
        layout: this.#bindGroupLayouts[1],
        entries: bindGroupEntries,
      };
      this.#perFaceBindGroups[face] = this.#device.createBindGroup(bindGroupDescriptor);
    }
  }

  /**
   * @brief Initialize the compute pipeline (async)
   * @private
   */
  async #initComputePipeline() {
    const shaderCode = await this.#loadShaderFile("./shaders/panorama_to_cubemap.wgsl");
    const shaderModule = this.#device.createShaderModule({ code: shaderCode });
    
    const pipelineBindGroups = [this.#bindGroupLayouts[0], this.#bindGroupLayouts[1]];
    const pipelineLayout = this.#device.createPipelineLayout({ bindGroupLayouts: pipelineBindGroups });
    
    const descriptor = {
      layout: pipelineLayout,
      compute: {
        module: shaderModule,
        entryPoint: "panoramaToCubemap",
      },
    };
    
    this.#computePipeline = this.#device.createComputePipeline(descriptor);
  }

  /**
   * @brief Load shader file from the shaders directory
   * @param {string} filepath - Path to the shader file
   * @returns {Promise<string>} The shader source code
   * @private
   */
  async #loadShaderFile(filepath) {
    try {
      // Add cache-busting parameter for development
      const cacheBuster = `?t=${Date.now()}`;
      const response = await fetch(filepath + cacheBuster, {
        cache: 'no-cache'
      });
      
      if (!response.ok) {
        throw new Error(`Failed to load shader: ${response.status} ${response.statusText}`);
      }
      
      return await response.text();
    } catch (error) {
      console.error(`Error loading shader file ${filepath}:`, error);
      throw error;
    }
  }
}
