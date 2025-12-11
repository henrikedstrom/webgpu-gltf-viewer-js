/**
 * @file EnvironmentPreprocessor.js
 * @brief Helper class for generating IBL maps (irradiance, specular, BRDF LUT) from an environment cube map.
 */

export default class EnvironmentPreprocessor {
  // === WebGPU Core ===
  #device;

  // === Compute Pipelines ===
  #pipelineIrradiance;
  #pipelinePrefilteredSpecular;
  #pipelineBRDFIntegrationLUT;

  // === Bind Groups & Layouts ===
  #bindGroupLayouts; // [common parameters, per-face uniforms]
  #perFaceBindGroups; // 6 bind groups for per-face parameters
  #perMipBindGroups; // Per-mip bind groups (created on demand)

  // === Uniform Buffers ===
  #uniformBuffer;
  #perMipUniformBuffers;
  #perFaceUniformBuffers; // 6 buffers, one per cubemap face

  // === Samplers ===
  #environmentSampler;

  // === Initialization State ===
  #isInitialized;
  #initializationPromise;
  
  /**
   * @brief Constructs a new environment preprocessor using the provided WebGPU device.
   * @param {GPUDevice} device - The WebGPU device
   */
  constructor(device) {
    this.#device = device;

    // Only initialize fields with non-null defaults
    this.#bindGroupLayouts = [];
    this.#perMipUniformBuffers = [];
    this.#perFaceUniformBuffers = [];
    this.#perFaceBindGroups = [];
    this.#isInitialized = false;
    
    // Initialize synchronous resources only (delay shader loading and pipeline creation)
    this.#initUniformBuffers();
    this.#initSampler();
    this.#initBindGroupLayouts();
    this.#initBindGroups();
  }

  /**
   * @brief Generates the IBL maps (irradiance, specular, BRDF LUT) from the provided environment cube map.
   * @param {Object} environmentCubemap - The environment cube map
   * @param {GPUTexture} irradianceCubemap - The irradiance cube map
   * @param {GPUTexture} specularCubemap - The specular cube map
   * @param {GPUTexture} brdfIntegrationLUT - The BRDF integration LUT
   */
  async generateMaps(environmentCubemap, irradianceCubemap, specularCubemap, brdfIntegrationLUT) {
    // Ensure async initialization is complete
    await this.#ensureInitialized();

    // Create views for the input cubemap and output cubemap.
    const inputViewDesc = {
      format: 'rgba16float',
      dimension: 'cube',
    };

    const outputCubeViewDesc = {
      format: 'rgba16float',
      dimension: '2d-array',
      baseMipLevel: 0,
      mipLevelCount: 1,
      baseArrayLayer: 0,
      arrayLayerCount: 6,
    };

    const output2DViewDesc = {
      format: 'rgba16float',
      dimension: '2d',
      baseMipLevel: 0,
      mipLevelCount: 1,
      baseArrayLayer: 0,
      arrayLayerCount: 1,
    };

    // Bind group 0 (common for all passes)
    const bindGroup0Entries = [
      { binding: 0, resource: this.#environmentSampler },
      { binding: 1, resource: environmentCubemap.createView(inputViewDesc) },
      { binding: 2, resource: this.#uniformBuffer },
      { binding: 3, resource: irradianceCubemap.createView(outputCubeViewDesc) },
      { binding: 4, resource: brdfIntegrationLUT.createView(output2DViewDesc) },
    ];

    const bindGroup0 = this.#device.createBindGroup({
      layout: this.#bindGroupLayouts[0], 
      entries: bindGroup0Entries,
    });
  
    // Bind group 2 (per-mip)
    const perMipBindGroups = this.#createPerMipBindGroups(specularCubemap);

    // Create a command encoder and compute pass.
    const encoder = this.#device.createCommandEncoder();
    const computePass = encoder.beginComputePass();

    // ---- Pass 1: Generate Irradiance Map (Diffuse IBL) ----

    // Set the pipeline for irradiance cubemap generation.
    computePass.setPipeline(this.#pipelineIrradiance);

    // Set bind groups common to all faces.
    computePass.setBindGroup(0, bindGroup0);
    computePass.setBindGroup(2, perMipBindGroups[0]); // Make sure BG2 is valid (not used in first pass)

    // Dispatch a compute shader for each face of the cubemap.
    const numFaces = 6;
    for (let face = 0; face < numFaces; face++) {
      // For each face, update the per-face uniform (bind group 1).
      computePass.setBindGroup(1, this.#perFaceBindGroups[face]);

      const workgroupSize = 8;
      const workgroupCountX = Math.ceil(irradianceCubemap.width / workgroupSize);
      const workgroupCountY = Math.ceil(irradianceCubemap.height / workgroupSize);
      computePass.dispatchWorkgroups(workgroupCountX, workgroupCountY, 1);
    }

    // ---- Pass 2: Generate Prefiltered Specular Map (Specular IBL) ----

    const mipLevelCount = specularCubemap.mipLevelCount;

    // Set the pipeline for prefiltered specular cubemap generation.
    computePass.setPipeline(this.#pipelinePrefilteredSpecular);

    // Dispatch a compute shader for each mip level of each face of the cubemap.
    for (let face = 0; face < numFaces; face++) {
      // Bind per-face uniform (bind group 1).
      computePass.setBindGroup(1, this.#perFaceBindGroups[face]);
      
      for (let mipLevel = 0; mipLevel < mipLevelCount; mipLevel++) {
        // Bind per-mip uniforms (bind group 2).
        computePass.setBindGroup(2, perMipBindGroups[mipLevel]);

        const mipWidth = Math.max(1, specularCubemap.width >> mipLevel);
        const mipHeight = Math.max(1, specularCubemap.height >> mipLevel);

        const workgroupSize = 8;
        const workgroupCountX = Math.ceil(mipWidth / workgroupSize);
        const workgroupCountY = Math.ceil(mipHeight / workgroupSize);
        computePass.dispatchWorkgroups(workgroupCountX, workgroupCountY, 1);
      }
    }

    // ---- Pass 3: Generate BRDF Integration LUT ----

    // Set the pipeline for BRDF integration LUT generation.
    computePass.setPipeline(this.#pipelineBRDFIntegrationLUT);

    // Dispatch a compute shader for the output texture.
    const width = brdfIntegrationLUT.width;
    const height = brdfIntegrationLUT.height;
    const workgroupSize = 8;
    const workgroupCountX = Math.ceil(width / workgroupSize);
    const workgroupCountY = Math.ceil(height / workgroupSize);
    computePass.dispatchWorkgroups(workgroupCountX, workgroupCountY, 1);

    // Finish the compute pass and submit the command buffer.
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
    this.#initializationPromise = this.#initComputePipelines();
    await this.#initializationPromise;
    this.#isInitialized = true;
  }
  
  /**
   * @brief Initialize uniform buffers for per-face parameters
   * @private
   */
  #initUniformBuffers() {
    const bufferDescriptor = {
      size: 4,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    };
    this.#uniformBuffer = this.#device.createBuffer(bufferDescriptor);
    this.#device.queue.writeBuffer(this.#uniformBuffer, 0, new Uint32Array([1024]));

    // Update descriptor for per-face uniform buffers
    bufferDescriptor.size = 4; // Face id
    for (let face = 0; face < 6; face++) {
      this.#perFaceUniformBuffers[face] = this.#device.createBuffer(bufferDescriptor);
      this.#device.queue.writeBuffer(this.#perFaceUniformBuffers[face], 0, new Uint32Array([face]));
    }
  }

  /**
   * @brief Initialize sampler for environment cube map
   * @private
   */
  #initSampler() {
    this.#environmentSampler = this.#device.createSampler({
      addressModeU: 'repeat',
      addressModeV: 'repeat',
      addressModeW: 'repeat',
      minFilter: 'linear',
      magFilter: 'linear',
      mipmapFilter: 'linear',
    });
  }

  /**
   * @brief Initialize bindGroupLayouts for the compute pipelines
   * @private
   */
  #initBindGroupLayouts() {
    // Bind group layout 0 (common parameters):
    // - binding 0: sampler
    // - binding 1: environment cubemap
    // - binding 2: number of samples
    // - binding 3: irradiance cubemap
    // - binding 4: BRDF integration LUT

    const samplerEntry = {
      binding: 0,
      visibility: GPUShaderStage.COMPUTE,
      sampler: { type: 'filtering' },
    };

    const cubemapEntry = {
      binding: 1,
      visibility: GPUShaderStage.COMPUTE,
      texture: { sampleType: 'float', viewDimension: 'cube' },
    };

    const numSamplesEntry = {
      binding: 2,
      visibility: GPUShaderStage.COMPUTE,
      buffer: { type: 'uniform', minBindingSize: 4 },
    };

    const irradianceEntry = {
      binding: 3,
      visibility: GPUShaderStage.COMPUTE,
      storageTexture: { access: 'write-only', format: 'rgba16float', viewDimension: '2d-array' },
    };

    const brdfLutEntry = {
      binding: 4,
      visibility: GPUShaderStage.COMPUTE,
      storageTexture: { access: 'write-only', format: 'rgba16float', viewDimension: '2d' },
    };

    const group0Entries = [samplerEntry, cubemapEntry, numSamplesEntry, irradianceEntry, brdfLutEntry];
    const group0LayoutDesc = { entries: group0Entries };
    this.#bindGroupLayouts[0] = this.#device.createBindGroupLayout(group0LayoutDesc);

    const faceIndexEntry = {
      binding: 0,
      visibility: GPUShaderStage.COMPUTE,
      buffer: { type: 'uniform', minBindingSize: 4 },
    };
    const group1Entries = [faceIndexEntry];
    const group1LayoutDesc = { entries: group1Entries };
    this.#bindGroupLayouts[1] = this.#device.createBindGroupLayout(group1LayoutDesc);

    const roughnessParamsEntry = {
      binding: 0,
      visibility: GPUShaderStage.COMPUTE,
      buffer: { type: 'uniform', minBindingSize: 4 },
    };

    const prefilteredSpecularEntry = {
      binding: 1,
      visibility: GPUShaderStage.COMPUTE,
      storageTexture: { access: 'write-only', format: 'rgba16float', viewDimension: '2d-array' },
    };

    const group2Entries = [roughnessParamsEntry, prefilteredSpecularEntry];
    const group2LayoutDesc = { entries: group2Entries };
    this.#bindGroupLayouts[2] = this.#device.createBindGroupLayout(group2LayoutDesc);
  }

  /**
   * @brief Initialize bindGroups for the compute pipelines
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
   * @brief Initialize compute pipelines for the environment preprocessor
   * @private
   */
  async #initComputePipelines() {
    const shaderCode = await this.#loadShaderFile("./shaders/environment_prefilter.wgsl");
    const shaderModule = this.#device.createShaderModule({ code: shaderCode });

    const pipelineBindGroups = [this.#bindGroupLayouts[0], this.#bindGroupLayouts[1], this.#bindGroupLayouts[2]];
    const pipelineLayout = this.#device.createPipelineLayout({ bindGroupLayouts: pipelineBindGroups });

    const computeIrradianceDescriptor = {
      layout: pipelineLayout,
      compute: {
        module: shaderModule,
        entryPoint: "computeIrradiance",
      },
    };
    this.#pipelineIrradiance = this.#device.createComputePipeline(computeIrradianceDescriptor);

    const computePrefilteredSpecularDescriptor = {
      layout: pipelineLayout,
      compute: {
        module: shaderModule,
        entryPoint: "computePrefilteredSpecular",
      },
    };
    this.#pipelinePrefilteredSpecular = this.#device.createComputePipeline(computePrefilteredSpecularDescriptor);

    const computeBRDFIntegrationLUTDescriptor = {
      layout: pipelineLayout,
      compute: {
        module: shaderModule,
        entryPoint: "computeLUT",
      },
    };
    this.#pipelineBRDFIntegrationLUT = this.#device.createComputePipeline(computeBRDFIntegrationLUTDescriptor);
  }



  /**
   * @brief Create bind groups for per-mip parameters
   * @param {GPUTexture} specularCubemap - The specular cube map
   * @private
   */
  #createPerMipBindGroups(specularCubemap) {
    const mipLevelCount = specularCubemap.mipLevelCount;
    this.#perMipUniformBuffers = new Array(mipLevelCount);
    this.#perMipBindGroups = new Array(mipLevelCount);

    // Create a buffer for the roughness parameter
    const bufferDescriptor = {
      size: 4,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    };
    for (let mipLevel = 0; mipLevel < mipLevelCount; mipLevel++) {
      this.#perMipUniformBuffers[mipLevel] = this.#device.createBuffer(bufferDescriptor);
      const roughness = mipLevel / (mipLevelCount - 1);
      this.#device.queue.writeBuffer(this.#perMipUniformBuffers[mipLevel], 0, new Float32Array([roughness])); 
    }

    // Create a texture view descriptor for the output cubemap
    const outputCubeViewDesc = {
      format: 'rgba16float',
      dimension: '2d-array',
      baseMipLevel: 0,
      mipLevelCount: 1,
      baseArrayLayer: 0,
      arrayLayerCount: 6,
    };

    // Create bind group descriptor
    const bindGroup2Entries = [
      { binding: 0, resource: this.#perMipUniformBuffers[0] },
      { binding: 1, resource: specularCubemap.createView(outputCubeViewDesc) },
    ];
    const bindGroup2Descriptor = {
      layout: this.#bindGroupLayouts[2],
      entries: bindGroup2Entries,
    };
    this.#perMipBindGroups[0] = this.#device.createBindGroup(bindGroup2Descriptor);

    // Create bind groups for each mip level
    for (let mipLevel = 1; mipLevel < mipLevelCount; mipLevel++) {
      outputCubeViewDesc.baseMipLevel = mipLevel;
      const bindGroupEntries = [
        { binding: 0, resource: this.#perMipUniformBuffers[mipLevel] },
        { binding: 1, resource: specularCubemap.createView(outputCubeViewDesc) },
      ];
      const bindGroupDescriptor = {
        layout: this.#bindGroupLayouts[2],
        entries: bindGroupEntries,
      };
      this.#perMipBindGroups[mipLevel] = this.#device.createBindGroup(bindGroupDescriptor);
    }
    return this.#perMipBindGroups;
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
