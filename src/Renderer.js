const { mat4 } = glMatrix;

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
    this.materialUniformBuffer = null;
    this.sampler = null;
    this.defaultTexture = null;
    this.defaultNormalTexture = null;
    this.bindGroup = null;
    
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

  async initialize(canvas, camera, model) {
    // Store canvas dimensions
    this.width = canvas.width;
    this.height = canvas.height;

    // Initialize WebGPU
    await this.#initWebGPU(canvas);
    
    // Create depth buffer
    this.#createDepthTexture();
    
    // Create resources first
    this.#createDefaultTexture();
    this.#createSampler();
    
    // Create rendering pipeline
    await this.#createPipeline();
    
    // Create uniform buffers (after pipeline, so we can create bind groups)
    this.#createUniformBuffers();
    
    // Setup model resources
    await this.updateModel(model);
    
    // Create render pass descriptor
    this.#createRenderPassDescriptor();
  }

  async updateModel(model) {
    if (!model.isLoaded()) return;

    // Get model data
    this.vertexData = model.getVertices();
    this.indices = model.getIndices();

    // Create vertex buffer
    this.#createVertexBuffer();
    
    // Create index buffer
    this.#createIndexBuffer();
    
    // Load model textures
    await this.#loadModelTextures(model);
    
    // Recreate bind group with actual textures
    this.#createBindGroup();
  }

  render(model, camera) {
    if (!model.isLoaded()) return;

    // Update all uniforms
    this.#updateUniforms(model, camera);

    // Get current texture
    const currentTexture = this.context.getCurrentTexture();
    this.renderPassDescriptor.colorAttachments[0].view = currentTexture.createView();

    // Create command encoder and render pass
    const commandEncoder = this.device.createCommandEncoder();
    const passEncoder = commandEncoder.beginRenderPass(this.renderPassDescriptor);

    // Issue drawing commands (per submesh with material)
    passEncoder.setPipeline(this.pipeline);
    passEncoder.setBindGroup(0, this.bindGroup);
    passEncoder.setVertexBuffer(0, this.vertexBuffer);

    const subMeshes = model.getSubMeshes();
    const materials = model.getMaterials();
    if (this.indices.length > 0 && subMeshes.length > 0) {
      passEncoder.setIndexBuffer(this.indexBuffer, "uint32");
      for (const sm of subMeshes) {
        const mat = materials[sm.materialIndex] || materials[0];
        this.#writeMaterialUniform(mat);
        passEncoder.drawIndexed(sm.indexCount, 1, sm.firstIndex, 0, 0);
      }
    } else if (this.indices.length > 0) {
      passEncoder.setIndexBuffer(this.indexBuffer, "uint32");
      this.#writeMaterialUniform(materials[0] || null);
      passEncoder.drawIndexed(this.indices.length);
    } else {
      this.#writeMaterialUniform(materials[0] || null);
      passEncoder.draw(this.vertexData.length / 18);
    }
    
    passEncoder.end();

    // Submit commands
    this.device.queue.submit([commandEncoder.finish()]);
  }

  resize(width, height) {
    this.width = width;
    this.height = height;
    
    // Recreate depth texture with new size
    this.#createDepthTexture();
    
    // Update render pass descriptor
    this.renderPassDescriptor.depthStencilAttachment.view = this.depthTextureView;
  }

  // Private methods
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
    const shaderCode = await this.#loadShaderFile('./shaders/gltf_pbr.wgsl');
    
    // Create shader module
    const shaderModule = this.device.createShaderModule({ code: shaderCode });

    // Create bind group layout for PBR with multiple textures
    const bindGroupLayout = this.device.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: "uniform" } }, // Global
        { binding: 1, visibility: GPUShaderStage.FRAGMENT, buffer: { type: "uniform" } }, // Material
        { binding: 2, visibility: GPUShaderStage.FRAGMENT, sampler: {} }, // Sampler
        { binding: 3, visibility: GPUShaderStage.FRAGMENT, texture: {} }, // BaseColor
        { binding: 4, visibility: GPUShaderStage.FRAGMENT, texture: {} }, // MetallicRoughness
        { binding: 5, visibility: GPUShaderStage.FRAGMENT, texture: {} }, // Normal
        { binding: 6, visibility: GPUShaderStage.FRAGMENT, texture: {} }, // Occlusion
        { binding: 7, visibility: GPUShaderStage.FRAGMENT, texture: {} }, // Emissive
      ],
    });

    // Create pipeline layout
    const pipelineLayout = this.device.createPipelineLayout({
      bindGroupLayouts: [bindGroupLayout],
    });

    // Create render pipeline
    this.pipeline = this.device.createRenderPipeline({
      vertex: {
        module: shaderModule,
        entryPoint: "vertexMain",
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
        entryPoint: "fragmentMain",
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
    // Global uniforms: 5 mat4 + 1 vec3 + padding = 80*4 + 16 = 336 bytes, round up to 256-byte alignment
    this.globalUniformBuffer = this.device.createBuffer({
      size: 512, // Enough space for all global uniforms
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    // Material uniforms: 4 vec4 = 64 bytes, we allocate 256 for alignment / future params
    this.materialUniformBuffer = this.device.createBuffer({
      size: 256,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    // Initial bind group with default texture
    this.#createBindGroup();
  }

  #createBindGroup() {
    // Find textures by type, use appropriate defaults
    const baseColorTexture = this.#findTextureByType('baseColor') || this.defaultTexture;
    const metallicRoughnessTexture = this.#findTextureByType('metallicRoughness') || this.defaultTexture;
    const normalTexture = this.#findTextureByType('normal') || this.defaultNormalTexture;
    const occlusionTexture = this.#findTextureByType('occlusion') || this.defaultTexture;
    const emissiveTexture = this.#findTextureByType('emissive') || this.defaultTexture;

    this.bindGroup = this.device.createBindGroup({
      layout: this.pipeline.getBindGroupLayout(0),
      entries: [
        {
          binding: 0, // Global uniforms
          resource: { buffer: this.globalUniformBuffer },
        },
        {
          binding: 1, // Material uniforms
          resource: { buffer: this.materialUniformBuffer },
        },
        {
          binding: 2, // Sampler
          resource: this.sampler,
        },
  { binding: 3, resource: baseColorTexture.createView() },
  { binding: 4, resource: metallicRoughnessTexture.createView() },
  { binding: 5, resource: normalTexture.createView() },
  { binding: 6, resource: occlusionTexture.createView() },
  { binding: 7, resource: emissiveTexture.createView() },
      ],
    });
  }

  #findTextureByType(type) {
    for (let i = 0; i < this.modelTextures.length; i++) {
      if (this.modelTextureTypes[i] === type) {
        return this.modelTextures[i];
      }
    }
    return null;
  }

  async #loadModelTextures(model) {
    this.modelTextures = [];
    this.modelTextureTypes = [];
    
    const textures = model.getTextures();
    
    for (const texture of textures) {
      if (texture.image) {
        // Create WebGPU texture from HTML Image
        const webgpuTexture = this.device.createTexture({
          size: [texture.width, texture.height, 1],
          format: 'rgba8unorm',
          usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST | GPUTextureUsage.RENDER_ATTACHMENT,
        });

        // Copy image data to WebGPU texture
        this.device.queue.copyExternalImageToTexture(
          { source: texture.image },
          { texture: webgpuTexture },
          [texture.width, texture.height]
        );

        this.modelTextures.push(webgpuTexture);
        this.modelTextureTypes.push(texture.type); // Track texture type
      }
    }
  }

  #createSampler() {
    this.sampler = this.device.createSampler({
      magFilter: 'linear',
      minFilter: 'linear',
      mipmapFilter: 'linear',
      addressModeU: 'repeat',
      addressModeV: 'repeat',
    });
  }

  #createDefaultTexture() {
    // Create a 1x1 white texture as default for base color
    this.defaultTexture = this.device.createTexture({
      size: [1, 1, 1],
      format: 'rgba8unorm',
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
      format: 'rgba8unorm',
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

  #updateUniforms(model, camera) {
    // Prepare global uniforms data
    const globalData = new Float32Array(80); // 5 matrices * 16 floats + 4 floats for camera pos
    
    const viewMatrix = camera.getViewMatrix();
    const projectionMatrix = camera.getProjectionMatrix();
    const modelMatrix = model.getTransform();
    // Compute normal matrix: inverse transpose of upper 3x3 of model matrix
    const modelMatrix3x3 = [
      modelMatrix[0], modelMatrix[1], modelMatrix[2],
      modelMatrix[4], modelMatrix[5], modelMatrix[6], 
      modelMatrix[8], modelMatrix[9], modelMatrix[10]
    ];
    
    // Create 3x3 matrix, invert it, then transpose
    const temp3x3 = mat4.create();
    mat4.set(temp3x3, 
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
    normalMatrix[0] = normalMatrix3x3[0]; normalMatrix[1] = normalMatrix3x3[1]; normalMatrix[2] = normalMatrix3x3[2];
    normalMatrix[4] = normalMatrix3x3[4]; normalMatrix[5] = normalMatrix3x3[5]; normalMatrix[6] = normalMatrix3x3[6];
    normalMatrix[8] = normalMatrix3x3[8]; normalMatrix[9] = normalMatrix3x3[9]; normalMatrix[10] = normalMatrix3x3[10];
    const cameraPos = camera.getWorldPosition();
    
    // Pack matrices (each matrix is 16 floats)
    globalData.set(viewMatrix, 0);           // offset 0-15
    globalData.set(projectionMatrix, 16);    // offset 16-31  
    globalData.set(modelMatrix, 32);         // offset 32-47
    globalData.set(normalMatrix, 48);        // offset 48-63
    globalData.set([cameraPos[0], cameraPos[1], cameraPos[2], 0], 64); // offset 64-67
    
    this.device.queue.writeBuffer(this.globalUniformBuffer, 0, globalData);
  }

  #writeMaterialUniform(material) {
    const data = new Float32Array(16);
    let o = 0;
    if (material) {
      data.set(material.baseColorFactor, o); o += 4;
      data[o++] = material.emissiveFactor[0];
      data[o++] = material.emissiveFactor[1];
      data[o++] = material.emissiveFactor[2];
      data[o++] = material.alphaMode;
      data[o++] = material.metallicFactor;
      data[o++] = material.roughnessFactor;
      data[o++] = material.normalScale;
      data[o++] = material.occlusionStrength;
      data[o++] = material.alphaCutoff;
      data[o++] = 0.0; // padding
      data[o++] = 0.0;
      data[o++] = 0.0;
    } else {
      data.set([
        1,1,1,1,
        0,0,0,0,
        1,1,1,1,
        0.5,0,0,0
      ]);
    }
    this.device.queue.writeBuffer(this.materialUniformBuffer, 0, data.buffer);
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



  // Load shader file from disk
  async #loadShaderFile(path) {
    try {
      const response = await fetch(path);
      if (!response.ok) {
        throw new Error(`Failed to load shader: ${response.status} ${response.statusText}`);
      }
      return await response.text();
    } catch (error) {
      console.error(`Error loading shader file ${path}:`, error);
      throw error;
    }
  }
}
