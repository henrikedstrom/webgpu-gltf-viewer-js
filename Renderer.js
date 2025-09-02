const { mat4 } = glMatrix;

// Shaders (WGSL) - Using structs like C++ version
const shaderCode = `
  @group(0) @binding(0) var<uniform> transformationMatrix: mat4x4<f32>;

  struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) tangent: vec4<f32>,
    @location(3) texCoord0: vec2<f32>,
    @location(4) texCoord1: vec2<f32>,
    @location(5) color: vec4<f32>
  };

  struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) fragColor: vec3<f32>,
    @location(1) texCoord0: vec2<f32>
  };

  @vertex
  fn vertexMain(input: VertexInput) -> VertexOutput {
    var output: VertexOutput;
    output.position = transformationMatrix * vec4<f32>(input.position, 1.0);
    output.fragColor = input.normal * 0.5 + 0.5; // Normalize normal to [0, 1] for now
    output.texCoord0 = input.texCoord0; // Pass through texture coordinates
    return output;
  }

  @fragment
  fn fragmentMain(
    @location(0) fragColor: vec3<f32>,
    @location(1) texCoord0: vec2<f32>
  ) -> @location(0) vec4<f32> {
    return vec4<f32>(fragColor, 1.0);
  }
`;

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
    this.uniformBuffer = null;
    this.bindGroup = null;
    
    // Model-specific resources
    this.vertexBuffer = null;
    this.indexBuffer = null;
    this.indices = null;
    this.vertexData = null;
    
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
    
    // Create rendering pipeline
    this.#createPipeline();
    
    // Create uniform buffer
    this.#createUniformBuffer();
    
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
  }

  render(model, camera) {
    if (!model.isLoaded()) return;

    // Create transformation matrix
    const transformationMatrix = this.#createTransformationMatrix(model, camera);
    
    // Update uniforms
    this.device.queue.writeBuffer(this.uniformBuffer, 0, transformationMatrix);

    // Get current texture
    const currentTexture = this.context.getCurrentTexture();
    this.renderPassDescriptor.colorAttachments[0].view = currentTexture.createView();

    // Create command encoder and render pass
    const commandEncoder = this.device.createCommandEncoder();
    const passEncoder = commandEncoder.beginRenderPass(this.renderPassDescriptor);

    // Issue drawing commands
    passEncoder.setPipeline(this.pipeline);
    passEncoder.setBindGroup(0, this.bindGroup);
    passEncoder.setVertexBuffer(0, this.vertexBuffer);
    
    if (this.indices.length > 0) {
      passEncoder.setIndexBuffer(this.indexBuffer, "uint32"); // Always 32-bit
      passEncoder.drawIndexed(this.indices.length);
    } else {
      passEncoder.draw(this.vertexData.length / 18); // 18 = full vertex struct
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

  #createPipeline() {
    // Create shader module
    const shaderModule = this.device.createShaderModule({ code: shaderCode });

    // Create bind group layout
    const bindGroupLayout = this.device.createBindGroupLayout({
      entries: [
        {
          binding: 0,
          visibility: GPUShaderStage.VERTEX,
          buffer: { type: "uniform" },
        },
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

  #createUniformBuffer() {
    this.uniformBuffer = this.device.createBuffer({
      size: 256, // Uniform buffers must be aligned to 256 bytes
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    // Create bind group
    this.bindGroup = this.device.createBindGroup({
      layout: this.pipeline.getBindGroupLayout(0),
      entries: [
        {
          binding: 0,
          resource: {
            buffer: this.uniformBuffer,
          },
        },
      ],
    });
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

  #createTransformationMatrix(model, camera) {
    const transformationMatrix = mat4.create();

    // Apply model transformation
    mat4.copy(transformationMatrix, model.getTransform());

    // Apply camera view matrix
    const viewMatrix = camera.getViewMatrix();
    mat4.multiply(transformationMatrix, viewMatrix, transformationMatrix);

    // Apply camera projection matrix
    const projectionMatrix = camera.getProjectionMatrix();
    mat4.multiply(transformationMatrix, projectionMatrix, transformationMatrix);

    return transformationMatrix;
  }
}
