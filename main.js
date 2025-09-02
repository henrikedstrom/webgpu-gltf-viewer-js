const { mat4, vec3 } = glMatrix;

import Camera from "./Camera.js";
import Model from "./Model.js";

//--------------------------------------------------------------------------------
// Shaders (WGSL)

const shaderCode = `

  @group(0) @binding(0) var<uniform> transformationMatrix: mat4x4<f32>;

  struct VertexOutput {
    @builtin(position) position: vec4<f32>,  // Position of the vertex
    @location(0) fragColor: vec3<f32>      // Color passed to the fragment shader
  };

  @vertex
  fn vertexMain(
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>
  ) -> VertexOutput {
    var output: VertexOutput;
    output.position = transformationMatrix * vec4<f32>(position, 1.0);
    output.fragColor = normal * 0.5 + 0.5; // Normalize normal to [0, 1]
    return output;
  }

  @fragment
  fn fragmentMain(
    @location(0) fragColor: vec3<f32>  // Input: interpolated color from the vertex shader
  ) -> @location(0) vec4<f32> {
    return vec4<f32>(fragColor, 1.0); // Output the color with full opacity
  }
`;

//--------------------------------------------------------------------------------
// Global objects

const camera = new Camera();
const model = new Model();
let isDragging = false;
let lastMouseX = 0;
let lastMouseY = 0;

function onMouseDown(event) {
  isDragging = true;
  lastMouseX = event.clientX;
  lastMouseY = event.clientY;
}

function onMouseUp() {
  isDragging = false;
}

function onMouseMove(event) {
  if (!isDragging) return;

  const deltaX = event.clientX - lastMouseX;
  const deltaY = event.clientY - lastMouseY;
  lastMouseX = event.clientX;
  lastMouseY = event.clientY;

  if (event.shiftKey) {
    camera.pan(deltaX, deltaY);
  } else {
    camera.tumble(-deltaX, -deltaY);
  }
}

function onMouseWheel(event) {
  const delta = -Math.sign(event.deltaY);
  camera.zoom(0, delta * 10);
}

function createTransformationMatrix(model, camera) {
  
  // Create the final transformation matrix
  const transformationMatrix = mat4.create();

  // Apply the model transformation
  mat4.copy(transformationMatrix, model.getTransform());

  // Multiply by the camera's view matrix
  const viewMatrix = camera.getViewMatrix();
  mat4.multiply(transformationMatrix, viewMatrix, transformationMatrix);

  // Multiply by the camera's projection matrix
  const projectionMatrix = camera.getProjectionMatrix();
  mat4.multiply(transformationMatrix, projectionMatrix, transformationMatrix);

  // Return the final transformation matrix as a Float32Array
  return transformationMatrix;
}


//--------------------------------------------------------------------------------
// Initialization

async function launchApp() {
  // Get the canvas and its WebGPU context
  const canvas = document.getElementById("gpuCanvas");

  // Event Listeners for camera controls
  canvas.addEventListener("mousedown", (event) => onMouseDown(event));
  canvas.addEventListener("mousemove", (event) => onMouseMove(event));
  canvas.addEventListener("mouseup", () => onMouseUp());
  canvas.addEventListener("wheel", (event) => onMouseWheel(event));
  canvas.addEventListener("contextmenu", (event) => event.preventDefault());
  canvas.addEventListener("wheel", (event) => event.preventDefault());

  camera.resizeViewport(canvas.width, canvas.height);

  // Adapter
  const adapter = await navigator.gpu.requestAdapter();
  if (!adapter) {
    console.error(
      "WebGPU adapter not available. Your hardware or browser may not support WebGPU."
    );
    return;
  }

  // Device
  const device = await adapter.requestDevice();
  if (!device) {
    console.error("Failed to create WebGPU device.");
    return;
  }

  // Define the format for rendering
  const format = navigator.gpu.getPreferredCanvasFormat();
  const context = canvas.getContext("webgpu");
  context.configure({
    device,
    format,
    alphaMode: "opaque",
  });

  // Create the depth texture
  const depthTexture = device.createTexture({
    size: [canvas.width, canvas.height, 1],
    format: "depth24plus", // Depth format
    usage: GPUTextureUsage.RENDER_ATTACHMENT, // Used as a render attachment
  });

  const depthTextureView = depthTexture.createView(); // Create a view of the depth texture

  // Create the shader module
  const shaderModule = device.createShaderModule({ code: shaderCode });

  // Load the glTF model
  try {
    await model.load(
      "https://raw.githubusercontent.com/KhronosGroup/glTF-Sample-Models/master/2.0/DamagedHelmet/glTF/DamagedHelmet.gltf"
    );
  } catch (error) {
    console.error("Failed to load model:", error);
    return;
  }

  // Get vertex data from the model
  const vertexData = model.getVertices();
  const indices = model.getIndices();

  // Create the vertex buffer
  const vertexBuffer = device.createBuffer({
    size: vertexData.byteLength,
    usage: GPUBufferUsage.VERTEX,
    mappedAtCreation: true,
  });

  new Float32Array(vertexBuffer.getMappedRange()).set(vertexData);
  vertexBuffer.unmap();

  // Create the index buffer (if indices are provided)
  const indexBuffer = indices.length > 0
    ? device.createBuffer({
        size: indices.byteLength,
        usage: GPUBufferUsage.INDEX | GPUBufferUsage.COPY_DST,
        mappedAtCreation: true,
      })
    : null;

  if (indices.length > 0) {
    if (indices instanceof Uint16Array) {
      new Uint16Array(indexBuffer.getMappedRange()).set(indices);
    } else {
      new Uint32Array(indexBuffer.getMappedRange()).set(indices);
    }
    indexBuffer.unmap();
  }

  // Create the bind group layout
  const bindGroupLayout = device.createBindGroupLayout({
    entries: [
      {
        binding: 0, // Matches @group(0) @binding(0) in the shader
        visibility: GPUShaderStage.VERTEX, // Accessible in the vertex shader
        buffer: { type: "uniform" }, // Specifies it's a uniform buffer
      },
    ],
  });

  // Create the pipeline layout
  const pipelineLayout = device.createPipelineLayout({
    bindGroupLayouts: [bindGroupLayout],
  });

  // Create the pipeline
  const pipeline = device.createRenderPipeline({
    vertex: {
      module: shaderModule,
      entryPoint: "vertexMain",
      buffers: [
        {
          arrayStride: 6 * Float32Array.BYTES_PER_ELEMENT,
          attributes: [
            {
              shaderLocation: 0, // Position: Matches @location(0) in the shader
              format: "float32x3", // vec3<f32>
              offset: 0, // Start at the beginning of each vertex
            },
            {
              shaderLocation: 1, // Normal: Matches @location(1) in the shader
              format: "float32x3", // vec3<f32>
              offset: 3 * Float32Array.BYTES_PER_ELEMENT, // Start after position
            },
          ],
        },
      ],
    },
    fragment: {
      module: shaderModule,
      entryPoint: "fragmentMain",
      targets: [{ format }],
    },
    primitive: {
      topology: "triangle-list",
    },
    depthStencil: {
      format: "depth24plus", // Match the depth texture format
      depthWriteEnabled: true, // Enable depth writes
      depthCompare: "less", // Compare depth values: closer fragments win
    },
    layout: pipelineLayout,
  });

  // Create the uniform buffer
  const uniformBuffer = device.createBuffer({
    size: 256, // Uniform buffers must be aligned to 256 bytes
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });

  // Create a bind group to bind the uniform buffer to the shader
  const bindGroup = device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      {
        binding: 0,
        resource: {
          buffer: uniformBuffer,
        },
      },
    ],
  });

  // Create the render pass descriptor
  const renderPassDescriptor = {
    colorAttachments: [
      {
        view: undefined, // Assigned per-frame
        loadOp: "clear",
        clearValue: { r: 0.0, g: 0.2, b: 0.4, a: 1.0 }, // Light blue background
        storeOp: "store",
      },
    ],
    depthStencilAttachment: {
      view: depthTextureView, // Use the depth texture
      depthLoadOp: "clear", // Clear depth at the start of the pass
      depthClearValue: 1.0, // Depth is cleared to the farthest value
      depthStoreOp: "store", // Store the depth values after the pass
    },
  };

  // Animation control
  let isAnimating = true; // Track whether the animation is active

  // Toggle animation state on key press
  window.addEventListener("keydown", (event) => {
    // Check if the pressed key is 'a'
    if (event.key === "a" || event.key === "A") {
      isAnimating = !isAnimating;
    }
  });

  // Rendering loop
  let lastTime = null;
  function frame(currentTime = performance.now()) {
    // Calculate deltaTime safely
    let deltaTime = 16.67; // Default to 60 FPS
    if (lastTime !== null) {
      deltaTime = currentTime - lastTime;
      // Clamp deltaTime to reasonable bounds
      if (deltaTime <= 0 || deltaTime > 100) {
        deltaTime = 16.67;
      }
    }
    lastTime = currentTime;

    // Skip if model not loaded yet
    if (!model.isLoaded()) {
      requestAnimationFrame(frame);
      return;
    }

    // Update the model and uniforms
    model.update(deltaTime, isAnimating);
    const transformationMatrix = createTransformationMatrix(model, camera);
    device.queue.writeBuffer(uniformBuffer, 0, transformationMatrix);

    // Get the current texture from the canvas
    const currentTexture = context.getCurrentTexture();
    renderPassDescriptor.colorAttachments[0].view = currentTexture.createView();

    // Create the command encoder
    const commandEncoder = device.createCommandEncoder();
    const passEncoder = commandEncoder.beginRenderPass(renderPassDescriptor);

    // Issue drawing commands
    passEncoder.setPipeline(pipeline);
    passEncoder.setBindGroup(0, bindGroup);
    passEncoder.setVertexBuffer(0, vertexBuffer);
    if (indices.length > 0) {
      passEncoder.setIndexBuffer(indexBuffer, indices instanceof Uint16Array ? "uint16" : "uint32");
      passEncoder.drawIndexed(indices.length);
    } else {
      passEncoder.draw(vertexData.length / 6); // 6 = 3 position + 3 normal
    }
    passEncoder.end();

    // Submit the command buffer
    device.queue.submit([commandEncoder.finish()]);

    // Request the next animation frame
    requestAnimationFrame(frame);
  }

  // Start rendering
  frame();
}

// Initialize App
if (navigator.gpu) {
  launchApp();
} else {
  console.error("WebGPU is not supported on this browser.");
}
