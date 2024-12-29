import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
const { mat4, vec3 } = glMatrix;

// Define shaders (WGSL)
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

async function loadGLTF(url) {
  return new Promise((resolve, reject) => {
    const loader = new GLTFLoader();
    loader.load(
      url,
      (gltf) => {
        const geometry = gltf.scene.children[0].geometry; // Assuming a single mesh
        const attributes = geometry.attributes;

        // Extract positions, normals, and indices
        const positions = attributes.position.array;
        const normals = attributes.normal.array;
        const indices = geometry.index ? geometry.index.array : null;

        resolve({ positions, normals, indices });
      },
      undefined,
      (error) => reject(error)
    );
  });
}

function createTransformationMatrix(rotation) {
  // Create an identity matrix
  const transformationMatrix = mat4.create();

  // Step 1: Apply 90-degree rotation around the X-axis
  const xRotationMatrix = mat4.create();
  mat4.rotateX(xRotationMatrix, xRotationMatrix, Math.PI / 2); // 90 degrees in radians
  mat4.multiply(transformationMatrix, transformationMatrix, xRotationMatrix);

  // Step 2: Apply the dynamic rotation around the Z-axis
  const zRotationMatrix = mat4.create();
  mat4.rotateZ(zRotationMatrix, zRotationMatrix, rotation);
  mat4.multiply(transformationMatrix, transformationMatrix, zRotationMatrix);

  // Return the final transformation matrix as a Float32Array
  return transformationMatrix;
}

async function initWebGPU() {

  // Get the canvas and its WebGPU context
  const canvas = document.getElementById('gpuCanvas');

  const adapter = await navigator.gpu.requestAdapter();
  if (!adapter) {
    console.error('WebGPU adapter not available. Your hardware or browser may not support WebGPU.');
    return;
  }
  
  const device = await adapter.requestDevice();
  if (!device) {
    console.error('Failed to create WebGPU device.');
    return;
  }

  const context = canvas.getContext('webgpu');

  // Define the format for rendering
  const format = navigator.gpu.getPreferredCanvasFormat();
  context.configure({
    device,
    format,
    alphaMode: 'opaque',
  });

  // Create the shader module
  const shaderModule = device.createShaderModule({ code: shaderCode });

  const { positions, normals, indices } = await loadGLTF('https://raw.githubusercontent.com/KhronosGroup/glTF-Sample-Models/master/2.0/DamagedHelmet/glTF/DamagedHelmet.gltf');

  //console.log('Loaded GLTF data:', positions, normals, indices);

  const vertexData = new Float32Array(positions.length + normals.length);
  for (let i = 0, j = 0; i < positions.length; i += 3, j += 6) {
    vertexData[j] = positions[i];     // x
    vertexData[j + 1] = positions[i + 1]; // y
    vertexData[j + 2] = positions[i + 2]; // z
    vertexData[j + 3] = normals[i];   // nx
    vertexData[j + 4] = normals[i + 1];   // ny
    vertexData[j + 5] = normals[i + 2];   // nz
  }


  // Create the vertex buffer
  const vertexBuffer = device.createBuffer({
    size: vertexData.byteLength,
    usage: GPUBufferUsage.VERTEX,
    mappedAtCreation: true,
  });

  new Float32Array(vertexBuffer.getMappedRange()).set(vertexData);
  vertexBuffer.unmap();

  // Create the index buffer (if indices are provided)
  const indexBuffer = indices
  ? device.createBuffer({
      size: indices.byteLength,
      usage: GPUBufferUsage.INDEX | GPUBufferUsage.COPY_DST,
      mappedAtCreation: true,
    })
  : null;

  if (indices) {
    new Uint16Array(indexBuffer.getMappedRange()).set(indices);
    indexBuffer.unmap();
  }

  // Create the bind group layout
  const bindGroupLayout = device.createBindGroupLayout({
    entries: [
      {
        binding: 0, // Matches @group(0) @binding(0) in the shader
        visibility: GPUShaderStage.VERTEX, // Accessible in the vertex shader
        buffer: { type: 'uniform' }, // Specifies it's a uniform buffer
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
      entryPoint: 'vertexMain',
      buffers: [{
        arrayStride: 6 * Float32Array.BYTES_PER_ELEMENT,
        attributes: [
          {
            shaderLocation: 0, // Position: Matches @location(0) in the shader
            format: 'float32x3', // vec3<f32>
            offset: 0, // Start at the beginning of each vertex
          },
          {
            shaderLocation: 1, // Normal: Matches @location(1) in the shader
            format: 'float32x3', // vec3<f32>
            offset: 3 * Float32Array.BYTES_PER_ELEMENT, // Start after position
          },
        ],
      }],
    },
    fragment: {
      module: shaderModule,
      entryPoint: 'fragmentMain',
      targets: [{ format }],
    },
    primitive: {
      topology: 'triangle-list',
    },
    layout: pipelineLayout
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
    colorAttachments: [{
      view: undefined, // Assigned per-frame
      loadOp: 'clear',
      clearValue: { r: 0.0, g: 0.2, b: 0.4, a: 1.0 }, // Light blue background
      storeOp: 'store',
    }],
  };

  // Rotation for the triangle
  let rotationAngle = 0;
  let isRotating = true; // Track whether the rotation is active

  // Toggle rotation state on canvas click
  canvas.addEventListener('click', () => {
    isRotating = !isRotating; // Toggle the rotation state
  });

  // Rendering loop
  function frame() {

    // Calculate the transformation matrix
    if (isRotating) {
      rotationAngle += 0.01; // Increment the angle for smooth rotation
      const transformationMatrix = createTransformationMatrix(rotationAngle);

      // Write the rotation matrix to the uniform buffer
      device.queue.writeBuffer(uniformBuffer, 0, transformationMatrix);
    }
    
    // Get the current texture from the canvas
    const currentTexture = context.getCurrentTexture();
    renderPassDescriptor.colorAttachments[0].view = currentTexture.createView();

    // Create the command encoder
    const commandEncoder = device.createCommandEncoder();
    const passEncoder = commandEncoder.beginRenderPass(renderPassDescriptor);

    // Issue drawing commands
    passEncoder.setPipeline(pipeline);
    passEncoder.setBindGroup(0, bindGroup); // Bind the uniform buffer
    passEncoder.setVertexBuffer(0, vertexBuffer); // Bind the vertex buffer
    if (indices) {
      passEncoder.setIndexBuffer(indexBuffer, "uint16");
      passEncoder.drawIndexed(indices.length);
    } else {
      passEncoder.draw(positions.length / 3);
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

// Initialize WebGPU
if (navigator.gpu) {
  initWebGPU();
} else {
  console.error('WebGPU is not supported on this browser.');
}
