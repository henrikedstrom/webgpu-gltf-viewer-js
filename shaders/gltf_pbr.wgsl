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
