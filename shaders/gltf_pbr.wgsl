//---------------------------------------------------------------------
// Uniforms

struct GlobalUniforms {
  viewMatrix: mat4x4<f32>,
  projectionMatrix: mat4x4<f32>,
  modelMatrix: mat4x4<f32>,
  normalMatrix: mat4x4<f32>,
  cameraPositionWorld: vec3<f32>
};

struct MaterialUniforms {
  baseColorFactor    : vec4<f32>, // RGBA
  emissiveFactor     : vec3<f32>, // RGB emissive factor
  alphaMode          : f32,       // 0=OPAQUE,1=MASK,2=BLEND
  metallicFactor     : f32,
  roughnessFactor    : f32,
  normalScale        : f32,       // Scale applied to sampled normal perturbation
  occlusionStrength  : f32,       // Strength of AO map (0..1)
  alphaCutoff        : f32,       // Used when alphaMode==MASK
  _pad0              : vec3<f32>  // Padding to maintain 16-byte alignment (unused)
};

//---------------------------------------------------------------------
// Constants

const pi = 3.141592653589793;
const lightDirection = vec3<f32>(-0.25, -0.8, -1.0); // Hard-coded directional light
const lightColor = 2.0 * vec3<f32>(1.0, 0.95, 0.8); // Warm white light

//---------------------------------------------------------------------
// Vertex Input/Output

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
  @location(0) normalWorld: vec3<f32>,
  @location(1) tangentWorld: vec4<f32>,
  @location(2) texCoord0: vec2<f32>,
  @location(3) viewDirectionWorld: vec3<f32>
};

//---------------------------------------------------------------------
// Bind Groups

@group(0) @binding(0) var<uniform> globalUniforms: GlobalUniforms;
@group(0) @binding(1) var<uniform> materialUniforms: MaterialUniforms;
@group(0) @binding(2) var textureSampler: sampler;
@group(0) @binding(3) var baseColorTexture: texture_2d<f32>;
@group(0) @binding(4) var metallicRoughnessTexture: texture_2d<f32>;
@group(0) @binding(5) var normalTexture: texture_2d<f32>;
@group(0) @binding(6) var occlusionTexture: texture_2d<f32>;
@group(0) @binding(7) var emissiveTexture: texture_2d<f32>;

//---------------------------------------------------------------------
// Utility Functions

fn clampedDot(a: vec3<f32>, b: vec3<f32>) -> f32 {
  return clamp(dot(a, b), 0.0, 1.0);
}

// Fresnel-Schlick approximation
fn fresnelSchlick(cosTheta: f32, f0: vec3<f32>) -> vec3<f32> {
  return f0 + (1.0 - f0) * pow(clamp(1.0 - cosTheta, 0.0, 1.0), 5.0);
}

// GGX Distribution function
fn distributionGGX(N: vec3<f32>, H: vec3<f32>, roughness: f32) -> f32 {
  let a = roughness * roughness;
  let a2 = a * a;
  let NdotH = max(dot(N, H), 0.0);
  let NdotH2 = NdotH * NdotH;
  
  let num = a2;
  var denom = (NdotH2 * (a2 - 1.0) + 1.0);
  denom = pi * denom * denom;
  
  return num / denom;
}

// Geometry function using Smith's method
fn geometrySchlickGGX(NdotV: f32, roughness: f32) -> f32 {
  let r = (roughness + 1.0);
  let k = (r * r) / 8.0;
  
  let num = NdotV;
  let denom = NdotV * (1.0 - k) + k;
  
  return num / denom;
}

fn geometrySmith(N: vec3<f32>, V: vec3<f32>, L: vec3<f32>, roughness: f32) -> f32 {
  let NdotV = max(dot(N, V), 0.0);
  let NdotL = max(dot(N, L), 0.0);
  let ggx2 = geometrySchlickGGX(NdotV, roughness);
  let ggx1 = geometrySchlickGGX(NdotL, roughness);
  
  return ggx1 * ggx2;
}

fn getNormal(input: VertexOutput) -> vec3<f32> {
  // Reconstruct the TBN matrix using interpolated normal and tangent
  let N = normalize(input.normalWorld);
  let T = normalize(input.tangentWorld.xyz);
  let B = cross(N, T) * input.tangentWorld.w; // Tangent.w is handedness
  let TBN = mat3x3<f32>(T, B, N);

  // Sample the normal map and remap from [0,1] to [-1,1]
  var sampledNormal = textureSample(normalTexture, textureSampler, input.texCoord0).xyz * 2.0 - 1.0;
  sampledNormal *= materialUniforms.normalScale; 

  // Compute the final normal in world space
  return normalize(TBN * sampledNormal);
}

//---------------------------------------------------------------------
// Vertex Shader

@vertex
fn vertexMain(input: VertexInput) -> VertexOutput {
  // Transform to world space
  let worldPosition = globalUniforms.modelMatrix * vec4<f32>(input.position, 1.0);
  let worldNormal = normalize((globalUniforms.normalMatrix * vec4<f32>(input.normal, 0.0)).xyz);
  let worldTangent = vec4<f32>(
    normalize((globalUniforms.normalMatrix * vec4<f32>(input.tangent.xyz, 0.0)).xyz),
    input.tangent.w
  );
  
  var output: VertexOutput;
  output.position = globalUniforms.projectionMatrix * globalUniforms.viewMatrix * worldPosition;
  output.normalWorld = worldNormal;
  output.tangentWorld = worldTangent;
  output.texCoord0 = input.texCoord0;
  output.viewDirectionWorld = globalUniforms.cameraPositionWorld - worldPosition.xyz;
  
  return output;
}

//---------------------------------------------------------------------
// Fragment Shader

@fragment
fn fragmentMain(input: VertexOutput) -> @location(0) vec4<f32> {
  // Sample textures
  let baseColor = textureSample(baseColorTexture, textureSampler, input.texCoord0) * materialUniforms.baseColorFactor;
  let metallicRoughness = textureSample(metallicRoughnessTexture, textureSampler, input.texCoord0);
  
  // Extract metallic and roughness from texture (B and G channels respectively)
  let metallic = metallicRoughness.b * materialUniforms.metallicFactor;
  let roughness = metallicRoughness.g * materialUniforms.roughnessFactor;
  
  // Get normal from normal map
  let normal = getNormal(input);
  
  // Lighting vectors
  let L = normalize(-lightDirection);
  let V = normalize(input.viewDirectionWorld);
  let H = normalize(L + V);
  
  // Calculate reflectance at normal incidence
  let F0 = mix(vec3<f32>(0.04), baseColor.rgb, metallic);
  
  // Cook-Torrance BRDF components
  let NDF = distributionGGX(normal, H, roughness);
  let G = geometrySmith(normal, V, L, roughness);
  let F = fresnelSchlick(max(dot(H, V), 0.0), F0);
  
  let kS = F;
  let kD = (1.0 - kS) * (1.0 - metallic);
  
  let NdotL = max(dot(normal, L), 0.0);
  let NdotV = max(dot(normal, V), 0.0);
  
  let numerator = NDF * G * F;
  let denominator = 4.0 * NdotV * NdotL + 0.0001;
  let specular = numerator / denominator;
  
  // Lambertian diffuse
  let diffuse = kD * baseColor.rgb / pi;
  
  // Ambient occlusion
  let aoSample = textureSample(occlusionTexture, textureSampler, input.texCoord0).r;
  let ao = mix(1.0, aoSample, materialUniforms.occlusionStrength);
  let ambient = vec3<f32>(0.01) * baseColor.rgb * ao;

  // Emissive
  let emissiveTex = textureSample(emissiveTexture, textureSampler, input.texCoord0).rgb;
  let emissive = emissiveTex * materialUniforms.emissiveFactor;
  
  // Alpha handling (mask discard)
  let alphaMode = materialUniforms.alphaMode;
  let alphaCutoff = materialUniforms.alphaCutoff;
  if (alphaMode == 1.0 && baseColor.a < alphaCutoff) { discard; }

  // Combine lighting
  let finalColor = ambient + (diffuse + specular) * lightColor * NdotL + emissive;
  
  // Simple tone mapping and gamma correction
  var color = finalColor / (finalColor + vec3<f32>(1.0));
  color = pow(color, vec3<f32>(1.0/2.2));

  return vec4<f32>(color, baseColor.a);
}