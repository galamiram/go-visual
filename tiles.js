const VERT = `#version 300 es
void main() {
  vec2 p = vec2(float((gl_VertexID << 1) & 2), float(gl_VertexID & 2));
  gl_Position = vec4(p * 2.0 - 1.0, 0.0, 1.0);
}`;

// Face field pass: paints the portrait as data, not colour.
// R = relief (dark..light), G = radial distance from the face centre (1 = face edge),
// B = figure presence, A = signed distance to the head/shoulder silhouette.
const FIELD = `#version 300 es
precision highp float;
uniform vec2 uScreen;
uniform vec2 uFieldRes;
uniform vec4 uFace;
uniform vec4 uP[2];
uniform sampler2D uEigen;
uniform sampler2D uCam;
uniform vec2 uCamO;
uniform float uCamK;
out vec4 outField;

float sdEll(vec2 p, vec2 r) { return (length(p / r) - 1.0) * min(r.x, r.y); }

float sdBox(vec2 p, vec2 b) {
  vec2 d = abs(p) - b;
  return length(max(d, 0.0)) + min(max(d.x, d.y), 0.0);
}

float cov(float d, float e) { return 1.0 - smoothstep(-e, e, d); }

float camLum(vec2 uv) { return dot(texture(uCam, uv).rgb, vec3(0.299, 0.587, 0.114)); }

float hash(vec2 p) {
  vec3 p3 = fract(vec3(p.xyx) * 0.1031);
  p3 += dot(p3, p3.yzx + 33.33);
  return fract((p3.x + p3.y) * p3.z);
}

float vnoise(vec2 p) {
  vec2 i = floor(p);
  vec2 f = fract(p);
  f = f * f * (3.0 - 2.0 * f);
  return mix(mix(hash(i), hash(i + vec2(1.0, 0.0)), f.x), mix(hash(i + vec2(0.0, 1.0)), hash(i + vec2(1.0, 1.0)), f.x), f.y);
}

// The model's 64x64 crop in head units: eyes land at (+-0.33, -0.1), mouth near 0.55
float eigen(vec2 q) { return texture(uEigen, (q - vec2(-0.682, -0.452)) / 1.364).r; }

void main() {
  vec2 uv = gl_FragCoord.xy / uFieldRes;
  vec2 d0 = (vec2(uv.x, 1.0 - uv.y) * uScreen - uFace.xy) / uFace.z;
  float ca = cos(uFace.w);
  float sa = sin(uFace.w);
  vec2 q = vec2(d0.x * ca + d0.y * sa, -d0.x * sa + d0.y * ca);

  float faceW = uP[0].x, faceH = uP[0].y, faceCY = uP[0].z, hairW = uP[0].w;
  float hairTop = uP[1].x, hairLen = uP[1].y, shoulderW = uP[1].z, neckW = uP[1].w;
  float wobble = (vnoise(q * 4.0) - 0.5) * 0.1 + (vnoise(q * 11.0) - 0.5) * 0.04;

  // Silhouette: soft masses for hair, neck and shoulders; nothing is outlined
  float dSh = sdEll(q - vec2(0.0, 2.15), vec2(shoulderW, 0.9));
  float dNeck = sdBox(q - vec2(0.0, faceCY + faceH + 0.2), vec2(neckW, 0.42));
  float dHead = sdEll(q - vec2(0.0, -0.14 + 0.22 * hairLen), vec2(hairW, 1.08 + 0.25 * hairLen)) + wobble;
  vec2 fc = vec2(0.0, faceCY + 0.05);
  float dFace = sdEll(q - fc, vec2(faceW, faceH));
  float dSil = min(min(dSh, dNeck), min(dHead, dFace));

  float r = 0.5;
  r = mix(r, (0.28 + 0.1 * vnoise(q * 3.0)) * (1.0 - 0.4 * clamp(abs(q.x) / shoulderW, 0.0, 1.0)), cov(dSh, 0.1));
  float nx = clamp(q.x / neckW, -1.0, 1.0);
  float underJaw = smoothstep(faceCY + faceH - 0.1, faceCY + faceH + 0.45, q.y);
  r = mix(r, (0.16 + 0.2 * underJaw) * (0.6 + 0.4 * (1.0 - nx * nx)), cov(dNeck, 0.16));
  r = mix(r, 0.1 + 0.12 * vnoise(q * 2.5 + 3.1) + 0.05 * vnoise(q * 7.0), cov(dHead, 0.08));

  // Face from the statistical model, feathered into the hair; the forehead is carried
  // up past the top of the learned crop
  float fr = length((q - fc) / vec2(faceW, faceH));
  float shade = 1.0 - 0.32 * smoothstep(0.5, 1.0, fr);
  float brow = (eigen(vec2(-0.25, -0.4)) + eigen(vec2(0.0, -0.42)) + eigen(vec2(0.25, -0.4))) / 3.0;
  float skin = (eigen(vec2(-0.36, 0.22)) + eigen(vec2(0.36, 0.22)) + brow) / 3.0;
  // The learned crop is square and narrower than the oval: fade it out before its edges and
  // let the face's own skin tone, shaded like a rounded surface, fill the rest of the oval
  float inCrop = (1.0 - smoothstep(0.4, 0.6, abs(q.x))) * (1.0 - smoothstep(0.66, 0.88, q.y));
  float e = mix(skin * shade, eigen(q), inCrop);
  e = mix(e, brow * shade, 1.0 - smoothstep(-0.68, -0.44, q.y));
  float hairline = faceCY - faceH * 0.72 + hairTop + wobble * 0.6;
  float wFace = (1.0 - smoothstep(0.8, 1.06, fr + wobble * 0.4)) * smoothstep(hairline - 0.1, hairline + 0.16, q.y);
  r = mix(r, clamp((e - 0.1) / 0.75, 0.0, 1.0), wFace);

  float mask = cov(dSil, 0.22);

  // Camera: the real person's shading takes over from the model; local contrast is measured
  // at the hexagon scale, so smaller cells pick up finer detail
  if (uCamK > 0.001) {
    vec2 cuv = vec2(uv.x, 1.0 - uv.y);
    vec2 o = uCamO;
    float lr = camLum(cuv);
    float lw = 0.25 * (camLum(cuv + vec2(o.x, 0.0)) + camLum(cuv - vec2(o.x, 0.0))
                     + camLum(cuv + vec2(0.0, o.y)) + camLum(cuv - vec2(0.0, o.y)));
    float dog = clamp(0.5 + (lr - lw) * 2.2 + (lr - 0.5) * 0.9, 0.0, 1.0);
    r = mix(r, dog, uCamK * mask * 0.85);
  }

  outField = vec4(clamp(r, 0.0, 1.0), clamp(fr / 1.5, 0.0, 1.0), mask, clamp(0.5 + 0.5 * dSil, 0.0, 1.0));
}`;

const FRAG = `#version 300 es
precision highp float;
uniform vec2 uRes;
uniform float uTime;
uniform float uR;
uniform float uHue;
uniform float uBreath;
uniform float uBreathK;
uniform float uTravel;
uniform float uPalW;
uniform vec2 uTilt;
uniform vec2 uLight;
uniform float uDepth;
uniform float uSurge;
uniform float uSurgeT;
uniform float uTileTilt;
uniform float uGloss;
uniform float uBob;
uniform float uEye;
uniform sampler2D uField;
out vec4 outColor;

#define TAU 6.28318530718
#define PI 3.14159265359
#define H(i) h13(sd + vec3(0.0, 0.0, (i) * 3.17))

float AA;

float h13(vec3 p3) {
  p3 = fract(p3 * 0.1031);
  p3 += dot(p3, p3.zyx + 31.32);
  return fract((p3.x + p3.y) * p3.z);
}

float vn3(vec3 p) {
  vec3 i = floor(p);
  vec3 f = fract(p);
  f = f * f * (3.0 - 2.0 * f);
  vec2 e = vec2(1.0, 0.0);
  return mix(
    mix(mix(h13(i), h13(i + e.xyy), f.x), mix(h13(i + e.yxy), h13(i + e.xxy), f.x), f.y),
    mix(mix(h13(i + e.yyx), h13(i + e.xyx), f.x), mix(h13(i + e.yxx), h13(i + e.xxx), f.x), f.y),
    f.z);
}

vec3 hsv2rgb(vec3 c) {
  vec3 p = abs(fract(c.xxx + vec3(0.0, 2.0 / 3.0, 1.0 / 3.0)) * 6.0 - 3.0);
  return c.z * mix(vec3(1.0), clamp(p - 1.0, 0.0, 1.0), c.y);
}

float fillMask(float d) { return 1.0 - smoothstep(-AA, AA, d); }
float lineMask(float d, float w) { return 1.0 - smoothstep(w - AA, w + AA, abs(d)); }

vec2 rot2(vec2 p, float a) {
  float c = cos(a);
  float s = sin(a);
  return vec2(c * p.x - s * p.y, s * p.x + c * p.y);
}

float sdHexP(vec2 p, float ap) {
  const vec3 k = vec3(-0.866025404, 0.5, 0.577350269);
  p = abs(p.yx);
  p -= 2.0 * min(dot(k.xy, p), 0.0) * k.xy;
  p -= vec2(clamp(p.x, -k.z * ap, k.z * ap), ap);
  return length(p) * sign(p.y);
}

float sdUnevenCapsule(vec2 p, float r1, float r2, float h) {
  p.x = abs(p.x);
  float b = (r1 - r2) / h;
  float a = sqrt(1.0 - b * b);
  float k = dot(p, vec2(-b, a));
  if (k < 0.0) return length(p) - r1;
  if (k > a * h) return length(p - vec2(0.0, h)) - r2;
  return dot(p, vec2(a, b)) - r1;
}

float polyD(float a, float r, float m, float R1) {
  float seg = TAU / m;
  float f = mod(a, seg) - seg * 0.5;
  return r * cos(f) - R1 * cos(PI / m);
}

float layerSD(int type, float a, float r, float n, float R0, float R1, float w) {
  float seg = TAU / n;
  float fa = mod(a, seg) - seg * 0.5;
  vec2 q = r * vec2(cos(fa), sin(fa));
  float band = max(R0 - r, r - R1);
  float halfW = sin(seg * 0.5);
  if (type == 0) {
    vec2 c = vec2(mix(R0, R1, 0.55), 0.0);
    float rx = (R1 - R0) * 0.55;
    float ry = min(rx * (0.35 + 0.5 * w), c.x * halfW * 0.95);
    return (length((q - c) / vec2(rx, ry)) - 1.0) * min(rx, ry);
  }
  if (type == 1) {
    float t = 1.0 - abs(fa) / (seg * 0.5);
    return r - mix(R0, R1, pow(t, 1.5 + 2.0 * w));
  }
  if (type == 2) {
    float f = (r - R0) / (R1 - R0);
    return max((abs(fract(f * 2.5 - w) - 0.5) - 0.22) * (R1 - R0) / 2.5, band);
  }
  if (type == 3) return max(abs(q.y) - (0.018 + 0.04 * w), band);
  if (type == 4) {
    vec2 c = vec2(mix(R0, R1, 0.6), 0.0);
    float rr = min((R1 - R0) * (0.28 + 0.16 * w), c.x * halfW * 0.9);
    return length(q - c) - rr;
  }
  if (type == 5) {
    return mix(polyD(a, r, 6.0, R1), r - R1 * 0.92, w);
  }
  if (type == 6) {
    float tt = clamp((q.x - R0) / (R1 - R0), 0.0, 1.0);
    float hw = min((R1 - R0) * (0.14 + 0.22 * w), q.x * halfW);
    return max(abs(q.y) - hw * sin(PI * tt), max(R0 - q.x, q.x - R1));
  }
  if (type == 7) {
    vec2 pr = r * vec2(cos(a), sin(a));
    float rr = R1 * 0.5;
    float d = abs(r - rr);
    for (int i = 0; i < 6; i++) {
      float b = float(i) * TAU / 6.0;
      d = min(d, abs(length(pr - rr * vec2(cos(b), sin(b))) - rr));
    }
    return d - (0.022 + 0.03 * w);
  }
  if (type == 8) return max(-sin(a * n) * r / n * 1.5, band);
  if (type == 9) {
    float f = fract(a * n / TAU + (r - R0) * (1.2 + 2.5 * w));
    return max((abs(f - 0.5) - 0.22) * TAU * r / n, band);
  }
  if (type == 10) {
    float rr = mix(R0, R1, 0.5) + (R1 - R0) * 0.4 * cos(a * n);
    return abs(r - rr) - (0.025 + 0.035 * w);
  }
  if (type == 11) return min(polyD(a, r, 3.0, R1), polyD(a + PI, r, 3.0, R1));
  if (type == 12) return min(length(q - vec2(R1 * 0.72, 0.0)) - min(R1 * 0.28, R1 * 0.72 * halfW), r - R1 * 0.72);
  if (type == 13) return max(-sin(a * n) * sin((r - R0) * (18.0 + 20.0 * w)) * 0.02, band);
  if (type == 14) {
    vec2 val = mix(R0, R1, 0.3 + 0.3 * w) * vec2(cos(seg * 0.5), sin(seg * 0.5));
    vec2 e = val - vec2(R1, 0.0);
    return max(dot(vec2(q.x, abs(q.y)) - vec2(R1, 0.0), normalize(vec2(e.y, -e.x))), R0 * 0.5 - r);
  }
  if (type == 15) {
    float rA = mix(R0, R1, 0.72);
    float rB = mix(R0, R1, 0.28);
    float s1 = min((R1 - R0) * 0.2 * (0.7 + 0.6 * w), rA * halfW * 0.85);
    float fb = abs(fa) - seg * 0.5;
    float d2 = length(r * vec2(cos(fb), sin(fb)) - vec2(rB, 0.0)) - min(s1 * 0.7, rB * halfW * 0.85);
    return min(length(q - vec2(rA, 0.0)) - s1, d2);
  }
  if (type == 16) {
    float tt = clamp((q.x - R0) / (R1 - R0), 0.0, 1.0);
    float hw = min((R1 - R0) * (0.05 + 0.08 * w), q.x * halfW) * (1.0 - 0.8 * tt);
    return max(abs(q.y) - hw, max(R0 - q.x, q.x - R1));
  }
  if (type == 17) {
    vec2 c = vec2(mix(R0, R1, 0.55), 0.0);
    float rr = min((R1 - R0) * 0.42, c.x * halfW * 0.95);
    return max(length(q - c) - rr, -(length(q - c - vec2(rr * (0.35 + 0.4 * w), 0.0)) - rr * 0.9));
  }
  if (type == 18) {
    float rx = (R1 - R0) * 0.5;
    float ry = min(rx * 0.3, mix(R0, R1, 0.5) * halfW * 0.5);
    vec2 qq = rot2(vec2(q.x, abs(q.y)) - vec2(mix(R0, R1, 0.5), ry * 0.9), 0.25 + 0.3 * w);
    return (length(qq / vec2(rx, ry)) - 1.0) * min(rx, ry);
  }
  if (type == 19) {
    float k = halfW * 1.15;
    float rc = R1 * k / (1.0 + k);
    float Rc = R1 - rc;
    return max(min(r - Rc, length(q - vec2(Rc, 0.0)) - rc), R0 * (0.3 + 0.5 * w) - r);
  }
  if (type == 20) {
    float rm = mix(R0, R1, 0.5);
    float amp = (R1 - R0) * 0.32 * cos(a * n);
    return min(abs(r - rm - amp), abs(r - rm + amp)) - (0.018 + 0.022 * w);
  }
  if (type == 21) {
    float pitch = (R1 - R0) * 0.34;
    float f = fract((q.x - abs(q.y) * (1.2 + 1.5 * w)) / pitch);
    return max((abs(f - 0.5) - 0.2) * pitch, max(band, abs(q.y) - q.x * halfW * 0.9));
  }
  if (type == 22) {
    float cx = mix(R0, R1, 0.35);
    float rr = min((R1 - R0) * (0.2 + 0.1 * w), cx * halfW * 0.9);
    return sdUnevenCapsule(vec2(q.y, q.x - cx), rr, 0.012, max(R1 - cx, rr + 0.02));
  }
  vec2 c = vec2(mix(R0, R1, 0.5), 0.0);
  float rx = (R1 - R0) * 0.5;
  float ry = min(rx * (0.3 + 0.4 * w), c.x * halfW * 0.95);
  vec2 d2 = abs(q - c);
  return (d2.x / rx + d2.y / ry - 1.0) * min(rx, ry) * 0.7;
}

vec2 hexRound(vec2 qr) {
  vec3 c = vec3(qr.x, qr.y, -qr.x - qr.y);
  vec3 rc = floor(c + 0.5);
  vec3 d = abs(rc - c);
  if (d.x > d.y && d.x > d.z) rc.x = -rc.y - rc.z;
  else if (d.y > d.z) rc.y = -rc.x - rc.z;
  return rc.xy;
}

const int RF[7] = int[7](2, 2, 4, 5, 6, 1, 3);
const vec2 HEX6[6] = vec2[6](
  vec2(0.866025, 0.5), vec2(0.0, 1.0), vec2(-0.866025, 0.5),
  vec2(-0.866025, -0.5), vec2(0.0, -1.0), vec2(0.866025, -0.5)
);

// Gentle organic curl modulation that keeps contour waves fluid without scrambling facial shapes
vec3 liquidFlow(vec2 u, float t, float relief, float mask) {
  float s = t * 0.45;
  vec2 q = vec2(
    sin(u.x * 1.7 + u.y * 1.2 + s * 0.8) + 0.5 * cos(u.y * 2.2 - s * 0.65),
    cos(u.x * 1.5 - u.y * 1.8 - s * 0.75) + 0.5 * sin(u.x * 2.1 + s * 0.6)
  ) * 0.35;
  vec2 p = u + q;
  float w1 = sin(p.x * 1.8 - p.y * 1.4 + s * 0.7)
           + 0.55 * cos(p.x * 1.3 + p.y * 2.1 - s * 0.55);
  float w2 = cos(p.x * 1.6 + p.y * 1.7 - s * 0.62)
           + 0.5 * sin(p.x * 2.2 - p.y * 1.3 + s * 0.48);
  return vec3(w1 * 0.14 + w2 * 0.10, w1 * 0.25 - w2 * 0.22, (w1 - w2) * 0.16);
}

const int NT = 24;
const float NS[10] = float[10](6.0, 12.0, 6.0, 8.0, 12.0, 5.0, 10.0, 4.0, 16.0, 6.0);

// One mandala design, fully determined by the seed sd; tile() swaps seeds over time.
vec3 design(vec2 p, vec3 sd, vec3 pal[7], vec3 col, vec3 flow, float breath) {
  float sym = NS[int(floor(H(1.0) * 10.0))];
  int frame = int(floor(H(3.0) * 10.0));
  int center = int(floor(H(5.0) * 9.0));
  int nL = 2 + int(floor(H(7.5) * 3.0));
  float cellPhase = H(1.5) * TAU;
  float r = length(p);

  float dir = H(0.5) < 0.5 ? -1.0 : 1.0;
  float a0 = atan(p.y, p.x);
  float fRot = uTime * 0.65 * dir + flow.z * 0.8;
  vec2 fp = rot2(p, fRot);
  float fang = a0 + fRot;
  float seg60 = PI / 3.0;
  float fPulse = 1.0 + 0.08 * sin(uTime * 1.1 + cellPhase);

  if (frame == 1) col = mix(col, pal[3], lineMask(sdHexP(fp, 0.78 * fPulse), 0.028));
  else if (frame == 2) {
    col = mix(col, pal[3], lineMask(sdHexP(fp, 0.82 * fPulse), 0.024));
    col = mix(col, pal[5], lineMask(sdHexP(fp, 0.65 / fPulse), 0.022));
  } else if (frame == 3) col = mix(col, pal[3], lineMask(r - 0.84 * fPulse, 0.03));
  else if (frame == 4) col = mix(col, pal[2], lineMask(sdHexP(fp.yx, 0.74 * fPulse), 0.026));
  else if (frame == 5) {
    float fa = mod(fang - PI / 6.0 + seg60 * 0.5, seg60) - seg60 * 0.5;
    col = mix(col, pal[3], fillMask(length(r * vec2(cos(fa), sin(fa)) - vec2(0.80 * fPulse, 0.0)) - 0.08));
  } else if (frame == 6) col = mix(col, pal[1], fillMask(sdHexP(fp, 0.74 * fPulse)));
  else if (frame == 7) {
    float fa = mod(fang + seg60 * 0.5, seg60) - seg60 * 0.5;
    col = mix(col, pal[1], fillMask(length(r * vec2(cos(fa), sin(fa)) - vec2(0.64 * fPulse, 0.0)) - 0.24));
  } else if (frame == 8) {
    float s12 = PI / 6.0;
    float fa = mod(fang + s12 * 0.5, s12) - s12 * 0.5;
    col = mix(col, pal[3], fillMask(length(r * vec2(cos(fa), sin(fa)) - vec2(0.84 * fPulse, 0.0)) - 0.05));
  } else if (frame == 9) {
    col = mix(col, pal[5], lineMask(r - 0.86 * fPulse, 0.018));
    col = mix(col, pal[3], lineMask(r - 0.78 * fPulse, 0.018));
  }

  for (int L = 0; L < 4; L++) {
    if (L >= nL) break;
    float base = 10.0 + float(L) * 12.0;
    float n = H(base + 3.0) < 0.3 ? NS[int(floor(H(base + 3.5) * 10.0))] : sym;

    // Outward-rippling radial bloom across the concentric layers
    float bloom = sin(uTime * 1.2 + flow.x * 2.1 + cellPhase - float(L) * 1.35);
    float R1 = mix(0.92, 0.3, float(L) / float(nL - 1)) * mix(0.86, 1.0, H(base + 4.0));
    R1 = clamp(R1 * (1.0 + 0.12 * bloom) * (0.84 + 0.24 * breath), 0.20, 0.96);
    float R0 = R1 * (0.20 + 0.36 * (0.5 + 0.5 * sin(uTime * 0.9 + H(base + 5.0) * TAU + float(L))));

    // Continuous shape-parameter breathing (w in [0.08, 0.92])
    float w = 0.5 + 0.42 * sin(uTime * 0.95 + flow.y * 1.7 + H(base + 6.0) * TAU);
    bool inner = L == nL - 1;
    int fr = inner ? (H(base + 7.0) < 0.55 ? 4 : 3) : RF[min(6, int(floor(H(base + 7.0) * 7.0)))];
    if (!inner && fr == 4) fr = 2;
    float hl = H(base + 8.0);
    int lr = hl < 0.65 ? 3 : (hl < 0.85 ? 1 : -1);

    // Counter-rotation + spiral petal curl inside each hexagon
    float lDir = (L == 1 || L == 3) ? -dir : dir;
    float spin = (0.9 + 1.2 * H(base + 10.0)) * lDir;
    float twist = 0.68 * sin(uTime * 0.75 + cellPhase + float(L) * 2.0);
    float rot0 = H(base + 11.0) < 0.5 ? 0.0 : PI / n;
    float a = a0 + rot0 + uTime * spin + flow.z * 1.5 * lDir + (r - R0) * twist;

    // Continuous transformation through the geometries, in a random order per layer
    float mPhase = uTime * (0.45 + 0.35 * H(base + 9.0)) + H(base + 2.0) * 97.0;
    float k = floor(mPhase);
    int t0 = int(floor(H(300.0 + base * 7.0 + mod(k, 97.0)) * float(NT)));
    int t1 = int(floor(H(300.0 + base * 7.0 + mod(k + 1.0, 97.0)) * float(NT)));
    float mf = smoothstep(0.12, 0.88, fract(mPhase));
    float d = mix(layerSD(t0, a, r, n, R0, R1, w), layerSD(t1, a, r, n, R0, R1, 1.0 - w), mf);

    float style = H(base + 12.0);
    if (style < 0.7) {
      col = mix(col, pal[fr], fillMask(d));
      if (lr >= 0) col = mix(col, pal[lr], lineMask(d, 0.024 + 0.024 * H(base + 1.5)));
    } else if (style < 0.88) {
      col = mix(col, pal[fr], lineMask(d, 0.03 + 0.02 * H(base + 1.5)));
    } else {
      col = mix(col, pal[fr], lineMask(d, 0.022));
      col = mix(col, pal[lr < 0 ? 3 : lr], lineMask(d + 0.07, 0.018));
    }
  }

  float cBloom = (1.0 + 0.22 * sin(uTime * 1.35 + flow.y * 2.0 + cellPhase)) * (0.78 + 0.4 * breath);
  float cr = r / cBloom;
  float cSpin = -uTime * (1.1 + 0.9 * H(9.0)) * dir + flow.z * 1.6;
  float ca = a0 + cSpin;
  vec2 cp = rot2(p, cSpin) / cBloom;
  if (center == 0) {
    col = mix(col, pal[4], fillMask(cr - 0.17));
    col = mix(col, pal[3], lineMask(cr - 0.17, 0.025));
    col = mix(col, pal[3], fillMask(cr - 0.07));
  } else if (center == 1) {
    col = mix(col, pal[3], fillMask((length(cp / vec2(1.0, 0.5)) - 0.26) * 0.5));
    col = mix(col, pal[4], fillMask(r - 0.09));
    col = mix(col, pal[1], fillMask(r - 0.045));
  } else if (center == 2) {
    float fa = mod(ca, seg60) - seg60 * 0.5;
    float t = 1.0 - abs(fa) / (seg60 * 0.5);
    float d = cr - mix(0.09, 0.24, t * t);
    col = mix(col, pal[5], fillMask(d));
    col = mix(col, pal[3], lineMask(d, 0.02));
  } else if (center == 3) {
    col = mix(col, pal[3], fillMask(cr - 0.2));
    col = mix(col, pal[4], fillMask(cr - 0.13));
    col = mix(col, pal[3], fillMask(cr - 0.06));
  } else if (center == 4) {
    float fa = mod(ca, seg60) - seg60 * 0.5;
    vec2 q = r * vec2(cos(fa), sin(fa));
    float d = min(length(q - vec2(0.15 * cBloom, 0.0)) - 0.05, r - 0.065);
    col = mix(col, pal[3], fillMask(d));
  } else if (center == 5) {
    float d = polyD(ca, cr, 6.0, 0.19);
    col = mix(col, pal[4], fillMask(d));
    col = mix(col, pal[3], lineMask(d, 0.02));
  } else if (center == 6) {
    float s5 = TAU / 5.0;
    float fa = mod(ca, s5) - s5 * 0.5;
    float d = cr * cos(fa) * (1.0 + 1.6 * abs(fa)) - 0.13;
    col = mix(col, pal[4], fillMask(d));
    col = mix(col, pal[1], fillMask(cr - 0.04));
  } else if (center == 7) {
    float d = max(length(cp - vec2(0.11, 0.0)) - 0.2, length(cp + vec2(0.11, 0.0)) - 0.2);
    col = mix(col, pal[4], fillMask(d));
    col = mix(col, pal[3], lineMask(d, 0.018));
    col = mix(col, pal[1], fillMask(cr - 0.04));
  } else {
    col = mix(col, pal[4], fillMask(cr - 0.2));
    col = mix(col, pal[3], lineMask(cr - 0.14, 0.014));
    col = mix(col, pal[3], lineMask(cr - 0.08, 0.014));
    col = mix(col, pal[3], fillMask(cr - 0.03));
  }
  return col;
}

// spread scales how far the mandala's colours stray from the cell's (h, s, v); low on the face,
// so each cell reads as its patch of shading rather than as its own picture.
vec3 tile(vec2 p, vec2 cid, vec3 hsv, vec3 flow, float breath, float spread) {
  vec3 sd = vec3(cid.x * 1.7 + 17.13, cid.y * 2.3 - 11.71, 0.0);
  float r = length(p);
  float cellPhase = H(1.5) * TAU;
  float radialWave = sin((flow.x * 1.8 - r * 0.75 + uTime * 0.38 + H(6.2)) * TAU);
  float h = fract(hsv.x + (H(6.0) - 0.5) * 0.045 - r * 0.08 + 0.04 * radialWave);
  float s = hsv.y;
  float v = clamp(hsv.z + 0.06 * radialWave, 0.06, 1.0);

  // Outer layers stay faithful to the hexagon's (h, s, v); pal[4] lights the core with a jewel accent
  float shift = (0.06 + 0.04 * sin((flow.z + H(6.5)) * TAU)) * spread;
  float coreHue = fract(h + (0.22 + 0.2 * (1.0 - uPalW) + 0.14 * sin(uTime * 0.35 + cellPhase)) * spread);
  #define SP(k) mix(1.0, k, spread)
  vec3 pal[7];
  pal[0] = hsv2rgb(vec3(h, s, v));
  pal[1] = hsv2rgb(vec3(fract(h - shift), min(1.0, s * SP(1.12)), clamp(v * SP(0.62), 0.05, 0.92)));
  pal[2] = hsv2rgb(vec3(fract(h + shift * 0.8), s * SP(0.76), min(1.0, v * SP(1.18) + 0.06 * spread)));
  pal[3] = hsv2rgb(vec3(fract(h + shift * 1.2), s * SP(0.45), min(1.0, v * SP(1.30) + 0.10 * spread)));
  pal[4] = hsv2rgb(vec3(coreHue, clamp(s * SP(0.95) + 0.1 * spread, 0.3, 1.0), clamp(v * SP(1.22) + 0.12 * spread, 0.06, 0.98)));
  pal[5] = hsv2rgb(vec3(fract(h + shift * 1.6), s * SP(0.86), clamp(v * SP(0.82), 0.06, 0.96)));
  pal[6] = hsv2rgb(vec3(fract(h - shift * 1.1), s * SP(0.92), clamp(v * SP(0.72), 0.05, 0.92)));
  #undef SP

  vec3 col = mix(pal[2], pal[0], smoothstep(0.0, 0.85, r));

  // Every cell is redesigned on its own random clock: the old mandala blooms outward and fades
  // while the next one grows in from the center.
  float period = 12.0 + 14.0 * H(40.0);
  float gt = uTime / period + H(41.0);
  float g = floor(gt);
  float x = smoothstep(1.0 - 1.8 / period, 1.0, fract(gt));
  vec3 sdg = sd + vec3(g * 5.31, g * 2.77, 0.0);
  vec3 outC = design(p / (1.0 + 0.35 * x), sdg, pal, col, flow, breath);
  if (x <= 0.0) return outC;
  vec3 inC = design(p * (1.0 + 0.35 * (1.0 - x)), sdg + vec3(5.31, 2.77, 0.0), pal, col, flow, breath);
  return mix(outC, inC, x);
}

vec4 fieldAt(vec2 px) {
  return texture(uField, clamp(vec2(px.x / uRes.x, 1.0 - px.y / uRes.y), vec2(0.001), vec2(0.999)));
}

// Floating tiles in true perspective. Each tile hovers at its own depth (toward the viewer is +z,
// in pixels): the face is a gentle sculpted layer, the background rolls with the halos, every
// tile drifts on its own, and now and then one surges forward (or sinks back) and tilts before
// settling. Nearer tiles grow and overlap their neighbours; receding ones open dark gaps.
const float TILE_A = 0.9;
const vec3 VOID = vec3(0.012, 0.01, 0.022);

vec2 cellCenter(vec2 c) { return uR * vec2(1.7320508 * (c.x + c.y * 0.5), 1.5 * c.y); }
vec2 cellOf(vec2 b) { return hexRound(vec2((0.57735027 * b.x - b.y / 3.0) / uR, (0.66666667 * b.y) / uR)); }

// (smooth base depth, tilt x, tilt y, the tile's own lift along the line of sight); depths in
// units of the eye distance
vec4 tilePose(vec2 c, vec2 cen) {
  vec4 F = fieldAt(cen);
  vec3 sd = vec3(c.x * 1.7 + 17.13, c.y * 2.3 - 11.71, 0.0);
  float mask = F.b;
  float face = mask * (1.0 - smoothstep(0.2, 1.0, F.g * 1.5));
  float sculpt = (F.r - 0.45) * 0.07 * mask + 0.04 * face;
  float wave = 0.035 * cos(TAU * (max(F.a * 2.0 - 1.0, 0.0) * 1.2 - uTravel)) * (1.0 - mask);
  float drift = 0.06 * (vn3(vec3(c * 0.31, uTime * 0.12)) - 0.5) * mix(1.0, 0.4, mask);
  vec2 fuv = (cen - uRes * vec2(0.5, 0.46)) / min(uRes.x, uRes.y);
  float br = 0.5 - 0.5 * uBreathK * cos(TAU * (uBreath - length(fuv) * 0.45));

  float rate = 0.04 + 0.06 * h13(sd + vec3(0.0, 0.0, 51.0));
  float surge = pow(max(0.0, sin(TAU * (uSurgeT * rate + h13(sd + vec3(0.0, 0.0, 53.0))))), 10.0);
  surge *= mix(1.0, 0.45, face);
  float sdir = h13(sd + vec3(0.0, 0.0, 57.0)) < 0.72 ? 1.0 : -0.5;

  surge *= step(0.001, uSurge);
  float z = (sculpt + wave + drift + 0.03 * (br - 0.5) * (0.4 + mask)) * uDepth;
  // Each tile also bobs toward the viewer and back on its own slow clock
  float bob = 0.075 * uBob * mix(1.0, 0.45, face)
              * sin(TAU * (uTime * (0.03 + 0.05 * h13(sd + vec3(0.0, 0.0, 71.0))) + h13(sd + vec3(0.0, 0.0, 73.0))));
  float lift = clamp(bob + 0.28 * uSurge * surge * sdir, -0.2, 0.4);
  vec2 tl = (vec2(vn3(vec3(c * 0.23, uTime * 0.09 + 1.7)), vn3(vec3(c * 0.23 + 9.1, uTime * 0.09 + 4.2))) - 0.5)
            * mix(0.55, 0.3, mask);
  tl += surge * 1.2 * (vec2(h13(sd + vec3(0.0, 0.0, 59.0)), h13(sd + vec3(0.0, 0.0, 61.0))) - 0.5);
  return vec4(z, tl * uTileTilt, lift);
}

const vec2 NB[7] = vec2[7](vec2(0.0), vec2(1.0, 0.0), vec2(-1.0, 0.0), vec2(0.0, 1.0), vec2(0.0, -1.0), vec2(1.0, -1.0), vec2(-1.0, 1.0));

// World centre of a tile. Its base hovers above its own cell; its own lift (bobbing, surges)
// carries it straight along the line of sight, so it stays put on screen, grows and shrinks by
// true perspective and really is nearer than everything it covers (and shadows).
vec3 tileCenter(vec2 ce, vec4 pose, vec3 eye) {
  float z0 = pose.x * eye.z;
  float z1 = z0 + pose.w * eye.z;
  vec2 onScreen = eye.xy + (ce - eye.xy) * eye.z / (eye.z - z0);
  return vec3(eye.xy + (onScreen - eye.xy) * (eye.z - z1) / eye.z, z1);
}

// Ray o + s*d against the plate of tile c: returns s; q is the hit in tile units (inside the
// hexagon when sdHexP(q, TILE_A) <= 0).
float rayTile(vec2 c, vec3 o, vec3 d, vec3 eye, out vec2 q, out vec3 n, out vec4 pose) {
  vec2 ce = cellCenter(c);
  pose = tilePose(c, ce);
  n = normalize(vec3(pose.yz, 1.0));
  vec3 C = tileCenter(ce, pose, eye);
  float s = dot(C - o, n) / dot(d, n);
  vec3 r = o + s * d - C;
  vec3 ux = normalize(cross(vec3(0.0, 1.0, 0.0), n));
  q = vec2(dot(r, ux), dot(r, cross(n, ux))) / uR;
  return s;
}

void main() {
  vec2 px = vec2(gl_FragCoord.x, uRes.y - gl_FragCoord.y);
  float minD = min(uRes.x, uRes.y);
  float Fd = uEye * minD;
  vec3 eye = vec3(uRes * 0.5 + uTilt * minD * 0.5, Fd);
  vec3 D = vec3(px, 0.0) - eye;

  // Find the tile under this pixel: refine which cell the line of sight meets at that cell's own
  // depth (a few fixed-point steps), then intersect it and its six neighbours exactly (tilted
  // plane, hexagon footprint); the nearest hit wins. Cost is independent of perspective.
  vec2 c0 = cellOf(px);
  for (int k = 0; k < 4; k++) {
    float z = tilePose(c0, cellCenter(c0)).x * Fd;
    c0 = cellOf(eye.xy + D.xy * (eye.z - z) / eye.z);
  }
  float bestZ = -1e9;
  bool behind = false;
  vec2 cid = c0;
  vec2 p = vec2(9.0);
  vec3 tN = vec3(0.0, 0.0, 1.0);
  vec3 tX = vec3(px, 0.0);
  vec3 tU = vec3(1.0, 0.0, 0.0), tV = vec3(0.0, 1.0, 0.0);
  float surge = 0.0, baseZ = 0.0;
  for (int i = 0; i < 7; i++) {
    vec2 c = c0 + NB[i];
    vec2 q;
    vec3 n;
    vec4 pose;
    float t = rayTile(c, eye, D, eye, q, n, pose);
    if (sdHexP(q, TILE_A) > 0.0) continue;
    vec3 X = eye + t * D;
    if (bestZ > -1e8) behind = true;
    if (X.z > bestZ) {
      bestZ = X.z;
      cid = c;
      p = q;
      tN = n;
      tX = X;
      baseZ = pose.x * Fd;
      surge = clamp(pose.w / 0.25, 0.0, 1.0);
    }
  }
  bool hit = bestZ > -1e8;
  tU = normalize(cross(vec3(0.0, 1.0, 0.0), tN));
  tV = cross(tN, tU);
  // A small lamp level with the viewer, off to one side (it circles slowly). Seen from here, a
  // tile's shadow is exactly as big as the tile looks, so it grows and shrinks with it, and it
  // slides out from under the tile the higher the tile rises.
  vec3 lamp = vec3(uRes * 0.5 + uLight * 0.42 * minD, Fd);
  vec3 L3 = normalize(lamp - tX);

  // Cast shadows: walk from the visible point toward the lamp up to the highest the tiles
  // reach; at each step test the tile seen there (tiles rise along the line of sight, so look
  // it up by screen position at the local base depth). The penumbra widens with the distance
  // to the occluder.
  float occl = 0.0;
  if (hit) {
    vec3 toLamp = lamp - tX;
    float zTop = (0.1 * uDepth + min(0.075 * uBob + 0.28 * uSurge, 0.4)) * Fd;
    float uTop = clamp((zTop - tX.z) / (Fd - tX.z), 0.0, 0.9);
    vec2 prev = cid;
    for (int i = 1; i <= 10; i++) {
      vec3 P = tX + toLamp * (uTop * pow(float(i) / 10.0, 1.3));
      vec2 sp = eye.xy + (P.xy - eye.xy) * Fd / (Fd - P.z);
      vec2 c = cellOf(eye.xy + (sp - eye.xy) * (Fd - baseZ) / Fd);
      if (c == cid || c == prev) continue;
      prev = c;
      vec2 q;
      vec3 n;
      vec4 pose;
      float s = rayTile(c, tX, toLamp, eye, q, n, pose);
      if (s < 1e-3 || s > 1.0) continue;
      float pen = 0.04 + 0.02 * Fd * s / uR;
      occl = max(occl, 1.0 - smoothstep(-pen, pen, sdHexP(q, TILE_A)));
      if (occl > 0.98) break;
    }
  }
  vec2 cen = cellCenter(cid);
  float scale = hit ? Fd / (Fd - bestZ) : 1.0;

  // Face field averaged over the whole hexagon, so features thinner than a cell still register
  vec4 F = fieldAt(cen) * 0.28;
  for (int k = 0; k < 6; k++) F += fieldAt(cen + HEX6[k] * (0.55 * uR)) * 0.12;
  float relief = F.r;
  float fd = F.g * 1.5;

  // Features stand out against their surroundings: eye sockets, nostrils and mouth sit below the
  // local mean, cheekbones, brow and nose ridge above it.
  float around = 0.0;
  for (int k = 0; k < 6; k++) around += fieldAt(cen + HEX6[k] * (2.6 * uR)).r;
  // Only inside the face oval: the skin-to-hair step at the contour is not a feature
  float feat = (relief - around / 6.0) * (1.0 - smoothstep(0.7, 0.92, fd));
  float mask = clamp(F.b, 0.0, 1.0);
  float dSil = F.a * 2.0 - 1.0;

  float minDim = min(uRes.x, uRes.y);
  vec2 fluidUV = (cen - uRes * vec2(0.5, 0.46)) / minDim;
  vec3 flow = liquidFlow(fluidUV, uTime, relief, mask);

  // Per-cell jitter is quieter on the figure, so neighbouring cells share a colour family and
  // the face's shading, not noise, decides the colour.
  float jit = mix(1.0, 0.3, mask);
  vec3 csd = vec3(cid.x * 1.7 + 17.13, cid.y * 2.3 - 11.71, 0.0);
  float cellPhaseOff = (h13(csd + vec3(0.0, 0.0, 19.1)) - 0.5) * 0.24;
  float cellHueOff = ((h13(csd + vec3(0.0, 0.0, 23.7)) - 0.5) * 0.18
                   + (h13(csd + vec3(0.0, 0.0, 29.3)) < 0.16 ? 0.34 : 0.0) * (1.0 - mask)) * jit;
  float cellSatOff = (h13(csd + vec3(0.0, 0.0, 31.9)) - 0.5) * 0.36 * jit;
  float cellValOff = (h13(csd + vec3(0.0, 0.0, 37.3)) - 0.5) * 0.14;

  // Colour bands follow the face: they are contours of its shading and of each feature against its
  // surroundings, plus halos echoing the head outline in the background. uTravel sweeps the bands
  // across those shapes at a randomly varying speed, outlining the features as they pass; a weak
  // evolving random field adds variety in which colours appear.
  vec2 du = fluidUV + flow.xy * 0.5;
  float drift = 0.35 * (vn3(vec3(du * 2.2, uTime * 0.045)) + 0.6 * vn3(vec3(du * 4.8 + 7.3, uTime * 0.075 + 3.1)));
  float cellDrift = (vn3(vec3(csd.xy * 0.37, uTime * 0.13)) - 0.5) * 0.12 * jit;
  float rings = mask * (relief * 0.25 + feat * 1.1) + (1.0 - mask) * max(dSil, 0.0) * 1.2;
  float basePhase = relief * 0.55 + rings + flow.x * mix(0.55, 0.1, mask) + drift - uTravel;
  float phase = basePhase + cellPhaseOff * jit + cellDrift;
  // Palette mood: the bands swing back and forth across an arc of uPalW around the drifting
  // uHue, so at any moment a few related hues dominate; the arc narrows to near two-tone and
  // opens to the full wheel over time.
  float h = fract(uHue + uPalW * (0.5 - 0.5 * cos(TAU * phase) - 0.5) + cellHueOff);

  // A soft bright crest rides each ring outward; it fades with distance and skips the darkest
  // cells, so eyes, nostrils and mouth keep their depth.
  float core = smoothstep(0.1, 0.32, relief);
  float near = mix(exp(-max(dSil, 0.0) * 2.2) * 0.7, exp(-fd * 1.3), mask);
  float crest = pow(0.5 + 0.5 * cos(TAU * basePhase), 3.0);
  float glint = crest * near * mix(1.0, core, mask);

  // Hair and shoulders breathe in value; the face keeps its contrast
  float mid = smoothstep(0.08, 0.2, relief) * (1.0 - smoothstep(0.45, 0.65, relief)) * smoothstep(0.85, 1.05, fd);
  float lifeV = mid * 0.22 * (0.5 + 0.5 * sin(TAU * (phase * 0.5 + 0.13)));

  // Brightness targets perceived luminance, not HSV value: a saturated blue and a yellow at the
  // same value read very differently, which would scramble the face as the hue cycles.
  // Features carve deep, saturated shadows (eyes, nostrils, mouth) and lift their ridges; only
  // they may go darker than the rest of the picture.
  float shadow = smoothstep(0.015, 0.11, -feat);
  float faceL = 0.26 + relief * 0.56 + feat * mix(0.4, 0.7, shadow) + glint * 0.11 + lifeV * 0.3 + cellValOff * 0.3;
  float bgL = 0.42 + 0.16 * (0.5 + 0.5 * sin((flow.y + cellPhaseOff - uTime * 0.2) * TAU)) + glint * 0.05 + cellValOff * 0.4;
  // Breath: each inhale swells outward from the face, lifting light and colour as it passes
  float breath = 0.5 - 0.5 * uBreathK * cos(TAU * (uBreath - length(fluidUV) * 0.45));
  float L = clamp(mix(bgL, faceL, mask) + (breath - 0.5) * 0.1, mix(0.24, 0.18, shadow * mask), 0.84);

  float faceS = clamp(1.0 - 0.5 * smoothstep(0.5, 1.0, relief) + cellSatOff - glint * 0.12 + 0.2 * shadow, 0.35, 1.0);
  float bgS = clamp(0.68 + 0.24 * (0.5 + 0.5 * cos((phase + flow.z) * TAU)) + cellSatOff, 0.4, 1.0);
  float s = clamp(mix(bgS, faceS, mask) + 0.12 * breath - 0.06, 0.0, 1.0);

  float hueL = dot(hsv2rgb(vec3(h, 1.0, 1.0)), vec3(0.299, 0.587, 0.114));
  s = min(s, (1.0 - L) / (1.0 - hueL));
  float v = min(1.0, L / (1.0 - s * (1.0 - hueL)));

  AA = 1.3 / (uR * scale);
  vec3 col;
  if (!hit) {
    col = VOID;
  } else {
    float faceIn = mask * (1.0 - smoothstep(0.85, 1.05, fd));
    col = tile(p, cid, vec3(h, s, v), flow, breath, 1.0 - 0.3 * faceIn * (1.0 - shadow));
    col *= 0.94 + 0.16 * (1.0 - smoothstep(0.0, 0.95, length(p)));
    vec3 edge = hsv2rgb(vec3(h, min(1.0, s * 1.05), v * 0.86));
    col = mix(col, edge, smoothstep(-0.03 - AA, -0.03 + AA, sdHexP(p, TILE_A)));

    // Lambert on the tilted plate (relative to a tile facing the viewer), a gentle enamel dome
    // on top, cast shadow from whatever hovers between it and the light; nearer tiles read
    // brighter
    float face = clamp(mix(1.0, dot(tN, L3) / L3.z, 0.9 * uGloss), 0.55, 1.3);
    vec3 N = normalize(tN + (tU * p.x + tV * p.y) * 0.45);
    float lit = face * max(0.9, 1.0 + 0.55 * uGloss * (dot(N, L3) - dot(tN, L3)));
    lit *= 1.0 - 0.6 * occl;
    vec3 V = normalize(eye - tX);
    float spec = pow(max(dot(N, normalize(L3 + V)), 0.0), 60.0) * (0.3 + 0.4 * surge) * uGloss * (1.0 - occl);
    col = col * lit + spec * mix(vec3(1.0), col, 0.25);
    col *= mix(0.72, 1.15, smoothstep(-0.1, 0.22, bestZ / Fd));
    // Silhouettes against the void are smoothed; edges over another tile stay crisp, so
    // touching tiles get no outline
    if (!behind) col = mix(VOID, col, fillMask(sdHexP(p, TILE_A)));
  }

  vec2 sv = gl_FragCoord.xy / uRes - 0.5;
  col *= 1.0 - 0.35 * dot(sv, sv);
  outColor = vec4(col, 1.0);
}`;

// Time-lapse morph: a triangle mesh over the face landmarks. Each vertex sits at the in-between
// landmark position and carries where that landmark was in each of the four snapshots, so every
// snapshot is warped onto the shared in-between shape before they are blended.
const WARP_VERT = `#version 300 es
layout(location = 0) in vec2 aPos;
layout(location = 1) in vec4 aUv01;
layout(location = 2) in vec4 aUv23;
out vec4 vUv01;
out vec4 vUv23;
void main() {
  vUv01 = aUv01;
  vUv23 = aUv23;
  gl_Position = vec4(aPos * 2.0 - 1.0, 0.0, 1.0);
}`;

const WARP_FRAG = `#version 300 es
precision highp float;
uniform sampler2D uS0;
uniform sampler2D uS1;
uniform sampler2D uS2;
uniform sampler2D uS3;
uniform vec4 uW;
in vec4 vUv01;
in vec4 vUv23;
out vec4 outColor;
void main() {
  vec3 c = texture(uS0, vUv01.xy).rgb * uW.x + texture(uS1, vUv01.zw).rgb * uW.y
         + texture(uS2, vUv23.xy).rgb * uW.z + texture(uS3, vUv23.zw).rgb * uW.w;
  outColor = vec4(clamp(c, 0.0, 1.0), 1.0);
}`;

const HOLD = 28;
const MORPH = 8;
const SRC_W = 768;

const canvas = document.getElementById("view");
const errorBox = document.getElementById("error");
const gl = canvas.getContext("webgl2", { antialias: false, alpha: false, powerPreference: "high-performance" });

// Camera downsample: enough rows for the finest hexagon setting, shaped like the screen.
const CAM_H = 192;
const camRaw = document.createElement("canvas");
camRaw.width = 256;
camRaw.height = CAM_H;
const rctx = camRaw.getContext("2d");
const camBtn = document.getElementById("cam");

const MP_BASE = "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14";
const MP_LANDMARKER = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task";
const MP_MODEL = "https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/1/blaze_face_short_range.tflite";

let cam = null;
let camOn = false;
let camK = 0;
let camCrop = null;
let camReady = false;
let camFace = null;
let camFaceK = 0;
const LOST_MS = 800;

// Time-lapse: a camera snapshot every tlGap seconds; the picture glides through them on a
// Catmull-Rom curve (between the 2nd and 3rd of the last four), so the morph never pauses.
// Snapshots live only in GPU textures and are dropped when time-lapse stops.
const TL_GAPS = [0.5, 1, 1.5, 2, 3, 5];
const TL_GAP_KEY = "go-visual-tl-gap";
const CAM_UNITS = [1, 3, 4, 5];
const tlBtn = document.getElementById("tl");
const tlGapSel = document.getElementById("tl-gap");
let tlGap = 1;
try { tlGap = TL_GAPS.includes(Number(localStorage.getItem(TL_GAP_KEY))) ? Number(localStorage.getItem(TL_GAP_KEY)) : 1; } catch {}
let tlOn = false;
let tlSnaps = null;
let tlT = 0;
let tlFace = null;
const camW = new Float32Array([1, 0, 0, 0]);
let warpProg, wloc, warpVao, warpVbo, warpIbo, morphTex, morphFbo;
let warpTris = null;
let warpCount = 0;
let warpData = new Float32Array(0);

// Face oval landmarks (MediaPipe face mesh); pushed outward they pin the hair and head.
const OVAL = [10, 297, 284, 389, 454, 361, 397, 379, 400, 152, 176, 150, 172, 132, 234, 162, 54, 67];
const N_MESH = 468;
const BORDER = [];
for (let i = 0; i < 4; i++) {
  const t = i / 4;
  BORDER.push([t, 0], [1, t], [1 - t, 1], [0, 1 - t]);
}
let landmarker = null;
let detector = null;
let detectorLoading = false;
let faceBox = null;
let faceMarks = null;
let lastDetect = -1;

let W = 0, H = 0, FW = 0, FH = 0, dpr = 1;

// Resolution: how many hexagons make up the picture (1 = default, higher = smaller cells).
const RES_KEY = "go-visual-res";
const RES_MIN = 0.6;
const RES_MAX = 3.5;
let resTarget = 1;
try { resTarget = clamp(Number(localStorage.getItem(RES_KEY)) || 1, RES_MIN, RES_MAX); } catch {}
let res = resTarget;
let faceA = makeFace();
let faceB = makeFace();
let faceT = 0;
let morphing = false;
let mixK = 0;
let simT = 0;
let hue = rand();
const hueSeed = rand() * 100;
const TAU = Math.PI * 2;
let breathPh = 0;
let travel = 0;
let last = performance.now();
let prog, fieldProg, loc, floc, fieldTex, camTex, eigenTex, fbo;
const packed = new Float32Array(FACE_KEYS.length);

// Face model: mean + weighted components. Each weight drifts on two slow incommensurate
// sines, so the face keeps morphing into new people and never repeats.
const EIG_N = EIGEN.size * EIGEN.size;
const EIG_AMP = 0.8;
const eigMean = Float32Array.from(atob(EIGEN.mean), (c) => c.charCodeAt(0) / 255);
const eigComps = (() => {
  const raw = Int8Array.from(atob(EIGEN.comps), (c) => (c.charCodeAt(0) << 24) >> 24);
  const out = new Float32Array(raw.length);
  for (let k = 0; k < EIGEN.k; k++) {
    for (let p = 0; p < EIG_N; p++) out[k * EIG_N + p] = raw[k * EIG_N + p] * EIGEN.scale[k];
  }
  return out;
})();
const eigDrift = Array.from({ length: EIGEN.k }, () => [range(0.04, 0.11), range(0.04, 0.11), range(0, 6.3), range(0, 6.3)]);
const eigSum = new Float32Array(EIG_N);
const eigTmp = new Float32Array(EIG_N);
const eigPix = new Uint8Array(EIG_N);

function fail(msg) {
  errorBox.hidden = false;
  errorBox.textContent = msg;
  throw new Error(msg);
}

function compile(type, src) {
  const s = gl.createShader(type);
  gl.shaderSource(s, src);
  gl.compileShader(s);
  if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) fail(gl.getShaderInfoLog(s) || "shader error");
  return s;
}

function makeProgram(fs, names, vs = VERT) {
  const p = gl.createProgram();
  gl.attachShader(p, compile(gl.VERTEX_SHADER, vs));
  gl.attachShader(p, compile(gl.FRAGMENT_SHADER, fs));
  gl.linkProgram(p);
  if (!gl.getProgramParameter(p, gl.LINK_STATUS)) fail(gl.getProgramInfoLog(p) || "link error");
  const u = {};
  for (const n of names) u[n] = gl.getUniformLocation(p, n);
  return [p, u];
}

function makeTexture() {
  const t = gl.createTexture();
  gl.bindTexture(gl.TEXTURE_2D, t);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
  return t;
}

function initGL() {
  if (!gl) fail("WebGL2 is not available in this browser.");
  [prog, loc] = makeProgram(FRAG, ["uRes", "uTime", "uR", "uHue", "uBreath", "uTravel", "uPalW", "uTilt", "uLight", "uDepth", "uSurge", "uSurgeT", "uTileTilt", "uGloss", "uEye", "uBreathK", "uField", "uBob"]);
  [fieldProg, floc] = makeProgram(FIELD, ["uScreen", "uFieldRes", "uFace", "uP", "uEigen", "uCam", "uCamO", "uCamK"]);
  [warpProg, wloc] = makeProgram(WARP_FRAG, ["uS0", "uS1", "uS2", "uS3", "uW"], WARP_VERT);
  warpVao = gl.createVertexArray();
  gl.bindVertexArray(warpVao);
  warpVbo = gl.createBuffer();
  gl.bindBuffer(gl.ARRAY_BUFFER, warpVbo);
  gl.enableVertexAttribArray(0);
  gl.vertexAttribPointer(0, 2, gl.FLOAT, false, 40, 0);
  gl.enableVertexAttribArray(1);
  gl.vertexAttribPointer(1, 4, gl.FLOAT, false, 40, 8);
  gl.enableVertexAttribArray(2);
  gl.vertexAttribPointer(2, 4, gl.FLOAT, false, 40, 24);
  warpIbo = gl.createBuffer();
  gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, warpIbo);
  gl.bindVertexArray(null);
  morphTex = makeTexture();
  morphFbo = gl.createFramebuffer();
  fieldTex = makeTexture();
  camTex = makeTexture();
  gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, 1, 1, 0, gl.RGBA, gl.UNSIGNED_BYTE, new Uint8Array([128, 128, 128, 255]));
  eigenTex = makeTexture();
  gl.texImage2D(gl.TEXTURE_2D, 0, gl.R8, EIGEN.size, EIGEN.size, 0, gl.RED, gl.UNSIGNED_BYTE, null);
  fbo = gl.createFramebuffer();
  gl.useProgram(prog);
  gl.uniform1i(loc.uField, 0);
  gl.useProgram(fieldProg);
  gl.uniform1i(floc.uCam, 1);
  gl.uniform1i(floc.uEigen, 2);
  gl.useProgram(warpProg);
  CAM_UNITS.forEach((unit, i) => gl.uniform1i(wloc[`uS${i}`], unit));
}

function updateEigen(t) {
  eigSum.set(eigMean);
  for (let k = 0; k < EIGEN.k; k++) {
    const [w1, w2, p1, p2] = eigDrift[k];
    const z = EIG_AMP * (Math.sin(w1 * t + p1) + Math.sin(w2 * t + p2));
    const off = k * EIG_N;
    for (let p = 0; p < EIG_N; p++) eigSum[p] += z * eigComps[off + p];
  }

  // Every blend gets the same contrast, and a local-contrast lift so eyes, nose and mouth
  // survive the per-hexagon averaging.
  const S = EIGEN.size;
  let m = 0, v = 0;
  for (let p = 0; p < EIG_N; p++) m += eigSum[p];
  m /= EIG_N;
  for (let p = 0; p < EIG_N; p++) v += (eigSum[p] - m) ** 2;
  const g = 0.2 / Math.sqrt(v / EIG_N + 1e-6);
  for (let y = 0; y < S; y++) {
    for (let x = 0; x < S; x++) {
      let s = 0;
      for (let k = -3; k <= 3; k++) s += eigSum[y * S + clamp(x + k, 0, S - 1)];
      eigTmp[y * S + x] = s / 7;
    }
  }
  for (let y = 0; y < S; y++) {
    for (let x = 0; x < S; x++) {
      let s = 0;
      for (let k = -3; k <= 3; k++) s += eigTmp[clamp(y + k, 0, S - 1) * S + x];
      const p = y * S + x;
      const sharp = eigSum[p] + 0.3 * (eigSum[p] - s / 7);
      eigPix[p] = clamp(0.5 + (sharp - m) * g, 0, 1) * 255;
    }
  }
  gl.activeTexture(gl.TEXTURE2);
  gl.bindTexture(gl.TEXTURE_2D, eigenTex);
  gl.texSubImage2D(gl.TEXTURE_2D, 0, 0, 0, EIGEN.size, EIGEN.size, gl.RED, gl.UNSIGNED_BYTE, eigPix);
}

// Render scale: big or slow screens drop pixels to hold the frame rate. After backing off from a
// scale it never climbs back to it, so the sharpness doesn't pump.
let renderScale = 1, scaleCeil = 1, frameMs = 16, scaleAt = 0;
function adaptScale(ms, now) {
  if (ms > 250) return;
  frameMs += (ms - frameMs) * 0.08;
  if (now - scaleAt < 1500) return;
  let next = renderScale;
  if (frameMs > 26) {
    scaleCeil = renderScale * 0.97;
    next = renderScale * 0.82;
  } else if (frameMs < 19 && now - scaleAt > 5000) {
    next = Math.min(scaleCeil, renderScale * 1.1);
  }
  next = Math.max(0.75 / Math.min(2.5, devicePixelRatio || 1), next);
  if (Math.abs(next - renderScale) < 0.01) return;
  renderScale = next;
  scaleAt = now;
  frameMs = 16;
  layout();
}

function layout() {
  dpr = Math.min(2.5, devicePixelRatio || 1) * renderScale;
  W = Math.max(1, Math.round(innerWidth * dpr));
  H = Math.max(1, Math.round(innerHeight * dpr));
  canvas.width = W;
  canvas.height = H;
  FW = SRC_W;
  FH = Math.max(120, Math.round((SRC_W * innerHeight) / innerWidth));
  gl.activeTexture(gl.TEXTURE0);
  gl.bindTexture(gl.TEXTURE_2D, fieldTex);
  gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA8, FW, FH, 0, gl.RGBA, gl.UNSIGNED_BYTE, null);
  gl.bindFramebuffer(gl.FRAMEBUFFER, fbo);
  gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, fieldTex, 0);

  camRaw.width = clamp(Math.round((CAM_H * innerWidth) / innerHeight), 96, 420);
  camRaw.height = CAM_H;
  gl.bindTexture(gl.TEXTURE_2D, morphTex);
  gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA8, camRaw.width, camRaw.height, 0, gl.RGBA, gl.UNSIGNED_BYTE, null);
  gl.bindFramebuffer(gl.FRAMEBUFFER, morphFbo);
  gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, morphTex, 0);
  gl.bindFramebuffer(gl.FRAMEBUFFER, null);
}

function headUnit(w, h) {
  return Math.min(w * 0.31, h * 0.3);
}

function cellRadius() {
  return Math.max(3, clamp(headUnit(innerWidth, innerHeight) * 0.074, 9, 24.0) / res) * dpr;
}

function setRes(step) {
  resTarget = clamp(resTarget * Math.pow(1.2, step), RES_MIN, RES_MAX);
  try { localStorage.setItem(RES_KEY, String(resTarget)); } catch {}
}

function advanceFaces(dt) {
  faceT += dt;
  if (!morphing && faceT >= HOLD) morphing = true;
  if (morphing) {
    mixK = smooth((faceT - HOLD) / MORPH);
    if (faceT >= HOLD + MORPH) {
      faceA = faceB;
      faceB = makeFace();
      faceT = 0;
      mixK = 0;
      morphing = false;
    }
  }
}

// Screen transform [cx, cy, unit, angle] of the current head; fills `packed` with its silhouette.
function currentFace() {
  let f = mixFace(faceA, faceB, mixK);
  const U = headUnit(W, H);
  let cx = W * 0.5 + Math.sin(simT * 0.19) * U * 0.012;
  let cy = H * 0.45 + Math.sin(simT * 0.13) * U * 0.008;
  let u = U * (1 + 0.008 * Math.sin(simT * 0.17) + 0.016 * fx.breath * (0.5 - 0.5 * Math.cos(TAU * breathPh)));
  let ang = f.tilt + 0.022 * Math.sin(simT * 0.23);
  const src = tlFace || camFace;
  const k = tlFace ? camK : camK * camFaceK;
  if (src && k > 0.001) {
    f = mixFace(f, src.f, k);
    cx = lerp(cx, src.cx, k);
    cy = lerp(cy, src.cy, k);
    u = lerp(u, src.U, k);
    ang = lerp(ang, src.angle, k);
  }
  packFace(f, packed);
  return [cx, cy, u, ang];
}

function draw() {
  const [cx, cy, u, ang] = currentFace();
  updateEigen(simT);
  if (tlSnaps) renderMorph();

  const cr = cellRadius();
  gl.bindFramebuffer(gl.FRAMEBUFFER, fbo);
  gl.viewport(0, 0, FW, FH);
  gl.useProgram(fieldProg);
  gl.activeTexture(gl.TEXTURE0);
  gl.bindTexture(gl.TEXTURE_2D, null);
  gl.activeTexture(gl.TEXTURE1);
  gl.bindTexture(gl.TEXTURE_2D, tlSnaps ? morphTex : camTex);
  gl.uniform2f(floc.uScreen, W, H);
  gl.uniform2f(floc.uFieldRes, FW, FH);
  gl.uniform4f(floc.uFace, cx, cy, u, ang);
  gl.uniform4fv(floc.uP, packed);
  gl.uniform2f(floc.uCamO, Math.max((1.6 * cr) / W, 1.2 / camRaw.width), Math.max((1.6 * cr) / H, 1.2 / camRaw.height));
  gl.uniform1f(floc.uCamK, tlSnaps ? camK : camReady ? camK * camFaceK : 0);
  gl.drawArrays(gl.TRIANGLES, 0, 3);

  gl.bindFramebuffer(gl.FRAMEBUFFER, null);
  gl.viewport(0, 0, W, H);
  gl.useProgram(prog);
  gl.activeTexture(gl.TEXTURE0);
  gl.bindTexture(gl.TEXTURE_2D, fieldTex);
  gl.uniform2f(loc.uRes, W, H);
  gl.uniform1f(loc.uTime, simT);
  gl.uniform1f(loc.uR, cr);
  gl.uniform1f(loc.uHue, hue);
  gl.uniform1f(loc.uTravel, travel);
  gl.uniform1f(loc.uPalW, paletteWidth(simT));
  gl.uniform2f(loc.uTilt, tilt[0], tilt[1]);
  gl.uniform1f(loc.uDepth, fx.depth);
  gl.uniform1f(loc.uSurge, fx.surge);
  gl.uniform1f(loc.uBob, fx.bob);
  gl.uniform1f(loc.uSurgeT, surgeT);
  gl.uniform1f(loc.uTileTilt, fx.tileTilt);
  gl.uniform1f(loc.uGloss, fx.gloss);
  gl.uniform1f(loc.uEye, 2.2 / fx.perspective);
  gl.uniform1f(loc.uBreathK, fx.breath);
  const la = -2.3 + 0.9 * Math.sin(simT * 0.045 + hueSeed) + 0.35 * Math.sin(simT * 0.13);
  gl.uniform2f(loc.uLight, 0.85 * Math.cos(la), 0.85 * Math.sin(la));
  gl.uniform1f(loc.uBreath, breathPh + 0.08 * Math.sin(TAU * breathPh));
  gl.drawArrays(gl.TRIANGLES, 0, 3);
}

// Share of the colour wheel in play: ~0.28 is a two-tone moment, 1 the full rainbow; it wanders
// on incommensurate periods (about 1-1.5 min), so the moods never come back in the same order.
function paletteWidth(t) {
  return clamp(0.6 + 0.3 * Math.sin(t * 0.071 + hueSeed * 1.9) + 0.16 * Math.sin(t * 0.173 + hueSeed * 0.6), 0.28, 1);
}

// Effect controls: every value is a multiplier on the tuned default (1).
const FX_KEY = "go-visual-fx";
const FX = [
  { key: "depth", label: "Depth", max: 3 },
  { key: "bob", label: "Front/back", max: 3 },
  { key: "surge", label: "Surge", max: 2.5 },
  { key: "surgeRate", label: "Surge rate", min: 0.2, max: 4 },
  { key: "tileTilt", label: "Tile tilt", max: 2.5 },
  { key: "swing", label: "View swing", max: 2.5 },
  { key: "perspective", label: "Perspective", min: 0.3, max: 3 },
  { key: "gloss", label: "Gloss", max: 2.5 },
  { key: "colorSpeed", label: "Color speed", max: 3 },
  { key: "breath", label: "Breathing", max: 2.5 },
];
const fx = Object.fromEntries(FX.map((p) => [p.key, 1]));
try { Object.assign(fx, JSON.parse(localStorage.getItem(FX_KEY)) || {}); } catch {}
let surgeT = 0;

const fxPanel = document.getElementById("fx");
const fxInputs = {};
for (const p of FX) {
  const row = document.createElement("label");
  const name = document.createElement("span");
  const input = document.createElement("input");
  const out = document.createElement("output");
  name.textContent = p.label;
  Object.assign(input, { type: "range", min: p.min ?? 0, max: p.max, step: 0.05, value: fx[p.key] });
  out.textContent = Number(fx[p.key]).toFixed(2);
  input.addEventListener("input", () => {
    fx[p.key] = Number(input.value);
    out.textContent = fx[p.key].toFixed(2);
    try { localStorage.setItem(FX_KEY, JSON.stringify(fx)); } catch {}
  });
  row.append(name, input, out);
  document.getElementById("fx-sliders").append(row);
  fxInputs[p.key] = { input, out };
}
document.getElementById("fx-reset").addEventListener("click", () => {
  for (const p of FX) {
    fx[p.key] = 1;
    fxInputs[p.key].input.value = 1;
    fxInputs[p.key].out.textContent = "1.00";
  }
  try { localStorage.removeItem(FX_KEY); } catch {}
});
const toggleFx = () => (fxPanel.hidden = !fxPanel.hidden);
document.getElementById("fx-btn").addEventListener("click", toggleFx);

// Viewpoint: drifts on its own, and follows the pointer while it moves (looking from the right
// shows the tiles' right-hand sides).
const tilt = [0, 0];
let pointer = null;
let pointerT = -Infinity;
addEventListener("pointermove", (e) => {
  pointer = [e.clientX / innerWidth - 0.5, e.clientY / innerHeight - 0.5];
  pointerT = performance.now();
});

function updateTilt(dt, now) {
  const follow = pointer && now - pointerT < 6000;
  const tx = follow ? -pointer[0] * 1.3 : 0.48 * Math.sin(simT * 0.09 + hueSeed) + 0.14 * Math.sin(simT * 0.23 + 1.3);
  const ty = follow ? -pointer[1] * 1.3 : 0.36 * Math.sin(simT * 0.071 + hueSeed * 2.1) + 0.1 * Math.sin(simT * 0.17 + 0.4) + 0.12;
  const e = 1 - Math.exp(-dt * (follow ? 5 : 1.5));
  tilt[0] += (tx * fx.swing - tilt[0]) * e;
  tilt[1] += (ty * fx.swing - tilt[1]) * e;
}

function frame(now) {
  const dt = Math.min(0.05, (now - last) / 1000);
  adaptScale(now - last, now);
  last = now;
  simT += dt;

  advanceFaces(dt);
  const hueSpeed = 0.008 + 0.012 * Math.sin(simT * 0.041 + hueSeed) + 0.008 * Math.sin(simT * 0.067 + hueSeed * 2.3);
  hue = (((hue + dt * hueSpeed * fx.colorSpeed) % 1) + 1) % 1;
  const travelSpeed = 0.25 + 0.09 * Math.sin(simT * 0.053 + hueSeed * 0.7) + 0.05 * Math.sin(simT * 0.121 + hueSeed * 1.3);
  travel = (travel + dt * travelSpeed * fx.colorSpeed) % 2;
  surgeT += dt * fx.surgeRate;
  updateTilt(dt, now);
  const breathPeriod = 6.4 + 1.6 * Math.sin(simT * 0.047 + hueSeed) + 0.8 * Math.sin(simT * 0.13 + hueSeed * 1.7);
  breathPh = (breathPh + dt / breathPeriod) % 1;
  res += (resTarget - res) * (1 - Math.exp(-dt * 6));
  camK += ((camOn ? 1 : 0) - camK) * (1 - Math.exp(-dt * 2));
  if (!camOn && cam && camK < 0.01) releaseCamera();
  if (cam) {
    trackFace(now);
    updateCamera(dt);
  }
  const tracked = cam && camFace && faceBox && now - faceBox.seen < LOST_MS;
  camFaceK += ((tracked ? 1 : 0) - camFaceK) * (1 - Math.exp(-dt * (tracked ? 7 : 2.5)));
  if (tlOn && camReady) stepTimelapse(dt, tracked);
  draw();
  requestAnimationFrame(frame);
}

function snapshotFace(tracked) {
  if (tracked) return { cx: camFace.cx, cy: camFace.cy, U: camFace.U, angle: camFace.angle, f: { ...camFace.f } };
  return tlSnaps ? tlSnaps[3].xf : null;
}

// Landmarks in the snapshot's own texture space (mirrored like camRaw, v down): the face mesh,
// a ring pushed out from the face oval for hair and head, and the fixed image border.
function snapshotPoints(tracked) {
  if (!tracked || !faceMarks || faceMarks.length < N_MESH || !camCrop) return null;
  const c = camCrop;
  const pts = new Float32Array((N_MESH + OVAL.length + BORDER.length) * 2);
  let mx = 0, my = 0;
  for (let i = 0; i < N_MESH; i++) {
    const u = 1 - (faceMarks[i][0] - c.sx) / c.cw;
    const v = (faceMarks[i][1] - c.sy) / c.ch;
    pts[i * 2] = u;
    pts[i * 2 + 1] = v;
    mx += u / N_MESH;
    my += v / N_MESH;
  }
  OVAL.forEach((idx, j) => {
    const o = (N_MESH + j) * 2;
    pts[o] = clamp(mx + (pts[idx * 2] - mx) * 1.6, 0.02, 0.98);
    pts[o + 1] = clamp(my + (pts[idx * 2 + 1] - my) * 1.6, 0.02, 0.98);
  });
  BORDER.forEach(([u, v], j) => {
    const o = (N_MESH + OVAL.length + j) * 2;
    pts[o] = u;
    pts[o + 1] = v;
  });
  return pts;
}

// Bowyer-Watson Delaunay triangulation of a flat [x0, y0, x1, y1, ...] point list.
function delaunay(pts) {
  const n = pts.length / 2;
  const P = Array.from(pts);
  P.push(-10, -10, 30, -10, -10, 30);
  const circ = (a, b, c) => {
    const ax = P[a * 2], ay = P[a * 2 + 1], bx = P[b * 2], by = P[b * 2 + 1], cx = P[c * 2], cy = P[c * 2 + 1];
    const d = 2 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by));
    if (Math.abs(d) < 1e-12) return { a, b, c, x: 0, y: 0, r2: Infinity };
    const a2 = ax * ax + ay * ay, b2 = bx * bx + by * by, c2 = cx * cx + cy * cy;
    const x = (a2 * (by - cy) + b2 * (cy - ay) + c2 * (ay - by)) / d;
    const y = (a2 * (cx - bx) + b2 * (ax - cx) + c2 * (bx - ax)) / d;
    return { a, b, c, x, y, r2: (ax - x) ** 2 + (ay - y) ** 2 };
  };
  let tris = [circ(n, n + 1, n + 2)];
  for (let i = 0; i < n; i++) {
    const px = P[i * 2], py = P[i * 2 + 1];
    const edges = new Map();
    const keep = [];
    for (const t of tris) {
      if ((px - t.x) ** 2 + (py - t.y) ** 2 < t.r2) {
        for (const [u, v] of [[t.a, t.b], [t.b, t.c], [t.c, t.a]]) {
          const key = u < v ? `${u},${v}` : `${v},${u}`;
          edges.set(key, edges.has(key) ? null : [u, v]);
        }
      } else keep.push(t);
    }
    for (const e of edges.values()) if (e) keep.push(circ(e[0], e[1], i));
    tris = keep;
  }
  const out = [];
  for (const t of tris) if (t.a < n && t.b < n && t.c < n) out.push(t.a, t.b, t.c);
  return new Uint16Array(out);
}

function captureSnapshot(tracked) {
  const xf = snapshotFace(tracked);
  const pts = snapshotPoints(tracked);
  if (pts && !warpTris) {
    warpTris = delaunay(pts);
    gl.bindVertexArray(warpVao);
    gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, warpTris, gl.STATIC_DRAW);
    gl.bindVertexArray(null);
    warpCount = pts.length / 2;
    warpData = new Float32Array(warpCount * 10);
  }
  gl.activeTexture(gl.TEXTURE0 + CAM_UNITS[0]);
  if (!tlSnaps) {
    tlSnaps = CAM_UNITS.map(() => {
      const tex = makeTexture();
      gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, camRaw);
      return { tex, xf, pts };
    });
    return;
  }
  const s = tlSnaps.shift();
  gl.bindTexture(gl.TEXTURE_2D, s.tex);
  gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, camRaw);
  s.xf = xf;
  s.pts = pts;
  tlSnaps.push(s);
}

// Snapshots without landmarks borrow the nearest neighbour's, so they cross-fade without warping.
function snapPoints(i) {
  for (let d = 0; d < 4; d++) {
    for (const j of [i - d, i + d]) if (j >= 0 && j < 4 && tlSnaps[j].pts) return tlSnaps[j].pts;
  }
  return null;
}

const QUAD = new Float32Array([0, 0, 1, 0, 1, 1, 0, 1]);
const QUAD_TRIS = new Uint16Array([0, 1, 2, 0, 2, 3]);

function renderMorph() {
  const meshed = warpTris && tlSnaps.some((s) => s.pts);
  const src = [0, 1, 2, 3].map((i) => (meshed ? snapPoints(i) : QUAD));
  const count = meshed ? warpCount : 4;
  const data = meshed ? warpData : new Float32Array(40);
  for (let v = 0; v < count; v++) {
    const o = v * 10;
    let x = 0, y = 0;
    for (let i = 0; i < 4; i++) {
      x += camW[i] * src[i][v * 2];
      y += camW[i] * src[i][v * 2 + 1];
      data[o + 2 + i * 2] = src[i][v * 2];
      data[o + 3 + i * 2] = src[i][v * 2 + 1];
    }
    data[o] = x;
    data[o + 1] = y;
  }

  gl.bindFramebuffer(gl.FRAMEBUFFER, morphFbo);
  gl.viewport(0, 0, camRaw.width, camRaw.height);
  gl.clearColor(0.5, 0.5, 0.5, 1);
  gl.clear(gl.COLOR_BUFFER_BIT);
  gl.useProgram(warpProg);
  CAM_UNITS.forEach((unit, i) => {
    gl.activeTexture(gl.TEXTURE0 + unit);
    gl.bindTexture(gl.TEXTURE_2D, tlSnaps[i].tex);
  });
  gl.uniform4fv(wloc.uW, camW);
  gl.bindVertexArray(warpVao);
  gl.bindBuffer(gl.ARRAY_BUFFER, warpVbo);
  gl.bufferData(gl.ARRAY_BUFFER, data, gl.DYNAMIC_DRAW);
  if (!meshed) gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, QUAD_TRIS, gl.DYNAMIC_DRAW);
  gl.drawElements(gl.TRIANGLES, meshed ? warpTris.length : 6, gl.UNSIGNED_SHORT, 0);
  if (!meshed && warpTris) gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, warpTris, gl.STATIC_DRAW);
  gl.bindVertexArray(null);
}

function stepTimelapse(dt, tracked) {
  tlT += dt;
  if (!tlSnaps) {
    captureSnapshot(tracked);
    tlT = 0;
  } else if (tlT >= tlGap) {
    tlT -= tlGap;
    captureSnapshot(tracked);
  }
  const t = Math.min(1, tlT / tlGap);
  const t2 = t * t, t3 = t2 * t;
  camW[0] = (-t3 + 2 * t2 - t) / 2;
  camW[1] = (3 * t3 - 5 * t2 + 2) / 2;
  camW[2] = (-3 * t3 + 4 * t2 + t) / 2;
  camW[3] = (t3 - t2) / 2;

  const xfs = tlSnaps.map((s) => s.xf);
  if (xfs.some((x) => !x)) {
    tlFace = null;
    return;
  }
  const blend = (get) => xfs.reduce((sum, x, i) => sum + camW[i] * get(x), 0);
  const f = {};
  for (const key in xfs[0].f) f[key] = blend((x) => x.f[key]);
  tlFace = { cx: blend((x) => x.cx), cy: blend((x) => x.cy), U: blend((x) => x.U), angle: blend((x) => x.angle), f };
}

function stopTimelapse() {
  tlOn = false;
  if (tlSnaps) for (const s of tlSnaps) gl.deleteTexture(s.tex);
  tlSnaps = null;
  tlFace = null;
  tlT = 0;
  camW.set([1, 0, 0, 0]);
  tlBtn.textContent = "Time-lapse";
  tlGapSel.hidden = true;
}

function toggleTimelapse() {
  if (tlOn) {
    stopTimelapse();
    return;
  }
  tlOn = true;
  tlBtn.textContent = "Live";
  tlGapSel.hidden = false;
  if (!camOn) startCamera();
}

// Keeps the current transition's progress when the gap changes, so the morph doesn't jump.
function setGap(gap) {
  tlT *= gap / tlGap;
  tlGap = gap;
  tlGapSel.value = String(gap);
  try { localStorage.setItem(TL_GAP_KEY, String(gap)); } catch {}
}

function stepGap(dir) {
  const i = clamp(TL_GAPS.indexOf(tlGap) + dir, 0, TL_GAPS.length - 1);
  setGap(TL_GAPS[i]);
}

async function startCamera() {
  if (cam) {
    camOn = true;
    camBtn.textContent = "Stop camera";
    return;
  }
  try {
    const stream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: "user", width: { ideal: 640 }, height: { ideal: 480 } },
      audio: false,
    });
    const video = document.createElement("video");
    video.muted = true;
    video.playsInline = true;
    video.srcObject = stream;
    await video.play();
    cam = { video, stream };
    camOn = true;
    camBtn.textContent = "Stop camera";
    loadDetector();
  } catch (err) {
    stopTimelapse();
    camBtn.textContent = "Camera blocked";
    setTimeout(() => { camBtn.textContent = "Camera"; }, 2500);
  }
}

function stopCamera() {
  camOn = false;
  camBtn.textContent = "Camera";
  stopTimelapse();
}

function releaseCamera() {
  stopTimelapse();
  for (const track of cam.stream.getTracks()) track.stop();
  cam = null;
  camCrop = null;
  camReady = false;
  camFace = null;
  camFaceK = 0;
  faceBox = null;
  faceMarks = null;
}

async function loadDetector() {
  if (landmarker || detector || detectorLoading) return;
  detectorLoading = true;
  try {
    const vision = await import(`${MP_BASE}/vision_bundle.mjs`);
    const files = await vision.FilesetResolver.forVisionTasks(`${MP_BASE}/wasm`);
    try {
      landmarker = await vision.FaceLandmarker.createFromOptions(files, {
        baseOptions: { modelAssetPath: MP_LANDMARKER, delegate: "GPU" },
        runningMode: "VIDEO",
        numFaces: 1,
        minFaceDetectionConfidence: 0.45,
        minTrackingConfidence: 0.45,
      });
    } catch (_) {
      detector = await vision.FaceDetector.createFromOptions(files, {
        baseOptions: { modelAssetPath: MP_MODEL, delegate: "GPU" },
        runningMode: "VIDEO",
        minDetectionConfidence: 0.5,
      });
    }
  } catch (err) {
    console.warn("Face detector unavailable; camera runs without tracking.", err);
  }
  detectorLoading = false;
}

// Smoothing weight per detection: steady when still (no jitter), near-instant once the head moves.
function followRate(dx, dy, size, still) {
  const moved = Math.hypot(dx, dy) / Math.max(1, size);
  return clamp(still + (moved - 0.02) * 8, still, 1);
}

function updateBox(target, now) {
  if (!faceBox) {
    faceBox = { ...target, seen: now };
    return;
  }
  const e = followRate(target.cx - faceBox.cx, target.cy - faceBox.cy, faceBox.h, 0.32);
  faceBox.cx += (target.cx - faceBox.cx) * e;
  faceBox.cy += (target.cy - faceBox.cy) * e;
  faceBox.h += (target.h - faceBox.h) * e;
  faceBox.seen = now;
}

function trackFace(now) {
  if ((!landmarker && !detector) || cam.video.readyState < 2 || now - lastDetect < 33) return;
  lastDetect = now;
  const vw = cam.video.videoWidth;
  const vh = cam.video.videoHeight;

  if (landmarker) {
    let res;
    try {
      res = landmarker.detectForVideo(cam.video, now);
    } catch (_) {
      return;
    }
    const lm = res && res.faceLandmarks && res.faceLandmarks[0];
    if (!lm || !lm.length) return;
    let x0 = 1, y0 = 1, x1 = 0, y1 = 0;
    for (let i = 0; i < lm.length; i++) {
      const p = lm[i];
      if (p.x < x0) x0 = p.x;
      if (p.y < y0) y0 = p.y;
      if (p.x > x1) x1 = p.x;
      if (p.y > y1) y1 = p.y;
    }
    if (!faceMarks || faceMarks.length !== lm.length) {
      faceMarks = lm.map((p) => [p.x * vw, p.y * vh]);
    } else if (faceBox) {
      const e = followRate((x0 + x1) * 0.5 * vw - faceBox.cx, (y0 + y1) * 0.5 * vh - faceBox.cy, faceBox.h, 0.4);
      for (let i = 0; i < lm.length; i++) {
        faceMarks[i][0] += (lm[i].x * vw - faceMarks[i][0]) * e;
        faceMarks[i][1] += (lm[i].y * vh - faceMarks[i][1]) * e;
      }
    }
    updateBox({ cx: ((x0 + x1) * 0.5) * vw, cy: ((y0 + y1) * 0.5) * vh, h: (y1 - y0) * vh }, now);
    return;
  }

  let result;
  try {
    result = detector.detectForVideo(cam.video, now);
  } catch (_) {
    return;
  }
  const det = result.detections && result.detections[0];
  if (!det) return;
  const b = det.boundingBox;
  const kp = det.keypoints || [];
  updateBox({ cx: b.originX + b.width / 2, cy: b.originY + b.height / 2, h: b.height }, now);
  faceBox.keys = kp.length >= 4 ? kp.slice(0, 4).map((p) => [p.x * vw, p.y * vh]) : null;
}

// Face frame from two eye centres: unit = eye gap / 0.66, origin = face-oval centre.
function faceFrame(a, b) {
  if (a[0] > b[0]) [a, b] = [b, a];
  const angle = Math.atan2(b[1] - a[1], b[0] - a[0]);
  const U = Math.max(1, Math.hypot(b[0] - a[0], b[1] - a[1]) / 0.66);
  const c = Math.cos(angle);
  const s = Math.sin(angle);
  const ly = -0.1 * U;
  const cx = (a[0] + b[0]) / 2 + ly * s;
  const cy = (a[1] + b[1]) / 2 - ly * c;
  const local = (p) => {
    const dx = p[0] - cx;
    const dy = p[1] - cy;
    return [(dx * c + dy * s) / U, (-dx * s + dy * c) / U];
  };
  return { cx, cy, U, angle, local };
}

function landmarkFace(map) {
  const P = (i) => map(faceMarks[i][0], faceMarks[i][1]);
  const mid = (i, j) => { const p = P(i), q = P(j); return [(p[0] + q[0]) / 2, (p[1] + q[1]) / 2]; };
  const fr = faceFrame(mid(33, 133), mid(362, 263));
  const L = (i) => fr.local(P(i));
  const top = L(10), chin = L(152);
  const f = {
    ...CAM_FACE,
    faceW: clamp(Math.abs(L(234)[0] - L(454)[0]) / 2, 0.55, 0.95),
    faceH: clamp(((chin[1] - top[1]) / 2) * 1.05, 0.8, 1.3),
    faceCY: clamp((chin[1] + top[1]) / 2 - 0.05, -0.2, 0.25),
  };
  f.hairW = f.faceW * 1.4;
  return { cx: fr.cx, cy: fr.cy, U: fr.U, angle: fr.angle, f };
}

function keypointFace(map) {
  const [eA, eB] = faceBox.keys.map((p) => map(p[0], p[1]));
  const fr = faceFrame(eA, eB);
  return { cx: fr.cx, cy: fr.cy, U: fr.U, angle: fr.angle, f: { ...CAM_FACE } };
}

function easeFace(cur, tgt, e) {
  cur.cx += (tgt.cx - cur.cx) * e;
  cur.cy += (tgt.cy - cur.cy) * e;
  cur.U += (tgt.U - cur.U) * e;
  cur.angle += (tgt.angle - cur.angle) * e;
  for (const key in tgt.f) cur.f[key] += (tgt.f[key] - cur.f[key]) * e;
}

function updateCamera(dt) {
  const v = cam.video;
  const vw = v.videoWidth;
  const vh = v.videoHeight;
  if (!vw || v.readyState < 2) return;
  const tracked = faceBox && performance.now() - faceBox.seen < LOST_MS;
  let target;
  if (tracked) {
    const ch = Math.min(vh * 1.25, faceBox.h / 0.6);
    const cw = (ch * W) / H;
    target = { sx: faceBox.cx - cw / 2, sy: faceBox.cy - ch * 0.45, cw, ch };
  } else {
    const s = Math.max(W / vw, H / vh);
    target = { sx: (vw - W / s) / 2, sy: (vh - H / s) / 2, cw: W / s, ch: H / s };
  }
  if (!camCrop) camCrop = { ...target };
  else {
    const off = Math.max(Math.abs(target.sx - camCrop.sx), Math.abs(target.sy - camCrop.sy), Math.abs(target.ch - camCrop.ch));
    const e = 1 - Math.exp(-dt * Math.min(24, 5 + 60 * (off / camCrop.ch)));
    for (const key of ["sx", "sy", "cw", "ch"]) camCrop[key] += (target[key] - camCrop[key]) * e;
  }
  const crop = camCrop;

  // Mirrored downsample around the face; the field pass band-passes it at the hexagon scale
  rctx.setTransform(1, 0, 0, 1, 0, 0);
  rctx.fillStyle = "#808080";
  rctx.fillRect(0, 0, camRaw.width, camRaw.height);
  rctx.setTransform(-1, 0, 0, 1, camRaw.width, 0);
  rctx.drawImage(v, crop.sx, crop.sy, crop.cw, crop.ch, 0, 0, camRaw.width, camRaw.height);
  gl.activeTexture(gl.TEXTURE1);
  gl.bindTexture(gl.TEXTURE_2D, camTex);
  gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, camRaw);
  camReady = true;

  if (!tracked) return;
  const map = (x, y) => [W - ((x - crop.sx) / crop.cw) * W, ((y - crop.sy) / crop.ch) * H];
  const next = faceMarks ? landmarkFace(map) : faceBox.keys ? keypointFace(map) : null;
  if (!next) return;
  if (!camFace) camFace = next;
  else easeFace(camFace, next, 1 - Math.exp(-dt * 20));
}

camBtn.addEventListener("click", () => (camOn ? stopCamera() : startCamera()));
tlBtn.addEventListener("click", toggleTimelapse);
for (const g of TL_GAPS) tlGapSel.add(new Option(`${g} s`, String(g)));
tlGapSel.value = String(tlGap);
tlGapSel.addEventListener("change", () => setGap(Number(tlGapSel.value)));
document.getElementById("res-up").addEventListener("click", () => setRes(1));
document.getElementById("res-down").addEventListener("click", () => setRes(-1));
addEventListener("keydown", (e) => {
  if (e.key === "c" || e.key === "C") camOn ? stopCamera() : startCamera();
  else if (e.key === "t" || e.key === "T") toggleTimelapse();
  else if (e.key === "e" || e.key === "E") toggleFx();
  else if (e.key === "[") stepGap(-1);
  else if (e.key === "]") stepGap(1);
  else if (e.key === "+" || e.key === "=") setRes(1);
  else if (e.key === "-" || e.key === "_") setRes(-1);
});

let idleTimer = 0;
function wake() {
  document.body.classList.remove("idle");
  clearTimeout(idleTimer);
  idleTimer = setTimeout(() => document.body.classList.add("idle"), 1500);
}
addEventListener("pointermove", wake);
canvas.addEventListener("dblclick", () => {
  if (document.fullscreenElement) document.exitFullscreen();
  else document.documentElement.requestFullscreen?.();
});

let resizeTimer = 0;
addEventListener("resize", () => {
  clearTimeout(resizeTimer);
  resizeTimer = setTimeout(() => {
    scaleCeil = 1;
    layout();
  }, 120);
});

initGL();
layout();
wake();
requestAnimationFrame(frame);
