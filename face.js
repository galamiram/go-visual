function mulberry32(a) {
  return function () {
    a |= 0;
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
const rand = mulberry32((Math.random() * 2 ** 31) | 0);
const clamp = (v, a, b) => (v < a ? a : v > b ? b : v);
const smooth = (x) => { x = clamp(x, 0, 1); return x * x * (3 - 2 * x); };
const lerp = (a, b, k) => a + (b - a) * k;
const range = (a, b) => a + (b - a) * rand();

// Head silhouette in head units (origin = face-oval centre, y grows downward); the face
// itself comes from the statistical model. Order matches the field shader's uP[0..1].
const FACE_KEYS = [
  "faceW", "faceH", "faceCY", "hairW",
  "hairTop", "hairLen", "shoulderW", "neckW",
];

function makeFace() {
  return {
    faceW: range(0.72, 0.8),
    faceH: range(0.98, 1.06),
    faceCY: range(-0.02, 0.04),
    hairW: range(0.9, 1.08),
    hairTop: range(0, 0.14),
    hairLen: rand() < 0.5 ? range(0.5, 1) : range(0, 0.3),
    shoulderW: range(1.5, 1.9),
    neckW: range(0.27, 0.34),
    tilt: range(-0.05, 0.05),
  };
}

// Neutral silhouette for camera mode; landmarks overwrite the measurable parts.
const CAM_FACE = {
  faceW: 0.74, faceH: 1.02, faceCY: 0.02, hairW: 0.98,
  hairTop: 0.05, hairLen: 0.2, shoulderW: 1.7, neckW: 0.31,
  tilt: 0,
};

function mixFace(a, b, k) {
  const out = {};
  for (const key in a) out[key] = lerp(a[key], b[key], k);
  return out;
}

function packFace(f, out) {
  for (let i = 0; i < FACE_KEYS.length; i++) out[i] = f[FACE_KEYS[i]];
  return out;
}
