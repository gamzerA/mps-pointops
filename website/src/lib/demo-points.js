/** Reproducible [0,1]² fixtures; no Math.random or platform-dependent sampling. */
export function generateDemoPoints({
  count = 2000,
  seed = 0x5eed2026,
  distribution = 'uniform',
} = {}) {
  if (!Number.isSafeInteger(count) || count < 0 || count > 1_000_000) {
    throw new RangeError('count must be an integer from 0 to 1,000,000');
  }
  if (!Number.isSafeInteger(seed) || seed < 0 || seed >= 2 ** 32) {
    throw new RangeError('seed must be a 32-bit unsigned integer');
  }
  if (!['uniform', 'clustered', 'coincident'].includes(distribution)) {
    throw new RangeError('distribution must be uniform, clustered, or coincident');
  }
  if (distribution === 'coincident') {
    return Array.from({ length: count }, () => ({ x: 0.5, y: 0.5 }));
  }

  // The product stays below Number.MAX_SAFE_INTEGER, so each LCG step is exact.
  let state = seed;
  const random = () => {
    state = (1664525 * state + 1013904223) % (2 ** 32);
    return state / (2 ** 32);
  };
  const centers = [[0.22, 0.22], [0.72, 0.30], [0.34, 0.74], [0.77, 0.77]];
  const clamp = (value) => Math.max(0, Math.min(1, value));
  const points = [];
  for (let i = 0; i < count; i += 1) {
    if (distribution === 'uniform' || i % 8 === 0) {
      points.push({ x: random(), y: random() });
      continue;
    }
    const center = centers[Math.floor(random() * centers.length)];
    const jitterX = (random() + random() + random() + random() - 2) * 0.065;
    const jitterY = (random() + random() + random() + random() - 2) * 0.065;
    points.push({ x: clamp(center[0] + jitterX), y: clamp(center[1] + jitterY) });
  }
  return points;
}
