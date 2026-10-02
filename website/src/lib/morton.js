/** Exact 63-bit, 3-axis Morton coding. Axis bits occupy 3i, 3i+1, and 3i+2. */
function validateBits(bits) {
  if (!Number.isInteger(bits) || bits < 1 || bits > 21) {
    throw new RangeError('bits must be an integer from 1 to 21');
  }
  return bits;
}

function validateCoordinate(value, bits, name) {
  if (typeof value !== 'bigint' &&
      (typeof value !== 'number' || !Number.isSafeInteger(value))) {
    throw new TypeError(`${name} must be a safe integer Number or BigInt`);
  }
  const coordinate = BigInt(value);
  const maximum = (1n << BigInt(bits)) - 1n;
  if (coordinate < 0n || coordinate > maximum) {
    throw new RangeError(`${name} must be between 0 and ${maximum}`);
  }
  return coordinate;
}

export function encodeMorton3D(x, y, z, bits = 21) {
  validateBits(bits);
  const axes = [
    validateCoordinate(x, bits, 'x'),
    validateCoordinate(y, bits, 'y'),
    validateCoordinate(z, bits, 'z'),
  ];
  let code = 0n;
  for (let i = 0; i < bits; i += 1) {
    const shift = BigInt(i);
    const position = BigInt(3 * i);
    code |= ((axes[0] >> shift) & 1n) << position;
    code |= ((axes[1] >> shift) & 1n) << (position + 1n);
    code |= ((axes[2] >> shift) & 1n) << (position + 2n);
  }
  return code;
}

export function decodeMorton3D(code, bits = 21) {
  validateBits(bits);
  if (typeof code !== 'bigint') {
    throw new TypeError('code must be a BigInt');
  }
  const maximum = (1n << BigInt(3 * bits)) - 1n;
  if (code < 0n || code > maximum) {
    throw new RangeError(`code must be between 0 and ${maximum}`);
  }
  const axes = [0n, 0n, 0n];
  for (let i = 0; i < bits; i += 1) {
    const shift = BigInt(i);
    const position = BigInt(3 * i);
    axes[0] |= ((code >> position) & 1n) << shift;
    axes[1] |= ((code >> (position + 1n)) & 1n) << shift;
    axes[2] |= ((code >> (position + 2n)) & 1n) << shift;
  }
  return { x: Number(axes[0]), y: Number(axes[1]), z: Number(axes[2]) };
}
