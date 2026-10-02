import test from 'node:test';
import assert from 'node:assert/strict';
import { decodeMorton3D, encodeMorton3D } from '../src/lib/morton.js';

test('Morton axis lanes use exact BigInt bits above Number precision', () => {
  assert.equal(encodeMorton3D(1, 0, 0), 1n);
  assert.equal(encodeMorton3D(0, 1, 0), 2n);
  assert.equal(encodeMorton3D(0, 0, 1), 4n);
  assert.equal(encodeMorton3D(2 ** 20, 0, 0), 1n << 60n);
  assert.equal(encodeMorton3D(0, 2 ** 20, 0), 1n << 61n);
  assert.equal(encodeMorton3D(0, 0, 2 ** 20), 1n << 62n);
  assert.equal(encodeMorton3D(2 ** 21 - 1, 2 ** 21 - 1, 2 ** 21 - 1),
    (1n << 63n) - 1n);
});

test('Morton decoding round trips 21-bit axes', () => {
  const values = [0, 1, 2, 17, 1024, 2 ** 20, 2 ** 21 - 1];
  for (const x of values) {
    for (const y of values) {
      const z = (x + y) % (2 ** 21);
      assert.deepEqual(decodeMorton3D(encodeMorton3D(x, y, z)), { x, y, z });
    }
  }
  assert.equal(encodeMorton3D(7n, 3n, 2n, 3), encodeMorton3D(7, 3, 2, 3));
});

test('Morton rejects out-of-range and imprecise inputs', () => {
  assert.throws(() => encodeMorton3D(2 ** 21, 0, 0), RangeError);
  assert.throws(() => encodeMorton3D(-1, 0, 0), RangeError);
  assert.throws(() => encodeMorton3D(0.5, 0, 0), TypeError);
  assert.throws(() => encodeMorton3D(2 ** 53, 0, 0), TypeError);
  assert.throws(() => encodeMorton3D(0, 0, 0, 22), RangeError);
  assert.throws(() => encodeMorton3D(0, 0, 0, 0), RangeError);
  assert.throws(() => decodeMorton3D(2 ** 53), TypeError);
  assert.throws(() => decodeMorton3D(1n << 63n), RangeError);
});
