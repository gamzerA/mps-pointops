import test from 'node:test';
import assert from 'node:assert/strict';
import { describeMortonGroup } from './formula-animation.js';

test('one group maps x, y, z into the actual 3i lanes', () => {
  const group = describeMortonGroup({ x: 1, y: 1, z: 1, bits: 21, index: 0 });
  assert.equal(group.groupBits, '111');
  assert.deepEqual(group.terms.map(term => term.position), [0, 1, 2]);
  assert.deepEqual(group.terms.map(term => term.value), [1n, 2n, 4n]);
  assert.equal(group.groupContribution, 7n);
  assert.equal(group.fullKey, 7n);
});

test('high bits remain exact beyond Number safe-integer key range', () => {
  const high = 1 << 20;
  const group = describeMortonGroup({ x: high, y: high, z: high, bits: 21, index: 20 });
  assert.equal(group.groupBits, '111');
  assert.deepEqual(group.terms.map(term => term.position), [60, 61, 62]);
  assert.equal(group.fullKey, 7n << 60n);
  assert.equal(group.groupContribution, 7n << 60n);
  assert.ok(group.fullKey > BigInt(Number.MAX_SAFE_INTEGER));
});

test('selected group equals the corresponding slice of the complete key', () => {
  for (let i = 0; i < 21; i += 1) {
    const group = describeMortonGroup({ x: 1_052_679, y: 1_055_707, z: 1_050_681, bits: 21, index: i });
    assert.equal((group.fullKey >> BigInt(3 * i)) & 7n, group.groupValue);
  }
  assert.throws(() => describeMortonGroup({ x: 0, y: 0, z: 0, bits: 21, index: 21 }), RangeError);
});
