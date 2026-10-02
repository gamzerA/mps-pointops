import test from 'node:test';
import assert from 'node:assert/strict';
import { mountMethodsPanels } from '../src/methods-panels.js';

function element() {
  const listeners = new Map();
  return {
    dataset: {}, attributes: {}, children: [], textContent: '',
    append(...children) { this.children.push(...children); },
    remove() { this.removed = true; },
    setAttribute(name, value) { this.attributes[name] = String(value); },
    getAttribute(name) { return this.attributes[name] ?? null; },
    addEventListener(name, listener) { listeners.set(name, listener); },
    click() { listeners.get('click')?.(); },
  };
}

function fixture(hash = '#methods') {
  const shell = element();
  const morton = element();
  const pruning = element();
  const content = element();
  content.querySelector = selector => ({ '#morton': morton, '#pruning': pruning })[selector];
  const root = element();
  root.parentElement = content;
  root.ownerDocument = { createElement: element };
  root.querySelector = selector => selector === '.shell' ? shell : null;
  const listeners = new Map();
  const windowRef = {
    location: { hash },
    addEventListener(name, listener) { listeners.set(name, listener); },
    removeEventListener(name) { listeners.delete(name); },
  };
  const changes = [];
  const mount = () => mountMethodsPanels(root, {
    windowRef, onPanelChange: panel => changes.push(panel),
  });
  const buttons = () => shell.children[0].children;
  return { content, shell, windowRef, listeners, changes, mount, buttons };
}

test('Methods starts with Morton and keeps both demos directly selectable', () => {
  const f = fixture();
  const panels = f.mount();
  assert.equal(panels.getSelectedPanel(), 'morton');
  assert.equal(f.content.dataset.methodsPanel, 'morton');
  assert.equal(f.buttons()[0].getAttribute('aria-pressed'), 'true');
  assert.equal(f.buttons()[1].getAttribute('aria-pressed'), 'false');
  assert.equal(f.buttons()[0].getAttribute('aria-controls'), 'morton');
  assert.equal(f.buttons()[0].children[1].textContent, 'Morton 키');

  f.buttons()[1].click();
  assert.equal(f.content.dataset.methodsPanel, 'pruning');
  assert.equal(f.windowRef.location.hash, '#pruning');
  assert.deepEqual(f.changes, ['morton', 'pruning']);
  f.buttons()[1].click();
  assert.deepEqual(f.changes, ['morton', 'pruning'], 'reselecting a panel avoids unnecessary canvas redraws');

  panels.setLanguage('en');
  assert.equal(f.buttons()[1].children[1].textContent, 'AABB pruning');
  panels.disconnect();
  assert.equal(f.listeners.has('hashchange'), false);
  assert.equal(f.shell.children[0].removed, true);
  assert.equal(f.content.dataset.methodsPanel, undefined);
});

test('Methods honors direct links and browser back/forward selection', () => {
  const f = fixture('#pruning');
  const panels = f.mount();
  assert.equal(panels.getSelectedPanel(), 'pruning');
  assert.deepEqual(f.changes, ['pruning']);
  f.windowRef.location.hash = '#morton';
  f.listeners.get('hashchange')();
  assert.equal(panels.getSelectedPanel(), 'morton');
  f.windowRef.location.hash = '#methods';
  f.listeners.get('hashchange')();
  assert.equal(panels.getSelectedPanel(), 'morton');
  f.windowRef.location.hash = '#route';
  f.listeners.get('hashchange')();
  assert.equal(panels.getSelectedPanel(), 'morton', 'other views leave the choice ready for return');
  assert.deepEqual(f.changes, ['pruning', 'morton']);
});
