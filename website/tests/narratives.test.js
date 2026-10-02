import test from 'node:test';
import assert from 'node:assert/strict';
import { mountExecutionStory, mountPhaseStory } from '../src/narratives.js';

function element() {
  return {
    textContent: '', attributes: {}, dataset: {}, children: [], events: {},
    setAttribute(name, value) { this.attributes[name] = String(value); },
    getAttribute(name) { return this.attributes[name]; },
    append(...children) { this.children.push(...children); },
    replaceChildren(...children) { this.children = [...children]; },
    addEventListener(name, listener) { this.events[name] = listener; },
    removeEventListener(name, listener) { if (this.events[name] === listener) delete this.events[name]; },
  };
}

function labeledItem(datasetKey, index) {
  const item = element();
  item.dataset[datasetKey] = String(index);
  item.parts = { span: element(), strong: element(), small: element() };
  item.querySelector = selector => item.parts[selector];
  item.click = () => item.events.click();
  return item;
}

function routeFixture() {
  const steps = Array.from({ length: 6 }, (_, index) => labeledItem('routeStep', index));
  const nodes = Array.from({ length: 6 }, (_, index) => labeledItem('routeNode', index));
  const title = element();
  const detail = element();
  const group = element();
  const readout = element();
  const root = element();
  root.querySelectorAll = selector => selector === '[data-route-step]' ? steps :
    selector === '[data-route-node]' ? nodes : [];
  root.querySelector = selector => ({
    '[data-route-title]': title,
    '[data-route-detail]': detail,
    '.story-steps': group,
    '.story-readout': readout,
  })[selector];
  return { root, steps, nodes, title, detail, group, readout };
}

function phaseFixture() {
  const steps = Array.from({ length: 5 }, (_, index) => labeledItem('phaseStep', index));
  const svg = element();
  const name = element();
  const detail = element();
  const status = element();
  const group = element();
  const root = element();
  root.querySelectorAll = selector => selector === '[data-phase-step]' ? steps : [];
  root.querySelector = selector => ({
    '[data-phase-scene]': svg,
    '[data-phase-name]': name,
    '[data-phase-detail]': detail,
    '[data-phase-status]': status,
    '.story-steps': group,
  })[selector];
  return { root, steps, svg, name, detail, status, group };
}

test('route and Phase controllers translate active details, buttons, and accessible names', () => {
  const previousWindow = globalThis.window;
  const previousDocument = globalThis.document;
  let now = 0;
  let nextTimerId = 1;
  const timers = new Map();
  const advance = milliseconds => {
    now += milliseconds;
    for (const [id, timer] of [...timers]) {
      if (timer.at <= now) { timers.delete(id); timer.fn(); }
    }
  };
  globalThis.window = {
    setTimeout(fn, delay) {
      const id = nextTimerId++;
      timers.set(id, { fn, at: now + delay });
      return id;
    },
    clearTimeout(id) { timers.delete(id); },
  };
  globalThis.document = { createElementNS: () => element() };
  try {
    const route = routeFixture();
    const routeController = mountExecutionStory(route.root, { language: 'ko' });
    assert.equal(route.title.textContent, 'CUDA 전용 호출');
    assert.equal(route.steps[0].getAttribute('aria-current'), 'step');
    routeController.select(4);
    assert.equal(route.title.textContent, 'Metal 커널');
    assert.equal(route.nodes[4].dataset.state, 'active');
    routeController.setLanguage('en');
    assert.equal(route.title.textContent, 'Metal kernel');
    assert.match(route.detail.textContent, /Apple GPU/);
    assert.equal(route.group.getAttribute('aria-label'), 'Choose an execution step');
    assert.match(route.steps[0].getAttribute('aria-label'), /CUDA-only operator/);
    assert.equal(route.nodes[0].parts.small.textContent, 'Unavailable on Apple Silicon');
    route.steps[2].click();
    assert.equal(route.title.textContent, 'Compatibility layer');
    assert.equal(route.root.dataset.routeActive, '2');

    route.steps[4].events.pointerenter({ pointerType: 'mouse' });
    assert.equal(route.root.dataset.routeActive, '2');
    advance(80);
    assert.equal(route.title.textContent, 'Metal kernel');
    assert.equal(route.nodes[4].dataset.state, 'active');
    assert.equal(route.readout.getAttribute('aria-live'), 'off');
    route.steps[4].events.pointerleave({ pointerType: 'mouse' });
    route.nodes[1].events.pointerenter({ pointerType: 'mouse' });
    advance(80);
    assert.equal(route.root.dataset.routeActive, '1');
    route.nodes[5].events.pointerenter({ pointerType: 'touch' });
    assert.equal(route.root.dataset.routeActive, '1');
    route.nodes[1].events.pointerleave({ pointerType: 'mouse' });
    advance(100);
    assert.equal(route.root.dataset.routeActive, '2');
    route.steps[5].click();
    assert.equal(route.root.dataset.routeActive, '5');
    assert.equal(route.readout.getAttribute('aria-live'), 'polite');
    route.steps[3].matches = selector => selector === ':focus-visible';
    globalThis.document.activeElement = route.steps[3];
    route.steps[3].events.focus();
    route.nodes[4].events.pointerenter({ pointerType: 'mouse' });
    advance(80);
    assert.equal(route.root.dataset.routeActive, '5');
    globalThis.document.activeElement = null;

    const phases = phaseFixture();
    const phaseController = mountPhaseStory(phases.root, { language: 'en' });
    phaseController.select(4);
    assert.equal(phases.name.textContent, 'Sparse 3D computation');
    assert.match(phases.status.textContent, /not implemented/);
    assert.match(phases.svg.getAttribute('aria-label'), /Sparse 3D computation/);
    assert.equal(phases.group.getAttribute('aria-label'), 'Choose a development phase');
    assert.match(phases.steps[4].getAttribute('aria-label'), /Planned/);
    phaseController.setLanguage('ko');
    assert.equal(phases.name.textContent, '희소 3D 계산');
    assert.match(phases.svg.getAttribute('aria-label'), /미구현/);
    assert.equal(phases.steps[4].getAttribute('aria-current'), 'step');
    phases.steps[1].click();
    assert.equal(phases.name.textContent, '점과 피처');
    routeController.disconnect();
    assert.equal(route.nodes[1].events.pointerenter, undefined);
    phaseController.disconnect();
  } finally {
    if (previousWindow === undefined) delete globalThis.window;
    else globalThis.window = previousWindow;
    if (previousDocument === undefined) delete globalThis.document;
    else globalThis.document = previousDocument;
  }
});
