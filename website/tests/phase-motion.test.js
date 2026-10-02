import test from 'node:test';
import assert from 'node:assert/strict';
import { mountPhaseStory } from '../src/narratives.js';

function element() {
  return {
    dataset: {}, attributes: {}, children: [], events: {}, textContent: '',
    setAttribute(name, value) { this.attributes[name] = String(value); },
    getAttribute(name) { return this.attributes[name]; },
    append(...children) { this.children.push(...children); },
    replaceChildren(...children) { this.children = [...children]; },
    addEventListener(name, handler) { this.events[name] = handler; },
    removeEventListener(name, handler) {
      if (this.events[name] === handler) delete this.events[name];
    },
  };
}

function phaseFixture(content) {
  const steps = Array.from({ length: 5 }, () => {
    const step = element();
    const parts = { span: element(), strong: element(), small: element() };
    step.querySelector = selector => parts[selector];
    return step;
  });
  const svg = element();
  const fields = {
    '[data-phase-scene]': svg,
    '[data-phase-name]': element(),
    '[data-phase-detail]': element(),
    '[data-phase-status]': element(),
    '.story-steps': element(),
  };
  const root = element();
  root.querySelector = selector => fields[selector];
  root.querySelectorAll = () => steps;
  root.closest = () => content;
  root.getClientRects = () => content.dataset.ecosystemPanel === 'phases' ? [{}] : [];
  return { root, svg };
}

function descendants(node) {
  return node.children.flatMap(child => [child, ...descendants(child)]);
}

test('phase motion starts on first visible panel, stays still on rerenders, and pauses when hidden', () => {
  const previous = {
    window: globalThis.window,
    document: globalThis.document,
    IntersectionObserver: globalThis.IntersectionObserver,
    MutationObserver: globalThis.MutationObserver,
  };
  const content = { dataset: { currentView: 'ecosystem', ecosystemPanel: 'capabilities' } };
  const windowEvents = {};
  const documentEvents = {};
  const mediaEvents = {};
  const media = { matches: false,
    addEventListener(name, callback) { mediaEvents[name] = callback; },
    removeEventListener(name) { delete mediaEvents[name]; },
  };
  const timers = new Map();
  let nextTimer = 0;
  let intersect;
  let mutate;
  globalThis.IntersectionObserver = class {
    constructor(callback) { intersect = visible => callback([{ isIntersecting: visible }]); }
    observe() {}
    disconnect() {}
  };
  globalThis.MutationObserver = class {
    constructor(callback) { mutate = callback; }
    observe() {}
    disconnect() {}
  };
  globalThis.window = {
    matchMedia(query) { return query.includes('max-width') ? { matches: this.compact ?? false } : media; },
    setTimeout(callback) { const id = ++nextTimer; timers.set(id, callback); return id; },
    clearTimeout(id) { timers.delete(id); },
    addEventListener(name, callback) { windowEvents[name] = callback; },
    removeEventListener(name) { delete windowEvents[name]; },
  };
  globalThis.document = {
    visibilityState: 'visible',
    createElementNS: () => element(),
    addEventListener(name, callback) { documentEvents[name] = callback; },
    removeEventListener(name) { delete documentEvents[name]; },
  };
  try {
    const { root, svg } = phaseFixture(content);
    const controller = mountPhaseStory(root, { language: 'en' });
    assert.equal(root.dataset.phaseMotion, 'paused');
    assert.equal(root.dataset.phaseReveal, 'off');
    controller.select(1);
    assert.equal(descendants(svg).filter(node => node.getAttribute('class')?.includes('phase-feature-packet')).length, 6);
    assert.equal(root.dataset.phaseReveal, 'off');

    content.dataset.ecosystemPanel = 'phases';
    mutate();
    intersect(true);
    assert.equal(root.dataset.phaseMotion, 'running');
    assert.equal(root.dataset.phaseReveal, 'on');
    for (const callback of timers.values()) callback();
    timers.clear();
    assert.equal(root.dataset.phaseReveal, 'off');

    const sceneBeforeTranslation = svg.children[0];
    controller.setLanguage('ko');
    assert.equal(svg.children[0], sceneBeforeTranslation);
    assert.equal(root.dataset.phaseReveal, 'off');
    globalThis.window.compact = true;
    windowEvents.resize();
    assert.notEqual(svg.children[0], sceneBeforeTranslation);
    assert.equal(root.dataset.phaseReveal, 'off');

    controller.select(2);
    assert.equal(root.dataset.phaseReveal, 'on');
    assert.equal(descendants(svg).filter(node => node.getAttribute('class')?.includes('phase-edge-packet')).length, 6);
    controller.select(4);
    assert.equal(descendants(svg).filter(node => node.getAttribute('class')?.includes('packet')).length, 0);
    assert.equal(descendants(svg).filter(node => node.getAttribute('class')?.includes('phase-voxel')).length, 15);

    globalThis.document.visibilityState = 'hidden';
    documentEvents.visibilitychange();
    assert.equal(root.dataset.phaseMotion, 'paused');
    assert.equal(root.dataset.phaseReveal, 'off');
    globalThis.document.visibilityState = 'visible';
    documentEvents.visibilitychange();
    assert.equal(root.dataset.phaseMotion, 'running');
    assert.equal(root.dataset.phaseReveal, 'off');
    media.matches = true;
    mediaEvents.change();
    assert.equal(root.dataset.phaseMotion, 'paused');
    controller.disconnect();
    assert.equal(windowEvents.resize, undefined);
    assert.equal(documentEvents.visibilitychange, undefined);
  } finally {
    for (const [key, value] of Object.entries(previous)) {
      if (value === undefined) delete globalThis[key];
      else globalThis[key] = value;
    }
  }
});
