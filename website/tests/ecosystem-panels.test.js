import test from 'node:test';
import assert from 'node:assert/strict';
import { mountEcosystemPanels } from '../src/ecosystem-panels.js';

function control(dataset, parts = {}) {
  const listeners = new Map();
  return {
    dataset, parts, attributes: {}, focusVisible: false,
    querySelector: selector => parts[selector],
    getAttribute(name) { return this.attributes[name] ?? null; },
    setAttribute(name, value) { this.attributes[name] = String(value); },
    matches(selector) { return selector === ':focus-visible' && this.focusVisible; },
    addEventListener(name, listener) {
      if (!listeners.has(name)) listeners.set(name, new Set());
      listeners.get(name).add(listener);
    },
    removeEventListener(name, listener) { listeners.get(name)?.delete(listener); },
    dispatch(name, event = {}) { for (const listener of listeners.get(name) ?? []) listener(event); },
    click() { this.dispatch('click'); },
    focus() { this.dispatch('focus'); },
    pointerEnter(pointerType = 'mouse', buttons = 0) { this.dispatch('pointerenter', { pointerType, buttons }); },
    pointerLeave() { this.dispatch('pointerleave'); },
  };
}

function fixture(hash = '#capabilities') {
  const listeners = new Map();
  const timers = new Map();
  let nextTimer = 0;
  let now = 0;
  const buttons = ['capabilities', 'phases'].map(panel => control(
    { ecosystemPanelOption: panel },
    { strong: { textContent: '' }, small: { textContent: '' } },
  ));
  const modelOptions = ['pointnet2', 'pyg'].map(id => control({ capability: id }));
  const phaseOptions = Array.from({ length: 5 }, (_, index) => control({ phaseStep: String(index) }));
  const selection = { model: 'pointnet2', phase: 0 };
  const calls = { model: 0, phase: 0 };
  const capabilities = {
    select(id) {
      calls.model += 1;
      selection.model = id;
      modelOptions.forEach(option => option.setAttribute('aria-pressed', String(option.dataset.capability === id)));
    },
  };
  const phases = {
    select(index) {
      calls.phase += 1;
      selection.phase = index;
      phaseOptions.forEach(option => option.setAttribute('aria-current', Number(option.dataset.phaseStep) === index ? 'step' : 'false'));
    },
  };
  capabilities.select(selection.model);
  phases.select(selection.phase);
  // The mounted explorers already own click activation before the hover layer is added.
  modelOptions.forEach(option => option.addEventListener('click', () => capabilities.select(option.dataset.capability)));
  phaseOptions.forEach(option => option.addEventListener('click', () => phases.select(Number(option.dataset.phaseStep))));
  const group = control({});
  const region = control({});
  const phaseRegion = control({});
  const detail = control({});
  const evidenceLink = control({});
  region.contains = node => modelOptions.includes(node) || node === evidenceLink;
  phaseRegion.contains = node => phaseOptions.includes(node);
  detail.contains = node => node === evidenceLink;
  const root = {
    dataset: {}, ownerDocument: { activeElement: null },
    querySelectorAll: selector => ({
      '[data-ecosystem-panel-option]': buttons,
      '#capability-map [data-capability]': modelOptions,
      '#phases [data-phase-step]': phaseOptions,
    })[selector] ?? [],
    querySelector: selector => ({
      '.ecosystem-switch': group, '#capability-map': region,
      '#phases': phaseRegion, '.capability-detail': detail,
    })[selector],
  };
  const windowRef = {
    location: { hash },
    addEventListener(name, listener) { listeners.set(name, listener); },
    removeEventListener(name) { listeners.delete(name); },
    setTimeout(callback, delay) {
      const id = ++nextTimer;
      timers.set(id, { callback, due: now + delay });
      return id;
    },
    clearTimeout(id) { timers.delete(id); },
  };
  const advanceHover = milliseconds => {
    now += milliseconds;
    for (const [id, timer] of [...timers]) {
      if (timer.due <= now) {
        timers.delete(id);
        timer.callback();
      }
    }
  };
  const flushHover = () => advanceHover(1000);
  return { root, buttons, modelOptions, phaseOptions, group, region, evidenceLink,
    capabilities, phases, selection, calls, windowRef, listeners, timers, advanceHover, flushHover };
}

function mount(f, language = 'ko') {
  return mountEcosystemPanels(f.root, {
    language, windowRef: f.windowRef, capabilities: f.capabilities, phases: f.phases,
  });
}

test('the ecosystem chooser selects a direct deep link and keeps both panels one click away', () => {
  const f = fixture('#phases');
  const panels = mount(f);
  assert.equal(panels.getSelectedPanel(), 'phases');
  assert.equal(f.root.dataset.ecosystemPanel, 'phases');
  assert.equal(f.buttons[1].attributes['aria-pressed'], 'true');
  assert.equal(f.buttons[0].parts.strong.textContent, '모델별 검증');

  f.buttons[0].click();
  assert.equal(panels.getSelectedPanel(), 'capabilities');
  panels.setLanguage('en');
  assert.equal(f.buttons[1].parts.strong.textContent, 'Compute stack');
  assert.equal(f.region.attributes['aria-label'], 'Model validation scope');

  f.windowRef.location.hash = '#route';
  f.listeners.get('hashchange')();
  assert.equal(panels.getSelectedPanel(), 'capabilities');
  f.windowRef.location.hash = '#phases';
  f.listeners.get('hashchange')();
  assert.equal(panels.getSelectedPanel(), 'phases');
  panels.disconnect();
  assert.equal(f.listeners.has('hashchange'), false);
});

test('settled mouse hover previews model and phase choices without flicker', () => {
  const f = fixture();
  const panels = mount(f);
  const initialModelRenders = f.calls.model;
  f.modelOptions[0].pointerEnter();
  f.flushHover();
  assert.equal(f.calls.model, initialModelRenders, 'hovering the selected card avoids a detail rebuild');
  f.modelOptions[1].pointerEnter();
  f.modelOptions[1].pointerLeave();
  f.flushHover();
  assert.equal(f.selection.model, 'pointnet2', 'passing across a card does not flash its detail');
  f.modelOptions[1].pointerEnter();
  f.flushHover();
  assert.equal(f.selection.model, 'pyg');
  f.modelOptions[1].pointerLeave();
  assert.equal(f.selection.model, 'pyg', 'preview stays selected after pointer exit');

  f.buttons[1].click();
  assert.equal(panels.getSelectedPanel(), 'phases');
  const initialPhaseRenders = f.calls.phase;
  f.phaseOptions[0].pointerEnter();
  f.flushHover();
  assert.equal(f.calls.phase, initialPhaseRenders, 'hovering the selected phase avoids a scene redraw');
  f.phaseOptions[4].pointerEnter();
  f.flushHover();
  assert.equal(f.selection.phase, 4);
  panels.disconnect();
});

test('panel chooser ignores pointer travel while model cards retain a fast preview', () => {
  const f = fixture();
  const panels = mount(f);
  f.buttons[1].pointerEnter('mouse', 1);
  f.flushHover();
  assert.equal(panels.getSelectedPanel(), 'capabilities', 'dragging across a chooser does not switch panels');

  f.buttons[1].pointerEnter();
  f.flushHover();
  assert.equal(panels.getSelectedPanel(), 'capabilities', 'hovering a chooser does not switch panels');

  f.modelOptions[1].pointerEnter();
  f.advanceHover(89);
  assert.equal(f.selection.model, 'pointnet2');
  f.advanceHover(1);
  assert.equal(f.selection.model, 'pyg', 'model hover still previews after 90 ms');

  f.buttons[1].click();
  assert.equal(panels.getSelectedPanel(), 'phases', 'click switches panels immediately');
  panels.disconnect();
});

test('touch clicks and keyboard focus are immediate; hover protects focused evidence', () => {
  const f = fixture();
  const panels = mount(f);
  f.buttons[1].pointerEnter('touch');
  f.flushHover();
  assert.equal(panels.getSelectedPanel(), 'capabilities');

  f.root.ownerDocument.activeElement = f.evidenceLink;
  f.buttons[1].pointerEnter();
  f.flushHover();
  assert.equal(panels.getSelectedPanel(), 'capabilities', 'hover does not hide a focused evidence link');
  f.modelOptions[1].pointerEnter();
  f.flushHover();
  assert.equal(f.selection.model, 'pointnet2', 'hover does not rebuild the focused detail');
  f.buttons[1].click();
  assert.equal(panels.getSelectedPanel(), 'phases', 'click remains an explicit panel choice');

  f.root.ownerDocument.activeElement = f.phaseOptions[2];
  f.phaseOptions[2].focusVisible = true;
  f.phaseOptions[2].focus();
  assert.equal(f.selection.phase, 2, 'keyboard focus previews the phase');
  f.phaseOptions[3].pointerEnter();
  f.flushHover();
  assert.equal(f.selection.phase, 2, 'hover does not override visible keyboard focus');
  f.phaseOptions[1].pointerEnter('touch');
  f.flushHover();
  assert.equal(f.selection.phase, 2);
  f.phaseOptions[1].click();
  assert.equal(f.selection.phase, 1, 'touch click activates the existing phase control');

  f.buttons[0].click();
  f.root.ownerDocument.activeElement = f.modelOptions[1];
  f.modelOptions[1].focusVisible = true;
  f.modelOptions[1].focus();
  assert.equal(f.selection.model, 'pyg', 'keyboard focus previews model evidence');
  f.modelOptions[0].pointerEnter();
  f.modelOptions[1].click();
  f.flushHover();
  assert.equal(f.selection.model, 'pyg', 'click cancels a pending hover');
  panels.disconnect();
});
