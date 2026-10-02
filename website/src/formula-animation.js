import { encodeMorton3D } from './lib/morton.js';

function node(tag, className, content) {
  const value = document.createElement(tag);
  if (className) value.className = className;
  if (content !== undefined) value.textContent = String(content);
  return value;
}

function reducedMotion() {
  return typeof window !== 'undefined'
    && (window.matchMedia?.('(prefers-reduced-motion: reduce)').matches ?? false);
}

let nextBitRangeId = 0;

function parseBitIndex(value, bits, clamp = false) {
  const text = String(value).trim();
  if (!/^[+-]?\d+$/.test(text)) return null;
  const parsed = BigInt(text);
  const maximum = BigInt(bits - 1);
  if (!clamp && (parsed < 0n || parsed > maximum)) return null;
  return Number(parsed < 0n ? 0n : parsed > maximum ? maximum : parsed);
}

/** Pure description of one three-bit Morton group, with exact BigInt terms. */
export function describeMortonGroup({ x, y, z, bits = 21, index = 0 }) {
  const fullKey = encodeMorton3D(x, y, z, bits);
  if (!Number.isInteger(index) || index < 0 || index >= bits) {
    throw new RangeError(`index must be an integer from 0 to ${bits - 1}`);
  }
  const coordinates = [BigInt(x), BigInt(y), BigInt(z)];
  const terms = coordinates.map((coordinate, axis) => {
    const bit = (coordinate >> BigInt(index)) & 1n;
    const position = 3 * index + axis;
    return { axis: ['x', 'y', 'z'][axis], bit, position, value: bit << BigInt(position) };
  });
  const groupValue = (terms[2].bit << 2n) | (terms[1].bit << 1n) | terms[0].bit;
  const groupBits = `${terms[2].bit}${terms[1].bit}${terms[0].bit}`;
  const groupContribution = groupValue << BigInt(3 * index);
  if (((fullKey >> BigInt(3 * index)) & 7n) !== groupValue) {
    throw new Error('Morton group does not match the full key');
  }
  if (terms.reduce((sum, term) => sum + term.value, 0n) !== groupContribution) {
    throw new Error('Morton terms do not sum to the group contribution');
  }
  return { terms, groupValue, groupBits, groupContribution, fullKey, index, bits };
}

/** Show bit extraction, lane shifts, one group, and the full BigInt key in order. */
export function mountMortonFormula(container, initial = {}) {
  if (!container || typeof container.replaceChildren !== 'function') {
    throw new TypeError('container must be a DOM element');
  }
  if (typeof initial.translate !== 'function') {
    throw new TypeError('translate(key, language) must be a function');
  }
  const state = {
    x: initial.x ?? 0, y: initial.y ?? 0, z: initial.z ?? 0,
    bits: initial.bits ?? 21, index: 0, stage: 0, timer: null, animation: null,
    language: initial.language ?? 'ko', translate: initial.translate,
  };
  const events = new AbortController();
  const section = node('section', 'morton-formula-explainer');
  const heading = node('h3', 'morton-formula-title');
  const equation = node('div', 'morton-formula-equation');
  equation.setAttribute('role', 'math');
  const controls = node('div', 'morton-formula-controls');
  const bitControl = node('div', 'morton-formula-bit-control');
  const bitLabel = node('label', 'morton-formula-select-label');
  const bitText = node('span', 'morton-formula-control-text');
  const bitInput = node('input', 'morton-formula-select morton-formula-bit-input');
  const bitRange = node('span', 'morton-formula-input-range');
  bitInput.type = 'number';
  bitInput.min = '0';
  bitInput.step = '1';
  bitInput.inputMode = 'numeric';
  bitInput.autocomplete = 'off';
  bitRange.id = `morton-formula-bit-range-${++nextBitRangeId}`;
  bitInput.setAttribute('aria-describedby', bitRange.id);
  bitLabel.append(bitText, bitInput);
  bitControl.append(bitLabel, bitRange);
  const previous = node('button', 'morton-formula-previous');
  const next = node('button', 'morton-formula-next');
  const play = node('button', 'morton-formula-play');
  for (const button of [previous, next, play]) button.type = 'button';
  controls.append(bitControl, previous, next, play);
  const status = node('p', 'morton-formula-progress');
  status.setAttribute('aria-live', 'polite');
  const display = node('div', 'morton-formula-stage');
  display.setAttribute('role', 'math');
  const exact = node('p', 'morton-formula-exact');
  section.append(heading, equation, controls, status, display, exact);
  container.replaceChildren(section);

  function copy(key) { return state.translate(key, state.language); }
  function pause() {
    if (state.timer !== null) clearInterval(state.timer);
    state.timer = null;
  }
  function render(animate = false) {
    const group = describeMortonGroup(state);
    heading.textContent = copy('formulaTitle');
    equation.innerHTML = `<i>M</i>(x, y, z) = <span class="math-sum">∑<sub>i=0</sub><sup>${state.bits - 1}</sup></span> (x<sub>i</sub>·2<sup>3i</sup> + y<sub>i</sub>·2<sup>3i+1</sup> + z<sub>i</sub>·2<sup>3i+2</sup>)`;
    equation.setAttribute('aria-label', copy('formulaEquationAria').replace('{last}', String(state.bits - 1)));
    bitText.textContent = copy('formulaChoose');
    bitInput.max = String(state.bits - 1);
    bitInput.value = String(state.index);
    bitRange.textContent = `0–${state.bits - 1}`;
    const [xTerm, yTerm, zTerm] = group.terms;
    const extract = `<span>x<sub>${state.index}</sub> = (x ≫ ${state.index}) &amp; 1 = ${xTerm.bit}</span><span>y<sub>${state.index}</sub> = (y ≫ ${state.index}) &amp; 1 = ${yTerm.bit}</span><span>z<sub>${state.index}</sub> = (z ≫ ${state.index}) &amp; 1 = ${zTerm.bit}</span>`;
    const shift = group.terms.map(term => `<span>${term.axis}<sub>${state.index}</sub>·2<sup>${term.position}</sup> = ${term.bit}·2<sup>${term.position}</sup> = ${term.value}</span>`).join('');
    const stages = [
      { name: 'formulaStageBits', html: extract, aria: copy('formulaBitRuleAria').replaceAll('{index}', String(state.index)) },
      { name: 'formulaStageLanes', html: shift, aria: copy('formulaLaneAria').replaceAll('{index}', String(state.index)) },
      { name: 'formulaStageGroup', html: `<span>z<sub>${state.index}</sub> y<sub>${state.index}</sub> x<sub>${state.index}</sub> = ${group.groupBits}<sub>2</sub></span><span>${group.groupBits}<sub>2</sub> · 2<sup>${3 * state.index}</sup> = ${group.groupContribution}</span>`, aria: copy('formulaGroupAria').replace('{bits}', group.groupBits).replace('{value}', String(group.groupContribution)) },
      { name: 'formulaStageSum', html: `<span><i>M</i>(x, y, z) = ${group.fullKey}</span>`, aria: `${copy('formulaFull')}: ${group.fullKey}` },
    ];
    const selected = stages[state.stage];
    status.textContent = `${state.stage + 1}/4 · ${copy(selected.name)}`;
    display.innerHTML = selected.html;
    display.setAttribute('aria-label', selected.aria);
    display.dataset.stage = String(state.stage);
    exact.textContent = copy('formulaExact');
    previous.textContent = copy('formulaPrevious');
    next.textContent = copy('formulaNext');
    play.textContent = copy(state.timer !== null ? 'formulaPause'
      : state.stage === 3 ? 'formulaReplay' : 'formulaPlay');
    previous.disabled = state.stage === 0;
    next.disabled = state.stage === 3;
    state.animation?.cancel();
    if (animate && !reducedMotion() && typeof display.animate === 'function') {
      state.animation = display.animate([
        { opacity: .55, transform: 'translateY(5px)' },
        { opacity: 1, transform: 'translateY(0)' },
      ], { duration: 240, easing: 'cubic-bezier(0.23, 1, 0.32, 1)' });
    }
  }
  function select(stage, animate = true) {
    state.stage = Math.max(0, Math.min(3, stage));
    render(animate);
  }
  function setBitIndex(index) {
    if (index === state.index) return;
    pause();
    state.index = index;
    select(0);
  }
  bitInput.addEventListener('focus', () => {
    if (state.timer !== null) { pause(); render(); }
  }, { signal: events.signal });
  bitInput.addEventListener('input', () => {
    const index = parseBitIndex(bitInput.value, state.bits);
    if (index !== null) setBitIndex(index);
  }, { signal: events.signal });
  function commitBitIndex() {
    const index = parseBitIndex(bitInput.value, state.bits, true);
    if (index !== null) setBitIndex(index);
    bitInput.value = String(state.index);
  }
  bitInput.addEventListener('change', commitBitIndex, { signal: events.signal });
  bitInput.addEventListener('keydown', event => {
    if (event.key === 'Enter') {
      event.preventDefault();
      commitBitIndex();
    } else if (event.key === 'Escape') {
      bitInput.value = String(state.index);
    }
  }, { signal: events.signal });
  previous.addEventListener('click', () => { pause(); select(state.stage - 1); }, { signal: events.signal });
  next.addEventListener('click', () => { pause(); select(state.stage + 1); }, { signal: events.signal });
  play.addEventListener('click', () => {
    if (state.timer !== null) { pause(); render(); return; }
    if (state.stage === 3) select(0, false);
    if (reducedMotion()) { select(3, false); return; }
    state.timer = setInterval(() => {
      if (state.stage === 3) { pause(); render(); return; }
      select(state.stage + 1);
      if (state.stage === 3) { pause(); render(); }
    }, 1200);
    render();
  }, { signal: events.signal });
  const update = (next = {}) => {
    pause();
    for (const axis of ['x', 'y', 'z', 'bits']) {
      if (Object.hasOwn(next, axis)) state[axis] = next[axis];
    }
    if (Object.hasOwn(next, 'language')) state.language = next.language;
    if (Object.hasOwn(next, 'translate')) state.translate = next.translate;
    const computed = encodeMorton3D(state.x, state.y, state.z, state.bits);
    if (Object.hasOwn(next, 'key') && BigInt(next.key) !== computed) {
      throw new RangeError('provided key does not match x, y, z, and bits');
    }
    state.index = Math.min(state.index, state.bits - 1);
    render();
  };
  update(initial);
  return {
    update,
    getStage: () => state.stage,
    destroy() {
      pause();
      state.animation?.cancel();
      events.abort();
      container.replaceChildren();
    },
  };
}

const OPERATOR_STAGES = [
  {
    id: 'distance', title: 'operatorDistanceTitle', note: 'operatorDistanceNote',
    aria: 'operatorDistanceAria',
    formula: `<span><i>s</i><sub>bij</sub> = <span class="math-sum">∑<sub>d=0</sub><sup>2</sup></span> (<i>q</i><sub>bid</sub> − <i>x</i><sub>bjd</sub>)<sup>2</sup></span>`,
  },
  {
    id: 'fps', title: 'operatorFpsTitle', note: 'operatorFpsNote',
    aria: 'operatorFpsAria',
    formula: `<span><i>m</i><sub>j</sub><sup>(t)</sup> = min<sub>0≤u≤t</sub> ∑<sub>d=0</sub><sup>2</sup> (<i>x</i><sub>bjd</sub> − <i>x</i><sub>b,cᵤ,d</sub>)<sup>2</sup></span><span><i>c</i><sub>t+1</sub> = min argmax<sub>j</sub> <i>m</i><sub>j</sub><sup>(t)</sup></span>`,
  },
  {
    id: 'knn', title: 'operatorKnnTitle', note: 'operatorKnnNote',
    aria: 'operatorKnnAria',
    formula: `<span><i>π</i><sub>bi</sub> = argsort<sub>j</sub>(<i>s</i><sub>bij</sub>, j)</span><span><i>I</i><sub>bik</sub> = <i>π</i><sub>bi</sub>[k] · <i>D</i><sub>bik</sub> = √<i>s</i><sub>bi,<span>I<sub>bik</sub></span></sub></span>`,
  },
  {
    id: 'ball', title: 'operatorBallTitle', note: 'operatorBallNote',
    aria: 'operatorBallAria',
    formula: `<span><i>R</i><sub>2</sub> = fl<sub>32</sub>(fl<sub>32</sub>(r) · fl<sub>32</sub>(r))</span><span><i>J</i><sub>bi</sub> = [j : <i>s</i><sub>bij</sub> &lt; <i>R</i><sub>2</sub>]<sub>input order</sub></span><span>(<i>I</i><sub>bik</sub>, <i>S</i><sub>bik</sub>) = (<i>J</i><sub>bi</sub>[k], <i>s</i><sub>bi,<span>J<sub>bi</sub>[k]</span></sub>) if k &lt; min(K, |<i>J</i><sub>bi</sub>|); else (−1, 0)</span>`,
  },
  {
    id: 'gradient', title: 'operatorGradientTitle', note: 'operatorGradientNote',
    aria: 'operatorGradientAria',
    formula: `<span>∂<i>L</i>/∂<i>q</i><sub>bid</sub> = 2∑<sub>k:<span>I<sub>bik</sub></span>≥0</sub> <i>G</i><sub>bik</sub>(<i>q</i><sub>bid</sub> − <i>x</i><sub>b,<span>I<sub>bik</sub></span>,d</sub>)</span><span>∂<i>L</i>/∂<i>x</i><sub>bjd</sub> = 2∑<sub>i,k:<span>I<sub>bik</sub></span>=j</sub> <i>G</i><sub>bik</sub>(<i>x</i><sub>bjd</sub> − <i>q</i><sub>bid</sub>)</span>`,
  },
];

/** Interactive derivation of the operator equations printed in the project README. */
export function mountOperatorDerivation(container, initial = {}) {
  if (!container || typeof container.replaceChildren !== 'function') {
    throw new TypeError('container must be a DOM element');
  }
  if (typeof initial.translate !== 'function') {
    throw new TypeError('translate(key, language) must be a function');
  }
  const state = { language: initial.language ?? 'ko', translate: initial.translate,
    stage: 0, timer: null, animation: null };
  const events = new AbortController();
  const section = node('section', 'operator-derivation');
  const heading = node('h3', 'operator-derivation-title');
  const stageButtons = node('div', 'operator-derivation-stages');
  stageButtons.setAttribute('role', 'group');
  const buttons = OPERATOR_STAGES.map((stage, index) => {
    const button = node('button', 'operator-stage-option');
    button.type = 'button';
    button.dataset.operatorStage = stage.id;
    button.addEventListener('click', () => { pause(); select(index); }, { signal: events.signal });
    stageButtons.append(button);
    return button;
  });
  const display = node('div', 'operator-derivation-display');
  const current = node('span', 'operator-derivation-current');
  const formula = node('div', 'operator-derivation-formula');
  formula.setAttribute('role', 'math');
  const note = node('p', 'operator-derivation-note');
  display.append(current, formula, note);
  const controls = node('div', 'operator-derivation-controls');
  const previous = node('button', 'operator-derivation-previous');
  const next = node('button', 'operator-derivation-next');
  const play = node('button', 'operator-derivation-play');
  for (const button of [previous, next, play]) button.type = 'button';
  const progress = node('span', 'operator-derivation-progress');
  progress.setAttribute('aria-live', 'polite');
  controls.append(previous, next, play, progress);
  section.append(heading, stageButtons, display, controls);
  container.replaceChildren(section);

  function copy(key) { return state.translate(key, state.language); }
  function pause() {
    if (state.timer !== null) clearInterval(state.timer);
    state.timer = null;
  }
  function renderControls() {
    previous.textContent = copy('operatorPrevious');
    next.textContent = copy('operatorNext');
    play.textContent = copy(state.timer !== null ? 'operatorPause'
      : state.stage === OPERATOR_STAGES.length - 1 ? 'operatorReplay' : 'operatorPlay');
    previous.disabled = state.stage === 0;
    next.disabled = state.stage === OPERATOR_STAGES.length - 1;
  }
  function select(index, animate = true) {
    state.stage = Math.max(0, Math.min(OPERATOR_STAGES.length - 1, index));
    const selected = OPERATOR_STAGES[state.stage];
    current.textContent = `${String(state.stage + 1).padStart(2, '0')} / 0${OPERATOR_STAGES.length} · ${copy(selected.title)}`;
    formula.innerHTML = selected.formula;
    formula.setAttribute('aria-label', copy(selected.aria));
    note.textContent = copy(selected.note);
    buttons.forEach((button, i) => button.setAttribute('aria-pressed', String(i === state.stage)));
    progress.textContent = copy('operatorProgress').replace('{current}', String(state.stage + 1))
      .replace('{total}', String(OPERATOR_STAGES.length));
    renderControls();
    state.animation?.cancel();
    if (animate && !reducedMotion() && typeof display.animate === 'function') {
      state.animation = display.animate([
        { opacity: .64, transform: 'translateY(5px)' },
        { opacity: 1, transform: 'translateY(0)' },
      ], { duration: 240, easing: 'cubic-bezier(0.23, 1, 0.32, 1)' });
    }
  }
  function renderLanguage() {
    heading.textContent = copy('operatorFormulaTitle');
    stageButtons.setAttribute('aria-label', copy('operatorStageGroup'));
    buttons.forEach((button, i) => { button.textContent = copy(OPERATOR_STAGES[i].title); });
    select(state.stage, false);
  }
  previous.addEventListener('click', () => { pause(); select(state.stage - 1); }, { signal: events.signal });
  next.addEventListener('click', () => { pause(); select(state.stage + 1); }, { signal: events.signal });
  play.addEventListener('click', () => {
    if (state.timer !== null) { pause(); select(state.stage, false); return; }
    if (state.stage === OPERATOR_STAGES.length - 1) select(0, false);
    if (reducedMotion()) { select(OPERATOR_STAGES.length - 1, false); return; }
    state.timer = setInterval(() => {
      select(state.stage + 1);
      if (state.stage === OPERATOR_STAGES.length - 1) { pause(); select(state.stage, false); }
    }, 1700);
    renderControls();
  }, { signal: events.signal });
  renderLanguage();
  return {
    update(next = {}) {
      pause();
      if (Object.hasOwn(next, 'language')) state.language = next.language;
      if (Object.hasOwn(next, 'translate')) state.translate = next.translate;
      renderLanguage();
    },
    getStage: () => OPERATOR_STAGES[state.stage].id,
    destroy() {
      pause();
      state.animation?.cancel();
      events.abort();
      container.replaceChildren();
    },
  };
}
