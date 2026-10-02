const COPY = {
  ko: {
    group: '적용 범위 내용 선택',
    capability: ['모델별 검증', '배포 · 테스트 범위'],
    phases: ['연산 단계', '현재 · 연구 · 계획'],
    capabilityRegion: '모델별 검증 범위',
  },
  en: {
    group: 'Choose What runs content',
    capability: ['Model paths', 'Shipped · tested scope'],
    phases: ['Compute stack', 'Public APIs · bounded experiments'],
    capabilityRegion: 'Model validation scope',
  },
};

const HOVER_INTENT_MS = 90;

/** Keep the model evidence and compute stack directly selectable in the same view. */
export function mountEcosystemPanels(root, {
  language = 'ko', windowRef = window, capabilities, phases,
} = {}) {
  const buttons = [...root.querySelectorAll('[data-ecosystem-panel-option]')];
  const modelOptions = [...root.querySelectorAll('#capability-map [data-capability]')];
  const phaseOptions = [...root.querySelectorAll('#phases [data-phase-step]')];
  const group = root.querySelector('.ecosystem-switch');
  const capabilityRegion = root.querySelector('#capability-map');
  const phaseRegion = root.querySelector('#phases');
  const capabilityDetail = root.querySelector('.capability-detail');
  if (buttons.length !== 2 || !group || !capabilityRegion || !phaseRegion || !modelOptions.length
      || !phaseOptions.length || typeof capabilities?.select !== 'function'
      || typeof phases?.select !== 'function') {
    throw new TypeError('Ecosystem panels require controls and mounted explorers');
  }
  const choices = new Set(['capabilities', 'phases']);
  let current = 'capabilities';
  let currentLanguage = language === 'en' ? 'en' : 'ko';
  let pendingHover = null;
  let pendingTarget = null;
  const cleanup = [];
  const schedule = windowRef.setTimeout?.bind(windowRef) ?? setTimeout;
  const unschedule = windowRef.clearTimeout?.bind(windowRef) ?? clearTimeout;

  function listen(element, event, handler) {
    element.addEventListener(event, handler);
    cleanup.push(() => element.removeEventListener?.(event, handler));
  }

  function cancelHover() {
    if (pendingHover !== null) unschedule(pendingHover);
    pendingHover = null;
    pendingTarget = null;
  }

  function bindHover(element, activate, { focus = false } = {}) {
    listen(element, 'pointerenter', event => {
      if (event.pointerType !== 'mouse' || event.buttons) return;
      cancelHover();
      pendingTarget = element;
      pendingHover = schedule(() => {
        pendingHover = null;
        pendingTarget = null;
        activate('hover');
      }, HOVER_INTENT_MS);
    });
    listen(element, 'pointerleave', () => {
      if (pendingTarget === element) cancelHover();
    });
    listen(element, 'click', cancelHover);
    if (focus) listen(element, 'focus', () => {
      cancelHover();
      if (element.matches?.(':focus-visible')) activate('focus');
    });
  }

  function protectedFocus(region) {
    const active = root.ownerDocument?.activeElement;
    return active && region?.contains?.(active)
      && (active.matches?.(':focus-visible') || capabilityDetail?.contains?.(active));
  }

  function select(panel) {
    if (!choices.has(panel)) return false;
    cancelHover();
    if (current === panel && root.dataset.ecosystemPanel === panel) return true;
    current = panel;
    root.dataset.ecosystemPanel = panel;
    for (const button of buttons) {
      button.setAttribute('aria-pressed', String(button.dataset.ecosystemPanelOption === panel));
    }
    return true;
  }

  function syncFromHash() {
    cancelHover();
    const hash = windowRef.location?.hash?.toLowerCase();
    if (hash === '#phases') select('phases');
    else if (hash === '#capabilities') select('capabilities');
  }

  function setLanguage(nextLanguage) {
    currentLanguage = nextLanguage === 'en' ? 'en' : 'ko';
    const copy = COPY[currentLanguage];
    group.setAttribute('aria-label', copy.group);
    capabilityRegion.setAttribute('aria-label', copy.capabilityRegion);
    for (const button of buttons) {
      const label = button.dataset.ecosystemPanelOption === 'phases' ? copy.phases : copy.capability;
      button.querySelector('strong').textContent = label[0];
      button.querySelector('small').textContent = label[1];
    }
    select(current);
  }

  for (const button of buttons) {
    listen(button, 'click', () => select(button.dataset.ecosystemPanelOption));
  }
  for (const button of modelOptions) {
    bindHover(button, source => {
      if (current === 'capabilities' && (source === 'focus' || !protectedFocus(capabilityRegion))
          && button.getAttribute('aria-pressed') !== 'true') {
        capabilities.select(button.dataset.capability);
      }
    }, { focus: true });
  }
  for (const button of phaseOptions) {
    bindHover(button, source => {
      if (current === 'phases' && (source === 'focus' || !protectedFocus(phaseRegion))
          && button.getAttribute('aria-current') !== 'step') {
        phases.select(Number(button.dataset.phaseStep));
      }
    }, { focus: true });
  }
  windowRef.addEventListener('hashchange', syncFromHash);
  syncFromHash();
  setLanguage(currentLanguage);
  return {
    select,
    setLanguage,
    getSelectedPanel: () => current,
    disconnect() {
      cancelHover();
      windowRef.removeEventListener('hashchange', syncFromHash);
      cleanup.forEach(remove => remove());
    },
  };
}
