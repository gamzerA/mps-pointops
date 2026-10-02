const COPY = {
  ko: {
    group: '계산 원리 장면 선택',
    morton: 'Morton 키',
    pruning: 'AABB 가지치기',
  },
  en: {
    group: 'Choose a Methods demonstration',
    morton: 'Morton key',
    pruning: 'AABB pruning',
  },
};

/** Keep each full-size teaching model directly reachable within the Methods view. */
export function mountMethodsPanels(root, {
  language = 'ko', windowRef = window, onPanelChange,
} = {}) {
  const content = root?.parentElement;
  const shell = root?.querySelector('.shell');
  const morton = content?.querySelector('#morton');
  const pruning = content?.querySelector('#pruning');
  if (!shell || !morton || !pruning || !windowRef?.location
      || (onPanelChange != null && typeof onPanelChange !== 'function')) {
    throw new TypeError('Methods panels require the heading, both demonstrations, and a window');
  }

  const documentRef = root.ownerDocument;
  const group = documentRef.createElement('div');
  group.className = 'methods-switch';
  group.setAttribute('role', 'group');
  const buttons = {};
  const names = {};
  for (const [index, panel] of ['morton', 'pruning'].entries()) {
    const button = documentRef.createElement('button');
    button.type = 'button';
    button.dataset.methodsPanelOption = panel;
    button.setAttribute('aria-controls', panel);
    button.setAttribute('aria-pressed', 'false');
    const number = documentRef.createElement('span');
    number.className = 'methods-switch-index';
    number.textContent = `0${index + 1}`;
    const name = documentRef.createElement('strong');
    button.append(number, name);
    group.append(button);
    buttons[panel] = button;
    names[panel] = name;
  }
  shell.append(group);

  let selected = null;
  function select(panel) {
    if (panel !== 'morton' && panel !== 'pruning') return false;
    if (selected === panel) return true;
    selected = panel;
    content.dataset.methodsPanel = panel;
    for (const [name, button] of Object.entries(buttons)) {
      button.setAttribute('aria-pressed', String(name === panel));
    }
    onPanelChange?.(panel);
    return true;
  }

  function syncFromHash() {
    const hash = String(windowRef.location.hash ?? '').toLowerCase();
    if (hash === '#pruning') select('pruning');
    else if (hash === '#methods' || hash === '#morton') select('morton');
  }

  function setLanguage(nextLanguage) {
    const copy = COPY[nextLanguage === 'en' ? 'en' : 'ko'];
    group.setAttribute('aria-label', copy.group);
    names.morton.textContent = copy.morton;
    names.pruning.textContent = copy.pruning;
  }

  for (const [panel, button] of Object.entries(buttons)) {
    button.addEventListener('click', () => {
      select(panel);
      const hash = `#${panel}`;
      if (windowRef.location.hash !== hash) windowRef.location.hash = hash;
    });
  }
  windowRef.addEventListener('hashchange', syncFromHash);
  syncFromHash();
  if (!selected) select('morton');
  setLanguage(language);

  return {
    select,
    setLanguage,
    getSelectedPanel: () => selected,
    disconnect() {
      windowRef.removeEventListener('hashchange', syncFromHash);
      group.remove();
      delete content.dataset.methodsPanel;
    },
  };
}
