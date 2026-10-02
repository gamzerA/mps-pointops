const HASH_TO_VIEW = Object.freeze({
  top: 'compute',
  compute: 'compute',
  route: 'execution',
  capabilities: 'ecosystem',
  phases: 'ecosystem',
  trust: 'measurements',
  benchmarks: 'measurements',
  provenance: 'measurements',
  methods: 'methods',
  morton: 'methods',
  pruning: 'methods',
  quickstart: 'install',
});

const VIEW_TO_HASH = Object.freeze({
  compute: '#top',
  execution: '#route',
  ecosystem: '#capabilities',
  measurements: '#benchmarks',
  methods: '#methods',
  install: '#quickstart',
});

export const SITE_VIEWS = Object.freeze(Object.keys(VIEW_TO_HASH));

function hashId(hash) {
  const raw = String(hash ?? '').replace(/^#/, '').trim();
  try {
    return decodeURIComponent(raw).toLowerCase();
  } catch {
    return raw.toLowerCase();
  }
}

export function resolveSiteView(hash, defaultView = 'compute') {
  if (!SITE_VIEWS.includes(defaultView)) throw new RangeError(`Unknown default view: ${defaultView}`);
  return HASH_TO_VIEW[hashId(hash)] ?? defaultView;
}

export function canonicalViewHash(view) {
  const hash = VIEW_TO_HASH[view];
  if (!hash) throw new RangeError(`Unknown site view: ${view}`);
  return hash;
}

/**
 * Show one site view at a time while preserving the existing section anchors.
 * Sections must be direct children of `main` and carry `data-site-view`.
 * The browser's normal hash history remains the source of truth.
 */
export function mountViewRouter({
  main, nav, skipLink, onViewChange, defaultView = 'compute', windowRef = window,
}) {
  if (!main || !windowRef?.location || typeof windowRef.addEventListener !== 'function') {
    throw new TypeError('mountViewRouter needs a main element and window');
  }
  if (!SITE_VIEWS.includes(defaultView)) throw new RangeError(`Unknown default view: ${defaultView}`);
  if (onViewChange != null && typeof onViewChange !== 'function') {
    throw new TypeError('onViewChange must be a function');
  }

  const sections = Array.from(main.children).filter(child => child.hasAttribute('data-site-view'));
  if (sections.length === 0) throw new Error('No main sections have data-site-view');
  const presentViews = new Set(sections.map(section => section.getAttribute('data-site-view')));
  for (const view of presentViews) {
    if (!SITE_VIEWS.includes(view)) throw new RangeError(`Unknown data-site-view: ${view}`);
  }
  if (!presentViews.has(defaultView)) throw new Error(`Default view ${defaultView} has no section`);

  const navLinks = nav ? Array.from(nav.querySelectorAll('a[href^="#"]')) : [];
  const activeSkipLink = skipLink ?? main.ownerDocument?.querySelector?.('.skip-link');
  const originalSkipHref = activeSkipLink?.getAttribute('href');
  let currentView = null;

  function syncFromLocation() {
    const previousView = currentView;
    const nextView = resolveSiteView(windowRef.location.hash, defaultView);
    const selectedView = presentViews.has(nextView) ? nextView : defaultView;
    const changed = selectedView !== currentView;
    currentView = selectedView;
    main.setAttribute('data-current-view', selectedView);
    activeSkipLink?.setAttribute('href', canonicalViewHash(selectedView));

    for (const section of sections) {
      const visible = section.getAttribute('data-site-view') === selectedView;
      section.hidden = !visible;
      section.toggleAttribute('data-current', visible);
    }

    let marked = false;
    for (const link of navLinks) {
      const view = link.getAttribute('data-view') || resolveSiteView(link.getAttribute('href'), defaultView);
      const active = view === selectedView && !marked;
      if (active) {
        link.setAttribute('aria-current', 'page');
        marked = true;
      } else {
        link.removeAttribute('aria-current');
      }
    }

    // Call after visibility changes so canvases can measure their new size.
    if (changed) onViewChange?.({ view: selectedView, previousView });

    // An anchor in a previously hidden section cannot receive the browser's
    // normal hash scroll. Unknown hashes fall back to the default view's top.
    const id = hashId(windowRef.location.hash);
    const knownHash = Object.hasOwn(HASH_TO_VIEW, id);
    const sharedWorkbench = selectedView === 'methods' || selectedView === 'measurements';
    if ((changed || !knownHash || sharedWorkbench) && id) {
      const firstSection = sections.find(section => section.getAttribute('data-site-view') === selectedView);
      const target = (sharedWorkbench ? firstSection
        : knownHash ? main.ownerDocument?.getElementById(id) : null) ?? firstSection;
      if (target && !target.closest?.('[hidden]') && typeof target.scrollIntoView === 'function') {
        windowRef.requestAnimationFrame?.(() => target.scrollIntoView({ block: 'start' }));
      }
    }
    return selectedView;
  }

  windowRef.addEventListener('hashchange', syncFromLocation);
  syncFromLocation();

  return {
    get currentView() { return currentView; },
    refresh: syncFromLocation,
    navigate(viewOrHash) {
      const hash = SITE_VIEWS.includes(viewOrHash) ? canonicalViewHash(viewOrHash) : String(viewOrHash);
      if (!hash.startsWith('#')) throw new TypeError('navigate expects a site view or hash');
      if (windowRef.location.hash !== hash) windowRef.location.hash = hash;
      return syncFromLocation();
    },
    destroy() {
      windowRef.removeEventListener('hashchange', syncFromLocation);
      for (const section of sections) {
        section.hidden = false;
        section.removeAttribute('data-current');
      }
      main.removeAttribute('data-current-view');
      for (const link of navLinks) link.removeAttribute('aria-current');
      if (activeSkipLink) {
        if (originalSkipHref === null) activeSkipLink.removeAttribute('href');
        else activeSkipLink.setAttribute('href', originalSkipHref);
      }
      currentView = null;
    },
  };
}
